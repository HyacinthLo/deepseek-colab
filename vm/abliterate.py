"""Plans and runs in-place mrfakename abliteration-edit.npz baking on GGUF shards via the compiled abliterate binary.
usage: python3 -I abliterate.py MODEL_DIR EDIT_NPZ ABLITERATE_BIN"""
import ast, glob, math, os, re, struct, subprocess, sys, tempfile, zipfile

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
         16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
         21: (256, 110), 22: (256, 82), 23: (256, 136), 26: (1, 4), 27: (1, 8), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
TARGET_RE = re.compile(r'^blk\.(\d+)\.(attn_output_b|ffn_down_shexp)\.weight$')


def load_npy(raw):
    """Minimal stdlib .npy reader for float32/float64 2D arrays (handles both C and Fortran order)."""
    assert raw[:6] == b'\x93NUMPY', 'not a .npy stream'
    major = raw[6]
    hlen = struct.unpack('<H' if major == 1 else '<I', raw[8:10 if major == 1 else 12])[0]
    off = (10 if major == 1 else 12) + hlen
    hdr = ast.literal_eval(raw[10 if major == 1 else 12:off].decode('latin1').strip())
    descr, fortran, shape = hdr['descr'], bool(hdr['fortran_order']), tuple(hdr['shape'])
    fmt_map = {'<f4': ('<', 'f', 4), '<f8': ('<', 'd', 8), '>f4': ('>', 'f', 4), '>f8': ('>', 'd', 8)}
    assert descr in fmt_map, 'unsupported npy dtype: ' + repr(descr)
    endian, code, esz = fmt_map[descr]
    n = 1
    for d in shape:
        n *= d
    payload = raw[off:off + n * esz]
    assert len(payload) == n * esz, 'truncated npy payload'
    vals = struct.unpack('%s%d%s' % (endian, n, code), payload)
    return shape, fortran, vals


def load_edit_npz(path):
    """Loads Q (dim, 2) as column vectors (q0, q1) and scales (n_layers, 2) from abliteration-edit.npz."""
    with zipfile.ZipFile(path, 'r') as zf:
        q_shape, q_f, q_vals = load_npy(zf.read('Q.npy'))
        s_shape, s_f, s_vals = load_npy(zf.read('scales.npy'))
    assert len(q_shape) == 2 and q_shape[1] == 2, 'unexpected Q shape: %r' % (q_shape,)
    assert len(s_shape) == 2 and s_shape[1] == 2, 'unexpected scales shape: %r' % (s_shape,)
    dim = q_shape[0]
    if q_f:
        q0 = list(q_vals[:dim])
        q1 = list(q_vals[dim:2 * dim])
    else:
        q0 = [q_vals[2 * i] for i in range(dim)]
        q1 = [q_vals[2 * i + 1] for i in range(dim)]
    # Orthonormality sanity check: Q^T Q == I_2
    n0 = sum(x * x for x in q0)
    n1 = sum(x * x for x in q1)
    dot = sum(a * b for a, b in zip(q0, q1))
    assert abs(n0 - 1.0) < 1e-4 and abs(n1 - 1.0) < 1e-4 and abs(dot) < 1e-4, (n0, n1, dot)
    nlayers = s_shape[0]
    if s_f:
        scales = [(float(s_vals[i]), float(s_vals[nlayers + i])) for i in range(nlayers)]
    else:
        scales = [(float(s_vals[2 * i]), float(s_vals[2 * i + 1])) for i in range(nlayers)]
    return dim, q0, q1, scales


def parse_gguf_tensors(path):
    """Returns list of (name, dims, gtype, abs_offset, nbytes) for all tensors in a GGUF shard."""
    with open(path, 'rb') as f:
        u = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        s = lambda: f.read(u('<Q'))

        def skip(t):
            if t == 8:
                s()
            elif t == 9:
                at, n = u('<I'), u('<Q')
                if at in (8, 9):
                    for _ in range(n):
                        skip(at)
                else:
                    f.seek(n * SCALAR[at], 1)
            else:
                f.seek(SCALAR[t], 1)

        assert u('<I') == 0x46554747, 'not a GGUF file: ' + path
        u('<I')
        nt, nkv = u('<Q'), u('<Q')
        for _ in range(nkv):
            s()
            skip(u('<I'))
        out = []
        for _ in range(nt):
            name = s().decode()
            dims = [u('<Q') for _ in range(u('<I'))]
            gt, off = u('<I'), u('<Q')
            n = 1
            for d in dims:
                n *= d
            pb, bb = BLOCK[gt]
            out.append((name, dims, gt, off, n // pb * bb))
        base = (f.tell() + 31) // 32 * 32
    return [(name, dims, gt, base + off, nb) for name, dims, gt, off, nb in out]


def build_plan(model_dir, npz_path):
    dim, q0, q1, scales = load_edit_npz(npz_path)
    shards = sorted(glob.glob(os.path.join(model_dir, '*.gguf')))
    assert shards, 'no .gguf shards found in ' + model_dir
    plan = []
    expected = sum(1 for a, b in scales for x in (a, b) if abs(x) > 0.0)
    for shard in shards:
        real = os.path.realpath(shard)
        for name, dims, gt, off, nb in parse_gguf_tensors(real):
            m = TARGET_RE.match(name)
            if not m:
                continue
            layer, kind = int(m.group(1)), m.group(2)
            if layer >= len(scales):
                continue
            scale = scales[layer][0 if kind == 'attn_output_b' else 1]
            if scale == 0.0:
                continue
            assert len(dims) == 2 and dims[1] == dim, (name, dims, dim)
            plan.append((real, name, gt, dims[0], dims[1], off, nb, scale))
    assert len(plan) == expected, 'expected %d non-zero abliteration targets, found %d' % (expected, len(plan))
    return dim, q0, q1, plan


def patch_chat_template(gguf_path, out_path):
    """Extracts tokenizer.chat_template from GGUF and enables numeric string reasoning_effort (e.g. '50')."""
    with open(gguf_path, 'rb') as f:
        u = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        s = lambda: f.read(u('<Q'))

        def skip(t):
            if t == 8:
                s()
            elif t == 9:
                at, n = u('<I'), u('<Q')
                if at in (8, 9):
                    for _ in range(n):
                        skip(at)
                else:
                    f.seek(n * SCALAR[at], 1)
            else:
                f.seek(SCALAR[t], 1)

        assert u('<I') == 0x46554747, 'not a GGUF file: ' + gguf_path
        u('<I')
        _, nkv = u('<Q'), u('<Q')
        tmpl = None
        for _ in range(nkv):
            k = s().decode()
            t = u('<I')
            if k == 'tokenizer.chat_template' and t == 8:
                tmpl = s().decode('utf-8')
                break
            skip(t)
    assert tmpl, 'tokenizer.chat_template not found in ' + gguf_path
    add = ('{%- elif reasoning_effort is string and (reasoning_effort | int | string) == reasoning_effort '
           'and (reasoning_effort | int) >= 1 and (reasoning_effort | int) <= 100 -%}\n'
           '    {%- set budget = reasoning_effort | int -%}\n')
    anchor = '{%- elif reasoning_effort is integer and reasoning_effort >= 1 and reasoning_effort <= 100 -%}\n    {%- set budget = reasoning_effort -%}\n'
    if '(reasoning_effort | int | string)' not in tmpl and anchor in tmpl:
        tmpl = tmpl.replace(anchor, anchor + add, 1)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tmpl)
    print('wrote patched chat template (%d bytes) to %s' % (len(tmpl), out_path))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--patch-template':
        patch_chat_template(sys.argv[2], sys.argv[3])
        return
    model_dir, npz_path, bin_path = sys.argv[1:4]
    dim, q0, q1, plan = build_plan(model_dir, npz_path)
    with tempfile.TemporaryDirectory(prefix='abliterate-') as td:
        q_bin = os.path.join(td, 'Q.bin')
        plan_txt = os.path.join(td, 'plan.txt')
        with open(q_bin, 'wb') as f:
            f.write(struct.pack('<q', dim))
            f.write(struct.pack('<%dd' % dim, *q0))
            f.write(struct.pack('<%dd' % dim, *q1))
        with open(plan_txt, 'w') as f:
            for real, name, gt, ne0, ne1, off, nb, scale in plan:
                f.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.12g\n' % (real, name, gt, ne0, ne1, off, nb, scale))
        r = subprocess.run([bin_path, q_bin, plan_txt])
        sys.exit(r.returncode)


if __name__ == '__main__':
    main()
