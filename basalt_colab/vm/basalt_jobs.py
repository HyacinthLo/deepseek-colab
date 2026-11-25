#!/usr/bin/env python3
"""Generate pfetch.py HTTP Range job JSONs and storage plans for Basalt DeepSeek-V4.1-Flash packs.

Reads only the 64 KiB BASALTMF v2 header via HTTP Range request and plans split-storage staging:
  - ram+ram (e.g. UD-IQ1_M, 139.1 + 110.6 GiB):
      Backbone -> /mnt/ram/model.basalt (punch-on-read) + Engram -> /mnt/ram/engram.bin
  - ram+ssd (e.g. UD-IQ2_XXS, 154.8 + 110.6 GiB):
      Backbone -> /mnt/ram/model.basalt (punch-on-read) + Engram -> /content/engram.bin (0 cold experts on SSD)
  - ssd+ssd (e.g. UD-Q2_K_XL, ~201 + 110.6 GiB):
      Backbone -> /content/model.basalt (punch-on-read) + Engram -> /content/engram.bin (0 cold experts on SSD)
  - Fail-fast guard rejects packs whose backbone exceeds zero-SSD-expert VRAM+RAM capacity (e.g. NVFP4 ~294 GiB).
"""
import argparse, ctypes, json, os, shlex, struct, sys, tempfile, urllib.request

PACKS = {
    'iq1_m': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt',
    'ud-iq1_m': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt',
    'iq2_xxs': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.basalt',
    'ud-iq2_xxs': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.basalt',
}

HEADER_BYTES = 65536
GIB = 1024 ** 3


def parse_header(data):
    assert len(data) == HEADER_BYTES, f'expected {HEADER_BYTES} header bytes, got {len(data)}'
    magic, ver, nsec, file_bytes = struct.unpack('<8sIIQ', data[:24])
    assert magic == b'BASALTMF' and ver == 2, f'bad basalt header: magic={magic!r} ver={ver}'
    sections = []
    for i in range(nsec):
        off = 64 + i * 128
        name_b, soff, sbytes, sha = struct.unpack('<64sQQ32s', data[off:off + 112])
        name = name_b.split(b'\x00', 1)[0].decode('utf-8')
        sections.append({'name': name, 'offset': soff, 'bytes': sbytes, 'sha256': sha.hex()})
    return file_bytes, sections


def read_header(url, hdr_file=None):
    headers = {'Range': f'bytes=0-{HEADER_BYTES - 1}'}
    if hdr_file and os.path.exists(hdr_file):
        with open(hdr_file) as f:
            for line in f:
                k, _, v = line.strip().partition(':')
                if k:
                    headers[k.strip()] = v.strip()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return parse_header(data)


def summarize_sections(sections):
    bb_bytes = HEADER_BYTES
    eng_bytes = 0
    for s in sections:
        if s['bytes'] == 0:
            continue
        if s['name'] == 'engram':
            eng_bytes += s['bytes']
        else:
            bb_bytes += s['bytes']
    return bb_bytes, eng_bytes


def plan_storage(backbone_bytes, engram_bytes, ram_total_gib, disk_free_gib, vram_gib=96.0, explicit_budget_gib=0):
    """Computes zero-SSD-expert staging plan (ram+ram, ram+ssd, or ssd+ssd) from real pack & VM sizes."""
    bb_gib = backbone_bytes / GIB
    eng_gib = engram_bytes / GIB
    # Conservative usable VRAM for weights after drafter (2.4 GiB), KV, workspace, and reserve.
    vram_weights_gib = max(0.0, float(vram_gib) - 10.0)
    ram_needed_gib = max(0.0, bb_gib - vram_weights_gib)
    max_ram_experts_gib = max(0.0, float(ram_total_gib) - 22.0)

    if ram_needed_gib > max_ram_experts_gib:
        max_bb = vram_weights_gib + max_ram_experts_gib
        raise RuntimeError(
            f'pack backbone ({bb_gib:.1f} GiB) exceeds zero-SSD-expert capacity on this VM '
            f'(VRAM {vram_gib:.0f} GiB + RAM {ram_total_gib:.0f} GiB -> max ~{max_bb:.0f} GiB backbone; '
            f'spilling experts to Colab SSD degrades decode to ~2-5 tok/s)'
        )

    bb_fits_tmpfs = bb_gib <= (float(ram_total_gib) - 12.0)
    both_fit_ram = bb_fits_tmpfs and ((ram_needed_gib + eng_gib) <= (float(ram_total_gib) - 8.0))

    if both_fit_ram:
        stage_mode = 'ram+ram'
        model_path = '/mnt/ram/model.basalt'
        engram_path = '/mnt/ram/engram.bin'
        punch_file = '/mnt/ram/model.basalt'
        auto_budget = max(55, min(160, int(ram_needed_gib + 4)))
    elif bb_fits_tmpfs:
        if eng_gib > float(disk_free_gib) - 6.0:
            raise RuntimeError(
                f'Engram ({eng_gib:.1f} GiB) does not fit in /content disk ({disk_free_gib:.1f} GiB free)'
            )
        stage_mode = 'ram+ssd'
        model_path = '/mnt/ram/model.basalt'
        engram_path = '/content/engram.bin'
        punch_file = '/mnt/ram/model.basalt'
        auto_budget = min(max(55, int(ram_total_gib - 26)), max(80, int(ram_needed_gib + 18)))
    else:
        if bb_gib > float(disk_free_gib) - 6.0:
            raise RuntimeError(
                f'pack backbone ({bb_gib:.1f} GiB) exceeds both /mnt/ram ({ram_total_gib:.0f} GiB) '
                f'and /content disk ({disk_free_gib:.1f} GiB free)'
            )
        if eng_gib > float(disk_free_gib) - 8.0:
            raise RuntimeError(
                f'Engram ({eng_gib:.1f} GiB) does not fit in /content disk ({disk_free_gib:.1f} GiB free)'
            )
        stage_mode = 'ssd+ssd'
        model_path = '/content/model.basalt'
        engram_path = '/content/engram.bin'
        punch_file = '/content/model.basalt'
        auto_budget = min(max(55, int(ram_total_gib - 24)), max(100, int(ram_needed_gib + 15)))

    ram_budget = int(explicit_budget_gib) if int(explicit_budget_gib) > 0 else int(auto_budget)
    return {
        'stage_mode': stage_mode,
        'model_path': model_path,
        'engram_path': engram_path,
        'punch_file': punch_file,
        'ram_budget': ram_budget,
        'backbone_gib': round(bb_gib, 2),
        'engram_gib': round(eng_gib, 2),
        'ram_needed_gib': round(ram_needed_gib, 2),
    }


def build_jobs(url, file_bytes, sections, backbone_dest, engram_dest):
    backbone_jobs = [{
        'url': url,
        'dest': backbone_dest,
        'dest_size': file_bytes,
        'src_off': 0,
        'dest_off': 0,
        'size': HEADER_BYTES,
    }]
    engram_jobs = []
    for s in sections:
        if s['bytes'] == 0:
            continue
        if s['name'] == 'engram':
            engram_jobs.append({
                'url': url,
                'dest': engram_dest,
                'dest_size': s['bytes'],
                'src_off': s['offset'],
                'dest_off': 0,
                'size': s['bytes'],
            })
        else:
            backbone_jobs.append({
                'url': url,
                'dest': backbone_dest,
                'dest_size': file_bytes,
                'src_off': s['offset'],
                'dest_off': s['offset'],
                'size': s['bytes'],
            })
    assert len(engram_jobs) == 1, f'expected 1 engram section, found {len(engram_jobs)}'
    return backbone_jobs, engram_jobs


def _fallocate_punch(fd, offset, length):
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    libc.fallocate.restype = ctypes.c_int
    FALLOC_FL_KEEP_SIZE = 0x01
    FALLOC_FL_PUNCH_HOLE = 0x02
    rc = libc.fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, offset, length)
    return rc, ctypes.get_errno()


def test_punch_support(directory):
    """Probes whether `directory` actually frees disk blocks on FALLOC_FL_PUNCH_HOLE."""
    os.makedirs(directory, exist_ok=True)
    size = 4 * 1024 * 1024
    fd, path = tempfile.mkstemp(prefix='.basalt_punch_probe_', dir=directory)
    try:
        os.write(fd, b'\x5a' * size)
        os.fsync(fd)
        before_blocks = os.fstat(fd).st_blocks
        rc, err = _fallocate_punch(fd, 0, size)
        os.fsync(fd)
        after_blocks = os.fstat(fd).st_blocks
        ok = (rc == 0 and before_blocks > 0 and after_blocks < before_blocks // 2)
        return ok, before_blocks, after_blocks, rc, err
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


def punch_sections(pack_path, names):
    """Inward-4K-aligned FALLOC_FL_PUNCH_HOLE on named BASALTMF sections (e.g. dspark/gguf, dense)."""
    with open(pack_path, 'r+b') as f:
        _, sections = parse_header(f.read(HEADER_BYTES))
        fd = f.fileno()
        punched = 0
        for s in sections:
            if s['name'] not in names or s['bytes'] == 0:
                continue
            pa = (s['offset'] + 4095) & ~4095
            pb = (s['offset'] + s['bytes']) & ~4095
            if pb > pa:
                rc, _ = _fallocate_punch(fd, pa, pb - pa)
                if rc == 0:
                    punched += pb - pa
    return punched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quant', default='ud-iq1_m')
    ap.add_argument('--url', default=None)
    ap.add_argument('--backbone-dest', default='/mnt/ram/model.basalt')
    ap.add_argument('--engram-dest', default='/mnt/ram/engram.bin')
    ap.add_argument('--backbone-jobs', default=None)
    ap.add_argument('--engram-jobs', default=None)
    ap.add_argument('--engram-size-out', default=None, help='Optional path to write exact Engram byte size')
    ap.add_argument('--punch-sections', nargs='+', default=None,
                    help='Punch named sections (e.g. dspark/gguf dense) in --backbone-dest after /load')
    ap.add_argument('--test-punch', default=None, help='Probe FALLOC_FL_PUNCH_HOLE support in directory')
    ap.add_argument('--plan', action='store_true',
                    help='Read remote 64 KiB header and emit shell exports for optimal storage staging')
    ap.add_argument('--ram-total-gib', type=float, default=176.0)
    ap.add_argument('--disk-free-gib', type=float, default=220.0)
    ap.add_argument('--vram-gib', type=float, default=96.0)
    ap.add_argument('--ram-budget', type=int, default=0)
    args = ap.parse_args()

    if args.test_punch:
        ok, before_b, after_b, rc, err = test_punch_support(args.test_punch)
        print(f'punch probe ({args.test_punch}): ok={ok} blocks={before_b}->{after_b} rc={rc} errno={err}')
        sys.exit(0 if ok else 1)

    if args.punch_sections:
        freed = punch_sections(args.backbone_dest, set(args.punch_sections))
        print(f'punched {args.punch_sections} in {args.backbone_dest}: freed {freed / GIB:.2f} GiB')
        return

    url = args.url or PACKS.get(args.quant.lower())
    if not url:
        sys.exit(f'unknown quant {args.quant!r} without --url')

    file_bytes, sections = read_header(url, os.environ.get('HF_HDR'))

    if args.plan:
        bb_bytes, eng_bytes = summarize_sections(sections)
        try:
            plan = plan_storage(
                bb_bytes,
                eng_bytes,
                ram_total_gib=args.ram_total_gib,
                disk_free_gib=args.disk_free_gib,
                vram_gib=args.vram_gib,
                explicit_budget_gib=args.ram_budget,
            )
        except RuntimeError as e:
            sys.stderr.write(f'PLAN ERROR: {e}\n')
            sys.exit(2)
        for k, v in (
            ('STAGE_MODE', plan['stage_mode']),
            ('MODEL_PATH', plan['model_path']),
            ('ENGRAM_PATH', plan['engram_path']),
            ('BASALT_PUNCH_FILE', plan['punch_file']),
            ('RAM_BUDGET', str(plan['ram_budget'])),
            ('BACKBONE_GIB', str(plan['backbone_gib'])),
            ('ENGRAM_GIB', str(plan['engram_gib'])),
            ('MODEL_URL', url),
        ):
            print(f'export {k}={shlex.quote(v)}')
        return

    assert args.backbone_jobs and args.engram_jobs, '--backbone-jobs and --engram-jobs are required'
    bb_jobs, eng_jobs = build_jobs(url, file_bytes, sections, args.backbone_dest, args.engram_dest)

    for path, data in ((args.backbone_jobs, bb_jobs), (args.engram_jobs, eng_jobs)):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    if args.engram_size_out:
        with open(args.engram_size_out, 'w') as f:
            f.write(str(eng_jobs[0]['size']) + '\n')

    bb_gb = sum(j['size'] for j in bb_jobs) / 1e9
    eng_gb = sum(j['size'] for j in eng_jobs) / 1e9
    print(f'basalt_jobs ({args.quant}): logical={file_bytes/1e9:.2f} GB | '
          f'backbone={bb_gb:.2f} GB ({len(bb_jobs)} ranges -> {args.backbone_dest}) | '
          f'engram={eng_gb:.2f} GB (1 range -> {args.engram_dest})')


if __name__ == '__main__':
    main()
