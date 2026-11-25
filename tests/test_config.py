"""Unit tests for basalt_colab config, paths, legacy migration, manifest, usage ledger, colab helpers, and VM payload."""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basalt_colab import colab, config, manifest, paths, proxy, usage  # noqa: E402
from basalt_colab.cli import parse_ask_args  # noqa: E402


class ConfigAndUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="bcolab-unit-"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.p = paths.get_paths(home_override=self.tmp, vm_override=ROOT / "basalt_colab" / "vm")
        self.p.ensure_dirs()
        self.p.key_file.write_text("sk-ds41-" + "a" * 48 + "\n", encoding="utf-8")

    def test_manifest_and_vm_files_in_sync(self) -> None:
        m = manifest.load_manifest(self.p.vm_dir)
        self.assertEqual(m.basalt_tag, "v0.5.7")
        self.assertEqual(m.pack_model_id, config.DEFAULT_MODEL_ALIAS)
        self.assertEqual(colab.missing_vm_files(self.p.vm_dir), [])

    def test_config_toml_roundtrip_validation_and_hot_cold_diff(self) -> None:
        cfg = config.Config()
        cfg.basalt.max_context = 1048576
        cfg.basalt.parallel = 3
        cfg.basalt.kv = "fp16"
        cfg.basalt.effort = "high"
        cfg.basalt.ref = "main"
        cfg.basalt.sampling.temperature = 0.7
        cfg.basalt.sampling.top_p = 0.95
        config.save_config(cfg, self.p)

        loaded = config.load_config(self.p, apply_env=False)
        self.assertEqual(loaded.to_dict(), cfg.to_dict())
        self.assertEqual(loaded.basalt.max_context, 1048576)
        self.assertEqual(loaded.basalt.sampling.to_basalt_dict(), {"temperature": 0.7, "top-p": 0.95})

        # Hot vs cold diff
        hot_only = loaded.clone()
        hot_only.basalt.effort = "max"
        hot_only.colab.idle_min = 25
        hot, cold = config.diff_hot_cold(loaded, hot_only)
        self.assertEqual(sorted(hot), ["basalt.effort", "colab.idle_min"])
        self.assertEqual(cold, [])

        cold_change = loaded.clone()
        cold_change.basalt.kv = "fp8"
        _, cold2 = config.diff_hot_cold(loaded, cold_change)
        self.assertEqual(cold2, ["basalt.kv"])

        # Context parser supports up to 1M
        self.assertEqual(
            [config.context_value(v) for v in ("32k", "262K", "512k", "1M", "5000", "4095", "2000000", "abc")],
            [32768, 262144, 524288, 1048576, 5120, None, None, None],
        )

        # Validation rejects out-of-range values
        bad = loaded.clone()
        bad.basalt.parallel = 8
        with self.assertRaises(config.ConfigError):
            config.validate_config(bad)

        bad_kv = loaded.clone()
        bad_kv.basalt.kv = "int4"
        with self.assertRaises(config.ConfigError):
            config.validate_config(bad_kv)

    def test_boot_vars_and_cmd(self) -> None:
        cfg = config.Config()
        cfg.basalt.max_context = 524288
        cfg.basalt.ref = "v0.5.8"
        cfg.basalt.sampling.temperature = 0.8
        bv = cfg.boot_vars()
        self.assertEqual(bv["CTX"], "524288")
        self.assertEqual(bv["BASALT_REF"], "v0.5.8")
        self.assertEqual(bv["VISION_MODE"], "dynamic")
        self.assertEqual(bv["BASALT_SAMPLING_JSON"], '{"temperature":0.8}')
        cmd = colab.boot_cmd(bv)
        self.assertIn("setsid nohup bash vm/boot.sh", cmd)
        self.assertNotIn("libpin.so", cmd)

    def test_legacy_state_migration(self) -> None:
        legacy = self.tmp / "old_state"
        legacy.mkdir()
        (legacy / "api_key").write_text("sk-ds41-legacy\n", encoding="utf-8")
        (legacy / "id_ed25519").write_text("PRIVATE-KEY\n", encoding="utf-8")
        new_p = paths.get_paths(home_override=self.tmp / "fresh_home", vm_override=self.p.vm_dir)
        migrated = paths.migrate_legacy_state(new_p, legacy_dir=legacy)
        self.assertEqual(sorted(migrated), ["api_key", "id_ed25519"])
        self.assertEqual(new_p.key_file.read_text(encoding="utf-8"), "sk-ds41-legacy\n")
        self.assertEqual(os.stat(new_p.key_file).st_mode & 0o777, 0o600)
        # Idempotent second run migrates nothing
        self.assertEqual(paths.migrate_legacy_state(new_p, legacy_dir=legacy), [])

    def test_usage_ledger_aggregation(self) -> None:
        ledger = usage.UsageLedger(self.p.usage_file)
        now = time.time()
        ledger.record(session="basalt-ds41", started_ts=now - 1800, stopped_ts=now - 600, reason="idle 10.0 min", ran_s=1200, cu_est=2.97, served=14)
        ledger.record(session="basalt-ds41", started_ts=now - 300, stopped_ts=now, reason="down", ran_s=300, cu_est=0.74, served=3)
        s = ledger.summary(days=7, active_ran_s=180, active_cu=0.45)
        self.assertEqual(s["sessions_count"], 2)
        self.assertAlmostEqual(s["today_cu"], 4.16, places=2)
        self.assertEqual(len(s["daily"]), 7)
        self.assertEqual(s["recent"][0]["reason"], "down")

    def test_rewrite_request_and_parse_ask(self) -> None:
        cfg = config.Config()
        cfg.basalt.effort = "high"
        chat = json.loads(proxy.rewrite_request(json.dumps({"model": "claude-haiku-4-5", "messages": []}).encode(), cfg))
        self.assertEqual((chat["model"], chat["reasoning_effort"]), (config.DEFAULT_MODEL_ALIAS, "high"))
        resp = json.loads(proxy.rewrite_request(json.dumps({"model": "gpt-5", "input": "hi"}).encode(), cfg, "/v1/responses"))
        self.assertEqual((resp["model"], resp["reasoning"]), (config.DEFAULT_MODEL_ALIAS, {"effort": "high"}))
        self.assertNotIn("reasoning_effort", resp)
        anth = json.loads(proxy.rewrite_request(json.dumps({"model": "claude-sonnet-4-5", "messages": []}).encode(), cfg, "/v1/messages"))
        self.assertEqual(anth["model"], config.DEFAULT_MODEL_ALIAS)
        self.assertNotIn("reasoning_effort", anth)
        cfg.proxy.strict_model = True
        strict = json.loads(proxy.rewrite_request(json.dumps({"model": "other", "messages": []}).encode(), cfg))
        self.assertEqual(strict["model"], "other")

        self.assertEqual(parse_ask_args(["--max", "--image", "a.png", "hi"]), ("max", ["a.png"], "hi"))
        self.assertEqual(parse_ask_args(["--effort", "75", "hello", "world"]), ("75", [], "hello world"))

    def test_bundle_lf_modes_and_log_redaction(self) -> None:
        vm_crlf = self.tmp / "vmcrlf"
        files = colab.vm_files(self.p.vm_dir)
        for f in files:
            dst = vm_crlf / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((self.p.vm_dir / f).read_bytes().replace(b"\n", b"\r\n"))
        p_crlf = paths.Paths(self.p.config_dir, self.p.state_dir, vm_crlf)
        out = self.tmp / "b.tgz"
        colab.build_bundle(p_crlf, out)
        with tarfile.open(out) as t:
            names = {m.name: m.mode for m in t.getmembers()}
            body = {m.name: t.extractfile(m).read() for m in t.getmembers()}
        self.assertIn("vm/manifest.json", names)
        self.assertFalse([n for n, b in body.items() if b"\r" in b])
        self.assertEqual(names["secrets/key"], 0o600)

        logger = colab.Logger(self.p, echo=False)
        key = colab.read_api_key(self.p)
        logger.log(f"secret {key} and Authorization: Bearer tok123")
        txt = self.p.log_file.read_text(encoding="utf-8")
        self.assertNotIn(key, txt)
        self.assertNotIn("tok123", txt)

    def test_serve_sh_v2_schema_jobs_and_abliterate(self) -> None:
        serve_sh = (self.p.vm_dir / "serve.sh").read_text(encoding="utf-8")
        m = re.search(r"python3 - \"\$R/basalt_config\.json\" <<'EOF'\n(.*?)\nEOF", serve_sh, re.S)
        self.assertIsNotNone(m)
        cfg_path = self.tmp / "basalt_config.json"
        env = dict(
            os.environ,
            PORT="18081", CTX="1048576", PARALLEL="2", KV_TYPE="fp8", VISION_MODE="dynamic",
            SPEC_MAX="6", DRAFT_EXPERTS="32", VRAM_RESERVE="700", RAM_BUDGET="55", ANSWER_ROOM="0",
            MODEL_ALIAS=config.DEFAULT_MODEL_ALIAS, CONV_CACHE_MIB="32768", CONV_CACHE_DIR="/content/conv_cache",
            MODEL_PATH="/mnt/ram/model.basalt", ENGRAM_PATH="/mnt/ram/engram.bin",
            BASALT_SAMPLING_JSON='{"temperature":0.7,"top-p":0.95}',
        )
        r = subprocess.run([sys.executable, "-c", m.group(1), str(cfg_path)], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], 2)
        m0 = doc["models"][0]
        self.assertEqual((m0["max-context"], m0["slot-full-context"], m0["kv"], m0["vision"], m0["temperature"], m0["top-p"]),
                         (1048576, True, "fp8", "dynamic", 0.7, 0.95))
        self.assertEqual((m0["answer-room"], m0["idle-unload"]), (0, 0))
        self.assertEqual(doc["engine"]["conversation-cache-disk-mib"], 32768)

        # Synthetic BASALTMF v2 pack + punch_sections + abliterate_basalt
        sys.path.insert(0, str(self.p.vm_dir))
        try:
            bj = importlib.import_module("basalt_jobs")
            ab = importlib.import_module("abliterate_basalt")
            hdr = bytearray(65536)
            sections = [
                ("meta", 65536, 4096),
                ("gguf", 131072, 1048576),
                ("dspark/gguf", 1179648, 262144),
                ("vision/mmproj", 1441792, 65536),
                ("experts", 2097152, 4194304),
                ("engram", 8388608, 16777216),
            ]
            file_bytes = 8388608 + 16777216
            struct.pack_into("<8sIIQ", hdr, 0, b"BASALTMF", 2, len(sections), file_bytes)
            for i, (name, soff, sbytes) in enumerate(sections):
                struct.pack_into("<64sQQ", hdr, 64 + i * 128, name.encode(), soff, sbytes)
            fb, parsed = bj.parse_header(bytes(hdr))
            bb_jobs, eng_jobs = bj.build_jobs("https://example.test/m.basalt", fb, parsed, "/mnt/ram/model.basalt", "/mnt/ram/engram.bin")
            self.assertEqual(len(eng_jobs), 1)
            self.assertEqual(sum(j["size"] for j in bb_jobs), 65536 + 4096 + 1048576 + 262144 + 65536 + 4194304)

            if sys.platform.startswith("linux"):
                pf = self.tmp / "punch.basalt"
                with pf.open("wb") as f:
                    f.write(hdr)
                    f.seek(1179648)
                    f.write(b"\xAA" * 262144)
                self.assertEqual(bj.punch_sections(str(pf), {"dspark/gguf"}), 262144)

            dim = 4
            def npy(shape, fortran, vals):
                h = (f"{{'descr': '<f4', 'fortran_order': {fortran}, 'shape': {shape!r}, }}").encode()
                h = h + b" " * ((64 - (10 + len(h) + 1) % 64) % 64) + b"\n"
                return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(h)) + h + struct.pack(f"<{len(vals)}f", *vals)

            npz = self.tmp / "edit.npz"
            with zipfile.ZipFile(npz, "w") as zf:
                zf.writestr("Q.npy", npy((dim, 2), True, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
                zf.writestr("scales.npy", npy((2, 2), False, [0.5, 0.0, 0.0, 0.75]))
            _, secs = ab.parse_basalt_header(bytes(hdr))
            meta = {"tensors": {
                "blk.0.attn_output_b.weight": {"section": "gguf", "offset": 0, "bytes": 210 * dim, "shape": [256, dim], "type_id": 14},
                "blk.1.ffn_down_shexp.weight": {"section": "gguf", "offset": 4096, "bytes": 176 * dim, "shape": [256, dim], "type_id": 13},
            }}
            _, _, _, plan = ab.build_plan_from_meta("/mnt/ram/model.basalt", secs, meta, str(npz))
            self.assertEqual([(p[1], p[5], p[7]) for p in plan],
                             [("blk.0.attn_output_b.weight", 131072, 0.5),
                              ("blk.1.ffn_down_shexp.weight", 131072 + 4096, 0.75)])
        finally:
            sys.path.remove(str(self.p.vm_dir))


if __name__ == "__main__":
    unittest.main()
