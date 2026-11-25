#!/usr/bin/env python3
"""Offline converter for experimental DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL.basalt.

Downloads Unsloth's 7-shard UD-Q2_K_XL GGUF (~335 GB) + mmproj-BF16.gguf (~1 GB) +
dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf (~8 GB), converts them into a single BASALTMF v2
`.basalt` pack via Basalt v0.5.7's `tools/convert/convert.py`, verifies every section SHA-256,
and optionally uploads the result to your Hugging Face repository.

Requirements:
  - ~720 GB free disk space on the conversion host (344 GB source + ~344 GB output),
    unless `--delete-sources` is used before upload (still ~700 GB peak during conversion).
  - `git`, `python3` (with `numpy` if Basalt's converter needs it), and `hf` (huggingface_hub CLI).

Usage:
  python3 scripts/build_q2k_xl_pack.py --work-dir /data/q2k_xl_build
  python3 scripts/build_q2k_xl_pack.py --work-dir /data/q2k_xl_build --upload-repo YourName/DeepSeek-V4.1-Flash-1017-Basalt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "basalt_colab" / "vm" / "manifest.json"
DEFAULT_SOURCE_REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
DEFAULT_OUT_NAME = "DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL.basalt"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def load_basalt_pin() -> tuple[str, str]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return raw["basalt_repo"], raw["basalt_tag"]


def check_disk(work_dir: Path, min_gib: int) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(work_dir).free / (1024 ** 3)
    print(f"Free disk at {work_dir}: {free_gib:.1f} GiB (recommended >= {min_gib} GiB)", flush=True)
    if free_gib < min_gib:
        sys.exit(
            f"Not enough free disk space ({free_gib:.1f} GiB < {min_gib} GiB). "
            f"Pass --skip-disk-check to override if you already downloaded source shards."
        )


def find_single(root: Path, patterns: list[str], label: str) -> Path:
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(Path(p) for p in sorted(glob.glob(str(root / pat), recursive=True)))
    if not matches:
        sys.exit(f"Could not locate {label} under {root} (patterns: {patterns})")
    return matches[0]


def main() -> None:
    repo_default, tag_default = load_basalt_pin()
    ap = argparse.ArgumentParser(description="Build DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL.basalt from Unsloth GGUFs")
    ap.add_argument("--work-dir", required=True, type=Path, help="Scratch + output directory (~720 GiB free recommended)")
    ap.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, help=f"HF GGUF repo (default {DEFAULT_SOURCE_REPO})")
    ap.add_argument("--quant-dir", default="UD-Q2_K_XL", help="Quant subfolder in source repo (default UD-Q2_K_XL)")
    ap.add_argument("--out", type=Path, default=None, help=f"Output .basalt path (default <work-dir>/{DEFAULT_OUT_NAME})")
    ap.add_argument("--basalt-repo", default=repo_default)
    ap.add_argument("--basalt-ref", default=tag_default)
    ap.add_argument("--workers", type=int, default=max(4, min(16, os.cpu_count() or 8)))
    ap.add_argument("--skip-download", action="store_true", help="Reuse already-downloaded GGUFs in <work-dir>/gguf")
    ap.add_argument("--skip-verify", action="store_true", help="Skip convert.py --verify pass")
    ap.add_argument("--skip-disk-check", action="store_true", help="Skip free disk space preflight")
    ap.add_argument("--delete-sources", action="store_true", help="Delete downloaded GGUFs after verification succeeds")
    ap.add_argument("--upload-repo", default="", help="Optional HF repo ID to upload the finished .basalt pack")
    ap.add_argument("--private", action="store_true", help="Create --upload-repo as private if it does not exist")
    args = ap.parse_args()

    work = args.work_dir.resolve()
    gguf_dir = work / "gguf"
    basalt_dir = work / "basalt"
    out_path = (args.out or (work / DEFAULT_OUT_NAME)).resolve()

    if not args.skip_disk_check:
        check_disk(work, 360 if args.skip_download else 700)

    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if (not args.skip_download or args.upload_repo) and not hf_bin:
        sys.exit("Missing Hugging Face CLI (`hf` or `huggingface-cli`). Install: pip install huggingface_hub")

    # 1. Fetch Basalt converter at pinned tag
    if not (basalt_dir / "tools" / "convert" / "convert.py").is_file():
        run(["git", "clone", "--depth", "1", "--branch", args.basalt_ref, args.basalt_repo, str(basalt_dir)])

    # 2. Download Unsloth UD-Q2_K_XL shards + mmproj + DSpark Q8_0 drafter
    if not args.skip_download:
        gguf_dir.mkdir(parents=True, exist_ok=True)
        run([
            hf_bin, "download", args.source_repo,
            "--local-dir", str(gguf_dir),
            "--include", f"{args.quant_dir}/*.gguf",
            "--include", "mmproj-BF16.gguf",
            "--include", "*dspark*Q8_0.gguf",
            "--include", "dspark/*Q8_0.gguf",
        ])

    first_shard = find_single(
        gguf_dir,
        [f"{args.quant_dir}/*-00001-of-*.gguf", f"**/*{args.quant_dir}*-00001-of-*.gguf"],
        "first UD-Q2_K_XL GGUF shard",
    )
    mmproj = find_single(gguf_dir, ["mmproj-BF16.gguf", "**/mmproj-BF16.gguf"], "mmproj-BF16.gguf")
    dspark = find_single(gguf_dir, ["*dspark*Q8_0.gguf", "**/*dspark*Q8_0.gguf"], "DSpark Q8_0 GGUF")

    # 3. Convert to BASALTMF v2 pack + attach DSpark drafter
    convert_py = basalt_dir / "tools" / "convert" / "convert.py"
    run([
        sys.executable, str(convert_py),
        "--gguf", str(first_shard),
        "--mmproj", str(mmproj),
        "--dspark", str(dspark),
        "--out", str(out_path),
        "--name", "DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL",
        "--workers", str(args.workers),
        "--force",
    ])

    # 4. Verify pack integrity against source GGUFs
    if not args.skip_verify:
        run([sys.executable, str(convert_py), "--verify", str(out_path), "--gguf", str(first_shard)])

    if args.delete_sources and gguf_dir.exists():
        print(f"Removing source GGUFs at {gguf_dir} ...", flush=True)
        shutil.rmtree(gguf_dir)

    size_gib = out_path.stat().st_size / (1024 ** 3)
    print(f"\nBuilt {out_path} ({size_gib:.2f} GiB)", flush=True)

    # 5. Optional upload to personal Hugging Face repository
    if args.upload_repo:
        if Path(hf_bin).name == "hf":
            create_cmd = [hf_bin, "repo", "create", args.upload_repo, "--repo-type", "model", "--exist-ok"]
            if args.private:
                create_cmd.append("--private")
            subprocess.run(create_cmd, check=False)
        run([hf_bin, "upload", args.upload_repo, str(out_path), out_path.name, "--repo-type", "model"])
        url = f"https://huggingface.co/{args.upload_repo}/resolve/main/{out_path.name}"
        print("\nUpload complete! Configure basalt-colab with:")
        print(f'  basalt-colab up --quant ud-q2_k_xl --pack-url "{url}"')
        print("\nOr persist in ~/.config/basalt-colab/config.toml under [basalt]:")
        print('  quant = "ud-q2_k_xl"')
        print(f'  pack_url = "{url}"')


if __name__ == "__main__":
    main()
