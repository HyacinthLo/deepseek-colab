# How `basalt-colab` Works

`basalt-colab` is engineered from first principles around **Basalt v0.5.7** and **DeepSeek-V4.1-Flash (`UD-IQ1_M`)** on a single Google Colab G4 (RTX PRO 6000 Blackwell 96 GB VRAM, ~176 GiB usable host RAM, slow local SSD).

## 1. Zero-SSD Two-Phase tmpfs Pipeline (`basalt_colab/vm/boot.sh`)

Unsloth's 249.6 GiB `DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt` (`BASALTMF v2`) exceeds 176 GiB host RAM when stored whole. `basalt_colab/vm/basalt_jobs.py` reads the 64 KiB header via HTTP range request and splits it into:
1. **Backbone (`/mnt/ram/model.basalt`, 139.1 GiB)**: `meta`, `gguf`, `dspark/gguf`, `vision/mmproj`, `experts`.
2. **Engram (`/mnt/ram/engram.bin`, 110.6 GiB)**: `engram` IQ4_NL lookup table.

Boot sequence (`139s` source build; `~95s` with GitHub Actions prebuilt `sm_120` tarball):
1. **Parallel**: 64-worker download of the 139.1 GiB backbone (`~42s`), 43 KB rank-2 `abliteration-edit.npz` fetch (`~1s`), and `vm/build.sh` (fetches prebuilt `basalt` + `abliterate` tarball with SHA-256 verification, or falls back to source compilation / `--basalt-ref`).
2. **In-place Abliteration (`~1.5–3s`)**: `abliterate_basalt.py` + standalone OpenMP `abliterate.c` applies mrfakename's 43 KB rank-2 edit directly across all 62/62 target `Q8_0`/`Q5_K`/`Q6_K` tensors via stochastic rounding.
3. **Load & Punch-on-Read (`~17s`)**: `POST /load` moves hot experts to VRAM (~87 GiB) and resident complement to pinned host RAM (55 GiB budget). Patch #3 (`BASALT_PUNCH_FILE`) calls `fallocate(FALLOC_FL_PUNCH_HOLE)` as chunks load, and `basalt_jobs.py --punch-sections dspark/gguf dense` reclaims another 8 GiB — shrinking `/mnt/ram/model.basalt` from 139.1 GiB down to **1.6 GiB**.
4. **Engram Stream (`~32s`)**: Streams the 110.6 GiB Engram table into `/mnt/ram/engram.bin`, leaving **~10 GiB host RAM free**.
5. **Warmup & Gate (`~1s`)**: Verifies generation, opens `gate.py` on `:18080`, and arms the VM-side hard-cap backstop (`MAX_HOURS * 3600 + 300s`).

## 2. Native Basalt v2 Configuration (`basalt_colab/vm/serve.sh`)

- **`slot-full-context: true`**: Each slot receives the full `basalt.max_context` (up to `1,048,576` / 1M tokens) without `llama.cpp`-style `parallel × context` multiplication.
- **Elastic FP8/FP16 KV**: Default `fp8` uses 920 B/token (`0.24 GiB` idle, `0.97 GiB` at 1M); `fp16` keeps unrounded KV (`3,200 B/token`).
- **Dynamic Vision**: `vision = "dynamic"` consumes **0 MiB idle VRAM and 0 MiB idle RAM** (identical memory plan to `vision = "off"`).
- **32 GiB Disk Conversation Cache**: `/content/conv_cache` snapshots evicted KV slots as `.bcs` files and restores them in `<1 ms` (99.8% hit verified across rotated multi-turn sessions).
- **4 Minimal Patches (`patch_basalt.py`)**: `BASALT_COLAB_ENGRAM_ENV`, `BASALT_COLAB_TMPFS_FALLBACK`, `BASALT_COLAB_PUNCH_ON_READ`, and `BASALT_COLAB_PUNCH_GUARD`.

## 3. Enhanced Local Gateway & 6-Layer Cost Protection

- **Unified Control Console (`/_console`) + Native Basalt Web UI (`/`)**: `/` proxies untouched to Basalt's built-in console once G4 is ready, and shows `/_console` while G4 is off/starting. `/_console` is authenticated with `api_key` (auto-loaded from `#key=` fragment by `basalt-colab console`), renders live Basalt telemetry, plots 14-day CU billing curves from `usage.jsonl`, and hot-reloads `config.toml` online.
- **6-Layer Cost Safety**:
  1. `0.5s` `MSG_PEEK` client-disconnect cancellation (aborts active GPU Prefill/Decode AND unwakes cold-start waiters).
  2. Orphan cold-start auto-cancel (`orphan_boot_s`, default `30s`).
  3. Idle auto-stop (`idle_min`, default `10 min`, hot-reloadable live).
  4. Proxy hard cap (`max_hours`, default `6 h`, hot-reloadable live).
  5. Detached local `_watchdog` process (survives proxy SIGKILL).
  6. VM-side hard-cap timer in `boot.sh` (kills Basalt and wipes `/content/secrets` if local machine sleeps).
