# How `basalt-colab` Works

`basalt-colab` is engineered from first principles around **Basalt v0.5.7** and **DeepSeek-V4.1-Flash** (`UD-IQ1_M`, `UD-IQ2_XXS`, and experimental self-hosted `UD-Q2_K_XL`) on a single Google Colab G4 (RTX PRO 6000 Blackwell 96 GB VRAM, ~176 GiB usable host RAM, ~220 GiB local SSD).

## 1. Adaptive 3-Mode Storage Pipeline (`basalt_colab/vm/boot.sh`)

Before downloading any weights, `basalt_colab/vm/basalt_jobs.py --plan` reads only the 64 KiB `BASALTMF v2` header via HTTP Range request and automatically chooses a zero-SSD-expert staging plan against the VM's real RAM, VRAM, and `/content` free space:

1. **`ram+ram` (`UD-IQ1_M`, 139.1 GiB backbone + 110.6 GiB Engram)**: Backbone downloads into `/mnt/ram/model.basalt`, shrinks to `~1.6 GiB` via punch-on-read during `POST /load`, then Engram streams into `/mnt/ram/engram.bin`. **100% RAM+VRAM, 0 SSD I/O.**
2. **`ram+ssd` (`UD-IQ2_XXS`, 154.8 GiB backbone + 110.6 GiB Engram)**: Backbone still downloads into `/mnt/ram/model.basalt` and punches down to `~1.6 GiB` during `POST /load` with an auto-raised `resident-budget-gib` (~87 GiB) so **100% of routed experts stay in VRAM+RAM (0 cold experts on SSD)**. Only `engram.bin` (288 B random read/token) streams to `/content/engram.bin`, backed by ~85 GiB of Linux Page Cache.
3. **`ssd+ssd` (experimental self-hosted `UD-Q2_K_XL`, ~201 GiB backbone + 110.6 GiB Engram)**: Probes `FALLOC_FL_PUNCH_HOLE` on `/content` first, stages backbone on `/content/model.basalt`, punches ~200 GiB free during `POST /load` (with auto-raised `resident-budget-gib` ~130 GiB), then streams `engram.bin` onto `/content/engram.bin`.
4. **Fail-fast guard**: Any pack whose backbone exceeds `VRAM + RAM` zero-SSD-expert capacity (e.g. `NVFP4`, ~294 GiB backbone) is rejected in 1 second before spending CU.

Boot sequence (`~139s` source build on `UD-IQ1_M`; `~95s` with GitHub Actions prebuilt `sm_120` tarball):
1. **Parallel**: 64-worker download of the sparse backbone, 43 KB rank-2 `abliteration-edit.npz` fetch (`~1s`), and `vm/build.sh` (fetches prebuilt `basalt` + `abliterate` tarball with SHA-256 verification, or falls back to source compilation / `--basalt-ref`).
2. **In-place Abliteration (`~1.5–3s`)**: `abliterate_basalt.py --scale $UNCENSOR_SCALE` + standalone OpenMP `abliterate.c` applies mrfakename's 43 KB rank-2 edit directly across all 62/62 target `Q8_0`/`Q5_K`/`Q6_K` tensors via stochastic rounding (0% decode slowdown).
3. **Load & Punch-on-Read (`~17s`)**: `POST /load` moves hot experts to VRAM (~87 GiB) and resident complement to pinned host RAM. Patch #3 (`BASALT_PUNCH_FILE`) calls `fallocate(FALLOC_FL_PUNCH_HOLE)` as chunks load, and `basalt_jobs.py --punch-sections dspark/gguf dense` reclaims another 8 GiB — shrinking the backbone file down to **~1.6 GiB**.
4. **Engram Stream**: Streams the 110.6 GiB Engram table into `/mnt/ram/engram.bin` (`ram+ram`) or `/content/engram.bin` (`ram+ssd` / `ssd+ssd`).
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
