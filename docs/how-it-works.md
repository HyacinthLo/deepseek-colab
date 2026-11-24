# How It Works (`basalt-native`)

DeepSeek V4.1 Flash has 552B parameters across 40 layers plus a 196B-parameter Engram lookup table (249.6 GiB total in Unsloth's `UD-IQ1_M.basalt` monolithic pack). A single Google Colab G4 VM provides one NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), ~190 GB host RAM, and slow local SSD (~250 MB/s). Here is how `basalt-native` boots in **~3 minutes** and serves at **150–190+ tok/s** with **~10 GiB host RAM free**.

## 1. Monolithic Pack Splitting & Punch-on-Read

Unsloth's `DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt` (`BASALTMF v2`) bundles all sections in one file:
- `meta`, `gguf` (attention + dense), `dspark/gguf` (draft model), `vision/mmproj` (vision encoder), `experts` (~122.4 GiB) → **139.1 GiB backbone**
- `engram` (IQ4_NL lookup table) → **110.6 GiB Engram**

Because 139.1 GiB + 110.6 GiB = 249.6 GiB exceeds 190 GiB host RAM, `vm/boot.sh` orchestrates a zero-SSD two-phase tmpfs pipeline:
1. [`vm/basalt_jobs.py`](../vm/basalt_jobs.py) reads the 64 KB `BASALTMF v2` header via HTTP range request and splits it into backbone jobs (`/mnt/ram/model.basalt`, 139.1 GiB) and Engram jobs (`/mnt/ram/engram.bin`, 110.6 GiB).
2. In parallel: 64 workers download the 139.1 GiB backbone at ~3.2 GB/s (~50s), `curl` fetches the 43 KB abliteration edit, and [`vm/build.sh`](../vm/build.sh) compiles patched Basalt v0.5.7 (`sm_120`, `-DBASALT_MODELS=deepseek41`) + standalone `abliterate` (~80s; or skips compilation when `BASALT_PREBUILT_URL` is configured).
3. An empty 0-block sparse `/mnt/ram/engram.bin` (`ftruncate`) is created so Basalt opens `PleReader` cleanly before Engram bytes arrive.
4. `basalt serve` starts and receives `POST /load`. As Basalt loads experts onto GPU VRAM (~87 GiB) and resident pinned host RAM (~55 GiB budget), patch #2 in [`vm/patch_basalt.py`](../vm/patch_basalt.py) calls `fallocate(FALLOC_FL_PUNCH_HOLE)` on each read chunk (`BASALT_PUNCH_FILE`). Immediately after `/load`, `basalt_jobs.py --punch-sections dspark/gguf dense` reclaims another 8 GiB, shrinking `/mnt/ram/model.basalt` from 139.1 GiB down to **1.6 GiB** (~140 GiB reclaimed in ~12s).
5. Now that ~120 GiB of host RAM is free, 64 workers stream the 110.6 GiB Engram table into `/mnt/ram/engram.bin` (~34s), leaving **~10 GiB host RAM available** at steady state.
6. A warmup request validates end-to-end generation, and [`vm/gate.py`](../vm/gate.py) opens port 18080 so no request ever touches a half-loaded model.

## 2. 43 KB In-Place Stochastic-Rounding Abliteration

Instead of downloading 1.7 GB of pre-modified tensors, [`vm/abliterate_basalt.py`](../vm/abliterate_basalt.py) + [`vm/abliterate.c`](../vm/abliterate.c) applies mrfakename's 43 KB rank-2 `abliteration-edit.npz` (SHA-256 verified) directly to all **62/62 target tensors** (`Q8_0`, `Q5_K`, `Q6_K`) inside `/mnt/ram/model.basalt` in **1.5 seconds** using OpenMP + stochastic rounding, with zero `llama.cpp` or `libggml` dependencies. Set `uncensor: false` (or `UNCENSOR=0`) to keep original weights.

## 3. Why Basalt v0.5.7 Is Faster & Cleaner

- **94–97% VRAM Expert Hit Rate (<1% PCIe):** Basalt's native Blackwell `sm_120` kernels, expert hotness residency, and compact `UD-IQ1_M` footprint pack far more active experts into 96 GB VRAM than legacy GGUF.
- **Elastic FP8 KV Cache:** 262,144 tokens/slot × 2 parallel slots allocates only **0.24 GiB** at idle and grows elastically on demand.
- **32 GiB Disk Conversation Cache (`/content/conv_cache`):** Evicted KV sessions spill to Colab SSD as `.bcs` snapshots and restore in **<1 ms** (99.8% cache hit measured across rotated multi-turn conversations).
- **Dynamic Vision (`0 MiB` idle VRAM):** The vision encoder stays unloaded until an image arrives, encodes in ~1s, and immediately returns VRAM to the expert cache.
- **4 Minimal Patches (`vm/patch_basalt.py`):**
  1. Split `BASALT_ENGRAM_FILE` support in `PleReader`.
  2. `BASALT_PUNCH_FILE` punch-on-read in weight loader.
  3. Colab RAM headroom guard in `auto_plan`.
  4. `BASALT_COLAB_PUNCH_GUARD` in `serve/service.py` blocking `POST /unload` and preventing silent engine respawn against a punched tmpfs file.

## 4. Proxy Scheduling & 6-Layer Cost Safety (`llm.py`)

- **Protocol & Routing:** Serves OpenAI `/v1/chat/completions`, OpenAI `/v1/responses`, Anthropic `/v1/messages`, Basalt Web UI (`GET /`), and `OPTIONS` CORS preflight. Automatically maps unknown model names (e.g. Claude Code's `claude-haiku-*`) to `DeepSeek-V4.1-Flash-1017-UD-IQ1_M` unless `LLM_STRICT_MODEL=1`.
- **0.5s Client-Disconnect Cancellation:** `select` + `MSG_PEEK` watches every active and cold-starting client socket; if a client hangs up or hits Ctrl+C/Esc, `llm.py` immediately `shutdown(SHUT_RDWR)`s the upstream socket so Basalt aborts GPU Prefill/Decode within 0.5s.
- **6-Layer Cost Protection:**
  1. Orphan cold-start auto-cancel (>30s with 0 waiting clients on request-triggered boot).
  2. Idle auto-stop (`idle_min`, default 10 min; hot-reloadable live via `POST /_llm/reload`).
  3. Proxy hard cap (`max_hours`, default 6 h; hot-reloadable live).
  4. Detached local `_watchdog` process (survives proxy SIGKILL).
  5. VM-side hard-cap backstop in `vm/boot.sh` (kills Basalt + wipes `/content/secrets` if your laptop sleeps).
  6. Proxy-mode auto-create detector (`drain()` immediately stops any session accidentally created by `colab ssh --proxy-mode`).
