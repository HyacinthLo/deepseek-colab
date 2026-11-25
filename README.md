<div align="center">

<h1>basalt-colab</h1>

<p><b>Uncensored DeepSeek-V4.1-Flash (UD-IQ1_M) on Google Colab G4 (RTX PRO 6000 96 GB), built from first principles around Basalt v0.5.7.</b><br>
150–190+ tok/s solo · 159 tok/s 2-stream concurrent · up to 1M elastic FP8 KV · 32 GiB Disk Conversation Cache · Dynamic Vision.<br>
Modular zero-dependency Python 3.11+ CLI · Unified Web Control Console · OpenAI / Anthropic Gateway · Claude Code, Codex &amp; opencode ready.</p>

</div>

## Install & Quick Start

Requires **Python 3.11+**, an OpenSSH client (`ssh`, `ssh-keygen`), and a Google AI Pro (200 CU/month) or Colab Pro (100 CU/month) subscription. Zero third-party Python dependencies.

```bash
# Install globally with uv (recommended)
uv tool install git+https://github.com/HyacinthLo/basalt-colab.git

# Or clone & install editable
git clone https://github.com/HyacinthLo/basalt-colab.git && cd basalt-colab
pip install --no-deps --no-build-isolation -e .
```

Start your Colab G4 endpoint (or open the Unified Control Console first without waking a GPU):

```bash
basalt-colab up            # sets up Colab CLI + OAuth on first run, boots G4 (~2-3 min)
basalt-colab console       # opens http://127.0.0.1:18080/_console (auto-authenticated)
```

```text
basalt-colab 0.1.0 · DeepSeek-V4.1-Flash (Basalt v0.5.7)

Starting a Colab G4 GPU · ~2-3 minutes (faster with prebuilt binary)

   0:11  ✓  G4 GPU allocated
   0:15  ✓  abliteration edit downloaded (43 KB)
   0:56  ✓  model downloaded (Basalt UD-IQ1_M)
   1:40  ✓  Basalt built / ready
   1:43  ✓  abliteration baked into weights (62/62)
   2:00  ✓  model loaded onto GPU + RAM
   2:00  ✓  tmpfs RAM punched & reclaimed
   2:32  ✓  Engram table loaded (110.6 GB)
   2:33  ✓  warmup answer done

✓ Ready after 2:39

  Console    http://127.0.0.1:18080/_console
  Web UI     http://127.0.0.1:18080/
  Base URL   http://127.0.0.1:18080/v1
  API key    sk-ds41-...
  Model      DeepSeek-V4.1-Flash-1017-UD-IQ1_M (effort 50)
  Engine     262K×2 FP8 KV · vision dynamic · ~160.4 tok/s

  Try it     basalt-colab ask "Hello!"
  Console    basalt-colab console
  Claude     basalt-colab claude
  Codex      basalt-colab codex
  opencode   basalt-colab opencode [project folder]
  Shell env  eval "$(basalt-colab env)"
  Stop       basalt-colab down
```

## Measured Performance (Colab G4 — RTX PRO 6000 Blackwell 96 GB)

| Metric | Measured (Basalt v0.5.7 + `UD-IQ1_M`) |
| --- | --- |
| VM cold start (source build / prebuilt) | **139s** (`~95s` with GitHub Release prebuilt `sm_120` binary) |
| Solo decode (`/v1/chat/completions`) | **148–160+ tok/s** (97.0% VRAM expert hit rate, 0.9% PCIe) |
| OpenAI Responses (`/v1/responses`) | **193.8 tok/s** (DSpark speculative draft + 32 draft experts) |
| Anthropic Messages (`/v1/messages`) | **152.5 tok/s** (native Claude Code compatible) |
| 2-stream concurrent decode | **159.0 tok/s aggregate** (`78.7 + 80.3 tok/s`, 94.1% hit, 0.8% PCIe) |
| Context & KV cache | **Up to 1,048,576 (1M) per slot** via `slot-full-context` (FP8 KV: 0.97 GiB @ 1M; FP16 optional) |
| Disk Conversation Cache | **32 GiB** `/content/conv_cache` (99.8% KV restored in **0.5 ms** across rotated sessions) |
| Vision (`dynamic` mode) | **0 MiB idle VRAM / 0 MiB idle RAM**, ~1.0–1.6s on-demand image encode |
| Client-disconnect cancel (`MSG_PEEK`) | **0.48s** measured GPU decode abort after client hangup |
| Steady-state host RAM free | **10 GiB available** after loading 110.6 GB Engram table |
| Cost | **~8.9 CU/hr** while GPU runs · **0 CU** when stopped |

## Unified Control Console (`/_console`) & Basalt Web UI (`/`)

- **Control Console (`/_console`)**: Self-contained dark UI (zero CDN deps) for starting/stopping G4 (with `--wait-gpu` toggle), live boot timeline, real-time Basalt `tok/s` + VRAM hit % telemetry, **14-day CU billing curve & session ledger** (`usage.jsonl`), **online `config.toml` editor with live hot-reload**, and local/VM log viewer. When G4 is off or booting, visiting `/` serves `/_console` directly.
- **Basalt Native Web UI (`/`)**: Once G4 is ready, `/` proxies 100% natively to Basalt's built-in chat & diagnostics console without rewriting any asset paths.
- Run `basalt-colab console` to open `/_console#key=...` auto-authenticated (the URL fragment is stored in `localStorage` and immediately stripped from the URL; never sent over HTTP).

## Commands

| Command | Description |
| --- | --- |
| `basalt-colab up [--quant Q] [--pack-url URL] [--uncensor-scale S] [--wait-gpu] [--basalt-ref REF]` | Start G4 (`--quant ud-iq1_m\|ud-iq2_xxs\|ud-q2_k_xl\|custom`; `--uncensor-scale` adjusts mrfakename strength; `--wait-gpu` retries up to 30 min; `--basalt-ref` builds an upstream Basalt tag/branch/SHA) |
| `basalt-colab down [--all]` | Stop G4, wipe VM secrets, record session in `usage.jsonl`, stop detached proxy |
| `basalt-colab console [--no-open]` | Ensure local proxy is up (without waking G4) and open `/_console` |
| `basalt-colab status` | Snapshot of Console/Web UI URLs, live Basalt telemetry, CU billing ledger, and Colab balance |
| `basalt-colab watch` | Live terminal status bar with `tok/s`, VRAM hit %, PCIe %, idle countdown, and CU cost |
| `basalt-colab config [show\|path\|wizard]` | Inspect or edit `config.toml` interactively (hot-reloads live fields without restart!) |
| `basalt-colab ask [opts] TEXT` | Stream an answer with `--max`, `--effort VAL`, `--image PATH` (repeatable), or piped `stdin` |
| `basalt-colab claude [...]` | Launch **Claude Code** pointed at Anthropic `/v1/messages` |
| `basalt-colab codex [...]` | Launch **Codex CLI** pointed at OpenAI `/v1/responses` |
| `basalt-colab opencode [...]` | Launch **opencode** with auto-generated `opencode.json` |
| `basalt-colab env [--powershell\|--sh]` | Print shell `export` or PowerShell `$env:` lines for OpenAI & Anthropic clients |
| `basalt-colab logs` | Local proxy log + VM boot timeline + Basalt server & engine logs |
| `basalt-colab doctor` | One-command check of Python 3.11+, OpenSSH, Colab CLI, OAuth balance, `config.toml`, VM payload & local port |
| `basalt-colab setup` | First-run setup + automatic one-time migration of legacy `./state/{api_key,id_ed25519}` |

## Quantization Tiers & Zero-SSD-Expert Staging

`basalt-colab` inspects the remote 64 KiB `.basalt` header before downloading and automatically plans storage so **0 routed experts spill to SSD**:

| Quant (`[basalt].quant`) | Pack Size | Storage Mode on G4 | Quality (vs BF16) | Notes |
| --- | --- | --- | --- | --- |
| **`ud-iq1_m`** *(default)* | 260 GB | **`ram+ram`** (100% RAM+VRAM) | KLD `0.2779` · Top-1 `83.88%` | Fastest (`150–190+ tok/s`), zero SSD I/O |
| **`ud-iq2_xxs`** | 285 GB | **`ram+ssd`** (Experts 100% RAM+VRAM; Engram on `/content` SSD + ~85 GiB RAM Page Cache) | KLD `0.2408` (**-13.4%**) · Top-1 `85.15%` | Official pack on `jesdga`, higher fidelity |
| **`ud-q2_k_xl`** *(experimental)* | ~335 GB | **`ssd+ssd`** (Requires `/content` punch-hole + self-hosted `pack_url`) | Highest 2-bit tier | Build & upload with [`scripts/build_q2k_xl_pack.py`](scripts/build_q2k_xl_pack.py) |
| `NVFP4` *(unsupported)* | 428 GB | **Rejected fail-fast** | — | Backbone (`~294 GiB`) > G4 VRAM+RAM (`~272 GiB`); Basalt requires ≥320 GB RAM |

## Declarative Configuration (`config.toml`)

Stored at `~/.config/basalt-colab/config.toml` (`%APPDATA%\basalt-colab\config.toml` on Windows; override via `BASALT_COLAB_HOME`):

```toml
[colab]
session = "basalt-ds41"
idle_min = 10            # hot-reloadable live
max_hours = 6            # hot-reloadable live
min_balance = 5
wait_gpu = false         # hot-reloadable live
orphan_boot_s = 30

[proxy]
bind = "127.0.0.1"
port = 18080
strict_model = false     # hot-reloadable live (false = route claude-haiku-* / gpt-* -> active model)

[basalt]
quant = "ud-iq1_m"       # "ud-iq1_m", "ud-iq2_xxs", "ud-q2_k_xl", or "custom"
pack_url = ""            # optional URL override (required for ud-q2_k_xl / custom)
ref = ""                 # empty = pinned manifest (v0.5.7 + prebuilt fast path)
model_alias = ""         # empty = auto from quant (e.g. DeepSeek-V4.1-Flash-1017-UD-IQ1_M)
max_context = 262144     # per-slot up to 1048576 (1M) via native slot-full-context
parallel = 2             # 1..4 concurrent slots
kv = "fp8"               # "fp8" or "fp16"
vision = "dynamic"       # 0 MiB idle VRAM/RAM
spec = 6                 # DSpark speculative window (0..8)
draft_experts = 32       # 32, 64, 96, 128, or 0
conv_cache_mib = 32768   # 32 GiB disk conversation cache
resident_budget_gib = 0  # 0 = auto-planned per storage mode (55 / ~87 / ~130 GiB)
uncensor = true          # 43 KB mrfakename rank-2 stochastic-rounding abliteration (0% decode slowdown)
uncensor_scale = 1.0     # strength multiplier (0.0=off, 1.0=default KL 0.067, 1.15-1.25=stronger)
effort = "50"            # hot-reloadable live (1-100, low, high, max)

[basalt.sampling]
# temperature = 1.0
# top_p = 0.95
```

## Architecture & 6-Layer Cost Safety

See [docs/how-it-works.md](docs/how-it-works.md) and [SETUP.md](SETUP.md) for deep dives into 3-mode automatic storage staging, in-place rank-2 abliteration, the offline `UD-Q2_K_XL` pack converter (`scripts/build_q2k_xl_pack.py`), the GitHub Actions prebuilt `sm_120` release pipeline (`.github/workflows/build-basalt-prebuilt.yml`), and the 6-layer cost protection system.

## Credits

Built on [Basalt](https://github.com/jesdga95/basalt) by jesdga95, DeepSeek-V4.1-Flash by DeepSeek, Unsloth's Dynamic V3.1 quantizations (`UD-IQ1_M` / `UD-IQ2_XXS` / `UD-Q2_K_XL`), mrfakename's 43 KB rank-2 `abliteration-edit.npz`, Google's Colab CLI, and inspired by [leonxger/deepseek-colab](https://github.com/leonxger/deepseek-colab).
