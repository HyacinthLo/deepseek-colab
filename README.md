<div align="center">

<h1>DeepSeek V4.1 Flash on Colab G4 (Basalt Native)</h1>

<p><b>Uncensored DeepSeek-V4.1-Flash (UD-IQ1_M) on a single Google Colab G4 (RTX PRO 6000 96 GB), powered by Basalt v0.5.7.</b><br>
150–190+ tok/s solo · 159 tok/s 2-stream concurrent · 262K elastic FP8 KV · 32 GiB Disk Conversation Cache · Dynamic Vision.<br>
OpenAI Chat &amp; Responses, Anthropic Messages, Claude Code, Codex CLI, opencode, and Basalt Web UI — all behind one local endpoint.</p>

</div>

## Quick start

Requires Python 3.9+, an OpenSSH client, and a Google AI Pro (200 CU/month) or Colab Pro (100 CU/month) subscription. Runs on Linux, macOS, and Windows.

```bash
git clone -b basalt-native https://github.com/HyacinthLo/deepseek-colab.git deepseek-colab && cd deepseek-colab
python3 llm.py up
```

The first run installs `google-colab-cli`, completes one-time Google OAuth login, lets you confirm defaults, and boots a Colab G4 in **~3 minutes**:

```text
Starting a Colab G4 GPU · about 2-3 minutes

   0:11  ✓  G4 GPU allocated
   1:08  ✓  model downloaded (Basalt UD-IQ1_M)
   1:39  ✓  Basalt v0.5.7 built
   1:41  ✓  abliteration baked into weights (62/62)
   1:59  ✓  model loaded onto GPU + RAM
   2:00  ✓  tmpfs RAM punched & reclaimed
   2:33  ✓  Engram table loaded (110.6 GB)
   2:34  ✓  warmup answer done

✓ Ready after 2:59

  Web UI     http://127.0.0.1:18080/
  Base URL   http://127.0.0.1:18080/v1
  API key    sk-ds41-...
  Model      DeepSeek-V4.1-Flash-1017-UD-IQ1_M (effort 50)
  Context    262,144 tokens/slot (2 slots) · dynamic vision on · ~160.4 tok/s

  Try it     python3 llm.py ask "Hello!"
  Claude     python3 llm.py claude
  Codex      python3 llm.py codex
  opencode   python3 llm.py opencode [project folder]
  Shell env  eval "$(python3 llm.py env)"
  Stop       python3 llm.py down

  ● GPU running   idle 0:12 · stops in 9:48   160.4 tok/s · hit 96% · pcie 0.8% · ran 3:11 · about 0.47 compute units
```

## Measured Performance (Colab G4 — RTX PRO 6000 Blackwell 96 GB)

| Metric | Measured (Basalt v0.5.7 + `UD-IQ1_M`) |
| --- | --- |
| End-to-end cold start | **2:59** (`143s` VM boot; optional `BASALT_PREBUILT_URL` skips ~80s build) |
| Solo decode (`/v1/chat/completions`) | **151–160+ tok/s** (96.5% VRAM expert hit rate, <1% PCIe) |
| OpenAI Responses (`/v1/responses`) | **193.8 tok/s** (speculative DSpark draft + 32 draft experts) |
| Anthropic Messages (`/v1/messages`) | **152.5 tok/s** (native Claude Code compatible) |
| 2-stream concurrent decode | **159.0 tok/s aggregate** (`78.7 + 80.3 tok/s`, 94.1% hit, 0.8% PCIe) |
| Context & KV cache | **262,144 tokens/slot × 2 slots** (524K total, elastic FP8 KV = 0.24 GiB idle) |
| Disk Conversation Cache | **32 GiB** `/content/conv_cache` (99.8% KV restored in **0.5 ms** across rotated sessions) |
| Vision (`dynamic` mode) | **0 MiB idle VRAM**, ~1.0s on-demand image encode |
| Steady-state host RAM free | **~10 GiB available** after loading 110.6 GB Engram table |
| Cost | **~8.9 CU/hr** while GPU runs · **0 CU** when stopped |

## Clients & CLI Commands

| Command | Description |
| --- | --- |
| `python3 llm.py up [--wait-gpu]` | Start the local proxy & G4 GPU (`--wait-gpu` retries with exponential backoff up to 30 min if Colab is busy) |
| `python3 llm.py down` | Stop the GPU, wipe VM secrets, log CU cost, and stop detached proxy |
| `python3 llm.py watch` | Live terminal status bar with Basalt `tok/s`, VRAM hit rate, PCIe %, idle countdown, and CU cost |
| `python3 llm.py status` | Snapshot of Web UI URL, endpoint, live Basalt telemetry, and Colab CU balance |
| `python3 llm.py settings` | Edit defaults interactively — `effort`, `idle_min`, and `max_hours` **hot-reload live** without restarting G4 |
| `python3 llm.py ask [opts] TEXT` | Stream an answer with `--max`, `--effort VAL`, `--image PATH` (repeatable), or piped `stdin` |
| `python3 llm.py claude [...]` | Launch **Claude Code** pointed at `/v1/messages` (`ANTHROPIC_*` env pre-wired) |
| `python3 llm.py codex [...]` | Launch **Codex CLI** pointed at `/v1/responses` (`OPENAI_*` env pre-wired) |
| `python3 llm.py opencode [...]` | Launch **opencode** with auto-generated `state/opencode.json` |
| `python3 llm.py env [--powershell]` | Emit `export` (or PowerShell `$env:`) lines for any OpenAI / Anthropic tool |
| `python3 llm.py logs` | Show local proxy log + VM boot timeline + Basalt server & engine logs |

### Built-in Web UI & Multi-Protocol Proxy

- **Basalt Web UI**: Open `http://127.0.0.1:18080/` in your browser once the GPU is ready for interactive chat, live slot telemetry, and metrics.
- **Single-model fallback**: By default, requests carrying other model names (such as Claude Code's `claude-haiku-*` / `claude-sonnet-*` background subagent calls) automatically route to `DeepSeek-V4.1-Flash-1017-UD-IQ1_M` instead of returning 404. Set `LLM_STRICT_MODEL=1` to disable.
- **CORS preflight**: `OPTIONS` preflight (`204`) and `Access-Control-Allow-Origin: *` are handled natively for browser frontends (Open WebUI, LobeChat, etc.) while loopback `Host` header validation continues to block DNS rebinding.
- **Multi-protocol SSE errors**: Cold-start heartbeats report live boot stages (`: starting gpu vm (5 s) - building Basalt`), and mid-stream errors format natively for OpenAI Chat (`data: {"error":...}` + `[DONE]`), OpenAI Responses (`event: error`), and Anthropic Messages (`event: error`).

## Architecture & Safety

1. **Monolithic Basalt pack + Punch-on-Read**: Unsloth's 249.6 GiB `DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt` bundles weights, DSpark draft, vision encoder, and 110.6 GiB Engram table in one `BASALTMF v2` file. [`vm/basalt_jobs.py`](vm/basalt_jobs.py) splits it into backbone (139.1 GiB) + Engram (110.6 GiB). While Basalt loads experts onto GPU VRAM + resident RAM, [`vm/patch_basalt.py`](vm/patch_basalt.py) injects `fallocate(FALLOC_FL_PUNCH_HOLE)` on read and punches `dspark/gguf` + `dense`, shrinking `/mnt/ram/model.basalt` from 139.1 GiB down to **1.6 GiB** before streaming the 110.6 GiB Engram table into `/mnt/ram/engram.bin`.
2. **43 KB In-Place Stochastic-Rounding Abliteration**: Instead of downloading gigabytes of patched tensors, [`vm/abliterate_basalt.py`](vm/abliterate_basalt.py) + [`vm/abliterate.c`](vm/abliterate.c) applies mrfakename's 43 KB rank-2 `abliteration-edit.npz` directly to all 62/62 target tensors (`Q8_0`/`Q5_K`/`Q6_K`) inside the `.basalt` pack in **1.5 seconds**.
3. **6-Layer Cost & Reliability Protection**:
   - **Client-disconnect GPU cancel**: `MSG_PEEK` socket watcher aborts the upstream connection within **0.5s** when a client disconnects or hits Ctrl+C/Esc, cancelling in-flight Prefill/Decode immediately.
   - **Orphan cold-start cancel**: If a request-triggered boot loses all waiting clients for >30s, the boot cancels automatically.
   - **Idle auto-stop**: Stops the VM after `idle_min` (default 10 min) with zero active requests; the next request wakes it seamlessly.
   - **Hard cap**: Stops the VM at `max_hours` (default 6 h).
   - **Detached local watchdog**: Survives proxy crashes/SIGKILL and stops the VM on idle or hard cap.
   - **VM-side hard-cap backstop**: Independent timer inside `vm/boot.sh` terminates Basalt and wipes `/content/secrets` if your local machine sleeps or loses power.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PORT` | `18080` | Local proxy port |
| `LLM_BIND` | `127.0.0.1` | Bind interface |
| `LLM_MODEL_ALIAS` | `DeepSeek-V4.1-Flash-1017-UD-IQ1_M` | Served model ID |
| `LLM_STRICT_MODEL` | `0` | Set `1` to disable automatic routing of unknown model names |
| `LLM_IDLE_MIN` | `10` | Idle auto-stop timeout (minutes) |
| `LLM_MAX_HOURS` | `6` | Hard-stop cap (hours) |
| `LLM_IMAGES` / `LLM_CONTEXT` | `1` / `262144` | Override dynamic vision / context per slot |
| `LLM_WAIT_GPU` | `0` | Set `1` (or pass `--wait-gpu`) for 30-min exponential backoff allocation |
| `HF_TOKEN` | — | Optional Hugging Face token (sent to VM only, redacted from logs) |
| `BASALT_PREBUILT_URL` | — | Optional URL to a prebuilt `basalt` sm_120 binary (skips ~80s compile) |
| `BASALT_PREBUILT_SHA256` | — | Optional SHA-256 verification for `BASALT_PREBUILT_URL` |

## Credits

Built on [Basalt](https://github.com/jesdga95/basalt) by jesdga95, DeepSeek-V4.1-Flash by DeepSeek, the `UD-IQ1_M.basalt` pack by Unsloth, rank-2 `abliteration-edit.npz` by mrfakename, Google's Colab CLI, and originally forked from [leonxger/deepseek-colab](https://github.com/leonxger/deepseek-colab).
