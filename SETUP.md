# Setup and Reference (`basalt-native`)

Everything beyond the quick start: platforms, commands, clients (Claude Code, Codex CLI, opencode, Web UI), live settings hot-reload, environment variables, and troubleshooting.

## What you need

- **Google AI Pro or Colab Pro.** Google AI Pro includes 200 Colab compute units every month (~22 GPU hours), Colab Pro 100 (~11 GPU hours). Colab G4 (RTX PRO 6000 Blackwell 96 GB VRAM, 190 GB host RAM) bills about 8.9 CU/hr.
- **Python 3.9 or newer.** `llm.py` uses only the Python standard library.
- **An OpenSSH client** (`ssh`, `ssh-keygen`). Built into macOS, Windows 10/11, and most Linux desktops.
- **Google's Colab CLI** (`google-colab-cli==0.7.5`). `llm.py` offers to install it on the first run via `uv` or `pipx`.

Nothing heavy runs on your machine. The model download, Basalt build, abliteration, and inference all run on the Colab G4 VM.

## Install

### Linux / macOS
```bash
git clone -b basalt-native https://github.com/HyacinthLo/deepseek-colab.git deepseek-colab && cd deepseek-colab
python3 llm.py up
```

### Windows (PowerShell or cmd)
```powershell
git clone -b basalt-native https://github.com/HyacinthLo/deepseek-colab.git deepseek-colab
cd deepseek-colab
python llm.py up
```

### First run
`python3 llm.py up` runs once:
1. Checks for `ssh` and `ssh-keygen`.
2. Offers to install `google-colab-cli==0.7.5` (set `LLM_YES=1` to skip prompts).
3. Runs one-time Google OAuth login (cached in `~/.config/colab-cli/`).
4. Generates `state/api_key`, `state/id_ed25519`, and `state/opencode.json`.
5. Prompts for your defaults (`images`, `context`, `effort`, `idle_min`, `max_hours`, `uncensor`).

## Commands

| Command | What it does |
| --- | --- |
| `python3 llm.py up [--wait-gpu]` | Set up if needed, start the G4 GPU, wait until ready (~3 min), print Web UI / URL / key, and open live status (`--wait-gpu` retries with backoff up to 30 min). |
| `python3 llm.py down` | Stop the G4 GPU, wipe `/content/secrets` on the VM, and stop the background proxy. |
| `python3 llm.py status` | Web UI URL, endpoint state, model/effort, live Basalt telemetry (`tok/s`, VRAM hit %, PCIe %), and Colab balance. |
| `python3 llm.py watch` | Live terminal status bar: GPU state, live `tok/s`, VRAM hit %, idle countdown, and CU cost. |
| `python3 llm.py settings` | Edit saved defaults interactively. `effort`, `idle_min`, and `max_hours` **hot-reload live** via `POST /_llm/reload`. |
| `python3 llm.py ask [opts] TEXT` | Stream a one-off question. Supports `--max`, `--effort VAL`, `--image PATH` (repeatable), and piped `stdin`. |
| `python3 llm.py claude [...]` | Start Claude Code pre-wired to Anthropic `/v1/messages`. |
| `python3 llm.py codex [...]` | Start Codex CLI pre-wired to OpenAI `/v1/responses`. |
| `python3 llm.py opencode [...]` | Start opencode with `state/opencode.json`. |
| `python3 llm.py env [--powershell]` | Print shell `export` (or PowerShell `$env:`) commands for OpenAI & Anthropic clients. |
| `python3 llm.py logs` | Local proxy log plus the VM's boot timeline, Basalt server log, and engine log. |
| `python3 llm.py serve` | Run the endpoint in the foreground (`Ctrl+C` stops the GPU too). |
| `python3 llm.py setup` | Install and log in only, without starting a GPU. |

## Connecting Clients

- **Web UI**: `http://127.0.0.1:18080/` (served directly by Basalt once the GPU is ready; paste the API key from `state/api_key` when prompted).
- **OpenAI Chat & Responses**: Base URL `http://127.0.0.1:18080/v1`, Model `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`.
- **Anthropic Messages**: Base URL `http://127.0.0.1:18080`, `x-api-key` or `Authorization: Bearer`, Model `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`.
- **Single-model fallback**: Any unrecognized model name (such as Claude Code's `claude-haiku-*` or `claude-sonnet-*` background calls) automatically routes to `DeepSeek-V4.1-Flash-1017-UD-IQ1_M` unless `LLM_STRICT_MODEL=1`.
- **Browser UIs (CORS)**: `OPTIONS` preflight and `Access-Control-Allow-Origin: *` are built in for Open WebUI, LobeChat, NextChat, etc.

### curl example
```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Authorization: Bearer $(cat state/api_key)" -H "Content-Type: application/json" \
  -d '{"model": "DeepSeek-V4.1-Flash-1017-UD-IQ1_M", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Multimodal CLI example
```bash
python3 llm.py ask --image screenshot.png "Explain the error in this trace"
git diff | python3 llm.py ask --max "Review this diff for edge cases"
```

## Settings & Environment Variables

| Setting | Default | Hot-reload? | Notes |
| --- | --- | --- | --- |
| `images` | `true` | Next restart | Dynamic Vision (`0 MiB` idle VRAM, on-demand ~1s image encode) |
| `context` | `262144` | Next restart | Per-slot context (`32K`–`262K`, 2 parallel slots, elastic FP8 KV) |
| `effort` | `50` | **Yes (live)** | `1`–`100`, `low`, `high`, or `max` (request's own `reasoning_effort` always wins) |
| `idle_min` | `10` | **Yes (live)** | Minutes without requests before automatic GPU stop |
| `max_hours` | `6` | **Yes (live)** | Hard cap hours (also enforced by detached watchdog and VM-side timer) |
| `uncensor` | `true` | Next restart | In-place 43 KB rank-2 stochastic-rounding abliteration (62/62 tensors in 1.5s) |

## Files

```text
llm.py                   local proxy, CLI, lifecycle manager, hot-reload, watchdog, telemetry
vm/boot.sh               G4 cold start: parallel download + build, abliterate, punch-on-read, Engram, warmup, VM backstop
vm/build.sh              clones/patches Basalt v0.5.7 (or fetches BASALT_PREBUILT_URL), builds sm_120 + standalone abliterate
vm/serve.sh              generates Basalt v2 config (elastic FP8 KV, Dynamic Vision, 32 GiB disk Conversation Cache)
vm/patch_basalt.py       4 minimal Colab patches (split Engram, punch-on-read, RAM guard, unload/respawn guard)
vm/basalt_jobs.py        BASALTMF v2 parser, backbone/Engram job splitter, section puncher
vm/abliterate_basalt.py  extracts 62/62 tensor targets from BASALTMF meta + mrfakename 43 KB edit
vm/abliterate.c          standalone OpenMP Q8_0/Q5_K/Q6_K stochastic-rounding rank-2 editor
vm/pfetch.py             64-worker parallel range fetcher
vm/gate.py               port 18080 gate (opens only after warmup succeeds)
tests/                   offline unit + integration test suite (fake Colab CLI + fake SSH + mock Basalt)
state/                   local API key, SSH key, settings, opencode.json, logs (never committed)
```

Run offline tests: `python3 -B -m unittest -v tests/test_llm.py`
