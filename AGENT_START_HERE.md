# AGENT_START_HERE (`basalt-native`)

You are an AI agent helping someone use this repo. Read this file fully; it is enough to set the project up, run it, connect apps (Claude Code, Codex CLI, opencode, Web UI, OpenAI/Anthropic SDKs), and troubleshoot issues. See also `SETUP.md` and `README.md`.

## What this is

`llm.py` turns a Google Colab G4 GPU (RTX PRO 6000 Blackwell 96 GB VRAM, 190 GB RAM) into a private local OpenAI + Anthropic + Web UI endpoint serving uncensored **DeepSeek-V4.1-Flash (`DeepSeek-V4.1-Flash-1017-UD-IQ1_M`)** on **Basalt v0.5.7**.

```text
app / browser --HTTP--> llm.py on 127.0.0.1:18080 --SSH tunnel (colab ssh --proxy-mode)--> G4 VM: gate.py :18080 --> basalt serve :18081
```

- `llm.py` (stdlib only) is the whole client: setup, Colab OAuth, G4 allocation (with exponential backoff / `--wait-gpu`), SSH tunnel, CORS + multi-protocol proxy (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, Web UI `/`), single-model fallback, `MSG_PEEK` client-disconnect cancel, live Basalt telemetry, hot-reload (`POST /_llm/reload`), and 6-layer cost protection.
- `vm/boot.sh` splits Unsloth's 249.6 GiB `UD-IQ1_M.basalt` pack (`vm/basalt_jobs.py`), fetches the 139.1 GiB backbone + builds patched Basalt v0.5.7 (`vm/build.sh`, `vm/patch_basalt.py`) in parallel, applies mrfakename's 43 KB rank-2 stochastic-rounding abliteration in-place (`vm/abliterate_basalt.py` + `vm/abliterate.c`, 1.5s across 62/62 tensors), loads weights with `BASALT_PUNCH_FILE` punch-on-read reclaiming ~140 GiB tmpfs down to 1.6 GiB, streams the 110.6 GiB Engram table into `/mnt/ram/engram.bin`, runs warmup, opens `vm/gate.py` on `:18080`, and arms a VM-side hard-cap backstop.

## Commands

| Command | Effect | Costs CU? |
| --- | --- | --- |
| `python3 llm.py setup` | installs Colab CLI (if needed), Google OAuth login, generates keys & `state/opencode.json` | no |
| `python3 llm.py up [--wait-gpu]` | starts G4 (~3 min), prints Web UI / Base URL / key / model, enters live view on TTY | **yes (~8.9 CU/hr)** |
| `python3 llm.py status` | Web UI URL, endpoint state, model/effort, live Basalt telemetry, Colab balance | no |
| `python3 llm.py watch` | live view: GPU state, live `tok/s`, VRAM hit %, idle countdown, cost | no |
| `python3 llm.py settings` | change defaults (`effort`, `idle_min`, `max_hours` hot-reload live via `/_llm/reload`) | no |
| `python3 llm.py ask [opts] TEXT` | stream answer (`--max`, `--effort VAL`, `--image PATH`, or piped stdin) | while GPU runs |
| `python3 llm.py claude [...]` | start Claude Code pointed at Anthropic `/v1/messages` | while GPU runs |
| `python3 llm.py codex [...]` | start Codex CLI pointed at OpenAI `/v1/responses` | while GPU runs |
| `python3 llm.py opencode [...]` | start opencode with `state/opencode.json` | while GPU runs |
| `python3 llm.py env [--powershell]` | print `export` / PowerShell `$env:` commands for OpenAI & Anthropic clients | no |
| `python3 llm.py logs` | local log + VM boot timeline + Basalt server & engine logs | no |
| `python3 llm.py down` | stops GPU, wipes `/content/secrets`, stops tunnel, watchdog, and detached proxy | no |
| `python3 llm.py serve` | foreground endpoint (GPU boots on first `/v1/*` POST; `Ctrl+C` stops GPU) | on request |

## Rules for agents

1. **Ask before `up` or `serve`.** A running G4 bills ~8.9 compute units/hour.
2. **Never leave a G4 running unintentionally** (and never stop a user's running G4 without explicit instruction if they are actively using it). Confirm with `status` when stopping.
3. **Never leak `state/api_key`, OAuth tokens, or `HF_TOKEN`.** `state/` is git-ignored and logs redact secrets automatically.
4. **Never invoke `colab ssh --proxy-mode` for a session `colab sessions` does not list** — it silently creates a new paid VM.
5. **Single-model fallback is on by default**: unknown model names (e.g. `claude-haiku-*`, `gpt-*`) rewrite to `DeepSeek-V4.1-Flash-1017-UD-IQ1_M` unless `LLM_STRICT_MODEL=1`.

## Code map

| File | Role |
| --- | --- |
| `llm.py` | `setup()`, `Manager` (lifecycle, backoff alloc, orphan-boot cancel, `basalt_stats()`), `Handler` (CORS, `rewrite_request`, multi-protocol SSE errors, `MSG_PEEK` cancel, `/_llm/reload`), `watchdog()`, CLI (`up`/`down`/`watch`/`status`/`settings`/`ask`/`claude`/`codex`/`opencode`/`env`) |
| `vm/boot.sh` | G4 cold-start orchestration + VM-side hard-cap backstop |
| `vm/build.sh` | clones/verifies/patches Basalt v0.5.7 (or downloads `BASALT_PREBUILT_URL`), compiles `sm_120` + standalone `abliterate` |
| `vm/serve.sh` | generates Basalt v2 config (elastic FP8 KV, Dynamic Vision, 32 GiB disk Conversation Cache, `answer-room: 0`, `idle-unload: 0`) |
| `vm/patch_basalt.py` | 4 minimal Colab patches (split Engram, punch-on-read, RAM guard, `BASALT_COLAB_PUNCH_GUARD` unload block) |
| `vm/basalt_jobs.py` | `BASALTMF v2` header parser, backbone/Engram job splitter, `punch_sections()` |
| `vm/abliterate_basalt.py` / `vm/abliterate.c` | in-place 43 KB rank-2 stochastic-rounding abliteration across 62/62 tensors (1.5s) |
| `vm/pfetch.py` / `vm/gate.py` | 64-worker parallel range downloader + post-warmup port 18080 gate |
| `tests/` | offline test suite: `python3 -B -m unittest -v tests/test_llm.py` |
