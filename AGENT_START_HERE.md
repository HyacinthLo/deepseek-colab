# AGENT_START_HERE (`basalt-colab`)

`basalt-colab` is a zero-dependency Python 3.11+ package that runs uncensored **DeepSeek-V4.1-Flash (`UD-IQ1_M`)** on a single Google Colab G4 (RTX PRO 6000 Blackwell 96 GB) via **Basalt v0.5.7**, behind a local OpenAI + Anthropic + Unified Control Console gateway.

```text
clients / browser --HTTP--> basalt-colab :18080 (/_console, /_control/*, /v1/*, /)
                              └── SSH tunnel --> G4 VM: gate.py :18080 --> basalt serve :18081
```

## Key Commands

| Command | Effect | Costs CU? |
| --- | --- | --- |
| `basalt-colab doctor` | verifies Python 3.11+, OpenSSH, Colab CLI, OAuth balance, `config.toml`, VM payload & port | no |
| `basalt-colab setup` | installs Colab CLI, runs Google OAuth, generates keys & `config.toml`, migrates legacy `./state/` | no |
| `basalt-colab console [--no-open]` | starts local proxy (no GPU) and opens `/_console#key=...` | no |
| `basalt-colab up [--wait-gpu] [--basalt-ref REF]` | allocates G4, boots Basalt (~2–3 min), prints endpoints | **yes (~8.9 CU/hr)** |
| `basalt-colab status` | endpoints, Basalt live telemetry, CU billing ledger, Colab balance | no |
| `basalt-colab config [show\|path\|wizard]` | inspect or edit `config.toml` (hot-reloads live fields via `/_control/reload`) | no |
| `basalt-colab ask [--max] [--effort V] [--image P] TEXT` | streams answer with multimodal image + piped stdin support | while GPU runs |
| `basalt-colab claude / codex / opencode / env` | launches or exports env for coding agents | while GPU runs |
| `basalt-colab down [--all]` | stops G4, wipes `/content/secrets`, writes `usage.jsonl` entry, stops proxy | no |

## Agent Rules

1. **Always ask before starting a G4** (`up` / `serve` / `/_control/up`) — G4 bills ~8.9 CU/hr.
2. **Never stop a user's running G4 without explicit instruction**, and always confirm `not running` after stopping when asked.
3. **Never leak `api_key`, OAuth tokens, or `HF_TOKEN`.**
4. **Never invoke `colab ssh --proxy-mode` for a session `colab sessions` does not list** — it silently creates a paid VM.

## Package Map

- [`basalt_colab/paths.py`](basalt_colab/paths.py): XDG config/state resolution (`BASALT_COLAB_HOME`) + legacy `state/` migration.
- [`basalt_colab/manifest.py`](basalt_colab/manifest.py) + [`basalt_colab/vm/manifest.json`](basalt_colab/vm/manifest.json): pinned Basalt tag/commit, prebuilt release archive URL/SHA-256, pack URL, abliteration edit URL/SHA-256.
- [`basalt_colab/config.py`](basalt_colab/config.py): typed `config.toml` (`tomllib` + TOML writer), validation, hot/cold diffing.
- [`basalt_colab/usage.py`](basalt_colab/usage.py): append-only `usage.jsonl` CU billing ledger & daily/monthly aggregation.
- [`basalt_colab/colab.py`](basalt_colab/colab.py): Colab CLI wrapper, `vmx()`, SSH builder, lockfile, tarball bundler, redacting logger.
- [`basalt_colab/lifecycle.py`](basalt_colab/lifecycle.py): `Manager` (backoff alloc, orphan-boot cancel, tunnel repair, cached Basalt telemetry, watchdog).
- [`basalt_colab/proxy.py`](basalt_colab/proxy.py) + [`basalt_colab/console.py`](basalt_colab/console.py): HTTP proxy, CORS, `rewrite_request`, 0.5s `MSG_PEEK` cancel, multi-protocol SSE errors, `/_control/*`, and embedded `/_console`.
- [`basalt_colab/cli.py`](basalt_colab/cli.py): full CLI entrypoint (`basalt-colab` / `python3 -m basalt_colab`).
- [`basalt_colab/vm/`](basalt_colab/vm/): packaged VM scripts (`boot.sh`, `build.sh`, `serve.sh`, `patch_basalt.py`, `basalt_jobs.py`, `abliterate_basalt.py`, `abliterate.c`, `pfetch.py`, `gate.py`).
