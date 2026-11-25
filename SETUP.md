# Setup & Reference (`basalt-colab`)

## Installation

Requires **Python 3.11+**, OpenSSH (`ssh`, `ssh-keygen`), and a Google AI Pro or Colab Pro subscription.

```bash
# Recommended: install as an isolated CLI tool with uv
uv tool install git+https://github.com/HyacinthLo/basalt-colab.git

# Or from a local clone
git clone https://github.com/HyacinthLo/basalt-colab.git && cd basalt-colab
pip install --no-deps --no-build-isolation -e .
```

Run `basalt-colab doctor` at any time to verify Python, OpenSSH, Colab CLI, OAuth balance, `config.toml`, VM payload integrity, and port availability.

## Paths & Legacy Migration

- **Config**: `~/.config/basalt-colab/config.toml` (`%APPDATA%\basalt-colab\config.toml` on Windows)
- **State**: `~/.local/state/basalt-colab/` (`api_key`, `id_ed25519`, `session.json`, `serve.lock`, `usage.jsonl`, `opencode.json`, `basalt-colab.log`)
- **Override**: Set `BASALT_COLAB_HOME=/path/to/dir` to keep config and state together in one custom folder.
- **Legacy migration**: On first `basalt-colab setup` / `up`, if `./state/api_key` or `./state/id_ed25519` exists from an older `deepseek-colab` checkout, it is safely copied (`0600`) into the new state directory.

## Control Console & Clients

- **Unified Control Console**: `basalt-colab console` starts the local proxy (without waking G4) and opens `http://127.0.0.1:18080/_console#key=...`. From there you can start/stop G4, inspect live Basalt telemetry, view your 14-day CU cost curve (`usage.jsonl`), edit & hot-reload `config.toml`, and tail local/VM logs.
- **Basalt Web UI**: `http://127.0.0.1:18080/` proxies natively to Basalt's web console once G4 is ready.
- **Claude Code**: `basalt-colab claude` pre-wires `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`. Background `claude-haiku-*` / `claude-sonnet-*` subagent calls automatically route to `DeepSeek-V4.1-Flash-1017-UD-IQ1_M` unless `proxy.strict_model = true`.
- **Codex CLI**: `basalt-colab codex` pre-wires `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`. `/v1/models` returns both `data` and `models` keys for newer Codex builds.
- **opencode**: `basalt-colab opencode [dir]` regenerates `opencode.json` and launches opencode.
- **Multimodal & Piped CLI**:
  ```bash
  basalt-colab ask --image screenshot.png "Explain this stack trace"
  git diff | basalt-colab ask --max "Review this diff for subtle bugs"
  ```

## Prebuilt Basalt Binary & `--basalt-ref`

- By default (`basalt.ref = ""`), `vm/build.sh` checks the pinned prebuilt release archive from `basalt_colab/vm/manifest.json` (`prebuilt_url` + `prebuilt_sha256`), skipping the ~85s C++ compile and bringing VM boot down to **~95s**. If unavailable or hash-mismatched, it transparently falls back to building from source on G4.
- To test an upstream Basalt branch, tag, or commit:
  ```bash
  basalt-colab up --basalt-ref main
  ```
  This skips the prebuilt cache, clones that ref, applies `patch_basalt.py` (with exact file/anchor mismatch diagnostics if upstream moved code), and compiles on G4.

## Development & Offline Tests

```bash
python3 -B -m unittest discover -s tests -v
```
