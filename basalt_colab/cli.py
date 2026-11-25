"""CLI entrypoint for basalt-colab."""
from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import re
import secrets
import select
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from typing import Any, Callable

from basalt_colab import __version__
from basalt_colab.colab import (
    Logger,
    alive,
    child_env,
    colab_cmd,
    colab_present,
    colab_usage,
    find_tool,
    http_call,
    kill_tree,
    load_session,
    missing_vm_files,
    read_api_key,
    read_bytes,
    redact,
    release_lock,
    run_cmd,
    serve_running,
    session_alive,
    spawn_detached,
    take_lock,
    vmx,
)
from basalt_colab.config import (
    Config,
    ConfigError,
    context_value,
    dump_toml,
    effort_value,
    load_config,
    quant_value,
    save_config,
    short_context,
    validate_config,
)
from basalt_colab.lifecycle import UPSTREAM_TIMEOUT, mmss, run_watchdog, stop_cloud
from basalt_colab.manifest import Manifest, load_manifest
from basalt_colab.paths import IS_WIN, Paths, get_paths, migrate_legacy_state
from basalt_colab.proxy import serve, write_opencode_config
from basalt_colab.usage import UsageLedger

USAGE = f"""basalt-colab {__version__} — DeepSeek-V4.1-Flash on Google Colab G4 via Basalt v0.5.7

usage: basalt-colab COMMAND [options]
  up [opts]                            start G4 (~2-3 min), print endpoints & live status
     --quant Q                         ud-iq1_m (default, 100% RAM) | ud-iq2_xxs (-13% KLD) | ud-q2_k_xl | custom
     --pack-url URL                    override .basalt URL (required for ud-q2_k_xl / custom)
     --uncensor-scale S                refusal-removal strength multiplier (0.0=off, 1.0=default, 1.15-1.25=stronger)
     --wait-gpu                        retry G4 allocation up to 30 min with backoff
     --basalt-ref REF                  build an upstream Basalt tag/branch/SHA from source
  down [--all]                         stop G4, wipe VM secrets, record CU ledger, stop detached proxy
  status                               Control Console & Web UI URLs, Basalt telemetry, billing ledger, Colab balance
  watch                                live terminal status bar (tok/s, VRAM hit %, idle countdown, CU cost)
  console [--no-open]                  ensure local proxy is running and open the Unified Control Console (/_console)
  config [show|path|wizard]            inspect or edit config.toml interactively (hot-reloads live!)
  ask [opts] TEXT                      stream an answer (--max, --effort VAL, --image PATH, or piped stdin)
  claude ...                           launch Claude Code pointed at /v1/messages
  codex ...                            launch Codex CLI pointed at /v1/responses
  opencode ...                         launch opencode with auto-generated opencode.json
  env [--powershell|--sh]              print export / PowerShell $env: lines for OpenAI & Anthropic clients
  logs                                 local proxy log plus VM boot timeline, Basalt server & engine logs
  doctor                               diagnose Python, OpenSSH, Colab CLI, OAuth balance, config.toml, VM payload & port
  setup                                install Colab CLI, complete Google OAuth, create keys & migrate legacy state/
  serve [--detached]                   run the local proxy in the foreground (Ctrl+C stops the GPU too)

env: BASALT_COLAB_HOME, BASALT_COLAB_PORT, BASALT_COLAB_BIND, BASALT_COLAB_QUANT, BASALT_COLAB_PACK_URL,
     BASALT_COLAB_UNCENSOR, BASALT_COLAB_UNCENSOR_SCALE, BASALT_COLAB_IDLE_MIN, BASALT_COLAB_MAX_HOURS,
     BASALT_COLAB_WAIT_GPU, BASALT_COLAB_REF, BASALT_COLAB_MAX_CONTEXT (32K..1M), BASALT_COLAB_PARALLEL (1-4),
     BASALT_COLAB_KV (fp8|fp16), BASALT_COLAB_EFFORT, BASALT_COLAB_STRICT_MODEL, HF_TOKEN, BASALT_COLAB_YES=1"""

TTY, COLOR = [False], [False]


def init_console() -> None:
    TTY[0] = bool(sys.stdout and sys.stdout.isatty())
    if not TTY[0] or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return
    if IS_WIN:
        try:
            import ctypes

            k32, mode = ctypes.windll.kernel32, ctypes.c_uint32()
            h = k32.GetStdHandle(-11)
            if not (k32.GetConsoleMode(h, ctypes.byref(mode)) and k32.SetConsoleMode(h, mode.value | 4)):
                return
        except (AttributeError, OSError):
            return
    COLOR[0] = True


def paint(text: str, *styles: str) -> str:
    codes = {"bold": "1", "dim": "2", "red": "31", "green": "32", "cyan": "36", "yellow": "33"}
    return f"\x1b[{';'.join(codes[s] for s in styles)}m{text}\x1b[0m" if COLOR[0] else text


def glyph(fancy: str, plain: str) -> str:
    utf = "utf" in (getattr(sys.stdout, "encoding", None) or "").lower()
    return fancy if utf and (not IS_WIN or os.environ.get("WT_SESSION")) else plain


def units(cu: float) -> str:
    return f"about {cu:.2f} compute units" if cu < 1 else f"about {cu:.1f} compute units"


def minutes(n: float) -> str:
    return f"{n:g} minute{'' if n == 1 else 's'}"


def cli_name() -> str:
    return "basalt-colab" if shutil.which("basalt-colab") else "python3 -m basalt_colab"


class Live:
    def __init__(self) -> None:
        self.width, self.n = 0, 0

    def clear(self) -> None:
        if self.width:
            sys.stdout.write("\r" + " " * self.width + "\r")
            self.width = 0

    def put(self, parts: list[tuple[str, tuple[str, ...]]]) -> None:
        if not TTY[0]:
            return
        cols = shutil.get_terminal_size((80, 24)).columns - 1
        plain = "".join(t for t, _ in parts)
        self.clear()
        sys.stdout.write(plain[:cols] if len(plain) > cols else "".join(paint(t, *st) for t, st in parts))
        sys.stdout.flush()
        self.width = min(len(plain), cols)

    def show(self, when: str, text: str) -> None:
        spin = glyph("⣾⣽⣻⢿⡿⣟⣯⣷", "|/-\\")
        self.n += 1
        self.put([("  ", ()), (f"{when:>5}", ("dim",)), ("  ", ()), (spin[self.n % len(spin)], ("cyan",)), ("  " + text, ())])

    def line(self, text: str) -> None:
        self.clear()
        print(text, flush=True)


def control_call(paths: Paths, cfg: Config, port: int, path: str, body: dict[str, Any] | None, timeout: float) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    data = None if body is None else json.dumps(body).encode()
    return http_call(
        port,
        "GET" if body is None else "POST",
        path,
        data,
        {"Authorization": "Bearer " + read_api_key(paths), "Content-Type": "application/json"},
        timeout,
        cfg.proxy.host,
    )


def follow(resp: http.client.HTTPResponse, header: str | None = None) -> tuple[str, str]:
    live = Live()
    for raw in resp:
        kind, _, rest = raw.decode("utf-8", "replace").strip().partition(" ")
        if kind in ("step", "wait"):
            if header:
                print(header + "\n", flush=True)
                header = None
            when, _, text = rest.partition(" ")
            if kind == "step":
                live.line(f"  {paint(f'{mmss(float(when)):>5}', 'dim')}  {paint(glyph('✓', '+'), 'green')}  {text}")
            else:
                live.show(mmss(float(when)), text)
        elif kind:
            live.clear()
            return kind, rest
    live.clear()
    return "", ""


def confirm(question: str) -> bool:
    if os.environ.get("BASALT_COLAB_YES") == "1" or os.environ.get("LLM_YES") == "1":
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    try:
        return input(question + " [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def install_colab(manifest: Manifest) -> bool:
    home_bin = os.path.join(os.path.expanduser("~"), ".local", "bin")
    uv = shutil.which("uv") or next((p for p in (os.path.join(home_bin, "uv.exe" if IS_WIN else "uv"),) if os.path.exists(p)), None)
    pipx = shutil.which("pipx")
    tool = "uv" if uv or not pipx else "pipx"
    print("Google's Colab CLI is not installed yet. basalt-colab uses it to start and stop the G4 GPU.", flush=True)
    if not confirm(f"Install {manifest.colab_cli} with {tool}{' (installs uv first)' if tool == 'uv' and not uv else ''}?"):
        return False
    print("Installing...", end=" ", flush=True)
    if tool == "uv" and not uv:
        rc, out = run_cmd(
            ["powershell", "-ExecutionPolicy", "ByPass", "-c", "irm https://astral.sh/uv/install.ps1 | iex"]
            if IS_WIN
            else ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            900,
        )
        if rc != 0:
            print("failed:\n" + out[-2000:], flush=True)
            return False
        uv = shutil.which("uv") or os.path.join(home_bin, "uv.exe" if IS_WIN else "uv")
    assert uv or pipx
    rc, out = run_cmd([uv, "tool", "install", manifest.colab_cli] if tool == "uv" and uv else [pipx or "pipx", "install", manifest.colab_cli], 900)
    print("done." if rc == 0 else "failed:\n" + out[-2000:], flush=True)
    return rc == 0


def colab_login() -> None:
    hide = (b"Current balance", b"Usage rate", b"Active assignments")
    p = subprocess.Popen(colab_cmd() + ["usage"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env({"PYTHONUNBUFFERED": "1"}))
    assert p.stdout is not None
    buf = b""
    while True:
        chunk = os.read(p.stdout.fileno(), 4096)
        if not chunk:
            break
        *lines, buf = (buf + chunk).split(b"\n")
        keep = [l + b"\n" for l in lines if not l.lstrip().startswith(hide)]
        if buf and not any(h.startswith(buf.lstrip()) for h in hide):
            keep, buf = keep + [buf], b""
        sys.stdout.flush()
        sys.stdout.buffer.write(b"".join(keep))
        sys.stdout.buffer.flush()
    p.wait()


def ensure_setup(paths: Paths, cfg: Config, manifest: Manifest) -> float:
    paths.ensure_dirs()
    migrated = migrate_legacy_state(paths)
    if migrated:
        print(paint(f"  Migrated legacy credentials into {paths.state_dir}: {', '.join(migrated)}", "dim"), flush=True)
    missing = [t for t in ("ssh", "ssh-keygen") if not (shutil.which(t) or os.path.exists(find_tool(t)))]
    if missing:
        sys.exit(f"Missing {' and '.join(missing)}. Install an OpenSSH client and run again.")
    if not colab_present():
        if not install_colab(manifest):
            sys.exit(f'The Colab CLI is needed. Install it with "uv tool install {manifest.colab_cli}" (or pipx), then run again.')
        if not colab_present():
            sys.exit('Installed, but "colab" is not on PATH yet. Open a new terminal and run again.')
    bal, _ = colab_usage()
    if bal is None:
        if not (sys.stdin and sys.stdin.isatty()):
            sys.exit(f"The Colab CLI is installed but not logged in yet. Run in a terminal: {cli_name()} setup")
        print(f"\n{paint('Google login (only once)', 'bold')}: open the link below, sign in and paste the code here.", flush=True)
        colab_login()
        bal, _ = colab_usage()
        if bal is None:
            sys.exit(f"The login did not work. Try again: {cli_name()} setup")
        print(f"{paint('Logged in.', 'bold', 'green')} Colab balance: {bal:.1f} compute units.\n", flush=True)
    if not paths.key_file.exists():
        fd = os.open(str(paths.key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(("sk-ds41-" + secrets.token_hex(24) + "\n").encode())
    if not paths.ssh_key.exists():
        rc, out = run_cmd([find_tool("ssh-keygen"), "-q", "-t", "ed25519", "-N", "", "-C", "basalt-colab", "-f", str(paths.ssh_key)], 30)
        if rc:
            sys.exit("ssh-keygen failed: " + out)
    if not paths.config_file.exists():
        save_config(cfg, paths)
    write_opencode_config(paths, cfg, cfg.proxy.port)
    return bal


def ensure_detached_proxy(paths: Paths) -> dict[str, Any]:
    held = serve_running(paths)
    if held:
        return held
    paths.ensure_dirs()
    extra = {"BASALT_COLAB_HOME": str(paths.config_dir)} if os.environ.get("BASALT_COLAB_HOME") else None
    with paths.serve_out_file.open("wb") as out:
        p = spawn_detached([sys.executable, "-m", "basalt_colab", "serve", "--detached"], out, extra_env=extra)
    t0 = time.time()
    while not held:
        if p.poll() is not None or time.time() - t0 > 30:
            raw = (read_bytes(paths.serve_out_file) or b"").decode("utf-8", "replace")
            sys.exit(f"The local endpoint did not start: {raw}")
        time.sleep(0.3)
        held = serve_running(paths)
    return held


def format_telemetry(b: dict[str, Any] | None) -> str:
    if not isinstance(b, dict):
        return ""
    parts: list[str] = []
    if b.get("busy") and b.get("live_tok_s") is not None:
        parts.append(f"{b['live_tok_s']:.1f} tok/s")
    elif b.get("last_tok_s") is not None:
        parts.append(f"{b['last_tok_s']:.1f} tok/s")
    if b.get("hit_rate") is not None:
        parts.append(f"hit {b['hit_rate'] * 100:.0f}%")
    if b.get("pcie_share") is not None:
        parts.append(f"pcie {b['pcie_share'] * 100:.1f}%")
    return " · ".join(parts)


def endpoint_info(paths: Paths, cfg: Config, manifest: Manifest, port: int, bal: float | None = None) -> None:
    me_cmd = cli_name()
    host = cfg.proxy.host
    rows = (
        ("Console", f"http://{host}:{port}/_console"),
        ("Web UI", f"http://{host}:{port}/"),
        ("Base URL", f"http://{host}:{port}/v1"),
        ("API key", read_api_key(paths)),
        ("Model", f"{cfg.basalt.effective_model_alias} (effort {cfg.basalt.effort}, {cfg.describe_uncensor()})"),
        ("Engine", f"{cfg.describe_engine()} · ~{(manifest.pack(cfg.basalt.quant).estimated_speed_tok_s if manifest.pack(cfg.basalt.quant) else manifest.estimated_speed_tok_s):.0f} tok/s"),
    )
    print()
    for name, value in rows:
        print(f"  {name:<10} {paint(value, 'bold')}")
    print()
    for name, value in (
        ("Try it", f'{me_cmd} ask "Hello!"'),
        ("Console", f"{me_cmd} console"),
        ("Claude", f"{me_cmd} claude"),
        ("Codex", f"{me_cmd} codex"),
        ("opencode", f"{me_cmd} opencode [project folder]"),
        ("Shell env", f'eval "$({me_cmd} env)"'),
        ("Stop", f"{me_cmd} down"),
    ):
        print(f"  {name:<10} {paint(value, 'cyan')}")
    bal_str = f" · balance {bal:.1f}" if bal is not None else ""
    print("\n" + paint(f"  About {manifest.g4_cu_per_hour:.0f} compute units/hr{bal_str} · stops after {minutes(cfg.colab.idle_min)} idle", "dim"))


def stopped_note(kind: str, rest: str) -> None:
    if kind != "STOPPED":
        sys.exit(f"The GPU may still be running. Run this again: {cli_name()} down")
    cu, ran = (rest.split() + ["None", "None"])[:2]
    done = paint(glyph("✓", "+") + " Stopped", "bold", "green")
    if cu != "None" and ran != "None":
        print(f"{done} · the GPU ran {mmss(float(ran))} · {units(float(cu))}")
    else:
        print(f"{done} · nothing is running or billed now")


def watch_loop(paths: Paths, cfg: Config, port: int) -> None:
    if not TTY[0]:
        return
    print(paint("\n  Ctrl+C closes this view; the endpoint keeps running.\n", "dim"))
    live, prev, dots = Live(), None, (glyph("●", "*"), glyph("◌", "~"), glyph("○", "o"))
    try:
        while True:
            try:
                conn, r = control_call(paths, cfg, port, "/_control/status", None, 5)
                st = json.loads(r.read())
                conn.close()
            except (OSError, ValueError, http.client.HTTPException):
                live.line(f"  {paint(time.strftime('%H:%M'), 'dim')}  the endpoint has stopped")
                return
            state = "ready" if st["state"].startswith("ready") else st["state"]
            if prev and state != prev:
                live.line(f"  {paint(time.strftime('%H:%M'), 'dim')}  {watch_event(prev, state, st)}")
            prev = state
            spent = f"ran {mmss(st['ran'])} · {units(st['cu'])}" if st.get("ran") is not None else ""
            if state == "ready":
                left = min(st["idle_stop_min"] * 60 - st["idle_s"], st["max_hours"] * 3600 - (st["ran"] or 0))
                doing = (
                    f"answering {st['active']} request{'' if st['active'] == 1 else 's'}"
                    if st["active"]
                    else f"idle {mmss(st['idle_s'])} · stops in {mmss(left)}"
                )
                perf = format_telemetry(st.get("basalt"))
                tail = " · ".join(p for p in (perf, spent) if p)
                live.put([("  ", ()), (dots[0] + " GPU running", ("green", "bold")), ("   " + doing + "   ", ()), (tail, ("dim",))])
            elif state in ("starting", "stopping"):
                what = "starting · " + st["stage"] if state == "starting" else "stopping"
                live.put([("  ", ()), (dots[1] + " GPU " + what, ("cyan",)), ("   " + spent, ("dim",))])
            else:
                live.put([("  ", ()), (dots[2] + " GPU off", ("bold",)), ("   starts on the next request (~2-3 min)", ("dim",))])
            time.sleep(1)
    except KeyboardInterrupt:
        live.clear()
        me_cmd = cli_name()
        print(
            f"\n  Closed live view; endpoint keeps running.\n"
            f"  {paint('Live view', 'dim')} {paint(me_cmd + ' watch', 'cyan')}   "
            f"{paint('Console', 'dim')} {paint(me_cmd + ' console', 'cyan')}   "
            f"{paint('Stop', 'dim')} {paint(me_cmd + ' down', 'cyan')}"
        )


def watch_event(prev: str, state: str, st: dict[str, Any]) -> str:
    stop = st.get("last_stop") or {}
    if state in ("stopping", "off") and prev in ("ready", "starting"):
        if st.get("last_error") and prev == "starting":
            return paint("start failed: " + st["last_error"], "red")
        cost = f": ran {mmss(stop['ran'])}, {units(stop['cu'])}" if stop.get("ran") is not None else ""
        why = (
            f"after {minutes(st['idle_stop_min'])} without requests"
            if stop.get("reason", "").startswith("idle")
            else stop.get("reason", "")
        )
        return f"GPU stopped {why}{cost}"
    if state == "starting":
        return "GPU starting for a new request"
    if state == "ready":
        return paint("GPU ready", "green")
    return "GPU " + state


def pick(label: str, hint: str, default: Any, parse: Callable[[str], Any]) -> Any:
    d_str = f"[{default}]"
    row = (
        f"  {paint(f'{label:<14}', 'bold')}{paint(f'{hint:<30}', 'dim')}"
        f"{paint(f'{d_str:>10}', 'cyan')} {paint(glyph('›', '>'), 'cyan')} "
    )
    while True:
        try:
            text = input(row).strip()
        except EOFError:
            text = ""
        val = parse(text or str(default))
        if val is not None:
            return val
        print(f"  {' ' * 14}{paint('not valid, try again', 'red')}")


def _parse_num(text: str, low: float, high: float) -> int | float | None:
    try:
        v = float(text)
    except ValueError:
        return None
    return (int(v) if v == int(v) else v) if low <= v <= high else None


def _parse_yes_no(text: str) -> bool | None:
    return {"y": True, "yes": True, "true": True, "n": False, "no": False, "false": False}.get(text.lower())


def interactive_config_wizard(paths: Paths, cfg: Config) -> Config | None:
    if not (sys.stdin and sys.stdin.isatty()):
        return None
    print(f"{paint('Configuration', 'bold')}  {paint('Enter keeps the value in brackets', 'dim')}\n")
    new = cfg.clone()
    new.basalt.quant = pick("Quant pack", "ud-iq1_m | ud-iq2_xxs | ud-q2_k_xl", cfg.basalt.quant, quant_value)
    if new.basalt.quant in ("ud-q2_k_xl", "custom"):
        new.basalt.pack_url = pick(
            "Pack URL",
            "https://.../model.basalt",
            cfg.basalt.pack_url or "",
            lambda t: t.strip() if re.match(r"^https?://", t.strip()) and "'" not in t else None,
        )
    else:
        new.basalt.pack_url = ""
    new.basalt.max_context = pick("Context/slot", "32K..1M (slot-full-context)", short_context(cfg.basalt.max_context), context_value)
    new.basalt.parallel = int(pick("Parallel", "1-4 concurrent slots", cfg.basalt.parallel, lambda t: int(t) if t.isdigit() and 1 <= int(t) <= 4 else None))
    new.basalt.kv = pick("KV precision", "fp8 (0.97G@1M) or fp16", cfg.basalt.kv, lambda t: t.lower() if t.lower() in ("fp8", "fp16") else None)
    new.basalt.effort = pick("Effort", "1-100, low, high, max (live)", cfg.basalt.effort, effort_value)
    new.colab.idle_min = float(pick("Idle stop", "minutes 1-120 (live)", f"{cfg.colab.idle_min:g}", lambda t: _parse_num(t, 1, 120)))
    new.colab.max_hours = float(pick("Hard stop", "hours 1-24 (live)", f"{cfg.colab.max_hours:g}", lambda t: _parse_num(t, 1, 24)))
    new.basalt.uncensor = bool(pick("Uncensored", "43KB mrfakename rank-2, y/n", "y" if cfg.basalt.uncensor else "n", _parse_yes_no))
    if new.basalt.uncensor:
        new.basalt.uncensor_scale = float(pick("Uncensor scale", "0.5-2.0 (1.0=std, 1.2=stronger)", f"{cfg.basalt.uncensor_scale:g}", lambda t: _parse_num(t, 0.0, 2.0)))
    new.basalt.ref = pick("Basalt ref", "blank=pinned, or tag/branch", cfg.basalt.ref or "-", lambda t: "" if t in ("-", "pinned", "") else t)
    save_config(new, paths)
    print(f"\n  {paint(glyph('✓', '+'), 'green')} Saved {paint(str(paths.config_file), 'dim')}\n")
    return new


def cmd_setup(paths: Paths, cfg: Config, manifest: Manifest) -> None:
    bal = ensure_setup(paths, cfg, manifest)
    print(f"Setup done. Config: {paths.config_file}")
    print(f"Colab balance: {bal:.1f} compute units (~{bal / manifest.g4_cu_per_hour:.0f} GPU hours).")
    print(f"Next: {cli_name()} up   (or {cli_name()} console)")


def cmd_up(
    paths: Paths,
    cfg: Config,
    manifest: Manifest,
    wait_gpu: bool,
    basalt_ref: str | None,
    quant: str | None = None,
    pack_url: str | None = None,
    uncensor_scale: float | None = None,
) -> None:
    want_cfg = cfg.clone()
    if basalt_ref is not None:
        want_cfg.basalt.ref = basalt_ref
    if quant is not None:
        q = quant_value(quant)
        if not q:
            sys.exit(f"invalid --quant {quant!r} (expected ud-iq1_m, ud-iq2_xxs, ud-q2_k_xl, or custom)")
        want_cfg.basalt.quant = q
    if pack_url is not None:
        want_cfg.basalt.pack_url = pack_url
    if uncensor_scale is not None:
        want_cfg.basalt.uncensor_scale = uncensor_scale
        want_cfg.basalt.uncensor = uncensor_scale > 0.0
    try:
        validate_config(want_cfg)
        want_cfg.boot_vars(manifest=manifest)
    except ConfigError as e:
        sys.exit(f"config error: {e}")

    print(paint(f"basalt-colab {__version__} · {want_cfg.basalt.effective_model_alias} (Basalt {want_cfg.basalt.ref or manifest.basalt_tag})", "bold") + "\n")
    bal = ensure_setup(paths, want_cfg, manifest)
    held = serve_running(paths)
    if held and held.get("cold_signature") and held["cold_signature"] != want_cfg.cold_signature():
        have = held.get("engine_summary") or "different engine settings"
        sys.exit(f"The endpoint runs with {have}. To change engine settings, stop it first: {cli_name()} down")
    if bal < want_cfg.colab.min_balance and not (held and load_session(paths)):
        sys.exit(f"Colab balance {bal:.2f} compute units is below min_balance {want_cfg.colab.min_balance:.0f}")
    if not held:
        held = ensure_detached_proxy(paths)
    up_opts: dict[str, Any] = {}
    if wait_gpu:
        up_opts["wait_gpu"] = True
    if basalt_ref is not None:
        up_opts["basalt_ref"] = basalt_ref
    if quant is not None:
        up_opts["quant"] = want_cfg.basalt.quant
    if pack_url is not None:
        up_opts["pack_url"] = want_cfg.basalt.pack_url
    if uncensor_scale is not None:
        up_opts["uncensor_scale"] = want_cfg.basalt.uncensor_scale
    conn, r = control_call(paths, want_cfg, held["port"], "/_control/up", up_opts, 120)
    kind, rest = follow(r, paint("Starting a Colab G4 GPU", "bold") + paint(" · ~2-3 minutes (faster with prebuilt binary)", "dim"))
    conn.close()
    if kind != "READY":
        print(paint("Start failed: " + (rest or "the local endpoint closed the connection"), "red"))
        sys.exit(f"Details: {cli_name()} logs")
    secs = int(rest or 0)
    ready_line = f"{glyph('●', '*')} The GPU is already running" if secs < 2 else f"\n{glyph('✓', '+')} Ready after {mmss(secs)}"
    print(paint(ready_line, "bold", "green"))
    endpoint_info(paths, want_cfg, manifest, held["port"], bal)
    watch_loop(paths, want_cfg, held["port"])


def cmd_down(paths: Paths, cfg: Config, manifest: Manifest, everything: bool = False) -> None:
    held = serve_running(paths)
    if held:
        leave = everything or bool(held.get("detached"))
        try:
            conn, r = control_call(paths, cfg, held["port"], "/_control/down", {"exit": leave}, 120)
            kind, rest = follow(r)
            conn.close()
            t0 = time.time()
            while leave and alive(held) and time.time() - t0 < 30:
                time.sleep(0.3)
            if leave and alive(held):
                kill_tree(int(held["pid"]), hard=True)
            return stopped_note(kind, rest)
        except (OSError, http.client.HTTPException) as e:
            Logger(paths).log(f"endpoint pid {held['pid']} did not answer ({e.__class__.__name__}); killing and stopping directly")
            kill_tree(int(held["pid"]), hard=True)
            time.sleep(1)
    held = take_lock(paths, {"port": 0})
    if held:
        sys.exit(f"Another basalt-colab command holds the lock (pid {held.get('pid')}).")
    try:
        still, cu, ran = stop_cloud(
            paths, cfg, manifest, UsageLedger(paths.usage_file), Logger(paths), load_session(paths), "down"
        )
    finally:
        release_lock(paths)
    stopped_note("STILL" if still else "STOPPED", f"{cu} {ran}")


def cmd_status(paths: Paths, cfg: Config) -> None:
    held = serve_running(paths)
    rows: list[tuple[str, str]] = []
    if held:
        try:
            conn, r = control_call(paths, cfg, held["port"], "/_control/status", None, 10)
            st = json.loads(r.read())
            conn.close()
            gpu = {"off": "off, starts on the next request", "starting": "starting: " + st["stage"]}.get(st["state"], st["state"])
            if st["state"].startswith("ready"):
                gpu = (
                    f"{st['state']}, idle {mmss(st['idle_s'])} of {minutes(st['idle_stop_min'])}, "
                    f"{st['served']} request{'' if st['served'] == 1 else 's'} served, "
                    f"ran {mmss(st['ran'] or 0)}, {units(st['cu'] or 0)}"
                )
            rows += [
                ("Console", f"http://{cfg.proxy.host}:{held['port']}/_console"),
                ("Web UI", f"http://{cfg.proxy.host}:{held['port']}/"),
                ("Endpoint", f"http://{cfg.proxy.host}:{held['port']}/v1 · {st.get('engine_summary') or ''}"),
                ("Model", f"{st.get('model') or cfg.basalt.effective_model_alias} (effort {st.get('effort', cfg.basalt.effort)}, {st.get('uncensor', cfg.describe_uncensor())}, ref {st.get('basalt_ref', 'pinned')})"),
                ("GPU", gpu),
            ]
            perf = format_telemetry(st.get("basalt"))
            if perf:
                rows.append(("Telemetry", perf))
            if st.get("last_error"):
                rows.append(("Last error", st["last_error"]))
        except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
            rows.append(("Endpoint", f"pid {held['pid']} not answering ({e.__class__.__name__})"))
    else:
        rows.append(("Endpoint", f"not running (start it: {cli_name()} up)"))
    ledger = UsageLedger(paths.usage_file).summary(days=30)
    if ledger["sessions_count"]:
        rows.append(("Ledger", f"today {ledger['today_cu']:.2f} CU · month {ledger['month_cu']:.2f} CU · {ledger['sessions_count']} sessions"))
    bal, _ = colab_usage()
    rows.append((
        "Colab",
        f"session {cfg.colab.session} {'running' if session_alive(cfg.colab.session) else 'not running'}, "
        f"balance {f'{bal:.1f}' if bal is not None else 'unknown'} compute units",
    ))
    if held and paths.key_file.exists():
        rows.append(("API key", read_api_key(paths)))
    for name, value in rows:
        print(f"  {name:<10} {value}")


def cmd_console(paths: Paths, cfg: Config, manifest: Manifest, no_open: bool = False) -> None:
    if not paths.key_file.exists():
        ensure_setup(paths, cfg, manifest)
    held = ensure_detached_proxy(paths)
    url = f"http://{cfg.proxy.host}:{held['port']}/_console#key={urllib.parse.quote(read_api_key(paths))}"
    clean_url = f"http://{cfg.proxy.host}:{held['port']}/_console"
    print(f"  {paint('Console', 'bold')}  {clean_url}")
    if not no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass


def cmd_config(paths: Paths, cfg: Config, args: list[str]) -> None:
    sub = args[0] if args else ("wizard" if (sys.stdin and sys.stdin.isatty()) else "show")
    if sub == "path":
        print(paths.config_file)
        return
    if sub == "show":
        print(dump_toml(cfg), end="")
        return
    if sub != "wizard":
        sys.exit(f"unknown config subcommand {sub!r} (use: show, path, wizard)")
    if not (sys.stdin and sys.stdin.isatty()):
        sys.exit(f"Config: {cfg.summary()}\nEdit {paths.config_file} or run '{cli_name()} config' in a terminal.")
    updated = interactive_config_wizard(paths, cfg)
    held = serve_running(paths)
    if updated and held:
        try:
            conn, r = control_call(paths, cfg, held["port"], "/_control/reload", {}, 10)
            res = json.loads(r.read())
            conn.close()
            ap = res.get("applied") or {}
            print(f"  {paint(glyph('✓', '+'), 'green')} Applied live (effort {ap.get('effort')} · idle {ap.get('idle_min')} min · cap {ap.get('max_hours')} h)")
            if res.get("restart_required"):
                print(paint(f"  Cold engine changes ({', '.join(res.get('cold_changed', []))}) apply on next GPU start ({cli_name()} down && {cli_name()} up).", "dim"))
        except (OSError, ValueError, http.client.HTTPException) as e:
            print(paint(f"  Saved to disk, but live reload failed ({e}).", "red"))


def cmd_logs(paths: Paths, cfg: Config) -> None:
    raw = read_bytes(paths.log_file)
    key = read_api_key(paths) if paths.key_file.exists() else ""
    print(redact(raw.decode("utf-8", "replace")[-4000:], key) if raw is not None else "(no local log)")
    if session_alive(cfg.colab.session):
        _, out = vmx(
            cfg.colab.session,
            "cat /content/res/timeline.txt 2>/dev/null; tail -n 25 /content/res/server.log 2>/dev/null; "
            "tail -n 15 /root/.cache/basalt/*/engine.log 2>/dev/null || true",
            40,
        )
        print(redact(out, key))


def cmd_watch(paths: Paths, cfg: Config) -> None:
    held = serve_running(paths)
    if not held:
        sys.exit(f"The endpoint is not running. Start it first: {cli_name()} up")
    if not TTY[0]:
        sys.exit(f'watch needs a terminal; use "{cli_name()} status" for a snapshot')
    watch_loop(paths, cfg, held["port"])


def parse_ask_args(words: list[str]) -> tuple[str | None, list[str], str]:
    effort, images, rest = None, [], []
    i = 0
    while i < len(words):
        w = words[i]
        if w == "--max":
            effort = "max"
        elif w == "--effort" and i + 1 < len(words):
            i += 1
            effort = effort_value(words[i]) or sys.exit(f"invalid --effort {words[i]!r} (use 1-100, low, high, max)")
        elif w == "--image" and i + 1 < len(words):
            i += 1
            images.append(words[i])
        else:
            rest.append(w)
        i += 1
    return effort, images, " ".join(rest).strip()


def encode_image_url(path: str) -> str:
    if not os.path.isfile(path):
        sys.exit(f"image file not found: {path}")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def read_piped_stdin(has_prompt: bool) -> str:
    if not sys.stdin or sys.stdin.isatty():
        return ""
    if has_prompt:
        if IS_WIN:
            return ""
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not rlist:
                return ""
        except (OSError, ValueError):
            return ""
    return sys.stdin.read().strip()


def cmd_ask(paths: Paths, cfg: Config, words: list[str]) -> None:
    held = serve_running(paths)
    if not held:
        sys.exit(f"The endpoint is not running. Start it first: {cli_name()} up")
    effort, images, prompt = parse_ask_args(words)
    piped = read_piped_stdin(bool(prompt or images))
    if piped and prompt:
        prompt = piped + "\n\n" + prompt
    elif piped:
        prompt = piped
    elif not prompt:
        prompt = "Describe this image." if images else "Hello!"
    content: Any = (
        [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": encode_image_url(p)}} for p in images]
        if images
        else prompt
    )
    body: dict[str, Any] = {
        "model": cfg.basalt.effective_model_alias,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": content}],
    }
    if effort:
        body["reasoning_effort"] = effort
    t_req = time.time()
    conn, r = http_call(
        held["port"],
        "POST",
        "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": "Bearer " + read_api_key(paths), "Content-Type": "application/json"},
        UPSTREAM_TIMEOUT,
        cfg.proxy.host,
    )
    if r.status != 200:
        sys.exit(f"HTTP {r.status}: {r.read()[:500].decode('utf-8', 'replace')}")
    live, part, timings, usage_obj, t_first = Live(), None, None, None, None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        wait = re.match(r": starting gpu vm \((\d+) s\)(?: - (.*))?", line)
        if wait:
            live.show(mmss(float(wait.group(1))), wait.group(2) or "starting the GPU (~2-3 min)")
        if not line.startswith("data: {"):
            continue
        event = json.loads(line[6:])
        if event.get("error"):
            live.clear()
            sys.exit(f"\nerror: {event['error'].get('message', event['error'])}")
        timings = event.get("timings") or timings
        if event.get("usage"):
            usage_obj = event["usage"]
        for ch in event.get("choices", []):
            delta = ch.get("delta") or {}
            for kind, text in (("think", delta.get("reasoning_content")), ("say", delta.get("content"))):
                if not text:
                    continue
                if t_first is None:
                    t_first = time.time()
                if kind == "think" and not TTY[0]:
                    continue
                live.clear()
                if part not in (None, kind):
                    sys.stdout.write("\n\n")
                part = kind
                sys.stdout.write(paint(text, "dim") if kind == "think" else text)
                sys.stdout.flush()
    t_end = time.time()
    conn.close()
    print()
    if TTY[0]:
        if timings and timings.get("predicted_n"):
            print(paint(f"{cfg.basalt.effective_model_alias}: {timings['predicted_n']} tokens at {timings.get('predicted_per_second', 0):.1f} tok/s", "dim"))
        elif usage_obj and usage_obj.get("completion_tokens") and t_first is not None:
            n = int(usage_obj["completion_tokens"])
            dec_s = max(0.001, t_end - t_first)
            ttft = max(0.0, t_first - t_req)
            print(paint(f"{cfg.basalt.effective_model_alias}: {n} tokens · TTFT {ttft:.2f}s · {n / dec_s:.1f} tok/s", "dim"))


def client_env_vars(paths: Paths, cfg: Config, port: int) -> tuple[tuple[str, str], ...]:
    base = f"http://{cfg.proxy.host}:{port}"
    key = read_api_key(paths)
    model = cfg.basalt.effective_model_alias
    return (
        ("OPENAI_BASE_URL", base + "/v1"),
        ("OPENAI_API_KEY", key),
        ("OPENAI_MODEL", model),
        ("ANTHROPIC_BASE_URL", base),
        ("ANTHROPIC_API_KEY", key),
        ("ANTHROPIC_AUTH_TOKEN", key),
        ("ANTHROPIC_MODEL", model),
        ("ANTHROPIC_SMALL_FAST_MODEL", model),
    )


def find_opencode() -> str | None:
    for p in (shutil.which("opencode"), os.path.join(os.path.expanduser("~"), ".opencode", "bin", "opencode.exe" if IS_WIN else "opencode")):
        if p and os.path.exists(p):
            return p
    return None


def cmd_opencode(paths: Paths, cfg: Config, args: list[str]) -> None:
    held = serve_running(paths)
    if not held:
        sys.exit(f"The endpoint is not running. Start it first: {cli_name()} up")
    exe = find_opencode()
    if not exe:
        sys.exit("opencode is not installed. Get it from https://opencode.ai, then run this again.")
    write_opencode_config(paths, cfg, held["port"])
    sys.exit(subprocess.call([exe] + args, env=dict(os.environ, OPENCODE_CONFIG=str(paths.opencode_file))))


def cmd_claude(paths: Paths, cfg: Config, args: list[str]) -> None:
    held = serve_running(paths)
    if not held:
        sys.exit(f"The endpoint is not running. Start it first: {cli_name()} up")
    exe = shutil.which("claude")
    if not exe:
        sys.exit("claude is not installed. Install Claude Code, then run this again.")
    env = dict(os.environ, **dict(client_env_vars(paths, cfg, held["port"])), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    sys.exit(subprocess.call([exe] + args, env=env))


def cmd_codex(paths: Paths, cfg: Config, args: list[str]) -> None:
    held = serve_running(paths)
    if not held:
        sys.exit(f"The endpoint is not running. Start it first: {cli_name()} up")
    exe = shutil.which("codex")
    if not exe:
        sys.exit("codex is not installed. Install Codex CLI (npm i -g @openai/codex), then run this again.")
    env = dict(os.environ, **dict(client_env_vars(paths, cfg, held["port"])))
    sys.exit(subprocess.call([exe] + args, env=env))


def cmd_env(paths: Paths, cfg: Config, args: list[str]) -> None:
    if not paths.key_file.exists():
        sys.exit(f"missing {paths.key_file}; run: {cli_name()} setup")
    held = serve_running(paths)
    port = held["port"] if held else cfg.proxy.port
    ps = "--powershell" in args or (IS_WIN and "--sh" not in args)
    for k, v in client_env_vars(paths, cfg, port):
        if ps:
            print(f"$env:{k} = '{v.replace(chr(39), chr(39) * 2)}'")
        else:
            print(f"export {k}={shlex.quote(v)}")


def cmd_doctor(paths: Paths, cfg: Config, manifest: Manifest) -> None:
    ok_all = True

    def check(label: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        if not ok:
            ok_all = False
        mark = paint(glyph("✓", "+"), "green") if ok else paint(glyph("✗", "!"), "red")
        print(f"  {mark}  {label:<18} {detail}")

    print(paint(f"basalt-colab {__version__} doctor", "bold") + "\n")
    check("Python", sys.version_info >= (3, 11), f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    ssh_ok = all(shutil.which(t) or os.path.exists(find_tool(t)) for t in ("ssh", "ssh-keygen"))
    check("OpenSSH", ssh_ok, "ssh + ssh-keygen found" if ssh_ok else "missing ssh or ssh-keygen")
    c_ok = colab_present()
    check("Colab CLI", c_ok, colab_cmd()[0] if c_ok else f"not installed (run {cli_name()} setup)")
    if c_ok:
        bal, _ = colab_usage()
        check("Colab OAuth", bal is not None, f"balance {bal:.1f} CU" if bal is not None else f"not logged in (run {cli_name()} setup)")
    missing_vm = missing_vm_files(paths.vm_dir)
    check("VM payload", not missing_vm, f"{paths.vm_dir} (Basalt {manifest.basalt_tag})" if not missing_vm else f"missing: {', '.join(missing_vm)}")
    check("Config", True, f"{paths.config_file} ({'saved' if paths.config_file.exists() else 'defaults'}) · {cfg.describe_engine()}")
    check("Credentials", paths.key_file.exists() and paths.ssh_key.exists(), str(paths.state_dir))
    held = serve_running(paths)
    if held:
        check("Local port", True, f"proxy already running on {cfg.proxy.host}:{held['port']} (pid {held['pid']})")
    else:
        s = socket.socket()
        try:
            s.bind((cfg.proxy.bind, cfg.proxy.port))
            port_free = True
        except OSError:
            port_free = False
        finally:
            s.close()
        check("Local port", port_free, f"{cfg.proxy.bind}:{cfg.proxy.port} available" if port_free else f"{cfg.proxy.bind}:{cfg.proxy.port} in use")
    if not ok_all:
        sys.exit(1)


def _extract_flag_val(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"basalt-colab needs Python 3.11+ (this is {sys.version_info.major}.{sys.version_info.minor})")
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "help"
    init_console()

    paths = get_paths()
    if cmd == "_watchdog" and len(args) == 6:
        run_watchdog(paths, args[1], float(args[2]), float(args[3]), float(args[4]), float(args[5]))
        return
    if cmd in ("help", "-h", "--help"):
        print(USAGE)
        return
    if cmd in ("-V", "--version"):
        print(f"basalt-colab {__version__}")
        return

    try:
        manifest = load_manifest(paths.vm_dir)
        cfg = load_config(paths)
    except (OSError, ConfigError, KeyError, ValueError) as e:
        sys.exit(f"config/manifest error: {e}")

    try:
        if cmd == "serve":
            serve(paths, cfg, manifest, detached="--detached" in args[1:])
        elif cmd == "setup":
            cmd_setup(paths, cfg, manifest)
        elif cmd == "up":
            raw_scale = _extract_flag_val(args[1:], "--uncensor-scale")
            try:
                parsed_scale = float(raw_scale) if raw_scale is not None else None
            except ValueError:
                sys.exit(f"invalid --uncensor-scale {raw_scale!r} (expected float 0.0-2.0)")
            cmd_up(
                paths,
                cfg,
                manifest,
                wait_gpu="--wait-gpu" in args[1:],
                basalt_ref=_extract_flag_val(args[1:], "--basalt-ref"),
                quant=_extract_flag_val(args[1:], "--quant"),
                pack_url=_extract_flag_val(args[1:], "--pack-url"),
                uncensor_scale=parsed_scale,
            )
        elif cmd == "down":
            cmd_down(paths, cfg, manifest, everything="--all" in args[1:])
        elif cmd == "status":
            cmd_status(paths, cfg)
        elif cmd == "watch":
            cmd_watch(paths, cfg)
        elif cmd == "console":
            cmd_console(paths, cfg, manifest, no_open="--no-open" in args[1:])
        elif cmd == "config":
            cmd_config(paths, cfg, args[1:])
        elif cmd == "logs":
            cmd_logs(paths, cfg)
        elif cmd == "ask":
            cmd_ask(paths, cfg, args[1:])
        elif cmd == "claude":
            cmd_claude(paths, cfg, args[1:])
        elif cmd == "codex":
            cmd_codex(paths, cfg, args[1:])
        elif cmd == "opencode":
            cmd_opencode(paths, cfg, args[1:])
        elif cmd == "env":
            cmd_env(paths, cfg, args[1:])
        elif cmd == "doctor":
            cmd_doctor(paths, cfg, manifest)
        else:
            sys.exit(USAGE)
    except KeyboardInterrupt:
        print()
        if cmd == "up" and serve_running(paths):
            print(f"Interrupted. The GPU may keep starting in the background. Stop it with: {cli_name()} down")
        sys.exit(130)
