"""Colab CLI wrapper, SSH tunnel builder, process identity, lockfile, bundle builder, and logger."""
from __future__ import annotations

import http.client
import io
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

from basalt_colab.paths import IS_WIN, Paths

NO_WINDOW, DETACHED, NEW_GROUP, BREAKAWAY = 0x08000000, 0x00000008, 0x00000200, 0x01000000
CHILD_FLAGS = NO_WINDOW if IS_WIN else 0
VM_PORT = 18080


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if extra:
        env.update(extra)
    return env


def split_cmd(s: str, win: bool = IS_WIN) -> list[str]:
    if os.path.exists(s):
        return [s]
    if win:
        return [t[1:-1] if t.startswith('"') else t for t in re.findall(r'"[^"]*"|\S+', s)]
    return shlex.split(s)


def join_cmd(args: list[str], win: bool = IS_WIN) -> str:
    return subprocess.list2cmdline(args) if win else shlex.join(args)


def find_tool(name: str) -> str:
    if IS_WIN:
        p = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "OpenSSH", name + ".exe")
        if os.path.exists(p):
            return p
    return shutil.which(name) or name


def colab_cmd() -> list[str]:
    override = os.environ.get("BASALT_COLAB_CLI") or os.environ.get("LLM_COLAB")
    if override:
        return split_cmd(override) + ["--auth", "oauth2"]
    exe = shutil.which("colab") or os.path.join(
        os.path.expanduser("~"), ".local", "bin", "colab.exe" if IS_WIN else "colab"
    )
    return [exe, "--auth", "oauth2"]


def colab_present() -> bool:
    if os.environ.get("BASALT_COLAB_CLI") or os.environ.get("LLM_COLAB"):
        return True
    cmd0 = colab_cmd()[0]
    return os.path.exists(cmd0) or bool(shutil.which(cmd0))


def kill_tree(pid: int, hard: bool = False) -> None:
    try:
        if IS_WIN:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CHILD_FLAGS,
                timeout=30,
            )
        else:
            os.killpg(pid, signal.SIGKILL if hard else signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def run_cmd(cmd: list[str], timeout: float, input_text: str | None = None) -> tuple[int, str]:
    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL if input_text is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env(),
            creationflags=CHILD_FLAGS,
            start_new_session=not IS_WIN,
        )
    except OSError as e:
        return 127, f"{cmd[0]}: {e}"
    try:
        out = p.communicate(None if input_text is None else input_text.encode(), timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        kill_tree(p.pid, hard=True)
        try:
            p.communicate(timeout=10)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        return 124, "timeout"
    return p.returncode, out.decode("utf-8", "replace")


def colab_run(args: list[str], timeout: float = 120, input_text: str | None = None) -> tuple[int, str]:
    return run_cmd(colab_cmd() + args, timeout, input_text)


def vmx(session: str, cmd: str, timeout: int = 100) -> tuple[int, str]:
    code = (
        f"import subprocess\n"
        f"r = subprocess.run({json.dumps(cmd)}, shell=True, capture_output=True, text=True, timeout={timeout})\n"
        f"print(r.stdout[-20000:] + r.stderr[-4000:])\n"
    )
    return colab_run(["exec", "-s", session, "--timeout", str(timeout + 20)], timeout + 40, code)


def colab_usage() -> tuple[float | None, float | None]:
    rc, out = colab_run(["usage"], 90)
    bal = re.search(r"balance: ([0-9.]+)", out)
    rate = re.search(r"rate: ([0-9.]+)", out)
    return (float(bal.group(1)) if bal else None), (float(rate.group(1)) if rate else None)


def session_alive(session: str) -> bool:
    rc, out = colab_run(["sessions"], 90)
    return f"[{session}]" in out


def vm_timeline(session: str) -> tuple[int, str]:
    return vmx(session, "cat /content/res/timeline.txt 2>/dev/null; ls /content/res/READY 2>/dev/null", 40)


def _win_start(pid: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return "?" if ctypes.get_last_error() == 5 else None
    try:
        code, times = wintypes.DWORD(), [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return "?"
        if code.value != 259:
            return None
        if not k32.GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
            return "?"
        return str(times[0].dwHighDateTime << 32 | times[0].dwLowDateTime)
    finally:
        k32.CloseHandle(h)


def proc_start(pid: int | None) -> str | None:
    """Start-time token of a live process ('?' if alive but unknown), None if gone."""
    if not pid:
        return None
    if IS_WIN:
        try:
            return _win_start(pid)
        except Exception:
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                    capture_output=True,
                    text=True,
                    creationflags=CHILD_FLAGS,
                    timeout=30,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                return "?"
            return "?" if f'"{pid}"' in out else None
    try:
        rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if rest[0] in ("Z", "X") else rest[19]
    except FileNotFoundError:
        if os.path.isdir("/proc"):
            return None
    except (OSError, IndexError):
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return "?"
    except OSError:
        return None
    return "?"


def me() -> dict[str, Any]:
    return {"pid": os.getpid(), "start": proc_start(os.getpid())}


def alive(rec: dict[str, Any] | None) -> bool:
    if not rec or not rec.get("pid"):
        return False
    st = proc_start(int(rec["pid"]))
    return st is not None and (st == "?" or rec.get("start") in (None, "?", st))


def kill_rec(rec: dict[str, Any] | None) -> None:
    if alive(rec):
        kill_tree(int(rec["pid"]))


def remove_file(path: Path) -> None:
    for _ in range(30):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.1)


def read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def read_api_key(paths: Paths) -> str:
    return paths.key_file.read_text(encoding="utf-8").strip()


def redact(text: str, key: str = "") -> str:
    text = re.sub(r"(Bearer\s+)\S+", r"\1***", text)
    if key:
        text = text.replace(key, "***")
    return text


class Logger:
    def __init__(self, paths: Paths, echo: bool = True) -> None:
        self.paths = paths
        self.echo = echo
        self._lock = threading.Lock()

    def _key(self) -> str:
        try:
            return read_api_key(self.paths)
        except OSError:
            return ""

    def log(self, msg: str, echo: bool | None = None) -> None:
        clean = redact(msg, self._key())
        with self._lock:
            try:
                if self.paths.log_file.exists() and self.paths.log_file.stat().st_size > 1_000_000:
                    os.replace(self.paths.log_file, str(self.paths.log_file) + ".1")
            except OSError:
                pass
            try:
                self.paths.state_dir.mkdir(parents=True, exist_ok=True)
                with self.paths.log_file.open("a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + clean + "\n")
            except OSError:
                pass
        if self.echo if echo is None else echo:
            print(clean, flush=True)


def read_lock(paths: Paths) -> dict[str, Any] | None:
    raw = read_bytes(paths.lock_file)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        try:
            young = time.time() - paths.lock_file.stat().st_mtime < 5
        except OSError:
            return None
        return {"pid": None, "busy": True} if young else {"pid": None}


def take_lock(paths: Paths, info: dict[str, Any]) -> dict[str, Any] | None:
    paths.ensure_dirs()
    for _ in range(5):
        try:
            fd = os.open(str(paths.lock_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raw, held = read_bytes(paths.lock_file), read_lock(paths)
            if held and (held.get("busy") or alive(held)):
                return held
            if read_bytes(paths.lock_file) == raw:
                try:
                    paths.lock_file.unlink()
                except OSError:
                    time.sleep(0.2)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(info, **me()), f)
        return None
    return read_lock(paths) or {"pid": None, "busy": True}


def release_lock(paths: Paths) -> None:
    held = read_lock(paths)
    if held and held.get("pid") == os.getpid():
        remove_file(paths.lock_file)


def serve_running(paths: Paths) -> dict[str, Any] | None:
    held = read_lock(paths)
    return held if held and held.get("port") and alive(held) else None


def load_session(paths: Paths) -> dict[str, Any] | None:
    for _ in range(5):
        try:
            return json.loads(paths.session_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.1)
    return None


def save_session(paths: Paths, s: dict[str, Any]) -> None:
    paths.ensure_dirs()
    tmp = Path(f"{paths.session_file}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    for _ in range(30):
        try:
            os.replace(tmp, paths.session_file)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, paths.session_file)


def cleanup_local(paths: Paths, s: dict[str, Any]) -> None:
    kill_rec(s.get("tunnel"))
    kill_rec(s.get("watchdog"))
    remove_file(paths.session_file)


def vm_files(vm_dir: Path) -> list[str]:
    out: list[str] = []
    for d, dirs, files in os.walk(vm_dir):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if not f.endswith(".pyc"):
                out.append(os.path.relpath(os.path.join(d, f), vm_dir).replace(os.sep, "/"))
    return out


def missing_vm_files(vm_dir: Path) -> list[str]:
    have = set(vm_files(vm_dir))
    refs: set[str] = set()
    scripts = ("boot.sh", "serve.sh", "build.sh")
    for f in scripts:
        p = vm_dir / f
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            refs |= set(re.findall(r"(?:\$V|/content/vm)/([\w.-]+\.(?:py|sh|txt|c|json))\b", text))
        except OSError:
            refs.add(f)
    refs |= set(scripts) | {"manifest.json"}
    return sorted(r for r in refs if r not in have and "$" not in r)


def build_bundle(paths: Paths, out_path: Path) -> None:
    """Writes VM tarball with LF line endings and strict 0600 secret permissions."""
    items: list[tuple[Path | None, str, int, bytes | None]] = [
        (paths.vm_dir / f, "vm/" + f, 0o644, None) for f in vm_files(paths.vm_dir)
    ] + [(paths.key_file, "secrets/key", 0o600, None)]
    if os.environ.get("HF_TOKEN"):
        hdr = f"Authorization: Bearer {os.environ['HF_TOKEN'].strip()}\n".encode()
        items.append((None, "secrets/hf_hdr", 0o600, hdr))
    fd = os.open(str(out_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f, tarfile.open(fileobj=f, mode="w:gz") as tar:
        for src, arc, mode, data in items:
            if src is not None:
                data = src.read_bytes().replace(b"\r\n", b"\n")
            assert data is not None
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mode, ti.mtime = len(data), mode, int(time.time())
            tar.addfile(ti, io.BytesIO(data))


def boot_cmd(env_vars: dict[str, str]) -> str:
    envs = " ".join(f"{k}='{v}'" for k, v in env_vars.items())
    return (
        "cd /content && umask 077 && tar xzf bundle.tgz && rm -f bundle.tgz && "
        "chmod 700 secrets && chmod 600 secrets/* && "
        f"mkdir -p res && ({envs} setsid nohup bash vm/boot.sh > res/boot.out 2>&1 < /dev/null &) && echo BOOT_STARTED"
    )


def ssh_cmd(session: str, ssh_key: Path, port: int, win: bool = IS_WIN) -> list[str]:
    proxy = join_cmd(colab_cmd() + ["ssh", "--proxy-mode", "-s", session, "-i", str(ssh_key)], win).replace("%", "%%")
    override = os.environ.get("BASALT_COLAB_SSH_CMD") or os.environ.get("LLM_SSH_CMD")
    base = split_cmd(override, win) if override else [find_tool("ssh")]
    null = "NUL" if win else "/dev/null"
    return base + [
        "-N", "-F", null,
        "-o", "ProxyCommand=" + proxy,
        "-o", "User=root",
        "-i", str(ssh_key),
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=" + null,
        "-o", "LogLevel=ERROR",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-L", f"127.0.0.1:{port}:127.0.0.1:{VM_PORT}",
        session,
    ]


def pick_port() -> int:
    base = int(os.environ.get("BASALT_COLAB_TUNNEL_PORT") or os.environ.get("LLM_TUNNEL_PORT") or "0")
    for p in (range(base, base + 50) if base else [0]):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            return int(s.getsockname()[1])
        except OSError:
            continue
        finally:
            s.close()
    raise RuntimeError("no free local port for the SSH tunnel")


def http_call(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
    host: str = "127.0.0.1",
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        return conn, conn.getresponse()
    except BaseException:
        conn.close()
        raise


def spawn_detached(args: list[str], out: Any = None, extra_env: dict[str, str] | None = None) -> subprocess.Popen[Any]:
    kw: dict[str, Any] = dict(
        stdin=subprocess.DEVNULL,
        stdout=out or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if out else subprocess.DEVNULL,
        env=child_env(extra_env),
    )
    if not IS_WIN:
        return subprocess.Popen(args, start_new_session=True, **kw)
    try:
        return subprocess.Popen(args, creationflags=DETACHED | NEW_GROUP | NO_WINDOW | BREAKAWAY, **kw)
    except OSError:
        return subprocess.Popen(args, creationflags=DETACHED | NEW_GROUP | NO_WINDOW, **kw)
