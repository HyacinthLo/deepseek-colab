"""End-to-end offline integration tests for basalt-colab proxy, lifecycle, Control Console, CLI, and safety nets."""
from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
VM_DIR = ROOT / "basalt_colab" / "vm"
PY = sys.executable
FAKE_COLAB = f"{PY} -B {TESTS / 'fake_colab.py'}"
FAKE_SSH = f"{PY} -B {TESTS / 'fake_ssh.py'}"
HF_SECRET = "hf_FAKE0123456789"
MODEL_ID = "DeepSeek-V4.1-Flash-1017-UD-IQ1_M"
TMP: str | None = None
BASE_HOME: str | None = None
COUNTER = [0]


def clean_env() -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("LLM_", "FAKE_", "BASALT_COLAB_")) and k not in ("HF_TOKEN", "CTX", "MMPROJ")
    }
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if ".local/bin" not in p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    return env


def kill_matching(token: str) -> None:
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            if token.encode() in Path(f"/proc/{d}/environ").read_bytes():
                os.kill(int(d), signal.SIGKILL)
        except OSError:
            pass


def procs_with(token: str, word: str) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            env = Path(f"/proc/{d}/environ").read_bytes()
            cmd = Path(f"/proc/{d}/cmdline").read_bytes()
            zombie = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
        except OSError:
            continue
        if token.encode() in env and word.encode() in cmd and not zombie:
            out.append(int(d))
    return out


def read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def setUpModule() -> None:
    global TMP, BASE_HOME
    TMP = tempfile.mkdtemp(prefix="bcolab-int-")
    BASE_HOME = os.path.join(TMP, "home0")
    os.makedirs(os.path.join(TMP, "fake0"))
    env = dict(
        clean_env(),
        BASALT_COLAB_HOME=BASE_HOME,
        BASALT_COLAB_VM_DIR=str(VM_DIR),
        BASALT_COLAB_CLI=FAKE_COLAB,
        FAKE_COLAB_DIR=os.path.join(TMP, "fake0"),
        HOME=os.path.join(TMP, "userhome0"),
    )
    r = subprocess.run([PY, "-B", "-m", "basalt_colab", "setup"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "Setup done" in r.stdout, r.stdout + r.stderr


def tearDownModule() -> None:
    if TMP:
        kill_matching(TMP)
        shutil.rmtree(TMP, ignore_errors=True)


class Resp:
    def __init__(self, status: int, t0: float, t_headers: float, chunks: list[tuple[float, bytes]], headers: dict[str, str]):
        self.status, self.t0, self.t_headers, self.chunks, self.headers = status, t0, t_headers, chunks, headers
        self.raw = b"".join(c for _, c in chunks)
        self.lines = self.raw.decode("utf-8", "replace").split("\n")
        self.comments = [l for l in self.lines if l.startswith(":")]
        self.data = [l[6:] for l in self.lines if l.startswith("data: ")]

    def content(self) -> str:
        if not self.data:
            return json.loads(self.raw)["choices"][0]["message"]["content"]
        out = ""
        for d in self.data:
            if d != "[DONE]":
                for ch in json.loads(d).get("choices", []):
                    out += ch["delta"].get("content") or ""
        return out


class Case:
    def __init__(self, test: unittest.TestCase, **env_overrides: str) -> None:
        COUNTER[0] += 1
        n = COUNTER[0]
        assert TMP and BASE_HOME
        self.dir = tempfile.mkdtemp(prefix=f"{test.id().split('.')[-1]}-", dir=TMP)
        self.home = os.path.join(self.dir, "bhome")
        self.state = os.path.join(self.home, "state")
        self.fake = os.path.join(self.dir, "fake")
        shutil.copytree(BASE_HOME, self.home)
        os.makedirs(self.fake)
        self.port, self.vm_port, self.name = 29100 + 3 * n, 29101 + 3 * n, f"case{n}"
        self.env = dict(
            clean_env(),
            BASALT_COLAB_HOME=self.home,
            BASALT_COLAB_VM_DIR=str(VM_DIR),
            BASALT_COLAB_CLI=FAKE_COLAB,
            BASALT_COLAB_SSH_CMD=FAKE_SSH,
            BASALT_COLAB_SESSION=self.name,
            BASALT_COLAB_PORT=str(self.port),
            BASALT_COLAB_TUNNEL_PORT=str(29500 + 5 * n),
            BASALT_COLAB_POLL_S="0.3",
            BASALT_COLAB_IDLE_MIN="10",
            BASALT_COLAB_MAX_HOURS="1",
            FAKE_COLAB_DIR=self.fake,
            FAKE_VM_PORT=str(self.vm_port),
            FAKE_VM_INNER_PORT=str(self.vm_port + 1),
            FAKE_BOOT_S="1.5",
            HOME=os.path.join(self.dir, "userhome"),
        )
        self.env.update(env_overrides)
        self.key = read(os.path.join(self.state, "api_key")).strip()
        self.procs: list[subprocess.Popen] = []
        self.outs: list = []
        test.addCleanup(self.cleanup)

    def cleanup(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                pass
        kill_matching(self.dir)
        for f in self.outs:
            f.close()

    def cli(self, *args: str, timeout: float = 60, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [PY, "-B", "-m", "basalt_colab"] + list(args),
            env=self.env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def serve(self) -> subprocess.Popen:
        out = open(os.path.join(self.dir, f"serve{len(self.procs)}.out"), "w")
        self.outs.append(out)
        p = subprocess.Popen(
            [PY, "-B", "-m", "basalt_colab", "serve"],
            env=self.env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.procs.append(p)
        self.wait(lambda: self.health_status() is not None or p.poll() is not None, 10)
        assert p.poll() is None, read(out.name)
        return p

    def health_status(self) -> int | None:
        try:
            return self.req(method="GET", path="/health", timeout=3).status
        except OSError:
            return None

    def req(
        self,
        stream: bool = False,
        path: str = "/v1/chat/completions",
        auth: bool = True,
        method: str = "POST",
        body: dict | None = None,
        timeout: float = 60,
        host: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Resp:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if host:
            headers["Host"] = host
        if auth:
            headers["Authorization"] = "Bearer " + self.key
        if extra_headers:
            headers.update(extra_headers)
        if method == "POST" and body is None:
            body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Hello!"}], "stream": stream}
        t0 = time.time()
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
        r = conn.getresponse()
        th, chunks = time.time() - t0, []
        while True:
            d = r.read1(65536)
            if not d:
                break
            chunks.append((time.time() - t0, d))
        conn.close()
        return Resp(r.status, t0, th, chunks, dict(r.getheaders()))

    def wait(self, fn, timeout: float, step: float = 0.1) -> float:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if fn():
                return time.time() - t0
            time.sleep(step)
        raise AssertionError(f"condition not met in {timeout:.0f} s")

    def lines(self, f: str) -> list[dict]:
        p = os.path.join(self.fake, f)
        if not os.path.exists(p):
            return []
        return [json.loads(l) for l in read(p).splitlines() if l.strip()]

    def calls(self, cmd: str | None = None) -> list[list[str]]:
        return [c["argv"] for c in self.lines("calls.log") if cmd is None or c["argv"][0] == cmd]

    def sessions(self) -> dict:
        p = os.path.join(self.fake, "sessions.json")
        return json.loads(read(p)) if os.path.exists(p) else {}

    def session_json(self) -> dict | None:
        p = os.path.join(self.state, "session.json")
        return json.loads(read(p)) if os.path.exists(p) else None

    def res(self, f: str) -> str:
        return os.path.join(self.fake, "vms", self.name, "content", "res", f)

    def log(self) -> str:
        text = ""
        for f in [os.path.join(self.state, "basalt-colab.log")] + [
            os.path.join(self.dir, x) for x in os.listdir(self.dir) if x.endswith(".out")
        ]:
            if os.path.exists(f):
                text += read(f)
        return text

    def assert_all_stopped(self, test: unittest.TestCase) -> None:
        test.assertEqual(self.sessions(), {})
        self.wait(lambda: not procs_with(self.dir, "_watchdog") and not procs_with(self.dir, "fake_ssh.py"), 10)
        test.assertEqual(self.lines("violations.log"), [])


class ProxyLifecycleTests(unittest.TestCase):
    def test_01_stream_cold_start_and_console_to_basalt_webui(self) -> None:
        c = Case(self, FAKE_BOOT_S="5.5")
        c.serve()
        self.assertEqual(c.health_status(), 503)
        # While GPU is OFF, GET / and GET /_console both serve the Unified Control Console
        cold_root = c.req(method="GET", path="/", auth=False)
        self.assertEqual(cold_root.status, 200)
        self.assertIn(b"basalt-colab", cold_root.raw)
        self.assertIn(b"Control Console", cold_root.raw)

        # Streaming first request wakes GPU with stage-aware heartbeat
        r = c.req(stream=True)
        self.assertEqual(r.status, 200)
        self.assertTrue(r.chunks[0][1].startswith(b": starting gpu vm (0 s) - "))
        self.assertTrue(any(cm.startswith(": starting gpu vm (5 s) - ") for cm in r.comments))
        self.assertEqual((r.content(), r.data[-1]), ("ok", "[DONE]"))

        check = json.loads(read(c.res("boot_check.json")))
        self.assertEqual(check["missing"], [])
        self.assertEqual(check["crlf"], [])
        self.assertEqual((check["env"]["CTX"], check["env"]["PARALLEL"], check["env"]["KV_TYPE"], check["env"]["VISION_MODE"]),
                         ("262144", "2", "fp8", "dynamic"))

        # Once READY, GET / proxies natively to Basalt Web UI while GET /_console stays Control Console!
        warm_root = c.req(method="GET", path="/", auth=False)
        self.assertEqual(warm_root.status, 200)
        self.assertIn(b"Basalt Native Web UI", warm_root.raw)
        cons = c.req(method="GET", path="/_console", auth=False)
        self.assertEqual(cons.status, 200)
        self.assertIn(b"Control Console", cons.raw)
        self.assertNotIn(c.key, c.log())

    def test_02_cors_models_and_single_model_routing(self) -> None:
        c = Case(self)
        c.serve()
        opt = c.req(method="OPTIONS", path="/v1/chat/completions", auth=False,
                    extra_headers={"Origin": "http://localhost:3000", "Access-Control-Request-Headers": "Authorization, x-api-key"})
        self.assertEqual(opt.status, 204)
        self.assertEqual(opt.headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("x-api-key", opt.headers.get("Access-Control-Allow-Headers", ""))

        r = c.req(method="GET", path="/v1/models", extra_headers={"Origin": "http://localhost:3000"})
        parsed = json.loads(r.raw)
        self.assertEqual((r.status, parsed["data"][0]["id"], parsed["models"][0]["id"]), (200, MODEL_ID, MODEL_ID))
        self.assertEqual(c.req(host=f"evil.example:{c.port}").status, 403)
        self.assertEqual(c.calls("new"), [])

        anth = json.loads(c.req(path="/v1/messages", auth=False, extra_headers={"x-api-key": c.key},
                                body={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}).raw)
        self.assertEqual(anth["model"], MODEL_ID)

    def test_03_boot_failure_multiprotocol_sse(self) -> None:
        c = Case(self, BASALT_COLAB_BOOT_ENV="FAKE_MODE=fail X_EXTRA=1", FAKE_BOOT_S="1")
        c.serve()
        out = {}
        specs = (("chat", "/v1/chat/completions", True), ("anth", "/v1/messages", True),
                 ("resp", "/v1/responses", True), ("json", "/v1/chat/completions", False))
        ts = [threading.Thread(target=lambda k=k, p=p, s=s: out.__setitem__(k, c.req(path=p, stream=s))) for k, p, s in specs]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual(out["chat"].data[-1], "[DONE]")
        self.assertEqual(json.loads(out["chat"].data[-2])["error"]["code"], 503)
        self.assertIn("event: error", out["anth"].lines)
        self.assertEqual(json.loads(out["anth"].data[-1])["type"], "error")
        self.assertIn("event: error", out["resp"].lines)
        self.assertEqual(json.loads(out["resp"].data[-1])["code"], 503)
        self.assertEqual(out["json"].status, 503)
        c.wait(lambda: c.sessions() == {} and c.session_json() is None, 15)
        c.assert_all_stopped(self)

    def test_04_console_config_hot_reload_usage_ledger_and_cli(self) -> None:
        c = Case(self)
        # doctor & config show/path work before starting
        self.assertEqual(c.cli("doctor").returncode, 0)
        self.assertIn("max_context = 262144", c.cli("config", "show").stdout)
        self.assertIn("config.toml", c.cli("config", "path").stdout)

        # Start detached proxy + G4 with --basalt-ref override
        r_up = c.cli("up", "--basalt-ref", "main")
        self.assertEqual(r_up.returncode, 0, r_up.stdout + r_up.stderr)
        self.assertIn(f"Console    http://127.0.0.1:{c.port}/_console", r_up.stdout)
        self.assertIn(f"Web UI     http://127.0.0.1:{c.port}/", r_up.stdout)
        boot_env = json.loads(read(c.res("boot_check.json")))["env"]
        self.assertEqual(boot_env["BASALT_REF"], "main")

        # Online Control Console config editor: GET -> POST hot change -> POST cold change
        cfg_get = json.loads(c.req(method="GET", path="/_control/config").raw)
        new_toml = cfg_get["toml"].replace('effort = "50"', 'effort = "max"')
        cfg_post = json.loads(c.req(path="/_control/config", body={"toml": new_toml}).raw)
        self.assertEqual((cfg_post["applied"]["effort"], cfg_post["restart_required"]), ("max", False))
        self.assertEqual(json.loads(c.req(body={"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]}).raw)["effort"], "max")

        cold_toml = new_toml.replace("parallel = 2", "parallel = 3")
        cold_post = json.loads(c.req(path="/_control/config", body={"toml": cold_toml}).raw)
        self.assertTrue(cold_post["restart_required"])
        self.assertIn("basalt.parallel", cold_post["cold_changed"])

        # CLI status shows Telemetry & logs API works
        st = c.cli("status")
        self.assertIn("Telemetry  160.4 tok/s · hit 96% · pcie 0.8%", st.stdout)
        logs_res = json.loads(c.req(method="GET", path="/_control/logs?vm=0").raw)
        self.assertIn("serve:", logs_res["local"])

        # Launchers (claude, codex, opencode, env, console --no-open, ask --image + stdin)
        fake_bin = os.path.join(c.dir, "bin")
        os.makedirs(fake_bin)
        for name, line in (
            ("claude", 'echo "url=$ANTHROPIC_BASE_URL model=$ANTHROPIC_MODEL args=$*"'),
            ("codex", 'echo "url=$OPENAI_BASE_URL model=$OPENAI_MODEL args=$*"'),
            ("opencode", 'echo "cfg=$OPENCODE_CONFIG args=$*"'),
        ):
            p = os.path.join(fake_bin, name)
            with open(p, "w") as f:
                f.write(f"#!/bin/sh\n{line}\n")
            os.chmod(p, 0o755)
        c.env["PATH"] = fake_bin + os.pathsep + c.env["PATH"]
        self.assertEqual(c.cli("claude", "-p", "hi").stdout.strip(), f"url=http://127.0.0.1:{c.port} model={MODEL_ID} args=-p hi")
        self.assertEqual(c.cli("codex", "exec").stdout.strip(), f"url=http://127.0.0.1:{c.port}/v1 model={MODEL_ID} args=exec")
        self.assertIn("opencode.json", c.cli("opencode", ".").stdout)
        self.assertIn("export ANTHROPIC_MODEL=", c.cli("env").stdout)
        self.assertIn("$env:OPENAI_BASE_URL", c.cli("env", "--powershell").stdout)
        self.assertIn("/_console", c.cli("console", "--no-open").stdout)

        img = os.path.join(c.dir, "dot.png")
        with open(img, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfake")
        self.assertEqual(c.cli("ask", "--max", "--image", img, "describe", input_text="one word").stdout.strip(), "ok")

        # Stop G4 -> verify usage.jsonl ledger recorded the session!
        r_down = c.cli("down")
        self.assertEqual(r_down.returncode, 0, r_down.stdout + r_down.stderr)
        c.wait(lambda: not procs_with(c.dir, "--detached"), 10)
        c.assert_all_stopped(self)
        usage_lines = [json.loads(l) for l in read(os.path.join(c.state, "usage.jsonl")).splitlines() if l.strip()]
        self.assertEqual(len(usage_lines), 1)
        self.assertEqual(usage_lines[0]["reason"], "down")

    def test_05_alloc_retry_orphan_boot_idle_and_watchdog(self) -> None:
        # 1) Backoff retry succeeds on attempt 2
        c = Case(self, FAKE_ALLOC_FAILS="1", BASALT_COLAB_IDLE_MIN="0.05")
        c.serve()
        self.assertEqual(c.req().content(), "ok")
        self.assertEqual(len(c.calls("new")), 2)
        # Idle auto-stop triggers & records usage.jsonl
        c.wait(lambda: c.sessions() == {}, 20)
        self.assertIn("idle", c.log())
        c.assert_all_stopped(self)
        c.cleanup()

        # 2) Orphan cold-start auto-cancel when waiter drops mid-boot
        c2 = Case(self, FAKE_BOOT_S="10", BASALT_COLAB_ORPHAN_BOOT_S="1.0")
        c2.serve()
        conn = http.client.HTTPConnection("127.0.0.1", c2.port, timeout=10)
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps({"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
                     headers={"Authorization": "Bearer " + c2.key, "Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read1(128)
        c2.wait(lambda: c2.calls("new"), 10)
        conn.close()
        c2.wait(lambda: "all waiting clients disconnected" in c2.log(), 15)
        c2.wait(lambda: c2.sessions() == {} and c2.session_json() is None, 15)
        c2.assert_all_stopped(self)
        c2.cleanup()

        # 3) Watchdog stops VM after proxy SIGKILL + HF_TOKEN never leaks
        c3 = Case(self, BASALT_COLAB_IDLE_MIN="0.05", HF_TOKEN=HF_SECRET)
        p = c3.serve()
        self.assertEqual(c3.req().status, 200)
        check = json.loads(read(c3.res("boot_check.json")))
        self.assertEqual(check["secrets"], ["hf_hdr", "key"])
        self.assertNotIn(HF_SECRET, c3.log())
        os.kill(p.pid, signal.SIGKILL)
        c3.wait(lambda: c3.sessions() == {}, 20)
        self.assertIn("proxy gone", c3.log())
        self.assertEqual(c3.cli("down").returncode, 0)
        c3.assert_all_stopped(self)


if __name__ == "__main__":
    unittest.main()
