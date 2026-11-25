"""Local HTTP gateway: OpenAI/Anthropic proxy, Basalt Web UI, Unified Control Console, and /_control/* API."""
from __future__ import annotations

import hmac
import http.client
import json
import os
import select
import signal
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from basalt_colab.colab import (
    Logger,
    read_api_key,
    read_bytes,
    redact,
    release_lock,
    session_alive,
    take_lock,
    vmx,
)
from basalt_colab.config import Config, ConfigError, dump_toml, load_config, parse_toml, save_config
from basalt_colab.console import render_console_html
from basalt_colab.lifecycle import Manager, UPSTREAM_TIMEOUT
from basalt_colab.manifest import Manifest
from basalt_colab.paths import IS_WIN, Paths
from basalt_colab.usage import UsageLedger

MAX_BODY = 64 << 20
HOP = {
    "connection", "keep-alive", "proxy-connection", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "host", "accept-encoding", "expect", "server", "date",
}


def write_opencode_config(paths: Paths, cfg: Config, port: int | None = None) -> None:
    p = port or cfg.proxy.port
    model_id = cfg.basalt.effective_model_alias
    image = {"attachment": True, "modalities": {"input": ["text", "image"], "output": ["text"]}} if cfg.basalt.vision != "off" else {}
    models = {
        model_id: dict(
            {
                "name": f"DeepSeek-V4.1-Flash {cfg.basalt.quant.upper()} (Basalt, effort {cfg.basalt.effort})",
                "tool_call": True,
                "reasoning": True,
                "interleaved": {"field": "reasoning_content"},
                "options": {"reasoningEffort": str(cfg.basalt.effort)},
                "limit": {"context": cfg.basalt.max_context, "output": 32768},
            },
            **image,
        )
    }
    doc = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "basalt": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DeepSeek-V4.1-Flash on Colab (Basalt)",
                "options": {
                    "baseURL": f"http://{cfg.proxy.host}:{p}/v1",
                    "apiKey": "{file:%s}" % str(paths.key_file).replace("\\", "/"),
                    "headerTimeout": 1800000,
                    "chunkTimeout": 900000,
                },
                "models": models,
            }
        },
        "model": f"basalt/{model_id}",
        "small_model": f"basalt/{model_id}",
    }
    paths.ensure_dirs()
    paths.opencode_file.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def rewrite_request(body: bytes, cfg: Config, path: str = "/v1/chat/completions") -> bytes:
    """Single-model routing to cfg.basalt.effective_model_alias (unless strict_model) + saved default effort injection."""
    try:
        req = json.loads(body)
    except ValueError:
        return body
    if not isinstance(req, dict):
        return body
    changed = False
    target_model = cfg.basalt.effective_model_alias
    if not cfg.proxy.strict_model and req.get("model") != target_model:
        req["model"] = target_model
        changed = True
    effort = str(cfg.basalt.effort)
    if effort != "50":
        if path == "/v1/chat/completions" and req.get("reasoning_effort") is None:
            req["reasoning_effort"] = effort
            changed = True
        elif path == "/v1/responses":
            reasoning = req.get("reasoning")
            if reasoning is None:
                req["reasoning"] = {"effort": effort}
                changed = True
            elif isinstance(reasoning, dict) and reasoning.get("effort") is None:
                req["reasoning"] = dict(reasoning, effort=effort)
                changed = True
    return json.dumps(req).encode() if changed else body


def abort_upstream(conn: http.client.HTTPConnection | None) -> None:
    if conn is None:
        return
    try:
        if conn.sock:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        conn.close()
    except OSError:
        pass


def error_obj(code: int, msg: str, kind: str = "unavailable_error") -> dict[str, Any]:
    return {"error": {"code": code, "message": msg, "type": kind}}


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not IS_WIN
    mgr: Manager
    paths: Paths

    def server_bind(self) -> None:
        if IS_WIN:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        e = sys.exc_info()[1]
        if not isinstance(e, (OSError, http.client.HTTPException)):
            self.mgr.logger.log(f"handler error: {e!r}", echo=False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 900
    disable_nagle_algorithm = True
    server_version, sys_version = "basalt-colab", ""
    server: ProxyServer

    def setup(self) -> None:
        self.request.settimeout(30)
        super().setup()

    def log_message(self, *args: Any) -> None:
        pass

    def chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

    def send_cors(self, preflight: bool = False) -> None:
        if not preflight and not self.headers.get("Origin"):
            return
        self.send_header("Access-Control-Allow-Origin", "*")
        if preflight:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header(
                "Access-Control-Allow-Headers",
                asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta",
            )
            self.send_header("Access-Control-Max-Age", "86400")

    def send_bytes(self, code: int, ctype: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, code: int, obj: Any) -> None:
        self.send_bytes(code, "application/json", json.dumps(obj).encode())

    def start_chunked(self, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_cors()
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def fail(self, code: int, msg: str, sse: bool = False, obj: dict[str, Any] | None = None) -> None:
        obj = obj or error_obj(code, msg)
        if not sse:
            self.send_json(code, obj)
            return
        path = self.path.split("?")[0]
        err = (
            obj.get("error")
            if isinstance(obj, dict) and isinstance(obj.get("error"), dict)
            else {"code": code, "message": msg or str(obj), "type": "unavailable_error"}
        )
        if path == "/v1/messages":
            payload = b"event: error\ndata: " + json.dumps({"type": "error", "error": err}).encode() + b"\n\n"
        elif path == "/v1/responses":
            payload = (
                b"event: error\ndata: "
                + json.dumps({"type": "error", "code": err.get("code", code), "message": err.get("message", msg)}).encode()
                + b"\n\n"
            )
        else:
            payload = b"data: " + json.dumps({"error": err}).encode() + b"\n\ndata: [DONE]\n\n"
        try:
            self.chunk(payload)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass
        self.close_connection = True

    def authed(self) -> bool:
        key = read_api_key(self.server.paths).encode()
        got = self.headers.get("Authorization", "").encode()
        xkey = self.headers.get("x-api-key", "").encode()
        if hmac.compare_digest(got, b"Bearer " + key) or (xkey and hmac.compare_digest(xkey, key)):
            return True
        self.send_json(401, error_obj(401, "Invalid API Key", "authentication_error"))
        return False

    def read_body(self) -> bytes | None:
        try:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                parts, total = [], 0
                while True:
                    n = int(self.rfile.readline(1024).split(b";")[0].strip(), 16)
                    if n == 0:
                        while self.rfile.readline(1024) not in (b"\r\n", b"\n", b""):
                            pass
                        return b"".join(parts)
                    total += n
                    if total > MAX_BODY:
                        break
                    parts.append(self.rfile.read(n))
                    self.rfile.readline(8)
            else:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= MAX_BODY:
                    return self.rfile.read(n) if n > 0 else b""
            self.close_connection = True
            self.fail(413, "request body too large", obj=error_obj(413, "request body too large", "invalid_request_error"))
        except ValueError:
            self.close_connection = True
            self.fail(400, "bad request body", obj=error_obj(400, "bad request body", "invalid_request_error"))
        return None

    def host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        return not self.server.mgr.cfg.proxy.loopback or name in ("", "127.0.0.1", "localhost", "[::1]")

    def client_gone(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (ConnectionError, TimeoutError):
            return True
        except (OSError, ValueError):
            return False

    def do_OPTIONS(self) -> None:
        if not self.host_ok():
            return self.fail(403, "unexpected Host header", obj=error_obj(403, "unexpected Host header", "permission_error"))
        self.send_response(204)
        self.send_cors(preflight=True)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self.route(None)

    def do_POST(self) -> None:
        body = self.read_body()
        if body is not None:
            self.route(body)

    def models_payload(self) -> dict[str, Any]:
        cfg = self.server.mgr.cfg
        entry = {
            "id": cfg.basalt.effective_model_alias,
            "object": "model",
            "owned_by": "basalt",
            "created": 0,
            "quant": cfg.basalt.quant,
            "reasoning_effort": str(cfg.basalt.effort),
            "context_length": cfg.basalt.max_context,
            "image_input": cfg.basalt.vision != "off",
        }
        return {"object": "list", "data": [entry], "models": [entry]}

    def route(self, body: bytes | None) -> None:
        mgr = self.server.mgr
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if not self.host_ok():
            return self.fail(403, "unexpected Host header", obj=error_obj(403, "unexpected Host header", "permission_error"))
        if path in ("/_console", "/_console/") and body is None:
            return self.send_bytes(200, "text/html; charset=utf-8", render_console_html())
        if path.startswith("/_control/"):
            return self.control(path, parsed.query, body)
        if path in ("/v1/models", "/models") and body is None:
            if self.authed():
                self.send_json(200, self.models_payload())
            return
        if body and path in ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses"):
            body = rewrite_request(body, mgr.cfg, path)
        if body is not None and path.startswith("/v1/"):
            if self.authed():
                t0 = time.time()
                self.held(body)
                mgr.logger.log(f"{self.command} {path} done in {time.time() - t0:.1f} s", echo=False)
            return
        if body is None and mgr.ready():
            return self.forward(None)
        if body is not None and mgr.enter():
            try:
                return self.forward(body)
            finally:
                mgr.leave()
        if path == "/" and body is None:
            # GPU is off or booting: serve the unified Control Console directly at /
            return self.send_bytes(200, "text/html; charset=utf-8", render_console_html())
        if path in ("/health", "/v1/health"):
            return self.send_json(503, dict(error_obj(503, f"gpu vm {mgr.state}"), status=mgr.state))
        self.fail(503, f"gpu vm {mgr.state}; a POST to /v1/... or /_console starts it")

    def held(self, body: bytes) -> None:
        mgr = self.server.mgr
        if mgr.enter():
            try:
                return self.forward(body)
            finally:
                mgr.leave()
        stream = False
        try:
            stream = bool(json.loads(body).get("stream")) if body else False
        except (ValueError, AttributeError):
            pass
        if not stream:
            err = mgr.acquire(is_gone=self.client_gone)
            if err == "client disconnected":
                self.close_connection = True
                return
            if err:
                return self.fail(503, err)
        else:
            self.start_chunked("text/event-stream")

            def tick(n: int) -> None:
                stage = mgr.activity()
                msg = f": starting gpu vm ({n} s) - {stage}\n\n" if stage else f": starting gpu vm ({n} s)\n\n"
                self.chunk(msg.encode("utf-8"))

            try:
                tick(0)
                err = mgr.acquire(tick, is_gone=self.client_gone)
            except OSError:
                self.close_connection = True
                return
            if err == "client disconnected":
                self.close_connection = True
                return
            if err:
                return self.fail(503, err, sse=True)
        try:
            self.forward(body, sse=stream)
        finally:
            mgr.leave()

    def forward(self, body: bytes | None, sse: bool = False) -> None:
        mgr = self.server.mgr
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        done, cur_conn, aborted = threading.Event(), [None], [False]

        def watch_client() -> None:
            sock = self.connection
            while not done.wait(0.5):
                try:
                    readable, _, _ = select.select([sock], [], [], 0)
                    if not readable:
                        continue
                    if sock.recv(1, socket.MSG_PEEK) == b"":
                        aborted[0] = True
                        abort_upstream(cur_conn[0])
                        return
                    return
                except (ConnectionError, TimeoutError):
                    aborted[0] = True
                    abort_upstream(cur_conn[0])
                    return
                except (OSError, ValueError):
                    return

        threading.Thread(target=watch_client, daemon=True).start()
        conn: http.client.HTTPConnection | None = None
        try:
            for attempt in range(2):
                assert mgr.port is not None
                c = http.client.HTTPConnection("127.0.0.1", mgr.port, timeout=UPSTREAM_TIMEOUT)
                cur_conn[0] = c
                try:
                    c.request(self.command, self.path, body=body, headers=headers)
                    resp = c.getresponse()
                    conn = c
                    break
                except (OSError, http.client.HTTPException) as e:
                    c.close()
                    if aborted[0]:
                        self.close_connection = True
                        return
                    if attempt == 0 and mgr.tunnel_failed():
                        continue
                    return self.fail(502, f"upstream unreachable ({e.__class__.__name__})", sse)
            if sse:
                if resp.status != 200:
                    raw = resp.read()
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        obj = None
                    if not isinstance(obj, dict) or "error" not in obj:
                        obj = error_obj(resp.status, raw.decode("utf-8", "replace")[:500], "upstream_error")
                    return self.fail(resp.status, "", True, obj)
                self.pipe(resp, True)
                return
            self.send_response(resp.status, resp.reason)
            has_acao = False
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    if k.lower() == "access-control-allow-origin":
                        has_acao = True
                    self.send_header(k, v)
            if not has_acao:
                self.send_cors()
            n = resp.getheader("Content-Length")
            if n is not None and not resp.chunked:
                self.send_header("Content-Length", n)
                self.end_headers()
                self.pipe(resp, False)
            else:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.pipe(resp, True)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            done.set()
            if conn is not None:
                conn.close()

    def pipe(self, resp: http.client.HTTPResponse, chunked: bool) -> None:
        while True:
            data = resp.read1(65536) if chunked else resp.read(65536)
            if not data:
                break
            if chunked:
                self.chunk(data)
            else:
                self.wfile.write(data)
        if chunked:
            self.wfile.write(b"0\r\n\r\n")

    def control(self, path: str, query: str, body: bytes | None) -> None:
        if not self.authed():
            return
        mgr = self.server.mgr
        paths = self.server.paths
        qs = urllib.parse.parse_qs(query)

        if path == "/_control/status":
            return self.send_json(200, mgr.status())

        if path == "/_control/usage":
            days = int((qs.get("days") or ["30"])[0])
            return self.send_json(200, mgr.usage_summary(days=days))

        if path == "/_control/logs":
            raw = read_bytes(paths.log_file)
            key = read_api_key(paths)
            local_txt = redact(raw.decode("utf-8", "replace")[-6000:], key) if raw is not None else ""
            vm_txt = ""
            if (qs.get("vm") or ["0"])[0] == "1" and session_alive(mgr.cfg.colab.session):
                _, out = vmx(
                    mgr.cfg.colab.session,
                    "cat /content/res/timeline.txt 2>/dev/null; tail -n 25 /content/res/server.log 2>/dev/null; "
                    "tail -n 15 /root/.cache/basalt/*/engine.log 2>/dev/null || true",
                    40,
                )
                vm_txt = redact(out, key)
            return self.send_json(200, {"local": local_txt, "vm": vm_txt})

        if path == "/_control/config":
            if body is None:
                disk_cfg = load_config(paths, apply_env=False)
                return self.send_json(200, {"config": disk_cfg.to_dict(), "toml": dump_toml(disk_cfg), "path": str(paths.config_file)})
            try:
                prev_disk = load_config(paths, apply_env=False)
                payload = json.loads(body or b"{}")
                if "toml" in payload:
                    new_cfg = parse_toml(str(payload["toml"]))
                elif "config" in payload:
                    new_cfg = Config.from_dict(payload["config"])
                else:
                    raise ConfigError("expected 'toml' or 'config' in request body")
                save_config(new_cfg, paths)
                hot, cold = mgr.reload_config(new_cfg, prev_cfg=prev_disk)
                try:
                    write_opencode_config(paths, mgr.cfg, self.server.server_port)
                except OSError:
                    pass
                mgr.logger.log(f"config saved: hot={hot} cold={cold}", echo=False)
                return self.send_json(200, {
                    "applied": {
                        "effort": mgr.cfg.basalt.effort,
                        "idle_min": mgr.cfg.colab.idle_min,
                        "max_hours": mgr.cfg.colab.max_hours,
                        "wait_gpu": mgr.cfg.colab.wait_gpu,
                        "strict_model": mgr.cfg.proxy.strict_model,
                    },
                    "hot_changed": hot,
                    "cold_changed": cold,
                    "restart_required": bool(cold),
                })
            except (ValueError, ConfigError) as e:
                return self.fail(400, str(e), obj=error_obj(400, str(e), "invalid_request_error"))

        if path == "/_control/reload" and body is not None:
            try:
                payload = json.loads(body or b"{}")
                new_cfg = load_config(paths, apply_env=False)
                if isinstance(payload, dict) and payload:
                    merged = new_cfg.to_dict()
                    for sec in ("colab", "proxy", "basalt"):
                        if isinstance(payload.get(sec), dict):
                            merged[sec].update(payload[sec])
                    # Also accept convenient flat hot keys: effort, idle_min, max_hours, wait_gpu, strict_model
                    if "effort" in payload:
                        merged["basalt"]["effort"] = payload["effort"]
                    if "idle_min" in payload:
                        merged["colab"]["idle_min"] = payload["idle_min"]
                    if "max_hours" in payload:
                        merged["colab"]["max_hours"] = payload["max_hours"]
                    if "wait_gpu" in payload:
                        merged["colab"]["wait_gpu"] = payload["wait_gpu"]
                    if "strict_model" in payload:
                        merged["proxy"]["strict_model"] = payload["strict_model"]
                    new_cfg = Config.from_dict(merged)
                hot, cold = mgr.reload_config(new_cfg)
                try:
                    write_opencode_config(paths, mgr.cfg, self.server.server_port)
                except OSError:
                    pass
                return self.send_json(200, {
                    "applied": {
                        "effort": mgr.cfg.basalt.effort,
                        "idle_min": mgr.cfg.colab.idle_min,
                        "max_hours": mgr.cfg.colab.max_hours,
                        "wait_gpu": mgr.cfg.colab.wait_gpu,
                        "strict_model": mgr.cfg.proxy.strict_model,
                    },
                    "hot_changed": hot,
                    "cold_changed": cold,
                    "restart_required": bool(cold),
                })
            except (ValueError, ConfigError) as e:
                return self.fail(400, str(e), obj=error_obj(400, str(e), "invalid_request_error"))

        if path not in ("/_control/up", "/_control/down") or body is None:
            return self.fail(404, "unknown control request", obj=error_obj(404, "unknown control request", "not_found_error"))

        try:
            opts = json.loads(body or b"{}")
        except ValueError:
            opts = {}
        self.start_chunked("text/plain; charset=utf-8")
        say = lambda text: self.chunk((text + "\n").encode())
        try:
            if path == "/_control/up":
                with mgr.cond:
                    joined = mgr.state == "starting"
                    t0, sent = (mgr.begun if joined else time.time()), [0 if joined else len(mgr.steps)]

                def emit_steps() -> None:
                    done = mgr.steps
                    sent[0] = 0 if sent[0] > len(done) else sent[0]
                    for _, at, label in done[sent[0]:]:
                        if label:
                            say(f"step {max(0, int(at - t0))} {label}")
                    sent[0] = len(done)

                def tick(n: int) -> None:
                    emit_steps()
                    say(f"wait {n} {mgr.activity()}")

                err = mgr.acquire(
                    tick,
                    every=1,
                    explicit=True,
                    wait_gpu=opts.get("wait_gpu"),
                    basalt_ref=opts.get("basalt_ref"),
                    quant=opts.get("quant"),
                    pack_url=opts.get("pack_url"),
                    uncensor_scale=opts.get("uncensor_scale"),
                )
                if not err:
                    mgr.leave(served=0)
                    emit_steps()
                say(f"FAILED {err}" if err else f"READY {int(time.time() - t0)}")
                self.wfile.write(b"0\r\n\r\n")
                return

            mgr.pop_last_stopped()
            mgr.request_stop("down")
            still = mgr.wait_off(lambda n: say(f"wait {n} stopping the GPU"), every=1)
            cu, ran = mgr.pop_last_stopped()
            say("STILL" if still else f"STOPPED {cu} {ran}")
            self.wfile.write(b"0\r\n\r\n")
            if opts.get("exit"):
                self.close_connection = True
                threading.Thread(target=self.server.shutdown, daemon=True).start()
        except OSError:
            self.close_connection = True


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def serve(paths: Paths, cfg: Config, manifest: Manifest, detached: bool = False) -> None:
    logger = Logger(paths, echo=not detached)
    if not paths.key_file.exists():
        sys.exit(f"missing {paths.key_file}; run: basalt-colab setup")
    held = take_lock(paths, {
        "port": cfg.proxy.port,
        "detached": detached,
        "cold_signature": cfg.cold_signature(),
        "engine_summary": cfg.describe_engine(),
    })
    if held:
        sys.exit(f"basalt-colab serve is already running (pid {held.get('pid')}, port {held.get('port')})")
    try:
        srv = ProxyServer((cfg.proxy.bind, cfg.proxy.port), Handler)
    except OSError as e:
        release_lock(paths)
        sys.exit(f"cannot serve on {cfg.proxy.bind}:{cfg.proxy.port}: {e} (set BASALT_COLAB_PORT)")
    ledger = UsageLedger(paths.usage_file)
    srv.paths = paths
    srv.mgr = Manager(paths, cfg, manifest, ledger, logger)
    for sig in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _interrupt)
    if paths.session_file.exists():
        srv.mgr.adopt()
    threading.Thread(target=srv.mgr.monitor, daemon=True).start()
    logger.log(
        f"serve: http://{cfg.proxy.host}:{cfg.proxy.port}/v1 (console /_console), "
        f"{cfg.basalt.effective_model_alias} ({cfg.describe_engine()}, {cfg.describe_uncensor()}), "
        f"session {cfg.colab.session} (idle {cfg.colab.idle_min:.1f} min, cap {cfg.colab.max_hours:.1f} h)"
    )
    try:
        srv.serve_forever(0.5)
    except KeyboardInterrupt:
        logger.log("serve: interrupted")
    finally:
        srv.server_close()
        try:
            srv.mgr.shutdown()
        except KeyboardInterrupt:
            logger.log("serve: second interrupt, leaving the VM to the watchdog")
        release_lock(paths)
        logger.log("serve: exit")
