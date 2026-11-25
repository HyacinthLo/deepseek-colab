"""Mock Basalt v0.5.7 HTTP server for offline tests: /, /health, /status, /metrics, /v1/*."""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, DELAY = int(sys.argv[1]), float(sys.argv[2])
KEY, START = os.environ.get("MOCK_KEY", ""), time.time()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_bytes(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, code, obj):
        self.send_bytes(code, "application/json", json.dumps(obj).encode())

    def _authed(self):
        return self.headers.get("Authorization") == "Bearer " + KEY or self.headers.get("x-api-key") == KEY

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send_bytes(200, "text/html; charset=utf-8", b"<html><title>Basalt Native Web UI</title></html>")
        if path in ("/health", "/v1/health"):
            if time.time() - START < DELAY:
                return self.send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
            return self.send_json(200, {"status": "ok", "service": "basalt"})
        if path == "/status":
            if not self._authed():
                return self.send_json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
            return self.send_json(200, {"busy": False})
        if path == "/metrics":
            if not self._authed():
                return self.send_json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
            return self.send_json(200, {
                "live": {"queued": 0},
                "requests": [{"decode_tok_s": 160.4, "hit_rate": 0.965, "pcie_share": 0.008, "draft_len": 3.5}],
            })
        self.send_json(404, {"error": {"code": 404, "message": "not found", "type": "not_found_error"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if not self._authed():
            return self.send_json(401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})
        effort = body.get("reasoning_effort")
        if effort is None and isinstance(body.get("reasoning"), dict):
            effort = body["reasoning"].get("effort")
        imgs = 0
        for m in body.get("messages") or []:
            if isinstance(m.get("content"), list):
                imgs += sum(1 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
        if not body.get("stream"):
            return self.send_json(200, {
                "object": "chat.completion",
                "model": body.get("model"),
                "effort": effort,
                "images": imgs,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in (
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "o"}}]},
            {"choices": [{"index": 0, "delta": {"content": "k"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        ):
            data = b"data: " + json.dumps(event).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.write(b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
