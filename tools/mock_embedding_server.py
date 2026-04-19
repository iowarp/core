"""
Tiny mock /v1/embeddings server for Acropolis E2E testing.

Returns a deterministic 8-dim embedding derived from a hash of the input text,
following the OpenAI-compatible response schema expected by EmbeddingClient.
"""

import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

DIM = 8


def embed(text: str):
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Map 8 bytes to 8 floats in [-1, 1], then L2-normalize.
    raw = [(b - 128) / 128.0 for b in h[:DIM]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        inp = body.get("input", "")
        if isinstance(inp, list):
            inp = inp[0] if inp else ""
        vec = embed(inp)
        out = {"data": [{"embedding": vec, "index": 0}], "model": body.get("model", "mock")}
        payload = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 9999), Handler).serve_forever()
