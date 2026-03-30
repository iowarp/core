#!/usr/bin/env python3
"""Lightweight OpenAI-compatible embedding server using sentence-transformers.

Usage:
    python3 embedding_server.py [--port 8090] [--model all-MiniLM-L6-v2]

Provides /v1/embeddings endpoint compatible with Qdrant backend.
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import SentenceTransformer

model = None
model_name = ""


class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/embeddings":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            text_input = body.get("input", "")
            if isinstance(text_input, list):
                texts = text_input
            else:
                texts = [text_input]

            embeddings = model.encode(texts).tolist()

            response = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": emb}
                    for i, emb in enumerate(embeddings)
                ],
                "model": model_name,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


def main():
    global model, model_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    model_name = args.model
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded: dim={dim}")
    print(f"Serving on http://0.0.0.0:{args.port}/v1/embeddings")

    server = HTTPServer(("0.0.0.0", args.port), EmbeddingHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
