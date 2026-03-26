#!/usr/bin/env python3
"""Minimal OpenAI-compatible inference server using Qwen2.5-0.5B-Instruct.

Runs on CPU. Exposes /v1/chat/completions endpoint that the CTE Summary
Operator can call via CAE_SUMMARY_ENDPOINT.

Usage:
    python3 qwen_server.py [--port 8080]
    export CAE_SUMMARY_ENDPOINT=http://localhost:8080/v1
    export CAE_SUMMARY_MODEL=qwen2.5-0.5b
"""

import argparse
import json
import time
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
)
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if "/chat/completions" not in self.path:
            self.send_error(404)
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 64)

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.01, do_sample=False
            )
        result = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        response = {
            "choices": [{"message": {"role": "assistant", "content": result}}],
            "model": body.get("model", "qwen2.5-0.5b"),
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving on port {args.port}", flush=True)
    server.serve_forever()
