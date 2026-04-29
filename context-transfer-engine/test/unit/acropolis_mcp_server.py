#!/usr/bin/env python3
"""MCP server exposing the Acropolis semantic_query tool.

Supports the full Acropolis matrix: 4 backends (bm25, elasticsearch, qdrant,
neo4j) × 3 indexing depths (0=name, 1=metadata, 2=content+LLM-summary).

The point of this server is to let Claude Code (or any MCP client) drive the
*same* model (Claude Sonnet) with and without Acropolis retrieval, isolating
retrieval as the only changed variable for the paper's A/B comparison.

Tool exposed:
    semantic_query(query: str,
                   k: int = 5,
                   backend: "bm25"|"elasticsearch-kw"|"elasticsearch-vec"|
                            "elasticsearch-rrf"|"qdrant"|"neo4j-kw"|"neo4j-rrf",
                   level: 0|1|2)
        Returns ranked list of {path, score, summary?}

Architecture:
    bm25            → reimplemented in Python over the path/summary set
                      (the C++ in-memory index is not persistent).
    elasticsearch*  → HTTPS to the matrix index `acropolis_matrix_l<level>`.
    qdrant          → HTTPS to the matrix collection `acropolis_matrix_L<level>`.
    neo4j*          → Cypher to the matrix Dataset nodes (one DB, level-tagged
                      by the text payload).

For L0 we walk the live filesystem to get paths.
For L1 we walk + tag with extension (cheap format hint).
For L2 we read the L2 summary cache the C++ bench produced
       (`/tmp/acropolis_summary_cache_<hash>.json` or copied to host).

Embeddings (for vector backends and ES-vec/rrf): Ollama nomic-embed-text.

Dependencies:
    pip install mcp
"""

import asyncio
import glob
import json
import math
import os
import re
import sys
import urllib.request
import urllib.error
from collections import Counter
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# --------------------------------------------------------------------------- #
# Configuration (env-overridable)
# --------------------------------------------------------------------------- #

REPO_ROOT      = os.environ.get("ACROPOLIS_REPO",        "/workspace")
CACHE_DIR      = os.environ.get("ACROPOLIS_CACHE_DIR",   "/tmp")
ES_BASE        = os.environ.get("ACROPOLIS_ES_URL",      "http://localhost:9200")
QDRANT_BASE    = os.environ.get("ACROPOLIS_QDRANT_URL",  "http://localhost:6333")
NEO4J_BASE     = os.environ.get("ACROPOLIS_NEO4J_URL",   "http://localhost:7474")
OLLAMA_BASE    = os.environ.get("ACROPOLIS_OLLAMA_URL",  "http://localhost:11434")
EMBED_MODEL    = os.environ.get("ACROPOLIS_EMBED_MODEL", "nomic-embed-text")
ES_INDEX_PFX   = os.environ.get("ACROPOLIS_ES_INDEX",    "acropolis_matrix")
QDRANT_COL_PFX = os.environ.get("ACROPOLIS_QDRANT_COL",  "acropolis_matrix")

SKIP_DIRS = {".git", "build", "node_modules", "__pycache__", ".cache"}
TOKEN_RE  = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# Helpers — filesystem walk + summary cache
# --------------------------------------------------------------------------- #

def walk_repo(root: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs
                   if x not in SKIP_DIRS and not x.startswith("build_")]
        for f in files:
            yield os.path.join(d, f)


def load_summary_cache(cache_dir: str) -> dict[str, str]:
    files = sorted(glob.glob(os.path.join(
        cache_dir, "acropolis_summary_cache_*.json")),
        key=os.path.getmtime, reverse=True)
    if not files:
        return {}
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


def docs_for_level(level: int) -> list[tuple[str, str]]:
    """(path, indexed_text) pairs for the requested depth level."""
    if level == 2:
        cache = load_summary_cache(CACHE_DIR)
        if cache:
            return [(p, f"{p}\n{s}") for p, s in cache.items()]
    paths = list(walk_repo(REPO_ROOT))
    if level == 0:
        return [(p, p) for p in paths]
    # L1 — path + extension hint
    return [(p, f"{p} {os.path.splitext(p)[1].lstrip('.')}") for p in paths]


# --------------------------------------------------------------------------- #
# BM25 in pure Python (used for `bm25` backend across all levels)
# --------------------------------------------------------------------------- #

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tok = [tokenize(t) for _, t in docs]
        self.dl = [len(t) for t in self.tok]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0
        self.df = Counter()
        for toks in self.tok:
            for term in set(toks):
                self.df[term] += 1
        self.tf = [Counter(t) for t in self.tok]
        self.N = len(docs)

    def score(self, q_terms: list[str], i: int) -> float:
        s = 0.0
        if self.dl[i] == 0:
            return 0.0
        for q in q_terms:
            if q not in self.df:
                continue
            idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
            tf = self.tf[i][q]
            denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
            s += idf * (tf * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        q_terms = tokenize(query)
        scored = sorted(
            ((self.score(q_terms, i), i) for i in range(self.N)),
            reverse=True)
        out = []
        for s, i in scored[:k]:
            if s <= 0:
                continue
            path, text = self.docs[i]
            hit = {"path": path, "score": round(s, 4)}
            if "\n" in text:
                hit["summary"] = text.split("\n", 1)[1][:1500]
            out.append(hit)
        return out


_bm25_cache: dict[int, BM25] = {}


def bm25_search(query: str, k: int, level: int) -> list[dict[str, Any]]:
    if level not in _bm25_cache:
        _bm25_cache[level] = BM25(docs_for_level(level))
    return _bm25_cache[level].search(query, k)


# --------------------------------------------------------------------------- #
# Embedding helper (for vector backends)
# --------------------------------------------------------------------------- #

def embed(text: str) -> list[float] | None:
    body = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        OLLAMA_BASE + "/v1/embeddings",
        data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
        return d["data"][0]["embedding"]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Elasticsearch (kw / vec / rrf)
# --------------------------------------------------------------------------- #

def http_post_json(url: str, body: dict, timeout: int = 30) -> dict | None:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"_error": str(e)}


def es_search(query: str, k: int, level: int, mode: str) -> list[dict[str, Any]]:
    index = f"{ES_INDEX_PFX}_l{level}"
    cache = load_summary_cache(CACHE_DIR) if level == 2 else {}
    hits_kw, hits_vec = [], []

    if mode in ("kw", "rrf"):
        body = {"query": {"match": {"text": query}},
                "size": max(k * 3, 10),
                "_source": ["tag_major", "tag_minor", "text"]}
        d = http_post_json(f"{ES_BASE}/{index}/_search", body) or {}
        for h in d.get("hits", {}).get("hits", []):
            txt = h["_source"].get("text", "")
            hits_kw.append({"path": txt.split("\n", 1)[0],
                            "score": h["_score"]})

    if mode in ("vec", "rrf"):
        qv = embed(query)
        if qv:
            body = {"knn": {"field": "embedding", "query_vector": qv,
                             "k": max(k * 3, 10),
                             "num_candidates": max(k * 10, 50)},
                    "size": max(k * 3, 10),
                    "_source": ["tag_major", "tag_minor", "text"]}
            d = http_post_json(f"{ES_BASE}/{index}/_search", body) or {}
            for h in d.get("hits", {}).get("hits", []):
                txt = h["_source"].get("text", "")
                hits_vec.append({"path": txt.split("\n", 1)[0],
                                 "score": h["_score"]})

    # RRF fuse if both, else use whichever ran.
    if mode == "rrf":
        rrf: dict[str, float] = {}
        for lst in (hits_kw, hits_vec):
            for i, h in enumerate(lst):
                rrf[h["path"]] = rrf.get(h["path"], 0.0) + 1.0 / (60 + i + 1)
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        out = [{"path": p, "score": round(s, 4)} for p, s in merged[:k]]
    else:
        out = (hits_kw if mode == "kw" else hits_vec)[:k]

    if level == 2:
        for h in out:
            if h["path"] in cache:
                h["summary"] = cache[h["path"]][:1500]
    return out


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #

def qdrant_search(query: str, k: int, level: int) -> list[dict[str, Any]]:
    collection = f"{QDRANT_COL_PFX}_L{level}"
    qv = embed(query)
    if not qv:
        return []
    body = {"vector": qv, "limit": k, "with_payload": True}
    d = http_post_json(f"{QDRANT_BASE}/collections/{collection}/points/search",
                       body) or {}
    out = []
    cache = load_summary_cache(CACHE_DIR) if level == 2 else {}
    for r in d.get("result", []):
        path = r.get("payload", {}).get("path") or r.get("payload", {}).get("text", "")
        if "\n" in path:
            path = path.split("\n", 1)[0]
        h = {"path": path, "score": round(r.get("score", 0.0), 4)}
        if level == 2 and path in cache:
            h["summary"] = cache[path][:1500]
        out.append(h)
    return out


# --------------------------------------------------------------------------- #
# Neo4j (kw / rrf)
# --------------------------------------------------------------------------- #

def neo4j_query(cypher: str, params: dict | None = None) -> dict | None:
    body = {"statements": [{"statement": cypher,
                              "parameters": params or {}}]}
    return http_post_json(f"{NEO4J_BASE}/db/neo4j/tx/commit", body)


def neo4j_search(query: str, k: int, level: int, mode: str) -> list[dict[str, Any]]:
    # Sanitize for Lucene: keep alphanumeric + space.
    lq = re.sub(r"[^A-Za-z0-9_\s]", " ", query)

    ft_rows: list[tuple[str, float]] = []
    if mode in ("kw", "rrf"):
        d = neo4j_query(
            "CALL db.index.fulltext.queryNodes('cte_kg_idx', $q) "
            "YIELD node, score WHERE node:Dataset "
            "RETURN node.text AS text, score LIMIT toInteger($k)",
            {"q": lq, "k": k * 3}) or {}
        for r in d.get("results", []):
            for row in r.get("data", []):
                txt, sc = row["row"][0] or "", row["row"][1]
                ft_rows.append((txt.split("\n", 1)[0], float(sc)))

    vec_rows: list[tuple[str, float]] = []
    if mode == "rrf":
        qv = embed(query)
        if qv:
            d = neo4j_query(
                "CALL db.index.vector.queryNodes('cte_kg_vec', toInteger($k), "
                "$vec) YIELD node, score WHERE node:Dataset "
                "RETURN node.text AS text, score",
                {"k": k * 3, "vec": qv}) or {}
            for r in d.get("results", []):
                for row in r.get("data", []):
                    txt, sc = row["row"][0] or "", row["row"][1]
                    vec_rows.append((txt.split("\n", 1)[0], float(sc)))

    if mode == "rrf":
        rrf: dict[str, float] = {}
        for lst in (ft_rows, vec_rows):
            for i, (p, _) in enumerate(lst):
                rrf[p] = rrf.get(p, 0.0) + 1.0 / (60 + i + 1)
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        out = [{"path": p, "score": round(s, 4)} for p, s in merged[:k]]
    else:
        out = [{"path": p, "score": round(s, 4)} for p, s in ft_rows[:k]]

    cache = load_summary_cache(CACHE_DIR) if level == 2 else {}
    for h in out:
        if h["path"] in cache:
            h["summary"] = cache[h["path"]][:1500]
    return out


# --------------------------------------------------------------------------- #
# Top-level dispatch
# --------------------------------------------------------------------------- #

def semantic_query(query: str, k: int, backend: str,
                    level: int) -> list[dict[str, Any]]:
    if backend == "bm25":
        return bm25_search(query, k, level)
    if backend == "elasticsearch-kw":
        return es_search(query, k, level, "kw")
    if backend == "elasticsearch-vec":
        return es_search(query, k, level, "vec")
    if backend == "elasticsearch-rrf":
        return es_search(query, k, level, "rrf")
    if backend == "qdrant":
        return qdrant_search(query, k, level)
    if backend == "neo4j-kw":
        return neo4j_search(query, k, level, "kw")
    if backend == "neo4j-rrf":
        return neo4j_search(query, k, level, "rrf")
    return [{"error": f"unknown backend: {backend}"}]


# --------------------------------------------------------------------------- #
# MCP server wiring
# --------------------------------------------------------------------------- #

server = Server("acropolis-search")

BACKENDS = ["bm25", "elasticsearch-kw", "elasticsearch-vec",
            "elasticsearch-rrf", "qdrant", "neo4j-kw", "neo4j-rrf"]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="semantic_query",
            description=(
                "Acropolis semantic search over a code repo. Returns up to k "
                "ranked hits with {path, score, optional summary}. Pick a "
                "backend (bm25 / elasticsearch-kw|vec|rrf / qdrant / "
                "neo4j-kw|rrf) and an indexing depth (0=name, 1=metadata, "
                "2=content+LLM-summary). Default backend=bm25, level=2 give "
                "the best precision/cost tradeoff."),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":   {"type": "string"},
                    "k":       {"type": "integer", "default": 5},
                    "backend": {"type": "string", "enum": BACKENDS,
                                "default": "bm25"},
                    "level":   {"type": "integer", "enum": [0, 1, 2],
                                "default": 2},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_acropolis_configs",
            description="Returns the list of supported Acropolis backends and "
                        "depth levels.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    if name == "semantic_query":
        q       = args.get("query", "")
        k       = int(args.get("k", 5))
        backend = args.get("backend", "bm25")
        level   = int(args.get("level", 2))
        results = semantic_query(q, k, backend, level)
        return [TextContent(type="text", text=json.dumps(results, indent=2))]
    if name == "list_acropolis_configs":
        return [TextContent(type="text", text=json.dumps(
            {"backends": BACKENDS, "levels": [0, 1, 2]}, indent=2))]
    return [TextContent(type="text", text=f"unknown tool: {name}")]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write,
                          server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
