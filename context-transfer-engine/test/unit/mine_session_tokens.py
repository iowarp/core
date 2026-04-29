#!/usr/bin/env python3
r"""Mine real billed tokens from Claude Code session JSONL transcripts.

Each assistant turn in a Claude Code session JSONL has a `usage` block:
{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}.

We attribute each turn's usage to a query bucket by inspecting the tool_use
inputs in that turn:

  - arm 2 (with Acropolis): query string is literally in the tool_use input
    (`mcp__acropolis__semantic_query` has `query`, `backend`, `level`). So we
    can split per-query AND per-(query,backend,level).

  - arm 1 (no Acropolis): no explicit query field — we infer the active
    query from the tool_use args (Glob patterns, Grep patterns, Read paths)
    by matching against per-query topic substrings. Best-effort but reliable
    for these 15 queries since each one is topically distinct.

Outputs:
  arm1_tokens.json  -- per-query totals (best-effort attribution)
  arm2_tokens.json  -- per-query totals AND per-config breakdown
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ARM1_JSONL = r"C:\Users\rajni\.claude\projects\c--Users-rajni-Documents-GPU-OS\899b03bb-c890-4326-b5c0-78cf11ca2a4b.jsonl"
ARM2_JSONL = r"C:\Users\rajni\.claude\projects\c--Users-rajni-Documents-GPU-OS\5d1bc0b8-914e-4753-aba1-d80e7a395eff.jsonl"
OUT_DIR = Path(__file__).parent

# Topic substrings used to attribute arm-1 tool calls to a query id.
# Each list is searched against tool_use input JSON (pattern, query, file path).
# Order matters: more specific patterns first (q11/q12/q13 etc. before q1).
ARM1_TOPICS = [
    (11, ["bm25"]),
    (12, ["rrf", "reciprocal", "hybrid"]),
    (13, ["depth_controller", "depth-controller", "xattr"]),
    (14, ["embedding_client", "embedding-client", "openai-compat", "openai_compat"]),
    (15, ["indexing_depth_config", "test_indexing_depth", "indexing-depth"]),
    (1,  ["qdrant"]),
    (2,  ["elasticsearch", "elastic"]),
    (3,  ["neo4j"]),
    (4,  ["hdf5"]),
    (5,  ["set_depth", "set-depth", "depth_cli", "acropolis_depth"]),
    (6,  ["gpu_submission", "submission_gpu", "test_gpu"]),
    (7,  ["ggml", "gpuvmm", "weights", "iowarp_backend"]),
    (8,  ["kvcache_manager", "kv-cache", "kvcache"]),
    (9,  ["e2e_gpu", "run_e2e", "kv cache restore", "restore"]),
    (10, ["bench_repo_scan", "bench_repo", "agent loop"]),
]

# arm2 query strings (truncated to first 30 chars for matching) -> query id
ARM2_QUERY_PREFIX_TO_ID = {
    "Find the Qdrant vector backend":           1,
    "Locate the Elasticsearch full-text":       2,
    "Show me the Neo4j knowledge graph":        3,
    "Where is the HDF5 metadata extractor":     4,
    "Find the CLI tool that sets indexing":     5,
    "Locate the unit test that exercises GPU":  6,
    "Find the code that streams per-layer":     7,
    "Where is the KV-cache manager":            8,
    "Find the end-to-end script that tests":    9,
    "Locate the benchmark that scans a repo":  10,
    "Find the file implementing BM25":         11,
    "Where is the hybrid retrieval that uses": 12,
    "Locate the depth controller that resolves": 13,
    "Find the OpenAI-compatible HTTP":         14,
    "Where is the unit test validating":       15,
}


def attribute_arm1(tool_inputs_json):
    """Return query_id (1..15) or None if not attributable."""
    haystack = tool_inputs_json.lower()
    for qid, keywords in ARM1_TOPICS:
        for kw in keywords:
            if kw.lower() in haystack:
                return qid
    return None


def attribute_arm2(tool_input):
    """Return query_id from sem_q tool input."""
    q = tool_input.get("query", "")
    for prefix, qid in ARM2_QUERY_PREFIX_TO_ID.items():
        if q.startswith(prefix):
            return qid
    return None


def empty_bucket():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "n_assistant_turns": 0,
        "n_tool_calls": 0,
    }


def add_usage(bucket, usage):
    bucket["input_tokens"]                += usage.get("input_tokens", 0) or 0
    bucket["output_tokens"]               += usage.get("output_tokens", 0) or 0
    bucket["cache_read_input_tokens"]     += usage.get("cache_read_input_tokens", 0) or 0
    bucket["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0) or 0
    bucket["n_assistant_turns"]           += 1


def billable_equiv(b):
    """Approx Anthropic-style billable-equivalent tokens (input + output +
    0.1 * cache_read + 1.25 * cache_creation 1h). Order-of-magnitude only."""
    return (b["input_tokens"] + b["output_tokens"]
            + 0.1 * b["cache_read_input_tokens"]
            + 1.25 * b["cache_creation_input_tokens"])


def mine_arm1():
    by_query = defaultdict(empty_bucket)
    unattributed = empty_bucket()
    total = empty_bucket()
    last_qid = None

    for line in open(ARM1_JSONL, encoding="utf-8"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") != "assistant":
            continue
        m = o.get("message", {})
        usage = m.get("usage", {})
        add_usage(total, usage)

        content = m.get("content", [])
        if not isinstance(content, list):
            content = []

        # Try to attribute this turn by examining its tool_use blocks
        qid_for_turn = None
        n_calls_in_turn = 0
        for c in content:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                n_calls_in_turn += 1
                inp = json.dumps(c.get("input", {}))
                qid = attribute_arm1(inp)
                if qid is not None:
                    qid_for_turn = qid
                    break

        # If no tool calls in this turn (pure-text reply, like a planning step
        # or final summary), inherit last_qid for continuity. Same for
        # ambiguous tool calls.
        if qid_for_turn is None:
            qid_for_turn = last_qid

        if qid_for_turn is None:
            add_usage(unattributed, usage)
            unattributed["n_tool_calls"] += n_calls_in_turn
        else:
            add_usage(by_query[qid_for_turn], usage)
            by_query[qid_for_turn]["n_tool_calls"] += n_calls_in_turn
            last_qid = qid_for_turn

    out = {
        "agent": "claude-sonnet-4.7",
        "arm": "without_acropolis",
        "source_jsonl": ARM1_JSONL,
        "total": total,
        "total_billable_equiv": int(billable_equiv(total)),
        "unattributed": unattributed,
        "per_query": [
            {"id": q, **by_query[q],
             "billable_equiv": int(billable_equiv(by_query[q]))}
            for q in sorted(by_query.keys())
        ],
    }
    out_path = OUT_DIR / "arm1_tokens.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return out


def mine_arm2():
    by_query = defaultdict(empty_bucket)
    by_config = defaultdict(empty_bucket)   # (qid, backend, level) -> bucket
    unattributed = empty_bucket()
    total = empty_bucket()

    for line in open(ARM2_JSONL, encoding="utf-8"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") != "assistant":
            continue
        m = o.get("message", {})
        usage = m.get("usage", {})
        add_usage(total, usage)

        content = m.get("content", [])
        if not isinstance(content, list):
            content = []

        # Collect sem_q calls in this turn
        sem_q_calls = []
        other_calls = 0
        for c in content:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                if c.get("name") == "mcp__acropolis__semantic_query":
                    sem_q_calls.append(c.get("input", {}))
                else:
                    other_calls += 1

        if not sem_q_calls:
            # No sem_q in this turn (e.g. TodoWrite, planning text).
            # Attribute to most-recent query if known, else unattributed.
            add_usage(unattributed, usage)
            unattributed["n_tool_calls"] += other_calls
            continue

        # Distribute usage equally across the sem_q calls in this turn.
        n = len(sem_q_calls)
        share = {k: (usage.get(k, 0) or 0) / n for k in
                 ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens")}

        for inp in sem_q_calls:
            qid = attribute_arm2(inp)
            backend = inp.get("backend", "?")
            level = inp.get("level", -1)

            for bucket in (by_query[qid] if qid else unattributed,
                           by_config[(qid, backend, level)]):
                bucket["input_tokens"]                += share["input_tokens"]
                bucket["output_tokens"]               += share["output_tokens"]
                bucket["cache_read_input_tokens"]     += share["cache_read_input_tokens"]
                bucket["cache_creation_input_tokens"] += share["cache_creation_input_tokens"]
                bucket["n_tool_calls"]                += 1
        # Count one assistant turn per turn (not per sem_q call)
        if qid:
            by_query[qid]["n_assistant_turns"] += 1
        else:
            unattributed["n_assistant_turns"] += 1

    out = {
        "agent": "claude-sonnet-4.7",
        "arm": "with_acropolis",
        "source_jsonl": ARM2_JSONL,
        "total": total,
        "total_billable_equiv": int(billable_equiv(total)),
        "unattributed": unattributed,
        "per_query": [
            {"id": q, **{k: int(v) if isinstance(v, float) else v
                         for k, v in by_query[q].items()},
             "billable_equiv": int(billable_equiv(by_query[q]))}
            for q in sorted(by_query.keys())
        ],
        "per_config": [
            {"id": qid, "backend": b, "level": lv,
             **{k: int(v) if isinstance(v, float) else v
                for k, v in by_config[(qid, b, lv)].items()},
             "billable_equiv": int(billable_equiv(by_config[(qid, b, lv)]))}
            for (qid, b, lv) in sorted(by_config.keys(),
                                       key=lambda x: (x[0] or 99, x[1], x[2]))
        ],
    }
    out_path = OUT_DIR / "arm2_tokens.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return out


def fmt(b):
    return (f"in={b['input_tokens']:>8,}  out={b['output_tokens']:>8,}  "
            f"cache_r={b['cache_read_input_tokens']:>10,}  "
            f"cache_c={b['cache_creation_input_tokens']:>9,}  "
            f"calls={b.get('n_tool_calls', 0):>3}  "
            f"~be={int(billable_equiv(b)):>10,}")


if __name__ == "__main__":
    print("=" * 95)
    print("ARM 1 (no Acropolis)")
    print("=" * 95)
    a1 = mine_arm1()
    print(f"\nTOTAL              : {fmt(a1['total'])}")
    print(f"UNATTRIBUTED       : {fmt(a1['unattributed'])}")
    print(f"\nper-query (~be = ~billable-equivalent tokens):")
    for r in a1["per_query"]:
        print(f"  q{r['id']:>2}: {fmt(r)}")

    print("\n" + "=" * 95)
    print("ARM 2 (with Acropolis sweep)")
    print("=" * 95)
    a2 = mine_arm2()
    print(f"\nTOTAL              : {fmt(a2['total'])}")
    print(f"UNATTRIBUTED       : {fmt(a2['unattributed'])}")
    print(f"\nper-query (across all 21 configs):")
    for r in a2["per_query"]:
        print(f"  q{r['id']:>2}: {fmt(r)}")

    print(f"\nper-config sample (first 5):")
    for r in a2["per_config"][:5]:
        print(f"  q{r['id']:>2} {r['backend']:<18} L{r['level']}: {fmt(r)}")
