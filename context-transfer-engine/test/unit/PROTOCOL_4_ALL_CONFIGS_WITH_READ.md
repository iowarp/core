# Protocol 4 — Acropolis full 21-config sweep WITH Read allowed

**Why:** Arm 2 (315 sem_q calls, no Read) and Arm 3 (1 config + Read) gave us
two extremes. This middle ground tests "what if agents had both Acropolis
AND verification" across every backend/level so we can see which combinations
benefit most from Read.

## Setup

1. Open a NEW Claude Code chat in workspace `C:\Users\rajni\Documents\GPU_OS`.
2. Acropolis MCP enabled (`/mcp` → connected). Docker Desktop running so the
   Elasticsearch / Qdrant / Neo4j containers respond.
3. Paste the prompt below as your first message.
4. When done, paste the printed JSON back into this chat.

---

## Prompt to paste

```
You are running a research benchmark on the Acropolis semantic-search
system. You may use ONLY these tools:
  - mcp__acropolis__semantic_query
  - Read

Do NOT use Glob, Grep, Bash, or any other tool.

For EACH query below, follow this exact procedure:

  STEP 1 — Run the full 21-config sweep.
  For EACH (backend × level) combination call mcp__acropolis__semantic_query
  with that backend, level, and k=3.
    backends: ["bm25", "elasticsearch-kw", "elasticsearch-vec",
               "elasticsearch-rrf", "qdrant", "neo4j-kw", "neo4j-rrf"]
    levels:   [0, 1, 2]
  That is 21 sem_q calls per query.

  STEP 2 — Collect candidate paths.
  Aggregate all unique file paths returned across the 21 sem_q calls into a
  candidate set.

  STEP 3 — Verify with Read (optional, capped).
  You may Read up to 3 unique candidate files per query to verify their
  contents match the query intent. Only Read if you genuinely need to
  disambiguate; if one candidate is clearly correct from path + summary
  alone, do not Read.

  STEP 4 — Commit per-config answers.
  For each of the 21 configs, decide which file best answers the query
  given that config's top-3 hits AND any Read knowledge you gained in
  step 3. Different configs may pick different paths.

For each query record:
  - id (1..15)
  - n_reads (0..3) — total Reads across all 21 configs for this query
  - configs: list of 21 entries {backend, level, answer_path}

QUERIES:
  q1:  "Find the Qdrant vector backend implementation file and give its path"
  q2:  "Locate the Elasticsearch full-text search backend"
  q3:  "Show me the Neo4j knowledge graph backend"
  q4:  "Where is the HDF5 metadata extractor?"
  q5:  "Find the CLI tool that sets indexing depth on files"
  q6:  "Locate the unit test that exercises GPU submission on an actual GPU"
  q7:  "Find the code that streams per-layer GGML weights through GpuVMM"
  q8:  "Where is the KV-cache manager that interacts with llama.cpp?"
  q9:  "Find the end-to-end script that tests KV cache restore on GPU"
  q10: "Locate the benchmark that scans a repo with an LLM agent loop"
  q11: "Find the file implementing BM25 scoring with distributed IDF synchronization"
  q12: "Where is the hybrid retrieval that uses reciprocal rank fusion?"
  q13: "Locate the depth controller that resolves xattr inheritance across directories"
  q14: "Find the OpenAI-compatible HTTP embeddings client shared across backends"
  q15: "Where is the unit test validating indexing-depth configuration parsing?"

After all 15 queries, print ONE final code block in this exact JSON shape,
with NO commentary:

{
  "agent": "claude-sonnet-4.7",
  "arm": "with_acropolis_full_sweep_and_read",
  "results": [
    {
      "id": 1,
      "n_reads": <int>,
      "configs": [
        {"backend": "bm25",              "level": 0, "answer_path": "<path or null>"},
        {"backend": "bm25",              "level": 1, "answer_path": "<path or null>"},
        {"backend": "bm25",              "level": 2, "answer_path": "<path or null>"},
        {"backend": "elasticsearch-kw",  "level": 0, "answer_path": "<path or null>"},
        ... 21 entries total in (backend × level) order ...
      ]
    },
    ...same for q2..q15...
  ]
}

Do NOT grade yourself; do NOT mention what file "should" be the answer.

Begin.
```

---

## When done

Paste the JSON back here. We'll grade it (with the multi-valid-answer fix
for q9 and q12), mine the new session JSONL for real billed tokens, and
render an updated comparison: arm 2 (no Read) vs arm 4 (with Read) per
backend × level cell.

The expected outcome: most configs hold their accuracy or improve slightly;
configs that were missing on disambiguation (e.g., L0 backends that returned
file lists without summaries) gain the most from Read.
