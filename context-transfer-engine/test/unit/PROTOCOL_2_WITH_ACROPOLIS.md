# Protocol 2 — fresh Claude WITH Acropolis sweep (15 queries × 7×3 configs)

Open ANOTHER NEW Claude Code chat in workspace `C:\Users\rajni\Documents\GPU_OS`.
The `acropolis` MCP must be enabled (default).

Paste the prompt below as your first message. When fresh-Claude finishes,
paste the printed JSON back here.

This is a 15-query × 21-config sweep = 315 `semantic_query` calls.
Plus `/cost` checkpoints around each query block.

---

## Prompt

```
You are running a research benchmark on the Acropolis semantic-search
system.  Use ONLY the tool `mcp__acropolis__semantic_query`.  Do NOT
use Glob, Grep, Read, or Bash.

QUERIES (just questions — do not assume any expected file name):
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

CONFIGURATIONS (21 total):
  backends: ["bm25", "elasticsearch-kw", "elasticsearch-vec",
             "elasticsearch-rrf", "qdrant", "neo4j-kw", "neo4j-rrf"]
  levels:   [0, 1, 2]

EXACT PROCEDURE — follow for EVERY query:
  1. Type `/cost` and record `tokens_in_start`, `tokens_out_start`.
  2. For EACH (backend × level) pair (21 of them):
     - Call mcp__acropolis__semantic_query with that query, backend,
       level, and k=3.
     - From the returned hits, decide the answer path based on the
       path + summary alone (no Read).  May be null if no hit looks
       plausible.
     - Record {backend, level, answer_path}.
  3. Type `/cost` and record `tokens_in_end`, `tokens_out_end`.

After all 15 queries, print ONE final code block in this exact JSON
shape, with NO commentary:

{
  "agent": "claude-sonnet-4.7",
  "arm": "with_acropolis",
  "results": [
    {
      "id": 1,
      "tokens_in_start": <int>, "tokens_in_end": <int>,
      "tokens_out_start": <int>, "tokens_out_end": <int>,
      "configs": [
        {"backend": "bm25",              "level": 0, "answer_path": "<path or null>"},
        {"backend": "bm25",              "level": 1, "answer_path": "<path or null>"},
        {"backend": "bm25",              "level": 2, "answer_path": "<path or null>"},
        {"backend": "elasticsearch-kw",  "level": 0, "answer_path": "<path or null>"},
        ... 21 entries total per query, in (backend × level) order above ...
      ]
    },
    ...same for q2..q15...
  ]
}

If `/cost` is unavailable, set the four token fields to null and we
will fall back to size estimates downstream.

Do NOT grade yourself; do NOT mention what file "should" be the answer.

Because this is 315 semantic_query calls, you may batch silently —
print reasoning only if you skip a config (state why).  The JSON at
the end is what matters.

Begin.
```

---

When done, paste the JSON back here.
