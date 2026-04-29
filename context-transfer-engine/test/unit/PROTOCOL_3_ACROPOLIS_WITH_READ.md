# Protocol 3 — Acropolis with verification (re-run of arm 2, fair comparison)

**Why:** the original arm 2 protocol forbade `Read`, forcing Acropolis to
commit blindly on a single semantic_query result. Baseline (arm 1) could
freely Read candidates to verify. This re-run lets Acropolis verify too,
producing a head-to-head comparison.

## Setup

1. Open a NEW Claude Code chat in workspace `C:\Users\rajni\Documents\GPU_OS`.
2. Make sure the `acropolis` MCP server is enabled (`/mcp` → connected).
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
  1. Call mcp__acropolis__semantic_query with backend="elasticsearch-vec",
     level=2, k=5.
  2. Look at the top-3 returned hits.
  3. If one hit clearly matches the query intent based on path + summary,
     commit that path as the answer. Set n_reads=0.
  4. If no single hit is obviously correct, you may Read up to TWO
     candidate files to verify. Pick the most plausible candidates.
     After Reading, commit the absolute path of whichever file actually
     answers the query.
  5. If even after Reads no candidate matches, you may make ONE additional
     mcp__acropolis__semantic_query with a rephrased query. Then repeat
     steps 2-4.
  6. If still no answer found after that, set answer_path=null.

For each query, record:
  - id (1..15)
  - n_sem_q_calls (1 or 2)
  - n_reads (0, 1, or 2)
  - answer_path (absolute path or null)

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
  "arm": "with_acropolis_and_read",
  "config": {"backend": "elasticsearch-vec", "level": 2, "k": 5},
  "results": [
    {"id": 1,  "n_sem_q_calls": <int>, "n_reads": <int>, "answer_path": "<path or null>"},
    {"id": 2,  "n_sem_q_calls": <int>, "n_reads": <int>, "answer_path": "<path or null>"},
    ...same for q3..q15...
  ]
}

Do NOT grade yourself; do NOT mention what file "should" be the answer.

Begin.
```

---

## When done

Paste the JSON back here. We'll grade against the same EXPECTED file set as
arms 1 and 2, plus mine the session JSONL for real billed tokens. Then we
can render an updated fig that compares:
  - arm 1 (baseline, Glob/Grep/Read)
  - arm 2 (Acropolis 1 sem_q, no Read)         ← old
  - arm 3 (Acropolis 1 sem_q + up to 2 Reads)  ← new, fair comparison
