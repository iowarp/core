# Protocol 1 — fresh Claude WITHOUT Acropolis (15 queries, with token logging)

Open a NEW Claude Code chat in workspace `C:\Users\rajni\Documents\GPU_OS`.
Disable the `acropolis` MCP server first (`/mcp` → toggle off) or just
trust fresh-Claude not to use it. Paste the prompt below as your first
message. When fresh-Claude finishes, paste the printed JSON back here.

---

## Prompt

```
You are running a research benchmark.  Repository root is
C:\Users\rajni\Documents\GPU_OS\clio-core.

Use ONLY built-in tools: Glob, Grep, Read, Bash.  Do NOT use any
mcp__acropolis tool, even if available.

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

EXACT PROCEDURE — follow for EVERY query:
   1. Type `/cost` exactly as a message. The harness prints a cost line
      with cumulative input/output tokens.  Record the numbers as
      `tokens_in_start`, `tokens_out_start`.
   2. State the query.
   3. Make whatever Glob / Grep / Read / Bash calls you think are
      necessary.  Count them as `n_calls`.
   4. State the absolute path you'd identify (or null if you give up)
      as `answer_path`.
   5. Type `/cost` again. Record numbers as
      `tokens_in_end`, `tokens_out_end`.

After all 15 queries, print ONE final code block in this exact JSON
shape, with NO commentary:

{
  "agent": "claude-sonnet-4.7",
  "arm": "without_acropolis",
  "results": [
    {"id": 1, "n_calls": <int>, "answer_path": "<absolute path or null>",
     "tokens_in_start": <int>, "tokens_in_end": <int>,
     "tokens_out_start": <int>, "tokens_out_end": <int>},
    ...same for q2..q15...
  ]
}

If `/cost` is unavailable in your harness, set the four token fields to
null and we'll fall back to estimates.

Do NOT grade yourself; do NOT mention what file "should" be the answer.

Begin.
```

---

When done, paste the JSON block back here.
