# Acropolis MCP server — same-model A/B testing

This MCP server exposes the entire Acropolis search system to Claude Code (or
any MCP client). One tool: `semantic_query(query, k, backend, level)` — works
across all 7 backend/mode combinations × 3 indexing depths = 21 configurations.

## Why

Lets you measure **the same Claude model with vs. without Acropolis** on the
same queries, isolating retrieval as the only variable. Run a sample of the 15
bench queries through Claude Code twice:

1. **Without Acropolis**: don't enable this MCP server. Claude only has its
   built-in tools (`Read`, `Grep`, `Glob`, `Bash`). Record per-query: tool-call
   count, tokens, whether the answer names the expected file.
2. **With Acropolis**: enable this MCP server. Claude has `semantic_query` in
   addition. Same queries, same model, same grader. Record same metrics.

The token + tool-call delta is attributable purely to retrieval.

## Prerequisites

```powershell
pip install mcp
```

The Acropolis backends should be running and populated. The matrix bench
populates them. The L2 summary cache file
(`acropolis_summary_cache_<hash>.json`) needs to be on the host where this
MCP server runs — copy it out of the docker container if you ran the matrix
in docker:

```powershell
docker cp iowarp-llm-test:/tmp/acropolis_summary_cache_<hash>.json C:\temp\
```

## Configuration

Add to your Claude Code MCP config (`%APPDATA%\Claude\claude_desktop_config.json`
on Windows, or use Claude Code's `claude mcp add` CLI):

```json
{
  "mcpServers": {
    "acropolis": {
      "command": "python",
      "args": [
        "C:\\Users\\rajni\\Documents\\GPU_OS\\clio-core\\context-transfer-engine\\test\\unit\\acropolis_mcp_server.py"
      ],
      "env": {
        "ACROPOLIS_REPO":       "C:\\Users\\rajni\\Documents\\GPU_OS\\clio-core",
        "ACROPOLIS_CACHE_DIR":  "C:\\temp",
        "ACROPOLIS_ES_URL":     "http://localhost:9200",
        "ACROPOLIS_QDRANT_URL": "http://localhost:6333",
        "ACROPOLIS_NEO4J_URL":  "http://localhost:7474",
        "ACROPOLIS_OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

Restart Claude Code. The `semantic_query` tool should appear when you start a
new session and Claude will use it for "find file" queries.

## A/B protocol for the paper

For each of the 15 bench queries (see `bench_repo_scan.cc:437`), run both
configurations and record:

| metric | how to capture |
|---|---|
| answer correctness | does the response include the expected substring? |
| total tool calls   | count `semantic_query` + `Grep` + `Glob` + `Read` etc. |
| total tokens       | from Claude Code's session metadata |
| wall-clock time    | from session timestamps |

The Acropolis-on configuration should show:
- ~1 tool call per query (just `semantic_query`)
- Drastically fewer tokens (no need to read full files)
- Same or higher correctness

The Acropolis-off configuration should show:
- 3-10+ tool calls per query (Glob → Read → Read → Read…)
- Many more tokens (file contents land in context)
- Possibly lower correctness (Claude can guess wrong on what to grep for)

## Sweeping backends/levels

The tool accepts `backend` and `level` parameters explicitly. Either let Claude
pick the defaults (`bm25`, level 2 — the best combo from our matrix) or
explicitly tell it in the prompt:

> "Use semantic_query with backend=elasticsearch-rrf, level=2."
