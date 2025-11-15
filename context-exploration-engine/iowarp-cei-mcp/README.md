# IOWarp Context Exploration Interface - MCP Server

Model Context Protocol (MCP) server wrapping the IOWarp Context Exploration Engine API, enabling AI assistants to interact with distributed context storage.

## Overview

This MCP server exposes IOWarp's Context Interface through the Model Context Protocol, providing tools for:
- **context_bundle**: Store data into IOWarp contexts
- **context_query**: Query contexts by regex patterns
- **context_retrieve**: Retrieve context data with batching
- **context_destroy**: Manage context lifecycles

## Quick Start

### Prerequisites

1. **Build IOWarp Core with Python bindings:**
   ```bash
   cd /workspace
   cmake --preset=debug -DWRP_CORE_ENABLE_PYTHON=ON
   cmake --build build -j8
   ```

2. **Install Python dependencies:**
   ```bash
   pip install pyyaml
   # Optional: For MCP protocol support
   pip install mcp
   ```

3. **Set environment variables:**
   ```bash
   export PYTHONPATH=/workspace/build/bin:$PYTHONPATH
   export CHI_REPO_PATH=/workspace/build/bin
   export LD_LIBRARY_PATH=/workspace/build/bin:${LD_LIBRARY_PATH}
   ```

### Running Tests

**Comprehensive end-to-end test:**
```bash
python3 test_mcp_end_to_end.py
```

**MCP with runtime initialization:**
```bash
python3 test_mcp_with_runtime.py
```

**Blob data test:**
```bash
python3 test_blob_data.py
```

## MCP Tools

### context_bundle

Bundle and assimilate data into IOWarp contexts.

**Parameters:**
- `bundle` (list[dict]): List of assimilation contexts, each containing:
  - `src` (str): Source URL (e.g., `file::/path/to/file`)
  - `dst` (str): Destination URL (e.g., `iowarp::tag_name`)
  - `format` (str, optional): Data format (default: `binary`)
  - `depends_on` (str, optional): Dependency identifier
  - `range_off` (int, optional): Byte offset in source file
  - `range_size` (int, optional): Number of bytes to read (0=full file)
  - `src_token` (str, optional): Source authentication token
  - `dst_token` (str, optional): Destination authentication token

**Returns:** Success message or error description

**Example:**
```python
from iowarp_cei_mcp import server

bundle = [
    {
        "src": "file::/tmp/data.bin",
        "dst": "iowarp::my_dataset",
        "format": "binary"
    }
]
result = server.context_bundle(bundle)
# Result: "Successfully assimilated 1 context(s)"
```

### context_query

Query IOWarp contexts for blobs matching tag and blob regex patterns.

**Parameters:**
- `tag_re` (str): Tag regex pattern to match
- `blob_re` (str): Blob regex pattern to match
- `max_results` (int, optional): Maximum number of results (0=unlimited, default: 0)

**Returns:** List of matching blob names or message if none found

**Example:**
```python
# Query all blobs in a tag
result = server.context_query("my_dataset", ".*")
# Result: "Found 2 blob(s):\n  - description\n  - chunk_0"

# Query specific pattern
result = server.context_query("experiment_.*", "result_[0-9]+", max_results=100)
```

### context_retrieve

Retrieve both identities and data of objects matching patterns.

**Parameters:**
- `tag_re` (str): Tag regex pattern to match
- `blob_re` (str): Blob regex pattern to match
- `max_results` (int, optional): Maximum number of blobs (default: 1024)
- `max_context_size` (int, optional): Maximum total size in bytes (default: 256MB)
- `batch_size` (int, optional): Concurrent AsyncGetBlob operations (default: 32)

**Returns:** Summary of retrieved data with size information and hex preview

**Example:**
```python
# Retrieve all data from a context
result = server.context_retrieve("my_dataset", ".*")
# Result: "Retrieved 1 packed context(s)\nTotal data size: 1,024 bytes (1.00 KB)\n..."

# Retrieve with limits
result = server.context_retrieve(
    "large_dataset", "chunk_.*",
    max_results=500,
    max_context_size=512 * 1024 * 1024
)
```

### context_destroy

Destroy IOWarp contexts by name.

**Parameters:**
- `context_names` (list[str]): List of context names to destroy

**Returns:** Success message or error description

**Example:**
```python
# Destroy single context
result = server.context_destroy(["my_old_dataset"])
# Result: "Successfully destroyed 1 context(s): my_old_dataset"

# Destroy multiple contexts
result = server.context_destroy(["temp_1", "temp_2", "temp_3"])
# Result: "Successfully destroyed 3 context(s): temp_1, temp_2, temp_3"
```

## API Coverage

The MCP server provides **100% coverage** of all implemented Context Interface methods:

| API Method | MCP Tool | Status |
|------------|----------|--------|
| ContextBundle | context_bundle | ✅ Fully implemented |
| ContextQuery | context_query | ✅ Fully implemented |
| ContextRetrieve | context_retrieve | ✅ Fully implemented |
| ContextDestroy | context_destroy | ✅ Fully implemented |
| ContextSplice | - | ⚠️ Not yet implemented in API |

## Test Results

### End-to-End Test (7/7 PASSED)

```
✓ Runtime initialization
✓ Storage registration
✓ MCP server import
✓ Test data creation
✓ context_bundle - Bundled 3 files
✓ context_query - Individual queries
✓ context_query - Pattern matching
✓ context_retrieve - Data retrieval
✓ context_retrieve - With size limits
✓ context_destroy - Cleanup
✓ Verification of deletion
```

### MCP with Runtime Test (5/5 PASSED)

```
✓ Runtime initialized and ContextInterface created
✓ context_bundle - Created test tag successfully
✓ context_query - Found created tag
✓ context_retrieve - Retrieved tag data
✓ context_destroy - Destroyed tag successfully
```

### Blob Data Test (ALL TESTS PASSED)

- Created 5 different data types (text, binary, JSON, large files, image data)
- Bundled multiple files
- Tested pattern-based queries
- Retrieved data with various limits
- Cleaned up all contexts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client (AI)                      │
└────────────────────┬────────────────────────────────────┘
                     │ MCP Protocol
┌────────────────────▼────────────────────────────────────┐
│              iowarp_cei_mcp/server.py                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  context_bundle | context_query | context_      │   │
│  │  retrieve | context_destroy                     │   │
│  └──────────────────┬──────────────────────────────┘   │
└─────────────────────┼────────────────────────────────────┘
                      │ Python Bindings (wrp_cee)
┌─────────────────────▼────────────────────────────────────┐
│         wrp_cee::ContextInterface (C++)                  │
│  ┌───────────────────────────────────────────────┐      │
│  │  ContextBundle | ContextQuery | ContextRetrieve│      │
│  │  ContextDestroy                                │      │
│  └──────┬────────────────────────┬─────────────────┘      │
└─────────┼────────────────────────┼────────────────────────┘
          │                        │
┌─────────▼────────┐     ┌────────▼─────────┐
│  CAE (Context    │     │  CTE (Context    │
│  Assimilation    │     │  Transfer        │
│  Engine)         │     │  Engine)         │
└──────────────────┘     └──────────────────┘
```

## URL Formats

**Source URLs:**
- `file::/absolute/path/to/file` - Local file system

**Destination URLs:**
- `iowarp::tag_name` - IOWarp context/tag

## Error Handling

All MCP tools return human-readable error messages:

- **Success**: Return code 0 → "Successfully ..."
- **Failure**: Non-zero code → "Error: ... with code: N"
- **Empty input**: "Error: Empty ... provided"
- **Not found**: "No ... found matching ..."

## Direct Usage (Without MCP)

The server functions can be called directly without MCP protocol:

```python
import sys
sys.path.insert(0, "/workspace/build/bin")
sys.path.insert(0, "/workspace/context-exploration-engine/iowarp-cei-mcp/src")

from iowarp_cei_mcp import server

# Initialize runtime first (see test files for examples)
# ...

# Use MCP functions directly
result = server.context_bundle([{"src": "file::/tmp/test.bin", "dst": "iowarp::test", "format": "binary"}])
print(result)
```

## Future Enhancements

When `ContextSplice` is implemented in the API, add corresponding MCP tool.

## License

Part of IOWarp Core framework.
