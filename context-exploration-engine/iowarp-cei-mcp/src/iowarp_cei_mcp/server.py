#!/usr/bin/env python3
"""MCP server wrapping IOWarp Context Exploration Interface.

This server exposes IOWarp's Context Interface functionality through the
Model Context Protocol (MCP), enabling AI assistants to:
- Store data into IOWarp contexts
- Query contexts by patterns
- Retrieve context data
- Manage context lifecycles
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iowarp-cei-mcp")

# Import MCP SDK (optional for direct testing)
try:
    from mcp import FastMCP
    MCP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MCP SDK not available: {e}")
    logger.warning("MCP server mode disabled. Direct function calls only.")
    logger.warning("Install with: pip install mcp")
    MCP_AVAILABLE = False
    FastMCP = None  # type: ignore

# Import IOWarp CEE API
try:
    import wrp_cee
except ImportError as e:
    logger.error(f"Failed to import wrp_cee module: {e}")
    logger.error("Make sure wrp_cee is built and in your PYTHONPATH")
    logger.error("Build with: cmake --preset=debug -DWRP_CORE_ENABLE_PYTHON=ON")
    sys.exit(1)


# Initialize FastMCP server (if available)
mcp = FastMCP("IOWarp Context Exploration Interface") if MCP_AVAILABLE else None

# Global context interface (initialized lazily)
_ctx_interface = None


def get_context_interface():
    """Get or initialize the ContextInterface singleton."""
    global _ctx_interface
    if _ctx_interface is None:
        try:
            _ctx_interface = wrp_cee.ContextInterface()
            logger.info("ContextInterface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContextInterface: {e}")
            raise
    return _ctx_interface


def context_bundle(bundle: list[dict]) -> str:
    """Bundle and assimilate data into IOWarp contexts.

    Stores files or data into named contexts for later retrieval.
    Each context item specifies source (file path), destination (context name),
    and format (binary, hdf5, etc.).

    Args:
        bundle: List of assimilation contexts. Each dict should contain:
            - src (str): Source URL (e.g., 'file::/path/to/file')
            - dst (str): Destination URL (e.g., 'iowarp::tag_name')
            - format (str, optional): Data format (default: 'binary')
            - depends_on (str, optional): Dependency identifier
            - range_off (int, optional): Byte offset in source file
            - range_size (int, optional): Number of bytes to read (0=full file)
            - src_token (str, optional): Source authentication token
            - dst_token (str, optional): Destination authentication token

    Returns:
        Success message or error description

    Example:
        bundle = [
            {
                "src": "file::/tmp/data.bin",
                "dst": "iowarp::my_dataset",
                "format": "binary"
            }
        ]
        result = context_bundle(bundle)
    """
    ctx_interface = get_context_interface()

    if not bundle:
        return "Error: Empty bundle provided"

    # Build list of AssimilationCtx objects
    ctx_list = []
    for item in bundle:
        ctx = wrp_cee.AssimilationCtx(
            src=item["src"],
            dst=item["dst"],
            format=item.get("format", "binary"),
            depends_on=item.get("depends_on", ""),
            range_off=item.get("range_off", 0),
            range_size=item.get("range_size", 0),
            src_token=item.get("src_token", ""),
            dst_token=item.get("dst_token", "")
        )
        ctx_list.append(ctx)

    # Execute bundling
    result = ctx_interface.context_bundle(ctx_list)

    if result == 0:
        msg = f"Successfully assimilated {len(ctx_list)} context(s)"
        logger.info(msg)
        return msg
    else:
        error_msg = f"Bundle failed with error code: {result}"
        logger.error(error_msg)
        return error_msg


# @mcp.tool()  # Decorator applied conditionally at module end
def context_query(tag_re: str, blob_re: str, max_results: int = 0) -> str:
    """Query IOWarp contexts for blobs matching tag and blob regex patterns.

    Returns a list of blob names that match the specified patterns.
    Use '.*' to match all blobs in a tag.

    Args:
        tag_re: Tag regex pattern to match
        blob_re: Blob regex pattern to match
        max_results: Maximum number of results (0=unlimited)

    Returns:
        List of matching blob names or message if none found

    Example:
        # Query all blobs in a tag
        result = context_query("my_dataset", ".*")

        # Query specific pattern
        result = context_query("experiment_.*", "result_[0-9]+", max_results=100)
    """
    ctx_interface = get_context_interface()

    # Execute query
    results = ctx_interface.context_query(tag_re, blob_re, max_results)

    if results:
        msg = f"Found {len(results)} blob(s):\n" + "\n".join(f"  - {blob}" for blob in results)
        logger.info(f"Query returned {len(results)} results")
    else:
        msg = f"No blobs found matching tag_re='{tag_re}', blob_re='{blob_re}'"
        logger.info("Query returned no results")

    return msg


# @mcp.tool()  # Decorator applied conditionally at module end
def context_retrieve(
    tag_re: str,
    blob_re: str,
    max_results: int = 1024,
    max_context_size: int = 256 * 1024 * 1024,
    batch_size: int = 32
) -> str:
    """Retrieve both identities and data of objects matching tag and blob patterns.

    Returns packed binary data containing all matching blobs.
    Automatically batches retrieval for efficiency.

    Args:
        tag_re: Tag regex pattern to match
        blob_re: Blob regex pattern to match
        max_results: Maximum number of blobs (0=unlimited, default: 1024)
        max_context_size: Maximum total size in bytes (default: 256MB)
        batch_size: Concurrent AsyncGetBlob operations (default: 32)

    Returns:
        Summary of retrieved data with size information and preview

    Example:
        # Retrieve all data from a context
        result = context_retrieve("my_dataset", ".*")

        # Retrieve with limits
        result = context_retrieve(
            "large_dataset", "chunk_.*",
            max_results=500,
            max_context_size=512 * 1024 * 1024
        )
    """
    ctx_interface = get_context_interface()

    # Execute retrieval
    packed_data = ctx_interface.context_retrieve(
        tag_re, blob_re, max_results, max_context_size, batch_size
    )

    if packed_data:
        total_bytes = sum(len(data) for data in packed_data)
        msg = (
            f"Retrieved {len(packed_data)} packed context(s)\n"
            f"Total data size: {total_bytes:,} bytes ({total_bytes / 1024:.2f} KB)"
        )
        logger.info(f"Retrieved {total_bytes} bytes")

        # For demonstration, show first 100 bytes as hex
        if packed_data[0]:
            preview = packed_data[0][:100]
            # Convert to bytes if it's a string
            if isinstance(preview, str):
                preview_bytes = preview.encode('latin-1')
            else:
                preview_bytes = preview
            hex_preview = " ".join(f"{b:02x}" for b in preview_bytes)
            msg += f"\n\nData preview (first {len(preview_bytes)} bytes):\n{hex_preview}"
            if len(packed_data[0]) > 100:
                msg += f"\n... ({len(packed_data[0]) - 100} more bytes)"
    else:
        msg = f"No data found matching tag_re='{tag_re}', blob_re='{blob_re}'"
        logger.info("Retrieve returned no data")

    return msg


# @mcp.tool()  # Decorator applied conditionally at module end
def context_destroy(context_names: list[str]) -> str:
    """Destroy IOWarp contexts by name.

    Permanently deletes the specified contexts and all their data.

    Args:
        context_names: List of context names to destroy

    Returns:
        Success message or error description

    Example:
        # Destroy single context
        result = context_destroy(["my_old_dataset"])

        # Destroy multiple contexts
        result = context_destroy(["temp_data_1", "temp_data_2", "temp_data_3"])
    """
    ctx_interface = get_context_interface()

    if not context_names:
        return "Error: Empty context list provided"

    # Execute destruction
    result = ctx_interface.context_destroy(context_names)

    if result == 0:
        msg = f"Successfully destroyed {len(context_names)} context(s): {', '.join(context_names)}"
        logger.info(msg)
        return msg
    else:
        error_msg = f"Destroy failed with error code: {result}"
        logger.error(error_msg)
        return error_msg


# Register tools with MCP if available
if MCP_AVAILABLE and mcp is not None:
    mcp.tool()(context_bundle)
    mcp.tool()(context_query)
    mcp.tool()(context_retrieve)
    mcp.tool()(context_destroy)


def main():
    """Entry point for the MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP SDK not available - cannot run server")
        logger.error("Install with: pip install mcp")
        sys.exit(1)

    logger.info("Starting IOWarp CEI MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
