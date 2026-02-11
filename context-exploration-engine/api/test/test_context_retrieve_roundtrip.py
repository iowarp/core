#!/usr/bin/env python3
"""
Regression test for context_retrieve double-free (issue #192).

Tests full data round-trip: bundle -> query -> retrieve -> verify -> destroy.
On main this test crashes with SIGABRT (double-free in ContextRetrieve).
With the fix (removing manual DelTask loop), it passes.

Requires a running IOWarp runtime with wrp_cte_core and wrp_cae_core modules.
"""

import sys
import os
import tempfile

# Add build directory to path for module import
sys.path.insert(0, os.path.join(os.getcwd(), "bin"))
sys.path.insert(0, "/usr/local/lib/python3.13/site-packages")


def test_retrieve_roundtrip():
    """Put data into IOWarp, retrieve it, and verify content matches."""
    import wrp_cee as cee

    ctx_interface = cee.ContextInterface()
    tag_name = "test_retrieve_roundtrip"

    # Use /dev/shm so the runtime container can also access the file
    tmpdir = tempfile.mkdtemp(prefix="iowarp_test_", dir="/dev/shm")
    test_file = os.path.join(tmpdir, "test_data.bin")

    # Create test file with known pattern
    pattern = b"IOWarp-Roundtrip-Test-" + b"0123456789ABCDEF"
    file_size = 64 * 1024  # 64 KB
    original_data = (pattern * (file_size // len(pattern) + 1))[:file_size]
    with open(test_file, "wb") as f:
        f.write(original_data)

    try:
        # 1. Bundle (put)
        ctx = cee.AssimilationCtx(
            src=f"file::{test_file}",
            dst=f"iowarp::{tag_name}",
            format="binary",
        )
        bundle_result = ctx_interface.context_bundle([ctx])
        assert bundle_result == 0, f"context_bundle failed with code {bundle_result}"

        # 2. Query (verify blobs exist)
        blobs = ctx_interface.context_query(tag_name, ".*", 0)
        assert len(blobs) > 0, "context_query returned no blobs after bundle"

        # 3. Retrieve (get data back) - this crashes on main with double-free
        packed_data = ctx_interface.context_retrieve(
            tag_name, ".*",
            max_results=1024,
            max_context_size=4 * 1024 * 1024,
            batch_size=32,
        )
        total_size = sum(len(d) for d in packed_data)
        assert total_size > 0, "context_retrieve returned no data"

        # 4. Verify content
        retrieved_bytes = b"".join(
            d.encode("latin-1") if isinstance(d, str) else d
            for d in packed_data
        )
        assert original_data in retrieved_bytes, (
            f"Retrieved data ({len(retrieved_bytes)} bytes) does not contain "
            f"original data ({len(original_data)} bytes)"
        )

        # 5. Destroy (cleanup)
        destroy_result = ctx_interface.context_destroy([tag_name])
        assert destroy_result == 0, f"context_destroy failed with code {destroy_result}"

        return True

    finally:
        os.unlink(test_file)
        os.rmdir(tmpdir)


def main():
    print("=" * 60)
    print(" Regression test: context_retrieve round-trip (#192)")
    print("=" * 60)

    try:
        success = test_retrieve_roundtrip()
        if success:
            print("\nPASS: Data round-trip verified successfully")
            return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nFAIL: {e}")
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
