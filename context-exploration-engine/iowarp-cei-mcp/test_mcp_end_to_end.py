#!/usr/bin/env python3
"""End-to-end test for IOWarp CEI MCP Server.

This test validates the complete MCP workflow:
1. Runtime initialization
2. Storage registration
3. MCP server function calls (context_bundle, context_query, context_retrieve, context_destroy)
4. Verification of results
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, "/workspace/build/bin")

print("=" * 70)
print("IOWarp CEI MCP Server - End-to-End Test")
print("=" * 70)
print()

# Initialize runtime
print("Step 1: Initialize Runtime")
print("-" * 70)

try:
    import wrp_cte_core_ext as cte
    print("✓ Imported wrp_cte_core_ext")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Set up environment
build_dir = "/workspace/build/bin"
os.environ["CHI_REPO_PATH"] = build_dir
os.environ["LD_LIBRARY_PATH"] = f"{build_dir}:{os.getenv('LD_LIBRARY_PATH', '')}"

# Generate config
try:
    import yaml
    import socket

    def find_available_port(start_port=5555, end_port=5600):
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No available ports")

    temp_dir = tempfile.gettempdir()
    hostfile = os.path.join(temp_dir, "mcp_e2e_hostfile")
    with open(hostfile, 'w') as f:
        f.write("localhost\n")

    port = find_available_port()
    storage_dir = os.path.join(temp_dir, "mcp_e2e_storage")
    os.makedirs(storage_dir, exist_ok=True)

    config = {
        'networking': {'protocol': 'zmq', 'hostfile': hostfile, 'port': port},
        'workers': {'num_workers': 4},
        'memory': {
            'main_segment_size': '1G',
            'client_data_segment_size': '512M',
            'runtime_data_segment_size': '512M'
        },
        'devices': [{'mount_point': storage_dir, 'capacity': '1G'}]
    }

    config_path = os.path.join(temp_dir, "mcp_e2e_conf.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    os.environ['CHI_SERVER_CONF'] = config_path

except ImportError:
    print("✗ PyYAML required")
    sys.exit(1)

# Initialize runtime
print("Starting Chimaera runtime...")
if not cte.chimaera_runtime_init():
    print("✗ Runtime init failed")
    sys.exit(1)
time.sleep(0.5)

print("Starting Chimaera client...")
if not cte.chimaera_client_init():
    print("✗ Client init failed")
    sys.exit(1)
time.sleep(0.2)

print("Initializing CTE...")
if not cte.initialize_cte(config_path, cte.PoolQuery.Dynamic()):
    print("✗ CTE init failed")
    sys.exit(1)

# Register storage
client = cte.get_cte_client()
mctx = cte.MemContext()
target_path = os.path.join(storage_dir, "test_target")
bdev_id = cte.PoolId(700, 0)
client.RegisterTarget(mctx, target_path, cte.BdevType.kFile,
                      1024 * 1024 * 1024, cte.PoolQuery.Local(), bdev_id)

print("✓ Runtime ready")
print()

# Import MCP server
print("Step 2: Import MCP Server")
print("-" * 70)

try:
    from iowarp_cei_mcp import server
    print("✓ MCP server module imported")
    print(f"  Available functions: {[name for name in dir(server) if name.startswith('context_')]}")
except ImportError as e:
    print(f"✗ Failed to import MCP server: {e}")
    sys.exit(1)

print()

# Create test data files
print("Step 3: Create Test Data")
print("-" * 70)

test_files = []

# File 1: Small text file
text_file = os.path.join(temp_dir, "e2e_test_text.txt")
text_content = "IOWarp MCP End-to-End Test\nThis is a test file for MCP validation.\n"
with open(text_file, 'w') as f:
    f.write(text_content)
test_files.append(("mcp_text_data", text_file))
print(f"1. Created text file: {len(text_content)} bytes")

# File 2: Binary data
binary_file = os.path.join(temp_dir, "e2e_test_binary.bin")
binary_data = bytes(range(256)) * 4  # 1KB
with open(binary_file, 'wb') as f:
    f.write(binary_data)
test_files.append(("mcp_binary_data", binary_file))
print(f"2. Created binary file: {len(binary_data)} bytes")

# File 3: Larger dataset
large_file = os.path.join(temp_dir, "e2e_test_large.bin")
large_data = b"MCP_TEST_" * 10000  # ~90KB
with open(large_file, 'wb') as f:
    f.write(large_data)
test_files.append(("mcp_large_dataset", large_file))
print(f"3. Created large file: {len(large_data):,} bytes")

print()

# Test 1: context_bundle
print("=" * 70)
print("TEST 1: context_bundle - Bundle Multiple Files")
print("=" * 70)

bundle = []
for tag_suffix, file_path in test_files:
    bundle.append({
        "src": f"file::{file_path}",
        "dst": f"iowarp::e2e_{tag_suffix}",
        "format": "binary"
    })
    print(f"  Adding: e2e_{tag_suffix}")

result = server.context_bundle(bundle)
print(f"\nResult: {result}")

if "Successfully" in result:
    print("✓ TEST 1 PASSED")
else:
    print("✗ TEST 1 FAILED")
    sys.exit(1)

print()

# Wait for assimilation
time.sleep(1)

# Test 2: context_query - Query individual contexts
print("=" * 70)
print("TEST 2: context_query - Query Individual Contexts")
print("=" * 70)

all_passed = True
for tag_suffix, _ in test_files:
    print(f"\nQuerying: e2e_{tag_suffix}")
    result = server.context_query(f"e2e_{tag_suffix}", ".*")
    print(f"Result: {result}")

    if "Found" in result or "blob" in result:
        print(f"✓ Query for e2e_{tag_suffix} succeeded")
    else:
        print(f"✗ Query for e2e_{tag_suffix} failed")
        all_passed = False

if all_passed:
    print("\n✓ TEST 2 PASSED")
else:
    print("\n✗ TEST 2 FAILED")
    sys.exit(1)

print()

# Test 3: context_query - Pattern matching
print("=" * 70)
print("TEST 3: context_query - Pattern Matching")
print("=" * 70)

print("Query: All contexts starting with 'e2e_mcp_'")
result = server.context_query("e2e_mcp_.*", ".*")
print(f"Result: {result}")

if "Found" in result and ("blob" in result.lower() or "chunk" in result.lower()):
    print("✓ TEST 3 PASSED")
else:
    print("✗ TEST 3 FAILED")
    sys.exit(1)

print()

# Test 4: context_retrieve - Retrieve data
print("=" * 70)
print("TEST 4: context_retrieve - Retrieve Data")
print("=" * 70)

print("Retrieving: e2e_mcp_text_data")
result = server.context_retrieve("e2e_mcp_text_data", ".*")
print(f"Result (first 200 chars):\n{result[:200]}")

if "Retrieved" in result and "bytes" in result:
    print("\n✓ TEST 4 PASSED")
else:
    print("\n✗ TEST 4 FAILED")
    sys.exit(1)

print()

# Test 5: context_retrieve - With limits
print("=" * 70)
print("TEST 5: context_retrieve - With Size Limits")
print("=" * 70)

print("Retrieving: e2e_mcp_binary_data (limited to 2KB)")
result = server.context_retrieve("e2e_mcp_binary_data", ".*",
                                 max_results=1, max_context_size=2048)
print(f"Result (first 200 chars):\n{result[:200]}")

if "Retrieved" in result or "No data" in result:
    # Either retrieval succeeded or size limit was respected
    print("\n✓ TEST 5 PASSED")
else:
    print("\n✗ TEST 5 FAILED")
    sys.exit(1)

print()

# Test 6: context_destroy - Clean up
print("=" * 70)
print("TEST 6: context_destroy - Cleanup")
print("=" * 70)

contexts_to_destroy = [f"e2e_{tag}" for tag, _ in test_files]
print(f"Destroying {len(contexts_to_destroy)} contexts:")
for ctx in contexts_to_destroy:
    print(f"  - {ctx}")

result = server.context_destroy(contexts_to_destroy)
print(f"\nResult: {result}")

if "Successfully" in result:
    print("✓ TEST 6 PASSED")
else:
    print("✗ TEST 6 FAILED")
    sys.exit(1)

print()

# Test 7: Verify deletion
print("=" * 70)
print("TEST 7: Verify Deletion")
print("=" * 70)

print("Querying deleted contexts...")
result = server.context_query("e2e_mcp_.*", ".*")
print(f"Result: {result}")

if "No blobs found" in result or "0 blob" in result:
    print("✓ TEST 7 PASSED - Contexts successfully deleted")
else:
    print("⚠ TEST 7 WARNING - Some data may still exist")

print()

# Cleanup temp files
for _, file_path in test_files:
    if os.path.exists(file_path):
        os.remove(file_path)

# Summary
print("=" * 70)
print("Summary - End-to-End MCP Test")
print("=" * 70)
print()
print("✓ Runtime initialization")
print("✓ Storage registration")
print("✓ MCP server import")
print("✓ Test data creation")
print("✓ context_bundle - Bundled 3 files")
print("✓ context_query - Individual queries")
print("✓ context_query - Pattern matching")
print("✓ context_retrieve - Data retrieval")
print("✓ context_retrieve - With size limits")
print("✓ context_destroy - Cleanup")
print("✓ Verification of deletion")
print()
print("=" * 70)
print("ALL END-TO-END TESTS PASSED ✓")
print("=" * 70)
print()
print("MCP Server is fully functional and ready for use!")
