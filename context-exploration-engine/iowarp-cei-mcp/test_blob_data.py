#!/usr/bin/env python3
"""Test script demonstrating MCP server with various blob data types.

This script shows:
1. Storing different types of data (text, binary, structured)
2. Querying and retrieving blobs
3. Working with multiple files
4. Pattern-based queries
"""

import os
import sys
import tempfile
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, "/workspace/build/bin")

print("=" * 70)
print("IOWarp CEI MCP Server - Blob Data Test")
print("=" * 70)
print()

# Initialize runtime
print("Initializing Runtime...")
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
    hostfile = os.path.join(temp_dir, "blob_test_hostfile")
    with open(hostfile, 'w') as f:
        f.write("localhost\n")

    port = find_available_port()
    storage_dir = os.path.join(temp_dir, "blob_test_storage")
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

    config_path = os.path.join(temp_dir, "blob_test_conf.yaml")
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
from iowarp_cei_mcp import server
import wrp_cee

print("=" * 70)
print("Creating Test Data")
print("=" * 70)
print()

# Create test files with different data types
test_files = []

# 1. Text file
text_file = os.path.join(temp_dir, "test_text.txt")
text_content = """IOWarp Context Interface Demo
============================

This is a sample text file demonstrating context storage.
It contains multiple lines and various characters.

Features:
- Line breaks
- Special chars: @#$%^&*()
- Numbers: 12345
- Unicode: café, 日本語

End of file.
"""
with open(text_file, 'w') as f:
    f.write(text_content)
test_files.append(("text_data", text_file, len(text_content)))
print(f"1. Created text file: {len(text_content)} bytes")

# 2. Binary file with pattern
binary_file = os.path.join(temp_dir, "test_binary.bin")
binary_pattern = bytes(range(256)) * 10  # All byte values repeated
with open(binary_file, 'wb') as f:
    f.write(binary_pattern)
test_files.append(("binary_data", binary_file, len(binary_pattern)))
print(f"2. Created binary file: {len(binary_pattern)} bytes")

# 3. JSON-like structured data
json_file = os.path.join(temp_dir, "test_json.txt")
json_content = json.dumps({
    "experiment": "iowarp_demo",
    "timestamp": "2025-11-15T17:00:00Z",
    "parameters": {
        "iterations": 1000,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "results": {
        "accuracy": 0.95,
        "loss": 0.05,
        "metrics": [0.92, 0.94, 0.95, 0.96, 0.95]
    }
}, indent=2)
with open(json_file, 'w') as f:
    f.write(json_content)
test_files.append(("json_metadata", json_file, len(json_content)))
print(f"3. Created JSON file: {len(json_content)} bytes")

# 4. Large binary file
large_file = os.path.join(temp_dir, "test_large.bin")
large_data = b"IOWARP" * 50000  # ~300KB
with open(large_file, 'wb') as f:
    f.write(large_data)
test_files.append(("large_dataset", large_file, len(large_data)))
print(f"4. Created large file: {len(large_data):,} bytes")

# 5. Image-like data (random bytes simulating an image)
import random
random.seed(42)
image_file = os.path.join(temp_dir, "test_image.raw")
image_data = bytes([random.randint(0, 255) for _ in range(10000)])
with open(image_file, 'wb') as f:
    f.write(image_data)
test_files.append(("image_data", image_file, len(image_data)))
print(f"5. Created image-like file: {len(image_data):,} bytes")

print()
print("=" * 70)
print("TEST 1: Bundle Multiple Files")
print("=" * 70)
print()

# Bundle all files into different contexts
bundle = []
for tag_suffix, file_path, size in test_files:
    bundle.append({
        "src": f"file::{file_path}",
        "dst": f"iowarp::test_{tag_suffix}",
        "format": "binary"
    })
    print(f"  Bundling: {tag_suffix} ({size:,} bytes)")

result = server.context_bundle(bundle)
print(f"\n{result}")
print()

print("=" * 70)
print("TEST 2: Query Individual Contexts")
print("=" * 70)
print()

for tag_suffix, _, _ in test_files:
    print(f"\nQuerying: test_{tag_suffix}")
    print("-" * 40)
    result = server.context_query(f"test_{tag_suffix}", ".*")
    print(result)

print()
print("=" * 70)
print("TEST 3: Query with Patterns")
print("=" * 70)
print()

# Query all test contexts
print("Query: All contexts starting with 'test_'")
print("-" * 40)
result = server.context_query("test_.*", ".*", max_results=100)
print(result)

print()
print("Query: Only data contexts (excluding metadata)")
print("-" * 40)
result = server.context_query("test_(text|binary|large|image)_data", "chunk_.*")
print(result)

print()
print("=" * 70)
print("TEST 4: Retrieve Specific Data")
print("=" * 70)
print()

# Retrieve text data
print("Retrieving: test_text_data")
print("-" * 40)
result = server.context_retrieve("test_text_data", ".*")
print(result)

print()
print("Retrieving: test_binary_data (first 256 bytes)")
print("-" * 40)
result = server.context_retrieve("test_binary_data", "chunk_0", max_context_size=512)
print(result)

print()
print("Retrieving: test_json_metadata")
print("-" * 40)
result = server.context_retrieve("test_json_metadata", ".*")
print(result)

print()
print("=" * 70)
print("TEST 5: Retrieve with Limits")
print("=" * 70)
print()

# Test max_results limit
print("Query: Limited to 2 results")
print("-" * 40)
result = server.context_query("test_.*", ".*", max_results=2)
print(result)

print()
print("Retrieve: Limited to 1KB")
print("-" * 40)
result = server.context_retrieve("test_large_dataset", ".*",
                                 max_results=1, max_context_size=1024)
print(result)

print()
print("=" * 70)
print("TEST 6: Batch Operations")
print("=" * 70)
print()

# Create multiple files in same context
print("Creating batch of files for single context...")
batch_files = []
for i in range(5):
    batch_file = os.path.join(temp_dir, f"batch_{i}.txt")
    content = f"Batch file {i}\n" + "Data " * 100
    with open(batch_file, 'w') as f:
        f.write(content)
    batch_files.append(batch_file)
    print(f"  Created batch_{i}.txt: {len(content)} bytes")

# Bundle all into one context
batch_bundle = [{
    "src": f"file::{f}",
    "dst": f"iowarp::batch_context",
    "format": "binary"
} for f in batch_files]

result = server.context_bundle(batch_bundle)
print(f"\n{result}")

print()
print("Querying batch context:")
print("-" * 40)
result = server.context_query("batch_context", ".*")
print(result)

print()
print("=" * 70)
print("TEST 7: Cleanup - Destroy All Test Contexts")
print("=" * 70)
print()

# Destroy all test contexts
contexts_to_destroy = [f"test_{tag}" for tag, _, _ in test_files]
contexts_to_destroy.append("batch_context")

print(f"Destroying {len(contexts_to_destroy)} contexts...")
for ctx in contexts_to_destroy:
    print(f"  - {ctx}")

result = server.context_destroy(contexts_to_destroy)
print(f"\n{result}")

print()
print("Verifying deletion:")
print("-" * 40)
result = server.context_query("test_.*", ".*")
print(result)
result = server.context_query("batch_context", ".*")
print(result)

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print()
print(f"✓ Created {len(test_files)} different data types")
print(f"✓ Bundled {len(test_files) + len(batch_files)} files")
print(f"✓ Tested pattern-based queries")
print(f"✓ Retrieved data with various limits")
print(f"✓ Cleaned up {len(contexts_to_destroy)} contexts")
print()
print("=" * 70)
print("ALL BLOB DATA TESTS COMPLETED ✓")
print("=" * 70)

# Cleanup temp files
for _, file_path, _ in test_files:
    os.remove(file_path)
for batch_file in batch_files:
    os.remove(batch_file)
