#!/usr/bin/env python3
"""Test script for IOWarp CEI MCP server with runtime initialization.

This script:
1. Initializes Chimaera runtime
2. Initializes CTE and CAE
3. Tests the MCP server tools directly (without full MCP client)
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
print("IOWarp CEI MCP Server Test (with Runtime Initialization)")
print("=" * 70)
print()

# Step 1: Initialize runtime using CTE bindings
print("Step 1: Initializing Chimaera Runtime")
print("-" * 70)

try:
    import wrp_cte_core_ext as cte
    print("✓ Imported wrp_cte_core_ext module")
except ImportError as e:
    print(f"✗ Failed to import wrp_cte_core_ext: {e}")
    print("Make sure it's built with: cmake --build build --target wrp_cte_core_ext")
    sys.exit(1)

# Set up environment for runtime
print("Setting up environment...")
build_dir = "/workspace/build/bin"
os.environ["CHI_REPO_PATH"] = build_dir
os.environ["LD_LIBRARY_PATH"] = f"{build_dir}:{os.getenv('LD_LIBRARY_PATH', '')}"
print(f"  CHI_REPO_PATH: {build_dir}")
print(f"  LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

# Generate test configuration
print("\nGenerating test configuration...")
try:
    import yaml
    import socket

    def find_available_port(start_port=5555, end_port=5600):
        """Find an available port"""
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No available ports in range {start_port}-{end_port}")

    temp_dir = tempfile.gettempdir()

    # Create hostfile
    hostfile = os.path.join(temp_dir, "cei_mcp_test_hostfile")
    with open(hostfile, 'w') as f:
        f.write("localhost\n")

    # Find available port
    port = find_available_port()
    print(f"  Using port: {port}")

    # Create storage directory
    storage_dir = os.path.join(temp_dir, "cei_mcp_test_storage")
    os.makedirs(storage_dir, exist_ok=True)
    print(f"  Storage dir: {storage_dir}")

    # Generate config
    config = {
        'networking': {
            'protocol': 'zmq',
            'hostfile': hostfile,
            'port': port
        },
        'workers': {
            'num_workers': 4
        },
        'memory': {
            'main_segment_size': '1G',
            'client_data_segment_size': '512M',
            'runtime_data_segment_size': '512M'
        },
        'devices': [
            {
                'mount_point': storage_dir,
                'capacity': '1G'
            }
        ]
    }

    # Write config
    config_path = os.path.join(temp_dir, "cei_mcp_test_conf.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    os.environ['CHI_SERVER_CONF'] = config_path
    print(f"  Config file: {config_path}")
    print("✓ Configuration generated")

except ImportError:
    print("✗ PyYAML not available - cannot generate config")
    print("Install with: pip install pyyaml")
    sys.exit(1)

# Initialize Chimaera runtime
print("\nInitializing Chimaera runtime...")
sys.stdout.flush()

try:
    if not cte.chimaera_runtime_init():
        print("✗ chimaera_runtime_init() returned False")
        sys.exit(1)
    print("✓ Chimaera runtime initialized")
    time.sleep(0.5)  # Give runtime time to initialize
except Exception as e:
    print(f"✗ Runtime initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize Chimaera client
print("\nInitializing Chimaera client...")
sys.stdout.flush()

try:
    if not cte.chimaera_client_init():
        print("✗ chimaera_client_init() returned False")
        sys.exit(1)
    print("✓ Chimaera client initialized")
    time.sleep(0.2)  # Give client time to connect
except Exception as e:
    print(f"✗ Client initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize CTE
print("\nInitializing CTE subsystem...")
sys.stdout.flush()

try:
    pool_query = cte.PoolQuery.Dynamic()
    if not cte.initialize_cte(config_path, pool_query):
        print("✗ initialize_cte() returned False")
        sys.exit(1)
    print("✓ CTE subsystem initialized")
except Exception as e:
    print(f"✗ CTE initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Register storage target
print("\nRegistering storage target...")
try:
    client = cte.get_cte_client()
    mctx = cte.MemContext()
    target_path = os.path.join(storage_dir, "test_target")
    bdev_id = cte.PoolId(700, 0)
    target_query = cte.PoolQuery.Local()
    result = client.RegisterTarget(mctx, target_path, cte.BdevType.kFile,
                                   1024 * 1024 * 1024, target_query, bdev_id)
    if result == 0:
        print("✓ Storage target registered")
    else:
        print(f"⚠ Storage target registration returned {result} (may already be registered)")
except Exception as e:
    print(f"⚠ Could not register storage target: {e}")
    print("  Continuing anyway...")

print()
print("=" * 70)
print("Runtime Initialization Complete!")
print("=" * 70)
print()

# Step 2: Test MCP server tools
print("Step 2: Testing MCP Server Tools")
print("-" * 70)

# Import MCP server module
try:
    from iowarp_cei_mcp import server
    print("✓ Imported MCP server module")
except ImportError as e:
    print(f"✗ Failed to import MCP server: {e}")
    print("Make sure to install: pip install -e .")
    sys.exit(1)

# Also import wrp_cee to verify it initializes now
try:
    import wrp_cee
    print("✓ Imported wrp_cee module")
except ImportError as e:
    print(f"✗ Failed to import wrp_cee: {e}")
    sys.exit(1)

print()

# Create test data
test_file = os.path.join(temp_dir, "cei_mcp_test_data.bin")
test_data = b"IOWarp MCP Test Data - " + b"X" * 1000

print(f"Creating test file: {test_file}")
with open(test_file, 'wb') as f:
    f.write(test_data)
print(f"✓ Created test file ({len(test_data)} bytes)")
print()

test_passed = 0
test_failed = 0

# Test 1: context_bundle
print("-" * 70)
print("TEST 1: context_bundle")
print("-" * 70)

try:
    bundle_data = [
        {
            "src": f"file::{test_file}",
            "dst": "iowarp::mcp_test_tag",
            "format": "binary"
        }
    ]

    result = server.context_bundle(bundle_data)
    print(f"Result: {result}")

    if "Successfully" in result:
        print("✓ TEST 1 PASSED")
        test_passed += 1
    else:
        print("✗ TEST 1 FAILED")
        test_failed += 1
except Exception as e:
    print(f"✗ TEST 1 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

print()

# Test 2: context_query
print("-" * 70)
print("TEST 2: context_query")
print("-" * 70)

try:
    result = server.context_query("mcp_test_tag", ".*")
    print(f"Result:\n{result}")

    if "Found" in result or "No blobs" in result:
        print("✓ TEST 2 PASSED")
        test_passed += 1
    else:
        print("✗ TEST 2 FAILED")
        test_failed += 1
except Exception as e:
    print(f"✗ TEST 2 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

print()

# Test 3: context_retrieve
print("-" * 70)
print("TEST 3: context_retrieve")
print("-" * 70)

try:
    result = server.context_retrieve("mcp_test_tag", ".*")
    print(f"Result:\n{result}")

    if "Retrieved" in result or "No data" in result:
        print("✓ TEST 3 PASSED")
        test_passed += 1
    else:
        print("✗ TEST 3 FAILED")
        test_failed += 1
except Exception as e:
    print(f"✗ TEST 3 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

print()

# Test 4: context_destroy
print("-" * 70)
print("TEST 4: context_destroy")
print("-" * 70)

try:
    result = server.context_destroy(["mcp_test_tag"])
    print(f"Result: {result}")

    if "Successfully" in result:
        print("✓ TEST 4 PASSED")
        test_passed += 1
    else:
        print("✗ TEST 4 FAILED")
        test_failed += 1
except Exception as e:
    print(f"✗ TEST 4 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

print()

# Verify deletion
print("-" * 70)
print("TEST 5: Verify deletion")
print("-" * 70)

try:
    result = server.context_query("mcp_test_tag", ".*")
    print(f"Result:\n{result}")

    if "No blobs found" in result:
        print("✓ TEST 5 PASSED - Context deleted successfully")
        test_passed += 1
    else:
        print("✗ TEST 5 FAILED - Context still exists")
        test_failed += 1
except Exception as e:
    print(f"✗ TEST 5 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

print()

# Clean up
os.remove(test_file)

# Summary
print("=" * 70)
print("Test Summary")
print("=" * 70)
print(f"Passed: {test_passed}/5")
print(f"Failed: {test_failed}/5")
print()

if test_failed == 0:
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    sys.exit(0)
else:
    print("=" * 70)
    print("SOME TESTS FAILED ✗")
    print("=" * 70)
    sys.exit(1)
