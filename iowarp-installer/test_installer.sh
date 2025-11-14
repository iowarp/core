#!/bin/bash
# Test script for IOWarp installer

set -e

echo "🧪 Testing IOWarp installer..."

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up test directory: $TEST_DIR"
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

cd "$TEST_DIR"

# Test 1: Check dependencies only
echo "Test 1: Checking dependencies..."
python3 -m iowarp.main install --check-deps-only

# Test 2: Install to temporary prefix
echo "Test 2: Installing to temporary prefix..."
TEMP_PREFIX="$TEST_DIR/iowarp-install"
python3 -m iowarp.main install --prefix "$TEMP_PREFIX" --keep-build

# Test 3: Check status
echo "Test 3: Checking installation status..."
python3 -m iowarp.main status --prefix "$TEMP_PREFIX"

# Test 4: Show environment
echo "Test 4: Showing environment variables..."
python3 -m iowarp.main config env --prefix "$TEMP_PREFIX"

echo "✅ All tests passed!"