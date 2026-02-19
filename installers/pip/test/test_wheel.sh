#!/bin/bash
# =============================================================================
# test_wheel.sh - Build and test pip wheel in Docker
# =============================================================================
# Uses a multi-stage Dockerfile to:
#   1. Build the wheel in iowarp/deps-cpu (static archives, no conda)
#   2. Verify external deps are statically linked (no yaml-cpp/zmq/sodium leaks)
#   3. Install in a clean python:slim container
#   4. Test Python import, shared lib resolution, and runtime start/stop
#
# Prerequisites:
#   - Docker
#   - iowarp/deps-cpu:latest image (build with docker/deps-cpu.Dockerfile)
#
# Usage (from the repo root):
#   bash installers/pip/test/test_wheel.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PIP_DIR/../.." && pwd)"

echo "======================================================================"
echo "IOWarp Pip Wheel Test"
echo "======================================================================"

# ------------------------------------------------------------------
# Step 1: Build the multi-stage Docker image (build + test stages)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 1: Building wheel and test container..."
echo "    Context: $PROJECT_ROOT"

cd "$PROJECT_ROOT"
docker build --progress=plain \
    -t iowarp/pip-test:latest \
    -f installers/pip/test/Dockerfile .

# ------------------------------------------------------------------
# Step 2: Run the tests
# ------------------------------------------------------------------
echo ""
echo ">>> Step 2: Running tests in clean container..."

docker run --rm --shm-size=256m iowarp/pip-test:latest

echo ""
echo "======================================================================"
echo "All tests passed!"
echo "======================================================================"
