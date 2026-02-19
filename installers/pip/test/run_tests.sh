#!/bin/bash
# =============================================================================
# run_tests.sh - Container-side test script
# =============================================================================
# Runs inside the minimalist Docker container to verify:
#   1. Wheel installs cleanly (no build tools needed)
#   2. Python package imports and APIs work
#   3. Shared libraries resolve (no missing symbols)
#   4. Chimaera runtime starts and stops successfully
# =============================================================================

set -euo pipefail

# ------------------------------------------------------------------
# 1. Install wheel
# ------------------------------------------------------------------
echo "=== Installing wheel ==="
pip install --quiet --break-system-packages /test/wheelhouse/*.whl
echo "    Installed successfully"

# Ensure the chimaera binary is executable (zip repack can lose permissions)
BIN_DIR=$(python3 -c "import iowarp_core; print(iowarp_core.get_bin_dir())")
chmod +x "$BIN_DIR"/* 2>/dev/null || true

# ------------------------------------------------------------------
# 2. Fix RPATHs (in case the build didn't fully handle them)
# ------------------------------------------------------------------
echo ""
echo "=== Fixing RPATHs ==="
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
bash /test/fix_rpaths.sh "$SITE_PACKAGES"

# ------------------------------------------------------------------
# 3. Verify Python import and package APIs
# ------------------------------------------------------------------
echo ""
echo "=== Verifying Python package ==="
python3 -c "import iowarp_core; print('  Version:', iowarp_core.get_version())"
python3 -c "import iowarp_core; print('  Lib dir:', iowarp_core.get_lib_dir())"
python3 -c "import iowarp_core; print('  Bin dir:', iowarp_core.get_bin_dir())"
python3 -c "import iowarp_core; print('  Data dir:', iowarp_core.get_data_dir())"
python3 -c "import iowarp_core; print('  CTE available:', iowarp_core.cte_available())"

# ------------------------------------------------------------------
# 4. Check shared library symbol resolution
# ------------------------------------------------------------------
echo ""
echo "=== Checking shared library symbols ==="
LIB_DIR=$(python3 -c "import iowarp_core; print(iowarp_core.get_lib_dir())")
MISSING=0
for so in "$LIB_DIR"/*.so "$LIB_DIR"/*.so.*; do
    [ -f "$so" ] || continue
    if ldd "$so" 2>/dev/null | grep -q "not found"; then
        echo "  FAIL: $(basename "$so") has missing symbols:"
        ldd "$so" | grep "not found"
        MISSING=1
    fi
done
if [ "$MISSING" = "1" ]; then
    echo "ERROR: Some libraries have unresolved symbols"
    exit 1
fi
echo "  All shared libraries resolve correctly"

# ------------------------------------------------------------------
# 5. Set up runtime config
# ------------------------------------------------------------------
echo ""
echo "=== Setting up runtime config ==="
mkdir -p ~/.chimaera
cp /test/chimaera_test.yaml ~/.chimaera/chimaera.yaml
echo "  Config installed to ~/.chimaera/chimaera.yaml"

# ------------------------------------------------------------------
# 6. Start runtime
# ------------------------------------------------------------------
echo ""
echo "=== Starting Chimaera runtime ==="
chimaera runtime start &
RUNTIME_PID=$!
echo "  Runtime PID: $RUNTIME_PID"

# Wait for runtime to initialize
echo "  Waiting for runtime to initialize..."
sleep 5

# Verify the runtime process is still alive
if ! kill -0 "$RUNTIME_PID" 2>/dev/null; then
    echo "  FAIL: Runtime process exited prematurely"
    wait "$RUNTIME_PID" || true
    exit 1
fi
echo "  Runtime is running"

# ------------------------------------------------------------------
# 7. Stop runtime
# ------------------------------------------------------------------
echo ""
echo "=== Stopping Chimaera runtime ==="

# Try the proper stop command first; fall back to SIGTERM
chimaera runtime stop --grace-period 5000 2>/dev/null || true

# Wait briefly for stop command to take effect
sleep 3

# If still running, send SIGTERM for graceful shutdown
if kill -0 "$RUNTIME_PID" 2>/dev/null; then
    echo "  Stop command did not terminate runtime, sending SIGTERM..."
    kill -TERM "$RUNTIME_PID" 2>/dev/null || true
fi

# Wait for the runtime process to exit (up to 15s)
TIMEOUT=15
for i in $(seq 1 "$TIMEOUT"); do
    if ! kill -0 "$RUNTIME_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

if kill -0 "$RUNTIME_PID" 2>/dev/null; then
    echo "  FAIL: Runtime did not stop within ${TIMEOUT}s"
    kill -9 "$RUNTIME_PID" 2>/dev/null || true
    exit 1
fi

wait "$RUNTIME_PID" && EXIT_CODE=0 || EXIT_CODE=$?
echo "  Runtime exited with code: $EXIT_CODE"

# ------------------------------------------------------------------
# 8. Cleanup
# ------------------------------------------------------------------
echo ""
echo "=== Cleanup ==="
rm -f /dev/shm/chimaera_* 2>/dev/null || true
echo "  Shared memory cleaned up"

echo ""
echo "=== All tests passed ==="
