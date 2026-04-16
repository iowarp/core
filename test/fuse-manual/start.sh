#!/bin/bash
# Start IOWarp runtime + FUSE filesystem
# Usage: ./start.sh [mount_point]
#
# After running this, the FUSE mount is live and you can:
#   cp -r /workspace/* /tmp/cte_fuse_mount/
#   ls /tmp/cte_fuse_mount/
#   diff /workspace/README.md /tmp/cte_fuse_mount/README.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_BIN="$REPO_ROOT/build/bin"
MOUNT_POINT="${1:-/tmp/cte_fuse_mount}"
PID_DIR="$SCRIPT_DIR/.pids"

export PATH="$BUILD_BIN:$PATH"
export LD_LIBRARY_PATH="$BUILD_BIN:${LD_LIBRARY_PATH:-}"
export WRP_RUNTIME_CONF="$SCRIPT_DIR/cte_config.yaml"
export HSHM_LOG_LEVEL=info

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
die()   { echo -e "${RED}[ERR]${NC}  $1" >&2; exit 1; }

# --- Preflight checks -------------------------------------------------------

[ -x "$BUILD_BIN/chimaera" ]      || die "chimaera not found — build with: cmake --preset release-fuse && cmake --build build -j\$(nproc)"
[ -x "$BUILD_BIN/wrp_cte_fuse" ]  || die "wrp_cte_fuse not found — build with: cmake --preset release-fuse && cmake --build build -j\$(nproc)"
command -v fusermount3 &>/dev/null || die "fusermount3 not found — install fuse3: sudo apt install fuse3"
[ -c /dev/fuse ]                   || die "/dev/fuse not available"

# --- Cleanup any previous run -----------------------------------------------

if [ -f "$PID_DIR/fuse.pid" ] && kill -0 "$(cat "$PID_DIR/fuse.pid")" 2>/dev/null; then
    info "Stopping previous FUSE daemon..."
    "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
fi

mkdir -p "$PID_DIR" "$MOUNT_POINT"

# --- Start Chimaera runtime -------------------------------------------------

info "Starting Chimaera runtime..."
export CHI_SERVER_CONF="$WRP_RUNTIME_CONF"
chimaera runtime start &
RUNTIME_PID=$!
echo "$RUNTIME_PID" > "$PID_DIR/runtime.pid"
sleep 3

if ! kill -0 "$RUNTIME_PID" 2>/dev/null; then
    die "Chimaera runtime failed to start"
fi
ok "Chimaera runtime started (PID $RUNTIME_PID)"

# --- Start FUSE daemon -------------------------------------------------------

info "Mounting FUSE filesystem at $MOUNT_POINT ..."
# Run FUSE daemon as a pure client — connect to the already-running
# runtime instead of trying to start its own.
CHI_WITH_RUNTIME=0 wrp_cte_fuse "$MOUNT_POINT" -f &
FUSE_PID=$!
echo "$FUSE_PID" > "$PID_DIR/fuse.pid"
echo "$MOUNT_POINT" > "$PID_DIR/mount_point"
sleep 2

if ! kill -0 "$FUSE_PID" 2>/dev/null; then
    die "wrp_cte_fuse failed to start"
fi
ok "FUSE filesystem mounted (PID $FUSE_PID)"

# --- Done --------------------------------------------------------------------

echo ""
echo "========================================="
echo "IOWarp FUSE filesystem is ready!"
echo "  Mount point:  $MOUNT_POINT"
echo "  Runtime PID:  $RUNTIME_PID"
echo "  FUSE PID:     $FUSE_PID"
echo ""
echo "Try:"
echo "  cp -r /workspace/README.md $MOUNT_POINT/"
echo "  cat $MOUNT_POINT/README.md"
echo "  rsync -av --exclude='build*' --exclude='.git' /workspace/ $MOUNT_POINT/workspace/"
echo ""
echo "Stop with:  ./stop.sh"
echo "========================================="
