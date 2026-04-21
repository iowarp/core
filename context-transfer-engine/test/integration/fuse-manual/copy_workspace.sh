#!/bin/bash
# Copy /workspace into the FUSE-mounted CTE filesystem and verify
# Usage: ./copy_workspace.sh [mount_point]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

if [ -f "$PID_DIR/mount_point" ]; then
    MOUNT_POINT="$(cat "$PID_DIR/mount_point")"
else
    MOUNT_POINT="${1:-/tmp/cte_fuse_mount}"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; }

# --- Preflight ---------------------------------------------------------------

if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo -e "${RED}[ERR]${NC}  FUSE not mounted at $MOUNT_POINT — run ./start.sh first" >&2
    exit 1
fi

# --- Copy --------------------------------------------------------------------

info "Copying /workspace into $MOUNT_POINT/workspace/ ..."
info "  (excluding build dirs, .git, caches)"

EXCLUDES="build|build_debug|build_release|build_socket|\.git|\.ppi-jarvis|\.jarvis-private|\.jarvis-shared|\.ssh-host|__pycache__|node_modules|\.cache"

if command -v rsync &>/dev/null; then
    rsync -a --info=progress2 \
        --exclude='build*' \
        --exclude='.git' \
        --exclude='.ppi-jarvis' \
        --exclude='.jarvis-private' \
        --exclude='.jarvis-shared' \
        --exclude='.ssh-host' \
        --exclude='__pycache__' \
        --exclude='node_modules' \
        --exclude='.cache' \
        /workspace/ "$MOUNT_POINT/workspace/"
else
    info "rsync not found, using find+cp (slower)..."
    cd /workspace
    find . -not -path "./.git/*" \
           -not -path "./build*" \
           -not -path "./.ppi-jarvis/*" \
           -not -path "./.jarvis-private/*" \
           -not -path "./.jarvis-shared/*" \
           -not -path "./.ssh-host/*" \
           -not -path "./__pycache__/*" \
           -not -path "./.cache/*" \
           -not -name "*.o" \
           -type f | while IFS= read -r f; do
        dir="$MOUNT_POINT/workspace/$(dirname "$f")"
        mkdir -p "$dir" 2>/dev/null || true
        cp "$f" "$dir/" 2>/dev/null || true
    done
    cd - >/dev/null
fi

ok "Copy complete"

# --- Verify ------------------------------------------------------------------

info "Verifying a sample of files..."

FAILURES=0
CHECKED=0

verify_file() {
    local src="$1"
    local dst="$MOUNT_POINT/workspace/${src#/workspace/}"
    if [ ! -f "$dst" ]; then
        fail "Missing: $dst"
        FAILURES=$((FAILURES + 1))
        return
    fi
    if ! cmp -s "$src" "$dst"; then
        fail "Mismatch: $src"
        FAILURES=$((FAILURES + 1))
        return
    fi
    CHECKED=$((CHECKED + 1))
}

# Check known files
verify_file /workspace/CMakeLists.txt
verify_file /workspace/README.md
verify_file /workspace/CMakePresets.json
verify_file /workspace/context-transfer-engine/adapter/libfuse/fuse_cte.h
verify_file /workspace/context-transfer-engine/adapter/libfuse/fuse_cte.cc
verify_file /workspace/context-transfer-engine/adapter/libfuse/CMakeLists.txt

# Check some random source files
for f in $(find /workspace/context-transfer-engine/core/src -name '*.cc' -maxdepth 1 2>/dev/null | head -5); do
    verify_file "$f"
done

for f in $(find /workspace/context-runtime/src -name '*.cc' -maxdepth 1 2>/dev/null | head -5); do
    verify_file "$f"
done

echo ""
if [ "$FAILURES" -eq 0 ]; then
    ok "All $CHECKED sampled files verified byte-for-byte"
else
    fail "$FAILURES of $((CHECKED + FAILURES)) files failed verification"
    exit 1
fi

# --- Summary -----------------------------------------------------------------

TOTAL_FILES=$(find "$MOUNT_POINT/workspace/" -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$MOUNT_POINT/workspace/" 2>/dev/null | cut -f1)

echo ""
echo "========================================="
echo "  Files copied:  $TOTAL_FILES"
echo "  Total size:    $TOTAL_SIZE"
echo "  Mount point:   $MOUNT_POINT/workspace/"
echo ""
echo "Browse with:  ls $MOUNT_POINT/workspace/"
echo "========================================="
