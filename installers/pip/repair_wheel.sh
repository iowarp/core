#!/bin/bash
# repair_wheel.sh - Fix RPATHs and repack wheel for cibuildwheel
#
# Called by CIBW_REPAIR_WHEEL_COMMAND_LINUX in place of auditwheel repair.
# auditwheel doesn't work well with our bundled shared libraries, so we
# fix RPATHs to use $ORIGIN and repack preserving Unix permissions.
#
# Usage (called by cibuildwheel):
#   bash repair_wheel.sh {dest_dir} {wheel}

set -euo pipefail

DEST_DIR="${1:?Usage: repair_wheel.sh <dest_dir> <wheel>}"
WHEEL="${2:?Usage: repair_wheel.sh <dest_dir> <wheel>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Repairing wheel: $(basename "$WHEEL") ==="

WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

# Unpack
python3 -m zipfile -e "$WHEEL" "$WORK_DIR/unpack/"

# Make binaries executable (zip loses permission bits)
chmod +x "$WORK_DIR"/unpack/iowarp_core/bin/* 2>/dev/null || true

# Fix RPATHs
echo "Fixing RPATHs..."

# Shared libraries: find each other via $ORIGIN
for so in "$WORK_DIR"/unpack/iowarp_core/lib/*.so*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
done

# Python extensions: find libs via $ORIGIN/../lib
for so in "$WORK_DIR"/unpack/iowarp_core/ext/*.so*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN/../lib' "$so" 2>/dev/null || true
done

# CLI binaries: find libs via $ORIGIN/../lib
for bin in "$WORK_DIR"/unpack/iowarp_core/bin/*; do
    [ -f "$bin" ] || continue
    patchelf --set-rpath '$ORIGIN/../lib' "$bin" 2>/dev/null || true
done

# Repack preserving permissions
WHEEL_NAME=$(basename "$WHEEL")
python3 -c "
import os, sys, zipfile
base = sys.argv[1]
out  = sys.argv[2]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(base):
        for f in files:
            full = os.path.join(root, f)
            arc = os.path.relpath(full, base)
            info = zipfile.ZipInfo(arc)
            st = os.stat(full)
            info.external_attr = (st.st_mode & 0xFFFF) << 16
            with open(full, 'rb') as fh:
                zf.writestr(info, fh.read())
" "$WORK_DIR/unpack" "$DEST_DIR/$WHEEL_NAME"

echo "=== Repaired wheel: $DEST_DIR/$WHEEL_NAME ==="
