#!/bin/bash
# repair_wheel.sh - Called by cibuildwheel's CIBW_REPAIR_WHEEL_COMMAND_LINUX.
# Usage: bash repair_wheel.sh {dest_dir} {wheel}
set -euo pipefail

DEST_DIR="$1"
WHEEL="$2"

# Add all build output directories so auditwheel can locate project .so files.
BUILD_LIBS=$(find /project/build -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')

# Add gcc-toolset-12 runtime libs so auditwheel can find and bundle the GCC 12
# libstdc++/libgcc_s (the project is compiled with GCC 12 for manylinux_2_28 ABI
# compatibility — GCC 14 generates GLIBC_2.38 symbols that exceed the policy).
GCC12_LIBS=""
for d in /opt/rh/gcc-toolset-12/root/usr/lib64 /opt/rh/gcc-toolset-12/root/usr/lib; do
    [ -d "$d" ] && GCC12_LIBS="${GCC12_LIBS}${d}:"
done

# Also add the custom deps prefix (yaml-cpp, zeromq, libsodium, etc.)
export LD_LIBRARY_PATH="${BUILD_LIBS}${GCC12_LIBS}/usr/local/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Show which specific symbols require newer GLIBC versions (useful for debugging)
echo "--- Symbols requiring GLIBC >= 2.29 (direct .so inspection) ---"
python3 -c "
import zipfile, subprocess, sys, os, re

whl = sys.argv[1]
found = False
with zipfile.ZipFile(whl) as z:
    names = [n for n in z.namelist() if n.endswith('.so') or '.so.' in n]
    print(f'  .so files in wheel: {names}', flush=True)
    for name in names:
        data = z.read(name)
        with open('/tmp/_inspect.so', 'wb') as f:
            f.write(data)
        # Use nm -D to get dynamic symbols
        out = subprocess.run(['nm', '-D', '--defined-only', '/tmp/_inspect.so'],
                             capture_output=True, text=True)
        undef = subprocess.run(['nm', '-D', '--undefined-only', '/tmp/_inspect.so'],
                               capture_output=True, text=True)
        for line in (out.stdout + undef.stdout).splitlines():
            if 'GLIBC_2.' in line:
                m = re.search(r'GLIBC_2\.(\d+)', line)
                if m and int(m.group(1)) >= 29:
                    print(f'  {os.path.basename(name)}: {line.strip()}')
                    found = True
if not found:
    print('  (none found - symbols may be in transitive deps)')
" "$WHEEL" 2>&1 || true

# Also check .so files directly in the build dir for GLIBC_2.38
echo "--- Build dir GLIBC >= 2.29 symbols ---"
for so in $(find /project/build -name "*.so" 2>/dev/null); do
    result=$(nm -D --undefined-only "$so" 2>/dev/null | grep 'GLIBC_2\.' | grep -v 'GLIBC_2\.[012][^0-9]' | head -3)
    if [ -n "$result" ]; then
        echo "  $(basename $so):"
        echo "$result" | head -5 | sed 's/^/    /'
    fi
done

# Detect the highest-priority manylinux tag this wheel actually qualifies for.
echo "--- auditwheel show ---"
auditwheel show "$WHEEL" 2>&1 || true

# Pick the most permissive qualifying manylinux tag so we don't fail with
# "too-recent versioned symbols" when the image's default policy is too strict.
PLAT=$(auditwheel show "$WHEEL" 2>&1 \
  | grep -oP 'manylinux_\d+_\d+_\w+' \
  | sort -t_ -k2,2n -k3,3n \
  | tail -1)

if [ -n "$PLAT" ]; then
    echo "Repairing to platform: $PLAT"
    auditwheel repair --plat "$PLAT" -w "$DEST_DIR" "$WHEEL"
else
    echo "Could not detect platform, using default"
    auditwheel repair -w "$DEST_DIR" "$WHEEL"
fi
