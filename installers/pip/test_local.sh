#!/bin/bash
# test_local.sh — run the cibuildwheel pip build locally using Docker.
# Must be run from the repo root, or it will navigate there automatically.
#
# Usage:
#   ./installers/pip/test_local.sh [BUILD_TARGET]
#
# BUILD_TARGET defaults to a single x86_64 wheel to keep things fast.
# Examples:
#   ./installers/pip/test_local.sh                          # cp312-manylinux_x86_64
#   ./installers/pip/test_local.sh cp311-manylinux_x86_64
#   ./installers/pip/test_local.sh "cp310-manylinux_x86_64 cp311-manylinux_x86_64"
#
# Requirements: Docker must be running, pip install cibuildwheel

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

BUILD_TARGET="${1:-cp312-manylinux_x86_64}"

echo "=== Building: $BUILD_TARGET ==="
echo "=== Repo root: $REPO_ROOT ==="

CIBW_BUILD="$BUILD_TARGET" \
CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_34 \
CIBW_MANYLINUX_AARCH64_IMAGE=manylinux_2_34 \
CIBW_BEFORE_ALL="yum install -y curl patchelf libaio-devel && bash {project}/installers/pip/build_deps_manylinux.sh" \
CIBW_ENVIRONMENT="CMAKE_PREFIX_PATH=/usr/local PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig CFLAGS=-march=x86-64 CXXFLAGS=-march=x86-64" \
CIBW_REPAIR_WHEEL_COMMAND_LINUX="bash {project}/installers/pip/repair_wheel.sh {dest_dir} {wheel}" \
CIBW_TEST_COMMAND="python -c \"import iowarp_core; print('Version:', iowarp_core.get_version())\"" \
cibuildwheel --platform linux --output-dir installers/pip/wheelhouse .

echo "=== Done. Wheels in installers/pip/wheelhouse/ ==="
ls -lh installers/pip/wheelhouse/
