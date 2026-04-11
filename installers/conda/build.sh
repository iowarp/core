#!/bin/bash
set -ex

PRESET="${IOWARP_PRESET:-release}"

# Clean any stale build directory (preset uses ${sourceDir}/build)
rm -rf build

# Suppress GCC false positive warnings from aggressive inlining
export CXXFLAGS="${CXXFLAGS:-} -Wno-array-bounds -Wno-maybe-uninitialized -Wno-stringop-overflow"

cmake --preset="${PRESET}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_FIND_ROOT_PATH="${PREFIX}" \
    -DWRP_CORE_ENABLE_CONDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build --parallel "${CPU_COUNT}"
cmake --install build
