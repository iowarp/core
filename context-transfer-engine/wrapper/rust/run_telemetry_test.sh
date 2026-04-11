#!/bin/bash
# Run telemetry capture example with proper environment

set -e

# Determine IOWarp root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOWARP_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${IOWARP_ROOT}/build"

# Find cereal from spack
CEREAL_DIR=$(find "${HOME}/spack/opt/spack" -path "*/cereal*/lib*/cmake/cereal" -type d 2>/dev/null | head -1)
if [ -z "${CEREAL_DIR}" ]; then
	echo "Error: Could not find cereal installation in spack"
	exit 1
fi
CEREAL_INCLUDE_DIR="$(cd "${CEREAL_DIR}/../../../include" && pwd)"

# Find chimaera modules
CHIMODS_ROOT="${IOWARP_ROOT}/context-runtime/modules"
CHIMAERA_ROOT="${IOWARP_ROOT}/context-runtime"
CTE_ROOT="${IOWARP_ROOT}/context-transfer-engine"
HSHM_ROOT="${IOWARP_ROOT}/context-transport-primitives"

# Export environment for build.rs
export IOWARP_INCLUDE_DIR="${HSHM_ROOT}/include"
export IOWARP_EXTRA_INCLUDES="${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
export IOWARP_LIB_DIR="${BUILD_DIR}/bin"

# Set library path for runtime
export LD_LIBRARY_PATH="${IOWARP_LIB_DIR}:${LD_LIBRARY_PATH}"

echo "=== IOWarp Telemetry Test ==="
echo "IOWARP_INCLUDE_DIR: ${IOWARP_INCLUDE_DIR}"
echo "IOWARP_EXTRA_INCLUDES: ${IOWARP_EXTRA_INCLUDES}"
echo "IOWARP_LIB_DIR: ${IOWARP_LIB_DIR}"
echo ""

# Build and run with embedded runtime
CHI_WITH_RUNTIME=1 cargo run --example telemetry_capture "$@"
