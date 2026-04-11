#!/bin/bash
# Test script for CTE Rust bindings
# Sets environment variables from CMake configuration before running cargo test

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine the build directory
# Try to find it relative to the wrapper directory
WRAPPER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$WRAPPER_DIR")"

# Look for build directory in common locations
if [ -d "$PROJECT_ROOT/build" ]; then
	BUILD_DIR="$PROJECT_ROOT/build"
elif [ -d "$PROJECT_ROOT/../build" ]; then
	BUILD_DIR="$PROJECT_ROOT/../build"
elif [ -d "$PROJECT_ROOT/../../build" ]; then
	BUILD_DIR="$PROJECT_ROOT/../../build"
else
	echo "Error: Could not find build directory"
	echo "Please build the project first with: cmake --preset=debug && cmake --build"
	exit 1
fi

echo "Using build directory: $BUILD_DIR"

# Set environment variables from CMake configuration
# These match the values set in wrapper/CMakeLists.txt via Corrosion

# Calculate paths from the build directory
HSHM_ROOT="$PROJECT_ROOT/context-transport-primitives"
CHIMAERA_ROOT="$PROJECT_ROOT/context-runtime"
CTE_ROOT="$WRAPPER_DIR"
CHIMODS_ROOT="$PROJECT_ROOT/context-runtime/modules"

# Get cereal include directory - try to find from build configuration
CEREAL_INCLUDE_DIR=""
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
	CEREAL_INCLUDE_DIR=$(grep -m1 "cereal_DIR:" "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2 | xargs dirname | xargs dirname)/include
fi

# Fallback if not found in cache
if [ -z "$CEREAL_INCLUDE_DIR" ] && [ -d "/usr/local/include" ]; then
	CEREAL_INCLUDE_DIR="/usr/local/include"
fi

# Export environment variables (same as Corrosion sets in parent CMakeLists.txt)
export IOWARP_INCLUDE_DIR="$HSHM_ROOT/include"
export IOWARP_EXTRA_INCLUDES="$CHIMAERA_ROOT/include:$CTE_ROOT/core/include:$CHIMODS_ROOT/admin/include:$CHIMODS_ROOT/bdev/include:$CEREAL_INCLUDE_DIR"
export IOWARP_LIB_DIR="$BUILD_DIR/bin"
export IOWARP_ZMQ_LIBS="${IOWARP_ZMQ_LIBS:-zmq}"
export IOWARP_ZMQ_LIB_DIRS="${IOWARP_ZMQ_LIB_DIRS:-/usr/local/lib}"

echo "Environment variables set:"
echo "  IOWARP_INCLUDE_DIR=$IOWARP_INCLUDE_DIR"
echo "  IOWARP_EXTRA_INCLUDES=$IOWARP_EXTRA_INCLUDES"
echo "  IOWARP_LIB_DIR=$IOWARP_LIB_DIR"
echo "  IOWARP_ZMQ_LIBS=$IOWARP_ZMQ_LIBS"
echo "  IOWARP_ZMQ_LIB_DIRS=$IOWARP_ZMQ_LIBS"

cd "$SCRIPT_DIR"

echo "Running unit tests..."
cargo test --lib

echo "Running integration tests (marked with #[ignore])..."
cargo test -- --ignored
