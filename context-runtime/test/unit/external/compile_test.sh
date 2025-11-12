#!/bin/bash

# Script to compile the external ChiMod test with proper environment setup
# This script loads the required modules and compiles the test

set -e  # Exit on any error

echo "Setting up environment for external ChiMod test compilation..."

# Load required modules
echo "Loading iowarp-runtime module..."
module load iowarp-runtime || {
    echo "Warning: Failed to load iowarp-runtime module - continuing anyway"
}

echo "Loading iowarp spack module..."
spack load iowarp || {
    echo "Warning: Failed to load iowarp spack module - continuing anyway"
}

echo "Environment setup complete."
echo ""

# Show current directory
echo "Current directory: $(pwd)"
echo ""

# Create build directory and compile
echo "Creating build directory..."
rm -rf build
mkdir -p build
cd build

echo "Configuring CMake..."
cmake .. || {
    echo "CMake configuration failed!"
    exit 1
}

echo "Building test..."
make || {
    echo "Build failed!"
    exit 1
}

echo ""
echo "Build completed successfully!"
echo "Test executable created: $(pwd)/test_external_chimod"
echo ""

# Optionally run the test
echo "Running external ChiMod test..."
./test_external_chimod || {
    echo "Test execution failed!"
    exit 1
}

echo ""
echo "External ChiMod test compilation and execution completed successfully!"