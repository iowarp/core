#==============================================================================
# IOWarp Core - Minimal Build Test Dockerfile
#==============================================================================
# This Dockerfile tests the minimalistic build configuration with only
# essential build dependencies and submodule-based libraries.
#
# Build Configuration:
# - Uses minimalist CMake preset (Release mode, no tests/benchmarks)
# - Only submodule dependencies: Boost, ZeroMQ, HDF5, cereal
# - Minimal system dependencies: cmake, gcc, make, git
#
# Build Command:
#   docker build -f docker/minimal.Dockerfile -t iowarp-minimal .
#
# Run Command:
#   docker run --rm iowarp-minimal
#==============================================================================

FROM ubuntu:22.04

#------------------------------------------------------------------------------
# Install Essential Build Dependencies
#------------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Build tools
    cmake \
    make \
    g++ \
    gcc \
    # Version control (for git submodules)
    git \
    # Required for Boost build
    python3 \
    # Required for ZeroMQ build
    pkg-config \
    libtool \
    autoconf \
    automake \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------------------------
# Set Working Directory
#------------------------------------------------------------------------------
WORKDIR /iowarp-core

#------------------------------------------------------------------------------
# Copy Source Code
#------------------------------------------------------------------------------
# Copy the entire source tree including submodules
COPY . .

#------------------------------------------------------------------------------
# Initialize Git Submodules
#------------------------------------------------------------------------------
# Initialize and update all submodules (Boost, ZeroMQ, HDF5, etc.)
RUN git submodule update --init --recursive

#------------------------------------------------------------------------------
# Configure Build with Minimalist Preset
#------------------------------------------------------------------------------
# Use the minimalist preset which:
# - Builds in Release mode
# - Disables tests and benchmarks
# - Uses only submodule dependencies (no system libraries)
# - Minimal feature set (no MPI, ELF, Python, etc.)
RUN cmake --preset=minimalist

#------------------------------------------------------------------------------
# Build
#------------------------------------------------------------------------------
# Build with parallel jobs for faster compilation
RUN cmake --build build -j$(nproc)

#------------------------------------------------------------------------------
# Verify Build Success
#------------------------------------------------------------------------------
# List built libraries to verify successful build
RUN echo "=== Build Successful ===" && \
    echo "Built libraries:" && \
    ls -lh build/bin/lib*.so 2>/dev/null || true && \
    echo "Built executables:" && \
    ls -lh build/bin/wrp_* 2>/dev/null || true

#------------------------------------------------------------------------------
# Default Command
#------------------------------------------------------------------------------
# Show build information when container runs
CMD ["sh", "-c", "echo 'IOWarp Core - Minimalist Build Container' && \
     echo 'Build configuration: Release mode, no tests/benchmarks' && \
     echo 'Dependencies: Boost, ZeroMQ, HDF5, cereal (from submodules)' && \
     echo '' && \
     echo 'Built libraries:' && \
     ls -lh /iowarp-core/build/bin/lib*.so 2>/dev/null || echo 'No libraries found' && \
     echo '' && \
     echo 'Built executables:' && \
     ls -lh /iowarp-core/build/bin/wrp_* 2>/dev/null || echo 'No executables found'"]
