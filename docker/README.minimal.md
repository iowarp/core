# IOWarp Core - Minimal Docker Build

This directory contains the minimal Docker configuration for testing the IOWarp Core minimalistic build.

## Overview

The `minimal.Dockerfile` provides a containerized build environment that:

- Uses only essential build dependencies (cmake, gcc, make, git)
- Builds with the **minimalist** CMake preset (Release mode)
- Uses **only submodule dependencies** (no system libraries except build tools)
- Disables tests, benchmarks, and optional features
- Validates that the project can be built with minimal external dependencies

## Dependencies

### Build Tools (from Ubuntu 22.04)
- `cmake` - Build system generator
- `make` - Build automation
- `g++` / `gcc` - C/C++ compilers
- `git` - Version control (for submodule initialization)
- `python3` - Required for Boost build
- `pkg-config`, `libtool`, `autoconf`, `automake` - Required for ZeroMQ build

### Libraries (from Git Submodules)
All libraries are built from source using git submodules:
- **Boost** - Required components: fiber, context, system
- **ZeroMQ** (libzmq) - Messaging library
- **HDF5** - Hierarchical data format (minimal build: C library only)
- **cereal** - C++ serialization library
- **nanobind** - Python bindings (not used in minimal build)
- **Catch2** - Testing framework (not used in minimal build)
- **yaml-cpp** - YAML parser

## Build Configuration

The minimal build uses the following CMake options (from `minimalist` preset):

```cmake
CMAKE_BUILD_TYPE=Release
WRP_CORE_ENABLE_TESTS=OFF
WRP_CORE_ENABLE_BENCHMARKS=OFF
WRP_CORE_ENABLE_PYTHON=OFF
WRP_CORE_ENABLE_MPI=OFF
WRP_CORE_ENABLE_ELF=OFF
WRP_CORE_ENABLE_RPATH=OFF
WRP_CORE_ENABLE_ZMQ=ON
WRP_CORE_ENABLE_HDF5=ON
WRP_CORE_ENABLE_CEREAL=ON
```

## Building the Docker Image

From the repository root:

```bash
docker build -f docker/minimal.Dockerfile -t iowarp-minimal .
```

## Running the Container

To see build information:

```bash
docker run --rm iowarp-minimal
```

Expected output:
```
IOWarp Core - Minimalist Build Container
Build configuration: Release mode, no tests/benchmarks
Dependencies: Boost, ZeroMQ, HDF5, cereal (from submodules)

Built libraries:
[List of shared libraries]

Built executables:
[List of executables]
```

## Size Optimization

The minimal build significantly reduces:
- **Build time**: No tests, benchmarks, or optional features
- **Dependencies**: Only essential submodules are built
- **Binary size**: Release mode with optimization
- **Attack surface**: Minimal feature set reduces potential vulnerabilities

## Use Cases

This minimal configuration is ideal for:
- **Production deployments** - Minimal dependencies and attack surface
- **CI/CD validation** - Quick smoke test that build works without system libraries
- **Dependency auditing** - Clear view of what's actually required
- **Embedded systems** - Minimal footprint for resource-constrained environments

## Differences from Full Build

The minimal build **excludes**:
- Unit tests and test frameworks
- Benchmarks
- Python bindings
- MPI support
- ELF/adapter support
- RPATH embedding
- Development tools (ASAN, coverage, doxygen)
- Accelerator support (CUDA, ROCm)

## Troubleshooting

### Submodule Issues
If you see errors about missing submodules:
```bash
# Ensure submodules are initialized before building
git submodule update --init --recursive
```

### Build Failures
Check the build output in the Docker logs:
```bash
docker build -f docker/minimal.Dockerfile -t iowarp-minimal . 2>&1 | tee build.log
```

### Container Won't Start
Verify the image was built successfully:
```bash
docker images | grep iowarp-minimal
```

## Related Files

- `minimal.Dockerfile` - Dockerfile for minimal build
- `.dockerignore` - Excludes build artifacts from Docker context
- `/CMakePresets.json` - Contains the `minimalist` preset definition
- `/CMakeLists.txt` - Root build configuration with all options

## Comparison with Other Presets

| Preset | Build Type | Tests | Dependencies | Use Case |
|--------|-----------|-------|--------------|----------|
| **minimalist** | Release | OFF | Submodules only | Production, minimal footprint |
| **debug** | Debug | ON | System + submodules | Development, debugging |
| **release** | Release | ON | System + submodules | Release validation, CI/CD |

## Future Enhancements

Potential improvements for the minimal build:
- Multi-stage build to reduce final image size
- Alpine Linux base for even smaller footprint
- Static linking to eliminate runtime dependencies
- Distroless final image for production security
