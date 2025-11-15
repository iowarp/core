
# Hermes Shared Memory (CTE)

[![IoWarp](https://img.shields.io/badge/IoWarp-GitHub-blue.svg)](http://github.com/iowarp)
[![GRC](https://img.shields.io/badge/GRC-Website-blue.svg)](https://grc.iit.edu/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Compatible-green.svg)](https://developer.nvidia.com/cuda-zone)
[![ROCm](https://img.shields.io/badge/ROCm-Compatible-red.svg)](https://rocmdocs.amd.com/)

A high-performance shared memory library containing data structures and synchronization primitives compatible with shared memory, CUDA, and ROCm.

## Dependencies

### System Requirements
- CMake >= 3.10
- C++17 compatible compiler
- Optional: CUDA toolkit (for GPU support)
- Optional: ROCm (for AMD GPU support)
- Optional: MPI, ZeroMQ, Thallium (for distributed computing)

Our docker container has all dependencies installed for you.
```bash
docker pull iowarp/iowarp-build:latest
```

## Building Manually

```bash
git clone https://github.com/grc-iit/cte-hermes-shm.git
cd cte-hermes-shm
mkdir build
cd build
cmake ../ -DHSHM_ENABLE_CUDA=OFF -DHSHM_ENABLE_ROCM=OFF
make -j8
```

## CMake Integration

HermesShm provides modular CMake targets for flexible dependency management. Link only what you need.

### Core Library (CPU-Only)
```cmake
find_package(HermesShm CONFIG REQUIRED)
target_link_libraries(your_target hshm::cxx)
```

### Modular Dependency Targets

HermesShm provides fine-grained modular targets for optional dependencies:

```cmake
find_package(HermesShm CONFIG REQUIRED)

target_link_libraries(your_target
  hshm::cxx              # Core library (required)
  hshm::configure        # YAML configuration parsing (instead of yaml-cpp directly)
  hshm::serialize        # Object serialization (instead of cereal directly)
  hshm::interceptor      # ELF interception for adapters
  hshm::lightbeam        # Network transport (ZMQ, libfabric, Thallium)
  hshm::thread_all       # Threading support (pthread, OpenMP)
  hshm::mpi              # MPI support (use only where needed)
  hshm::compress         # Compression utilities
  hshm::encrypt          # Encryption utilities
)
```

**Key Guidelines:**
- Always link `hshm::cxx` as the base
- Use `hshm::configure` instead of linking to `yaml-cpp` directly
- Use `hshm::serialize` instead of linking to `cereal` directly
- Link only the modular targets you actually need
- Each modular target includes appropriate compile definitions

### GPU Support

**CUDA Version:**
```cmake
find_package(HermesShm CONFIG REQUIRED)
target_link_libraries(your_target hshm::cudacxx)
```

**ROCm Version:**
```cmake
find_package(HermesShm CONFIG REQUIRED)
target_link_libraries(your_target hshm::rocmcxx_gpu)
```

## Tests

To run the tests, do the following:
```
ctest
```

To run the MPSC queue tests, do the following:
```
ctest -VV -R test_mpsc_queue_mpi
```

## Project Structure

- `include/hermes_shm/` - Public API headers organized by functional modules
  - `data_structures/` - Shared memory compatible data structures
    - `ipc/` - IPC-safe containers (vector, list, unordered_map, ring queues, etc.)
    - `internal/` - Internal implementation details
    - `serialization/` - Serialization utilities
  - `memory/` - Memory management subsystem
    - `allocator/` - Allocator implementations (malloc, GPU stack, heap allocators)
    - `backend/` - Memory backend implementations
  - `thread/` - Threading and synchronization primitives
    - `lock/` - Lock implementations
    - `thread_model/` - Thread model abstractions
  - `util/` - Utility functions
    - `compress/` - Compression utilities
    - `encrypt/` - Encryption utilities
  - `lightbeam/` - Networking layer (ZMQ transport, networking utilities)
  - `types/` - Type definitions and utilities
  - `introspect/` - System introspection capabilities
- `src/` - Core library implementation files
