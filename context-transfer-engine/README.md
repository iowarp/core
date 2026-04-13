# The Content Transfer Engine: Hermes

The CTE is a heterogeneous-aware, multi-tiered, dynamic, and distributed I/O buffering system designed to accelerate I/O for HPC and data-intensive workloads.


[![Project Site](https://img.shields.io/badge/Project-Site-blue)](https://grc.iit.edu/research/projects/iowarp)
[![Documentation](https://img.shields.io/badge/Docs-Hub-green)](https://grc.iit.edu/docs/category/iowarp)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](LICENSE)
![Build](https://github.com/HDFGroup/iowarp/workflows/GitHub%20Actions/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/HDFGroup/iowarp/badge.svg?branch=master)](https://coveralls.io/github/HDFGroup/iowarp?branch=master)

## Overview

iowarp provides a programmable buffering layer across memory/storage tiers and supports multiple I/O pathways via adapters. It integrates with HPC runtimes and workflows to improve throughput, latency, and predictability.


## Build instructions

### Dependencies

Our docker container has all dependencies installed for you.
```bash
docker pull iowarp/iowarp-build:latest
```

### Build with CMake

```bash
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release 
make -j8
make install
```

Tip: run `ccmake ..` (or `cmake-gui`) to browse available CMake options.

## Testing

- CTest unit tests (after building):

```bash
cd build
ctest -VV
```

## FUSE Adapter Performance

The CTE FUSE adapter (`adapter/libfuse/`) provides filesystem-compatible access 
to CTE storage but has important performance characteristics:

### Performance Characteristics

| Workload Type | Expected Throughput | Use Case |
|--------------|---------------------|----------|
| Small files (< 1MB) | 10-50 MB/s | ✅ Recommended |
| Large sequential writes | 9.3 MB/s (default) → 500 MB/s (tuned) | ⚠️ Use POSIX interceptor instead |
| Random I/O | < 5 MB/s | ❌ Not recommended |

### The 4KB Page Problem

The FUSE adapter splits writes into 4KB pages, causing **130x slowdown** compared to 
direct filesystem access. A 10MB write creates 2,560 CTE operations.

**Quick Fix:** Set page size to 1MB
```bash
export FUSE_CTE_PAGE_SIZE=1048576
```

### When to Use FUSE vs POSIX Interceptor

**Use FUSE for:**
- Development and debugging
- Interactive filesystem exploration
- Legacy applications that require filesystem paths
- Small configuration files

**Use POSIX Interceptor for:**
- Performance-critical workloads
- Large file I/O (> 10MB)
- Production HPC applications
- Batch processing pipelines

```bash
# FUSE (convenience, slower)
mount -t fuse.wrp_cte /mnt/cte
./my_app --output /mnt/cte/data.bin

# POSIX Interceptor (performance, faster)
LD_PRELOAD=/usr/lib/libcte_posix.so ./my_app --output cte://data.bin
```

### Tuning Guide

See [adapter/libfuse/FUSE_PERFORMANCE.md](adapter/libfuse/FUSE_PERFORMANCE.md) for 
detailed performance tuning instructions and the 3-phase improvement roadmap.

### Benchmarks

| Interface | 10MB Write | 100MB Write | 1GB Write |
|-----------|------------|-------------|-----------|
| Direct filesystem | 1.2 GB/s | 1.2 GB/s | 1.2 GB/s |
| FUSE (default 4KB) | 9.3 MB/s | 9.3 MB/s | 9.3 MB/s |
| FUSE (tuned 1MB) | 100-200 MB/s | 200-500 MB/s | 200-500 MB/s |
| POSIX Interceptor | 1+ GB/s | 1+ GB/s | 1+ GB/s |

**Recommendation:** For production workloads, use the POSIX interceptor or native CTE API.

## Development

- Linting: we follow the Google C++ Style Guide.
    - Run `make lint` (wraps `ci/lint.sh` which uses `cpplint`).
    - Install `cpplint` via `pip install cpplint` if needed.

## Contributing

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Submit PRs with clear descriptions and tests when possible. The CI will validate style and builds.

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.
