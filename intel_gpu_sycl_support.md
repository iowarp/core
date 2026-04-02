# Intel GPU SYCL Support for clio-core on Aurora

## Overview

Added Intel GPU (oneAPI/SYCL) support to clio-core, enabling GPU testing on
Aurora's Intel Data Center GPU Max 1550 (Ponte Vecchio / PVC). The
implementation follows the existing CUDA/ROCm pattern and uses `#ifdef` guards
throughout so no existing code is affected.

## New Script: `~/bin/core_gpu`

Submits a `ctest -D Experimental` PBS job using the Intel oneAPI compiler
(`icpx -fsycl`) targeting Aurora's Intel GPU.

```bash
~/bin/core_gpu          # submit job
qstat -x <JOBID>        # monitor
# output: ~/clio-core/ctest_experimental_gpu.out
```

PBS configuration mirrors `~/bin/core_icx`:
- Queue: `debug`, Account: `gpu_hack`
- Modules: `gcc/13.4.0`, `oneapi/release/2025.3.1`
- Conda env: `iowarp`
- CDash: site=`aurora`, buildname=`sycl/r/gpu`
- Build dir: `build_gpu` (separate from `build` and `build_icx`)
- CTest filter: `-L gpu -LE docker`

## CMake Changes

### New option (`CMakeLists.txt`)

```cmake
option(WRP_CORE_ENABLE_SYCL "Enable Intel GPU support via SYCL/oneAPI (icpx -fsycl)" OFF)
set(HSHM_ENABLE_SYCL ${WRP_CORE_ENABLE_SYCL} CACHE BOOL "..." FORCE)
```

### New preset (`CMakePresets.json`)

```json
{
    "name": "sycl-debug",
    "binaryDir": "${sourceDir}/build_gpu",
    "cacheVariables": {
        "WRP_CORE_ENABLE_SYCL": "ON",
        "WRP_CORE_ENABLE_CAE": "OFF",
        "WRP_CORE_ENABLE_CEE": "OFF",
        "WRP_CTE_ENABLE_COMPRESS": "OFF",
        ...
    }
}
```

### New macros (`cmake/IowarpCoreCommon.cmake`)

```cmake
# Sets CXX standard and SYCL AOT cache variables (no global flags).
macro(wrp_core_enable_sycl CXX_STANDARD)

# Applies -fsycl flags to a specific target only.
function(wrp_core_apply_sycl_flags target)
```

> **Important:** `-fsycl` is applied **per-target**, not globally.
> Setting it globally breaks precompiled headers via `clang-offload-bundler`.

### GPU subdirectory inclusion (`context-transport-primitives/test/unit/CMakeLists.txt`)

```cmake
# Before
if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
    add_subdirectory(gpu)
endif()

# After
if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM OR WRP_CORE_ENABLE_SYCL)
    add_subdirectory(gpu)
endif()
```

### Disabled-component CMake bug fix (`CMakeLists.txt`)

The `_chimaera_lock_tests_recursive` function previously called
`get_directory_property(DIRECTORY ...)` on directories that were never added
via `add_subdirectory` (e.g., CAE/CEE when disabled), causing a fatal CMake
error. Fixed by only iterating over enabled components:

```cmake
set(_lock_dirs context-runtime context-transfer-engine)
if(WRP_CORE_ENABLE_CAE)
    list(APPEND _lock_dirs context-assimilation-engine)
endif()
if(WRP_CORE_ENABLE_CEE)
    list(APPEND _lock_dirs context-exploration-engine)
endif()
```

## Header Changes

### `constants/macros.h`

Added SYCL compiler detection after the existing ROCm block:

```cpp
#if HSHM_ENABLE_SYCL && defined(SYCL_LANGUAGE_VERSION)
#define HSHM_IS_SYCL_COMPILER 1
#else
#define HSHM_IS_SYCL_COMPILER 0
#endif

#if HSHM_IS_SYCL_COMPILER
#include <sycl/sycl.hpp>
#endif
```

Added `HSHM_ENABLE_GPU` combining all three backends:

```cpp
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM || HSHM_ENABLE_SYCL
#define HSHM_ENABLE_GPU 1
#endif
```

> **Critical:** `HSHM_IS_SYCL_COMPILER` is **not** included in
> `HSHM_IS_GPU_COMPILER`. That macro gates `__device__`/`__host__`
> attributes, which are CUDA/ROCm-only. SYCL uses standard C++ lambdas
> and does not support these keywords.

### `util/gpu_api.h`

Added SYCL implementations for all `GpuApi` methods via `#elif HSHM_ENABLE_SYCL`:

| Method | SYCL implementation |
|--------|---------------------|
| `SetDevice` | no-op (device selected at queue construction) |
| `GetDeviceCount` | `sycl::platform::get_platforms()` iteration |
| `Synchronize` | `SyclQueue().wait_and_throw()` |
| `MallocManaged` | `sycl::malloc_shared(size, SyclQueue())` |
| `RegisterHostMemory` | no-op (USM host memory needs no registration) |
| `UnregisterHostMemory` | no-op |
| `Memcpy` | `SyclQueue().memcpy(...).wait_and_throw()` |
| `Free` | `sycl::free(ptr, SyclQueue())` |

Added a static queue helper:

```cpp
#if HSHM_ENABLE_SYCL
  static sycl::queue &SyclQueue() {
    static sycl::queue q{sycl::gpu_selector_v};
    return q;
  }
#endif
```

### `memory/backend/gpu_shm_mmap.h`

Extended the include guard:

```cpp
// Before
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

// After
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM || HSHM_ENABLE_SYCL
```

Note: `GpuApi::RegisterHostMemory` is a no-op for SYCL, so `GpuShmMmap`
compiles but POSIX-shm memory is **not** accessible from SYCL kernels
(not a USM allocation). Use `sycl::malloc_shared` for GPU-accessible buffers.

## New SYCL Test

### `context-transport-primitives/test/unit/gpu/test_gpu_sycl.cc`

Tests Intel GPU kernels using SYCL USM. The entire test body is guarded with
`#ifdef HSHM_ENABLE_SYCL` so it compiles as a no-op in non-SYCL builds.

Two `parallel_for` tests:
1. **ParallelFill** — fill 256 elements with their index, verify on CPU
2. **ParallelScaleAndAccumulate** — scale 512 elements by 2, verify on CPU

> **Important:** The `sycl::queue` is declared `static` to ensure it is
> constructed only once per process. Catch2 re-executes the `TEST_CASE` body
> for each `SECTION`, and constructing a new `sycl::queue{sycl::gpu_selector_v}`
> on each re-entry causes a SEGFAULT on Intel GPU (PVC).

### `context-transport-primitives/test/unit/gpu/CMakeLists.txt`

```cmake
if(WRP_CORE_ENABLE_SYCL)
  add_executable(test_gpu_sycl test_gpu_sycl.cc)
  target_compile_definitions(test_gpu_sycl PRIVATE HSHM_ENABLE_SYCL=1)
  wrp_core_apply_sycl_flags(test_gpu_sycl)
  target_link_libraries(test_gpu_sycl Catch2::Catch2WithMain)
  add_test(NAME test_gpu_sycl COMMAND test_gpu_sycl)
  set_tests_properties(test_gpu_sycl PROPERTIES LABELS "gpu;sycl")
endif()
```

## Test Results

Verified on Aurora compute node `x4418c6s1b0n0` (PBS job 8409833):

```
Device: Intel(R) Data Center GPU Max 1550
Compiler: Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2

100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.24 sec
```

Results submitted to CDash at `my.cdash.org` under project `HERMES`,
site `aurora`, buildname `sycl/r/gpu`.

## Lessons Learned

| Issue | Root cause | Fix |
|-------|-----------|-----|
| PCH build failure (`clang-offload-bundler: invalid file type 'pchi'`) | `-fsycl` applied globally via `CMAKE_CXX_FLAGS` | Apply SYCL flags per-target with `wrp_core_apply_sycl_flags()` |
| `unknown type name '__device__'` compile error | `HSHM_IS_SYCL_COMPILER` included in `HSHM_IS_GPU_COMPILER`, enabling CUDA-style macros | Exclude SYCL from `HSHM_IS_GPU_COMPILER` |
| CMake configure error on `sycl-debug` preset | `_chimaera_lock_tests_recursive` called on disabled CAE/CEE directories | Only iterate enabled component directories |
| `test_gpu_sycl` target not found | GPU test subdir gated on CUDA/ROCm only | Add `OR WRP_CORE_ENABLE_SYCL` to the subdirectory guard |
| SEGFAULT on second Catch2 SECTION | `sycl::queue` reconstructed per section re-run | Declare queue as `static` |
