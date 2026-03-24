/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * BuddyAllocator GPU benchmark-mimic test
 *
 * Reproduces the exact memory access pattern of bench_gpu_runtime to isolate
 * the cudaErrorMisalignedAddress (error 716) bug introduced when
 * HSHM_DEFAULT_ALLOC_GPU_T was switched to hipc::BuddyAllocator.
 *
 * Key differences from test_buddy_alloc_gpu (context-transport-primitives):
 *
 *   1. Uses CHIMAERA_GPU_ORCHESTRATOR_INIT — the real macro from the
 *      benchmark.  This adjusts block_info.backend.data_ per block:
 *        data_ += blockIdx.x * per_block
 *      so each block's IpcManager sees a shifted base pointer. The existing
 *      single-block test never exercises this path.
 *
 *   2. Multi-block launch (kNumBlocks = 4, kThreadsPerBlock = 32).
 *
 *   3. Allocates 168-byte objects via CHI_IPC->AllocateBuffer(168), which
 *      is exactly what SendGpuForward does when creating a FutureShm.
 *
 *   4. Uses the chimaera_cxx_gpu shared library so HSHM_DEFAULT_ALLOC_GPU_T
 *      resolves to whatever the runtime was built with.  To test with
 *      BuddyAllocator, rebuild with -DHSHM_DEFAULT_ALLOC_GPU_T=hipc::BuddyAllocator.
 *
 * To reproduce the misaligned-address error:
 *   cmake -DHSHM_DEFAULT_ALLOC_GPU_T=hipc::BuddyAllocator ...
 *   cmake --build . --target test_buddy_benchmark_gpu
 *   compute-sanitizer --tool memcheck bin/test_buddy_benchmark_gpu
 *
 * The test verifies:
 *   - All kNumAllocs allocations per thread succeed (non-null)
 *   - Write+read-back of a (block,thread,alloc) magic value matches
 *   - No CUDA error on DeviceSynchronize
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/chimaera.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/memory/backend/gpu_shm_mmap.h>
#include <hermes_shm/util/gpu_api.h>

#include "../simple_test.h"

namespace {

// ─── Benchmark-mimic kernel ──────────────────────────────────────────────────

/**
 * Result codes written to d_results[global_tid]:
 *   >= 0   number of successful allocations
 *  -1      AllocateBuffer returned null (out of memory)
 *  -2      write/read-back mismatch (data corruption)
 */
__global__ void bench_mimic_kernel(
    chi::IpcManagerGpu gpu_info,
    int                num_blocks,   ///< = gridDim.x, passed explicitly for ORCHESTRATOR_INIT
    int                alloc_size,   ///< bytes per allocation (168 = sizeof FutureShm)
    int                num_allocs,   ///< allocations per thread
    int               *d_results) {

  // ── Exact CHIMAERA_GPU_ORCHESTRATOR_INIT pattern ─────────────────────────
  // Splits the backend per block (data_ += blockIdx.x * per_block), then
  // calls ClientInitGpu which placement-news one HSHM_DEFAULT_ALLOC_GPU_T
  // per thread and calls shm_init with the per-block sub-backend.
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  // ── Repeated AllocateBuffer calls ─────────────────────────────────────────
  // This is the hot path: SendGpuForward allocates a FutureShm here.
  for (int i = 0; i < num_allocs; ++i) {
    hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(alloc_size);

    if (buf.IsNull()) {
      d_results[global_tid] = -1;
      return;
    }

    // Write a unique 4-byte magic derived from block, thread, and alloc index.
    // Cast is safe: AllocateBuffer guarantees at least alloc_size bytes.
    int magic = (blockIdx.x << 20) | (threadIdx.x << 10) | (i & 0x3FF);
    *reinterpret_cast<int *>(buf.ptr_) = magic;

    // Immediate read-back: catches corruption or wrong pointer resolution.
    int read_back = *reinterpret_cast<int *>(buf.ptr_);
    if (read_back != magic) {
      d_results[global_tid] = -2;
      return;
    }
  }

  d_results[global_tid] = num_allocs;
}

// ─── Test fixture helper ─────────────────────────────────────────────────────

struct BenchMimicResult {
  int  code;       ///< raw result code from kernel
  bool passed() const { return code == kNumAllocs; }
  static constexpr int kNumAllocs = 100;
};

}  // namespace

// ─── Tests ───────────────────────────────────────────────────────────────────

/**
 * Single-block control: should pass with both ArenaAllocator and BuddyAllocator.
 * Equivalent to the existing test_buddy_alloc_gpu but routed through the real
 * IpcManager / AllocateBuffer path.
 */
TEST_CASE("BuddyBenchmark - single block (control)",
          "[gpu][buddy][benchmark]") {
  constexpr int    kNumBlocks      = 1;
  constexpr int    kThreadsPerBlock = 32;
  constexpr int    kNumAllocs      = 100;
  constexpr int    kAllocSize      = 168;  // sizeof(FutureShm)
  constexpr size_t kPerBlockBytes  = 10u * 1024u * 1024u;   // 10 MB
  constexpr size_t kBackendSize    = kNumBlocks * kPerBlockBytes;

  cudaDeviceSetLimit(cudaLimitStackSize, 4096);

  hipc::GpuShmMmap backend;
  hipc::MemoryBackendId backend_id(50, 0);
  REQUIRE(backend.shm_init(backend_id, kBackendSize,
                           "/test_buddy_bench_single", 0));

  chi::IpcManagerGpu gpu_info(backend, nullptr);

  constexpr int kTotal = kNumBlocks * kThreadsPerBlock;
  int *d_results = nullptr;
  cudaMallocHost(&d_results, kTotal * sizeof(int));
  REQUIRE(d_results != nullptr);
  memset(d_results, -1, kTotal * sizeof(int));

  bench_mimic_kernel<<<kNumBlocks, kThreadsPerBlock>>>(
      gpu_info, kNumBlocks, kAllocSize, kNumAllocs, d_results);

  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  for (int i = 0; i < kTotal; ++i) {
    INFO("Thread " << i << " result: " << d_results[i]);
    REQUIRE(d_results[i] == kNumAllocs);
  }

  cudaFreeHost(d_results);
}

/**
 * Multi-block test: 4 blocks × 32 threads.
 * CHIMAERA_GPU_ORCHESTRATOR_INIT shifts data_ per block; this is the path
 * that was NOT covered by the single-block test and is suspected to trigger
 * the misaligned-address bug with BuddyAllocator.
 *
 * Expected to pass with ArenaAllocator and reveal the bug with BuddyAllocator.
 * Run under compute-sanitizer for precise error location:
 *   compute-sanitizer --tool memcheck bin/test_buddy_benchmark_gpu
 *     "[BuddyBenchmark - multi block (benchmark workload)]"
 */
TEST_CASE("BuddyBenchmark - multi block (benchmark workload)",
          "[gpu][buddy][benchmark]") {
  constexpr int    kNumBlocks      = 4;
  constexpr int    kThreadsPerBlock = 32;
  constexpr int    kNumAllocs      = 100;
  constexpr int    kAllocSize      = 168;  // sizeof(FutureShm)
  constexpr size_t kPerBlockBytes  = 10u * 1024u * 1024u;   // 10 MB (matches benchmark)
  constexpr size_t kBackendSize    = kNumBlocks * kPerBlockBytes;  // 40 MB

  cudaDeviceSetLimit(cudaLimitStackSize, 4096);

  hipc::GpuShmMmap backend;
  hipc::MemoryBackendId backend_id(51, 0);
  REQUIRE(backend.shm_init(backend_id, kBackendSize,
                           "/test_buddy_bench_multi", 0));

  chi::IpcManagerGpu gpu_info(backend, nullptr);

  constexpr int kTotal = kNumBlocks * kThreadsPerBlock;
  int *d_results = nullptr;
  cudaMallocHost(&d_results, kTotal * sizeof(int));
  REQUIRE(d_results != nullptr);
  memset(d_results, -1, kTotal * sizeof(int));

  bench_mimic_kernel<<<kNumBlocks, kThreadsPerBlock>>>(
      gpu_info, kNumBlocks, kAllocSize, kNumAllocs, d_results);

  cudaError_t sync_err = cudaDeviceSynchronize();
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(sync_err == cudaSuccess);

  for (int i = 0; i < kTotal; ++i) {
    INFO("Thread " << i << " result: " << d_results[i]);
    REQUIRE(d_results[i] == kNumAllocs);
  }

  cudaFreeHost(d_results);
}

/**
 * Stress test: 8 blocks × 32 threads, 500 allocations each.
 * Exercises BuddyAllocator RepopulateSmallArena multiple times per thread.
 */
TEST_CASE("BuddyBenchmark - stress (8 blocks, 500 allocs)",
          "[gpu][buddy][benchmark][stress]") {
  constexpr int    kNumBlocks      = 8;
  constexpr int    kThreadsPerBlock = 32;
  constexpr int    kNumAllocs      = 500;
  constexpr int    kAllocSize      = 168;
  constexpr size_t kPerBlockBytes  = 10u * 1024u * 1024u;
  constexpr size_t kBackendSize    = kNumBlocks * kPerBlockBytes;  // 80 MB

  cudaDeviceSetLimit(cudaLimitStackSize, 4096);

  hipc::GpuShmMmap backend;
  hipc::MemoryBackendId backend_id(52, 0);
  REQUIRE(backend.shm_init(backend_id, kBackendSize,
                           "/test_buddy_bench_stress", 0));

  chi::IpcManagerGpu gpu_info(backend, nullptr);

  constexpr int kTotal = kNumBlocks * kThreadsPerBlock;
  int *d_results = nullptr;
  cudaMallocHost(&d_results, kTotal * sizeof(int));
  REQUIRE(d_results != nullptr);
  memset(d_results, -1, kTotal * sizeof(int));

  bench_mimic_kernel<<<kNumBlocks, kThreadsPerBlock>>>(
      gpu_info, kNumBlocks, kAllocSize, kNumAllocs, d_results);

  cudaError_t sync_err = cudaDeviceSynchronize();
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(sync_err == cudaSuccess);

  for (int i = 0; i < kTotal; ++i) {
    INFO("Thread " << i << " result: " << d_results[i]);
    REQUIRE(d_results[i] == kNumAllocs);
  }

  cudaFreeHost(d_results);
}

SIMPLE_TEST_MAIN()

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
