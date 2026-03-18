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
 * GPU benchmark kernels for CTE PutBlob data placement.
 *
 * Each warp in the client kernel:
 *   1. memsets its slice of a pre-allocated device array
 *   2. Lane 0 calls AsyncPutBlob to store the slice via CTE
 *   3. Signals completion via atomicAdd on a pinned done counter
 *
 * Supports both GPU->CPU (ToLocalCpu) and GPU-local (Local) routing.
 *
 * Compiled via add_cuda_library (clang-cuda dual-pass).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <thread>
#include <chrono>

/**
 * Poll the pinned done flag until all warps complete or timeout.
 */
static bool PollDone(volatile int *d_done, int total_warps, int timeout_us) {
  int elapsed_us = 0;
  while (*d_done < total_warps && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }
  return *d_done >= total_warps;
}

/**
 * Kernel 1: Initialize a BuddyAllocator over device memory and allocate
 * a contiguous array of `total_bytes` bytes.  Returns the FullPtr via
 * pinned host memory so the CPU can read it.
 */
__global__ void gpu_putblob_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) {
    d_out_ptr->SetNull();
    return;
  }
  auto result = alloc->AllocateObjs<char>(total_bytes);
  *d_out_ptr = result;
}

/**
 * Kernel 2: Each warp memsets its slice of A to a constant, then calls
 * AsyncPutBlob to store that slice as a blob via the CTE runtime.
 * Only the warp scheduler (lane 0) submits the PutBlob task.
 */
__global__ void gpu_putblob_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> array_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 total_bytes,
    chi::u32 total_warps,
    bool to_cpu,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    // Compute this warp's slice of the array
    chi::u64 slice_size = total_bytes / total_warps;
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * slice_size;
    char *my_data = array_ptr.ptr_ + my_offset;

    // All lanes participate in memset
    for (chi::u64 i = lane_id; i < slice_size; i += 32) {
      my_data[i] = static_cast<char>(warp_id & 0xFF);
    }
    __syncwarp();

    // Only lane 0 submits PutBlob
    if (chi::IpcManager::IsWarpScheduler()) {
      wrp_cte::core::Client cte_client(cte_pool_id);

      // Build ShmPtr referencing the data allocator backend
      hipc::ShmPtr<> blob_shm;
      blob_shm.alloc_id_ = data_alloc_id;
      // offset = distance from backend base
      size_t base_off = array_ptr.shm_.off_.load();
      blob_shm.off_.exchange(base_off + my_offset);

      // Build blob name: "w_<id>"
      char name_buf[32];
      int pos = 0;
      name_buf[pos++] = 'w';
      name_buf[pos++] = '_';
      chi::u32 wid = warp_id;
      char digits[10];
      int nd = 0;
      do { digits[nd++] = '0' + (wid % 10); wid /= 10; } while (wid > 0);
      for (int d = nd - 1; d >= 0; --d) name_buf[pos++] = digits[d];
      name_buf[pos] = '\0';

      auto future = cte_client.AsyncPutBlob(
          tag_id, name_buf,
          /*offset=*/0, /*size=*/slice_size,
          blob_shm, /*score=*/-1.0f,
          wrp_cte::core::Context(), /*flags=*/0,
          to_cpu ? chi::PoolQuery::ToLocalCpu()
                 : chi::PoolQuery::Local());

      future.Wait();
    }
  }

  __syncwarp();
  if (chi::IpcManager::IsWarpScheduler()) {
    __threadfence();
    int prev = atomicAdd(d_done, 1);
    if (prev == static_cast<int>(total_warps) - 1) {
      __threadfence_system();
    }
  }
}

/**
 * Kernel 3: Direct copy baseline — all threads cooperatively copy from
 * device memory to pinned host memory using 4-byte coalesced stores.
 * 32 threads × 4 bytes = 128 bytes per iteration = one PCIe cache line.
 */
__global__ void gpu_direct_memcpy_kernel(
    const char *d_src,
    char *h_dst,
    chi::u64 total_bytes,
    chi::u32 total_threads_used,
    int *d_done) {
  chi::u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  chi::u32 stride = blockDim.x * gridDim.x;

  // Coalesced 4-byte stores across all threads
  chi::u64 n_words = total_bytes / 4;
  const unsigned int *src4 = reinterpret_cast<const unsigned int *>(d_src);
  unsigned int *dst4 = reinterpret_cast<unsigned int *>(h_dst);
  for (chi::u64 i = tid; i < n_words; i += stride) {
    dst4[i] = src4[i];
  }

  // Tail bytes
  if (tid == 0) {
    chi::u64 tail = n_words * 4;
    for (chi::u64 i = tail; i < total_bytes; ++i) {
      h_dst[i] = d_src[i];
    }
  }

  __threadfence_system();
  atomicAdd(d_done, 1);
}

/**
 * CPU-side launcher for the PutBlob data placement benchmark.
 *
 * 1. Allocates a device memory backend + BuddyAllocator
 * 2. Runs alloc kernel to allocate array A
 * 3. Registers backend with runtime
 * 4. Launches data placement kernel (memset + PutBlob per warp)
 */
extern "C" int run_cte_gpu_bench_putblob(
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    bool to_cpu,
    float *out_elapsed_ms) {
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Pause GPU orchestrator before any cudaDeviceSynchronize / GPU init.
  CHI_IPC->PauseGpuOrchestrator();

  // --- 1. Data backend: device memory for array A ---
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  size_t data_backend_size = total_bytes + 4 * 1024 * 1024;
  if (!data_backend.shm_init(data_backend_id, data_backend_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 2. Client scratch backend (for FutureShm, serialization) ---
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t scratch_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 3. GPU heap backend (for ThreadAllocator) ---
  constexpr size_t kPerBlockHeapBytes = 4 * 1024 * 1024;
  size_t heap_size = static_cast<size_t>(client_blocks) * kPerBlockHeapBytes;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 4. Run alloc kernel to initialize allocator + allocate A ---
  hipc::FullPtr<char> *d_array_ptr;
  cudaMallocHost(&d_array_ptr, sizeof(hipc::FullPtr<char>));
  d_array_ptr->SetNull();

  gpu_putblob_alloc_kernel<<<1, 1>>>(
      static_cast<hipc::MemoryBackend &>(data_backend),
      total_bytes, d_array_ptr);
  cudaDeviceSynchronize();

  if (d_array_ptr->IsNull()) {
    cudaFreeHost(d_array_ptr);
    return -2;
  }

  hipc::FullPtr<char> array_ptr = *d_array_ptr;
  cudaFreeHost(d_array_ptr);

  // --- 5. Register data backend with runtime for ShmPtr resolution ---
  CHI_IPC->RegisterGpuAllocator(data_backend_id, data_backend.data_,
                                 data_backend.data_capacity_);

  // --- 6. Build GPU info and launch data placement kernel ---
  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;
  gpu_info.gpu_heap_backend = heap_backend;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  gpu_putblob_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, client_blocks,
      array_ptr,
      hipc::AllocatorId(data_backend_id.major_, data_backend_id.minor_),
      total_bytes, total_warps, to_cpu, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_IPC->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;  // 60s
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * CPU-side launcher for the direct memcpy baseline benchmark.
 *
 * Mirrors PutBlob's structure: same block/thread config, same warp-slicing.
 * Each warp's lane 0 calls memcpy to copy its slice to pinned host memory.
 */
extern "C" int run_cte_gpu_bench_direct(
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    float *out_elapsed_ms) {
  // Pause GPU orchestrator to allow cudaDeviceSynchronize
  CHI_IPC->PauseGpuOrchestrator();

  // Allocate device source buffer
  char *d_src = nullptr;
  cudaError_t err = cudaMalloc(&d_src, total_bytes);
  if (err != cudaSuccess || !d_src) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // Fill with pattern
  cudaMemset(d_src, 0xAB, total_bytes);
  cudaDeviceSynchronize();

  // Allocate pinned host destination buffer
  char *h_dst = nullptr;
  err = cudaMallocHost(reinterpret_cast<void **>(&h_dst), total_bytes);
  if (err != cudaSuccess || !h_dst) {
    cudaFree(d_src);
    CHI_IPC->ResumeGpuOrchestrator();
    return -2;
  }
  memset(h_dst, 0, total_bytes);

  chi::u32 total_threads_used = client_blocks * client_threads;
  if (total_threads_used == 0) total_threads_used = 1;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  gpu_direct_memcpy_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      d_src, h_dst, total_bytes, total_threads_used, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    cudaFreeHost(d_done);
    cudaFreeHost(h_dst);
    cudaFree(d_src);
    hshm::GpuApi::DestroyStream(stream);
    CHI_IPC->ResumeGpuOrchestrator();
    return -3;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;  // 60s
  bool completed = PollDone(d_done, static_cast<int>(total_threads_used), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);

  cudaFreeHost(d_done);
  cudaFreeHost(h_dst);
  cudaFree(d_src);
  hshm::GpuApi::DestroyStream(stream);
  CHI_IPC->ResumeGpuOrchestrator();

  return completed ? 0 : -4;
}

/**
 * Kernel 4: Write to managed memory — same warp structure as PutBlob.
 * All lanes cooperatively memset their slice. Pages reside in VRAM,
 * so writes happen at full VRAM bandwidth (~256 GB/s).
 */
__global__ void gpu_managed_write_kernel(
    char *managed_buf,
    chi::u64 total_bytes,
    chi::u32 total_warps,
    int *d_done) {
  chi::u32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  chi::u32 lane_id = threadIdx.x % 32;

  if (warp_id < total_warps) {
    chi::u64 slice_size = total_bytes / total_warps;
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * slice_size;
    char *my_data = managed_buf + my_offset;

    // All lanes participate in write (same as PutBlob kernel memset)
    for (chi::u64 i = lane_id; i < slice_size; i += 32) {
      my_data[i] = static_cast<char>(warp_id & 0xFF);
    }
    __syncwarp();

    if (lane_id == 0) {
      __threadfence();
      atomicAdd(d_done, 1);
    }
  }
}

/**
 * CPU-side launcher for managed memory benchmark.
 *
 * Measures three phases:
 *   1. GPU write to managed memory (pages in VRAM)
 *   2. Prefetch to host via cudaMemPrefetchAsync (DMA migration)
 *   3. Total (write + prefetch)
 */
extern "C" int run_cte_gpu_bench_managed(
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    float *out_write_ms,
    float *out_prefetch_ms,
    float *out_total_ms) {
  CHI_IPC->PauseGpuOrchestrator();

  // Allocate managed memory, prefer GPU location
  char *managed_buf = nullptr;
  cudaError_t err = cudaMallocManaged(&managed_buf, total_bytes);
  if (err != cudaSuccess || !managed_buf) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // Advise preferred location = GPU so pages stay in VRAM during writes
  int device = 0;
  cudaMemAdvise(managed_buf, total_bytes, cudaMemAdviseSetPreferredLocation, device);
  // Pre-fault pages onto GPU
  cudaMemPrefetchAsync(managed_buf, total_bytes, device, 0);
  cudaDeviceSynchronize();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();

  // Phase 1: GPU write
  auto t0 = std::chrono::high_resolution_clock::now();

  gpu_managed_write_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      managed_buf, total_bytes, total_warps, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    cudaFreeHost(d_done);
    cudaFree(managed_buf);
    hshm::GpuApi::DestroyStream(stream);
    CHI_IPC->ResumeGpuOrchestrator();
    return -2;
  }

  bool completed = PollDone(d_done, static_cast<int>(total_warps), 60000000);
  hshm::GpuApi::Synchronize(stream);

  auto t1 = std::chrono::high_resolution_clock::now();

  if (!completed) {
    cudaFreeHost(d_done);
    cudaFree(managed_buf);
    hshm::GpuApi::DestroyStream(stream);
    CHI_IPC->ResumeGpuOrchestrator();
    return -3;
  }

  // Phase 2: Prefetch to host (DMA migration)
  cudaMemPrefetchAsync(managed_buf, total_bytes, cudaCpuDeviceId,
                       static_cast<cudaStream_t>(stream));
  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

  auto t2 = std::chrono::high_resolution_clock::now();

  double write_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  double prefetch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t0).count();

  *out_write_ms = static_cast<float>(write_ns / 1e6);
  *out_prefetch_ms = static_cast<float>(prefetch_ns / 1e6);
  *out_total_ms = static_cast<float>(total_ns / 1e6);

  cudaFreeHost(d_done);
  cudaFree(managed_buf);
  hshm::GpuApi::DestroyStream(stream);
  CHI_IPC->ResumeGpuOrchestrator();

  return 0;
}

/**
 * CPU-side launcher for cudaMemcpyAsync baseline.
 * This is the theoretical maximum for device→host transfer over PCIe.
 */
extern "C" int run_cte_gpu_bench_cudamemcpy(
    chi::u64 total_bytes,
    float *out_elapsed_ms) {
  CHI_IPC->PauseGpuOrchestrator();

  char *d_src = nullptr;
  cudaMalloc(&d_src, total_bytes);
  cudaMemset(d_src, 0xAB, total_bytes);

  char *h_dst = nullptr;
  cudaMallocHost(reinterpret_cast<void **>(&h_dst), total_bytes);

  void *stream = hshm::GpuApi::CreateStream();

  auto t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpyAsync(h_dst, d_src, total_bytes, cudaMemcpyDeviceToHost,
                  static_cast<cudaStream_t>(stream));
  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  auto t_end = std::chrono::high_resolution_clock::now();

  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  cudaFreeHost(h_dst);
  cudaFree(d_src);
  hshm::GpuApi::DestroyStream(stream);
  CHI_IPC->ResumeGpuOrchestrator();

  return 0;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
