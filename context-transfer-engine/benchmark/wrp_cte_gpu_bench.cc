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
 * CTE GPU Benchmark
 *
 * Measures the performance of GPU-initiated PutBlob operations through CTE.
 * Supports multiple modes:
 *   putblob     -- GPU client -> CTE via GPU->CPU path (ToLocalCpu)
 *   putblob_gpu -- GPU client -> CTE via GPU-local path (Local)
 *   direct      -- GPU kernel writes directly to pinned host memory (baseline)
 *   cudamemcpy  -- cudaMemcpyAsync baseline (theoretical PCIe max)
 *   managed     -- CUDA managed memory write + prefetch to host
 *
 * Usage:
 *   wrp_cte_gpu_bench [options]
 *
 * Options:
 *   --test-case <case>       putblob, putblob_gpu, direct, cudamemcpy, or managed
 *   --rt-blocks <N>          GPU runtime orchestrator blocks (default: 1)
 *   --rt-threads <N>         GPU runtime threads per block (default: 32)
 *   --client-blocks <N>      GPU client kernel blocks (default: 1)
 *   --client-threads <N>     GPU client kernel threads per block (default: 32)
 *   --io-size <bytes>        Per-warp I/O size (default: 64M, supports k/m/g)
 *
 * Compiled as a single file via add_cuda_executable (clang-cuda dual-pass).
 * GPU kernels and host-side launcher/main are in the same translation unit;
 * host-only code is guarded by #if HSHM_IS_HOST.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/logging.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>

// Needed for Transport::Send template (host-only, guarded by HSHM_IS_HOST
// inside the header).  Only available when HSHM_ENABLE_LIGHTBEAM is defined,
// which the build system now propagates from linked interface targets.
#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

using namespace std::chrono_literals;

//==============================================================================
// GPU Kernels
//==============================================================================

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
    chi::u64 warp_bytes,
    chi::u32 total_warps,
    chi::u32 iterations,
    bool to_cpu,
    int *d_done,
    volatile int *d_progress) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  // Write progress to pinned memory (visible to host immediately)
  if (lane_id == 0 && warp_id < total_warps) {
    d_progress[warp_id] = 1;  // init complete
    __threadfence_system();
  }
  __syncwarp();

  if (warp_id < total_warps) {
    // Compute this warp's slice of the array
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * warp_bytes;
    char *my_data = array_ptr.ptr_ + my_offset;

    for (chi::u32 iter = 0; iter < iterations; ++iter) {
      // All lanes participate in memset
      for (chi::u64 i = lane_id; i < warp_bytes; i += 32) {
        my_data[i] = static_cast<char>((warp_id + iter) & 0xFF);
      }
      __syncwarp();

      // Only lane 0 submits PutBlob
      if (chi::IpcManager::IsWarpScheduler()) {
        wrp_cte::core::Client cte_client(cte_pool_id);

        // Build ShmPtr referencing the data allocator backend
        hipc::ShmPtr<> blob_shm;
        blob_shm.alloc_id_ = data_alloc_id;
        size_t base_off = array_ptr.shm_.off_.load();
        blob_shm.off_.exchange(base_off + my_offset);

        // Build blob name: "w_<warp_id>_<iter>"
        char name_buf[32];
        int pos = 0;
        name_buf[pos++] = 'w';
        name_buf[pos++] = '_';
        chi::u32 wid = warp_id;
        char digits[10];
        int nd = 0;
        do { digits[nd++] = '0' + (wid % 10); wid /= 10; } while (wid > 0);
        for (int d = nd - 1; d >= 0; --d) name_buf[pos++] = digits[d];
        name_buf[pos++] = '_';
        chi::u32 it = iter;
        nd = 0;
        do { digits[nd++] = '0' + (it % 10); it /= 10; } while (it > 0);
        for (int d = nd - 1; d >= 0; --d) name_buf[pos++] = digits[d];
        name_buf[pos] = '\0';

        auto *ipc = CHI_IPC;
        auto task = ipc->NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), cte_pool_id,
            to_cpu ? chi::PoolQuery::ToLocalCpu() : chi::PoolQuery::Local(),
            tag_id, name_buf,
            (chi::u64)0, warp_bytes,
            blob_shm, -1.0f,
            wrp_cte::core::Context(), (chi::u32)0);
        auto future = ipc->Send(task);
        future.Wait();
      }
      __syncwarp();
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
 * Kernel 2b: Multi-block allocator stress test.
 * Each warp initializes, allocates from scratch and heap, then frees.
 * Used to isolate multi-block ThreadAllocator bugs.
 */
__global__ void gpu_alloc_test_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::u32 num_blocks,
    chi::u32 total_warps,
    int *d_done,
    volatile int *d_progress) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (lane_id == 0 && warp_id < total_warps) {
    d_progress[warp_id] = 1;  // init done
    __threadfence_system();

    // Test scratch allocator
    auto *scratch = CHI_IPC->gpu_alloc_;
    if (scratch) {
      auto p = scratch->AllocateObjs<char>(1024);
      if (!p.IsNull()) {
        d_progress[warp_id] = 2;  // scratch alloc ok
        __threadfence_system();
        hipc::OffsetPtr<> off; off = p.shm_.off_.load();
        scratch->FreeOffset(off);
      } else {
        d_progress[warp_id] = -2;  // scratch alloc failed
        __threadfence_system();
      }
    }

    // Test heap allocator
    auto *heap = CHI_IPC->gpu_heap_alloc_;
    if (heap) {
      auto p = heap->AllocateObjs<char>(256);
      if (!p.IsNull()) {
        d_progress[warp_id] = 3;  // heap alloc ok
        __threadfence_system();
        hipc::OffsetPtr<> hoff; hoff = p.shm_.off_.load();
        heap->FreeOffset(hoff);
      } else {
        d_progress[warp_id] = -3;  // heap alloc failed
        __threadfence_system();
      }
    }

    d_progress[warp_id] = 10;  // all done
    __threadfence_system();
    int prev = atomicAdd(d_done, 1);
    if (prev == static_cast<int>(total_warps) - 1) {
      __threadfence_system();
    }
  }
}

/**
 * Kernel 3: Direct copy baseline -- all threads cooperatively copy from
 * device memory to pinned host memory using 4-byte coalesced stores.
 * 32 threads x 4 bytes = 128 bytes per iteration = one PCIe cache line.
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
 * Kernel 4: Write to managed memory -- same warp structure as PutBlob.
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

//==============================================================================
// Host-only code: CPU-side launchers, CLI, and main()
//==============================================================================
#if HSHM_IS_HOST

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

static bool PollDone(volatile int *d_done, int total_warps, int timeout_us) {
  int elapsed_us = 0;
  while (*d_done < total_warps && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }
  return *d_done >= total_warps;
}

//==============================================================================
// CPU-side Benchmark Launchers
//==============================================================================

static int run_cte_gpu_bench_putblob(
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 warp_bytes,
    chi::u32 iterations,
    bool to_cpu,
    float *out_elapsed_ms) {
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Pause GPU orchestrator before any cudaDeviceSynchronize / GPU init.
  CHI_IPC->PauseGpuOrchestrator();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;
  chi::u64 total_data_bytes = warp_bytes * total_warps;

  // --- 1. Data backend: device memory for array A ---
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  size_t data_backend_size = total_data_bytes + 4 * 1024 * 1024;
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
      total_data_bytes, d_array_ptr);
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

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  // Per-warp progress array (pinned host)
  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int) * 64);
  memset((void*)d_progress, 0, sizeof(int) * 64);

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  // Zero client scratch/heap backends (device memory) so non-block-0
  // blocks don't see stale heap_ready_ flags from previous allocations.
  if (scratch_backend.data_ != nullptr) {
    cudaMemset(scratch_backend.data_, 0, sizeof(hipc::ThreadAllocator));
  }
  if (heap_backend.data_ != nullptr) {
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::ThreadAllocator));
  }

  // Record CUDA event on client stream just before kernel launch
  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);
  cudaEventRecord(ev_start, static_cast<cudaStream_t>(stream));

  // Launch client kernel (it blocks in INIT until orchestrator starts)
  gpu_putblob_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, client_blocks,
      array_ptr,
      hipc::AllocatorId(data_backend_id.major_, data_backend_id.minor_),
      warp_bytes, total_warps, iterations, to_cpu, d_done, d_progress);

  // Record end event — fires after kernel finishes
  cudaEventRecord(ev_end, static_cast<cudaStream_t>(stream));

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "ERROR: Client kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    hshm::GpuApi::DestroyStream(stream);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    return -3;
  }

  auto *orchestrator = static_cast<chi::gpu::WorkOrchestrator *>(
      CHI_IPC->gpu_orchestrator_);
  auto *ctrl = orchestrator ? orchestrator->control_ : nullptr;

  CHI_IPC->ResumeGpuOrchestrator();
  if (ctrl) {
    int wait_ms = 0;
    while (ctrl->running_flag == 0 && wait_ms < 5000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ++wait_ms;
    }
    if (ctrl->running_flag == 0) {
      fprintf(stderr, "ERROR: Orchestrator failed to start after %dms\n", wait_ms);
    }
  }

  constexpr int kTimeoutUs = 30000000;  // 30s
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  // Wait for end event and compute GPU-side elapsed time
  cudaEventSynchronize(ev_end);
  float gpu_elapsed_ms = 0;
  cudaEventElapsedTime(&gpu_elapsed_ms, ev_start, ev_end);
  *out_elapsed_ms = gpu_elapsed_ms;
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  // On timeout: pause orchestrator FIRST to flush GPU printf, then sync client
  if (!completed) {
    if (ctrl) {
      fprintf(stderr, "TIMEOUT: d_done=%d/%u running_flag=%d\n",
              *d_done, total_warps, ctrl->running_flag);
      for (chi::u32 i = 0; i < (rt_blocks * rt_threads) / 32 && i < 32; ++i) {
        fprintf(stderr, "  warp[%u]: polls=%llu qpop=%u pop=%u done=%u "
                "state=%u step=%u method=%u pool=%u.%u "
                "tail=%llu head=%llu\n",
                i, (unsigned long long)ctrl->dbg_poll_count[i],
                ctrl->dbg_queue_pops[i],
                ctrl->dbg_tasks_popped[i],
                ctrl->dbg_tasks_completed[i],
                ctrl->dbg_last_state[i],
                ctrl->dbg_dispatch_step[i],
                ctrl->dbg_last_method[i],
                (unsigned int)(ctrl->dbg_resume_checks[i] >> 32),
                (unsigned int)(ctrl->dbg_resume_checks[i] & 0xFFFFFFFF),
                (unsigned long long)ctrl->dbg_input_tw[i],
                (unsigned long long)ctrl->dbg_input_cs[i]);
      }
      fflush(stderr);
    }
    // Print per-warp progress
    fprintf(stderr, "Client warp progress (1=init,2=newtask,3=send,4=wait,5=done):\n");
    for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
      fprintf(stderr, "  warp[%u]: %d\n", i, d_progress[i]);
    }
    fflush(stderr);
    CHI_IPC->PauseGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    return -4;
  }

  hshm::GpuApi::Synchronize(stream);
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  cudaFreeHost((void*)d_progress);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

static int run_cte_gpu_bench_direct(
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    float *out_elapsed_ms) {
  CHI_IPC->PauseGpuOrchestrator();

  char *d_src = nullptr;
  cudaError_t err = cudaMalloc(&d_src, total_bytes);
  if (err != cudaSuccess || !d_src) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  cudaMemset(d_src, 0xAB, total_bytes);
  cudaDeviceSynchronize();

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

static int run_cte_gpu_bench_managed(
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    float *out_write_ms,
    float *out_prefetch_ms,
    float *out_total_ms) {
  CHI_IPC->PauseGpuOrchestrator();

  char *managed_buf = nullptr;
  cudaError_t err = cudaMallocManaged(&managed_buf, total_bytes);
  if (err != cudaSuccess || !managed_buf) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  int device = 0;
  cudaMemAdvise(managed_buf, total_bytes, cudaMemAdviseSetPreferredLocation, device);
  cudaMemPrefetchAsync(managed_buf, total_bytes, device, 0);
  cudaDeviceSynchronize();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();

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

static int run_cte_gpu_bench_cudamemcpy(
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

//==============================================================================
// CLI and main()
//==============================================================================

enum class TestCase { kPutBlob, kPutBlobGpu, kDirect, kCudaMemcpy, kManaged, kAllocTest };

struct BenchConfig {
  TestCase test_case = TestCase::kPutBlob;
  chi::u32 rt_blocks = 1;
  chi::u32 rt_threads = 32;
  chi::u32 client_blocks = 1;
  chi::u32 client_threads = 32;
  chi::u64 warp_bytes = 128 * 1024;  // per-warp I/O size
  chi::u32 iterations = 16;          // iterations per warp
};

namespace {

chi::u64 ParseSize(const std::string &s) {
  double val = 0.0;
  chi::u64 mult = 1;
  std::string num;
  char suffix = 0;
  for (char c : s) {
    if (std::isdigit(c) || c == '.') num += c;
    else { suffix = std::tolower(c); break; }
  }
  if (num.empty()) return 0;
  val = std::stod(num);
  switch (suffix) {
    case 'k': mult = 1024; break;
    case 'm': mult = 1024 * 1024; break;
    case 'g': mult = 1024ULL * 1024 * 1024; break;
    default: break;
  }
  return static_cast<chi::u64>(val * mult);
}

void PrintUsage(const char *prog) {
  HIPRINT("Usage: {} [options]", prog);
  HIPRINT("Options:");
  HIPRINT("  --test-case <case>     putblob, putblob_gpu, direct, cudamemcpy, or managed (default: putblob)");
  HIPRINT("  --rt-blocks <N>        GPU runtime orchestrator blocks (default: 1)");
  HIPRINT("  --rt-threads <N>       GPU runtime threads/block (default: 32)");
  HIPRINT("  --client-blocks <N>    GPU client kernel blocks (default: 1)");
  HIPRINT("  --client-threads <N>   GPU client kernel threads/block (default: 32)");
  HIPRINT("  --io-size <bytes>      Per-warp I/O size (default: 128K, supports k/m/g suffixes)");
  HIPRINT("  --iterations <N>       Iterations per warp (default: 16)");
  HIPRINT("  --help, -h             Show this help");
}

bool ParseArgs(int argc, char **argv, BenchConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--test-case" && i + 1 < argc) {
      std::string tc = argv[++i];
      if (tc == "putblob") cfg.test_case = TestCase::kPutBlob;
      else if (tc == "putblob_gpu") cfg.test_case = TestCase::kPutBlobGpu;
      else if (tc == "direct") cfg.test_case = TestCase::kDirect;
      else if (tc == "cudamemcpy") cfg.test_case = TestCase::kCudaMemcpy;
      else if (tc == "managed") cfg.test_case = TestCase::kManaged;
      else if (tc == "alloc_test") cfg.test_case = TestCase::kAllocTest;
      else {
        HLOG(kError, "Unknown test case '{}'; use putblob, putblob_gpu, direct, cudamemcpy, managed, or alloc_test", tc);
        return false;
      }
    } else if (arg == "--rt-blocks" && i + 1 < argc) {
      cfg.rt_blocks = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--rt-threads" && i + 1 < argc) {
      cfg.rt_threads = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--client-blocks" && i + 1 < argc) {
      cfg.client_blocks = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--client-threads" && i + 1 < argc) {
      cfg.client_threads = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--io-size" && i + 1 < argc) {
      cfg.warp_bytes = ParseSize(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      cfg.iterations = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else {
      HLOG(kError, "Unknown option: {}", arg);
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  BenchConfig cfg;
  if (!ParseArgs(argc, argv, cfg)) return 1;

  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    HLOG(kError, "No GPUs available");
    return 1;
  }

  // Load GPU config if CHI_SERVER_CONF is not already set
  if (!std::getenv("CHI_SERVER_CONF")) {
    std::string config_dir = std::string(__FILE__);
    config_dir = config_dir.substr(0, config_dir.rfind('/'));
    std::string gpu_config = config_dir + "/cte_config_gpu.yaml";
    setenv("CHI_SERVER_CONF", gpu_config.c_str(), 1);
    HLOG(kInfo, "Using GPU benchmark config: {}", gpu_config);
  }

  // Initialize Chimaera runtime
  HLOG(kInfo, "Initializing Chimaera runtime...");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }
  std::this_thread::sleep_for(500ms);

  const char *tc_name = (cfg.test_case == TestCase::kPutBlob) ? "putblob" :
                         (cfg.test_case == TestCase::kPutBlobGpu) ? "putblob_gpu" :
                         (cfg.test_case == TestCase::kCudaMemcpy) ? "cudamemcpy" :
                         (cfg.test_case == TestCase::kManaged) ? "managed" :
                         (cfg.test_case == TestCase::kAllocTest) ? "alloc_test" :
                         "direct";

  HIPRINT("\n=== CTE GPU Benchmark ===");
  HIPRINT("Test case:           {}", tc_name);
  HIPRINT("RT blocks:           {}", cfg.rt_blocks);
  HIPRINT("RT threads/block:    {}", cfg.rt_threads);
  HIPRINT("Client blocks:       {}", cfg.client_blocks);
  HIPRINT("Client threads/block:{}", cfg.client_threads);
  HIPRINT("Per-warp I/O size:   {} bytes ({} KB)", cfg.warp_bytes,
          cfg.warp_bytes / 1024);
  HIPRINT("Iterations:          {}", cfg.iterations);

  // Compute total I/O: per-warp size x warps x iterations
  chi::u32 client_warps = (cfg.client_blocks * cfg.client_threads) / 32;
  if (client_warps == 0) client_warps = 1;
  chi::u64 total_bytes = cfg.warp_bytes * client_warps * cfg.iterations;
  HIPRINT("Total I/O size:      {} bytes ({} MB)", total_bytes,
          total_bytes / (1024 * 1024));

  float elapsed_ms = 0;
  int rc = 0;

  if (cfg.test_case == TestCase::kAllocTest) {
    // Minimal multi-block allocator stress test
    CHI_IPC->PauseGpuOrchestrator();

    chi::u32 tw = (cfg.client_blocks * cfg.client_threads) / 32;
    if (tw == 0) tw = 1;

    // Scratch backend
    size_t scratch_size = static_cast<size_t>(cfg.client_blocks) * 10 * 1024 * 1024;
    hipc::MemoryBackendId scratch_id(201, 0);
    hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, scratch_size, "", 0);

    // Heap backend
    size_t heap_size = static_cast<size_t>(cfg.client_blocks) * 4 * 1024 * 1024;
    hipc::MemoryBackendId heap_id(202, 0);
    hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, heap_size, "", 0);

    chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;
    gpu_info.gpu_heap_backend = heap_backend;
    gpu_info.gpu_heap_partitions = tw;

    int *d_done;
    cudaMallocHost(&d_done, sizeof(int));
    *d_done = 0;
    volatile int *d_progress;
    cudaMallocHost((void**)&d_progress, sizeof(int) * 64);
    memset((void*)d_progress, 0, sizeof(int) * 64);

    if (scratch_backend.data_ != nullptr)
      cudaMemset(scratch_backend.data_, 0, sizeof(hipc::ThreadAllocator));
    if (heap_backend.data_ != nullptr)
      cudaMemset(heap_backend.data_, 0, sizeof(hipc::ThreadAllocator));
    cudaDeviceSynchronize();

    void *stream = hshm::GpuApi::CreateStream();
    gpu_alloc_test_kernel<<<
        cfg.client_blocks, cfg.client_threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        gpu_info, cfg.client_blocks, tw, d_done, d_progress);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: alloc_test kernel launch failed: %s\n",
              cudaGetErrorString(err));
      cudaFreeHost(d_done);
      cudaFreeHost((void*)d_progress);
      hshm::GpuApi::DestroyStream(stream);
      CHI_IPC->ResumeGpuOrchestrator();
      return 1;
    }

    bool completed = PollDone(d_done, static_cast<int>(tw), 10000000);

    // Print progress BEFORE sync (kernel may have crashed)
    printf("\n=== alloc_test Results ===\n");
    printf("Completed: %s (d_done=%d/%u)\n", completed ? "YES" : "NO", *d_done, tw);
    for (chi::u32 i = 0; i < tw && i < 64; ++i) {
      printf("  warp[%u]: progress=%d\n", i, d_progress[i]);
    }

    fflush(stdout);
    // Don't sync if not completed — kernel may have crashed
    if (completed) {
      hshm::GpuApi::Synchronize(stream);
    }
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    hshm::GpuApi::DestroyStream(stream);
    CHI_IPC->ResumeGpuOrchestrator();
    return completed ? 0 : 1;
  } else if (cfg.test_case == TestCase::kManaged) {
    float write_ms = 0, prefetch_ms = 0, total_ms = 0;
    rc = run_cte_gpu_bench_managed(cfg.client_blocks, cfg.client_threads,
                                    total_bytes,
                                    &write_ms, &prefetch_ms, &total_ms);
    if (rc != 0) {
      HLOG(kError, "Benchmark failed with error: {}", rc);
      return 1;
    }
    double bytes = total_bytes;
    printf("\n=== %s Results ===\n", tc_name);
    printf("GPU write:           %.3f ms  (%.3f GB/s)\n",
           static_cast<double>(write_ms), (bytes / 1e9) / (write_ms / 1e3));
    printf("Prefetch to host:    %.3f ms  (%.3f GB/s)\n",
           static_cast<double>(prefetch_ms), (bytes / 1e9) / (prefetch_ms / 1e3));
    printf("Total:               %.3f ms  (%.3f GB/s)\n",
           static_cast<double>(total_ms), (bytes / 1e9) / (total_ms / 1e3));
    printf("=========================\n");
    return 0;
  } else if (cfg.test_case == TestCase::kCudaMemcpy) {
    rc = run_cte_gpu_bench_cudamemcpy(total_bytes, &elapsed_ms);
  } else if (cfg.test_case == TestCase::kDirect) {
    rc = run_cte_gpu_bench_direct(cfg.client_blocks, cfg.client_threads,
                                   total_bytes, &elapsed_ms);
  } else {
    // CTE putblob path -- need pool, target, and tag setup
    chi::PoolId gpu_pool_id(wrp_cte::core::kCtePoolId.major_ + 1,
                             wrp_cte::core::kCtePoolId.minor_);
    wrp_cte::core::Client cte_client(gpu_pool_id);
    wrp_cte::core::CreateParams params;
    auto create_task = cte_client.AsyncCreate(
        chi::PoolQuery::Dynamic(),
        "cte_gpu_bench_pool", gpu_pool_id, params);
    create_task.Wait();
    if (create_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create CTE GPU pool: {}", create_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    chi::PoolId bdev_pool_id(800, 0);
    chi::u64 data_footprint = cfg.warp_bytes * client_warps;
    chi::u64 bdev_size = std::max(data_footprint + 64ULL * 1024 * 1024,
                                    256ULL * 1024 * 1024);
    auto reg_task = cte_client.AsyncRegisterTarget(
        "pinned::cte_gpu_bench_target",
        chimaera::bdev::BdevType::kPinned,
        bdev_size,
        chi::PoolQuery::Local(),
        bdev_pool_id);
    reg_task.Wait();
    if (reg_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to register target (CPU): {}", reg_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    auto gpu_reg_task = cte_client.AsyncRegisterTarget(
        "pinned::cte_gpu_bench_target",
        chimaera::bdev::BdevType::kPinned,
        bdev_size,
        chi::PoolQuery::Local(),
        bdev_pool_id,
        chi::PoolQuery::LocalGpuBcast());
    gpu_reg_task.Wait();
    if (gpu_reg_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to register target (GPU): {}", gpu_reg_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    auto tag_task = cte_client.AsyncGetOrCreateTag(
        "gpu_bench_tag", wrp_cte::core::TagId::GetNull(),
        chi::PoolQuery::Local());
    tag_task.Wait();
    if (tag_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create tag");
      return 1;
    }
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    auto gpu_tag_task = cte_client.AsyncGetOrCreateTag(
        "gpu_bench_tag", tag_id, chi::PoolQuery::LocalGpuBcast());
    gpu_tag_task.Wait();
    if (gpu_tag_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create tag on GPU: {}", gpu_tag_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    HIPRINT("Pool ID: {}.{}", gpu_pool_id.major_, gpu_pool_id.minor_);
    HIPRINT("Tag ID:  {}.{}", tag_id.major_, tag_id.minor_);

    bool to_cpu = (cfg.test_case == TestCase::kPutBlob);
    rc = run_cte_gpu_bench_putblob(
        cte_client.pool_id_, tag_id,
        cfg.rt_blocks, cfg.rt_threads,
        cfg.client_blocks, cfg.client_threads,
        cfg.warp_bytes, cfg.iterations, to_cpu, &elapsed_ms);
  }

  if (rc != 0) {
    HLOG(kError, "Benchmark failed with error: {}", rc);
    return 1;
  }

  double bw_gbps = (total_bytes / 1e9) / (elapsed_ms / 1e3);

  printf("\n=== %s Results ===\n", tc_name);
  printf("Elapsed:             %.3f ms\n", static_cast<double>(elapsed_ms));
  printf("Bandwidth:           %.3f GB/s\n", bw_gbps);
  printf("=========================\n");

  return 0;
}

#endif  // HSHM_IS_HOST

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
