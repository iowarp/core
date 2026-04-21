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
 * GPU kernels for queue stress test.
 * N client warps × M iterations submit GpuSubmitTask to 1 RT warp.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/task.h>
#include <chimaera/local_task_archives.h>
#include <hermes_shm/util/gpu_api.h>

/**
 * GPU kernel: each warp submits iterations GpuSubmitTask roundtrips.
 * Only lane 0 of each warp does the actual submit+wait.
 *
 * @param gpu_info     IPC manager GPU info (scratch backend, queues)
 * @param pool_id      MOD_NAME pool to submit tasks to
 * @param iterations   Number of submit+wait cycles per warp
 * @param num_blocks   Client block count (for CHIMAERA_GPU_CLIENT_INIT)
 * @param d_done       Pinned host counter — each warp atomicAdds 1 on completion
 * @param d_progress   Per-warp progress (pinned host): 0=init, 1=running, 2=done
 */
__global__ void gpu_queue_stress_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    chi::PoolId pool_id,
    chi::u32 iterations,
    int num_blocks,
    int *d_done,
    volatile int *d_progress) {
  // Initialize IPC — multi-block aware, partitions allocators per warp
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();

  // Only lane 0 does the work
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;

  d_progress[warp_id] = 1;  // running

  auto *ipc = CHI_IPC;

  for (chi::u32 i = 0; i < iterations; ++i) {
    chi::u32 test_value = warp_id * 1000 + i;
    auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
        chi::u32(0), test_value);
    auto future = ipc->Send(task);
    future.Wait();
  }

  d_progress[warp_id] = 2;  // done
  atomicAdd_system(d_done, 1);
  __threadfence_system();
}

/**
 * C++ wrapper: launches the stress kernel and polls for completion.
 *
 * @param pool_id          MOD_NAME pool
 * @param client_blocks    Client kernel block count
 * @param client_threads   Client kernel threads per block
 * @param iterations       Submit+wait cycles per warp
 * @param out_elapsed_ms   Output: wall-clock time in ms
 * @return 1 on success, negative on error
 */
extern "C" int run_gpu_queue_stress_test(
    chi::PoolId pool_id,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 iterations,
    float *out_elapsed_ms) {

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) return -1;

  // Pause GPU orchestrator — cudaMallocHost/cudaMalloc on the default stream
  // implicitly syncs with all streams, which blocks on the persistent kernel.
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  // Client scratch backend (pinned host memory, 10MB per block)
  size_t scratch_size = static_cast<size_t>(client_blocks) * 10 * 1024 * 1024;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuShmMmap scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size,
                                 "/gpu_stress_scratch", 0))
    return -100;
  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(scratch_id, scratch_backend.data_,
                                 scratch_backend.data_capacity_);

  // Client heap backend for serialization scratch (4MB per block)
  size_t heap_size = static_cast<size_t>(client_blocks) * 4 * 1024 * 1024;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0))
    return -101;

  // Build GPU info for client kernel
  chi::IpcManagerGpuInfo gpu_info;
  gpu_info.backend = scratch_backend;
  gpu_info.gpu2gpu_queue = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0).gpu2gpu_queue;
  gpu_info.gpu2gpu_num_lanes = 1;

  // Pinned host memory for completion tracking
  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  volatile int *d_progress;
  cudaMallocHost((void **)&d_progress, sizeof(int) * total_warps);
  memset((void *)d_progress, 0, sizeof(int) * total_warps);

  // Launch on a dedicated stream — create BEFORE any cudaMemset to avoid
  // default-stream sync with the persistent orchestrator kernel.
  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  // Zero allocator headers so non-block-0 blocks spin-wait correctly.
  if (scratch_backend.data_)
    memset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  if (heap_backend.data_)
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);
  cudaEventRecord(ev_start, static_cast<cudaStream_t>(stream));

  gpu_queue_stress_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, iterations, client_blocks,
      d_done, d_progress);

  cudaEventRecord(ev_end, static_cast<cudaStream_t>(stream));

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "ERROR: stress kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    cudaFreeHost((void *)d_progress);
    hshm::GpuApi::DestroyStream(stream);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    return -201;
  }

  // Resume GPU orchestrator so it can process the client's tasks
  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
  fprintf(stderr, "[STRESS] %u warps × %u iters launched\n",
          total_warps, iterations);

  // Poll pinned host memory for completion (10s timeout)
  int timeout_us = 10000000;
  int elapsed_us = 0;
  int cur_done = 0;
  while (cur_done < static_cast<int>(total_warps) && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
    cur_done = __atomic_load_n(d_done, __ATOMIC_ACQUIRE);
  }

  bool completed = (cur_done == static_cast<int>(total_warps));

  float gpu_ms = static_cast<float>(elapsed_us) / 1000.0f;
  if (completed) {
    // Don't sync stream — the polling loop already confirmed all warps
    // finished via pinned host memory. Stream sync may hang if GPU
    // cleanup races with the orchestrator.
    cudaError_t ev_err = cudaEventQuery(ev_end);
    if (ev_err == cudaSuccess) {
      cudaEventElapsedTime(&gpu_ms, ev_start, ev_end);
    }
    fprintf(stderr, "[STRESS] Completed: %u warps in %.1f ms\n",
            total_warps, gpu_ms);
  }
  *out_elapsed_ms = gpu_ms;

  if (!completed) {
    fprintf(stderr, "TIMEOUT: d_done=%d/%u after %dms\n",
            cur_done, total_warps, elapsed_us / 1000);
    for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
      fprintf(stderr, "  warp[%u]: progress=%d\n", i, d_progress[i]);
    }
  }

  // Skip CUDA cleanup — cudaFreeHost/cudaEventDestroy may implicitly
  // sync with the persistent GPU orchestrator and block indefinitely.
  // These resources are cleaned up at CUDA context teardown.

  return completed ? 1 : -3;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
