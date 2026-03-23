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
 * GPU kernels for GPU-initiated PutBlob/GetBlob tests.
 *
 * These kernels run on the GPU, create CTE tasks using the Client API
 * (AsyncPutBlob / AsyncGetBlob), and await completion via Future::Wait().
 * This exercises the full GPU->GPU roundtrip: GPU client kernel submits
 * tasks via SendGpu, the GPU orchestrator dispatches to GpuRuntime, and
 * results are returned through FutureShm completion.
 *
 * Compiled via add_cuda_library (clang-cuda dual-pass).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

// cereal type headers must precede core_client.h so that SendZmq
// instantiations (compiled on the host pass) can serialize types like
// std::vector<std::string> used by ListTargetsTask::SerializeOut.
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <thread>
#include <chrono>

/**
 * GPU kernel: PutBlob + GetBlob roundtrip initiated from GPU.
 *
 * Uses the CTE Client API (AsyncPutBlob, AsyncGetBlob) which internally
 * call NewTask + Send (dispatched to SendGpu on the GPU path).
 * Future::Wait() handles the completion wait and output deserialization.
 */
__global__ void gpu_putblob_getblob_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    char *blob_data,
    char *out_data,
    chi::u64 blob_size,
    int *d_result) {
  *d_result = -1;
  __threadfence_system();
  CHIMAERA_GPU_INIT(gpu_info);

  wrp_cte::core::Client client(pool_id);

  // PutBlob
  auto put_future = client.AsyncPutBlob(
      tag_id, "gpu_initiated_blob",
      /*offset=*/chi::u64(0), blob_size,
      hipc::ShmPtr<>::FromRaw(blob_data), /*score=*/0.5f,
      wrp_cte::core::Context(),
      /*flags=*/chi::u32(0),
      chi::PoolQuery::Local());
  if (put_future.IsNull()) {
    *d_result = -2;
    __threadfence_system();
    return;
  }
  put_future.Wait();
  chi::u32 put_rc = put_future->GetReturnCode();
  if (put_rc != 0) {
    *d_result = -10 - (int)put_rc;
    __threadfence_system();
    return;
  }

  // GetBlob into out_data
  auto get_future = client.AsyncGetBlob(
      tag_id, "gpu_initiated_blob",
      /*offset=*/chi::u64(0), blob_size,
      /*flags=*/chi::u32(0),
      hipc::ShmPtr<>::FromRaw(out_data),
      chi::PoolQuery::Local());
  if (get_future.IsNull()) {
    *d_result = -3;
    __threadfence_system();
    return;
  }
  get_future.Wait();
  chi::u32 get_rc = get_future->GetReturnCode();
  if (get_rc != 0) {
    *d_result = -20 - (int)get_rc;
    __threadfence_system();
    return;
  }

  // Verify data roundtrip
  __threadfence_system();  // Ensure all writes visible
  int mismatches = 0;
  for (chi::u64 i = 0; i < blob_size; ++i) {
    if (out_data[i] != blob_data[i]) {
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    *d_result = -4;
    __threadfence_system();
    return;
  }

  *d_result = 1;  // Success
  __threadfence_system();
}

/**
 * Host wrapper: launches the GPU-initiated PutBlob+GetBlob kernel.
 *
 * Sets up GPU memory backends, fills a pinned source buffer with 0xAB,
 * pauses/resumes the GPU orchestrator around the kernel launch, and polls
 * pinned host memory for the result (avoids cudaStreamSynchronize deadlock
 * with the persistent GPU orchestrator kernel).
 *
 * @param pool_id   CTE core pool ID
 * @param tag_id    Tag ID for the blob
 * @param blob_size Size of the test blob
 * @return 1 on success, negative on error
 */
extern "C" int run_gpu_initiated_putblob_getblob_test(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u64 blob_size) {

  // Create GPU memory backend for kernel allocations (NewTask, FutureShm, etc.)
  hipc::MemoryBackendId backend_id(20, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024,
                             "/cte_gpu_initiated", 0))
    return -100;

  // Register backend so host-side ShmPtr resolution works
  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Allocate GPU heap for CHI_PRIV_ALLOC (BuddyAllocator for serialization)
  hipc::MemoryBackendId heap_id(21, 0);
  hipc::GpuMalloc gpu_heap;
  if (!gpu_heap.shm_init(heap_id, 4 * 1024 * 1024, "", 0))
    return -102;

  // Use GetClientGpuInfo for complete queue/backend setup, then override
  // the primary backend and heap backend with our custom ones
  chi::IpcManagerGpuInfo gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;
  gpu_info.gpu_priv_backend = gpu_heap;

  // Allocate UVM buffers (accessible from both CPU and GPU)
  // Using cudaMallocManaged so device-scope fences ensure cross-warp
  // visibility (pinned host memory would require system-scope fences).
  char *blob_data = nullptr;
  char *out_data = nullptr;
  cudaMallocManaged(&blob_data, blob_size);
  cudaMallocManaged(&out_data, blob_size);
  if (!blob_data || !out_data) return -101;

  // Fill source with test pattern, zero output
  memset(blob_data, 0xAB, blob_size);
  memset(out_data, 0x00, blob_size);

  // Result in pinned memory (CPU polls it instead of cudaStreamSync)
  // Use volatile to prevent compiler from caching the read in the poll loop
  volatile int *d_result;
  cudaMallocHost(const_cast<int **>(&d_result), sizeof(int));
  *d_result = 0;

  // Pause GPU orchestrator to free SMs for kernel launch, then resume
  CHI_IPC->PauseGpuOrchestrator();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_putblob_getblob_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, tag_id, blob_data, out_data, blob_size,
      const_cast<int *>(d_result));

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  // Resume GPU orchestrator so it processes the GPU->GPU tasks
  CHI_IPC->ResumeGpuOrchestrator();

  // Poll pinned host memory for kernel completion
  int timeout_us = 10000000;  // 10 seconds
  int elapsed_us = 0;
  while (*d_result != 1 && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }

  int result = *d_result;

  // Sync the client stream if kernel is still running
  cudaError_t sync_err = cudaStreamQuery(static_cast<cudaStream_t>(stream));
  if (sync_err == cudaErrorNotReady) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    result = *d_result;
  }

  hshm::GpuApi::DestroyStream(stream);
  // Intentional leak of pinned buffers: cudaFreeHost blocks on persistent kernel
  return (result == 0) ? -4 : result;  // 0 means timeout
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
