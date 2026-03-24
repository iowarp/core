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
 * GPU kernel for testing AsyncCreate from a GPU kernel.
 *
 * Uses admin task directly (GetOrCreatePoolTask) to avoid pulling in
 * CTE-specific bdev headers that are not GPU-compatible. The kernel
 * calls the Create task with ToLocalCpu() routing so it gets processed
 * by a CPU worker in the admin pool.
 */

#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_tasks.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/memory/backend/gpu_shm_mmap.h>
#include <hermes_shm/memory/backend/gpu_malloc.h>
#include <time.h>

/**
 * Minimal GPU-safe CreateParams for wrp_cte_core.
 * Only contains chimod_lib_name (no bdev/STL types).
 * serialize() is empty — chimod_params_ will be empty string.
 */
struct GpuCoreCreateParams {
  static constexpr const char *chimod_lib_name = "wrp_cte_core";

  HSHM_CROSS_FUN GpuCoreCreateParams() = default;

  template <class Archive>
  HSHM_CROSS_FUN void serialize(Archive &ar) {
    (void)ar;
  }
};

// GPU-callable CreateTask type: routes to admin's kGetOrCreatePool handler
using GpuCoreCreateTask =
    chimaera::admin::GetOrCreatePoolTask<GpuCoreCreateParams>;

/** Trivial kernel to test if any CUDA kernel can run at all. */
__global__ void trivial_test_kernel(int *d_result) {
  *d_result = 42;
}

/**
 * GPU kernel: calls AsyncCreate from device code with ToLocalCpu() routing.
 *
 * d_result encodes progress stages:
 *   0  = kernel started
 *   10 = after CHIMAERA_GPU_INIT
 *   20 = after NewTask (task != null)
 *   30 = after Send (entering Wait)
 *   40 = Wait completed successfully
 *   1  = completed successfully
 *   -1 = NewTask returned null
 *   -2 = Send returned null future
 *   -3 = Wait timed out (d_stop set by CPU)
 *
 * d_stop: CPU sets this to 1 to signal the kernel to exit early.
 */
__global__ void gpu_create_kernel(chi::IpcManagerGpu gpu_info,
                                  const char *pool_name,
                                  chi::PoolId target_pool_id,
                                  int *d_result,
                                  int *d_return_code,
                                  volatile int *d_stop) {
  // Use volatile stores so values are visible to CPU via UVM polling
  volatile int *vresult = d_result;
  *vresult = 0;
  *d_return_code = -1;

  CHIMAERA_GPU_INIT(gpu_info);
  *vresult = 10;  // CHIMAERA_GPU_INIT completed

  // Construct and submit a GetOrCreatePool task directly via admin pool.
  // Note: use chi::PoolId(1,0) directly — chi::kAdminPoolId is not device-linked
  auto task = CHI_IPC->NewTask<GpuCoreCreateTask>(
      chi::CreateTaskId(),
      chi::PoolId(1, 0),              // kAdminPoolId inline
      chi::PoolQuery::ToLocalCpu(),
      "wrp_cte_core",                 // chimod_lib_name string literal
      pool_name,                      // const char*
      target_pool_id,
      static_cast<chi::ContainerClient *>(nullptr));

  if (task.IsNull()) {
    *vresult = -1;
    return;
  }
  *vresult = 20;  // NewTask succeeded

  auto future = CHI_IPC->Send(task);
  if (future.IsNull()) {
    *vresult = -2;
    return;
  }
  *vresult = 30;  // Send completed, entering wait loop

  // Manual wait loop with stop flag so CPU can terminate us if needed.
  // This replaces future.Wait() to avoid an unbreakable infinite loop.
  {
    auto fshm_full = future.GetFutureShm();
    chi::FutureShm *fshm = fshm_full.ptr_;
    while (fshm && !fshm->flags_.AnySystem(chi::FutureShm::FUTURE_COMPLETE)) {
      // Use atomicAdd_system(0) to bypass GPU L2 cache so we see the
      // CPU-written stop flag without a stale L2-cached value.
      int stop = atomicAdd_system(const_cast<int *>(d_stop), 0);
      if (stop) {
        *vresult = -3;  // CPU told us to stop
        return;
      }
      // Yield to avoid starving other GPU work
      HSHM_THREAD_MODEL->Yield();
    }
  }
  *vresult = 40;  // Wait completed

  *d_return_code = static_cast<int>(future->GetReturnCode());
  *vresult = 1;  // kernel completed
}

/**
 * C++ wrapper: sets up GPU backend and queues, launches the kernel, polls result.
 * Returns 1 on success, negative on error.
 *
 * Return codes:
 *   1   = success
 *   -1  = gpu_backend init failed
 *  -100 = gpu_backend shm_init failed
 *  -104 = g2c_backend shm_init failed
 *  -105 = gpu_priv_backend shm_init failed
 *  -200 = CUDA error after launch
 *  -201 = CUDA kernel launch error
 *  -110 = kernel stuck after CHIMAERA_GPU_INIT (stage 10)
 *  -120 = kernel stuck after NewTask (stage 20)
 *  -130 = kernel stuck in Wait (stage 30) — CPU worker never responded
 */
extern "C" int run_gpu_create_test(const char *pool_name,
                                   chi::PoolId target_pool_id,
                                   int *out_return_code) {
  fprintf(stderr, "[GPU_CREATE_DEBUG] run_gpu_create_test started\n"); fflush(stderr);

  // Primary backend for GPU kernel task allocations
  hipc::MemoryBackendId backend_id(20, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024,
                             "/gpu_create_test", 0))
    return -100;

  // FutureShm backend (UVM, CPU+GPU accessible)
  hipc::MemoryBackendId g2c_backend_id(21, 0);
  hipc::GpuShmMmap g2c_backend;
  if (!g2c_backend.shm_init(g2c_backend_id, 4 * 1024 * 1024,
                              "/gpu_create_g2c", 0))
    return -104;

  // GPU heap backend (device memory, for BuddyAllocator serialization scratch)
  hipc::MemoryBackendId heap_backend_id(22, 0);
  hipc::GpuMalloc gpu_priv_backend;
  if (!gpu_priv_backend.shm_init(heap_backend_id, 4 * 1024 * 1024,
                                  "/gpu_create_heap", 0))
    return -105;

  fprintf(stderr, "[GPU_CREATE_DEBUG] backends initialized\n"); fflush(stderr);

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Set up IpcManagerGpuInfo for GPU→CPU path
  chi::IpcManagerGpuInfo gpu_info;
  gpu_info.backend = static_cast<hipc::MemoryBackend &>(gpu_backend);
  gpu_info.gpu2cpu_queue = CHI_IPC->GetGpuQueue(0);
  gpu_info.gpu2cpu_backend = static_cast<hipc::MemoryBackend &>(g2c_backend);
  gpu_info.gpu_priv_backend = static_cast<hipc::MemoryBackend &>(gpu_priv_backend);

  fprintf(stderr, "[GPU_CREATE_DEBUG] gpu_info set up, launching kernel\n"); fflush(stderr);

  // Allocate result, return_code, and stop flag in MANAGED memory so
  // CPU can poll d_result and set d_stop without cudaStreamSynchronize.
  int *d_result = nullptr;
  int *d_return_code = nullptr;
  int *d_stop = nullptr;
  cudaError_t ma1 = cudaMallocManaged(&d_result, sizeof(int));
  cudaError_t ma2 = cudaMallocManaged(&d_return_code, sizeof(int));
  cudaError_t ma3 = cudaMallocManaged(&d_stop, sizeof(int));
  fprintf(stderr, "[GPU_CREATE_DEBUG] cudaMallocManaged: %d %d %d, ptrs: %p %p %p\n",
          ma1, ma2, ma3, d_result, d_return_code, d_stop); fflush(stderr);
  if (!d_result || !d_return_code || !d_stop) return -106;
  *d_result = 0;
  *d_return_code = -1;
  *d_stop = 0;
  // Note: NO cudaDeviceSynchronize() here — a persistent orchestrator GPU kernel
  // may be running, and synchronizing would block forever. UVM ensures the
  // initial values are visible to the GPU kernel via hardware coherence.

  // Clear sticky CUDA errors from previous tests
  cudaGetLastError();

  // Use cudaStreamNonBlocking to avoid implicit serialization with the default
  // stream (stream 0). The orchestrator kernel runs on a blocking stream; if
  // we also use a blocking stream, we'd implicitly wait for the default stream,
  // which could have pending work. Non-blocking streams are independent.
  cudaStream_t stream_handle;
  cudaError_t sc_err = cudaStreamCreateWithFlags(&stream_handle,
                                                  cudaStreamNonBlocking);
  fprintf(stderr, "[GPU_CREATE_DEBUG] stream created: %p err=%d\n",
          (void*)stream_handle, (int)sc_err); fflush(stderr);
  if (sc_err != cudaSuccess) return -202;

  // Copy pool_name to managed memory so the GPU kernel can dereference it.
  // String literals in host .rodata are NOT accessible from GPU device code.
  char *d_pool_name = nullptr;
  size_t pool_name_len = strlen(pool_name) + 1;
  cudaMallocManaged(&d_pool_name, pool_name_len);
  memcpy(d_pool_name, pool_name, pool_name_len);

  fprintf(stderr, "[GPU_CREATE_DEBUG] launching main kernel\n"); fflush(stderr);
  gpu_create_kernel<<<1, 1, 0, stream_handle>>>(
      gpu_info, d_pool_name, target_pool_id,
      d_result, d_return_code, (volatile int *)d_stop);

  cudaError_t launch_err = cudaGetLastError();
  fprintf(stderr, "[GPU_CREATE_DEBUG] kernel launched, err=%s\n",
          cudaGetErrorString(launch_err)); fflush(stderr);
  if (launch_err != cudaSuccess) {
    cudaFree(d_result);
    cudaFree(d_return_code);
    cudaFree(d_stop);
    cudaStreamDestroy(stream_handle);
    return -201;
  }

  fprintf(stderr, "[GPU_CREATE_DEBUG] entering poll loop\n"); fflush(stderr);

  // Poll d_result from CPU using UVM coherence to detect which stage the
  // kernel is stuck at. We poll for up to 20 seconds total.
  // Stage values: 0=start, 10=init, 20=NewTask, 30=Send, 40=Wait, 1=done
  // Negative: error (-1=NewTask null, -2=Send null future, -3=stopped)
  int last_printed_stage = -99;
  for (int poll_ms = 0; poll_ms < 20000; poll_ms += 100) {
    struct timespec ts = {0, 100 * 1000000L};
    nanosleep(&ts, nullptr);
    int stage = *((volatile int *)d_result);
    if (stage != last_printed_stage) {
      fprintf(stderr, "[GPU_CREATE_DEBUG] stage=%d (poll_ms=%d)\n",
              stage, poll_ms);
      fflush(stderr);
      last_printed_stage = stage;
    }
    // Terminal states: completed (1), error (negative), or Wait done (40)
    if (stage == 1 || stage < 0 || stage == 40) break;
  }

  int h_result = *((volatile int *)d_result);
  int h_return_code = *((volatile int *)d_return_code);
  fprintf(stderr, "[GPU_CREATE_DEBUG] final: h_result=%d h_return_code=%d\n",
          h_result, h_return_code);
  fflush(stderr);

  if (h_result != 1 && h_result >= 0) {
    // Kernel did not complete within timeout — encode hang stage as -(stage+100)
    // e.g. -130 = hung at stage 30 (CPU worker never responded to GPU Send)
    int hung_at = h_result;
    *out_return_code = h_return_code;
    // Use _exit() to immediately terminate without calling atexit handlers
    // (avoids CHIMAERA_FINALIZE and CUDA cleanup hanging on the stuck kernel)
    fprintf(stderr, "[GPU_CREATE_DEBUG] kernel hung at stage %d, aborting\n",
            hung_at);
    fflush(stderr);
    _exit(-(hung_at + 100));
  }

  *out_return_code = h_return_code;
  cudaFree(d_pool_name);
  cudaFree(d_result);
  cudaFree(d_return_code);
  cudaFree(d_stop);
  cudaStreamDestroy(stream_handle);
  return h_result;
}
