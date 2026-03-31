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
 * GPU kernels for Part 3: GPU Task Submission tests
 * This file contains only GPU kernel code and is compiled as CUDA
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/task.h>
#include <chimaera/types.h>
#include <chimaera/local_task_archives.h>
#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/lightbeam/shm_transport.h>
#include <chrono>
#include <thread>

/**
 * GPU kernel that submits a task from within the kernel
 * Tests Part 3: GPU kernel calling NewTask and Send
 */
__global__ void gpu_submit_task_kernel(chi::IpcManagerGpu gpu_info,
                                       chi::PoolId pool_id, chi::u32 test_value,
                                       int *result) {
  *result = 100;  // Kernel started

  // Step 1: Initialize IPC manager (no queue needed for NewTask-only test)
  CHIMAERA_GPU_INIT(gpu_info);

  *result = 200;  // After CHIMAERA_GPU_INIT

  // Step 2: Create task using NewTask
  chi::TaskId task_id = chi::CreateTaskId();
  chi::PoolQuery query = chi::PoolQuery::Local();

  *result = 300;  // Before NewTask
  hipc::FullPtr<chimaera::MOD_NAME::GpuSubmitTask> task;
  task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                      task_id, pool_id, query, 0, test_value);

  if (task.ptr_ == nullptr) {
    *result = -1;  // NewTask failed
    return;
  }

  *result = 1;  // Success - NewTask works
}

/**
 * C++ wrapper function to run the GPU kernel test
 * This allows the CPU test file to call this without needing CUDA headers
 */
extern "C" int run_gpu_kernel_task_submission_test(chi::PoolId pool_id,
                                                   chi::u32 test_value) {
  // Create GPU memory backend using GPU-registered shared memory
  hipc::MemoryBackendId backend_id(2, 0);
  size_t gpu_memory_size = 10 * 1024 * 1024;  // 10MB
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_kernel_submit",
                            0)) {
    return -100;  // Backend init failed
  }

  // Allocate result on GPU
  int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
  int h_result = 0;
  hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));

  // Create IpcManagerGpu for kernel
  chi::IpcManagerGpu gpu_info(gpu_backend, nullptr);

  // Launch kernel on a dedicated stream (cudaDeviceSynchronize would block
  // on the persistent GPU orchestrator running on another stream)
  void *stream = hshm::GpuApi::CreateStream();
  gpu_submit_task_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, test_value, d_result);

  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::DestroyStream(stream);
    return -201;  // Kernel launch error
  }

  // Synchronize only this stream
  hshm::GpuApi::Synchronize(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::DestroyStream(stream);
    return -200;  // CUDA error
  }

  // Get result
  hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));

  // Cleanup
  hshm::GpuApi::Free(d_result);
  hshm::GpuApi::DestroyStream(stream);

  return h_result;
}

/**
 * GPU kernel that tests full end-to-end runtime roundtrip using client API:
 * GPU kernel calls AsyncGpuSubmit() -> worker processes -> Wait() -> verify
 */
__global__ void gpu_full_runtime_kernel(chi::IpcManagerGpu gpu_info,
                                         chi::PoolId pool_id,
                                         chi::u32 test_value,
                                         int *d_result,
                                         chi::u32 *d_result_value) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);
  chimaera::MOD_NAME::Client client(pool_id);
  auto future = client.AsyncGpuSubmit(chi::PoolQuery::Local(), 0, test_value);
  future.Wait();
  *d_result_value = future->result_value_;
  __threadfence_system();  // Ensure writes visible to CPU
  *d_result = 1;  // success
}

/**
 * C++ wrapper to launch the full runtime roundtrip GPU kernel.
 * GPU→GPU path: kernel uses Local() → SendGpuLocal() → GPU orchestrator processes.
 * Does NOT pause GPU orchestrator (it must be running to process GPU→GPU tasks).
 */
extern "C" int run_gpu_full_runtime_test(chi::PoolId pool_id,
                                          chi::u32 test_value,
                                          chi::u32 *out_result_value) {

  // Create GPU memory backend for kernel allocations
  hipc::MemoryBackendId backend_id(3, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024, "/gpu_rt_test", 0))
    return -100;

  // Register GPU backend memory for host-side ShmPtr resolution.
  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Set up IpcManagerGpuInfo for GPU→GPU path:
  // - gpu2cu_queue = nullptr (not using GPU→CPU in this test)
  // - cpu2gpu_queue = nullptr (not receiving CPU→GPU in this test)
  // - gpu2gpu_queue = GPU orchestrator's GPU→GPU queue
  chi::IpcManagerGpuInfo gpu_info;
  gpu_info.backend = gpu_backend;
  gpu_info.gpu2cpu_queue = nullptr;
  gpu_info.cpu2gpu_queue = nullptr;
  gpu_info.gpu2gpu_queue = CHI_IPC->GetGpuToGpuQueue(0);

  // Use pinned host memory so CPU can poll result directly without
  // cudaStreamSynchronize (which can hang with persistent GPU orchestrator).
  // Kernel must use __threadfence_system() before writing to ensure visibility.
  int *d_result;
  chi::u32 *d_rv;
  cudaMallocHost(&d_result, sizeof(int));
  cudaMallocHost(&d_rv, sizeof(chi::u32));
  *d_result = 0;
  *d_rv = 0;

  // Pause GPU orchestrator to free SMs for the test kernel launch,
  // then resume so the GPU orchestrator can process the GPU→GPU task.
  CHI_IPC->PauseGpuOrchestrator();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_full_runtime_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, test_value, d_result, d_rv);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_result);
    cudaFreeHost(d_rv);
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  // Resume GPU orchestrator so it can process the GPU→GPU task
  CHI_IPC->ResumeGpuOrchestrator();

  // Poll pinned host memory for kernel completion instead of
  // cudaStreamSynchronize (persistent GPU orchestrator causes stream sync to hang)
  int timeout_us = 10000000;  // 10 seconds
  int elapsed_us = 0;
  while (*d_result == 0 && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }

  if (*d_result == 0) {
    // Timeout: kernel didn't complete
    cudaFreeHost(d_result);
    cudaFreeHost(d_rv);
    hshm::GpuApi::DestroyStream(stream);
    return -3;  // Timeout
  }

  *out_result_value = *d_rv;
  int h_result = *d_result;
  cudaFree(d_result);
  cudaFree(d_rv);
  hshm::GpuApi::DestroyStream(stream);
  return h_result;
}

/**
 * CPU→GPU test: CPU calls SendToGpu to push task to GPU orchestrator.
 * GPU orchestrator's gpu::Worker dispatches to MOD_NAME GpuRuntime.
 * CPU polls FUTURE_COMPLETE and deserializes output.
 */
extern "C" int run_cpu_to_gpu_test(chi::PoolId pool_id,
                                    chi::u32 test_value,
                                    chi::u32 *out_result_value) {
  auto *ipc = CHI_IPC;

  // Create the task with LocalGpuBcast routing
  auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::LocalGpuBcast(),
      0, test_value);
  if (task.IsNull()) {
    return -1;  // NewTask failed
  }

  // Send to GPU orchestrator via to_gpu_queue
  auto future = ipc->SendGpuCpuCopy(task, 0);
  if (future.GetFutureShmPtr().IsNull()) {
    return -2;  // SendToGpu failed
  }

  // Poll for FUTURE_COMPLETE
  auto fshm_full = future.GetFutureShm();
  chi::FutureShm *fshm = fshm_full.ptr_;
  int attempts = 0;
  while (!fshm->flags_.Any(chi::FutureShm::FUTURE_COMPLETE)) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    ++attempts;
    if (attempts > 50000) {  // 5 second timeout
      return -3;  // Timeout
    }
  }

  // Deserialize output from FutureShm ring buffer
  hshm::lbm::LbmContext ctx;
  ctx.copy_space = fshm->copy_space;
  ctx.shm_info_ = &fshm->output_;

  fprintf(stderr, "[CPU2GPU-DIAG] output total_written=%zu copy_space_size=%zu flags=%u\n",
          (size_t)fshm->output_.total_written_.load(),
          (size_t)fshm->output_.copy_space_size_.load(),
          (unsigned)fshm->flags_.bits_.load());

  chi::priv::vector<char> recv_buf;
  recv_buf.reserve(256);
  chi::DefaultLoadArchive load_ar(recv_buf);
  hshm::lbm::ShmTransport::Recv(load_ar, ctx);
  load_ar.SetMsgType(chi::LocalMsgType::kSerializeOut);
  task.ptr_->SerializeOut(load_ar);

  *out_result_value = task->result_value_;
  return 1;  // Success
}

/**
 * GPU kernel: GPU submits task to CPU with ToLocalCpu routing.
 * Uses AsyncGpuSubmit → SendGpu → gpu_worker_queue → CPU worker processes.
 */
__global__ void gpu_to_cpu_kernel(chi::IpcManagerGpu gpu_info,
                                   chi::PoolId pool_id,
                                   chi::u32 test_value,
                                   int *d_result,
                                   chi::u32 *d_result_value) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);

  chimaera::MOD_NAME::Client client(pool_id);
  auto future = client.AsyncGpuSubmit(
      chi::PoolQuery::ToLocalCpu(), 0, test_value);
  future.Wait();

  *d_result_value = future->result_value_;
  *d_result = 1;  // success
}

/**
 * C++ wrapper to launch the GPU→CPU test kernel.
 * Sets up GPU backend and queue, launches kernel, waits for result.
 */
extern "C" int run_gpu_to_cpu_test(chi::PoolId pool_id,
                                    chi::u32 test_value,
                                    chi::u32 *out_result_value) {

  // Primary backend: GPU kernel task allocations (NewTask objects)
  hipc::MemoryBackendId backend_id(5, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024, "/gpu_to_cpu", 0))
    return -100;

  // FutureShm backend: GPU allocates FutureShm here (UVM, CPU+GPU accessible).
  // Separate from queue_backend so InitAllocTable doesn't overwrite the queue.
  hipc::MemoryBackendId g2c_backend_id(9, 0);
  hipc::GpuShmMmap g2c_backend;
  if (!g2c_backend.shm_init(g2c_backend_id, 4 * 1024 * 1024,
                              "/gpu_to_cpu_g2c", 0))
    return -104;

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Use the system's pre-existing GPU→CPU queue (backend 4000).
  // The CPU GPU worker (worker 2) already polls it via AssignGpuLanesToWorker
  // called during runtime startup. No need to register a custom queue.
  chi::IpcManagerGpuInfo gpu_info;
  gpu_info.backend = static_cast<hipc::MemoryBackend &>(gpu_backend);
  gpu_info.gpu2cpu_queue = CHI_IPC->GetGpuQueue(0);
  gpu_info.gpu2cpu_backend = static_cast<hipc::MemoryBackend &>(g2c_backend);

  int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
  chi::u32 *d_rv = hshm::GpuApi::Malloc<chi::u32>(sizeof(chi::u32));
  int h_result = 0;
  chi::u32 h_rv = 0;
  hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));
  hshm::GpuApi::Memcpy(d_rv, &h_rv, sizeof(chi::u32));

  // Clear any sticky CUDA error from previous tests
  cudaGetLastError();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_to_cpu_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, test_value, d_result, d_rv);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::Free(d_rv);
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  hshm::GpuApi::Synchronize(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::Free(d_rv);
    hshm::GpuApi::DestroyStream(stream);
    return -200;
  }

  hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
  hshm::GpuApi::Memcpy(&h_rv, d_rv, sizeof(chi::u32));

  *out_result_value = h_rv;
  hshm::GpuApi::Free(d_result);
  hshm::GpuApi::Free(d_rv);
  hshm::GpuApi::DestroyStream(stream);
  return h_result;
}

/**
 * CPU→GPU test via client API: CPU calls AsyncGpuSubmit with LocalGpuBcast routing.
 * Must be compiled in the GPU compilation unit where HSHM_ENABLE_CUDA=1 so
 * that Send() detects LocalGpuBcast and redirects to SendGpuCpuCopy().
 */
extern "C" int run_async_gpu_submit_local_gpu_bcast_test(
    chi::PoolId pool_id,
    chi::u32 test_value,
    chi::u32 *out_result_value) {
  chimaera::MOD_NAME::Client client(pool_id);
  auto future = client.AsyncGpuSubmit(
      chi::PoolQuery::LocalGpuBcast(), 0, test_value);

  // Diagnostic: check FutureShm state before waiting
  auto fshm_full = future.GetFutureShm();
  chi::FutureShm *fshm = fshm_full.ptr_;
  fprintf(stderr, "[ASYNC-BCAST-DIAG] fshm=%p future_shm_null=%d to_gpu_size=%zu\n",
          (void*)fshm,
          (int)future.GetFutureShmPtr().IsNull(),
          (size_t)CHI_IPC->cpu2gpu_queues_.size());
  if (fshm) {
    fprintf(stderr, "[ASYNC-BCAST-DIAG] flags=%u copy_from_client=%u in.size=%zu out.size=%zu\n",
            (unsigned)fshm->flags_.bits_.load(),
            (unsigned)chi::FutureShm::FUTURE_COPY_FROM_CLIENT,
            (size_t)fshm->input_.copy_space_size_.load(),
            (size_t)fshm->output_.copy_space_size_.load());
  }

  bool completed = future.Wait(10.0f);

  if (fshm) {
    fprintf(stderr, "[ASYNC-BCAST-DIAG] after wait: flags=%u total_written=%zu\n",
            (unsigned)fshm->flags_.bits_.load(),
            (size_t)fshm->output_.total_written_.load());
  }

  if (!completed) {
    return -3;  // Timeout
  }

  *out_result_value = future->result_value_;
  return 1;  // Success
}

/**
 * GPU kernel: GPU submits task to CPU with ToLocalCpu routing via AsyncGpuSubmit.
 * Validates the clean client API path (same as gpu_to_cpu_local but via AsyncGpuSubmit).
 */
__global__ void async_gpu_submit_to_local_cpu_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 test_value,
    int *d_result,
    chi::u32 *d_result_value) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);

  chimaera::MOD_NAME::Client client(pool_id);
  auto future = client.AsyncGpuSubmit(
      chi::PoolQuery::ToLocalCpu(), 0, test_value);
  future.Wait();

  *d_result_value = future->result_value_;
  *d_result = 1;  // success
}

/**
 * C++ wrapper to launch the async_gpu_submit_to_local_cpu kernel.
 * Sets up GPU backend and queue, launches kernel, waits for result.
 */
extern "C" int run_async_gpu_submit_to_local_cpu_test(
    chi::PoolId pool_id,
    chi::u32 test_value,
    chi::u32 *out_result_value) {

  // Primary backend: GPU kernel task allocations (NewTask objects)
  hipc::MemoryBackendId backend_id(7, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024,
                             "/async_gpu_to_cpu", 0))
    return -100;

  // FutureShm backend: GPU allocates FutureShm here (UVM, CPU+GPU accessible)
  hipc::MemoryBackendId g2c_backend_id(10, 0);
  hipc::GpuShmMmap g2c_backend;
  if (!g2c_backend.shm_init(g2c_backend_id, 4 * 1024 * 1024,
                              "/async_gpu_to_cpu_g2c", 0))
    return -104;

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Use the system's pre-existing GPU→CPU queue (backend 4000).
  // The CPU GPU worker (worker 2) already polls it via AssignGpuLanesToWorker
  // called during runtime startup. No need to register a custom queue.
  chi::IpcManagerGpuInfo gpu_info;
  gpu_info.backend = static_cast<hipc::MemoryBackend &>(gpu_backend);
  gpu_info.gpu2cpu_queue = CHI_IPC->GetGpuQueue(0);
  gpu_info.gpu2cpu_backend = static_cast<hipc::MemoryBackend &>(g2c_backend);

  int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
  chi::u32 *d_rv = hshm::GpuApi::Malloc<chi::u32>(sizeof(chi::u32));
  int h_result = 0;
  chi::u32 h_rv = 0;
  hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));
  hshm::GpuApi::Memcpy(d_rv, &h_rv, sizeof(chi::u32));

  // Clear any sticky CUDA error from previous tests
  cudaGetLastError();

  void *stream = hshm::GpuApi::CreateStream();
  async_gpu_submit_to_local_cpu_kernel<<<1, 1, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, test_value, d_result, d_rv);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::Free(d_rv);
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  hshm::GpuApi::Synchronize(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::Free(d_rv);
    hshm::GpuApi::DestroyStream(stream);
    return -200;
  }

  hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
  hshm::GpuApi::Memcpy(&h_rv, d_rv, sizeof(chi::u32));

  *out_result_value = h_rv;
  hshm::GpuApi::Free(d_result);
  hshm::GpuApi::Free(d_rv);
  hshm::GpuApi::DestroyStream(stream);
  return h_result;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
