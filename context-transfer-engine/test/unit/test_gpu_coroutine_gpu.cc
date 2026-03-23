/*
 * GPU kernels for coroutine subtask spawning tests.
 *
 * Tests the GPU→GPU dispatch path and coroutine suspend/resume machinery.
 * Compiled via add_cuda_library (clang-cuda dual-pass).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <thread>
#include <chrono>

/**
 * Test 1: Basic GPU→GPU leaf task.
 * GPU kernel sends MOD_NAME::GpuSubmit via gpu2gpu queue and waits.
 * Tests: task creation, serialization, gpu2gpu dispatch, deserialization.
 */
__global__ void gpu_leaf_task_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    chi::PoolId pool_id,
    int *d_result) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);

  auto *ipc = CHI_IPC;
  printf("[GPU kernel] Calling AsyncGpuSubmit(gpu_id=0, test_value=7)\n");
  auto sub = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
      chi::u32(0), chi::u32(7));
  auto future = ipc->Send(sub);

  if (future.IsNull()) {
    printf("[GPU kernel] future is null\n");
    *d_result = -2;
    return;
  }

  printf("[GPU kernel] Waiting for GpuSubmit\n");
  future.Wait();
  printf("[GPU kernel] GpuSubmit done, rc=%d result=%u\n",
         (int)future->return_code_, (unsigned)future->result_value_);

  // GpuSubmit computes: test_value * 3 + gpu_id = 7*3+0 = 21
  if (future->return_code_ == 0 && future->result_value_ == 21) {
    *d_result = 1;  // Success
  } else {
    *d_result = -3;
  }
}

extern "C" int run_gpu_leaf_task_test(chi::PoolId pool_id) {
  // GPU memory backend for kernel allocations
  hipc::MemoryBackendId backend_id(30, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024,
                             "/gpu_coro_leaf", 0))
    return -100;

  // GPU heap for serialization
  hipc::MemoryBackendId heap_id(31, 0);
  hipc::GpuMalloc gpu_heap;
  if (!gpu_heap.shm_init(heap_id, 4 * 1024 * 1024, "", 0))
    return -102;

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpuInfo gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;
  gpu_info.gpu_priv_backend = gpu_heap;

  // Pinned result
  int *d_result;
  cudaMallocHost(&d_result, sizeof(int));
  *d_result = 0;

  // Pause orchestrator, launch kernel, resume
  CHI_IPC->PauseGpuOrchestrator();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_leaf_task_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, d_result);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  CHI_IPC->ResumeGpuOrchestrator();

  // Poll for completion (10s timeout)
  for (int i = 0; i < 100000 && *d_result == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  int result = *d_result;
  hshm::GpuApi::DestroyStream(stream);
  return (result == 0) ? -4 : result;  // 0 = timeout
}

/**
 * Test 2: Coroutine subtask spawning.
 * GPU kernel sends MOD_NAME::SubtaskTest via gpu2gpu queue.
 * SubtaskTest's GPU Run() co_awaits GpuSubmit, testing
 * the full coroutine suspend/resume path inside the GPU worker.
 */
__global__ void gpu_subtask_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    chi::PoolId pool_id,
    chi::u32 test_value,
    int *d_result,
    chi::u32 *d_result_value) {
  *d_result = 0;
  *d_result_value = 0;
  CHIMAERA_GPU_INIT(gpu_info);

  auto *ipc = CHI_IPC;
  printf("[GPU kernel] Calling AsyncSubtaskTest(%u)\n", (unsigned)test_value);
  auto sub = ipc->NewTask<chimaera::MOD_NAME::SubtaskTestTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(), test_value);
  auto future = ipc->Send(sub);

  if (future.IsNull()) {
    printf("[GPU kernel] subtask future is null\n");
    *d_result = -2;
    return;
  }

  printf("[GPU kernel] Waiting for SubtaskTest\n");
  future.Wait();
  printf("[GPU kernel] SubtaskTest done, rc=%d result=%u\n",
         (int)future->return_code_, (unsigned)future->result_value_);

  *d_result_value = future->result_value_;
  __threadfence_system();
  *d_result = (future->return_code_ == 0) ? 1 : -3;
}

extern "C" int run_gpu_subtask_test(chi::PoolId pool_id,
                                     chi::u32 test_value,
                                     chi::u32 *out_result_value) {
  // GPU memory backend for kernel allocations
  hipc::MemoryBackendId backend_id(32, 0);
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, 10 * 1024 * 1024,
                             "/gpu_coro_sub", 0))
    return -100;

  // GPU heap for serialization
  hipc::MemoryBackendId heap_id(33, 0);
  hipc::GpuMalloc gpu_heap;
  if (!gpu_heap.shm_init(heap_id, 4 * 1024 * 1024, "", 0))
    return -102;

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpuInfo gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;
  gpu_info.gpu_priv_backend = gpu_heap;

  // Pinned results
  int *d_result;
  chi::u32 *d_rv;
  cudaMallocHost(&d_result, sizeof(int));
  cudaMallocHost(&d_rv, sizeof(chi::u32));
  *d_result = 0;
  *d_rv = 0;

  // Pause orchestrator, launch kernel, resume
  CHI_IPC->PauseGpuOrchestrator();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_subtask_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, test_value, d_result, d_rv);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return -201;
  }

  CHI_IPC->ResumeGpuOrchestrator();

  // Poll for completion (10s timeout)
  for (int i = 0; i < 100000 && *d_result == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  *out_result_value = *d_rv;
  int result = *d_result;
  hshm::GpuApi::DestroyStream(stream);
  return (result == 0) ? -4 : result;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
