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
 * GPU task trace tests (single CUDA source file).
 *
 * Compiled by nvcc so HSHM_ENABLE_CUDA=1 is active on host code.
 * Tests CPU->GPU, GPU->GPU, and GPU->CPU task paths with tracing.
 */

#include "../../simple_test.h"

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/task.h>
#include <chimaera/gpu/future.h>
#include <chimaera/gpu/gpu_info.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/memory/backend/gpu_shm_mmap.h>

#include <chrono>
#include <thread>
#include <cstdio>

using namespace std::chrono_literals;

// ============================================================================
// Shared initialization
// ============================================================================

static bool g_initialized = false;
static chi::PoolId g_pool_id(10000, 1);

/**
 * Initialize Chimaera server and create MOD_NAME pool.
 * Shared across all tests.
 */
static void EnsureInit() {
  if (g_initialized) return;

  fprintf(stderr, "[INIT] Initializing Chimaera server...\n");
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer);
  REQUIRE(success);
  g_initialized = true;
  fprintf(stderr, "[INIT] Waiting 500ms for GPU orchestrator\n");
  std::this_thread::sleep_for(500ms);

  auto *ipc = CHI_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() > 0);

  // Create MOD_NAME pool via CHI_CPU_IPC (not client.AsyncCreate,
  // because nvcc redefines CHI_IPC to nullptr on host).
  fprintf(stderr, "[INIT] Creating MOD_NAME pool (%u,%u)\n",
          g_pool_id.major_, g_pool_id.minor_);
  static chimaera::MOD_NAME::Client client(g_pool_id);
  {
    using CreateTask = chimaera::MOD_NAME::CreateTask;
    using CreateParams = chimaera::MOD_NAME::CreateParams;
    auto task = ipc->NewTask<CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId,
        chi::PoolQuery::Dynamic(),
        CreateParams::chimod_lib_name,
        std::string("gpu_trace_pool"),
        g_pool_id, &client);
    auto future = ipc->Send(task);
    future.Wait();
  }
  fprintf(stderr, "[INIT] Waiting 200ms for GPU container registration\n");
  std::this_thread::sleep_for(200ms);
  fprintf(stderr, "[INIT] Ready\n");
}

// ============================================================================
// Test 1: CPU -> GPU via PoolQuery::ToLocalGpu(0)
// ============================================================================

/**
 * CPU sends GpuSubmitTask to GPU via ToLocalGpu(0).
 *
 * Flow: NewTask -> Send() -> SendCpuToGpu -> cpu2gpu_queue ->
 *       GPU worker -> CDP RunTask -> GpuRuntime::GpuSubmit ->
 *       FUTURE_COMPLETE relay -> CPU polls pinned-host -> D2H copy
 *
 * GPU handler: result = (test_value * 3) + gpu_id
 */
TEST_CASE("cpu2gpu_trace", "[gpu][cpu2gpu][trace]") {
  fprintf(stderr, "\n=== cpu2gpu_trace START ===\n");
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  const chi::u32 test_value = 42;
  fprintf(stderr, "[TRACE] NewTask<GpuSubmitTask>(ToLocalGpu(0), val=%u)\n",
          test_value);
  auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), g_pool_id, chi::PoolQuery::ToLocalGpu(0),
      chi::u32(0), test_value);
  REQUIRE(!task.IsNull());
  fprintf(stderr, "[TRACE]   method=%u routing=%u parallelism=%u\n",
          (unsigned)task->method_,
          (unsigned)task->pool_query_.GetRoutingMode(),
          (unsigned)task->pool_query_.GetParallelism());

  fprintf(stderr, "[TRACE] Send(task)\n");
  auto future = ipc->Send(task);
  REQUIRE(!future.GetFutureShmPtr().IsNull());

  auto fshm_sptr = future.GetFutureShmPtr();
  auto sentinel = chi::gpu::FutureShm::GetCpu2GpuAllocId();
  REQUIRE(fshm_sptr.alloc_id_.major_ == sentinel.major_);

  auto *host_fshm = reinterpret_cast<chi::gpu::FutureShm *>(
      fshm_sptr.off_.load());
  fprintf(stderr, "[TRACE]   host_fshm=%p origin=%u device_task=0x%lx\n",
          (void *)host_fshm, host_fshm->origin_,
          (unsigned long)host_fshm->client_task_vaddr_);

  // Poll for FUTURE_COMPLETE
  fprintf(stderr, "[TRACE] Polling (10s timeout)...\n");
  auto t0 = std::chrono::steady_clock::now();
  chi::u32 flags_val = 0;
  size_t flags_off = offsetof(chi::gpu::FutureShm, flags_);
  while (!(flags_val & chi::gpu::FutureShm::FUTURE_COMPLETE)) {
    auto *p = reinterpret_cast<volatile chi::u32 *>(
        reinterpret_cast<char *>(host_fshm) + flags_off);
    flags_val = *p;
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= 10.0f) {
      fprintf(stderr, "[TRACE] FAIL: Timeout flags=%u\n", flags_val);
      REQUIRE(false);
    }
    std::this_thread::yield();
  }
  float ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "[TRACE] COMPLETE in %.2f ms\n", ms);

  // D2H copy-back
  auto *dev_task = reinterpret_cast<chimaera::MOD_NAME::GpuSubmitTask *>(
      host_fshm->task_device_ptr_);
  chimaera::MOD_NAME::GpuSubmitTask host_copy;
  hshm::GpuApi::Memcpy(
      reinterpret_cast<char *>(&host_copy),
      reinterpret_cast<const char *>(dev_task),
      sizeof(chimaera::MOD_NAME::GpuSubmitTask));

  chi::u32 expected = (test_value * 3) + 0;
  fprintf(stderr, "[TRACE] result=%u expected=%u\n",
          host_copy.result_value_, expected);
  REQUIRE(host_copy.result_value_ == expected);

  // Cleanup: pause orchestrator so cudaFree won't block
  ipc->GetGpuIpcManager()->PauseGpuOrchestrator();
  hshm::GpuApi::Free(dev_task);
  hshm::GpuApi::FreeHost(host_fshm);
  ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();

  fprintf(stderr, "=== cpu2gpu_trace PASS ===\n");
}

// ============================================================================
// GPU kernels for GPU->GPU and GPU->CPU tests
// ============================================================================

/**
 * GPU kernel: submits GpuSubmitTask with Local() routing (GPU->GPU).
 * GPU orchestrator processes the task on GPU.
 * GPU handler: result = (test_value * 3) + gpu_id
 */
__global__ void gpu2gpu_kernel(chi::IpcManagerGpu gpu_info,
                               chi::PoolId pool_id,
                               chi::u32 test_value,
                               int *d_result,
                               chi::u32 *d_result_value) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);

  auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
      chi::u32(0), test_value);
  auto future = CHI_IPC->Send(task);
  future.Wait();

  *d_result_value = future->result_value_;
  __threadfence_system();
  *d_result = 1;
}

/**
 * GPU kernel: submits GpuSubmitTask with ToLocalCpu() routing (GPU->CPU).
 * CPU runtime processes the task.
 * CPU handler: result = (test_value * 2) + gpu_id
 */
__global__ void gpu2cpu_kernel(chi::IpcManagerGpu gpu_info,
                               chi::PoolId pool_id,
                               chi::u32 test_value,
                               int *d_result,
                               chi::u32 *d_result_value) {
  *d_result = 0;
  CHIMAERA_GPU_INIT(gpu_info);

  auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::ToLocalCpu(),
      chi::u32(0), test_value);
  auto future = CHI_IPC->Send(task);
  future.Wait();

  *d_result_value = future->result_value_;
  __threadfence_system();
  *d_result = 1;
}

// ============================================================================
// Test 2: GPU -> GPU via PoolQuery::Local()
// ============================================================================

/**
 * GPU kernel submits GpuSubmitTask with Local() -> GPU orchestrator processes.
 *
 * Flow: GPU kernel -> CHIMAERA_GPU_INIT -> AsyncGpuSubmit(Local()) ->
 *       gpu2gpu_queue -> GPU worker -> CDP RunTask -> GpuRuntime::GpuSubmit ->
 *       RecvGpu -> kernel reads result
 *
 * GPU handler: result = (test_value * 3) + gpu_id
 */
TEST_CASE("gpu2gpu_trace", "[gpu][gpu2gpu][trace]") {
  fprintf(stderr, "\n=== gpu2gpu_trace START ===\n");
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->CreateGpuAllocator(10 * 1024 * 1024, 0);
  const chi::u32 test_value = 55;

  // Pause orchestrator — cudaMallocHost and cudaLaunchKernel are
  // device-synchronizing and deadlock with the persistent CDP kernel.
  ipc->GetGpuIpcManager()->PauseGpuOrchestrator();

  int *d_result;
  chi::u32 *d_rv;
  cudaMallocHost(&d_result, sizeof(int));
  cudaMallocHost(&d_rv, sizeof(chi::u32));
  *d_result = 0;
  *d_rv = 0;

  // Launch kernel while orchestrator is paused
  cudaGetLastError();
  void *stream = hshm::GpuApi::CreateStream();
  gpu2gpu_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, g_pool_id, test_value, d_result, d_rv);
  cudaError_t err = cudaGetLastError();
  REQUIRE(err == cudaSuccess);

  // Resume orchestrator to process the gpu2gpu task
  ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();

  // Poll pinned-host for kernel completion (10s timeout)
  auto t0 = std::chrono::steady_clock::now();
  while (*d_result == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= 10.0f) {
      fprintf(stderr, "[TRACE] FAIL: Timeout d_result=%d d_rv=%u\n",
              *d_result, *d_rv);
      REQUIRE(false);
    }
  }
  float ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

  chi::u32 result = *d_rv;
  chi::u32 expected = (test_value * 3) + 0;
  fprintf(stderr, "[TRACE] COMPLETE in %.2f ms result=%u expected=%u\n",
          ms, result, expected);
  REQUIRE(result == expected);

  // Cleanup
  ipc->GetGpuIpcManager()->PauseGpuOrchestrator();
  cudaFreeHost(d_result);
  cudaFreeHost(d_rv);
  hshm::GpuApi::DestroyStream(stream);
  ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();

  fprintf(stderr, "=== gpu2gpu_trace PASS ===\n");
}

// ============================================================================
// Test 3: GPU -> CPU via PoolQuery::ToLocalCpu()
// ============================================================================

/**
 * GPU kernel submits GpuSubmitTask with ToLocalCpu() -> CPU worker processes.
 *
 * Flow: GPU kernel -> CHIMAERA_GPU_INIT -> AsyncGpuSubmit(ToLocalCpu()) ->
 *       gpu2cpu_queue -> CPU GPU worker -> MOD_NAME::Runtime::GpuSubmit ->
 *       completion -> GPU kernel reads result
 *
 * CPU handler: result = (test_value * 2) + gpu_id
 */
TEST_CASE("gpu2cpu_trace", "[gpu][gpu2cpu][trace]") {
  fprintf(stderr, "\n=== gpu2cpu_trace START ===\n");
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  // Create GPU allocator backend and get populated IpcManagerGpuInfo
  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->CreateGpuAllocator(10 * 1024 * 1024, 0);

  const chi::u32 test_value = 88;

  // Pause orchestrator FIRST — cudaMallocHost is device-synchronizing
  // and would deadlock with the persistent GPU orchestrator kernel.
  ipc->GetGpuIpcManager()->PauseGpuOrchestrator();

  // Allocate pinned-host result slots (GPU writes, CPU polls)
  int *d_result;
  chi::u32 *d_rv;
  cudaMallocHost(&d_result, sizeof(int));
  cudaMallocHost(&d_rv, sizeof(chi::u32));
  *d_result = 0;
  *d_rv = 0;

  fprintf(stderr, "[TRACE] Launching gpu2cpu_kernel(val=%u)\n", test_value);
  cudaGetLastError();  // Clear sticky errors
  void *stream = hshm::GpuApi::CreateStream();
  gpu2cpu_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, g_pool_id, test_value, d_result, d_rv);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[TRACE] FAIL: kernel launch error %d\n", (int)err);
    ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();
  }
  REQUIRE(err == cudaSuccess);

  // Resume orchestrator — it's not needed for GPU→CPU processing
  // (CPU worker handles gpu2cpu_queue), but the test kernel needs it
  // not to block SM resources.
  ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();

  // Poll pinned-host for kernel completion (10s timeout)
  fprintf(stderr, "[TRACE] Polling kernel result (10s timeout)...\n");
  auto t0 = std::chrono::steady_clock::now();
  while (*d_result == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= 10.0f) {
      fprintf(stderr, "[TRACE] FAIL: Timeout d_result=%d d_rv=%u\n",
              *d_result, *d_rv);
      REQUIRE(false);
    }
  }
  float ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

  chi::u32 expected = (test_value * 2) + 0;
  fprintf(stderr, "[TRACE] COMPLETE in %.2f ms result=%u expected=%u rc=%d\n",
          ms, *d_rv, expected, *d_result);
  REQUIRE(*d_result == 1);
  REQUIRE(*d_rv == expected);

  // Cleanup
  ipc->GetGpuIpcManager()->PauseGpuOrchestrator();
  cudaFreeHost(d_result);
  cudaFreeHost(d_rv);
  hshm::GpuApi::DestroyStream(stream);
  ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();

  fprintf(stderr, "=== gpu2cpu_trace PASS ===\n");
}

SIMPLE_TEST_MAIN()
