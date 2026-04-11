/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * GPU-compiled wrapper for cross-warp parallelism validation test.
 * Must be compiled as CUDA so Send() detects ToLocalGpu routing
 * and redirects to SendToGpu().
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/task.h>
#include <hermes_shm/util/gpu_api.h>

// Persistent device counter — allocated once, never freed, so the
// persistent orchestrator kernel can always safely atomicAdd on it.
static unsigned int *g_d_counter = nullptr;

/**
 * Run the cross-warp parallelism test.
 *
 * @param pool_id       Pool ID of the MOD_NAME container
 * @param parallelism   Number of GPU threads to use (e.g., 2048)
 * @param out_counter   Output: number of lanes that executed the handler
 * @return 1 on success, negative on error
 */
extern "C" int run_gpu_parallelism_test(
    chi::PoolId pool_id,
    chi::u32 parallelism,
    chi::u32 *out_counter) {

  // Allocate device counter once — never freed so the persistent
  // orchestrator kernel can always safely access it via atomicAdd.
  if (!g_d_counter) {
    cudaError_t err = cudaMalloc(&g_d_counter, sizeof(unsigned int));
    if (err != cudaSuccess) {
      return -100;
    }
  }
  cudaMemset(g_d_counter, 0, sizeof(unsigned int));

  // Direct NewTask/Send — use CHI_CPU_IPC (not CHI_IPC which is nullptr in nvcc host pass)
  auto *ipc = CHI_CPU_IPC;
  chi::u32 gpu_id = 0;
  chi::u32 test_value = 42;
  chi::u64 counter_addr = reinterpret_cast<chi::u64>(g_d_counter);

  auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id,
      chi::PoolQuery::ToLocalGpu(gpu_id, parallelism),
      gpu_id, test_value, counter_addr);
  auto future = ipc->Send(task);

  // Wait with timeout
  bool completed = future.Wait(30.0f);
  if (!completed) {
    return -3;  // Timeout
  }

  // Copy counter from device to host
  unsigned int h_counter = 0;
  cudaMemcpy(&h_counter, g_d_counter, sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  *out_counter = h_counter;

  // Verify result_value_ is correct (last warp's output)
  chi::u32 expected = (test_value * 3) + gpu_id;
  if (future->result_value_ != expected) {
    fprintf(stderr, "[PARALLELISM] result_value_=%u expected=%u\n",
            future->result_value_, expected);
    return -4;  // Wrong result value
  }

  return 1;  // Success
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
