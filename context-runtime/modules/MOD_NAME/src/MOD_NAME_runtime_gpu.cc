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
 * GPU implementation of MOD_NAME ChiMod methods.
 *
 * GpuSubmit computes a simple polynomial on the GPU to verify end-to-end
 * GPU task submission and execution.
 */

#include "chimaera/MOD_NAME/MOD_NAME_gpu_runtime.h"
#include "chimaera/singletons.h"
#include "chimaera/gpu/container.h"
#include "hermes_shm/util/gpu_intrinsics.h"

namespace chimaera::MOD_NAME {

HSHM_GPU_FUN void GpuRuntime::SubtaskTest(
    hipc::FullPtr<SubtaskTestTask> task,
    chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under SYCL resolves
  // to the kernel-scope IpcManager pointer (set by the worker via
  // rctx.ipc_mgr_). On CUDA/ROCm the rctx field is nullptr and CHI_IPC
  // continues to reach the per-block __shared__ singleton.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  auto *ipc = CHI_IPC;
  chi::u32 num_subtasks = task->num_subtasks_;
  chi::u32 last_result = 0;

  // Backend-portable thread-0 guard: GetGpuThreadId returns threadIdx.x on
  // CUDA/ROCm and 0 under SYCL single_task. HSHM_DEVICE_PRINTF is a no-op
  // under SYCL since kernels can't call variadic ::printf.
  const bool is_thread0 = (chi::gpu::IpcManager::GetGpuThreadId() == 0);
  if (is_thread0) {
    HSHM_DEVICE_PRINTF(
        "[SubtaskTest] START num_subtasks=%u test_value=%u "
        "internal_queue=%p gpu_alloc=%p\n",
        num_subtasks, task->test_value_,
        (void*)ipc->gpu_info_.internal_queue,
        (void*)ipc->gpu_alloc_);
  }
  HSHM_DEVICE_SYNCWARP();

  for (chi::u32 s = 0; s < num_subtasks; ++s) {
    if (is_thread0) HSHM_DEVICE_PRINTF("[SubtaskTest] iter %u: NewTask\n", s);
    auto sub = ipc->NewTask<GpuSubmitTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
        /*gpu_id=*/chi::u32(0), task->test_value_);
    if (is_thread0) HSHM_DEVICE_PRINTF("[SubtaskTest] iter %u: Send, null=%d\n",
                                       s, (int)sub.IsNull());
    auto future = ipc->Send(sub);
    if (is_thread0) HSHM_DEVICE_PRINTF("[SubtaskTest] iter %u: Wait\n", s);
    future.Wait();
    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      last_result = future.get()->result_value_;
      HSHM_DEVICE_PRINTF("[SubtaskTest] iter %u: done result=%u\n",
                         s, last_result);
    }
    HSHM_DEVICE_SYNCWARP();
  }

  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    task->result_value_ = last_result + 1;
    task->return_code_ = 0;
  }
  (void)rctx;
}

}  // namespace chimaera::MOD_NAME
