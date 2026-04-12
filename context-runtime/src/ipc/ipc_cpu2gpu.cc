/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "chimaera/ipc/ipc_cpu2gpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/ipc_manager.h"

namespace chi {

void IpcCpu2Gpu::RuntimeSend(
    IpcManager *ipc, const FullPtr<Task> &task_ptr,
    RunContext *run_ctx, Container *container) {
  auto future_shm = run_ctx->future_.GetFutureShm();

  // CPU->GPU: set complete (D2H copy handled in WaitCpu2Gpu)
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
  task_ptr->ClearFlags(TASK_DATA_OWNER);
  container->DelTask(task_ptr->method_, task_ptr);
}

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
