/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "chimaera/ipc/ipc_gpu2cpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/ipc_manager.h"
#include "chimaera/gpu/future.h"

namespace chi {

hipc::FullPtr<Task> IpcGpu2Cpu::RuntimeRecv(
    IpcManager *ipc, Future<Task> &future, Container *container,
    u32 method_id, hshm::lbm::Transport *recv_transport) {
  auto future_shm = future.GetFutureShm();
  FullPtr<Task> task_full_ptr = future.GetTaskPtr();

  if (!future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) ||
      future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED)) {
    return task_full_ptr;
  }

  // GPU->CPU: task was serialized with DefaultSaveArchive
  hshm::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->input_;

  chi::priv::vector<char> recv_buf;
  recv_buf.reserve(256);
  DefaultLoadArchive local_archive(recv_buf);
  recv_transport->Recv(local_archive, ctx);
  task_full_ptr = container->LocalAllocLoadTask(method_id, local_archive);

  future.GetTaskPtr() = task_full_ptr;
  future_shm->flags_.SetBits(FutureShm::FUTURE_WAS_COPIED);
  return task_full_ptr;
}

void IpcGpu2Cpu::RuntimeSend(
    IpcManager *ipc, const FullPtr<Task> &task_ptr,
    RunContext *run_ctx, Container *container) {
  auto future_shm = run_ctx->future_.GetFutureShm();
  HLOG(kInfo, "IpcGpu2Cpu::RuntimeSend: pool={} method={} device_ptr=0x{:x}",
       task_ptr->pool_id_, task_ptr->method_,
       (size_t)future_shm->task_device_ptr_);

  // Signal the device-side gpu::FutureShm so the GPU waiter sees COMPLETE
  if (future_shm->task_device_ptr_) {
    auto *gpu_fshm = reinterpret_cast<gpu::FutureShm *>(
        future_shm->task_device_ptr_);
    gpu_fshm->flags_.SetBitsSystem(gpu::FutureShm::FUTURE_COMPLETE);
  }

  // Also mark chi-side future complete for host-side waiters
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
  task_ptr->ClearFlags(TASK_DATA_OWNER);
  container->DelTask(task_ptr->method_, task_ptr);
}

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
