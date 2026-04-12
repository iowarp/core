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

  chi::priv::vector<char> recv_buf(CHI_PRIV_ALLOC);
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

  // Signal the device-side gpu::FutureShm so the GPU waiter sees COMPLETE.
  // Use direct volatile write to the flags field. The gpu::FutureShm lives
  // in pinned host memory (cudaMallocHost), so CPU stores are visible to
  // GPU volatile reads through PCIe cache snooping.
  if (future_shm->task_device_ptr_) {
    auto *gpu_fshm = reinterpret_cast<gpu::FutureShm *>(
        future_shm->task_device_ptr_);
    // Direct volatile RMW on the raw storage to ensure the write is not
    // reordered or held in a CPU store buffer.
    volatile u32 *flags_ptr = reinterpret_cast<volatile u32 *>(
        &gpu_fshm->flags_.bits_.x);
    __sync_fetch_and_or(flags_ptr, gpu::FutureShm::FUTURE_COMPLETE);
  }

  // Also mark chi-side future complete for host-side waiters
  future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);

  // Free the task from the GPU orchestrator's scratch allocator.
  // The task lives in pinned host memory allocated by the GPU-side
  // RoundRobinAllocator. The GPU kernel does NOT free it (ClientRecv
  // skips Destroy), so the CPU handles cleanup here.
  auto *gpu_mgr = ipc->GetGpuIpcManager();
  if (gpu_mgr) {
    auto *alloc = reinterpret_cast<hipc::RoundRobinAllocator *>(
        gpu_mgr->gpu_orchestrator_info_.backend.data_);
    if (alloc && alloc->ContainsPtr(task_ptr.ptr_)) {
      task_ptr->ClearFlags(TASK_DATA_OWNER);
      alloc->Free(task_ptr.template Cast<char>());
    }
  }
}

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
