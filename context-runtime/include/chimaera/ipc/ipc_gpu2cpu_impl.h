/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_IMPL_H_

#include "chimaera/ipc/ipc_gpu2cpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {

#if HSHM_IS_GPU_COMPILER
/**
 * GPU-side ClientSend: enqueue task to gpu2cpu_queue (pinned host).
 * The CPU GPU worker polls this queue and dispatches on the CPU side.
 */
template <typename TaskT>
HSHM_GPU_FUN gpu::Future<TaskT> IpcGpu2Cpu::ClientSend(
    gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr) {
  gpu::Future<TaskT> future;

  if (threadIdx.x == 0) {
    if (!task_ptr.IsNull() && ipc->gpu_info_.gpu2cpu_queue) {
      gpu::FutureShm *fshm = reinterpret_cast<gpu::FutureShm *>(
          reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
      fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
      fshm->client_task_vaddr_ =
          reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
      fshm->flags_.SetBits(gpu::FutureShm::FUTURE_DEVICE_SCOPE);

      hipc::ShmPtr<gpu::FutureShm> fshmptr;
      fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
      fshmptr.off_ = reinterpret_cast<size_t>(fshm);
      future = gpu::Future<TaskT>(fshmptr, task_ptr);

      auto &qlane = ipc->gpu_info_.gpu2cpu_queue->GetLane(0, 0);
      gpu::Future<Task> task_future(future.GetFutureShmPtr());
      hipc::threadfence_system();
      qlane.Push(task_future);
    }
  }
  return future;
}
#endif  // HSHM_IS_GPU_COMPILER

#if HSHM_IS_GPU_COMPILER
/**
 * GPU-side ClientRecv: poll gpu::FutureShm FUTURE_COMPLETE.
 * Unlike IpcGpu2Gpu::ClientRecv, this does NOT mark the future as consumed
 * because the CPU RuntimeSend owns task cleanup for gpu2cpu tasks. The GPU
 * kernel must not free the task — the pinned host memory is freed by the
 * CPU after signaling completion.
 */
template <typename TaskT>
HSHM_GPU_FUN void IpcGpu2Cpu::ClientRecv(
    gpu::IpcManager *ipc, gpu::Future<TaskT> &future, TaskT *task_ptr) {
  (void)ipc; (void)task_ptr;
  if (threadIdx.x != 0) return;

  hipc::FullPtr<gpu::FutureShm> fshm_full = future.GetFutureShm();
  if (fshm_full.IsNull()) return;
  gpu::FutureShm *fshm = fshm_full.ptr_;
  // Poll FUTURE_COMPLETE via volatile read (safe for pinned host memory).
  volatile unsigned int *fp =
      reinterpret_cast<volatile unsigned int *>(&fshm->flags_.bits_.x);
  while (!((*fp) & gpu::FutureShm::FUTURE_COMPLETE)) {}
  __threadfence_system();
  // Do NOT call future.Destroy(true): that sets consumed_=true which
  // causes ~Future to call DelTask. The CPU RuntimeSend handles cleanup.
}
#endif  // HSHM_IS_GPU_COMPILER

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_IMPL_H_
