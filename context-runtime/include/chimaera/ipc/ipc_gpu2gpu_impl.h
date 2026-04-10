/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_IMPL_H_

#include "chimaera/ipc/ipc_gpu2gpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {
namespace gpu {

#if HSHM_IS_GPU_COMPILER

/**
 * GPU→GPU ClientSend: thread 0 enqueues task to gpu2gpu_queue.
 * Only thread 0 allocates and pushes. No warp-level primitives.
 */
template <typename TaskT>
HSHM_GPU_FUN Future<TaskT> IpcGpu2Gpu::ClientSend(
    IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr) {
  Future<TaskT> future;
  if (threadIdx.x == 0) {
    if (!task_ptr.IsNull() && ipc->gpu_info_.gpu2gpu_queue) {
      FutureShm *fshm = reinterpret_cast<FutureShm *>(
          reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
      fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
      fshm->client_task_vaddr_ =
          reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
      fshm->flags_.SetBits(FutureShm::FUTURE_DEVICE_SCOPE);
      hipc::ShmPtr<FutureShm> fshmptr;
      fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
      fshmptr.off_ = reinterpret_cast<size_t>(fshm);
      future = Future<TaskT>(fshmptr, task_ptr);
      u32 queue_lane_id = 0;
      if (ipc->gpu_info_.gpu2gpu_num_lanes > 1) {
        queue_lane_id =
            IpcManager::GetWarpId() % ipc->gpu_info_.gpu2gpu_num_lanes;
      }
      auto &qlane = ipc->gpu_info_.gpu2gpu_queue->GetLane(queue_lane_id, 0);
      Future<Task> task_future(future.GetFutureShmPtr());
      hipc::threadfence_system();
      qlane.PushSystem(task_future);
    }
  }
  return future;
}

/**
 * GPU→GPU ClientRecv: thread 0 polls FutureShm for FUTURE_COMPLETE.
 * No warp-level primitives — works in both client and runtime contexts.
 */
template <typename TaskT>
HSHM_GPU_FUN void IpcGpu2Gpu::ClientRecv(
    IpcManager *ipc, Future<TaskT> &future, TaskT *task_ptr) {
  (void)ipc; (void)task_ptr;
  if (threadIdx.x != 0) return;

  hipc::FullPtr<FutureShm> fshm_full = future.GetFutureShm();
  if (fshm_full.IsNull()) return;
  FutureShm *fshm = fshm_full.ptr_;
  // Poll FUTURE_COMPLETE.
  // For gpu2gpu (device memory): use volatile read (device-scope).
  // For gpu2cpu (pinned host memory): use volatile read. System-scope
  // atomics (atomicAdd_system) can hang on pinned host memory on some
  // GPU architectures, so we use volatile reads which bypass GPU L1
  // cache. For pinned host memory, volatile reads go through PCIe and
  // snoop CPU caches.
  volatile unsigned int *fp =
      reinterpret_cast<volatile unsigned int *>(&fshm->flags_.bits_.x);
  while (!((*fp) & FutureShm::FUTURE_COMPLETE)) {}
  __threadfence_system();
  future.Destroy(true);
}

/**
 * GPU→Self ClientSend: thread 0 enqueues task to internal_queue.
 * No warp-level primitives.
 */
template <typename TaskT>
HSHM_GPU_FUN Future<TaskT> IpcGpu2Self::ClientSend(
    IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr) {
  if (task_ptr.IsNull()) return Future<TaskT>();
  GpuTaskQueue *queue = ipc->gpu_info_.internal_queue
                            ? ipc->gpu_info_.internal_queue
                            : ipc->gpu_info_.gpu2gpu_queue;
  if (!queue) return Future<TaskT>();
  FutureShm *fshm = reinterpret_cast<FutureShm *>(
      reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
  fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
  fshm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
  fshm->client_task_vaddr_ =
      reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
  fshm->flags_.SetBits(FutureShm::FUTURE_DEVICE_SCOPE);
  hipc::ShmPtr<FutureShm> fshmptr;
  fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
  fshmptr.off_ = reinterpret_cast<size_t>(fshm);
  Future<TaskT> future(fshmptr, task_ptr);
  u32 lane_id = 0;
  if (queue == ipc->gpu_info_.internal_queue) {
    if (ipc->gpu_info_.internal_num_lanes > 1)
      lane_id =
          IpcManager::GetWarpId() % ipc->gpu_info_.internal_num_lanes;
  } else {
    if (ipc->gpu_info_.gpu2gpu_num_lanes > 1)
      lane_id =
          IpcManager::GetWarpId() % ipc->gpu_info_.gpu2gpu_num_lanes;
  }
  auto &qlane = queue->GetLane(lane_id, 0);
  Future<Task> task_future(future.GetFutureShmPtr());
  hipc::threadfence_system();
  qlane.PushSystem(task_future);
  return future;
}

#endif  // HSHM_IS_GPU_COMPILER

}  // namespace gpu
}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_IMPL_H_
