/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_IMPL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_IMPL_H_

#include "chimaera/ipc/ipc_cpu2gpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "hermes_shm/util/gpu_api.h"

namespace chi {

#if HSHM_IS_HOST

/**
 * CPU→GPU ClientSend using pre-allocated pools.
 *
 * - Task data allocated from device memory pool (no cudaMalloc)
 * - FutureShm allocated from pinned host pool (no cudaMallocHost)
 * - H2D copy via cudaMemcpyAsync on dedicated non-blocking stream
 * - No device-synchronizing calls — safe while orchestrator is running
 */
template <typename TaskT>
chi::Future<TaskT> IpcCpu2Gpu::ClientSend(
    gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr, u32 gpu_id) {
  if (task_ptr.IsNull() || gpu_id >= ipc->gpu_devices_.size()) {
    return chi::Future<TaskT>();
  }
  auto &dev = ipc->gpu_devices_[gpu_id];

  size_t task_size = sizeof(TaskT);
  size_t total_size = task_size + sizeof(gpu::FutureShm);

  // Allocate [Task | gpu::FutureShm] from the pinned-host pool.
  // Pinned host memory is GPU-accessible via UVM — no cudaMemcpy needed.
  // The GPU kernel reads the task directly from pinned host.
  if (!dev.cpu2gpu_fshm_pool ||
      dev.cpu2gpu_fshm_next + total_size > dev.cpu2gpu_fshm_pool_size) {
    dev.cpu2gpu_fshm_next = 0;  // Wrap around
  }
  char *pinned_buf = dev.cpu2gpu_fshm_pool + dev.cpu2gpu_fshm_next;
  dev.cpu2gpu_fshm_next += total_size;

  // Copy task data to pinned host (plain memcpy, no CUDA API)
  memcpy(pinned_buf, reinterpret_cast<const char *>(task_ptr.ptr_),
         task_size);

  // Initialize FutureShm (co-located after task in pinned host).
  // Cast to void* to silence -Wclass-memaccess: the struct has non-trivial
  // members (atomics) but zero-initialization of its bytes is the intended
  // start state; the field assignments below set the live values.
  auto *host_fshm = reinterpret_cast<gpu::FutureShm *>(
      pinned_buf + task_size);
  memset(static_cast<void *>(host_fshm), 0, sizeof(gpu::FutureShm));
  host_fshm->pool_id_ = task_ptr->pool_id_;
  host_fshm->method_id_ = task_ptr->method_;
  host_fshm->origin_ = gpu::FutureShm::FUTURE_CLIENT_CPU2GPU;
  host_fshm->client_task_vaddr_ = reinterpret_cast<uintptr_t>(pinned_buf);
  host_fshm->task_device_ptr_ = reinterpret_cast<uintptr_t>(pinned_buf);
  host_fshm->task_size_ = static_cast<u32>(task_size);
  host_fshm->flags_.SetBits(gpu::FutureShm::FUTURE_POD_COPY);

  // Push to cpu2gpu_queue (pinned host queue, GPU-accessible)
  hipc::ShmPtr<gpu::FutureShm> gpu_fshmptr;
  gpu_fshmptr.alloc_id_ = gpu::FutureShm::GetCpu2GpuAllocId();
  gpu_fshmptr.off_ = reinterpret_cast<size_t>(host_fshm);
  auto &lane = dev.cpu2gpu_queue.ptr_->GetLane(0, 0);
  gpu::Future<Task> task_future(gpu_fshmptr);
  lane.Push(task_future);

  // Return chi::Future (polls pinned-host FutureShm for completion)
  hipc::ShmPtr<chi::FutureShm> chi_fshmptr;
  chi_fshmptr.alloc_id_ = gpu::FutureShm::GetCpu2GpuAllocId();
  chi_fshmptr.off_ = reinterpret_cast<size_t>(host_fshm);
  return chi::Future<TaskT>(chi_fshmptr, task_ptr);
}

template <typename TaskT, typename AllocT>
bool IpcCpu2Gpu::ClientRecv(Future<TaskT, AllocT> &future, float max_sec) {
  // ShmPtr offset points to pinned-host gpu::FutureShm
  void *host_fshm = reinterpret_cast<void *>(future.future_shm_.off_.load());
  auto start = std::chrono::steady_clock::now();

  // Poll flags on pinned-host gpu::FutureShm (direct CPU read, no cudaMemcpy)
  auto *fshm = reinterpret_cast<gpu::FutureShm *>(host_fshm);
  while (!fshm->flags_.AnySystem(gpu::FutureShm::FUTURE_COMPLETE)) {
    HSHM_THREAD_MODEL->Yield();
    if (max_sec > 0) {
      float elapsed = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      if (elapsed >= max_sec) return false;
    }
  }

  // Task lives on pinned host — GPU wrote results there directly.
  // CPU can read pinned host without cudaMemcpy.
  void *pinned_task = reinterpret_cast<void *>(fshm->task_device_ptr_);
  if (future.task_ptr_.ptr_ && pinned_task) {
    // Intentional raw POD copy: TaskT is layout-compatible with bytes on the
    // GPU side, and FixupAfterCopy() below re-seats any inline pointer fields
    // (e.g. priv::vector SVO). Cast to void* to silence -Wclass-memaccess.
    memcpy(static_cast<void *>(future.task_ptr_.ptr_), pinned_task,
           sizeof(TaskT));
    future.task_ptr_->FixupAfterCopy();
  }
  return true;
}

#endif  // HSHM_IS_HOST

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_IMPL_H_
