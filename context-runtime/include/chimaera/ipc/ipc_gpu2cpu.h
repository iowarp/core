/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {

class IpcManager;
namespace gpu { class IpcManager; }

/**
 * IPC transport for GPU client → CPU runtime.
 */
struct IpcGpu2Cpu {
  /** GPU-side: enqueue task to gpu2cpu_queue for CPU worker pickup. */
  template <typename TaskT>
  static HSHM_GPU_FUN gpu::Future<TaskT> ClientSend(
      gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr);

  /** CPU-side: deserialize GPU-originated task. */
  static hipc::FullPtr<Task> RuntimeRecv(
      IpcManager *ipc, Future<Task> &future, Container *container,
      u32 method_id, hshm::lbm::Transport *recv_transport);

  /** CPU-side: signal gpu::FutureShm COMPLETE and clean up. */
  static void RuntimeSend(
      IpcManager *ipc, const FullPtr<Task> &task_ptr,
      RunContext *run_ctx, Container *container);

  /** GPU-side wait: polls gpu::FutureShm FUTURE_COMPLETE (device-scope).
   *  Same mechanism as IpcGpu2Gpu::ClientRecv — the CPU worker signals
   *  completion on the gpu::FutureShm via system-scope atomics. */
  template <typename TaskT>
  static HSHM_GPU_FUN void ClientRecv(
      gpu::IpcManager *ipc, gpu::Future<TaskT> &future, TaskT *task_ptr);
};

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_H_
