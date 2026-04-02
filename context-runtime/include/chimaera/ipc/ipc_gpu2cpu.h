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
 *
 * GPU kernel enqueues to gpu2cpu_queue (pinned host).
 * CPU GPU worker dequeues, deserializes, and dispatches.
 * CPU runtime serializes outputs back and signals gpu::FutureShm COMPLETE.
 * GPU kernel polls FUTURE_COMPLETE via system-scope atomics.
 */
struct IpcGpu2Cpu {
  /** GPU-side: enqueue task to gpu2cpu_queue for CPU worker pickup. */
  template <typename TaskT>
  static HSHM_GPU_FUN gpu::Future<TaskT> ClientSend(
      gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr);

  /** Deserialize GPU-originated task on CPU runtime. */
  static hipc::FullPtr<Task> RuntimeRecv(
      IpcManager *ipc, Future<Task> &future, Container *container,
      u32 method_id, hshm::lbm::Transport *recv_transport);

  /** Signal gpu::FutureShm COMPLETE and clean up. */
  static void RuntimeSend(
      IpcManager *ipc, const FullPtr<Task> &task_ptr,
      RunContext *run_ctx, Container *container);
};

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2CPU_H_
