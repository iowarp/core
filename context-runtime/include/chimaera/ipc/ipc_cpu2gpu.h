/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {

class IpcManager;
namespace gpu { class IpcManager; }

/**
 * IPC transport for CPU client → GPU runtime.
 */
struct IpcCpu2Gpu {
  /** Allocate device buffer, copy H2D, push to cpu2gpu_queue. */
  template <typename TaskT>
  static chi::Future<TaskT> ClientSend(
      gpu::IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr,
      u32 gpu_id = 0);

  /** RuntimeRecv: handled by gpu::Worker::TryPopFromQueue (not called directly). */

  /** Set FUTURE_COMPLETE on CPU side. */
  static void RuntimeSend(
      IpcManager *ipc, const FullPtr<Task> &task_ptr,
      RunContext *run_ctx, Container *container);

  /** Client-side wait: poll pinned-host gpu::FutureShm, copy result D2H. */
  template <typename TaskT, typename AllocT>
  static bool ClientRecv(Future<TaskT, AllocT> &future, float max_sec);
};

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_CPU2GPU_H_
