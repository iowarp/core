/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_H_
#define CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {
namespace gpu {

class IpcManager;

/**
 * IPC transport for GPU client → GPU runtime (inter-warp via gpu2gpu queue).
 * All GPU-side methods are HSHM_GPU_FUN. Runtime methods run on the GPU
 * orchestrator (CDP child kernels).
 */
struct IpcGpu2Gpu {
  template <typename TaskT>
  static HSHM_GPU_FUN Future<TaskT> ClientSend(
      IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr);

  /** RuntimeRecv: handled by gpu::Worker::TryPopFromQueue (no separate call). */

  /** RuntimeSend: GPU orchestrator sets FUTURE_COMPLETE via device-scope atomics. */

  template <typename TaskT>
  static HSHM_GPU_FUN void ClientRecv(
      IpcManager *ipc, Future<TaskT> &future, TaskT *task_ptr);
};

/**
 * IPC transport for GPU runtime self-dispatch (intra-GPU via internal queue).
 */
struct IpcGpu2Self {
  template <typename TaskT>
  static HSHM_GPU_FUN Future<TaskT> ClientSend(
      IpcManager *ipc, const hipc::FullPtr<TaskT> &task_ptr);

  /** RuntimeRecv/RuntimeSend/ClientRecv: same as IpcGpu2Gpu
   *  (orchestrator handles both paths identically). */
};

}  // namespace gpu
}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_IPC_GPU2GPU_H_
