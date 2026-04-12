/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_INFO_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_INFO_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

namespace chi {

/**
 * GPU data transfer object for IpcManager initialization.
 *
 * Passed by value to GPU kernels. Contains queue pointers and backend
 * references needed by gpu::IpcManager::ClientInitGpu to set up the
 * per-block IPC state.
 *
 * Memory topology:
 *   - backend: primary scratch / GPU->GPU alloc backend
 *   - gpu2gpu_queue: device memory (orchestrator polls, client pushes)
 *   - internal_queue: device memory (orchestrator polls, runtime pushes)
 *   - cpu2gpu_queue: pinned host (orchestrator polls, CPU pushes)
 *   - gpu2cpu_queue: pinned host (CPU worker polls, GPU pushes)
 *   - gpu2cpu_backend: pinned host for GPU->CPU FutureShm allocation
 */
struct IpcManagerGpuInfo {
  /** Primary backend: orchestrator scratch or client GPU->GPU alloc memory */
  hipc::MemoryBackend backend;

  /** GPU->GPU queue in device memory (orchestrator polls, client pushes) */
  GpuTaskQueue *gpu2gpu_queue = nullptr;

  /** Internal subtask queue in device memory (orchestrator polls) */
  GpuTaskQueue *internal_queue = nullptr;

  /** CPU->GPU queue in pinned host (orchestrator polls, CPU pushes) */
  GpuTaskQueue *cpu2gpu_queue = nullptr;

  /** GPU->CPU queue in pinned host (CPU worker polls, GPU pushes) */
  GpuTaskQueue *gpu2cpu_queue = nullptr;

  /**
   * Pinned-host backend for GPU->CPU FutureShm + copy_space allocation.
   * GPU client allocates from this when routing ToLocalCpu.
   */
  hipc::MemoryBackend gpu2cpu_backend;

  /** Depth of task queues */
  u32 gpu_queue_depth = 16;

  /** Number of lanes in the gpu2gpu GpuTaskQueue */
  u32 gpu2gpu_num_lanes = 1;

  /** Number of lanes in the internal subtask queue */
  u32 internal_num_lanes = 1;

  HSHM_CROSS_FUN IpcManagerGpuInfo() = default;
};

/** Backward compatibility alias */
using IpcManagerGpu = IpcManagerGpuInfo;

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_INFO_H_
