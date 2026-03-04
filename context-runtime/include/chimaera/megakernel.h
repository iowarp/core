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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MEGAKERNEL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MEGAKERNEL_H_

#include "chimaera/types.h"
#include <string>

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

namespace chi {

// Forward declare IpcManagerGpuInfo (defined in ipc_manager.h)
struct IpcManagerGpuInfo;

/**
 * Control structure for megakernel lifecycle (pinned host memory)
 * Shared between CPU and GPU for signaling exit
 */
struct MegakernelControl {
  volatile int exit_flag;
  volatile int running_flag;
};

/**
 * Host-side megakernel launcher
 * Manages lifecycle of the persistent GPU megakernel
 */
class MegakernelLauncher {
 public:
  MegakernelControl *control_ = nullptr;
  void *d_pool_mgr_ = nullptr;  // gpu::PoolManager* on device (opaque for header)
  void *stream_ = nullptr;      // cudaStream_t for dedicated megakernel stream
  bool is_launched_ = false;

  // Saved launch parameters for Pause/Resume
  u32 blocks_ = 0;
  u32 threads_per_block_ = 0;

  /**
   * Launch the megakernel on the GPU
   * @param gpu_info IPC info with queue pointers
   * @param blocks Number of GPU blocks
   * @param threads_per_block Threads per block
   * @return true if launch successful
   */
  bool Launch(const IpcManagerGpuInfo &gpu_info, u32 blocks,
              u32 threads_per_block);

  /**
   * Stop the megakernel and free resources
   */
  void Finalize();

  /**
   * Pause the megakernel (signal exit + wait for completion).
   * Frees SMs so other kernels (e.g., GPU container allocation) can run.
   * The device-side PoolManager and control structure are preserved.
   */
  void Pause();

  /**
   * Resume a paused megakernel with the same parameters.
   * @param gpu_info IPC info with queue pointers
   */
  void Resume(const IpcManagerGpuInfo &gpu_info);

  /**
   * Register a GPU container with the device-side PoolManager.
   * Also fixes up the container's function pointer dispatch table with
   * addresses from the megakernel's CUDA module context (required because
   * device function pointers from companion .so libraries are not callable
   * from the megakernel's persistent kernel).
   *
   * @param pool_id Pool identifier
   * @param gpu_container_ptr Device pointer to gpu::Container
   * @param chimod_name Name of the ChiMod (e.g., "chimaera_admin")
   */
  void RegisterGpuContainer(const PoolId &pool_id, void *gpu_container_ptr,
                             const std::string &chimod_name);
};

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_MEGAKERNEL_H_
