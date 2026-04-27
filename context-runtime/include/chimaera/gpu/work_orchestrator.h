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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_WORK_ORCHESTRATOR_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_WORK_ORCHESTRATOR_H_

#include "chimaera/types.h"
#include "chimaera/task.h"
#include "chimaera/gpu/gpu_info.h"
#include <string>

#if HSHM_ENABLE_GPU

namespace chi {

namespace gpu {

/**
 * Control structure for GPU work orchestrator lifecycle (pinned host memory)
 * Shared between CPU and GPU for signaling exit
 */
struct WorkOrchestratorControl {
  volatile int exit_flag;
  volatile int running_flag;

  /** Debug state written by GPU workers, readable from CPU (pinned memory) */
  static constexpr int kMaxDebugWorkers = 32;
  volatile unsigned long long dbg_poll_count[kMaxDebugWorkers];
  volatile unsigned int dbg_num_suspended[kMaxDebugWorkers];
  volatile unsigned int dbg_last_method[kMaxDebugWorkers];
  /** Last state: 0=idle, 1=popped_task, 2=dispatched, 3=suspended, 4=resumed, 5=completed */
  volatile unsigned int dbg_last_state[kMaxDebugWorkers];
  volatile unsigned long long dbg_resume_checks[kMaxDebugWorkers];
  /** Debug: total_written_ after SerializeAndComplete for last task */
  volatile unsigned long long dbg_ser_total_written[kMaxDebugWorkers];
  /** Debug: method_id of last serialized task */
  volatile unsigned int dbg_ser_method[kMaxDebugWorkers];
  /** Debug: step within DispatchTask (10=enter, 11=lbm_ctx, 12=pre_recv, ...) */
  volatile unsigned int dbg_dispatch_step[kMaxDebugWorkers];
  /** Debug: input ring buffer values */
  volatile unsigned long long dbg_input_tw[kMaxDebugWorkers];
  volatile unsigned long long dbg_input_cs[kMaxDebugWorkers];
  /** Debug: task lifecycle counters */
  volatile unsigned int dbg_tasks_popped[kMaxDebugWorkers];
  volatile unsigned int dbg_tasks_completed[kMaxDebugWorkers];
  volatile unsigned int dbg_tasks_resumed[kMaxDebugWorkers];
  volatile unsigned int dbg_alloc_failures[kMaxDebugWorkers];
  /** Debug: queue pop counters (before container lookup) */
  volatile unsigned int dbg_queue_pops[kMaxDebugWorkers];
  /** Debug: no-container counter */
  volatile unsigned int dbg_no_container[kMaxDebugWorkers];
  /** Debug: internal queue head/tail (separate from gpu2gpu) */
  volatile unsigned long long dbg_iq_head[kMaxDebugWorkers];
  volatile unsigned long long dbg_iq_tail[kMaxDebugWorkers];
  /** Debug: internal queue pop counters */
  volatile unsigned int dbg_iq_pops[kMaxDebugWorkers];
  /** Debug: internal queue push counters (from SendGpu) */
  volatile unsigned int dbg_iq_pushes[kMaxDebugWorkers];

  /** Base of cpu2gpu copy-space backend for GPU-side ShmPtr->ptr conversion.
   *  Set by host before orchestrator launch; used by gpu::Worker. */
  char *cpu2gpu_queue_base = nullptr;

  /** Scratch allocator generation — incremented every time scratch is
   *  reinitialized (pause/resume).  GPU containers compare this against
   *  a saved value to detect when their scratch-backed metadata became
   *  invalid and needs to be recreated. */
  volatile unsigned int scratch_gen;

  /** Orchestrator profiling counters (written by GPU, read by CPU) */
  volatile long long prof_queue_pop[kMaxDebugWorkers];
  volatile long long prof_recv_device[kMaxDebugWorkers];
  volatile long long prof_alloc_task[kMaxDebugWorkers];
  volatile long long prof_load_task[kMaxDebugWorkers];
  volatile long long prof_alloc_ctx[kMaxDebugWorkers];
  volatile long long prof_coro_create[kMaxDebugWorkers];
  volatile long long prof_coro_resume[kMaxDebugWorkers];
  volatile long long prof_coro_destroy[kMaxDebugWorkers];
  volatile long long prof_save_task[kMaxDebugWorkers];
  volatile long long prof_send_device[kMaxDebugWorkers];
  volatile long long prof_complete[kMaxDebugWorkers];
  volatile long long prof_task_count[kMaxDebugWorkers];
  volatile long long prof_ctx_alloc[kMaxDebugWorkers];
  volatile long long prof_ctx_copy[kMaxDebugWorkers];
  volatile long long prof_ctx_zero[kMaxDebugWorkers];

  /** AllocTask sub-breakdown */
  volatile long long prof_alloc_task_buddy[kMaxDebugWorkers];   /**< BuddyAlloc */
  volatile long long prof_alloc_task_ctor[kMaxDebugWorkers];    /**< placement new RunContext */
  volatile long long prof_alloc_task_deser[kMaxDebugWorkers];   /**< LoadTaskTmpl (deser) */
};

/**
 * Host-side GPU work orchestrator
 * Manages lifecycle of the persistent GPU orchestrator kernel
 */
class WorkOrchestrator {
 public:
  WorkOrchestratorControl *control_ = nullptr;
  void *d_pool_mgr_ = nullptr;  // gpu::PoolManager* on device (opaque for header)
  void *stream_ = nullptr;      // cudaStream_t / sycl::queue* for dedicated orchestrator stream
  // Host USM staging for IpcManagerGpuInfo (SYCL only — captured by pointer
  // in the persistent kernel since the struct isn't device-copyable).
  // Always nullptr on the CUDA/ROCm path.
  IpcManagerGpuInfo *gpu_info_storage_ = nullptr;
  // Host USM IpcManager (SYCL only). The persistent kernel captures this
  // pointer so CHI_IPC under SYCL resolves to it. Phase 10 plumbing —
  // CUDA/ROCm reach the IpcManager through __shared__ + GetBlockIpcManager.
  IpcManager *ipc_storage_ = nullptr;
  bool is_launched_ = false;

  // Saved launch parameters for Pause/Resume
  u32 blocks_ = 0;
  u32 threads_per_block_ = 0;

  // Cross-warp scheduling resources (freed in Finalize)
  void *warp_group_load_ = nullptr;
  void *warp_load_ = nullptr;
  void *warp_group_queue_data_ = nullptr;
  void *warp_group_queue_ptr_ = nullptr;  // GpuTaskQueue* on device
  u32 launched_num_warps_ = 0;            // Warp count at last Launch/Resume

  /**
   * Launch the GPU work orchestrator
   * @param gpu_info IPC info with queue pointers
   * @param blocks Number of GPU blocks
   * @param threads_per_block Threads per block
   * @return true if launch successful
   */
  bool Launch(const IpcManagerGpuInfo &gpu_info, u32 blocks,
              u32 threads_per_block,
              char *cpu2gpu_queue_base = nullptr);

  /**
   * Stop the orchestrator and free resources
   */
  void Finalize();

  /**
   * Pause the orchestrator (signal exit + wait for completion).
   * Frees SMs so other kernels (e.g., GPU container allocation) can run.
   * The device-side PoolManager and control structure are preserved.
   */
  void Pause();

  /**
   * Pre-allocate cross-warp resources after Pause but before Resume.
   * Must be called while NO persistent GPU kernels are running, as it
   * uses cudaMalloc/cudaFree which synchronize with the default stream.
   */
  void PrepareResume();

  /**
   * Resume a paused orchestrator with the same parameters.
   * Safe to call while other GPU kernels are running (no implicit sync).
   * @param gpu_info IPC info with queue pointers
   */
  void Resume(const IpcManagerGpuInfo &gpu_info);

  /**
   * Register a GPU container with the device-side PoolManager.
   * Since all containers are allocated within this CUDA module,
   * vtables are already correct -- no fixup needed.
   *
   * @param pool_id Pool identifier
   * @param gpu_container_ptr Device pointer to gpu::Container
   */
  void RegisterGpuContainer(const PoolId &pool_id, void *gpu_container_ptr);

  /**
   * Allocate a GPU container by module name.
   * Launches a kernel in the orchestrator's CUDA module context so vtables
   * are valid for the persistent orchestrator kernel.
   *
   * @param pool_id Pool identifier
   * @param container_id Container ID (typically node_id)
   * @param chimod_name Name of the ChiMod (e.g., "chimaera_admin")
   * @return Device pointer to allocated gpu::Container, or nullptr
   */
  void *AllocGpuContainer(const PoolId &pool_id, u32 container_id,
                            const std::string &chimod_name);

};

/**
 * Initialize an ArenaAllocator + GpuTaskQueue on device memory via a GPU kernel.
 * Called from CPU; launches a one-shot kernel to construct the allocator and
 * queue in-place on the device buffer, then returns the queue FullPtr.
 *
 * @param device_data  Pointer to device memory (from GpuMalloc)
 * @param capacity     Size of device_data in bytes
 * @param queue_depth  Depth of the GpuTaskQueue ring buffer
 * @return FullPtr<GpuTaskQueue> with shm_ offset valid for this allocator
 */
hipc::FullPtr<GpuTaskQueue> InitQueueOnDevice(char *device_data, size_t capacity,
                                            u32 num_lanes, u32 queue_depth);

}  // namespace gpu
}  // namespace chi

#endif  // HSHM_ENABLE_GPU
#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORK_ORCHESTRATOR_H_
