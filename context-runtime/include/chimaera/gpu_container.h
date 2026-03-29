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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_

// gpu_coroutine.h MUST be included first — it blocks libstdc++ <coroutine>
// and provides GPU-compatible std::coroutine_handle for Clang CUDA.
#include "chimaera/gpu_coroutine.h"
#include "chimaera/types.h"
#include "chimaera/pool_query.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"

namespace chi {
namespace gpu {

/**
 * GPU-side container base class
 *
 * Uses virtual method dispatch for task execution, deserialization, and
 * serialization. All GPU containers are allocated within the work
 * orchestrator's CUDA module context, so vtables are correct.
 *
 * Run() returns chi::gpu::TaskResume (a C++20 coroutine type) to support
 * yielding and cooperative multitasking on the GPU.  Methods that complete
 * synchronously simply co_return at the end.
 */
class Container {
 public:
  PoolId pool_id_;
  u32 container_id_;
  HSHM_DEFAULT_ALLOC_GPU_T *gpu_alloc_ = nullptr;  /**< Set by worker before dispatch */

  HSHM_GPU_FUN Container() : container_id_(0), gpu_alloc_(nullptr) {}
  HSHM_GPU_FUN virtual ~Container() = default;

  /**
   * Initialize the GPU container
   * @param pool_id Pool identifier
   * @param container_id Container ID (typically node_id)
   */
  HSHM_GPU_FUN void Init(const PoolId &pool_id, u32 container_id) {
    pool_id_ = pool_id;
    container_id_ = container_id;
  }

  /**
   * Virtual dispatch for task execution
   * @param method Method ID to execute
   * @param task_ptr Full pointer to the task
   * @param rctx GPU run context (coroutine-capable)
   * @return TaskResume coroutine handle
   */
  HSHM_GPU_FUN virtual TaskResume Run(u32 method, hipc::FullPtr<Task> task_ptr,
                                       RunContext &rctx) {
    (void)method; (void)task_ptr; (void)rctx;
    co_return;
  }

  /**
   * Virtual dispatch for task deserialization
   * @param method Method ID identifying the task type
   * @param archive LocalLoadTaskArchive containing serialized input
   * @return FullPtr to the deserialized task, or null on failure
   */
  HSHM_GPU_FUN virtual hipc::FullPtr<Task> LocalAllocLoadTask(
      u32 method, LocalLoadTaskArchive &archive) {
    (void)method; (void)archive;
    return hipc::FullPtr<Task>::GetNull();
  }

  /**
   * Allocate + construct a task without deserializing (lane 0 only).
   * Used with LocalLoadTask for warp-parallel deserialization.
   */
  HSHM_GPU_FUN virtual hipc::FullPtr<Task> LocalAllocTask(u32 method) {
    (void)method;
    return hipc::FullPtr<Task>::GetNull();
  }

  /**
   * Deserialize input into an existing task (warp-parallel safe).
   * All lanes call this with warp_converged set on the archive.
   */
  HSHM_GPU_FUN virtual void LocalLoadTask(
      u32 method, LocalLoadTaskArchive &archive,
      const hipc::FullPtr<Task> &task) {
    (void)method; (void)archive; (void)task;
  }

  /**
   * Virtual dispatch for task serialization
   * @param method Method ID identifying the task type
   * @param archive LocalSaveTaskArchive to write output into
   * @param task FullPtr to the completed task
   */
  HSHM_GPU_FUN virtual void LocalSaveTask(
      u32 method, LocalSaveTaskArchive &archive,
      const hipc::FullPtr<Task> &task) {
    (void)method; (void)archive; (void)task;
  }

  /**
   * Virtual dispatch for deserializing task output onto an existing task.
   * Called by Worker before resuming a suspended coroutine, so the
   * deserialization runs outside the coroutine frame (avoids GPU allocator
   * issues in resumed coroutines).
   *
   * @param method Method ID identifying the task type
   * @param archive LocalLoadTaskArchive containing serialized output
   * @param task FullPtr to the task to populate output fields on
   */
  HSHM_GPU_FUN virtual void LocalLoadTaskOutput(
      u32 method, LocalLoadTaskArchive &archive,
      const hipc::FullPtr<Task> &task) {
    (void)method; (void)archive; (void)task;
  }

  /**
   * Virtual dispatch for typed task destruction.
   * Called by Worker::SerializeAndComplete to properly destroy deserialized
   * tasks via their derived destructor (Task has a non-virtual destructor).
   * Without this, priv::vector/priv::string members in derived tasks leak.
   *
   * @param method Method ID identifying the task type
   * @param task FullPtr to the task to destroy
   */
  HSHM_GPU_FUN virtual void LocalDestroyTask(
      u32 method, hipc::FullPtr<Task> &task) {
    (void)method;
    if (!task.IsNull()) {
      task.ptr_->~Task();
    }
  }

  HSHM_GPU_FUN virtual TaskStat GetTaskStats(u32 method_id) const {
    (void)method_id;
    return TaskStat();
  }

  /**
   * Get remaining work for load balancing
   * @return Amount of work remaining (0 = idle)
   */
  HSHM_GPU_FUN u64 GetWorkRemaining() const { return 0; }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_
