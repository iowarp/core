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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_FUTURE_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_FUTURE_H_

#include "chimaera/types.h"
#include "hermes_shm/memory/allocator/allocator.h"

namespace chi {
namespace gpu {

// Forward declarations
struct RunContext;

// ============================================================================
// gpu::FutureShm - Lightweight shared memory structure for GPU task futures
// ============================================================================

/**
 * gpu::FutureShm - Compact future metadata for GPU task paths
 *
 * Contains only the fields needed for GPU task routing and completion
 * signaling. Omits all network/serialization fields (response_*, input_,
 * output_, copy_space[]) that are only used by CPU SHM/ZMQ/IPC paths.
 *
 * Used by: Cpu2Gpu, Gpu2Gpu, Gpu2Cpu task paths.
 */
struct FutureShm {
  // Completion flags (shared with chi::FutureShm)
  static constexpr u32 FUTURE_COMPLETE = 1;
  static constexpr u32 FUTURE_DEVICE_SCOPE = 16;
  static constexpr u32 FUTURE_POD_COPY = 32;

  // Origin constants (shared with chi::FutureShm)
  static constexpr u32 FUTURE_CLIENT_SHM = 0;
  static constexpr u32 FUTURE_CLIENT_CPU2GPU = 3;
  static constexpr u32 FUTURE_CLIENT_GPU2CPU = 4;

  /**
   * Get the sentinel AllocatorId used by SendCpuToGpu to mark ShmPtrs
   * whose offset is a raw pinned-host address (not an SHM offset).
   * @return AllocatorId sentinel {UINT32_MAX-1, 0}
   */
  HSHM_CROSS_FUN static hipc::AllocatorId GetCpu2GpuAllocId() {
    hipc::AllocatorId id;
    id.major_ = UINT32_MAX - 1;
    id.minor_ = 0;
    return id;
  }

  /** Pool ID for the task */
  PoolId pool_id_;
  /** Method ID for the task */
  u32 method_id_;
  /** Origin transport mode */
  u32 origin_;
  /** Virtual address of the task (device or host depending on path) */
  uintptr_t client_task_vaddr_;
  /** sizeof(TaskT) for POD copy sizing (CPU2GPU only) */
  u32 task_size_;
  /** Atomic bitfield for completion flags */
  hshm::abitfield32_t flags_;
  /**
   * Opaque pointer to the parent's GPU RunContext.
   * Set by gpu::Future::await_suspend so that the worker completing
   * this sub-task can directly resume the parent coroutine.
   * Null for top-level (client-originated) tasks.
   */
  void *parent_gpu_rctx_;
  /** Device pointer to POD task for cudaMemcpy (POD copy paths) */
  uintptr_t task_device_ptr_;

  /**
   * Default constructor - initializes all fields
   */
  HSHM_CROSS_FUN FutureShm() {
    pool_id_ = PoolId::GetNull();
    method_id_ = 0;
    origin_ = FUTURE_CLIENT_SHM;
    client_task_vaddr_ = 0;
    task_size_ = 0;
    parent_gpu_rctx_ = nullptr;
    task_device_ptr_ = 0;
    flags_.Clear();
  }

  /**
   * Lightweight reset for per-task reuse on GPU.
   * Only resets fields that change between tasks.
   * @param pool_id Pool ID for the new task
   * @param method_id Method ID for the new task
   */
  HSHM_CROSS_FUN void Reset(PoolId pool_id, u32 method_id) {
    pool_id_ = pool_id;
    method_id_ = method_id;
    client_task_vaddr_ = 0;
    parent_gpu_rctx_ = nullptr;
    task_device_ptr_ = 0;
    task_size_ = 0;
    flags_.Clear();
  }
};

// ============================================================================
// gpu::Future - Lightweight future for GPU task paths
// ============================================================================

/**
 * gpu::Future - Template class for GPU asynchronous task operations
 *
 * Handles Cpu2Gpu, Gpu2Gpu, and Gpu2Cpu wait/completion paths.
 * Simpler than chi::Future: no network response routing, no serialization
 * buffer, no cross-warp range tracking.
 *
 * @tparam TaskT The task type (e.g., Task, CreateTask)
 * @tparam AllocT The allocator type (defaults to CHI_QUEUE_ALLOC_T)
 */
template <typename TaskT, typename AllocT = CHI_QUEUE_ALLOC_T>
class Future {
 public:
  using FutureT = FutureShm;

  template <typename OtherTaskT, typename OtherAllocT>
  friend class Future;

 private:
  /** FullPtr to the task */
  hipc::FullPtr<TaskT> task_ptr_;
  /** ShmPtr to the gpu::FutureShm object */
  hipc::ShmPtr<FutureT> future_shm_;
  /** Whether Destroy(true) was called (via Wait/await_resume) */
  bool consumed_;

 public:
  /**
   * Constructor from ShmPtr<FutureShm> and FullPtr<Task>
   * @param future_shm ShmPtr to existing gpu::FutureShm object
   * @param task_ptr FullPtr to the task
   */
  HSHM_CROSS_FUN Future(hipc::ShmPtr<FutureT> future_shm,
                        const hipc::FullPtr<TaskT>& task_ptr)
      : future_shm_(future_shm), consumed_(false) {
    task_ptr_.shm_ = task_ptr.shm_;
    task_ptr_.ptr_ = task_ptr.ptr_;
  }

  /** Default constructor - creates null future */
  HSHM_CROSS_FUN Future() : consumed_(false) {}

  /**
   * Constructor from ShmPtr<FutureShm> only (queue deserialization)
   * @param future_shm_ptr ShmPtr to gpu::FutureShm object
   */
  HSHM_CROSS_FUN explicit Future(const hipc::ShmPtr<FutureT>& future_shm_ptr)
      : future_shm_(future_shm_ptr), consumed_(false) {
    task_ptr_.SetNull();
  }

  /**
   * Destructor - defined out-of-line in ipc_manager.h where CHI_IPC
   * is available.
   */
  HSHM_CROSS_FUN ~Future();

  /** Mark consumed and run PostWait on the task */
  HSHM_CROSS_FUN void Destroy(bool post_wait = false);

  /** Explicitly delete the underlying task */
  HSHM_CROSS_FUN void DelTask();

  /** Copy constructor - does not transfer ownership */
  HSHM_CROSS_FUN Future(const Future& other)
      : future_shm_(other.future_shm_), consumed_(false) {
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
  }

  /** Copy assignment - does not transfer ownership */
  HSHM_CROSS_FUN Future& operator=(const Future& other) {
    if (this != &other) {
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      future_shm_ = other.future_shm_;
      consumed_ = false;
    }
    return *this;
  }

  /** Move constructor - transfers ownership */
  HSHM_CROSS_FUN Future(Future&& other) noexcept
      : future_shm_(std::move(other.future_shm_)),
        consumed_(other.consumed_) {
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
    other.task_ptr_.SetNull();
    other.consumed_ = false;
  }

  /** Move assignment - transfers ownership */
  HSHM_CROSS_FUN Future& operator=(Future&& other) noexcept {
    if (this != &other) {
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      future_shm_ = std::move(other.future_shm_);
      consumed_ = other.consumed_;
      other.task_ptr_.SetNull();
      other.future_shm_.SetNull();
      other.consumed_ = false;
    }
    return *this;
  }

  /** Get raw pointer to the task */
  HSHM_CROSS_FUN TaskT* get() const { return task_ptr_.ptr_; }

  /** Get the FullPtr to the task */
  hipc::FullPtr<TaskT>& GetTaskPtr() { return task_ptr_; }

  /** Get the FullPtr to the task (const) */
  const hipc::FullPtr<TaskT>& GetTaskPtr() const { return task_ptr_; }

  /** Dereference operator */
  HSHM_CROSS_FUN TaskT& operator*() const { return *task_ptr_.ptr_; }

  /** Arrow operator */
  HSHM_CROSS_FUN TaskT* operator->() const { return task_ptr_.ptr_; }

  /** Check if this future is null */
  HSHM_CROSS_FUN bool IsNull() const { return task_ptr_.IsNull(); }

  /** Get the internal ShmPtr to gpu::FutureShm */
  HSHM_CROSS_FUN hipc::ShmPtr<FutureT> GetFutureShmPtr() const {
    return future_shm_;
  }

  /**
   * Get the gpu::FutureShm FullPtr.
   * Defined out-of-line in ipc_manager.h where CHI_IPC is available.
   */
  HSHM_CROSS_FUN hipc::FullPtr<FutureT> GetFutureShm() const;

  // ----------------------------------------------------------------
  // IsComplete variants
  // ----------------------------------------------------------------

  /**
   * Check if the task is complete.
   * Dispatches to the correct variant based on context.
   * @return True if task has completed
   */
  HSHM_CROSS_FUN bool IsComplete() const;

  /**
   * CPU-to-GPU completion check.
   * Polls device-resident flags via D-to-H memcpy.
   * @return True if task has completed
   */
  HSHM_HOST_FUN bool IsCompleteCpu2Gpu() const;

  /**
   * GPU-to-CPU completion check.
   * Reads flags via system-scope atomic.
   * @return True if task has completed
   */
  HSHM_HOST_FUN bool IsCompleteGpu2Cpu() const;

  /**
   * GPU-to-GPU completion check.
   * Reads flags via device-scope atomic.
   * @return True if task has completed
   */
  HSHM_GPU_FUN bool IsCompleteGpu2Gpu() const;

  // ----------------------------------------------------------------
  // Wait variants
  // ----------------------------------------------------------------

  /**
   * Wait for task completion (blocking with optional timeout).
   * Dispatches to the correct path based on origin.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  HSHM_CROSS_FUN bool Wait(float max_sec = 0, bool reuse_task = false);

  /**
   * Inline GPU wait: volatile poll for FUTURE_COMPLETE.
   * Use this when the library-linked Wait doesn't see completion flags
   * (cross-TU CUDA device-linking visibility issue).
   */
  /**
   * Inline GPU wait: volatile poll for FUTURE_COMPLETE.
   * Must be fully inlined at the call site to avoid cross-TU CUDA
   * device-linking visibility issues with completion flags.
   */
#if HSHM_IS_GPU_COMPILER
  __device__ __forceinline__
#endif
  void WaitGpu() {
#if HSHM_IS_GPU
    if (threadIdx.x != 0) return;
    if (future_shm_.IsNull()) return;
    auto fshm_full = GetFutureShm();
    if (fshm_full.IsNull()) return;
    volatile unsigned int *fp =
        reinterpret_cast<volatile unsigned int *>(
            &fshm_full.ptr_->flags_.bits_.x);
    while (!((*fp) & FutureT::FUTURE_COMPLETE)) {}
    hipc::threadfence();
#endif
  }

  /**
   * CPU-to-GPU wait path (POD transfer via cudaMemcpy).
   * Polls device-resident flags via D-to-H copy, then copies
   * the completed task back to host memory.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  HSHM_HOST_FUN bool WaitCpu2Gpu(float max_sec = 0, bool reuse_task = false);

  /**
   * GPU-to-CPU wait path (GPU future polled from client on host).
   * Polls flags with system-scope atomics.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  HSHM_HOST_FUN bool WaitGpu2Cpu(float max_sec = 0, bool reuse_task = false);

  /**
   * GPU-to-GPU wait path (task submitted and completed on GPU).
   * All warp lanes enter RecvGpu for warp-cooperative deserialization.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  HSHM_GPU_FUN bool WaitGpu2Gpu(float max_sec = 0, bool reuse_task = false);

  /** Wait phase 1: spin until FUTURE_COMPLETE (no deserialization) */
  HSHM_CROSS_FUN void WaitPoll(float max_sec = 0, bool reuse_task = false);

  /** Wait phase 2: deserialize output + cleanup (call after WaitPoll) */
  HSHM_CROSS_FUN void WaitRecv(float max_sec = 0, bool reuse_task = false);

  // ----------------------------------------------------------------
  // Conversion to chi::Future (for host client return types)
  // ----------------------------------------------------------------

  /**
   * Implicit conversion to chi::Future<TaskT>.
   * Enables gpu::IpcManager::Send() (which returns gpu::Future) to be
   * assigned to chi::Future return types in client methods.
   * On host this produces an empty chi::Future (host-side Send is a stub).
   */
  HSHM_CROSS_FUN operator chi::Future<TaskT, AllocT>() const {
    return chi::Future<TaskT, AllocT>();
  }

  // ----------------------------------------------------------------
  // Cast
  // ----------------------------------------------------------------

  /**
   * Cast this gpu::Future to a different task type.
   * Does not transfer ownership.
   * @tparam NewTaskT The new task type
   * @return gpu::Future<NewTaskT> with same underlying state
   */
  template <typename NewTaskT>
  Future<NewTaskT, AllocT> Cast() const {
    Future<NewTaskT, AllocT> result;
    result.task_ptr_ = task_ptr_.template Cast<NewTaskT>();
    result.future_shm_ = future_shm_;
    result.consumed_ = false;
    return result;
  }

  // ----------------------------------------------------------------
  // C++20 Coroutine Awaitable Interface (GPU-only)
  // ----------------------------------------------------------------

  /**
   * Check if the awaitable is ready (coroutine await_ready)
   * @return True if task is complete, false if coroutine should suspend
   */
  HSHM_CROSS_FUN bool await_ready() const noexcept {
    if (future_shm_.IsNull() && task_ptr_.IsNull()) return true;
    if (IsComplete()) return true;
    return false;
  }

  /**
   * Suspend the coroutine and register for resumption
   * @param handle The coroutine handle to resume when task completes
   * @return True to suspend, false to continue
   */
  template <typename PromiseT>
  HSHM_GPU_FUN bool await_suspend(
      std::coroutine_handle<PromiseT> handle) noexcept {
#if HSHM_IS_HOST
    // GPU Future coroutines are not used on host
    (void)handle;
    return false;
#else
    auto *ctx = handle.promise().get_run_context();
    if (ctx) {
#if HSHM_IS_GPU_COMPILER
      u32 lane = threadIdx.x % 32;
#else
      u32 lane = 0;
#endif
      ctx->coro_handles_[lane] = handle;
      ctx->is_yielded_ = true;
      auto fshm_full = GetFutureShm();
      ctx->awaited_fshm_ = fshm_full.IsNull() ? nullptr : fshm_full.ptr_;
      ctx->awaited_task_ = task_ptr_.IsNull() ? nullptr : task_ptr_.ptr_;
      if (!fshm_full.IsNull()) {
        fshm_full.ptr_->parent_gpu_rctx_ = ctx;
      }
      return true;
    }
    return false;
#endif
  }

  /**
   * Get the result after resumption (coroutine await_resume)
   */
  HSHM_GPU_FUN void await_resume() noexcept {
    Destroy(true);
  }
};

}  // namespace gpu

// ============================================================================
// GPU Task Queue typedefs
// ============================================================================

/** Queue type for GPU task paths (stores gpu::Future<Task>) */
using GpuTaskQueue =
    hipc::multi_mpsc_ring_buffer<gpu::Future<Task>, CHI_QUEUE_ALLOC_T>;
/** Single lane within a GpuTaskQueue */
using GpuTaskLane = GpuTaskQueue::ring_buffer_type;

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_FUTURE_H_
