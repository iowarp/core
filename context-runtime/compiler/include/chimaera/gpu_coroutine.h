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

#ifndef CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_
#define CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_

/**
 * @file gpu_coroutine.h
 * @brief C++20 coroutine support for CUDA device code, compiled with Clang.
 *
 * Provides chi::gpu::TaskResume, chi::gpu::RunContext, and chi::gpu::yield()
 * for GPU-side coroutine execution.  Types live in the chi::gpu namespace to
 * coexist with the CPU-side chi::TaskResume / chi::RunContext from task.h.
 *
 * === Memory Allocation ===
 *
 * Coroutine frames and scheduler entries are allocated through
 * CHI_PRIV_ALLOC (PrivateBuddyAllocator) directly.  No function
 * pointers or indirection — the macro resolves to the per-warp
 * cached allocator on GPU.
 *
 * === Clang Requirement ===
 *
 * NVCC does not support C++20 coroutines in device code. Files that use
 * GPU coroutines must be compiled with Clang via add_cuda_library() /
 * add_cuda_executable() after calling wrp_core_find_clang_cuda().
 */

// Must be included first -- blocks libstdc++ <coroutine> and provides
// GPU-compatible std::coroutine_handle with __host__ __device__ annotations.
#include <cstddef>
#include <cstdint>

#include "chimaera/gpu_coroutine_handle.h"
#include "chimaera/task.h"

// Set to 1 to enable verbose GPU coroutine debug printf (very slow)
#ifndef CHI_GPU_CORO_DEBUG
#define CHI_GPU_CORO_DEBUG 0
#endif

#if CHI_GPU_CORO_DEBUG
#define GPU_CORO_DPRINTF(...) printf(__VA_ARGS__)
#else
#define GPU_CORO_DPRINTF(...) ((void)0)
#endif

namespace chi {
namespace gpu {

/** Unsigned 32-bit integer (self-contained, no dependency on hshm types) */
using u32 = uint32_t;

// Forward declarations
class TaskResume;
class Container;

static constexpr u32 kWarpSize = 32;

// Wrapper functions for coroutine frame allocation via CHI_PRIV_ALLOC.
// Defined in gpu_coroutine_gpu.cc to avoid pulling ipc_manager.h here.
HSHM_GPU_FUN hipc::FullPtr<char> GpuCoroAlloc(size_t size);
HSHM_GPU_FUN void GpuCoroFree(hipc::FullPtr<char> fp);
HSHM_GPU_FUN void GpuCoroFreeRaw(void *ptr);

// ============================================================================
// FrameHeader -- prepended to each frame block for bulk vs single tracking
// ============================================================================

/**
 * Prepended to each coroutine frame (block).  Stores parallelism so that
 * operator delete knows whether to do a bulk free (lane 0 only after
 * __syncwarp) or a single free.
 */
struct alignas(8) FrameHeader {
  u32 parallelism_;
  u32 lane_id_; /**< Hardware lane at frame allocation (set in device pass) */
};

// ============================================================================
// RunContext -- execution context for coroutines (GPU-side)
// ============================================================================

/**
 * GPU-side execution context for coroutine-based task methods.
 *
 * Serves as both the coroutine RunContext (captured by promise_type for
 * yield/await) and the task dispatch context (used by Worker to manage
 * the task lifecycle).  One RunContext is allocated per active task.
 *
 * Layout is GPU-friendly: no STL, no virtual functions, trivially copyable.
 * Heavy types (Container*, FullPtr<Task>, FutureShm*) are stored as void*
 * to avoid pulling in runtime headers from this low-level coroutine header.
 */
struct RunContext {
  // ==== Per-lane coroutine handles ====

  /** Per-lane innermost active coroutine handle (for chain-resume) */
  std::coroutine_handle<> coro_handles_[kWarpSize];

  /** Per-lane top-level coroutine handle (Worker manages lifetime) */
  std::coroutine_handle<> task_coros_[kWarpSize];

  // ==== Coroutine state (shared, set by lane 0) ====

  /** Set by YieldAwaiter::await_suspend when the coroutine yields */
  bool is_yielded_;

  /** Number of PollOnce iterations to skip before resuming (0 = immediate) */
  u32 yield_spin_count_;

  /** Countdown decremented by the scheduler each iteration */
  u32 spins_remaining_;

  // ==== Thread identity ====

  u32 block_id_;
  u32 thread_id_;
  u32 warp_id_; /**< Warp index within the grid */
  u32 lane_id_; /**< Thread lane within the warp (0-31) */

  // ==== Awaited sub-task (co_await tracking) ====

  chi::FutureShm *awaited_fshm_;
  chi::Task *awaited_task_;

  // ==== Task dispatch (set by Worker) ====

  Container *container_;
  chi::FutureShm *task_fshm_;

  /** Full pointer to the task being executed */
  hipc::FullPtr<chi::Task> task_ptr_;

  /** Method ID to dispatch */
  u32 method_id_;

  /** 1 = lane-0-only, 32 = full warp participation */
  u32 parallelism_;

  bool is_gpu2gpu_;
  bool is_copy_path_;

  // ==== Cross-warp range ====

  u32 range_off_; /**< Start of this warp's sub-range within [0, parallelism) */
  u32 range_width_; /**< Width of this warp's sub-range (typically <=32) */

  // ==== Constructors ====

  __host__ __device__ RunContext()
      : is_yielded_(false),
        yield_spin_count_(0),
        spins_remaining_(0),
        block_id_(0),
        thread_id_(0),
        warp_id_(0),
        lane_id_(0),
        awaited_fshm_(nullptr),
        awaited_task_(nullptr),
        container_(nullptr),
        task_fshm_(nullptr),
        method_id_(0),
        parallelism_(1),
        is_gpu2gpu_(false),
        is_copy_path_(false),
        range_off_(0),
        range_width_(0) {
    for (u32 i = 0; i < kWarpSize; ++i) {
      coro_handles_[i] = nullptr;
      task_coros_[i] = nullptr;
    }
  }

  __host__ __device__ RunContext(u32 block_id, u32 thread_id, u32 warp_id,
                                 u32 lane_id)
      : is_yielded_(false),
        yield_spin_count_(0),
        spins_remaining_(0),
        block_id_(block_id),
        thread_id_(thread_id),
        warp_id_(warp_id),
        lane_id_(lane_id),
        awaited_fshm_(nullptr),
        awaited_task_(nullptr),
        container_(nullptr),
        task_fshm_(nullptr),
        method_id_(0),
        parallelism_(1),
        is_gpu2gpu_(false),
        is_copy_path_(false),
        range_off_(0),
        range_width_(0) {
    for (u32 i = 0; i < kWarpSize; ++i) {
      coro_handles_[i] = nullptr;
      task_coros_[i] = nullptr;
    }
  }

  /** Get the warp offset index for this context's range */
  __host__ __device__ u32 GetTaskWarpOffset() const {
    return range_off_ / kWarpSize;
  }

  /** Allocate memory via CHI_PRIV_ALLOC (PrivateBuddyAllocator) */
  __device__ void *Alloc(size_t size) {
    auto fp = GpuCoroAlloc(size);
    return fp.IsNull() ? nullptr : fp.ptr_;
  }

  /** Free memory via CHI_PRIV_ALLOC (PrivateBuddyAllocator) */
  __device__ void Free(void *ptr) { GpuCoroFreeRaw(ptr); }

  /**
   * Free the top-level coroutine frame block directly (lane-0-only path).
   * Used when only lane 0 is active (e.g., suspended task cleanup) and
   * we can't do __syncwarp for the full warp destroy path.
   */
  __device__ void FreeFramesDirect() {
    if (task_coros_[0]) {
      char *frame = static_cast<char *>(task_coros_[0].address());
      GpuCoroFreeRaw(frame - sizeof(FrameHeader));
      for (u32 i = 0; i < kWarpSize; ++i) {
        task_coros_[i] = nullptr;
        coro_handles_[i] = nullptr;
      }
    }
  }
};

// ============================================================================
// TaskResume -- the coroutine return type for GPU task methods
// ============================================================================

/**
 * Coroutine return type for GPU task methods.  A container method returning
 * TaskResume is a C++20 coroutine that can co_await chi::gpu::yield() or
 * co_await another TaskResume (nested coroutines).
 */
class TaskResume {
 public:
  struct promise_type {
    RunContext *run_ctx_ = nullptr;
    std::coroutine_handle<> caller_handle_ = nullptr;
    u32 lane_id_ = 0; /**< Hardware lane set by Worker before resume */

    /** Capture RunContext from the coroutine's first parameter. */
    template <typename... Args>
    __device__ promise_type(RunContext &ctx, Args &&...)
        : run_ctx_(&ctx), caller_handle_(nullptr), lane_id_(0) {}

    /** Capture RunContext from the second parameter (FullPtr<Task>,
     * RunContext&). */
    template <typename TaskT, typename... Args>
    __device__ promise_type(hipc::FullPtr<TaskT>, RunContext &ctx, Args &&...)
        : run_ctx_(&ctx), caller_handle_(nullptr), lane_id_(0) {}

    __device__ promise_type()
        : run_ctx_(nullptr), caller_handle_(nullptr), lane_id_(0) {}

    /**
     * Warp-level bulk allocation helper shared by all operator new overloads.
     * If parallelism > 1, lane 0 allocates 32 frames in one call and
     * broadcasts the base via __shfl_sync. Each lane gets its own frame.
     */
    __device__ static void *AllocFrame(size_t size, u32 parallelism) noexcept {
      size_t total = sizeof(FrameHeader) + size;
      u32 lane = threadIdx.x % kWarpSize;
      if (lane == 0) {
        printf("[AllocFrame] par=%u lane=%u size=%llu\n",
               parallelism, lane, (unsigned long long)size);
      }
      if (parallelism > 1) {
        unsigned long long base_ull = 0;
        if (lane == 0) {
          auto fp = GpuCoroAlloc(kWarpSize * total);
          base_ull =
              fp.IsNull() ? 0 : reinterpret_cast<unsigned long long>(fp.ptr_);
        }
        base_ull = __shfl_sync(0xFFFFFFFF, base_ull, 0);
        if (base_ull == 0) return nullptr;
        char *my_frame = reinterpret_cast<char *>(base_ull) + lane * total;
        auto *hdr = reinterpret_cast<FrameHeader *>(my_frame);
        hdr->parallelism_ = parallelism;
        hdr->lane_id_ = lane;
        return my_frame + sizeof(FrameHeader);
      }
      auto fp = GpuCoroAlloc(total);
      if (fp.IsNull()) return nullptr;
      auto *hdr = reinterpret_cast<FrameHeader *>(fp.ptr_);
      hdr->parallelism_ = 1;
      hdr->lane_id_ = 0;
      return fp.ptr_ + sizeof(FrameHeader);
    }

    /** operator new: catch-all for member function coroutines.
     * C++20 passes coroutine function params to operator new.
     * For member functions, the implicit object ref is first.
     * We accept it as Self&& and scan remaining args for RunContext. */
    template <typename Self, typename... Args>
    __device__ static void *operator new(size_t size, Self&&,
                                         Args &&...args) noexcept {
      return AllocFrame(size, ExtractParallelism(args...));
    }

    /** operator new: fallback for parameterless coroutines. */
    __device__ static void *operator new(size_t size) noexcept {
      return AllocFrame(size, 1);
    }

   private:
    __device__ static u32 ExtractParallelism() { return 1; }

    template <typename... Rest>
    __device__ static u32 ExtractParallelism(RunContext &ctx, Rest &&...) {
      return ctx.parallelism_;
    }

    template <typename First, typename... Rest>
    __device__ static u32 ExtractParallelism(First &&, Rest &&...rest) {
      return ExtractParallelism(rest...);
    }

   public:

    /**
     * Warp-level bulk free: if the frame was bulk-allocated, __syncwarp
     * ensures all lanes finish cleanup, then lane 0 frees the entire block.
     * For single allocations, free directly.
     */
    __device__ static void operator delete(void *ptr, size_t) {
      if (!ptr) return;
      char *raw = static_cast<char *>(ptr) - sizeof(FrameHeader);
      auto *hdr = reinterpret_cast<FrameHeader *>(raw);
#if HSHM_IS_GPU_COMPILER
      if (hdr->parallelism_ > 1) {
        __syncwarp();
        u32 lane = threadIdx.x % kWarpSize;
        if (lane == 0) {
          GpuCoroFreeRaw(raw);
        }
        return;
      }
#endif
      GpuCoroFreeRaw(raw);
    }

    __device__ TaskResume get_return_object() {
      return TaskResume(
          std::coroutine_handle<promise_type>::from_promise(*this));
    }

    __device__ std::suspend_always initial_suspend() noexcept { return {}; }

    struct FinalAwaiter {
      __device__ bool await_ready() noexcept { return false; }

      /**
       * On GPU, symmetric transfer is NOT a tail call — .resume() returns
       * normally through the call stack. If we returned the caller's handle
       * here, the caller's await_resume would destroy our frame while our
       * .resume() is still on the stack (use-after-free).
       *
       * Instead, always return noop_coroutine(). The Worker detects that the
       * inner coroutine is done and chain-resumes callers explicitly, with
       * the stack fully unwound between each step.
       */
      __device__ std::coroutine_handle<> await_suspend(
          std::coroutine_handle<> h) noexcept {
        GPU_CORO_DPRINTF(
            "[FinalAwaiter] coroutine %p reached final_suspend, returning "
            "noop\n",
            h.address());
        return std::noop_coroutine();
      }

      __device__ void await_resume() noexcept {}
    };

    __device__ FinalAwaiter final_suspend() noexcept { return FinalAwaiter{}; }

    __device__ void return_void() {}

    __device__ void unhandled_exception() { __trap(); }

    __host__ __device__ void set_run_context(RunContext *ctx) {
      run_ctx_ = ctx;
    }
    __host__ __device__ RunContext *get_run_context() const { return run_ctx_; }
    __host__ __device__ void set_lane_id(u32 id) { lane_id_ = id; }
    __host__ __device__ u32 get_lane_id() const { return lane_id_; }
    __host__ __device__ void set_caller(std::coroutine_handle<> caller) {
      caller_handle_ = caller;
    }
  };

  using handle_type = std::coroutine_handle<promise_type>;

 private:
  handle_type handle_;
  std::coroutine_handle<>
      caller_handle_; /**< Stored by await_suspend for await_resume */

 public:
  __device__ explicit TaskResume(handle_type h)
      : handle_(h), caller_handle_(nullptr) {}
  __device__ TaskResume() : handle_(nullptr), caller_handle_(nullptr) {}

  __device__ TaskResume(TaskResume &&other) noexcept
      : handle_(other.handle_), caller_handle_(other.caller_handle_) {
    other.handle_ = nullptr;
    other.caller_handle_ = nullptr;
  }

  __device__ TaskResume &operator=(TaskResume &&other) noexcept {
    if (this != &other) {
      if (handle_) handle_.destroy();
      handle_ = other.handle_;
      caller_handle_ = other.caller_handle_;
      other.handle_ = nullptr;
      other.caller_handle_ = nullptr;
    }
    return *this;
  }

  TaskResume(const TaskResume &) = delete;
  TaskResume &operator=(const TaskResume &) = delete;

  __device__ ~TaskResume() {
    if (handle_) handle_.destroy();
  }

  __device__ handle_type get_handle() const { return handle_; }
  __device__ bool done() const { return handle_ && handle_.done(); }

  __device__ void resume() {
    if (handle_ && !handle_.done()) handle_.resume();
  }

  __device__ void destroy() {
    if (handle_) {
      handle_.destroy();
      handle_ = nullptr;
    }
  }

  __device__ explicit operator bool() const { return handle_ != nullptr; }

  __device__ handle_type release() {
    handle_type h = handle_;
    handle_ = nullptr;
    return h;
  }

  // ==========================================================================
  // Awaiter interface -- co_await TaskResume for nested coroutines
  // ==========================================================================

  __device__ bool await_ready() const noexcept {
    return handle_ && handle_.done();
  }

  /**
   * Suspend the calling coroutine and run the inner to completion or
   * suspension.
   *
   * Matches the CPU pattern (task.h): manually resumes the inner coroutine
   * and returns bool. This avoids symmetric transfer, which on GPU is NOT
   * a tail call and would cause use-after-free when await_resume destroys
   * the inner frame while inner's .resume() is still on the stack.
   */
  template <typename PromiseT>
  __device__ bool await_suspend(
      std::coroutine_handle<PromiseT> caller_handle) noexcept {
    if (!handle_) return false;

    // Store caller handle for await_resume to use
    caller_handle_ = caller_handle;

    // Propagate RunContext and lane_id from caller to inner
    handle_.promise().set_run_context(
        caller_handle.promise().get_run_context());
    handle_.promise().set_lane_id(caller_handle.promise().get_lane_id());

    // NOTE: Do NOT set caller on inner's promise yet. If inner completes
    // synchronously, FinalAwaiter would try to access a stale caller.
    // We set it only after confirming inner suspended.

    // Manually resume the inner coroutine
    handle_.resume();
    GPU_CORO_DPRINTF("[TaskResume::await_suspend] inner resumed, done=%d\n",
                     (int)handle_.done());

    // Check if inner completed synchronously
    if (handle_.done()) {
      handle_.destroy();
      handle_ = nullptr;
      return false;  // Don't suspend caller — inner already done
    }

    // Inner suspended (on co_await Future or yield).
    // NOW safe to set caller — inner will complete asynchronously.
    handle_.promise().set_caller(caller_handle);
    GPU_CORO_DPRINTF(
        "[TaskResume::await_suspend] set caller=%p on inner, returning true\n",
        caller_handle.address());
    return true;  // Suspend caller
  }

  /**
   * Resume after inner coroutine completes.
   *
   * Safe to destroy inner here because FinalAwaiter returns noop_coroutine()
   * on GPU — inner's .resume() call has fully unwound before we get here.
   * The Worker chain-resumes callers explicitly.
   */
  __device__ void await_resume() noexcept {
    GPU_CORO_DPRINTF(
        "[TaskResume::await_resume] handle_=%p caller_handle_=%p\n",
        handle_ ? handle_.address() : nullptr,
        caller_handle_ ? caller_handle_.address() : nullptr);
    if (handle_) {
      auto *ctx = handle_.promise().get_run_context();
      char *frame = static_cast<char *>(handle_.address());
      auto *hdr = reinterpret_cast<FrameHeader *>(frame - sizeof(FrameHeader));
      u32 lane = hdr->lane_id_;
      GPU_CORO_DPRINTF("[TaskResume::await_resume] destroying inner %p\n",
                       handle_.address());
      handle_.destroy();
      handle_ = nullptr;

      // Update per-lane coro_handle to this coroutine (the caller) so that
      // subsequent yields or co_awaits are tracked correctly.
      if (ctx && caller_handle_) {
        ctx->coro_handles_[lane] = caller_handle_;
        GPU_CORO_DPRINTF(
            "[TaskResume::await_resume] updated coro_handles_[%u] to caller "
            "%p\n",
            lane, caller_handle_.address());
      }
    }
  }
};

// ============================================================================
// YieldAwaiter / yield() -- same API as CPU side
// ============================================================================

class YieldAwaiter {
 private:
  u32 spin_count_;

 public:
  __device__ explicit YieldAwaiter(u32 spins = 0) : spin_count_(spins) {}

  __device__ bool await_ready() const noexcept { return false; }

  __device__ void await_suspend(std::coroutine_handle<> handle) noexcept {
    auto typed = std::coroutine_handle<TaskResume::promise_type>::from_address(
        handle.address());
    auto *ctx = typed.promise().get_run_context();
    if (!ctx) return;
    char *frame = static_cast<char *>(handle.address());
    auto *hdr = reinterpret_cast<FrameHeader *>(frame - sizeof(FrameHeader));
    u32 lane = hdr->lane_id_;
    ctx->coro_handles_[lane] = handle;
    if (lane == 0) {
      ctx->is_yielded_ = true;
      ctx->yield_spin_count_ = spin_count_;
      ctx->spins_remaining_ = spin_count_;
    }
  }

  __device__ void await_resume() noexcept {}
};

/**
 * Yield control from a coroutine back to the worker.
 * @param spins Number of poll iterations to skip before resume (default 0)
 */
__device__ inline YieldAwaiter yield(u32 spins = 0) {
  return YieldAwaiter(spins);
}

// ============================================================================
// GetLaneIdAwaiter -- retrieve lane ID from coroutine promise
// ============================================================================

/**
 * Awaitable that retrieves the lane ID from the promise.
 *
 * threadIdx.x is unreliable inside clang-cuda coroutine bodies.
 * The Worker sets the correct lane_id on each coroutine's promise
 * before resume, and propagates it to nested coroutines via
 * TaskResume::await_suspend.
 *
 * Usage: chi::u32 lane = co_await chi::gpu::get_lane_id();
 */
class GetLaneIdAwaiter {
 private:
  u32 lane_id_ = 0;

 public:
  __device__ bool await_ready() noexcept { return false; }

  __device__ bool await_suspend(std::coroutine_handle<> handle) noexcept {
    auto typed = std::coroutine_handle<TaskResume::promise_type>::from_address(
        handle.address());
    lane_id_ = typed.promise().lane_id_;  // Direct field access
    return false;
  }

  __device__ u32 await_resume() noexcept { return lane_id_; }
};

__device__ inline GetLaneIdAwaiter get_lane_id() { return GetLaneIdAwaiter{}; }

// ============================================================================
// CoroutineEntry / CoroutineScheduler
// ============================================================================

struct CoroutineEntry {
  std::coroutine_handle<TaskResume::promise_type> handle_;
  RunContext run_ctx_;
  bool occupied_;

  __host__ __device__ CoroutineEntry() : handle_(nullptr), occupied_(false) {}
};

class CoroutineScheduler {
 public:
  static constexpr u32 kMaxSuspended = 32;

 private:
  CoroutineEntry *entries_;
  u32 capacity_;
  u32 count_;

 public:
  __host__ __device__ CoroutineScheduler()
      : entries_(nullptr), capacity_(0), count_(0) {}

  __host__ __device__ void Init(CoroutineEntry *entries, u32 capacity) {
    entries_ = entries;
    capacity_ = capacity;
    count_ = 0;
    for (u32 i = 0; i < capacity_; ++i) {
      entries_[i].occupied_ = false;
      entries_[i].handle_ = nullptr;
    }
  }

  __device__ void Init(RunContext &ctx, u32 capacity = kMaxSuspended) {
    entries_ = static_cast<CoroutineEntry *>(
        ctx.Alloc(sizeof(CoroutineEntry) * capacity));
    capacity_ = capacity;
    count_ = 0;
    for (u32 i = 0; i < capacity_; ++i) {
      entries_[i].occupied_ = false;
      entries_[i].handle_ = nullptr;
    }
  }

  __device__ void Destroy(RunContext &ctx) {
    if (entries_) {
      ctx.Free(entries_);
      entries_ = nullptr;
    }
  }

  __host__ __device__ bool Insert(
      std::coroutine_handle<TaskResume::promise_type> handle,
      const RunContext &ctx) {
    for (u32 i = 0; i < capacity_; ++i) {
      if (!entries_[i].occupied_) {
        entries_[i].handle_ = handle;
        entries_[i].run_ctx_ = ctx;
        entries_[i].occupied_ = true;
        ++count_;
        return true;
      }
    }
    return false;
  }

  __device__ bool ResumeSuspended() {
    bool did_work = false;

    for (u32 i = 0; i < capacity_; ++i) {
      if (!entries_[i].occupied_) continue;

      auto &entry = entries_[i];

      if (entry.run_ctx_.spins_remaining_ > 0) {
        --entry.run_ctx_.spins_remaining_;
        continue;
      }

      entry.run_ctx_.is_yielded_ = false;
      entry.handle_.promise().set_run_context(&entry.run_ctx_);

      entry.handle_.resume();
      did_work = true;

      if (entry.handle_.done()) {
        entry.handle_.destroy();
        entry.occupied_ = false;
        --count_;
      } else if (entry.run_ctx_.is_yielded_) {
        entry.run_ctx_.spins_remaining_ = entry.run_ctx_.yield_spin_count_;
      }
    }

    return did_work;
  }

  __host__ __device__ u32 Count() const { return count_; }
  __host__ __device__ bool HasSuspended() const { return count_ > 0; }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_
