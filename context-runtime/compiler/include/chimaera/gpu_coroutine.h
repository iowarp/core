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
 * Provides chi::TaskResume, chi::RunContext, and chi::yield() -- the same
 * API surface as the CPU-side coroutine primitives in task.h.  Module code
 * written against this API is source-portable between CPU and GPU: the same
 * function returning chi::TaskResume and using co_await chi::yield() compiles
 * on both targets.
 *
 * === Memory Allocation ===
 *
 * Coroutine frames and scheduler entries are allocated through the
 * RunContext::alloc_fn_ / free_fn_ function pointers.  By default these
 * use device malloc/free.  The runtime sets them to CHI_IPC wrappers
 * (AllocateBuffer / FreeBuffer) for proper memory management.
 *
 * === Clang Requirement ===
 *
 * NVCC does not support C++20 coroutines in device code. Files that use
 * GPU coroutines must be compiled with Clang via add_cuda_library() /
 * add_cuda_executable() after calling wrp_core_find_clang_cuda().
 */

// Must be included first -- blocks libstdc++ <coroutine> and provides
// GPU-compatible std::coroutine_handle with __device__ annotations.
#include "chimaera/gpu_coroutine_handle.h"

#include <cstdint>
#include <cstddef>

namespace chi {

/** Unsigned 32-bit integer (self-contained, no dependency on hshm types) */
using u32 = uint32_t;

// Forward declarations
class TaskResume;

// ============================================================================
// Default device malloc/free allocator
// ============================================================================

__device__ inline void *DefaultGpuAlloc(size_t size, void *) {
  return malloc(size);
}

__device__ inline void DefaultGpuFree(void *ptr, void *) {
  free(ptr);
}

// Allocator function pointer types
using GpuAllocFn = void *(*)(size_t size, void *alloc_ctx);
using GpuFreeFn = void (*)(void *ptr, void *alloc_ctx);

// ============================================================================
// RunContext -- execution context for coroutines (GPU-side)
// ============================================================================

/**
 * GPU-side execution context, matching the chi::RunContext interface from
 * task.h.  Coroutine methods receive RunContext& as a parameter; the
 * promise_type captures it so that yield() can access it without TLS.
 *
 * Layout is GPU-friendly: no STL, no virtual functions, trivially copyable.
 */
struct RunContext {
  /** Coroutine handle for the current coroutine */
  std::coroutine_handle<> coro_handle_;

  /** Set by YieldAwaiter::await_suspend when the coroutine yields */
  bool is_yielded_;

  /** Number of PollOnce iterations to skip before resuming (0 = immediate) */
  u32 yield_spin_count_;

  /** Countdown decremented by the scheduler each iteration */
  u32 spins_remaining_;

  /** Block and thread identity */
  u32 block_id_;
  u32 thread_id_;

  /**
   * Memory allocator interface.
   * Default: device malloc/free.
   * Runtime sets these to CHI_IPC->AllocateBuffer / FreeBuffer wrappers.
   * The alloc_ctx_ is passed as the second argument (e.g., IpcManager*).
   */
  GpuAllocFn alloc_fn_;
  GpuFreeFn free_fn_;
  void *alloc_ctx_;

  __host__ __device__ RunContext()
      : coro_handle_(nullptr),
        is_yielded_(false),
        yield_spin_count_(0),
        spins_remaining_(0),
        block_id_(0),
        thread_id_(0),
        alloc_fn_(nullptr),
        free_fn_(nullptr),
        alloc_ctx_(nullptr) {}

  __host__ __device__ RunContext(u32 block_id, u32 thread_id)
      : coro_handle_(nullptr),
        is_yielded_(false),
        yield_spin_count_(0),
        spins_remaining_(0),
        block_id_(block_id),
        thread_id_(thread_id),
        alloc_fn_(nullptr),
        free_fn_(nullptr),
        alloc_ctx_(nullptr) {}

  /** Allocate memory using the configured allocator */
  __device__ void *Alloc(size_t size) {
    if (alloc_fn_) return alloc_fn_(size, alloc_ctx_);
    return malloc(size);
  }

  /** Free memory using the configured allocator */
  __device__ void Free(void *ptr) {
    if (free_fn_) { free_fn_(ptr, alloc_ctx_); return; }
    free(ptr);
  }
};

// ============================================================================
// FrameHeader -- stored before each coroutine frame for deallocation
// ============================================================================

/**
 * Prepended to every coroutine frame allocation.  Stores the free function
 * and context so that promise_type::operator delete can deallocate without
 * access to the RunContext (which doesn't exist at delete time).
 *
 * When the runtime uses CHI_IPC, it stores the FullPtr data in
 * opaque_[0..23] so FreeBuffer can be called without recomputing offsets.
 */
struct FrameHeader {
  GpuFreeFn free_fn_;
  void *alloc_ctx_;
  /** Opaque storage for runtime allocator metadata (e.g., hipc::FullPtr).
   *  The runtime's alloc/free callbacks interpret this data. */
  alignas(8) char opaque_[24];
};

// ============================================================================
// TaskResume -- the coroutine return type (same name as CPU side)
// ============================================================================

/**
 * Coroutine return type for GPU task methods, API-compatible with the
 * CPU-side chi::TaskResume.  A container method returning TaskResume is
 * a C++20 coroutine that can co_await chi::yield() or co_await another
 * TaskResume (nested coroutines).
 */
class TaskResume {
 public:
  struct promise_type {
    RunContext *run_ctx_ = nullptr;
    std::coroutine_handle<> caller_handle_ = nullptr;

    /** Capture RunContext from the coroutine's first parameter. */
    template <typename... Args>
    __device__ promise_type(RunContext &ctx, Args &&...)
        : run_ctx_(&ctx), caller_handle_(nullptr) {}

    __device__ promise_type()
        : run_ctx_(nullptr), caller_handle_(nullptr) {}

    /**
     * Allocate coroutine frame via RunContext's allocator.
     * C++20 allows operator new to take the coroutine function parameters.
     * A FrameHeader is prepended to store the free function for operator
     * delete, which doesn't have access to the RunContext.
     */
    template <typename... Args>
    __device__ static void *operator new(size_t size, RunContext &ctx,
                                         Args &&...) noexcept {
      size_t total = sizeof(FrameHeader) + size;
      char *raw = static_cast<char *>(ctx.Alloc(total));
      if (!raw) return nullptr;
      auto *header = reinterpret_cast<FrameHeader *>(raw);
      header->free_fn_ = ctx.free_fn_;
      header->alloc_ctx_ = ctx.alloc_ctx_;
      return raw + sizeof(FrameHeader);
    }

    /** Fallback for coroutines without RunContext (shouldn't happen). */
    __device__ static void *operator new(size_t size) noexcept {
      size_t total = sizeof(FrameHeader) + size;
      char *raw = static_cast<char *>(malloc(total));
      if (!raw) return nullptr;
      auto *header = reinterpret_cast<FrameHeader *>(raw);
      header->free_fn_ = nullptr;
      header->alloc_ctx_ = nullptr;
      return raw + sizeof(FrameHeader);
    }

    /**
     * Free coroutine frame.  Recovers the FrameHeader stored before
     * the frame and calls the allocator's free function.
     */
    __device__ static void operator delete(void *ptr, size_t) {
      char *raw = static_cast<char *>(ptr) - sizeof(FrameHeader);
      auto *header = reinterpret_cast<FrameHeader *>(raw);
      if (header->free_fn_) {
        header->free_fn_(raw, header->alloc_ctx_);
      } else {
        free(raw);
      }
    }

    __device__ TaskResume get_return_object() {
      return TaskResume(
          std::coroutine_handle<promise_type>::from_promise(*this));
    }

    __device__ std::suspend_always initial_suspend() noexcept { return {}; }

    struct FinalAwaiter {
      std::coroutine_handle<> caller_;

      __device__ bool await_ready() noexcept { return false; }

      __device__ std::coroutine_handle<>
      await_suspend(std::coroutine_handle<>) noexcept {
        return caller_ ? caller_ : std::noop_coroutine();
      }

      __device__ void await_resume() noexcept {}
    };

    __device__ FinalAwaiter final_suspend() noexcept {
      return FinalAwaiter{caller_handle_};
    }

    __device__ void return_void() {}

    __device__ void unhandled_exception() { __trap(); }

    __device__ void set_run_context(RunContext *ctx) { run_ctx_ = ctx; }
    __device__ RunContext *get_run_context() const { return run_ctx_; }
    __device__ void set_caller(std::coroutine_handle<> caller) {
      caller_handle_ = caller;
    }
  };

  using handle_type = std::coroutine_handle<promise_type>;

 private:
  handle_type handle_;

 public:
  __device__ explicit TaskResume(handle_type h) : handle_(h) {}
  __device__ TaskResume() : handle_(nullptr) {}

  __device__ TaskResume(TaskResume &&other) noexcept
      : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  __device__ TaskResume &operator=(TaskResume &&other) noexcept {
    if (this != &other) {
      if (handle_) handle_.destroy();
      handle_ = other.handle_;
      other.handle_ = nullptr;
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
    if (handle_) { handle_.destroy(); handle_ = nullptr; }
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

  __device__ bool await_ready() const noexcept { return false; }

  template <typename PromiseT>
  __device__ std::coroutine_handle<>
  await_suspend(std::coroutine_handle<PromiseT> caller_handle) noexcept {
    handle_.promise().set_caller(caller_handle);

    // Propagate RunContext from caller to inner
    if constexpr (requires { caller_handle.promise().get_run_context(); }) {
      auto *ctx = caller_handle.promise().get_run_context();
      if (ctx) handle_.promise().set_run_context(ctx);
    }

    return handle_;  // symmetric transfer
  }

  __device__ void await_resume() noexcept {
    if (handle_) { handle_.destroy(); handle_ = nullptr; }
  }
};

// ============================================================================
// YieldAwaiter / yield() -- same API as CPU side
// ============================================================================

/**
 * Awaitable that yields control from a coroutine back to the worker.
 * Obtains the RunContext from the promise, matching the CPU-side pattern.
 *
 * Usage:
 *   co_await chi::yield();       // Yield, resume next iteration
 *   co_await chi::yield(10);     // Yield, skip 10 poll iterations
 */
class YieldAwaiter {
 private:
  u32 spin_count_;

 public:
  __device__ explicit YieldAwaiter(u32 spins = 0) : spin_count_(spins) {}

  __device__ bool await_ready() const noexcept { return false; }

  /**
   * Suspend the coroutine and mark context as yielded.
   * Clang type-erases the handle to coroutine_handle<void> for custom
   * awaiters, so we recover the typed handle via from_address() to access
   * the promise and its RunContext.
   */
  __device__ void
  await_suspend(std::coroutine_handle<> handle) noexcept {
    auto typed = std::coroutine_handle<TaskResume::promise_type>::from_address(
        handle.address());
    auto *ctx = typed.promise().get_run_context();
    if (!ctx) return;
    ctx->coro_handle_ = handle;
    ctx->is_yielded_ = true;
    ctx->yield_spin_count_ = spin_count_;
    ctx->spins_remaining_ = spin_count_;
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
// CoroutineEntry / CoroutineScheduler
// ============================================================================

struct CoroutineEntry {
  std::coroutine_handle<TaskResume::promise_type> handle_;
  RunContext run_ctx_;
  bool occupied_;

  __host__ __device__ CoroutineEntry()
      : handle_(nullptr), occupied_(false) {}
};

/**
 * Fixed-capacity scheduler for suspended coroutines.
 * One per GPU block.  Manages yield/resume lifecycle.
 *
 * Entries are allocated via RunContext::Alloc (CHI_IPC->AllocateBuffer
 * in the runtime, device malloc in standalone tests).  This avoids
 * placing ~1.5KB on the CUDA kernel stack.
 */
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

  /**
   * Initialize with externally-allocated entries buffer.
   * @param entries Pointer to array of CoroutineEntry (allocated by caller)
   * @param capacity Number of entries in the array
   */
  __host__ __device__ void Init(CoroutineEntry *entries, u32 capacity) {
    entries_ = entries;
    capacity_ = capacity;
    count_ = 0;
    for (u32 i = 0; i < capacity_; ++i) {
      entries_[i].occupied_ = false;
      entries_[i].handle_ = nullptr;
    }
  }

  /**
   * Allocate entries from RunContext's allocator and initialize.
   * @param ctx RunContext whose allocator to use
   * @param capacity Number of entries (default kMaxSuspended)
   */
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

  /**
   * Free entries using RunContext's allocator.
   */
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

}  // namespace chi

#endif  // CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_
