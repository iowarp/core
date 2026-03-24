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

#ifndef CHIMAERA_COMPILER_GPU_COROUTINE_HANDLE_H_
#define CHIMAERA_COMPILER_GPU_COROUTINE_HANDLE_H_

/**
 * @file gpu_coroutine_handle.h
 * @brief GPU-compatible C++20 coroutine primitives using Clang builtins.
 *
 * Neither libstdc++ nor libc++ annotate std::coroutine_handle methods with
 * __device__, so we provide our own implementation that wraps Clang's
 * __builtin_coro_* builtins directly with proper __host__ __device__
 * annotations.
 *
 * This header MUST be included BEFORE any standard library header that
 * might transitively include <coroutine>. It defines the libstdc++ and
 * libc++ include guards to prevent their versions from loading.
 *
 * Only used in translation units compiled by Clang as CUDA (not nvcc).
 */

// Prevent libstdc++ and libc++ <coroutine> from being included.
// Their implementations lack __device__ annotations.
#define _GLIBCXX_COROUTINE 1
#define _LIBCPP_COROUTINE 1

namespace std {

/**
 * Type-erased coroutine handle (void specialization).
 * Wraps a raw coroutine frame pointer with GPU-callable resume/destroy/done.
 */
template <typename Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  void *frame_ = nullptr;

  __host__ __device__ constexpr coroutine_handle() noexcept = default;
  __host__ __device__ constexpr coroutine_handle(decltype(nullptr)) noexcept
      : frame_(nullptr) {}

  __host__ __device__ constexpr explicit operator bool() const noexcept {
    return frame_ != nullptr;
  }

  __host__ __device__ constexpr void *address() const noexcept {
    return frame_;
  }

  /**
   * Reconstruct a handle from a raw frame address.
   * @param addr The raw coroutine frame pointer
   * @return A coroutine_handle wrapping the address
   */
  __host__ __device__ static constexpr coroutine_handle
  from_address(void *addr) noexcept {
    coroutine_handle h;
    h.frame_ = addr;
    return h;
  }

  /**
   * Resume the suspended coroutine.
   * Must be called from device code only.
   */
  __host__ __device__ void resume() const { __builtin_coro_resume(frame_); }

  /**
   * Destroy the coroutine frame and free its memory.
   * Calls promise_type::operator delete if defined.
   */
  __host__ __device__ void destroy() const { __builtin_coro_destroy(frame_); }

  /**
   * Check if the coroutine has reached its final suspension point.
   * @return true if the coroutine is done
   */
  __host__ __device__ bool done() const { return __builtin_coro_done(frame_); }

  __host__ __device__ friend constexpr bool
  operator==(coroutine_handle a, coroutine_handle b) noexcept {
    return a.frame_ == b.frame_;
  }

  __host__ __device__ friend constexpr bool
  operator!=(coroutine_handle a, coroutine_handle b) noexcept {
    return a.frame_ != b.frame_;
  }
};

/**
 * Typed coroutine handle with access to the promise object.
 * @tparam Promise The promise_type of the coroutine
 */
template <typename Promise>
struct coroutine_handle : coroutine_handle<void> {
  using coroutine_handle<void>::coroutine_handle;

  /**
   * Reconstruct a typed handle from a raw frame address.
   * Hides the base-class version so the return type is correct.
   */
  __host__ __device__ static constexpr coroutine_handle
  from_address(void *addr) noexcept {
    coroutine_handle h;
    h.frame_ = addr;
    return h;
  }

  /**
   * Create a handle from a reference to the promise object.
   * Uses __builtin_coro_promise with from_promise=true to compute the
   * frame pointer from the promise address.
   * @param p Reference to the promise object
   * @return A typed coroutine_handle
   */
  __host__ __device__ static coroutine_handle from_promise(Promise &p) noexcept {
    coroutine_handle h;
    h.frame_ = __builtin_coro_promise(&p, alignof(Promise), true);
    return h;
  }

  /**
   * Access the promise object of the coroutine.
   * Uses __builtin_coro_promise with from_promise=false to compute the
   * promise address from the frame pointer.
   * @return Reference to the promise object
   */
  __host__ __device__ Promise &promise() const {
    return *static_cast<Promise *>(
        __builtin_coro_promise(frame_, alignof(Promise), false));
  }
};

// -- noop_coroutine --

struct __noop_coro_promise {};

template <>
struct coroutine_handle<__noop_coro_promise> : coroutine_handle<void> {
  using promise_type = __noop_coro_promise;

  /**
   * Construct a handle to the builtin noop coroutine.
   * The noop coroutine's resume() and destroy() are no-ops.
   */
  __host__ __device__ coroutine_handle() noexcept {
    frame_ = __builtin_coro_noop();
  }
};

using noop_coroutine_handle = coroutine_handle<__noop_coro_promise>;

/**
 * Get a handle to the noop coroutine.
 * @return A noop_coroutine_handle whose resume/destroy are no-ops
 */
__host__ __device__ inline noop_coroutine_handle noop_coroutine() noexcept {
  return noop_coroutine_handle{};
}

// -- Trivial awaitables --

/**
 * Awaitable that always suspends.
 */
struct suspend_always {
  __host__ __device__ constexpr bool await_ready() const noexcept {
    return false;
  }
  __host__ __device__ constexpr void
  await_suspend(coroutine_handle<>) const noexcept {}
  __host__ __device__ constexpr void await_resume() const noexcept {}
};

/**
 * Awaitable that never suspends.
 */
struct suspend_never {
  __host__ __device__ constexpr bool await_ready() const noexcept {
    return true;
  }
  __host__ __device__ constexpr void
  await_suspend(coroutine_handle<>) const noexcept {}
  __host__ __device__ constexpr void await_resume() const noexcept {}
};

/**
 * Coroutine traits: extracts promise_type from the coroutine return type.
 * Clang's coroutine lowering looks for std::coroutine_traits.
 */
template <typename T, typename... Args>
struct coroutine_traits {
  using promise_type = typename T::promise_type;
};

}  // namespace std

#endif  // CHIMAERA_COMPILER_GPU_COROUTINE_HANDLE_H_
