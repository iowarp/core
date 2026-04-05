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

#ifndef HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_
#define HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_

#include <atomic>
#include <type_traits>

#include "hermes_shm/constants/macros.h"
#include "numbers.h"
#if HSHM_IS_CUDA_COMPILER
#include <cuda/atomic>
#endif
#if HSHM_IS_ROCM_COMPILER
#include <hip/hip_runtime.h>
#endif

namespace hshm::ipc {

/** Provides the API of an atomic, without being atomic */
template <typename T>
struct nonatomic {
  T x;

  /** Serialization */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(x);
  }

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE_CROSS_FUN nonatomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN nonatomic(const nonatomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS_FUN nonatomic(nonatomic &&other) : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(const nonatomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(nonatomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add(U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    T orig_x = x;
    x += (T)count;
    return orig_x;
  }

  /** System-scope fetch_add (same as fetch_add for nonatomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T fetch_add_system(U count) {
    return fetch_add(count);
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_sub(U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    T orig_x = x;
    x -= (T)count;
    return orig_x;
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    (void)order;
    return x;
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void
  store(U val, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    x = (T)val;
  }

  /** System-scope store (same as store for nonatomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store_system(U val) { x = (T)val; }

  /** System-scope load (same as load for nonatomic) */
  HSHM_INLINE_CROSS_FUN T load_system() const { return x; }

  /** Device-scope load (same as load for nonatomic) */
  HSHM_INLINE_CROSS_FUN T load_device() const { return x; }

  /** Get reference to x */
  HSHM_INLINE_CROSS_FUN T &ref() { return x; }

  /** Get const reference to x */
  HSHM_INLINE_CROSS_FUN const T &ref() const { return x; }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void exchange(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    x = count;
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    if (x == expected) {
      x = (T)desired;
      return true;
    } else {
      expected = x;
      return false;
    }
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    if (x == expected) {
      x = (T)desired;
      return true;
    } else {
      expected = x;
      return false;
    }
  }

  /** System-scope compare exchange strong (same as strong for nonatomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong_system(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_strong(expected, desired, order);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS_FUN nonatomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS_FUN nonatomic operator--(int) {
    nonatomic orig_x(x);
    --x;
    return orig_x;
  }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator+(U count) const {
    return nonatomic(x + count);
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator-(U count) const {
    return nonatomic(x - count);
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator+=(U count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator-=(U count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(U count) {
    x = count;
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator==(U other) const {
    return (static_cast<T>(other) == x);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator!=(U other) const {
    return (static_cast<T>(other) != x);
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const nonatomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const nonatomic &other) const {
    return (other.x != x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator&(U other) const {
    return nonatomic(x & other);
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator|(U other) const {
    return nonatomic(x | other);
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator^(U other) const {
    return nonatomic(x ^ other);
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator&=(U other) {
    x &= other;
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator|=(U other) {
    x |= other;
    return *this;
  }

  /** System-scope bitwise or assign (same as |= for nonatomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &or_system(U other) {
    x |= other;
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator^=(U other) {
    x ^= other;
    return *this;
  }
};

/** A wrapper for CUDA atomic operations.
 * Guarded by HSHM_IS_GPU_COMPILER because CUDA device builtins (atomicAdd,
 * atomicExch, atomicCAS, etc.) are only available when compiling with nvcc/hipcc.
 * Regular g++/clang++ compilations with HSHM_ENABLE_CUDA set should not parse
 * this class since it's only used as hipc::atomic<T> inside device code. */
#if HSHM_IS_GPU_COMPILER
template <typename T>
struct rocm_atomic {
  T x;

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic(const rocm_atomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic(rocm_atomic &&other)
      : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(const rocm_atomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(rocm_atomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add(U count, std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      return (T)atomicAdd(reinterpret_cast<unsigned long long*>(&x),
                          static_cast<unsigned long long>(
                              static_cast<T>(count)));
    } else {
      return atomicAdd(&x, static_cast<T>(count));
    }
#else
    T old = x;
    x += static_cast<T>(count);
    return old;
#endif
  }

  /** System-scope atomic fetch_add: atomicAdd_system for cross-device
   *  visibility. Use when GPU must increment a counter visible to CPU
   *  (e.g., tail_ of a GPU→CPU ring buffer in managed/UVM memory). */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add_system(U count) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      return (T)atomicAdd_system(
          reinterpret_cast<unsigned long long*>(&x),
          static_cast<unsigned long long>(static_cast<T>(count)));
    } else {
      return (T)atomicAdd_system(
          reinterpret_cast<unsigned int*>(&x),
          static_cast<unsigned int>(static_cast<T>(count)));
    }
#else
    return x.fetch_add(static_cast<T>(count), std::memory_order_seq_cst);
#endif
  }

  /** Atomic fetch_sub wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_sub(U count, std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      return (T)atomicAdd(reinterpret_cast<unsigned long long*>(&x),
                          static_cast<unsigned long long>(
                              static_cast<T>(-count)));
    } else {
      return atomicAdd(&x, static_cast<T>(-count));
    }
#else
    T old = x;
    x -= static_cast<T>(count);
    return old;
#endif
  }

  /** Atomic load wrapper (volatile to prevent compiler caching in loops) */
  HSHM_INLINE_CROSS_FUN T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return *reinterpret_cast<const volatile T*>(&x);
  }

  /**
   * Device-scope atomic load: bypasses per-SM L1 cache and loads directly
   * from L2 using PTX ld.global.cg (cache-global). This is necessary when
   * the producer is on a different SM or in a different concurrent kernel,
   * because volatile loads hit L1 which is NOT coherent across SMs.
   *
   * Unlike atomicOr(&x, 0), this is a pure read — no read-modify-write,
   * so it doesn't contend with writers on the same cache line.
   *
   * Falls back to volatile read on host.
   */
  HSHM_INLINE_CROSS_FUN T
  load_device() const {
#if HSHM_IS_GPU
    // Device-scope atomic read: atomicAdd(&x, 0) bypasses L1 and reads
    // from L2 (which is coherent across SMs). This is a read-modify-write
    // but with 0 addend, so it doesn't change the value.
    // Only use this where cross-SM/cross-kernel visibility is needed;
    // hot spin loops (e.g. WaitForSpace) should use volatile load() instead.
    if constexpr (sizeof(T) == 8) {
      return (T)atomicAdd(
          const_cast<unsigned long long*>(
              reinterpret_cast<const unsigned long long*>(&x)),
          0ULL);
    } else if constexpr (sizeof(T) == 4) {
      return (T)atomicAdd(
          const_cast<unsigned int*>(
              reinterpret_cast<const unsigned int*>(&x)),
          0u);
    } else {
      return *reinterpret_cast<const volatile T*>(&x);
    }
#else
    return *reinterpret_cast<const volatile T*>(&x);
#endif
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    exchange(count);
  }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  exchange(U count, std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      return (T)atomicExch(reinterpret_cast<unsigned long long*>(&x),
                           static_cast<unsigned long long>(
                               static_cast<T>(count)));
    } else {
      return atomicExch(&x, static_cast<T>(count));
    }
#else
    T old = x;
    x = static_cast<T>(count);
    return old;
#endif
  }

  /** System-scope atomic store: fence first so prior writes are globally
   *  visible before the signal value is updated. Uses volatile write +
   *  threadfence_system for cross-device visibility. atomicExch_system
   *  can hang on pinned host memory in persistent kernels. */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store_system(U count) {
#if HSHM_IS_GPU
    __threadfence_system();
    *reinterpret_cast<volatile T*>(&x) = static_cast<T>(count);
    __threadfence_system();
#else
    x = static_cast<T>(count);
#endif
  }

  /** System-scope atomic load: bypasses GPU L2 cache so GPU can observe
   *  CPU-written values in pinned host memory without stale L2 reads.
   *  Uses volatile read which bypasses L2 cache on NVIDIA GPUs for
   *  pinned host memory accessed via UVA. atomicAdd_system(&x, 0) can
   *  hang on pinned host memory in persistent kernels. */
  HSHM_INLINE_CROSS_FUN T load_system() const {
    return *reinterpret_cast<const volatile T*>(&x);
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      auto old = atomicCAS(reinterpret_cast<unsigned long long*>(
                               const_cast<T*>(&x)),
                           *reinterpret_cast<unsigned long long*>(&expected),
                           static_cast<unsigned long long>(
                               static_cast<T>(desired)));
      T old_t = *reinterpret_cast<T*>(&old);
      if (old_t == expected) return true;
      expected = old_t;
      return false;
    } else {
      T old = atomicCAS(const_cast<T*>(&x), expected,
                        static_cast<T>(desired));
      if (old == expected) return true;
      expected = old;
      return false;
    }
#else
    if (x == expected) {
      x = static_cast<T>(desired);
      return true;
    }
    expected = x;
    return false;
#endif
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      auto old = atomicCAS(reinterpret_cast<unsigned long long*>(
                               const_cast<T*>(&x)),
                           *reinterpret_cast<unsigned long long*>(&expected),
                           static_cast<unsigned long long>(
                               static_cast<T>(desired)));
      T old_t = *reinterpret_cast<T*>(&old);
      if (old_t == expected) return true;
      expected = old_t;
      return false;
    } else {
      T old = atomicCAS(const_cast<T*>(&x), expected,
                        static_cast<T>(desired));
      if (old == expected) return true;
      expected = old;
      return false;
    }
#else
    if (x == expected) {
      x = static_cast<T>(desired);
      return true;
    }
    expected = x;
    return false;
#endif
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator++() {
    fetch_add(1);
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic operator++(int) {
    T old = fetch_add(1);
    return rocm_atomic(old);
  }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator--() {
    fetch_sub(1);
    return (*this);
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic operator--(int) {
    T old = fetch_sub(1);
    return rocm_atomic(old);
  }

  /** Atomic add operator (non-destructive, reads then adds) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator+(U count) const {
    return rocm_atomic(load() + count);
  }

  /** Atomic subtract operator (non-destructive, reads then subtracts) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator-(U count) const {
    return rocm_atomic(load() - count);
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator+=(U count) {
    fetch_add(count);
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator-=(U count) {
    fetch_sub(count);
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(U count) {
    store(count);
    return *this;
  }

  /** Equality check (non-destructive: load then compare) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator==(U other) const {
    return load() == static_cast<T>(other);
  }

  /** Inequality check (non-destructive: load then compare) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator!=(U other) const {
    return load() != static_cast<T>(other);
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const rocm_atomic &other) const {
    return load() == other.load();
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const rocm_atomic &other) const {
    return load() != other.load();
  }

  /** Bitwise and (non-destructive: load then AND) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator&(U other) const {
    return rocm_atomic(load() & static_cast<T>(other));
  }

  /** Bitwise or (non-destructive: load then OR) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator|(U other) const {
    return rocm_atomic(load() | static_cast<T>(other));
  }

  /** Bitwise xor (non-destructive: load then XOR) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator^(U other) const {
    return rocm_atomic(load() ^ static_cast<T>(other));
  }

  /** Bitwise and assign (device-scope on GPU, plain on host) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator&=(U other) {
#if HSHM_IS_GPU
    atomicAnd(reinterpret_cast<unsigned int*>(&x),
              static_cast<unsigned int>(other));
#else
    x &= static_cast<T>(other);
#endif
    return *this;
  }

  /** Bitwise or assign (device-scope on GPU, plain on host) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator|=(U other) {
#if HSHM_IS_GPU
    atomicOr(reinterpret_cast<unsigned int*>(&x),
             static_cast<unsigned int>(other));
#else
    x |= static_cast<T>(other);
#endif
    return *this;
  }

  /** System-scope bitwise or assign: fence first (prior GPU writes globally
   *  visible), then volatile RMW so CPU can observe the flag update.
   *  atomicOr_system can hang on pinned host memory in persistent kernels. */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &or_system(U other) {
#if HSHM_IS_GPU
    __threadfence_system();
    volatile T *vptr = reinterpret_cast<volatile T*>(&x);
    *vptr = *vptr | static_cast<T>(other);
    __threadfence_system();
#else
    x |= static_cast<T>(other);
#endif
    return *this;
  }

  /** System-scope compare-exchange strong: bypasses GPU L2 so GPU can
   *  atomically claim entries written by CPU in pinned host memory. */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong_system(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      auto old = atomicCAS_system(
          reinterpret_cast<unsigned long long*>(const_cast<T*>(&x)),
          *reinterpret_cast<unsigned long long*>(&expected),
          static_cast<unsigned long long>(static_cast<T>(desired)));
      T old_t = *reinterpret_cast<T*>(&old);
      if (old_t == expected) return true;
      expected = old_t;
      return false;
    } else {
      T old = atomicCAS_system(const_cast<T*>(&x), expected,
                               static_cast<T>(desired));
      if (old == expected) return true;
      expected = old;
      return false;
    }
#else
    if (x == expected) {
      x = static_cast<T>(desired);
      return true;
    }
    expected = x;
    return false;
#endif
  }

  /** Bitwise xor assign (device-scope on GPU, plain on host) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator^=(U other) {
#if HSHM_IS_GPU
    atomicXor(reinterpret_cast<unsigned int*>(&x),
              static_cast<unsigned int>(other));
#else
    x ^= static_cast<T>(other);
#endif
    return *this;
  }

  /** Serialization */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(x);
  }
};
#endif

/** A wrapper around std::atomic */
template <typename T>
struct std_atomic {
  std::atomic<T> x;

  /** Serialization - properly handles std::atomic by loading/storing value */
  template <typename Ar>
  void save(Ar &ar) const {
    T val = x.load(std::memory_order_relaxed);
    ar(val);
  }

  /** Deserialization - properly handles std::atomic by loading/storing value */
  template <typename Ar>
  void load(Ar &ar) {
    T val{};
    ar(val);
    x.store(val, std::memory_order_relaxed);
  }

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE std_atomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE std_atomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE std_atomic(const std_atomic &other) : x(other.x.load()) {}

  /* Move constructor */
  HSHM_INLINE std_atomic(std_atomic &&other) : x(other.x.load()) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN std_atomic &operator=(const std_atomic &other) {
#if HSHM_IS_HOST
    x = other.x.load();
#endif
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN std_atomic &operator=(std_atomic &&other) {
#if HSHM_IS_HOST
    x = other.x.load();
#endif
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE T fetch_add(U count,
                          std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_add(count, order);
  }

  /** System-scope fetch_add (same as seq_cst fetch_add for std_atomic) */
  template <typename U>
  HSHM_INLINE T fetch_add_system(U count) {
    return x.fetch_add(count, std::memory_order_seq_cst);
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE T fetch_sub(U count,
                          std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_sub(count, order);
  }

  /** Atomic load wrapper */
  HSHM_INLINE T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return x.load(order);
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store(U count,
                         std::memory_order order = std::memory_order_seq_cst) {
#if HSHM_IS_HOST
    x.store(count, order);
#endif
  }

  /** System-scope store (same as store for std_atomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store_system(U count) {
#if HSHM_IS_HOST
    x.store(count, std::memory_order_seq_cst);
#else
    (void)count;
#endif
  }

  /** System-scope load (same as load for std_atomic) */
  HSHM_INLINE T load_system() const {
    return x.load(std::memory_order_seq_cst);
  }

  /** Device-scope load (same as load on host; PTX ld.global.cg on GPU) */
  HSHM_INLINE T load_device() const {
    return x.load(std::memory_order_seq_cst);
  }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE void exchange(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    x.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_strong(expected, desired, order);
  }

  /** System-scope compare exchange strong (same as strong for std_atomic) */
  template <typename U>
  HSHM_INLINE bool compare_exchange_strong_system(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_strong(expected, desired, order);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE std_atomic &operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE std_atomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE std_atomic &operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE std_atomic operator--(int) { return atomic(x - 1); }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE std_atomic operator+(U count) const {
    return x + count;
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE std_atomic operator-(U count) const {
    return x - count;
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator+=(U count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator-=(U count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN std_atomic &operator=(U count) {
#if HSHM_IS_HOST
    x.exchange(count);
#else
    (void)count;
#endif
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE bool operator==(U other) const {
    return (other == x);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE bool operator!=(U other) const {
    return (other != x);
  }

  /** Equality check */
  HSHM_INLINE bool operator==(const std_atomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE bool operator!=(const std_atomic &other) const {
    return (other.x != x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE std_atomic operator&(U other) const {
    return x & other;
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE std_atomic operator|(U other) const {
    return x | other;
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE std_atomic operator^(U other) const {
    return x ^ other;
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator&=(U other) {
    x &= other;
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator|=(U other) {
    x |= other;
    return *this;
  }

  /** System-scope bitwise or assign (same as |= for std_atomic) */
  template <typename U>
  HSHM_INLINE std_atomic &or_system(U other) {
    x |= other;
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator^=(U other) {
    x ^= other;
    return *this;
  }
};

#if HSHM_IS_HOST
template <typename T>
using atomic = std_atomic<T>;
#endif

#if HSHM_IS_GPU && HSHM_ENABLE_CUDA_OR_ROCM
template <typename T>
using atomic = rocm_atomic<T>;
#endif

#if HSHM_IS_GPU && !HSHM_ENABLE_CUDA_OR_ROCM
// Fallback for nvcc's device-compilation pass when HSHM_ENABLE_CUDA=0.
// HSHM_IS_GPU=1 (via __CUDA_ARCH__) but no GPU atomic backend is configured.
// nonatomic<T> is HSHM_CROSS_FUN-safe and prevents "atomic is not a template"
// cascade errors in downstream lock/allocator headers.
template <typename T>
using atomic = nonatomic<T>;
#endif

template <typename T, bool is_atomic>
using opt_atomic =
    typename std::conditional<is_atomic, atomic<T>, nonatomic<T>>::type;

/** Device-scope memory fence */
#if HSHM_IS_GPU
HSHM_GPU_FUN static void threadfence() { __threadfence(); }
#else
HSHM_INLINE static void threadfence() {
  std::atomic_thread_fence(std::memory_order_release);
}
#endif

/** System-scope memory fence (ensures GPU writes are visible to CPU) */
#if HSHM_IS_GPU
HSHM_GPU_FUN static void threadfence_system() { __threadfence_system(); }
#else
HSHM_INLINE static void threadfence_system() {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}
#endif

/**
 * Safe 64-bit warp shuffle broadcast.
 * Splits into two 32-bit shuffles to avoid potential issues with
 * __shfl_sync for 64-bit types on some GPU architectures.
 */
#if HSHM_IS_GPU
HSHM_GPU_FUN static unsigned long long shfl_sync_u64(
    unsigned mask, unsigned long long val, int src_lane) {
  unsigned int lo = __shfl_sync(mask, static_cast<unsigned int>(val), src_lane);
  unsigned int hi = __shfl_sync(mask, static_cast<unsigned int>(val >> 32), src_lane);
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#else
HSHM_INLINE static unsigned long long shfl_sync_u64(
    unsigned mask, unsigned long long val, int src_lane) {
  (void)mask; (void)src_lane;
  return val;
}
#endif

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_
