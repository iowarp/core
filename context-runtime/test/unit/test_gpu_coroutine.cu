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

/**
 * @file test_gpu_coroutine.cu
 * @brief Unit tests for GPU C++20 coroutines compiled with Clang.
 *
 * Uses the chi::gpu::TaskResume / chi::gpu::yield() API for GPU-side
 * coroutines.
 *
 * Tests verify:
 * 1. Basic coroutine creation and completion
 * 2. Yielding with co_await chi::gpu::yield()
 * 3. Nested coroutines via co_await TaskResume
 * 4. CoroutineScheduler with spin-based yield
 * 5. Multiple concurrent suspended coroutines
 */

#include <chimaera/gpu_coroutine.h>
#include <cstdio>

using chi::gpu::RunContext;
using chi::gpu::CoroutineScheduler;
using chi::gpu::TaskResume;

// ============================================================================
// Test 1: Basic coroutine -- create, resume, complete
// ============================================================================

__device__ TaskResume basic_coro(RunContext &ctx) {
  co_return;
}

__global__ void test_basic(int *result) {
  RunContext ctx;
  auto c = basic_coro(ctx);
  c.resume();
  *result = c.done() ? 1 : -1;
}

// ============================================================================
// Test 2: Coroutine that yields twice
// ============================================================================

__device__ TaskResume yielding_coro(RunContext &ctx, int *counter) {
  (*counter)++;  // 0 -> 1
  co_await chi::gpu::yield();
  (*counter)++;  // 1 -> 2
  co_await chi::gpu::yield();
  (*counter)++;  // 2 -> 3
  co_return;
}

__global__ void test_yield(int *result) {
  int counter = 0;
  RunContext ctx;
  auto c = yielding_coro(ctx, &counter);

  // Resume 1: initial_suspend -> first yield (counter = 1)
  c.resume();
  if (c.done() || counter != 1) { *result = -1; return; }

  // Resume 2: first yield -> second yield (counter = 2)
  c.resume();
  if (c.done() || counter != 2) { *result = -2; return; }

  // Resume 3: second yield -> co_return (counter = 3)
  c.resume();
  if (!c.done() || counter != 3) { *result = -3; return; }

  *result = 1;
}

// ============================================================================
// Test 3: Nested coroutines (synchronous inner)
// ============================================================================

__device__ TaskResume inner_sync(RunContext &ctx, int *counter) {
  (*counter) += 10;
  co_return;
}

__device__ TaskResume outer_sync(RunContext &ctx, int *counter) {
  (*counter) += 1;
  co_await inner_sync(ctx, counter);  // +10 via symmetric transfer
  (*counter) += 1;
  co_return;
}

__global__ void test_nested_sync(int *result) {
  int counter = 0;
  RunContext ctx;
  auto c = outer_sync(ctx, &counter);
  c.resume();
  // counter should be 12 (1 + 10 + 1), coroutine should be done
  *result = (c.done() && counter == 12) ? 1 : -counter;
}

// ============================================================================
// Test 4: CoroutineScheduler with spin-based yield
// ============================================================================

__device__ TaskResume spin_coro(RunContext &ctx, int *counter) {
  (*counter)++;
  co_await chi::gpu::yield(2);  // Skip 2 poll iterations
  (*counter)++;
  co_return;
}

__global__ void test_scheduler(int *result) {
  int counter = 0;
  RunContext ctx;
  CoroutineScheduler sched;
  sched.Init(ctx);

  auto c = spin_coro(ctx, &counter);
  c.resume();  // Runs to first yield (counter = 1)

  if (c.done()) { *result = -1; return; }
  if (counter != 1) { *result = -2; return; }

  // Insert into scheduler
  auto typed_handle = c.release();
  sched.Insert(typed_handle, ctx);

  // Iteration 1: spins_remaining = 2, decrements to 1
  sched.ResumeSuspended();
  if (counter != 1) { *result = -3; return; }

  // Iteration 2: spins_remaining = 1, decrements to 0
  sched.ResumeSuspended();
  if (counter != 1) { *result = -4; return; }

  // Iteration 3: spins_remaining = 0, resumes coroutine (counter = 2, done)
  sched.ResumeSuspended();
  if (counter != 2) { *result = -5; return; }
  if (sched.HasSuspended()) { *result = -6; return; }

  sched.Destroy(ctx);
  *result = 1;
}

// ============================================================================
// Test 5: Multiple concurrent coroutines in scheduler
// ============================================================================

__device__ TaskResume multi_coro(RunContext &ctx, int *val, int increment) {
  *val += increment;
  co_await chi::gpu::yield();
  *val += increment;
  co_return;
}

__global__ void test_multi_suspended(int *result) {
  int val_a = 0, val_b = 0;
  RunContext ctx_a, ctx_b;
  CoroutineScheduler sched;
  sched.Init(ctx_a);

  auto ca = multi_coro(ctx_a, &val_a, 1);
  ca.resume();  // val_a = 1, yielded

  auto cb = multi_coro(ctx_b, &val_b, 10);
  cb.resume();  // val_b = 10, yielded

  if (val_a != 1 || val_b != 10) { *result = -1; return; }

  // Insert both
  sched.Insert(ca.release(), ctx_a);
  sched.Insert(cb.release(), ctx_b);

  if (sched.Count() != 2) { *result = -2; return; }

  // Resume both (yield spin = 0, so immediate)
  sched.ResumeSuspended();

  if (val_a != 2 || val_b != 20) { *result = -3; return; }
  if (sched.Count() != 0) { *result = -4; return; }

  sched.Destroy(ctx_a);
  *result = 1;
}

// ============================================================================
// Test runner
// ============================================================================

__host__ static bool run_test(const char *name,
                               void (*kernel)(int *), int expected) {
  int *d_result;
  cudaMallocManaged(&d_result, sizeof(int));
  *d_result = 0;

  kernel<<<1, 1>>>(d_result);
  cudaError_t err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    fprintf(stderr, "  FAIL %s: CUDA error: %s\n", name,
            cudaGetErrorString(err));
    cudaFree(d_result);
    return false;
  }

  bool pass = (*d_result == expected);
  fprintf(stderr, "  %s %s (result=%d)\n",
          pass ? "PASS" : "FAIL", name, *d_result);
  cudaFree(d_result);
  return pass;
}

__host__ int main() {
  fprintf(stderr, "GPU Coroutine Tests\n");
  fprintf(stderr, "===================\n");

  int failures = 0;
  if (!run_test("basic_coroutine", test_basic, 1)) ++failures;
  if (!run_test("yield_twice", test_yield, 1)) ++failures;
  if (!run_test("nested_sync", test_nested_sync, 1)) ++failures;
  if (!run_test("scheduler_spin", test_scheduler, 1)) ++failures;
  if (!run_test("multi_suspended", test_multi_suspended, 1)) ++failures;

  fprintf(stderr, "===================\n");
  if (failures == 0) {
    fprintf(stderr, "All tests passed\n");
  } else {
    fprintf(stderr, "%d test(s) FAILED\n", failures);
  }

  return failures;
}
