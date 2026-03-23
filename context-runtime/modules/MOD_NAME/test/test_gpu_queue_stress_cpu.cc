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
 * CPU-side test for GPU queue stress: 64 client warps × 16 iterations
 * submitting GpuSubmitTask to 1 RT warp.
 *
 * This validates that the MPSC gpu2gpu queue handles high fan-in correctly
 * when many client warps overwhelm a single runtime warp.
 */

#include "simple_test.h"
#include <chrono>
#include <thread>
#include <hermes_shm/util/logging.h>

using namespace std::chrono_literals;

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/task.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
extern "C" int run_gpu_queue_stress_test(
    chi::PoolId pool_id,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 iterations,
    float *out_elapsed_ms);
#else
extern "C" __attribute__((weak)) int run_gpu_queue_stress_test(
    chi::PoolId, chi::u32, chi::u32, chi::u32, float *) {
  return -200;
}
#endif

static bool g_initialized = false;
static int g_test_counter = 0;

/**
 * Helper: initialize runtime, create MOD_NAME pool, return pool_id.
 */
static chi::PoolId SetupPool(const char *name_prefix) {
  if (!g_initialized) {
    // Set GPU config for max warp count needed across all tests (4 warps).
    setenv("CHI_GPU_BLOCKS", "1", 0);
    setenv("CHI_GPU_THREADS", "128", 0);
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    g_initialized = true;
    SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
    std::this_thread::sleep_for(500ms);
  }

  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);
  chimaera::MOD_NAME::Client client(pool_id);
  std::string pool_name = std::string(name_prefix) + "_" +
                           std::to_string(pool_id.ToU64());
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), pool_name, pool_id);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  std::this_thread::sleep_for(200ms);
  return pool_id;
}

/**
 * Test: 64 client warps (8×256) × 16 iterations with 1 RT warp (1×32).
 * GpuSubmitTask is trivial (multiply + add), so the bottleneck is purely
 * queue throughput. Should complete well within 10 seconds.
 */
TEST_CASE("gpu_queue_stress_64w_1rt", "[gpu][stress][queue]") {
  chi::PoolId pool_id = SetupPool("queue_stress_64w");

  // RT stays at default 1×32 = 1 warp (set during CHIMAERA_INIT)

  float elapsed_ms = 0;
  int result = run_gpu_queue_stress_test(
      pool_id,
      8,     // client_blocks
      256,   // client_threads (= 64 warps)
      16,    // iterations per warp
      &elapsed_ms);

  INFO("64 warps × 16 iters with 1 RT warp: " +
       std::to_string(elapsed_ms) + " ms, result=" +
       std::to_string(result));

  REQUIRE(result == 1);
  // 64 warps × 16 iters = 1024 trivial tasks; should finish in <5s
  REQUIRE(elapsed_ms < 5000.0f);
  fprintf(stderr, "[TEST PASS] gpu_queue_stress_64w_1rt: %.1f ms\n", elapsed_ms);
}

/**
 * Test: 16 client warps (1×512) × 16 iterations with 1 RT warp (1×32).
 * Smaller fan-in sanity check.
 */
TEST_CASE("gpu_queue_stress_16w_1rt", "[gpu][stress][queue]") {
  chi::PoolId pool_id = SetupPool("queue_stress_16w");

  float elapsed_ms = 0;
  int result = run_gpu_queue_stress_test(
      pool_id,
      2,     // client_blocks
      256,   // client_threads (= 16 warps)
      16,    // iterations
      &elapsed_ms);

  INFO("16 warps × 16 iters with 1 RT warp: " +
       std::to_string(elapsed_ms) + " ms, result=" +
       std::to_string(result));

  REQUIRE(result == 1);
  REQUIRE(elapsed_ms < 3000.0f);
  fprintf(stderr, "[TEST PASS] gpu_queue_stress_16w_1rt: %.1f ms\n", elapsed_ms);
}

/**
 * Test: 64 client warps × 16 iterations with 4 RT warps (1×128).
 * More RT capacity should improve throughput.
 */
TEST_CASE("gpu_queue_stress_64w_4rt", "[gpu][stress][queue]") {
  chi::PoolId pool_id = SetupPool("queue_stress_64w_4rt");

  // Upgrade RT to 4 warps for this test
  CHI_IPC->SetGpuOrchestratorBlocks(1, 128);

  float elapsed_ms = 0;
  int result = run_gpu_queue_stress_test(
      pool_id,
      8,     // client_blocks
      256,   // client_threads (= 64 warps)
      16,    // iterations
      &elapsed_ms);

  INFO("64 warps × 16 iters with 4 RT warps: " +
       std::to_string(elapsed_ms) + " ms, result=" +
       std::to_string(result));

  REQUIRE(result == 1);
  REQUIRE(elapsed_ms < 3000.0f);
  fprintf(stderr, "[TEST PASS] gpu_queue_stress_64w_4rt: %.1f ms\n", elapsed_ms);
}

SIMPLE_TEST_MAIN()
