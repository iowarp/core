/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * CPU-side test for cross-warp parallelism validation.
 *
 * Launches the GPU orchestrator with 64 blocks x 32 threads (2048 threads
 * = 64 warps), submits a single GpuSubmitTask with parallelism=2048,
 * and verifies that every lane executed the handler via an atomic counter.
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
extern "C" int run_gpu_parallelism_test(
    chi::PoolId pool_id,
    chi::u32 parallelism,
    chi::u32 *out_counter);
#else
extern "C" __attribute__((weak)) int run_gpu_parallelism_test(
    chi::PoolId, chi::u32, chi::u32 *) {
  return -200;
}
#endif

static bool g_initialized = false;
static int g_test_counter = 0;

/**
 * Test: Cross-warp parallelism with 64 warps (2048 threads).
 * Each lane atomically increments a device counter.
 * Verifies counter == parallelism after task completion.
 */
TEST_CASE("gpu_cross_warp_parallelism_2048", "[gpu][parallelism][cross_warp]") {
  // Configure GPU orchestrator: 64 blocks x 32 threads = 64 warps
  setenv("CHI_GPU_BLOCKS", "64", 1);
  setenv("CHI_GPU_THREADS", "32", 1);

  if (!g_initialized) {
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    g_initialized = true;
    SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
    std::this_thread::sleep_for(500ms);
  }

  // Create MOD_NAME pool
  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);
  chimaera::MOD_NAME::Client client(pool_id);
  std::string pool_name = "parallelism_test_" + std::to_string(pool_id.ToU64());
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), pool_name, pool_id);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  std::this_thread::sleep_for(200ms);

  // Run the parallelism test: 2048 threads = 64 warps x 32 lanes
  const chi::u32 parallelism = 64 * 32;
  chi::u32 counter = 0;
  int result = run_gpu_parallelism_test(pool_id, parallelism, &counter);

  INFO("Parallelism test result: " + std::to_string(result));
  INFO("Counter value: " + std::to_string(counter));
  INFO("Expected: " + std::to_string(parallelism));

  REQUIRE(result == 1);
  REQUIRE(counter == parallelism);

}

/**
 * Test: Single-warp parallelism (32 threads) as a baseline.
 * Verifies counter == 32 for a standard single-warp task.
 */
TEST_CASE("gpu_single_warp_parallelism_32", "[gpu][parallelism][single_warp]") {
  if (!g_initialized) {
    setenv("CHI_GPU_BLOCKS", "64", 1);
    setenv("CHI_GPU_THREADS", "32", 1);
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    g_initialized = true;
    SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
    std::this_thread::sleep_for(500ms);
  }

  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);
  chimaera::MOD_NAME::Client client(pool_id);
  std::string pool_name = "parallelism_1w_" + std::to_string(pool_id.ToU64());
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), pool_name, pool_id);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  std::this_thread::sleep_for(200ms);

  const chi::u32 parallelism = 32;
  chi::u32 counter = 0;
  int result = run_gpu_parallelism_test(pool_id, parallelism, &counter);

  INFO("Single-warp test result: " + std::to_string(result));
  INFO("Counter value: " + std::to_string(counter));

  REQUIRE(result == 1);
  REQUIRE(counter == parallelism);

}

SIMPLE_TEST_MAIN()
