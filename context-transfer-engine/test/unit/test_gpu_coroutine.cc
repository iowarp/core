/*
 * GPU Coroutine Subtask Spawning Tests
 *
 * Tests the GPU-only task dispatch and coroutine suspend/resume:
 *   1. Leaf task: GPU kernel → MOD_NAME::GpuSubmit via gpu2gpu → result
 *   2. Subtask:  GPU kernel → MOD_NAME::SubtaskTest via gpu2gpu →
 *                SubtaskTest co_awaits GpuSubmit → resume → result
 *
 * No CPU-side client task flow. CPU only does setup (pools, containers).
 */

#include "simple_test.h"
#include <chrono>
#include <thread>
#include <hermes_shm/util/logging.h>

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

using namespace std::chrono_literals;

// Forward declare GPU wrapper functions
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
extern "C" int run_gpu_leaf_task_test(chi::PoolId pool_id);
extern "C" int run_gpu_subtask_test(chi::PoolId pool_id,
                                     chi::u32 test_value,
                                     chi::u32 *out_result_value);
#else
extern "C" __attribute__((weak)) int run_gpu_leaf_task_test(chi::PoolId) {
  return -200;
}
extern "C" __attribute__((weak)) int run_gpu_subtask_test(chi::PoolId,
                                                           chi::u32,
                                                           chi::u32 *) {
  return -200;
}
#endif

static bool g_initialized = false;
static chi::PoolId g_pool_id;

/**
 * One-time setup: start runtime, create a MOD_NAME pool, register GPU container.
 */
static void EnsureInit() {
  if (g_initialized) return;

  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer, true);
  REQUIRE(success);
  SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
  std::this_thread::sleep_for(500ms);

  // Create a MOD_NAME pool. The pool auto-registers a GPU container via
  // direct cudaMemcpy (no cpu2gpu queue involvement).
  g_pool_id = chi::PoolId(900, 0);
  chimaera::MOD_NAME::Client client(g_pool_id);

  auto create_future = client.AsyncCreate(
      chi::PoolQuery::Local(),
      "mod_name::gpu_coroutine_test",
      g_pool_id);
  create_future.Wait();
  INFO("MOD_NAME create return_code: " + std::to_string(create_future->return_code_));
  REQUIRE(create_future->return_code_ == 0);

  // Wait for GPU container registration (orchestrator pause/resume cycle)
  std::this_thread::sleep_for(500ms);

  g_initialized = true;
}

/**
 * Test 1: GPU kernel dispatches a leaf MOD_NAME::GpuSubmit via gpu2gpu.
 * No coroutine yielding — the container method runs to completion.
 */
TEST_CASE("GpuCoroutine - Leaf Task", "[gpu][coroutine]") {
  EnsureInit();

  int result = run_gpu_leaf_task_test(g_pool_id);

  if (result == -200) {
    INFO("GPU not compiled, skipping");
    return;
  }
  INFO("Leaf task result: " + std::to_string(result));
  REQUIRE(result == 1);
}

/**
 * Test 2: GPU kernel dispatches MOD_NAME::SubtaskTest via gpu2gpu.
 * SubtaskTest's Run() co_awaits GpuSubmit on itself.
 * Tests: coroutine suspension, gpu2gpu sub-task dispatch,
 *        resume after sub-task completion, output propagation.
 */
TEST_CASE("GpuCoroutine - Subtask Spawn", "[gpu][coroutine]") {
  EnsureInit();

  chi::u32 test_value = 42;
  chi::u32 result_value = 0;
  int result = run_gpu_subtask_test(g_pool_id, test_value, &result_value);

  if (result == -200) {
    INFO("GPU not compiled, skipping");
    return;
  }
  INFO("Subtask test result: " + std::to_string(result));
  INFO("Result value: " + std::to_string(result_value));
  REQUIRE(result == 1);
  // GpuSubmit computes: test_value * 3 + gpu_id(0) = 126
  // SubtaskTest adds 1: 127
  REQUIRE(result_value == (test_value * 3) + 1);
}

SIMPLE_TEST_MAIN()
