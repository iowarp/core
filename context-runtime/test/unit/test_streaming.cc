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
 * Test task output streaming functionality
 * Uses only client APIs to test large output streaming (>4KB)
 */

#include "../simple_test.h"
#include "chimaera/chimaera.h"
#include "chimaera/task.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/MOD_NAME/MOD_NAME_client.h"
#include "chimaera/MOD_NAME/MOD_NAME_tasks.h"

#include <vector>
#include <chrono>
#include <thread>

using namespace chi;
using namespace std::chrono_literals;

// Test pool ID for MOD_NAME
constexpr chi::PoolId kTestModNamePoolId = chi::PoolId(200, 0);

// Global flag to track runtime initialization
static bool g_initialized = false;

/**
 * Fixture for streaming tests
 */
class StreamingTestFixture {
public:
  StreamingTestFixture() {
    if (!g_initialized) {
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }
};

TEST_CASE("Task Streaming - Small Output", "[streaming][small]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  // Initialize MOD_NAME client
  chimaera::MOD_NAME::Client client(kTestModNamePoolId);

  // Create the MOD_NAME container
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_small";
  auto create_task = client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Submit CustomTask with small data (< 4KB)
  std::string input_data = "small test data for streaming";
  auto task = client.AsyncCustom(pool_query, input_data, 42);
  task.Wait();

  // Verify task completed successfully
  REQUIRE(task->return_code_ == 0);
  REQUIRE(task->data_.size() > 0);
  INFO("Small output test completed with " << task->data_.size() << " bytes");
}

TEST_CASE("Task Streaming - Large Output (1MB)", "[streaming][large]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  // Initialize MOD_NAME client
  chimaera::MOD_NAME::Client client(kTestModNamePoolId);

  // Create the MOD_NAME container
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_large";
  auto create_task = client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Submit TestLargeOutput task - this generates 1MB output
  auto task = client.AsyncTestLargeOutput(pool_query);

  // Wait for task completion (this tests streaming if output > 4KB)
  task.Wait();

  // Verify the large output was received correctly
  REQUIRE(task->data_.size() == 1024 * 1024);  // 1MB

  // Verify the pattern: data[i] = i % 256
  bool pattern_valid = true;
  size_t error_count = 0;
  constexpr size_t MAX_ERRORS_TO_SHOW = 5;

  for (size_t i = 0; i < task->data_.size(); ++i) {
    if (task->data_[i] != static_cast<uint8_t>(i % 256)) {
      pattern_valid = false;
      if (error_count < MAX_ERRORS_TO_SHOW) {
        INFO("Pattern mismatch at index " << i
             << ": expected " << static_cast<int>(i % 256)
             << ", got " << static_cast<int>(task->data_[i]));
      }
      error_count++;
    }
  }

  if (!pattern_valid) {
    INFO("Total pattern mismatches: " << error_count);
  }

  REQUIRE(pattern_valid);

  INFO("Large output test completed: received and verified 1MB output");
  INFO("Streaming mechanism tested successfully for output > 4KB copy space");
}

TEST_CASE("FutureShm Bitfield Operations", "[streaming][bitfield]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  // Initialize MOD_NAME client
  chimaera::MOD_NAME::Client client(kTestModNamePoolId);

  // Create the MOD_NAME container
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_bitfield";
  auto create_task = client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Submit a simple task
  auto task = client.AsyncCustom(pool_query, "bitfield test", 1);

  // Get the future's shared memory structure
  auto future_shm = task.GetFutureShm();
  REQUIRE(!future_shm.IsNull());

  // Test initial state (task not complete yet)
  INFO("Testing initial bitfield state");
  // Note: We can't reliably test the state before Wait() as the task might complete very fast

  // Wait for task to complete
  task.Wait();

  // After completion, FUTURE_COMPLETE should be set
  INFO("Testing FUTURE_COMPLETE flag after Wait()");
  using FutureShm = chi::FutureShm;
  REQUIRE(future_shm->flags_.Any(FutureShm::FUTURE_COMPLETE));

  // Test manual flag operations on a separate bitfield
  hshm::abitfield32_t test_flags;
  test_flags.SetBits(0);  // Initialize

  INFO("Testing manual FUTURE_COMPLETE flag set");
  test_flags.SetBits(FutureShm::FUTURE_COMPLETE);
  REQUIRE(test_flags.Any(FutureShm::FUTURE_COMPLETE));

  INFO("Testing FUTURE_NEW_DATA flag set");
  test_flags.SetBits(FutureShm::FUTURE_NEW_DATA);
  REQUIRE(test_flags.Any(FutureShm::FUTURE_NEW_DATA));
  REQUIRE(test_flags.Any(FutureShm::FUTURE_COMPLETE)); // Should still be set

  INFO("Testing flag unset");
  test_flags.UnsetBits(FutureShm::FUTURE_NEW_DATA);
  REQUIRE_FALSE(test_flags.Any(FutureShm::FUTURE_NEW_DATA));
  REQUIRE(test_flags.Any(FutureShm::FUTURE_COMPLETE)); // Should still be set

  INFO("Bitfield operations verified successfully");
}

SIMPLE_TEST_MAIN()
