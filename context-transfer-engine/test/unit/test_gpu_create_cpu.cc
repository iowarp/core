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
 * GPU Create test: verifies that AsyncCreate called from a GPU kernel
 * correctly creates a pool via the CPU admin worker.
 *
 * The test uses fork-client mode (kClient, fork=true) which spawns a background
 * chimaera server. The GPU kernel calls AsyncCreate with ToLocalCpu() routing,
 * which routes through the gpu2cpu queue to a CPU worker in the server process.
 * The CPU admin worker handles pool creation and marks the task complete.
 */

#include "simple_test.h"

#include <chimaera/chimaera.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/singletons.h>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

// Forward declaration of GPU kernel wrapper (defined in test_gpu_create_gpu.cu)
extern "C" int run_gpu_create_test(const char *pool_name,
                                   chi::PoolId target_pool_id,
                                   int *out_return_code);

namespace {
  bool g_initialized = false;
}

class GpuCreateFixture {
 public:
  GpuCreateFixture() {
    if (g_initialized) return;

    INFO("Initializing Chimaera as fork client for GPU create test...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      INFO("CHIMAERA_INIT failed");
      return;
    }

    std::this_thread::sleep_for(500ms);
    g_initialized = true;
    INFO("GpuCreateFixture ready");
  }
};

/**
 * Test: GPU kernel calls AsyncCreate with ToLocalCpu() routing.
 * The CPU admin worker processes the GetOrCreatePool request and returns 0.
 *
 * The GPU orchestrator is paused before launching the test kernel because
 * persistent spinning kernels prevent concurrent kernel execution within
 * the same CUDA context (RTX 4070 Laptop / Ada Lovelace limitation).
 * The CPU worker thread (not the GPU orchestrator) handles the gpu2cpu queue,
 * so pausing the orchestrator does not affect task processing.
 */
TEST_CASE("GpuCreate - AsyncCreate from GPU kernel", "[gpu][create]") {
  auto *f = hshm::Singleton<GpuCreateFixture>::GetInstance();
  REQUIRE(g_initialized);

  // Pause the GPU orchestrator so its persistent kernel does not block
  // the test kernel from being scheduled on the GPU.
  CHI_IPC->PauseGpuOrchestrator();

  // Use a unique pool ID for this test
  chi::PoolId target_pool_id(999, 0);
  int return_code = -1;

  int result = run_gpu_create_test("wrp_cte_core_gpu_test",
                                   target_pool_id,
                                   &return_code);

  // Resume the GPU orchestrator after the test kernel has completed.
  CHI_IPC->ResumeGpuOrchestrator();

  INFO("run_gpu_create_test returned: " << result
       << ", task return_code: " << return_code);

  REQUIRE(result == 1);       // Kernel completed successfully
  REQUIRE(return_code == 0);  // Task succeeded (pool created or already exists)
}

SIMPLE_TEST_MAIN()
