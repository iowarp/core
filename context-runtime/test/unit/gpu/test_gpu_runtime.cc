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
 * Unit tests for GPU orchestrator runtime support
 *
 * Tests:
 * - Runtime initialization with GPU support
 * - Admin pool exists after init
 * - GPU queues are initialized
 * - GPU orchestrator is launched
 * - Clean finalize
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "simple_test.h"

#include <chimaera/chimaera.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/pool_manager.h>
#include <chimaera/config_manager.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>

#include <chrono>
#include <thread>

using namespace std::chrono_literals;

namespace {
  bool g_initialized = false;
}

class GpuRuntimeFixture {
 public:
  GpuRuntimeFixture() {
    if (!g_initialized) {
      INFO("Initializing Chimaera in Server mode for GPU runtime test...");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
        std::this_thread::sleep_for(500ms);
        INFO("Chimaera server initialization successful");
      } else {
        INFO("Failed to initialize Chimaera server");
      }
    }
  }
};

/**
 * Test: Server initializes successfully with GPU support
 */
TEST_CASE("GpuRuntime - Server Initialization", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
}

/**
 * Test: Admin pool exists after server init
 */
TEST_CASE("GpuRuntime - Admin Pool Exists", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *pool_mgr = CHI_POOL_MANAGER;
  REQUIRE(pool_mgr != nullptr);
  REQUIRE(pool_mgr->IsInitialized());

  // Admin pool should exist (kAdminPoolId)
  bool has_admin = pool_mgr->HasPool(chi::kAdminPoolId);
  REQUIRE(has_admin);
}

/**
 * Test: GPU queues are initialized
 */
TEST_CASE("GpuRuntime - GPU Queues Initialized", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *ipc = CHI_IPC;
  size_t gpu_count = ipc->GetGpuQueueCount();
  INFO("Number of GPU queues: " + std::to_string(gpu_count));

  // Should have at least one GPU queue (since we're running with GPU support)
  REQUIRE(gpu_count > 0);
}

/**
 * Test: GPU config parameters are accessible
 */
TEST_CASE("GpuRuntime - GPU Config Parameters", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *config = CHI_CONFIG_MANAGER;
  REQUIRE(config != nullptr);

  chi::u32 blocks = config->GetGpuBlocks();
  chi::u32 threads = config->GetGpuThreadsPerBlock();
  chi::u32 depth = config->GetGpuQueueDepth();

  INFO("GPU config: blocks=" + std::to_string(blocks) +
       ", threads_per_block=" + std::to_string(threads) +
       ", queue_depth=" + std::to_string(depth));

  REQUIRE(blocks > 0);
  REQUIRE(threads > 0);
  REQUIRE(depth > 0);
}

/**
 * Test: GPU Orchestrator is active
 */
TEST_CASE("GpuRuntime - GPU Orchestrator Active", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *ipc = CHI_IPC;
  // Check that the GPU orchestrator was created
  REQUIRE(ipc->GetGpuIpcManager()->gpu_orchestrator_ != nullptr);
}

/**
 * Test: CPU→GPU queues (to_gpu_queues_) are initialized
 */
TEST_CASE("GpuRuntime - ToGpu Queues Initialized", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *ipc = CHI_IPC;
  auto *gpu_ipc = ipc->GetGpuIpcManager();
  size_t to_gpu_count = gpu_ipc ? gpu_ipc->gpu_devices_.size() : 0;
  INFO("Number of to_gpu queues: " + std::to_string(to_gpu_count));
  REQUIRE(to_gpu_count > 0);

  // First queue should be non-null
  chi::GpuTaskQueue *q = gpu_ipc->gpu_devices_[0].cpu2gpu_queue.ptr_;
  REQUIRE(q != nullptr);
}

/**
 * Test: GPU→GPU queues (gpu_to_gpu_queues_) are initialized
 */
TEST_CASE("GpuRuntime - GpuToGpu Queues Initialized", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *ipc = CHI_IPC;
  auto *gpu_ipc = ipc->GetGpuIpcManager();
  size_t g2g_count = gpu_ipc ? gpu_ipc->gpu_devices_.size() : 0;
  INFO("Number of gpu_to_gpu queues: " + std::to_string(g2g_count));
  REQUIRE(g2g_count > 0);

  // First queue should be non-null
  chi::GpuTaskQueue *q = gpu_ipc->gpu_devices_[0].gpu2gpu_queue.ptr_;
  REQUIRE(q != nullptr);
}

/**
 * Test: GPU Orchestrator running_flag is set after launch
 */
TEST_CASE("GpuRuntime - GPU Orchestrator Running Flag", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc->GetGpuIpcManager()->gpu_orchestrator_ != nullptr);

  // Cast to gpu::WorkOrchestrator and check running_flag
  auto *launcher =
      static_cast<chi::gpu::WorkOrchestrator *>(ipc->GetGpuIpcManager()->gpu_orchestrator_);
  REQUIRE(launcher->control_ != nullptr);

  // Give the kernel time to set running_flag (it's async)
  // The 500ms sleep in GpuRuntimeFixture should be enough,
  // but poll briefly if needed.
  int attempts = 0;
  while (launcher->control_->running_flag == 0 && attempts < 100) {
    std::this_thread::sleep_for(10ms);
    ++attempts;
  }
  INFO("running_flag poll attempts: " + std::to_string(attempts));
  REQUIRE(launcher->control_->running_flag == 1);
}

/**
 * Test: Clean finalize
 * This test verifies that CHIMAERA_FINALIZE doesn't crash.
 * The actual finalize happens in SimpleTest::g_test_finalize cleanup.
 */
TEST_CASE("GpuRuntime - Ready For Finalize", "[gpu]") {
  GpuRuntimeFixture fixture;
  REQUIRE(g_initialized);

  // If we got here, the runtime is functional.
  // Finalize will be called by the test framework cleanup.
  INFO("GPU runtime is functional and ready for clean finalize");
}

#else  // No CUDA or ROCm

#include "simple_test.h"

TEST_CASE("GpuRuntime - Skipped (no GPU support)", "[gpu]") {
  INFO("GPU runtime tests skipped: CUDA/ROCm not enabled");
  REQUIRE(true);
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

SIMPLE_TEST_MAIN()
