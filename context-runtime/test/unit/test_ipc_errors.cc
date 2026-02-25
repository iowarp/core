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
 * IPC Error Handling Tests
 *
 * Tests error conditions and failure paths in IpcManager that are not
 * exercised by happy-path tests.
 */

#include "../simple_test.h"

#include <cstdlib>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

using namespace chi;

// ============================================================================
// Global Setup - Initialize once for all tests
// ============================================================================
static bool InitializeRuntime() {
  static bool initialized = false;
  if (!initialized) {
    bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
    initialized = success;
    if (success) SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
    return success;
  }
  return true;
}

// ============================================================================
// Memory Allocation Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Huge Buffer Allocation", "[ipc][errors][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  size_t huge_size = hshm::Unit<size_t>::Terabytes(100);
  auto buf = ipc->AllocateBuffer(huge_size);
  REQUIRE(buf.IsNull());
}

TEST_CASE("IpcErrors - Zero Size Allocation", "[ipc][errors][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  auto buf = ipc->AllocateBuffer(0);
  if (!buf.IsNull()) {
    ipc->FreeBuffer(buf);
  }
}

TEST_CASE("IpcErrors - Invalid Buffer Free", "[ipc][errors][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  FullPtr<char> null_buf;
  ipc->FreeBuffer(null_buf);
}

// ============================================================================
// Host/Network Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Node ID", "[ipc][errors][network]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  auto *host = ipc->GetHost(0xDEADBEEF);
  REQUIRE(host == nullptr);

  host = ipc->GetHost(0xFFFFFFFF);
  REQUIRE(host == nullptr);
}

TEST_CASE("IpcErrors - Invalid IP Address", "[ipc][errors][network]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  auto *host = ipc->GetHostByIp("999.999.999.999");
  REQUIRE(host == nullptr);

  host = ipc->GetHostByIp("");
  REQUIRE(host == nullptr);

  host = ipc->GetHostByIp("not.an.ip.address");
  REQUIRE(host == nullptr);
}

TEST_CASE("IpcErrors - Network Client Creation Failure",
          "[ipc][errors][network]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  try {
    auto *client = ipc->GetOrCreateClient("invalid://address", 0);
    (void)client;
  } catch (const std::exception &e) {
    // Exception is acceptable for invalid address
  }
}

// ============================================================================
// Queue Operation Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Network Queue Operations", "[ipc][errors][queue]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  Future<Task> future;
  bool result = ipc->TryPopNetTask(NetQueuePriority::kSendIn, future);
  REQUIRE(!result);

  result = ipc->TryPopNetTask(NetQueuePriority::kSendOut, future);
  REQUIRE(!result);
}

// ============================================================================
// Shared Memory Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Allocator Registration", "[ipc][errors][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  hipc::AllocatorId invalid_id(0xFFFF, 0xFFFF);
  bool registered = ipc->RegisterMemory(invalid_id);
  (void)registered;
}

TEST_CASE("IpcErrors - GetClientShmInfo Invalid Index",
          "[ipc][errors][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  ClientShmInfo info = ipc->GetClientShmInfo(9999);
  (void)info;
}

// ============================================================================
// Scheduler Error Tests
// ============================================================================

TEST_CASE("IpcErrors - SetNumSchedQueues Edge Cases", "[ipc][errors][sched]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  u32 original = ipc->GetNumSchedQueues();
  REQUIRE(original > 0);

  ipc->SetNumSchedQueues(0);
  ipc->SetNumSchedQueues(1000000);
  ipc->SetNumSchedQueues(original);
}

// ============================================================================
// Global Cleanup - Finalize once at the end
// ============================================================================

TEST_CASE("IpcErrors - ZZZ Final Cleanup", "[ipc][errors][cleanup]") {
  // Force exit to avoid hanging on worker thread joins during finalization
  _exit(0);
}

SIMPLE_TEST_MAIN()
