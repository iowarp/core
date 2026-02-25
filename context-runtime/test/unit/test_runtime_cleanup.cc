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
 * Runtime Cleanup and Finalization Tests
 *
 * Tests proper cleanup paths including ServerFinalize() and ClientFinalize()
 * which are currently not exercised by other tests.
 */

#include "../simple_test.h"

#include <cstdlib>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/singletons.h"

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

TEST_CASE("Cleanup - ClearClientPool", "[cleanup][ipc][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  std::vector<FullPtr<char>> buffers;
  for (int i = 0; i < 10; ++i) {
    auto buf = ipc->AllocateBuffer(1024);
    if (!buf.IsNull()) {
      buffers.push_back(buf);
    }
  }

  REQUIRE(buffers.size() > 0);

  for (auto &buf : buffers) {
    ipc->FreeBuffer(buf);
  }

  ipc->ClearClientPool();
}

// ============================================================================
// IPC Cleanup Utility Tests
// ============================================================================

TEST_CASE("Cleanup - ClearUserIpcs", "[cleanup][ipc][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  size_t cleared = ipc->ClearUserIpcs();
  (void)cleared;
}

TEST_CASE("Cleanup - WreapDeadIpcs", "[cleanup][ipc][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  size_t reaped = ipc->WreapDeadIpcs();
  (void)reaped;
}

TEST_CASE("Cleanup - WreapAllIpcs", "[cleanup][ipc][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  size_t reaped = ipc->WreapAllIpcs();
  (void)reaped;
}

// ============================================================================
// Memory Registration Tests
// ============================================================================

TEST_CASE("Cleanup - RegisterMemory", "[cleanup][ipc][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  hipc::AllocatorId custom_alloc_id(1, 42);
  bool registered = ipc->RegisterMemory(custom_alloc_id);
  (void)registered;

  registered = ipc->RegisterMemory(custom_alloc_id);
  (void)registered;
}

TEST_CASE("Cleanup - GetClientShmInfo", "[cleanup][ipc][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  ClientShmInfo info = ipc->GetClientShmInfo(0);
  (void)info;
}

// ============================================================================
// Thread Management Tests
// ============================================================================

TEST_CASE("Cleanup - Client Thread Flags", "[cleanup][ipc][threads]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  ipc->SetIsClientThread(true);
  REQUIRE(ipc->GetIsClientThread() == true);

  ipc->SetIsClientThread(false);
  REQUIRE(ipc->GetIsClientThread() == false);
}

// ============================================================================
// Global Cleanup - Finalize once at the end
// ============================================================================

TEST_CASE("Cleanup - ZZZ Final Cleanup", "[cleanup][ipc]") {
  chi::CHIMAERA_FINALIZE();
  SIMPLE_TEST_HARD_EXIT(0);
}

SIMPLE_TEST_MAIN()
