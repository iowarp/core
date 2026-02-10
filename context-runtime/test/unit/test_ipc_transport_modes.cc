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
 * IPC Transport Mode Tests
 *
 * Tests that each IPC transport mode (SHM, TCP, IPC) initializes correctly
 * and that the correct transport path is active. Each test case forks a
 * server, sets CHI_IPC_MODE, connects as client, and verifies mode state.
 */

#include "../simple_test.h"

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

using namespace chi;

/**
 * Helper to start server in background process
 * Returns server PID
 */
pid_t StartServerProcess() {
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Redirect child's stdout/stderr to /dev/null to prevent massive
    // worker log output from flooding shared pipes and blocking parent
    freopen("/dev/null", "w", stdout);
    freopen("/dev/null", "w", stderr);

    // Child process: Start runtime server
    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      _exit(1);
    }

    // Keep server alive for tests
    // Server will be killed by parent process
    sleep(300);  // 5 minutes max
    _exit(0);
  }
  return server_pid;
}

/**
 * Helper to wait for server to be ready
 */
bool WaitForServer(int max_attempts = 50) {
  // The main shared memory segment name is "chi_main_segment_${USER}"
  const char *user = std::getenv("USER");
  std::string shm_name = std::string("/chi_main_segment_") + (user ? user : "");

  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Check if shared memory exists (indicates server is ready)
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd >= 0) {
      close(fd);
      // Give it a bit more time to fully initialize
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return true;
    }
  }
  return false;
}

/**
 * Helper to cleanup shared memory
 */
void CleanupSharedMemory() {
  const char *user = std::getenv("USER");
  std::string main_seg = std::string("/chi_main_segment_") + (user ? user : "");
  shm_unlink(main_seg.c_str());
}

/**
 * Helper to cleanup server process
 */
void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(server_pid, SIGTERM);
    int status;
    waitpid(server_pid, &status, 0);
    CleanupSharedMemory();
  }
}

// ============================================================================
// IPC Transport Mode Tests
// ============================================================================

TEST_CASE("IpcTransportMode - SHM Client Connection",
          "[ipc_transport][shm]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Set SHM mode and connect as external client
  setenv("CHI_IPC_MODE", "SHM", 1);
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kShm);

  // SHM mode attaches to shared queues
  REQUIRE(ipc->GetTaskQueue() != nullptr);

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("IpcTransportMode - TCP Client Connection",
          "[ipc_transport][tcp]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Set TCP mode and connect as external client
  setenv("CHI_IPC_MODE", "TCP", 1);
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);

  // TCP mode does not attach to shared queues
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("IpcTransportMode - IPC Client Connection",
          "[ipc_transport][ipc]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Set IPC (Unix Domain Socket) mode and connect as external client
  setenv("CHI_IPC_MODE", "IPC", 1);
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kIpc);

  // IPC mode does not attach to shared queues
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("IpcTransportMode - Default Mode Is TCP",
          "[ipc_transport][default]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Unset CHI_IPC_MODE to test default behavior
  unsetenv("CHI_IPC_MODE");
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);

  // Cleanup
  CleanupServer(server_pid);
}

SIMPLE_TEST_MAIN()
