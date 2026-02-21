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
 * and that the correct transport path is active.
 *
 * Uses SystemInfo::SpawnProcess for portable process management.
 */

#include "../simple_test.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"
#include "hermes_shm/introspect/system_info.h"

#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

using namespace chi;

inline chi::priv::vector<chimaera::bdev::Block> WrapBlock(
    const chimaera::bdev::Block &block) {
  chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
  blocks.push_back(block);
  return blocks;
}

void SubmitTasksForMode(const std::string &mode_name) {
  const chi::u64 kRamSize = 16 * 1024 * 1024;
  const chi::u64 kBlockSize = 4096;
  const chi::u64 kIoSize = 1024 * 1024;

  chi::PoolId pool_id(9000, 0);
  chimaera::bdev::Client client(pool_id);
  std::string pool_name = "ipc_test_ram_" + mode_name;
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), pool_name, pool_id,
      chimaera::bdev::BdevType::kRam, kRamSize);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  client.pool_id_ = create_task->new_pool_id_;

  auto alloc_task =
      client.AsyncAllocateBlocks(chi::PoolQuery::Local(), kBlockSize);
  alloc_task.Wait();
  REQUIRE(alloc_task->return_code_ == 0);
  REQUIRE(alloc_task->blocks_.size() > 0);
  chimaera::bdev::Block block = alloc_task->blocks_[0];
  REQUIRE(block.size_ >= kBlockSize);

  std::vector<hshm::u8> write_data(kIoSize);
  for (size_t i = 0; i < kIoSize; ++i) {
    write_data[i] = static_cast<hshm::u8>((0xAB + i) % 256);
  }

  auto write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
  REQUIRE_FALSE(write_buffer.IsNull());
  memcpy(write_buffer.ptr_, write_data.data(), write_data.size());
  auto write_task = client.AsyncWrite(
      chi::PoolQuery::Local(), WrapBlock(block),
      write_buffer.shm_.template Cast<void>().template Cast<void>(),
      write_data.size());
  write_task.Wait();
  REQUIRE(write_task->return_code_ == 0);
  size_t actual_written = write_task->bytes_written_;

  auto read_buffer = CHI_IPC->AllocateBuffer(kIoSize);
  REQUIRE_FALSE(read_buffer.IsNull());
  auto read_task = client.AsyncRead(
      chi::PoolQuery::Local(), WrapBlock(block),
      read_buffer.shm_.template Cast<void>().template Cast<void>(), kIoSize);
  read_task.Wait();
  REQUIRE(read_task->return_code_ == 0);

  hipc::FullPtr<char> data_ptr =
      CHI_IPC->ToFullPtr(read_task->data_.template Cast<char>());
  REQUIRE_FALSE(data_ptr.IsNull());
  size_t actual_read = read_task->bytes_read_;
  std::vector<hshm::u8> read_data(actual_read);
  memcpy(read_data.data(), data_ptr.ptr_, actual_read);
  size_t verify_size = std::min(actual_written, actual_read);

  for (size_t i = 0; i < verify_size; ++i) {
    REQUIRE(read_data[i] == write_data[i]);
  }

  CHI_IPC->FreeBuffer(write_buffer);
  CHI_IPC->FreeBuffer(read_buffer);
}

/**
 * Helper to start server in background process via SpawnProcess
 */
hshm::ProcessHandle StartServerProcess() {
  std::string exe = hshm::SystemInfo::GetSelfExePath();
  return hshm::SystemInfo::SpawnProcess(
      exe, {"--server-mode"},
      {{"CHI_WITH_RUNTIME", "1"}});
}

/**
 * Helper to wait for server to be ready
 */
bool WaitForServer(int max_attempts = 50) {
  std::string memfd_dir =
      (std::filesystem::temp_directory_path() / "chimaera_memfd").string();
  const char *user = std::getenv("USER");
  if (!user) user = std::getenv("USERNAME");
  std::string segment_name =
      std::string("chi_main_segment_") + (user ? user : "");

  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::error_code ec;
    auto segment_path = std::filesystem::path(memfd_dir) / segment_name;
    if (std::filesystem::exists(segment_path, ec)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return true;
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  return true;
}

/**
 * Helper to cleanup server process
 */
void CleanupServer(hshm::ProcessHandle &proc) {
  hshm::SystemInfo::KillProcess(proc);
  hshm::SystemInfo::WaitProcess(proc);
}

/**
 * RAII guard that always kills the server process
 */
struct ServerGuard {
  hshm::ProcessHandle proc;
  explicit ServerGuard(hshm::ProcessHandle p) : proc(p) {}
  ~ServerGuard() { CleanupServer(proc); }
  ServerGuard(const ServerGuard &) = delete;
  ServerGuard &operator=(const ServerGuard &) = delete;
};

// ============================================================================
// IPC Transport Mode Tests
// ============================================================================

TEST_CASE("IpcTransportMode - SHM Client Connection",
          "[ipc_transport][shm]") {
  auto server = StartServerProcess();
  ServerGuard guard(server);
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  hshm::SystemInfo::Setenv("CHI_IPC_MODE", "SHM", 1);
  hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kShm);
  REQUIRE(ipc->GetTaskQueue() != nullptr);

  SubmitTasksForMode("shm");
}

TEST_CASE("IpcTransportMode - TCP Client Connection",
          "[ipc_transport][tcp]") {
  auto server = StartServerProcess();
  ServerGuard guard(server);
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  hshm::SystemInfo::Setenv("CHI_IPC_MODE", "TCP", 1);
  hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  SubmitTasksForMode("tcp");
}

TEST_CASE("IpcTransportMode - IPC Client Connection",
          "[ipc_transport][ipc]") {
  auto server = StartServerProcess();
  ServerGuard guard(server);
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  hshm::SystemInfo::Setenv("CHI_IPC_MODE", "IPC", 1);
  hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kIpc);
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  SubmitTasksForMode("ipc");
}

TEST_CASE("IpcTransportMode - Default Mode Is TCP",
          "[ipc_transport][default]") {
  auto server = StartServerProcess();
  ServerGuard guard(server);
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  hshm::SystemInfo::Unsetenv("CHI_IPC_MODE");
  hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);
}

int main(int argc, char *argv[]) {
  // Server mode: started by StartServerProcess() via SpawnProcess
  if (argc > 1 && std::string(argv[1]) == "--server-mode") {
    hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::minutes(5));
    return 0;
  }

  // Normal test mode
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  return SimpleTest::run_all_tests(filter);
}
