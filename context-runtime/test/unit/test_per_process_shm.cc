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
 * Unit tests for per-process shared memory functionality
 *
 * Tests the IpcManager's per-process shared memory allocation with:
 * - AllocateBuffer() with allocations larger than 1GB to trigger IncreaseClientShm
 * - Multiple segment creation and allocation fallback strategies
 *
 * Uses SystemInfo::SpawnProcess for portable process management.
 */

#include <chimaera/chimaera.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hermes_shm/introspect/system_info.h"

#include "../simple_test.h"

namespace {
// Test setup helper - same pattern as other tests
bool initialize_chimaera() {
  return chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
}

/**
 * Start a Chimaera server in a child process via SpawnProcess
 */
hshm::ProcessHandle StartServerProcess() {
  std::string exe = hshm::SystemInfo::GetSelfExePath();
  return hshm::SystemInfo::SpawnProcess(
      exe, {"--server-mode"},
      {{"CHI_WITH_RUNTIME", "1"}});
}

/**
 * Wait for the server's shared memory segment to become available
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
 * Kill the server process and clean up
 */
void CleanupServer(hshm::ProcessHandle &proc) {
  hshm::SystemInfo::KillProcess(proc);
  hshm::SystemInfo::WaitProcess(proc);
}

// Constants for testing
constexpr size_t k1MB = 1ULL * 1024 * 1024;
constexpr size_t k100MB = 100ULL * 1024 * 1024;
constexpr size_t k500MB = 500ULL * 1024 * 1024;
constexpr size_t k1GB = 1ULL * 1024 * 1024 * 1024;
constexpr size_t k1_5GB = 1536ULL * 1024 * 1024;  // 1.5 GB
}  // namespace

// This test MUST be first: it spawns server+client processes and requires
// that no runtime has been initialized in the parent yet.
TEST_CASE("Per-process shared memory GetClientShmInfo",
          "[ipc][per_process_shm][shm_info][fork]") {
  // Spawn a server, then spawn a client child to test GetClientShmInfo.
  auto server = StartServerProcess();
  REQUIRE(WaitForServer());

  // Spawn a client child process
  std::string exe = hshm::SystemInfo::GetSelfExePath();
  auto client_proc = hshm::SystemInfo::SpawnProcess(
      exe, {"--client-shm-info-mode"},
      {{"CHI_WITH_RUNTIME", "0"},
       {"CHI_IPC_MODE", "SHM"}});

  int exit_code = hshm::SystemInfo::WaitProcess(client_proc);
  INFO("Client child exit code: " << exit_code);
  REQUIRE(exit_code == 0);

  CleanupServer(server);
}

TEST_CASE("Per-process shared memory AllocateBuffer medium sizes",
          "[ipc][per_process_shm][allocate][medium]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Allocate 100MB buffer") {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k100MB);

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    buffer.ptr_[0] = 0x01;
    buffer.ptr_[k100MB - 1] = static_cast<char>(0xFF);

    REQUIRE(buffer.ptr_[0] == 0x01);
    REQUIRE(static_cast<unsigned char>(buffer.ptr_[k100MB - 1]) == 0xFF);
  }

  SECTION("Allocate 500MB buffer") {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k500MB);

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    buffer.ptr_[0] = static_cast<char>(0xAA);
    buffer.ptr_[k500MB - 1] = static_cast<char>(0xBB);

    REQUIRE(static_cast<unsigned char>(buffer.ptr_[0]) == 0xAA);
    REQUIRE(static_cast<unsigned char>(buffer.ptr_[k500MB - 1]) == 0xBB);
  }
}

TEST_CASE("Per-process shared memory AllocateBuffer exceeding 1GB",
          "[ipc][per_process_shm][allocate][large]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Allocate 1.5GB buffer triggers IncreaseMemory") {
    auto start_time = std::chrono::high_resolution_clock::now();
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k1_5GB);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    INFO("Allocation took " << duration.count() << " ms");

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    buffer.ptr_[0] = 'A';
    buffer.ptr_[k1GB] = 'B';
    buffer.ptr_[k1_5GB - 1] = 'Z';

    REQUIRE(buffer.ptr_[0] == 'A');
    REQUIRE(buffer.ptr_[k1GB] == 'B');
    REQUIRE(buffer.ptr_[k1_5GB - 1] == 'Z');
  }
}

TEST_CASE("Per-process shared memory multiple large allocations",
          "[ipc][per_process_shm][allocate][multiple_large]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Multiple allocations spanning segments") {
    std::vector<hipc::FullPtr<char>> buffers;

    auto buffer1 = ipc_manager->AllocateBuffer(600ULL * k1MB);
    REQUIRE_FALSE(buffer1.IsNull());
    buffers.push_back(buffer1);

    auto buffer2 = ipc_manager->AllocateBuffer(600ULL * k1MB);
    REQUIRE_FALSE(buffer2.IsNull());
    buffers.push_back(buffer2);

    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].ptr_[0] = static_cast<char>(i);
      buffers[i].ptr_[600ULL * k1MB - 1] = static_cast<char>(i + 100);
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
      REQUIRE(buffers[i].ptr_[0] == static_cast<char>(i));
      REQUIRE(buffers[i].ptr_[600ULL * k1MB - 1] ==
              static_cast<char>(i + 100));
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
      for (size_t j = i + 1; j < buffers.size(); ++j) {
        REQUIRE(buffers[i].ptr_ != buffers[j].ptr_);
      }
    }
  }
}

TEST_CASE("Per-process shared memory allocation patterns",
          "[ipc][per_process_shm][allocate][patterns]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Mixed small and large allocations") {
    std::vector<hipc::FullPtr<char>> small_buffers;
    std::vector<hipc::FullPtr<char>> large_buffers;

    for (int i = 0; i < 5; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(k1MB);
      REQUIRE_FALSE(buffer.IsNull());
      small_buffers.push_back(buffer);
    }

    auto large1 = ipc_manager->AllocateBuffer(k500MB);
    REQUIRE_FALSE(large1.IsNull());
    large_buffers.push_back(large1);

    for (int i = 0; i < 3; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(k1MB);
      REQUIRE_FALSE(buffer.IsNull());
      small_buffers.push_back(buffer);
    }

    auto large2 = ipc_manager->AllocateBuffer(k500MB);
    REQUIRE_FALSE(large2.IsNull());
    large_buffers.push_back(large2);

    for (size_t i = 0; i < small_buffers.size(); ++i) {
      small_buffers[i].ptr_[0] = static_cast<char>(i);
      REQUIRE(small_buffers[i].ptr_[0] == static_cast<char>(i));
    }

    large_buffers[0].ptr_[0] = 'X';
    large_buffers[0].ptr_[k500MB - 1] = 'Y';
    large_buffers[1].ptr_[0] = 'A';
    large_buffers[1].ptr_[k500MB - 1] = 'B';

    REQUIRE(large_buffers[0].ptr_[0] == 'X');
    REQUIRE(large_buffers[0].ptr_[k500MB - 1] == 'Y');
    REQUIRE(large_buffers[1].ptr_[0] == 'A');
    REQUIRE(large_buffers[1].ptr_[k500MB - 1] == 'B');
  }
}

TEST_CASE("Per-process shared memory FreeBuffer",
          "[ipc][per_process_shm][free]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Free allocated buffer") {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k100MB);
    REQUIRE_FALSE(buffer.IsNull());
    buffer.ptr_[0] = 'T';
    buffer.ptr_[k100MB - 1] = 'E';
    ipc_manager->FreeBuffer(buffer);
  }

  SECTION("Allocate-free-allocate cycle") {
    hipc::FullPtr<char> buffer1 = ipc_manager->AllocateBuffer(k100MB);
    REQUIRE_FALSE(buffer1.IsNull());
    ipc_manager->FreeBuffer(buffer1);

    hipc::FullPtr<char> buffer2 = ipc_manager->AllocateBuffer(k100MB);
    REQUIRE_FALSE(buffer2.IsNull());
    buffer2.ptr_[0] = 'R';
    REQUIRE(buffer2.ptr_[0] == 'R');
  }
}

TEST_CASE("Per-process shared memory ToFullPtr conversion",
          "[ipc][per_process_shm][tofullptr]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("ToFullPtr from raw pointer") {
    hipc::FullPtr<char> original = ipc_manager->AllocateBuffer(k1MB);
    REQUIRE_FALSE(original.IsNull());

    char *raw_ptr = original.ptr_;
    hipc::FullPtr<char> converted = ipc_manager->ToFullPtr(raw_ptr);

    REQUIRE_FALSE(converted.IsNull());
    REQUIRE(converted.ptr_ == raw_ptr);

    original.ptr_[0] = 'X';
    REQUIRE(converted.ptr_[0] == 'X');

    converted.ptr_[100] = 'Y';
    REQUIRE(original.ptr_[100] == 'Y');
  }
}

TEST_CASE("Per-process shared memory stress test",
          "[ipc][per_process_shm][stress]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Many small allocations") {
    const size_t num_allocs = 100;
    const size_t alloc_size = 10 * k1MB;
    std::vector<hipc::FullPtr<char>> buffers;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocs; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(alloc_size);
      REQUIRE_FALSE(buffer.IsNull());
      buffer.ptr_[0] = static_cast<char>(i & 0xFF);
      buffer.ptr_[alloc_size - 1] = static_cast<char>((i + 1) & 0xFF);
      buffers.push_back(buffer);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    INFO("Allocated " << num_allocs << " buffers in " << duration.count()
                      << " ms");

    for (size_t i = 0; i < num_allocs; ++i) {
      REQUIRE(static_cast<unsigned char>(buffers[i].ptr_[0]) == (i & 0xFF));
      REQUIRE(static_cast<unsigned char>(buffers[i].ptr_[alloc_size - 1]) ==
              ((i + 1) & 0xFF));
    }
  }
}

TEST_CASE("Per-process shared memory ClientShmInfo",
          "[ipc][per_process_shm][shm_info]") {
  REQUIRE(initialize_chimaera());

  auto *ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("ClientShmInfo struct creation") {
    chi::ClientShmInfo info;
    info.shm_name = "test_shm";
    info.owner_pid = hshm::SystemInfo::GetPid();
    info.shm_index = 0;
    info.size = k100MB;
    info.alloc_id = hipc::AllocatorId(
        static_cast<chi::u32>(hshm::SystemInfo::GetPid()), 0);

    REQUIRE(info.shm_name == "test_shm");
    REQUIRE(info.owner_pid == hshm::SystemInfo::GetPid());
    REQUIRE(info.shm_index == 0);
    REQUIRE(info.size == k100MB);
  }
}

int main(int argc, char *argv[]) {
  // Server mode
  if (argc > 1 && std::string(argv[1]) == "--server-mode") {
    hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "1", 1);
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer, true);
    if (!success) return 1;
    std::this_thread::sleep_for(std::chrono::minutes(5));
    return 0;
  }

  // Client SHM info test mode
  if (argc > 1 && std::string(argv[1]) == "--client-shm-info-mode") {
    hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "0", 1);
    hshm::SystemInfo::Setenv("CHI_IPC_MODE", "SHM", 1);
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) return 1;

    auto *client_ipc = CHI_IPC;
    if (!client_ipc) return 2;

    auto buffer = client_ipc->AllocateBuffer(1024 * 1024);
    if (buffer.IsNull()) return 3;

    chi::ClientShmInfo info = client_ipc->GetClientShmInfo(0);
    if (info.owner_pid != hshm::SystemInfo::GetPid()) return 4;
    if (info.shm_index != 0) return 5;
    if (info.size == 0) return 6;

    std::string expected_prefix =
        "chimaera_" + std::to_string(hshm::SystemInfo::GetPid()) + "_";
    if (info.shm_name.find(expected_prefix) != 0) return 7;

    return 0;
  }

  // Normal test mode
  hshm::SystemInfo::SuppressErrorDialogs();
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  int rc = SimpleTest::run_all_tests(filter);
  chi::CHIMAERA_FINALIZE();
  SIMPLE_TEST_HARD_EXIT(rc);
  return rc;
}
