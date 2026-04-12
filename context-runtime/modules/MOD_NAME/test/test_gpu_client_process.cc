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
 * Separate-Process Client GPU Submit Tests
 *
 * Fork-based test where child process runs as a standalone server and
 * the parent process connects as a pure client to submit GpuSubmitTask
 * via SHM transport with PoolQuery::Local() routing.
 *
 * Pattern borrowed from test_client_retry.cc.
 */

#include "simple_test.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/admin/admin_tasks.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/util/gpu_api.h"
#endif

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

using namespace chi;
using namespace std::chrono_literals;

// ============================================================================
// Helper functions (same pattern as test_client_retry.cc)
// ============================================================================

static pid_t g_server_pid = -1;
static bool g_initialized = false;
static int g_test_counter = 0;

void CleanupSharedMemory() {
  const char *user = std::getenv("USER");
  std::string memfd_path =
      std::string("/tmp/chimaera_") + (user ? user : "unknown") +
      "/chi_main_segment_" + (user ? user : "");
  unlink(memfd_path.c_str());
}

pid_t StartServerProcess() {
  pid_t server_pid = fork();
  if (server_pid == 0) {
    setpgid(0, 0);

    (void)freopen("/dev/null", "w", stdout);
    (void)freopen("/tmp/chimaera_server_gpu_client_test.log", "w", stderr);

    setenv("CHI_WITH_RUNTIME", "1", 1);
    execl("/proc/self/exe", "chimaera_gpu_client_process_tests",
          "--server-mode", nullptr);
    _exit(1);
  }
  setpgid(server_pid, server_pid);
  return server_pid;
}

bool WaitForServer(int max_attempts = 50) {
  const char *user = std::getenv("USER");
  std::string memfd_path =
      std::string("/tmp/chimaera_") + (user ? user : "unknown") +
      "/chi_main_segment_" + (user ? user : "");

  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(200ms);

    int fd = open(memfd_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      close(fd);
      std::this_thread::sleep_for(1000ms);
      return true;
    }
  }
  return false;
}

void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(-server_pid, SIGKILL);
    int status;
    waitpid(server_pid, &status, 0);
    CleanupSharedMemory();
    unlink("/tmp/chimaera_9413.ipc");
    (void)system("rm -f /dev/shm/chimaera_* 2>/dev/null");
  }
}

/** Ensure server + client are initialized (called once, shared across tests) */
void EnsureInitialized() {
  if (g_initialized) return;

  // Start server child process
  g_server_pid = StartServerProcess();
  REQUIRE(g_server_pid > 0);
  bool ready = WaitForServer();
  REQUIRE(ready);

  // Init pure client (no runtime)
  setenv("CHI_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);
  REQUIRE(CHI_IPC != nullptr);
  REQUIRE(CHI_IPC->IsInitialized());

  g_initialized = true;
}

// ============================================================================
// Tests
// ============================================================================

/**
 * Test: Separate client process submits GpuSubmitTask via SHM with Local()
 * routing. The server's CPU worker handles the task.
 *
 * CPU handler: result_value = test_value * 2 + gpu_id
 */
TEST_CASE("client_process_gpu_submit_local", "[gpu][client_process]") {
  EnsureInitialized();

  // Create MOD_NAME pool on the server
  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);
  chimaera::MOD_NAME::Client client(pool_id);
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_client_test", pool_id);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  INFO("Pool created on server");

  std::this_thread::sleep_for(100ms);

  // Submit GpuSubmitTask with Local() routing -> SHM -> server CPU handler
  u32 test_value = 42;
  u32 gpu_id = 0;
  auto future = client.AsyncGpuSubmit(
      chi::PoolQuery::Local(), gpu_id, test_value);
  future.Wait();

  // Verify CPU handler result: test_value * 2 + gpu_id
  REQUIRE(future->GetReturnCode() == 0);
  u32 expected = (test_value * 2) + gpu_id;
  REQUIRE(future->result_value_ == expected);
  INFO("GpuSubmitTask from client process succeeded: result=" +
       std::to_string(future->result_value_));
}

/**
 * Test: Multiple GpuSubmitTask submissions from separate client process
 */
TEST_CASE("client_process_gpu_submit_multiple", "[gpu][client_process]") {
  EnsureInitialized();

  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);
  chimaera::MOD_NAME::Client client(pool_id);
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_client_multi_test", pool_id);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);

  std::this_thread::sleep_for(100ms);

  // Submit multiple tasks with different values
  const int num_tasks = 5;
  for (int i = 0; i < num_tasks; ++i) {
    u32 test_value = 100 + i;
    u32 gpu_id = 0;
    auto future = client.AsyncGpuSubmit(
        chi::PoolQuery::Local(), gpu_id, test_value);
    future.Wait();

    REQUIRE(future->GetReturnCode() == 0);
    u32 expected = (test_value * 2) + gpu_id;
    REQUIRE(future->result_value_ == expected);
  }

  INFO("All " + std::to_string(num_tasks) +
       " GpuSubmitTasks from client process succeeded");
}

/**
 * Test: Verify client GPU queue attachment via ClientConnect.
 * After EnsureInitialized(), the client should have GPU queue info
 * if the server has GPUs.
 */
TEST_CASE("client_process_gpu_queue_attachment", "[gpu][client_process]") {
  EnsureInitialized();

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  auto *ipc = CHI_IPC;
  size_t num_to_gpu = ipc->GetGpuIpcManager()->gpu_devices_.size();
  size_t num_g2g = ipc->GetGpuIpcManager()->gpu_devices_.size();

  INFO("Client GPU queues: cpu2gpu=" + std::to_string(num_to_gpu) +
       ", gpu2gpu=" + std::to_string(num_g2g));

  // If server has GPUs, queues should be attached
  if (num_to_gpu > 0) {
    REQUIRE(ipc->GetGpuIpcManager()->gpu_devices_[0].cpu2gpu_queue.ptr_ != nullptr);
    REQUIRE(ipc->GetGpuIpcManager()->gpu_devices_[0].gpu2gpu_queue.ptr_ != nullptr);
    INFO("Client GPU queue attachment verified");
  } else {
    INFO("No GPUs on server, skipping queue verification");
  }
#else
  INFO("GPU support not compiled, skipping");
#endif
}

/**
 * Test: Client process submits GpuSubmitTask via SendToGpu (LocalGpuBcast routing).
 * The server's GPU orchestrator handles the task.
 *
 * GPU handler: result_value = test_value * 2 + gpu_id
 */
TEST_CASE("client_process_gpu_submit_to_gpu", "[gpu][client_process]") {
  EnsureInitialized();

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  auto *ipc = CHI_IPC;
  if (ipc->GetGpuIpcManager()->gpu_devices_.size() == 0) {
    INFO("No GPU queues available, skipping SendToGpu test");
    return;
  }

  // Create MOD_NAME pool on the server
  g_test_counter++;
  chi::PoolId pool_id(10000, g_test_counter);
  chimaera::MOD_NAME::Client client(pool_id);
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_client_sendtogpu_test", pool_id);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  INFO("Pool created on server");

  std::this_thread::sleep_for(100ms);

  // Submit GpuSubmitTask with LocalGpuBcast routing → SendToGpu path
  u32 test_value = 77;
  u32 gpu_id = 0;
  auto future = client.AsyncGpuSubmit(
      chi::PoolQuery::LocalGpuBcast(), gpu_id, test_value);
  future.Wait();

  // Verify GPU handler result: test_value * 2 + gpu_id
  REQUIRE(future->GetReturnCode() == 0);
  u32 expected = (test_value * 2) + gpu_id;
  REQUIRE(future->result_value_ == expected);
  INFO("GpuSubmitTask via SendToGpu from client process succeeded: result=" +
       std::to_string(future->result_value_));
#else
  INFO("GPU support not compiled, skipping");
#endif
}

/**
 * Test: Client registers a GPU device memory backend with the server
 * via RegisterMemory(kGpuDeviceMemory).
 */
TEST_CASE("client_process_register_gpu_memory", "[gpu][client_process]") {
  EnsureInitialized();

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  auto *ipc = CHI_IPC;

  // Create a GpuMalloc backend in this client process
  hipc::MemoryBackendId backend_id(static_cast<u32>(getpid()), 100);
  size_t data_size = hshm::Unit<size_t>::Megabytes(4);

  hipc::GpuMalloc gpu_backend;
  bool init_ok = gpu_backend.shm_init(backend_id, data_size, "", 0);
  REQUIRE(init_ok);

  // Get IPC handle from the private header
  hipc::GpuMallocPrivateHeader priv_header;
  char *priv_header_gpu = gpu_backend.region_ + gpu_backend.priv_header_off_;
  hshm::GpuApi::Memcpy(&priv_header,
                        reinterpret_cast<hipc::GpuMallocPrivateHeader *>(priv_header_gpu),
                        sizeof(hipc::GpuMallocPrivateHeader));

  // Send RegisterMemory task to server
  auto reg_task = ipc->NewTask<chimaera::admin::RegisterMemoryTask>(
      chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
      backend_id,
      chimaera::admin::MemoryType::kGpuDeviceMemory,
      0,  // gpu_id
      data_size,
      &priv_header.ipc_handle_);
  auto future = ipc->SendZmq(reg_task, chi::IpcMode::kTcp);
  future.Wait();

  REQUIRE(reg_task->success_ == true);
  INFO("GPU device memory registered with server successfully");

  // Keep gpu_backend alive for the duration of the test
  // (it's cleaned up when the unique_ptr goes out of scope)
#else
  INFO("GPU support not compiled, skipping");
#endif
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
  // Server mode: started by StartServerProcess() via exec
  if (argc > 1 && std::string(argv[1]) == "--server-mode") {
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      return 1;
    }
    sleep(300);  // 5 minutes max, parent will SIGKILL us
    return 0;
  }

  // Normal test mode
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  int rc = SimpleTest::run_all_tests(filter);
  CleanupServer(g_server_pid);
  return rc;
}
