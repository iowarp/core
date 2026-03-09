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
 * GPU CTE Core Client API Tests
 *
 * Verifies that AsyncPutBlob and AsyncGetBlob with PoolQuery::Local() work
 * end-to-end with the CTE core runtime. Tests use a fork client (background
 * runtime spawned by CHIMAERA_INIT with fork=true).
 *
 * The 4KB blob is allocated in shared memory via CHI_IPC->AllocateBuffer,
 * routed through PoolQuery::Local() to CPU workers, and verified by checking
 * the task return code.
 */

#include "simple_test.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <thread>

#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {
  bool g_initialized = false;
}  // namespace

/**
 * Test fixture: initializes chimaera as a fork client (background runtime)
 * and creates a CTE core pool with a file-based storage target.
 */
class GpuCoreFixture {
 public:
  static constexpr size_t kBlobSize = 4096;  // 4KB test blob

  std::unique_ptr<wrp_cte::core::Client> core_client_;
  chi::PoolId core_pool_id_;
  wrp_cte::core::TagId tag_id_{};
  std::string test_storage_path_;

  GpuCoreFixture() {
    if (g_initialized) return;

    test_storage_path_ = "/tmp/cte_gpu_core_test.dat";
    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
    }

    INFO("Initializing Chimaera as fork client for GPU core test...");
    bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!ok) {
      INFO("CHIMAERA_INIT failed");
      return;
    }
    std::this_thread::sleep_for(500ms);

    // Create CTE core pool using canonical pool ID and name
    core_pool_id_ = wrp_cte::core::kCtePoolId;
    core_client_ = std::make_unique<wrp_cte::core::Client>(core_pool_id_);
    wrp_cte::core::CreateParams params;
    auto create_task = core_client_->AsyncCreate(
        chi::PoolQuery::Dynamic(),
        wrp_cte::core::kCtePoolName, core_pool_id_, params);
    create_task.Wait();
    if (create_task->GetReturnCode() != 0) {
      INFO("Failed to create CTE core pool: " << create_task->GetReturnCode());
      return;
    }

    // Register a file-based bdev target for storage
    auto reg_task = core_client_->AsyncRegisterTarget(
        test_storage_path_,
        chimaera::bdev::BdevType::kFile,
        /*total_size=*/chi::u64{1024 * 1024 * 16},  // 16MB
        chi::PoolQuery::Local(),
        chi::PoolId(700, 0));
    reg_task.Wait();
    if (reg_task->GetReturnCode() != 0) {
      INFO("Failed to register target: " << reg_task->GetReturnCode());
      return;
    }

    // Create a tag for blob operations
    auto tag_task = core_client_->AsyncGetOrCreateTag("gpu_test_tag");
    tag_task.Wait();
    tag_id_ = tag_task->tag_id_;

    g_initialized = true;
    INFO("GpuCoreFixture ready (pool_id=" << core_pool_id_.ToU64() << ")");
  }
};

/**
 * Test: AsyncPutBlob with PoolQuery::Local() on a 4KB blob succeeds.
 */
TEST_CASE("GpuCore - AsyncPutBlob Local 4KB", "[gpu][cte][core]") {
  auto *f = hshm::Singleton<GpuCoreFixture>::GetInstance();
  REQUIRE(g_initialized);

  const size_t blob_size = GpuCoreFixture::kBlobSize;

  // Allocate 4KB in shared memory and fill with a test pattern
  hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(blob_size);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0xAB, blob_size);
  hipc::ShmPtr<> blob_data = buf.shm_.template Cast<void>();

  auto task = f->core_client_->AsyncPutBlob(
      f->tag_id_, "gpu_blob_4kb",
      /*offset=*/0,
      /*size=*/blob_size,
      blob_data,
      /*score=*/-1.0f,
      wrp_cte::core::Context(),
      /*flags=*/0,
      chi::PoolQuery::Local());
  REQUIRE(!task.IsNull());
  task.Wait();
  REQUIRE(task->GetReturnCode() == 0);

  INFO("AsyncPutBlob Local succeeded (return_code=0)");
}

/**
 * Test: AsyncGetBlob with PoolQuery::Local() retrieves the previously stored
 * 4KB blob successfully.
 */
TEST_CASE("GpuCore - AsyncGetBlob Local 4KB", "[gpu][cte][core]") {
  auto *f = hshm::Singleton<GpuCoreFixture>::GetInstance();
  REQUIRE(g_initialized);

  const size_t blob_size = GpuCoreFixture::kBlobSize;

  // Allocate output buffer
  hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(blob_size);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0x00, blob_size);
  hipc::ShmPtr<> blob_data = buf.shm_.template Cast<void>();

  auto task = f->core_client_->AsyncGetBlob(
      f->tag_id_, "gpu_blob_4kb",
      /*offset=*/0,
      /*size=*/blob_size,
      /*flags=*/0,
      blob_data,
      chi::PoolQuery::Local());
  REQUIRE(!task.IsNull());
  task.Wait();
  REQUIRE(task->GetReturnCode() == 0);

  INFO("AsyncGetBlob Local succeeded (return_code=0)");
}

SIMPLE_TEST_MAIN()
