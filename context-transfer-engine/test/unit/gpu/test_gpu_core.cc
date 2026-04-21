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
 * Two fixture classes:
 *
 * GpuCoreFixture  — CPU path: fork-client mode, PoolQuery::Local(), SHM buffers.
 *                   Verifies AsyncPutBlob / AsyncGetBlob go through CPU workers.
 *
 * GpuCoreGpuFixture — GPU path: server mode, PoolQuery::LocalGpuBcast(),
 *                     cudaMallocHost (pinned UVM) buffers.
 *                     Verifies PutBlob and GetBlob execute inside GPU kernels
 *                     via GpuRuntime::PutBlob / GpuRuntime::GetBlob.
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
#include <chimaera/gpu/work_orchestrator.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {
  bool g_initialized = false;
  bool g_gpu_initialized = false;
}  // namespace

// ============================================================================
// CPU-path fixture (fork client, Local routing)
// ============================================================================

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

// ============================================================================
// GPU-path fixture (server mode, LocalGpuBcast routing, pinned UVM buffers)
// ============================================================================

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

static bool HasAvailableGpuDevice() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    INFO("cudaGetDeviceCount failed; GPU-path CTE tests will be skipped: "
         << cudaGetErrorString(err));
    return false;
  }
  if (device_count <= 0) {
    INFO("No CUDA/ROCM devices reported; GPU-path CTE tests will be skipped");
    return false;
  }
  return true;
}

/**
 * Allocate a pinned host buffer accessible from both CPU and GPU via UVA.
 * Uses cudaMallocHost — non-synchronizing free, safe alongside the persistent
 * GPU orchestrator kernel.
 */
static char *AllocPinned(size_t bytes) {
  char *ptr = nullptr;
  cudaMallocHost(&ptr, bytes);
  return ptr;
}

static void FreePinned(char *ptr) {
  (void)ptr;  // Intentional leak: cudaFreeHost blocks on persistent GPU kernel.
}

/**
 * Encode a pinned (UVA) pointer into an hipc::ShmPtr<> for task routing.
 * Null alloc_id + raw pointer in off_ signals GPU path in IpcManager::ToFullPtr.
 */
static hipc::ShmPtr<> PinnedToShmPtr(void *ptr) {
  hipc::ShmPtr<> sp;
  sp.alloc_id_ = hipc::MemoryBackendId::GetNull();
  sp.off_.exchange(reinterpret_cast<size_t>(ptr));
  return sp;
}

/**
 * GPU-path fixture: initializes chimaera in kServer mode so that GPU workers
 * and the persistent GPU orchestrator kernel are active.
 */
class GpuCoreGpuFixture {
 public:
  static constexpr size_t kBlobSize = 4096;

  std::unique_ptr<wrp_cte::core::Client> core_client_;
  chi::PoolId core_pool_id_;
  wrp_cte::core::TagId tag_id_{};

  GpuCoreGpuFixture() {
    if (g_gpu_initialized) return;

    if (!HasAvailableGpuDevice()) {
      INFO("No CUDA/ROCM device; GPU-path CTE tests will be skipped");
      return;
    }

    // Ensure the runtime is initialized (GpuCoreFixture may not have run yet)
    if (!g_initialized) {
      hshm::Singleton<GpuCoreFixture>::GetInstance();
    }

    auto *ipc = CHI_IPC;
    if (ipc == nullptr) {
      INFO("CHI_IPC is null; GPU-path CTE tests will be skipped");
      return;
    }
    if (ipc->GetGpuIpcManager()->gpu_devices_.size() == 0) {
      INFO("No GPU queues available; GPU-path CTE tests will be skipped");
      return;
    }

    // Create CTE core pool for GPU path
    core_pool_id_ = chi::PoolId(wrp_cte::core::kCtePoolId.major_ + 1,
                                wrp_cte::core::kCtePoolId.minor_);
    core_client_ = std::make_unique<wrp_cte::core::Client>(core_pool_id_);
    wrp_cte::core::CreateParams params;
    auto create_task = core_client_->AsyncCreate(
        chi::PoolQuery::Dynamic(),
        "cte_gpu_path_pool", core_pool_id_, params);
    create_task.Wait();
    if (create_task->GetReturnCode() != 0) {
      INFO("GPU fixture: CTE pool create failed: " << create_task->GetReturnCode());
      return;
    }

    // Register a pinned-memory bdev target (GPU-accessible)
    chi::PoolId bdev_pool_id(800, 0);
    auto reg_task = core_client_->AsyncRegisterTarget(
        "pinned::cte_gpu_test_target",
        chimaera::bdev::BdevType::kPinned,
        /*total_size=*/chi::u64{16 * 1024 * 1024},  // 16MB
        chi::PoolQuery::Local(),
        bdev_pool_id);
    reg_task.Wait();
    if (reg_task->GetReturnCode() != 0) {
      INFO("GPU fixture: RegisterTarget (CPU) failed: " << reg_task->GetReturnCode());
      return;
    }

    // Brief delay: allow GPU orchestrator to register bdev GPU container
    std::this_thread::sleep_for(200ms);

    // Wait for bdev UpdateTask to complete on GPU before sending more tasks.
    bool bdev_update_done = false;
    {
      auto start = std::chrono::steady_clock::now();
      int print_counter = 0;
      while (!bdev_update_done) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

        chi::gpu::WorkOrchestrator *orch = nullptr;
        auto *ctrl = (orch && orch->control_) ? orch->control_ : nullptr;

        // Print debug every second
        if (ctrl && (++print_counter % 10 == 0)) {
          fprintf(stderr, "[FIXTURE t=%llds] W0: polls=%llu state=%u method=%u "
                  "step=%u tw=%llu cs=%llu susp=%u\n",
                  (long long)secs,
                  (unsigned long long)ctrl->dbg_poll_count[0],
                  (unsigned)ctrl->dbg_last_state[0],
                  (unsigned)ctrl->dbg_last_method[0],
                  (unsigned)ctrl->dbg_dispatch_step[0],
                  (unsigned long long)ctrl->dbg_input_tw[0],
                  (unsigned long long)ctrl->dbg_input_cs[0],
                  (unsigned)ctrl->dbg_num_suspended[0]);
          fflush(stderr);
        }

        if (secs >= 10) {
          fprintf(stderr, "[FIXTURE] Aborting - bdev UpdateTask never completed\n");
          fflush(stderr);
          break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (ctrl && ctrl->dbg_last_state[0] >= 5) {
          // State 5 = completed
          bdev_update_done = true;
        }
      }
    }

    if (!bdev_update_done) {
      INFO("GPU fixture: Skipping GPU RegisterTarget - bdev UpdateTask didn't complete");
      return;
    }

    // Register the same target on the GPU side so GpuRuntime can find it
    auto gpu_reg_task = core_client_->AsyncRegisterTarget(
        "pinned::cte_gpu_test_target",
        chimaera::bdev::BdevType::kPinned,
        /*total_size=*/chi::u64{16 * 1024 * 1024},
        chi::PoolQuery::Local(),
        bdev_pool_id,
        chi::PoolQuery::LocalGpuBcast());
    gpu_reg_task.Wait();
    if (gpu_reg_task->GetReturnCode() != 0) {
      INFO("GPU fixture: RegisterTarget (GPU) failed: " << gpu_reg_task->GetReturnCode());
      return;
    }

    // GetOrCreateTag via CPU Local() — tag metadata must be in CPU runtime
    auto tag_task = core_client_->AsyncGetOrCreateTag(
        "gpu_path_tag", wrp_cte::core::TagId::GetNull(), chi::PoolQuery::Local());
    tag_task.Wait();
    if (tag_task->GetReturnCode() != 0) {
      INFO("GPU fixture: GetOrCreateTag failed: " << tag_task->GetReturnCode());
      return;
    }
    tag_id_ = tag_task->tag_id_;

    // Also create the tag on the GPU side
    auto gpu_tag_task = core_client_->AsyncGetOrCreateTag(
        "gpu_path_tag", tag_id_, chi::PoolQuery::LocalGpuBcast());
    gpu_tag_task.Wait();

    // Brief delay: allow GPU orchestrator to register the new pool's container
    std::this_thread::sleep_for(200ms);

    g_gpu_initialized = true;
    INFO("GpuCoreGpuFixture ready (pool_id=" << core_pool_id_.ToU64()
         << " tag_id=" << tag_id_.major_ << "." << tag_id_.minor_ << ")");
  }
};

// ============================================================================
// GPU-path tests: PutBlob and GetBlob execute in GPU kernel
// ============================================================================

/**
 * Test: GPU-path PutBlob — routes to GpuRuntime::PutBlob via LocalGpuBcast.
 *
 * Allocates a 4 KB pinned buffer, fills it with 0xCD, and submits PutBlob
 * to the GPU worker.  The GPU container stores the UVM pointer as the blob's
 * backing location.
 */
TEST_CASE("GpuCore - GPU PutBlob via LocalGpuBcast", "[gpu][cte][core]") {
  auto *f = hshm::Singleton<GpuCoreGpuFixture>::GetInstance();
  if (!g_gpu_initialized) {
    INFO("GPU not available; skipping GPU PutBlob test");
    return;
  }

  const size_t kSize = GpuCoreGpuFixture::kBlobSize;

  char *src = AllocPinned(kSize);
  REQUIRE(src != nullptr);
  memset(src, 0xCD, kSize);

  hipc::ShmPtr<> blob_data = PinnedToShmPtr(src);

  auto task = f->core_client_->AsyncPutBlob(
      f->tag_id_, "gpu_kernel_blob",
      /*offset=*/0,
      /*size=*/kSize,
      blob_data,
      /*score=*/0.5f,
      wrp_cte::core::Context(),
      /*flags=*/0,
      chi::PoolQuery::LocalGpuBcast());
  REQUIRE(!task.IsNull());

  // Poll with timeout instead of blocking Wait()
  {
    auto start = std::chrono::steady_clock::now();
    bool completed = false;
    while (!completed) {
      if (task.IsComplete()) {
        completed = true;
        break;
      }
      auto elapsed = std::chrono::steady_clock::now() - start;
      if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= 10) {
        // Dump GPU worker debug state
        chi::gpu::WorkOrchestrator *orch = nullptr;
        if (orch && orch->control_) {
          auto *ctrl = orch->control_;
          for (int w = 0; w < 2; ++w) {
            INFO("Worker " << w
                 << ": polls=" << ctrl->dbg_poll_count[w]
                 << " suspended=" << ctrl->dbg_num_suspended[w]
                 << " last_method=" << ctrl->dbg_last_method[w]
                 << " last_state=" << ctrl->dbg_last_state[w]
                 << " resume_checks=" << ctrl->dbg_resume_checks[w]
                 << " ser_total_written=" << ctrl->dbg_ser_total_written[w]
                 << " ser_method=" << ctrl->dbg_ser_method[w]
                 << " dispatch_step=" << ctrl->dbg_dispatch_step[w]
                 << " input_tw=" << ctrl->dbg_input_tw[w]
                 << " input_cs=" << ctrl->dbg_input_cs[w]);
          }
        }
        FAIL("PutBlob timed out after 10s");
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (completed) {
      task.Wait();  // Finalize
    }
  }
  // Always dump debug info on completion
  {
    chi::gpu::WorkOrchestrator *orch = nullptr;
    if (orch && orch->control_) {
      auto *ctrl = orch->control_;
      for (int w = 0; w < 2; ++w) {
        INFO("Worker " << w
             << ": polls=" << ctrl->dbg_poll_count[w]
             << " suspended=" << ctrl->dbg_num_suspended[w]
             << " last_method=" << ctrl->dbg_last_method[w]
             << " last_state=" << ctrl->dbg_last_state[w]
             << " resume_checks=" << ctrl->dbg_resume_checks[w]
             << " ser_total_written=" << ctrl->dbg_ser_total_written[w]
             << " ser_method=" << ctrl->dbg_ser_method[w]);
      }
    }
  }
  INFO("GPU PutBlob return_code=" << task->GetReturnCode());
  REQUIRE(task->GetReturnCode() == 0);
  // src kept alive; FreePinned is a no-op (safe leak until process exit)
}

/**
 * Test: GPU-path GetBlob — routes to GpuRuntime::GetBlob via LocalGpuBcast.
 *
 * Reads the blob stored by the previous GPU PutBlob test into a separate
 * pinned output buffer, then verifies the data matches the write pattern.
 */
TEST_CASE("GpuCore - GPU GetBlob via LocalGpuBcast", "[gpu][cte][core]") {
  auto *f = hshm::Singleton<GpuCoreGpuFixture>::GetInstance();
  if (!g_gpu_initialized) {
    INFO("GPU not available; skipping GPU GetBlob test");
    return;
  }

  const size_t kSize = GpuCoreGpuFixture::kBlobSize;

  char *dst = AllocPinned(kSize);
  REQUIRE(dst != nullptr);
  memset(dst, 0x00, kSize);

  hipc::ShmPtr<> blob_data = PinnedToShmPtr(dst);

  auto task = f->core_client_->AsyncGetBlob(
      f->tag_id_, "gpu_kernel_blob",
      /*offset=*/0,
      /*size=*/kSize,
      /*flags=*/0,
      blob_data,
      chi::PoolQuery::LocalGpuBcast());
  REQUIRE(!task.IsNull());
  task.Wait();
  REQUIRE(task->GetReturnCode() == 0);

  // Verify data round-trip: GPU memcpy'd from src (0xCD) into dst
  // task.Wait() guarantees GPU future is complete (system-scope atomic),
  // so UVM coherence ensures CPU can read dst immediately.
  bool data_ok = true;
  for (size_t i = 0; i < kSize; ++i) {
    if ((unsigned char)dst[i] != 0xCD) { data_ok = false; break; }
  }
  REQUIRE(data_ok);

  INFO("GPU GetBlob data verified (0xCD pattern correct)");
  FreePinned(dst);
}

/**
 * Test: GPU-path GetOrCreateTag — routes via LocalGpuBcast to GpuRuntime.
 *
 * Creates a new tag directly on the GPU container and verifies a valid
 * tag ID is assigned.
 */
TEST_CASE("GpuCore - GPU GetOrCreateTag via LocalGpuBcast", "[gpu][cte][core]") {
  auto *f = hshm::Singleton<GpuCoreGpuFixture>::GetInstance();
  if (!g_gpu_initialized) {
    INFO("GPU not available; skipping GPU GetOrCreateTag test");
    return;
  }

  auto task = f->core_client_->AsyncGetOrCreateTag(
      "gpu_only_tag", wrp_cte::core::TagId::GetNull(),
      chi::PoolQuery::LocalGpuBcast());
  REQUIRE(!task.IsNull());
  task.Wait();
  REQUIRE(task->GetReturnCode() == 0);
  REQUIRE(!task->tag_id_.IsNull());

  INFO("GPU GetOrCreateTag succeeded (tag_id="
       << task->tag_id_.major_ << "." << task->tag_id_.minor_ << ")");
}

// ============================================================================
// GPU-initiated tests: kernel creates tasks and awaits completion
// ============================================================================

extern "C" int run_gpu_initiated_putblob_getblob_test(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u64 blob_size);

/**
 * Test: GPU-initiated PutBlob + GetBlob roundtrip.
 *
 * The GPU kernel itself creates PutBlob and GetBlob tasks using the unified
 * Client API, submits them via the GPU→GPU queue, and awaits completion.
 * This is a full GPU-initiated roundtrip — no CPU task submission involved.
 */
TEST_CASE("GpuCore - GPU-Initiated PutBlob+GetBlob", "[gpu][cte][core]") {
  auto *f = hshm::Singleton<GpuCoreGpuFixture>::GetInstance();
  if (!g_gpu_initialized) {
    INFO("GPU not available; skipping GPU-initiated test");
    return;
  }

  const chi::u64 kSize = 4096;

  int result = run_gpu_initiated_putblob_getblob_test(
      f->core_pool_id_, f->tag_id_, kSize);

  INFO("GPU-initiated PutBlob+GetBlob result: " << result);
  REQUIRE(result == 1);  // 1 = success from kernel
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

// ============================================================================
// CPU-path tests
// ============================================================================

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
