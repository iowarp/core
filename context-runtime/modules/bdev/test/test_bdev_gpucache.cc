/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * GPU bdev integration tests.
 *
 * Tests kHbm (cudaMalloc device memory) and kPinned (cudaMallocHost) bdev
 * types, exercising:
 *   Create → AllocateBlocks → Write → Read → FreeBlocks → data correctness
 *
 * Compiled as CPU code (no CUDA kernel launches needed): the chimaera task
 * infrastructure routes Write/Read to the GPU GpuRuntime container when using
 * LocalGpuBcast, and to the CPU Runtime when using Local routing.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "../../../test/simple_test.h"

#include <chrono>
#include <cstring>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

#include <hermes_shm/util/gpu_api.h>
#include <cuda_runtime.h>

using namespace std::chrono_literals;

// ============================================================================
// Shared fixture state
// ============================================================================

namespace {
bool g_initialized = false;
int  g_pool_counter = 0;
}

void EnsureServerInitialized() {
  if (g_initialized) return;
  bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer);
  REQUIRE(ok);
  SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
  std::this_thread::sleep_for(500ms);
  g_initialized = true;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Allocate a pinned host buffer that both CPU and GPU can access via UVA.
 * Uses cudaMallocHost (not cudaMallocManaged) so that cudaFreeHost is
 * non-synchronizing — safe to call while the persistent GPU orchestrator
 * kernel is still running (cudaFree on managed memory would block forever).
 */
static char *AllocUvm(size_t bytes) {
  char *ptr = nullptr;
  cudaMallocHost(&ptr, bytes);
  return ptr;
}

static void FreeUvm(char *ptr) {
  // cudaFreeHost can block if a persistent GPU kernel holds the device context.
  // Use cudaFreeAsync on the null stream to defer the free until the stream
  // catches up — or just skip the free (small test buffer, freed at exit).
  (void)ptr;  // leak intentionally; freed at process exit
}

/**
 * Encode a UVA (UVM) pointer into an hipc::ShmPtr<> for passing to bdev tasks.
 * The GPU container reads raw pointer from off_ when alloc_id is null.
 */
static hipc::ShmPtr<> UvmToShmPtr(void *ptr) {
  hipc::ShmPtr<> sp;
  sp.alloc_id_ = hipc::MemoryBackendId::GetNull();
  sp.off_.exchange(reinterpret_cast<size_t>(ptr));
  return sp;
}

// ============================================================================
// Test: kRam bdev (baseline — no CUDA required)
// ============================================================================

TEST_CASE("bdev_gpu_ram_baseline", "[gpu_bdev]") {
  EnsureServerInitialized();

  ++g_pool_counter;
  chi::PoolId pool_id(20000, g_pool_counter);
  chimaera::bdev::Client client(pool_id);

  constexpr size_t kBufSize = 64 * 1024;  // 64 KB

  auto create_f = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_test_ram_bdev",
      pool_id, chimaera::bdev::BdevType::kRam, kBufSize);
  create_f.Wait();
  REQUIRE(create_f->return_code_ == 0);

  // Allocate 4 KB of blocks
  auto alloc_f = client.AsyncAllocateBlocks(chi::PoolQuery::Local(), 4096);
  alloc_f.Wait();
  REQUIRE(alloc_f->return_code_ == 0);
  REQUIRE(alloc_f->blocks_.size() > 0);
  chi::priv::vector<chimaera::bdev::Block> blocks = alloc_f->blocks_;

  auto *ipc = CHI_IPC;
  auto src_shm = ipc->AllocateBuffer(4096);
  auto dst_shm = ipc->AllocateBuffer(4096);
  memset(src_shm.ptr_, 0x42, 4096);
  memset(dst_shm.ptr_, 0x00, 4096);

  auto write_f = client.AsyncWrite(chi::PoolQuery::Local(), blocks,
                                    src_shm.shm_.template Cast<void>().template Cast<void>(),
                                    4096);
  write_f.Wait();
  REQUIRE(write_f->return_code_ == 0);
  REQUIRE(write_f->bytes_written_ == 4096);

  auto read_f = client.AsyncRead(chi::PoolQuery::Local(), blocks,
                                   dst_shm.shm_.template Cast<void>().template Cast<void>(),
                                   4096);
  read_f.Wait();
  REQUIRE(read_f->return_code_ == 0);
  REQUIRE(read_f->bytes_read_ == 4096);
  REQUIRE(memcmp(src_shm.ptr_, dst_shm.ptr_, 4096) == 0);

  ipc->FreeBuffer(src_shm);
  ipc->FreeBuffer(dst_shm);

  auto free_f = client.AsyncFreeBlocks(chi::PoolQuery::Local(),
                                         std::vector<chimaera::bdev::Block>(
                                             blocks.begin(), blocks.end()));
  free_f.Wait();
  INFO("bdev_gpu_ram_baseline passed");
}

// ============================================================================
// Test: kHbm bdev — CPU-side Write/Read via cudaMemcpy
// ============================================================================

#if HSHM_ENABLE_CUDA
TEST_CASE("bdev_gpu_hbm_cpu_write_read", "[gpu_bdev]") {
  EnsureServerInitialized();

  ++g_pool_counter;
  chi::PoolId pool_id(20000, g_pool_counter);
  chimaera::bdev::Client client(pool_id);

  constexpr size_t kBufSize = 256 * 1024;  // 256 KB HBM

  auto create_f = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_test_hbm_bdev",
      pool_id, chimaera::bdev::BdevType::kHbm, kBufSize);
  create_f.Wait();
  REQUIRE(create_f->return_code_ == 0);
  INFO("HBM bdev created (256 KB device memory)");

  // Allocate blocks via CPU runtime
  auto alloc_f = client.AsyncAllocateBlocks(chi::PoolQuery::Local(), 4096);
  alloc_f.Wait();
  REQUIRE(alloc_f->return_code_ == 0);
  REQUIRE(alloc_f->blocks_.size() > 0);
  chi::priv::vector<chimaera::bdev::Block> blocks = alloc_f->blocks_;

  // Source and destination buffers in SHM (CPU-accessible)
  auto *ipc = CHI_IPC;
  auto src_shm = ipc->AllocateBuffer(4096);
  auto dst_shm = ipc->AllocateBuffer(4096);
  memset(src_shm.ptr_, 0x77, 4096);
  memset(dst_shm.ptr_, 0x00, 4096);

  // CPU Write: cudaMemcpy host→device (HSHM_ENABLE_CUDA path in Runtime::Write)
  auto write_f = client.AsyncWrite(chi::PoolQuery::Local(), blocks,
                                    src_shm.shm_.template Cast<void>().template Cast<void>(),
                                    4096);
  write_f.Wait();
  REQUIRE(write_f->return_code_ == 0);
  REQUIRE(write_f->bytes_written_ == 4096);

  // CPU Read: cudaMemcpy device→host
  auto read_f = client.AsyncRead(chi::PoolQuery::Local(), blocks,
                                   dst_shm.shm_.template Cast<void>().template Cast<void>(),
                                   4096);
  read_f.Wait();
  REQUIRE(read_f->return_code_ == 0);
  REQUIRE(read_f->bytes_read_ == 4096);
  REQUIRE(memcmp(src_shm.ptr_, dst_shm.ptr_, 4096) == 0);

  ipc->FreeBuffer(src_shm);
  ipc->FreeBuffer(dst_shm);

  auto free_f = client.AsyncFreeBlocks(chi::PoolQuery::Local(),
                                         std::vector<chimaera::bdev::Block>(
                                             blocks.begin(), blocks.end()));
  free_f.Wait();
  INFO("bdev_gpu_hbm_cpu_write_read passed");
}

// ============================================================================
// Test: kPinned bdev — CPU-side Write/Read via memcpy
// ============================================================================

TEST_CASE("bdev_gpu_pinned_cpu_write_read", "[gpu_bdev]") {
  EnsureServerInitialized();

  ++g_pool_counter;
  chi::PoolId pool_id(20000, g_pool_counter);
  chimaera::bdev::Client client(pool_id);

  constexpr size_t kBufSize = 256 * 1024;  // 256 KB pinned

  auto create_f = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_test_pinned_bdev",
      pool_id, chimaera::bdev::BdevType::kPinned, kBufSize);
  create_f.Wait();
  REQUIRE(create_f->return_code_ == 0);
  INFO("Pinned bdev created (256 KB pinned host memory)");

  auto alloc_f = client.AsyncAllocateBlocks(chi::PoolQuery::Local(), 4096);
  alloc_f.Wait();
  REQUIRE(alloc_f->return_code_ == 0);
  REQUIRE(alloc_f->blocks_.size() > 0);
  chi::priv::vector<chimaera::bdev::Block> blocks = alloc_f->blocks_;

  auto *ipc = CHI_IPC;
  auto src_shm = ipc->AllocateBuffer(4096);
  auto dst_shm = ipc->AllocateBuffer(4096);
  memset(src_shm.ptr_, 0x55, 4096);
  memset(dst_shm.ptr_, 0x00, 4096);

  auto write_f = client.AsyncWrite(chi::PoolQuery::Local(), blocks,
                                    src_shm.shm_.template Cast<void>().template Cast<void>(),
                                    4096);
  write_f.Wait();
  REQUIRE(write_f->return_code_ == 0);
  REQUIRE(write_f->bytes_written_ == 4096);

  auto read_f = client.AsyncRead(chi::PoolQuery::Local(), blocks,
                                   dst_shm.shm_.template Cast<void>().template Cast<void>(),
                                   4096);
  read_f.Wait();
  REQUIRE(read_f->return_code_ == 0);
  REQUIRE(read_f->bytes_read_ == 4096);
  REQUIRE(memcmp(src_shm.ptr_, dst_shm.ptr_, 4096) == 0);

  ipc->FreeBuffer(src_shm);
  ipc->FreeBuffer(dst_shm);

  auto free_f = client.AsyncFreeBlocks(chi::PoolQuery::Local(),
                                         std::vector<chimaera::bdev::Block>(
                                             blocks.begin(), blocks.end()));
  free_f.Wait();
  INFO("bdev_gpu_pinned_cpu_write_read passed");
}

// ============================================================================
// Test: kHbm bdev — CPU→GPU: AllocateBlocks + Write + Read via LocalGpuBcast
// ============================================================================

TEST_CASE("bdev_gpu_hbm_cpu2gpu_write_read", "[gpu_bdev]") {
  EnsureServerInitialized();

  auto *ipc = CHI_IPC;
  if (ipc->GetGpuIpcManager()->gpu_devices_.size() == 0) {
    INFO("No GPU queues available, skipping GPU-path bdev test");
    return;
  }

  ++g_pool_counter;
  chi::PoolId pool_id(20000, g_pool_counter);
  chimaera::bdev::Client client(pool_id);

  constexpr size_t kBufSize = 256 * 1024;
  constexpr size_t kDataSize = 4096;

  // Create HBM bdev (CPU allocates cudaMalloc, fires UpdateTask to GPU container)
  auto create_f = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "gpu_test_hbm_gpu_bdev",
      pool_id, chimaera::bdev::BdevType::kHbm, kBufSize);
  create_f.Wait();
  REQUIRE(create_f->return_code_ == 0);

  // Brief delay to ensure UpdateTask reaches and is processed by GPU worker
  std::this_thread::sleep_for(200ms);

  // GPU-side AllocateBlocks (no pause needed — IpcCpu2Gpu uses pinned host)
  auto alloc_f = client.AsyncAllocateBlocks(chi::PoolQuery::LocalGpuBcast(), kDataSize);
  alloc_f.Wait();
  REQUIRE(alloc_f->return_code_ == 0);
  REQUIRE(alloc_f->blocks_.size() > 0);
  chi::priv::vector<chimaera::bdev::Block> gpu_blocks = alloc_f->blocks_;
  INFO("GPU AllocateBlocks succeeded: " + std::to_string(gpu_blocks.size()) + " block(s)");

  // Pause orchestrator for UVM allocation (cudaMallocManaged is device-sync)
  ipc->GetGpuIpcManager()->PauseGpuOrchestrator();
  char *src_uvm = AllocUvm(kDataSize);
  char *dst_uvm = AllocUvm(kDataSize);
  REQUIRE(src_uvm != nullptr);
  REQUIRE(dst_uvm != nullptr);
  memset(src_uvm, 0xAB, kDataSize);
  memset(dst_uvm, 0x00, kDataSize);
  ipc->GetGpuIpcManager()->ResumeGpuOrchestrator();

  // GPU-side Write (no pause — IpcCpu2Gpu uses pinned host, no CUDA calls)
  hipc::ShmPtr<> write_data = UvmToShmPtr(src_uvm);
  auto write_f = client.AsyncWrite(chi::PoolQuery::LocalGpuBcast(),
                                    gpu_blocks, write_data, kDataSize);
  write_f.Wait();
  REQUIRE(write_f->return_code_ == 0);
  REQUIRE(write_f->bytes_written_ == kDataSize);
  INFO("GPU Write succeeded");

  // GPU-side Read (no pause)
  hipc::ShmPtr<> read_data = UvmToShmPtr(dst_uvm);
  auto read_f = client.AsyncRead(chi::PoolQuery::LocalGpuBcast(),
                                   gpu_blocks, read_data, kDataSize);
  read_f.Wait();
  REQUIRE(read_f->return_code_ == 0);
  REQUIRE(read_f->bytes_read_ == kDataSize);
  INFO("GPU Read succeeded");

  // Verify data round-trip (UVM is coherent; CPU can read directly).
  // read_f.Wait() already ensured GPU writes are system-scope visible via
  // FUTURE_COMPLETE system atomic — no cudaDeviceSynchronize needed here
  // (it would block forever on the persistent GPU orchestrator kernel).
  REQUIRE(memcmp(src_uvm, dst_uvm, kDataSize) == 0);
  INFO("GPU Write→Read data verified correctly");

  FreeUvm(src_uvm);
  FreeUvm(dst_uvm);

  // GPU-side FreeBlocks (no-op but completes cleanly)
  auto free_f = client.AsyncFreeBlocks(
      chi::PoolQuery::LocalGpuBcast(),
      std::vector<chimaera::bdev::Block>(gpu_blocks.begin(), gpu_blocks.end()));
  free_f.Wait();
  REQUIRE(free_f->return_code_ == 0);
  INFO("bdev_gpu_hbm_gpu_write_read passed");
}

#endif  // HSHM_ENABLE_CUDA

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char *argv[]) {
  std::string filter = (argc > 1) ? argv[1] : "";
  return SimpleTest::run_all_tests(filter);
}

#else  // !HSHM_ENABLE_CUDA && !HSHM_ENABLE_ROCM

int main() {
  // GPU support not compiled; nothing to test.
  return 0;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
