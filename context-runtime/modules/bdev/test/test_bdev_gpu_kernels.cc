/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GPU kernel bdev tests — GPU→GPU and GPU→CPU paths.
 *
 * Compiled by nvcc. Tests exercise bdev AllocateBlocks + Write + Read
 * from within GPU kernels using PoolQuery::Local() (GPU→GPU) and
 * PoolQuery::ToLocalCpu() (GPU→CPU).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "../../../test/simple_test.h"

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <hermes_shm/util/gpu_api.h>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

// ============================================================================
// Shared state
// ============================================================================

static bool g_initialized = false;
static int g_pool_counter = 100;  // Offset from CPU-side test counters
static chimaera::bdev::Block g_block;  // Block allocated by CPU for GPU→CPU test

static void EnsureInit() {
  if (g_initialized) return;
  bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer);
  REQUIRE(ok);
  g_initialized = true;
  std::this_thread::sleep_for(500ms);
}

// ============================================================================
// GPU→GPU kernel: AllocateBlocks + Write + Read via PoolQuery::Local()
// ============================================================================

/**
 * GPU kernel that performs bdev allocate + write + read using Local() routing.
 * All operations go through the GPU orchestrator (gpu2gpu queue).
 *
 * d_result: 0=running, 1=success, <0=error
 * d_dbg: [0]=step, [1]=alloc_rc, [2]=write_rc, [3]=read_rc, [4]=verify
 */
__global__ void bdev_gpu2gpu_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u64 data_size,
    char *src_buf,    // UVM: source data for write
    char *dst_buf,    // UVM: destination buffer for read
    int *d_result,
    volatile int *d_dbg) {
  *d_result = 0;
  d_dbg[0] = 1;
  __threadfence_system();
  CHIMAERA_GPU_INIT(gpu_info);

  // Step 1: AllocateBlocks
  d_dbg[0] = 10;
  __threadfence_system();
  auto alloc_task = CHI_IPC->NewTask<chimaera::bdev::AllocateBlocksTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(), data_size);
  if (alloc_task.IsNull()) { *d_result = -1; __threadfence_system(); return; }
  auto alloc_f = CHI_IPC->Send(alloc_task);
  alloc_f.Wait();
  d_dbg[1] = alloc_task->return_code_;
  d_dbg[0] = 11;
  __threadfence_system();
  if (alloc_task->return_code_ != 0 || alloc_task->blocks_.size() == 0) {
    *d_result = -2;
    __threadfence_system();
    return;
  }

  // Get allocated blocks
  chi::priv::vector<chimaera::bdev::Block> blocks = alloc_task->blocks_;

  // Step 2: Write
  d_dbg[0] = 20;
  __threadfence_system();
  hipc::ShmPtr<> write_data;
  write_data.alloc_id_.SetNull();
  write_data.off_ = reinterpret_cast<size_t>(src_buf);
  auto write_task = CHI_IPC->NewTask<chimaera::bdev::WriteTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
      blocks, write_data, data_size);
  if (write_task.IsNull()) { *d_result = -3; __threadfence_system(); return; }
  auto write_f = CHI_IPC->Send(write_task);
  write_f.Wait();
  d_dbg[2] = write_task->return_code_;
  d_dbg[0] = 21;
  __threadfence_system();
  if (write_task->return_code_ != 0) {
    *d_result = -4;
    __threadfence_system();
    return;
  }

  // Step 3: Read
  d_dbg[0] = 30;
  __threadfence_system();
  hipc::ShmPtr<> read_data;
  read_data.alloc_id_.SetNull();
  read_data.off_ = reinterpret_cast<size_t>(dst_buf);
  auto read_task = CHI_IPC->NewTask<chimaera::bdev::ReadTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
      blocks, read_data, data_size);
  if (read_task.IsNull()) { *d_result = -5; __threadfence_system(); return; }
  auto read_f = CHI_IPC->Send(read_task);
  read_f.Wait();
  d_dbg[3] = read_task->return_code_;
  d_dbg[0] = 31;
  __threadfence_system();
  if (read_task->return_code_ != 0) {
    *d_result = -6;
    __threadfence_system();
    return;
  }

  // Step 4: Verify (compare src and dst on GPU)
  d_dbg[0] = 40;
  __threadfence_system();
  bool match = true;
  for (chi::u64 i = 0; i < data_size; ++i) {
    if (src_buf[i] != dst_buf[i]) {
      match = false;
      break;
    }
  }
  d_dbg[4] = match ? 1 : 0;
  d_dbg[0] = 41;
  __threadfence_system();

  *d_result = match ? 1 : -7;
  __threadfence_system();
}

// ============================================================================
// Test: GPU→GPU bdev write/read
// ============================================================================

TEST_CASE("bdev_gpu2gpu_write_read", "[gpu_bdev][gpu2gpu]") {
  fprintf(stderr, "\n=== bdev_gpu2gpu_write_read START ===\n");
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;
  auto *gpu_ipc = ipc->GetGpuIpcManager();

  ++g_pool_counter;
  chi::PoolId pool_id(20000, g_pool_counter);
  chimaera::bdev::Client client(pool_id);

  constexpr size_t kBufSize = 256 * 1024;
  constexpr size_t kDataSize = 4096;

  // Create HBM bdev (use CHI_CPU_IPC — CHI_IPC is nullptr on nvcc host)
  {
    auto *cpu_ipc = CHI_CPU_IPC;
    auto task = cpu_ipc->NewTask<chimaera::bdev::CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Dynamic(),
        chimaera::bdev::CreateParams::chimod_lib_name,
        std::string("gpu2gpu_bdev"), pool_id, &client,
        chimaera::bdev::BdevType::kHbm, (chi::u64)kBufSize,
        (chi::u32)32, (chi::u32)4096,
        (const chimaera::bdev::PerfMetrics*)nullptr);
    auto f = cpu_ipc->Send(task);
    f.Wait();
    REQUIRE(f->return_code_ == 0);
  }
  std::this_thread::sleep_for(200ms);

  // Get GPU info (shared orchestrator allocator)
  chi::IpcManagerGpuInfo gpu_info = gpu_ipc->CreateGpuAllocator(0, 0);

  // Pause for UVM allocation
  gpu_ipc->PauseGpuOrchestrator();

  // Allocate UVM buffers
  char *src_uvm, *dst_uvm;
  cudaMallocManaged(&src_uvm, kDataSize);
  cudaMallocManaged(&dst_uvm, kDataSize);
  REQUIRE(src_uvm != nullptr);
  REQUIRE(dst_uvm != nullptr);
  memset(src_uvm, 0xCD, kDataSize);
  memset(dst_uvm, 0x00, kDataSize);

  // Allocate pinned result slots
  int *d_result;
  volatile int *d_dbg;
  cudaMallocHost(&d_result, sizeof(int));
  cudaMallocHost((void**)&d_dbg, sizeof(int) * 8);
  *d_result = 0;
  for (int i = 0; i < 8; i++) d_dbg[i] = 0;

  // Launch kernel while paused, then resume
  void *stream = hshm::GpuApi::CreateStream();
  bdev_gpu2gpu_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, kDataSize, src_uvm, dst_uvm, d_result, d_dbg);
  cudaError_t err = cudaGetLastError();
  REQUIRE(err == cudaSuccess);

  gpu_ipc->ResumeGpuOrchestrator();

  // Poll for completion
  auto t0 = std::chrono::steady_clock::now();
  while (*d_result == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= 30.0f) {
      fprintf(stderr, "[FAIL] Timeout d_result=%d step=%d "
              "alloc_rc=%d write_rc=%d read_rc=%d verify=%d\n",
              *d_result, d_dbg[0], d_dbg[1], d_dbg[2], d_dbg[3], d_dbg[4]);
      REQUIRE(false);
    }
  }

  float ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "[TRACE] GPU→GPU bdev: d_result=%d step=%d "
          "alloc_rc=%d write_rc=%d read_rc=%d verify=%d (%.1f ms)\n",
          *d_result, d_dbg[0], d_dbg[1], d_dbg[2], d_dbg[3], d_dbg[4], ms);
  REQUIRE(*d_result == 1);

  // Cleanup
  gpu_ipc->PauseGpuOrchestrator();
  cudaFree(src_uvm);
  cudaFree(dst_uvm);
  cudaFreeHost(d_result);
  cudaFreeHost((void*)d_dbg);
  hshm::GpuApi::DestroyStream(stream);
  gpu_ipc->ResumeGpuOrchestrator();

  fprintf(stderr, "=== bdev_gpu2gpu_write_read PASS ===\n");
}

// ============================================================================
// GPU→CPU kernel: AllocateBlocks + Write + Read via PoolQuery::ToLocalCpu()
// (Same kernel structure but with ToLocalCpu routing)
// ============================================================================

/**
 * GPU→CPU kernel: Write + Read via ToLocalCpu.
 * Blocks are pre-allocated on the CPU and passed to the kernel.
 * (AllocateBlocks returns priv::vector which uses CPU-only allocator,
 * so GPU kernels can't read the result.)
 */
__global__ void bdev_gpu2cpu_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u64 data_size,
    chimaera::bdev::Block block,   // Single block passed by value
    char *src_buf,
    char *dst_buf,
    int *d_result,
    volatile int *d_dbg) {
  *d_result = 0;
  d_dbg[0] = 1;
  __threadfence_system();
  CHIMAERA_GPU_INIT(gpu_info);

  // Build a single-element blocks vector on the GPU
  chi::priv::vector<chimaera::bdev::Block> blocks;
  blocks.push_back(block);

  // Step 1: Write via CPU
  d_dbg[0] = 20;
  __threadfence_system();
  hipc::ShmPtr<> write_data;
  write_data.alloc_id_.SetNull();
  write_data.off_ = reinterpret_cast<size_t>(src_buf);
  auto write_task = CHI_IPC->NewTask<chimaera::bdev::WriteTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::ToLocalCpu(),
      blocks, write_data, data_size);
  if (write_task.IsNull()) { *d_result = -3; __threadfence_system(); return; }
  auto write_f = CHI_IPC->Send(write_task);
  write_f.Wait();
  d_dbg[2] = write_task->return_code_;
  d_dbg[0] = 21;
  __threadfence_system();
  if (write_task->return_code_ != 0) {
    *d_result = -4;
    __threadfence_system();
    return;
  }

  // Step 2: Read via CPU
  d_dbg[0] = 30;
  __threadfence_system();
  hipc::ShmPtr<> read_data;
  read_data.alloc_id_.SetNull();
  read_data.off_ = reinterpret_cast<size_t>(dst_buf);
  auto read_task = CHI_IPC->NewTask<chimaera::bdev::ReadTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::ToLocalCpu(),
      blocks, read_data, data_size);
  if (read_task.IsNull()) { *d_result = -5; __threadfence_system(); return; }
  auto read_f = CHI_IPC->Send(read_task);
  read_f.Wait();
  d_dbg[3] = read_task->return_code_;
  d_dbg[0] = 31;
  __threadfence_system();
  if (read_task->return_code_ != 0) {
    *d_result = -6;
    __threadfence_system();
    return;
  }

  // Step 3: Verify
  d_dbg[0] = 40;
  __threadfence_system();
  bool match = true;
  for (chi::u64 i = 0; i < data_size; ++i) {
    if (src_buf[i] != dst_buf[i]) {
      match = false;
      break;
    }
  }
  d_dbg[4] = match ? 1 : 0;
  d_dbg[0] = 41;
  __threadfence_system();

  *d_result = match ? 1 : -7;
  __threadfence_system();
}

// ============================================================================
// Test: GPU→CPU bdev write/read
// ============================================================================

TEST_CASE("bdev_gpu2cpu_write_read", "[gpu_bdev][gpu2cpu]") {
  fprintf(stderr, "\n=== bdev_gpu2cpu_write_read START ===\n");
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;
  auto *gpu_ipc = ipc->GetGpuIpcManager();

  ++g_pool_counter;
  chi::PoolId pool_id(20000, g_pool_counter);
  chimaera::bdev::Client client(pool_id);

  constexpr size_t kBufSize = 256 * 1024;
  constexpr size_t kDataSize = 4096;

  // Create HBM bdev (use CHI_CPU_IPC — CHI_IPC is nullptr on nvcc host)
  {
    auto *cpu_ipc = CHI_CPU_IPC;
    auto task = cpu_ipc->NewTask<chimaera::bdev::CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Dynamic(),
        chimaera::bdev::CreateParams::chimod_lib_name,
        std::string("gpu2cpu_bdev"), pool_id, &client,
        chimaera::bdev::BdevType::kHbm, (chi::u64)kBufSize,
        (chi::u32)32, (chi::u32)4096,
        (const chimaera::bdev::PerfMetrics*)nullptr);
    auto f = cpu_ipc->Send(task);
    f.Wait();
    REQUIRE(f->return_code_ == 0);
  }
  std::this_thread::sleep_for(200ms);

  // Allocate blocks on the CPU side first
  // (AllocateBlocks returns priv::vector which uses CPU allocator)
  {
    auto *cpu_ipc = CHI_CPU_IPC;
    auto task = cpu_ipc->NewTask<chimaera::bdev::AllocateBlocksTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Dynamic(),
        (chi::u64)kDataSize);
    auto f = cpu_ipc->Send(task);
    f.Wait();
    REQUIRE(f->return_code_ == 0);
    REQUIRE(f->blocks_.size() > 0);
    // Save the first block to pass to the GPU kernel
    g_block = f->blocks_[0];
  }
  fprintf(stderr, "[TRACE] CPU AllocateBlocks: offset=%llu size=%llu\n",
          (unsigned long long)g_block.offset_,
          (unsigned long long)g_block.size_);

  chi::IpcManagerGpuInfo gpu_info = gpu_ipc->CreateGpuAllocator(0, 0);

  // Pause for UVM/pinned allocation
  gpu_ipc->PauseGpuOrchestrator();

  char *src_uvm, *dst_uvm;
  cudaMallocManaged(&src_uvm, kDataSize);
  cudaMallocManaged(&dst_uvm, kDataSize);
  REQUIRE(src_uvm != nullptr);
  REQUIRE(dst_uvm != nullptr);
  memset(src_uvm, 0xEF, kDataSize);
  memset(dst_uvm, 0x00, kDataSize);

  int *d_result;
  volatile int *d_dbg;
  cudaMallocHost(&d_result, sizeof(int));
  cudaMallocHost((void**)&d_dbg, sizeof(int) * 8);
  *d_result = 0;
  for (int i = 0; i < 8; i++) d_dbg[i] = 0;

  void *stream = hshm::GpuApi::CreateStream();
  bdev_gpu2cpu_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, kDataSize, g_block, src_uvm, dst_uvm,
      d_result, d_dbg);
  cudaError_t err = cudaGetLastError();
  REQUIRE(err == cudaSuccess);

  gpu_ipc->ResumeGpuOrchestrator();

  // Poll for completion
  auto t0 = std::chrono::steady_clock::now();
  while (*d_result == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= 30.0f) {
      fprintf(stderr, "[FAIL] Timeout d_result=%d step=%d "
              "alloc_rc=%d write_rc=%d read_rc=%d verify=%d\n",
              *d_result, d_dbg[0], d_dbg[1], d_dbg[2], d_dbg[3], d_dbg[4]);
      REQUIRE(false);
    }
  }

  float ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "[TRACE] GPU→CPU bdev: d_result=%d step=%d "
          "alloc_rc=%d write_rc=%d read_rc=%d verify=%d (%.1f ms)\n",
          *d_result, d_dbg[0], d_dbg[1], d_dbg[2], d_dbg[3], d_dbg[4], ms);
  REQUIRE(*d_result == 1);

  // Cleanup
  gpu_ipc->PauseGpuOrchestrator();
  cudaFree(src_uvm);
  cudaFree(dst_uvm);
  cudaFreeHost(d_result);
  cudaFreeHost((void*)d_dbg);
  hshm::GpuApi::DestroyStream(stream);
  gpu_ipc->ResumeGpuOrchestrator();

  fprintf(stderr, "=== bdev_gpu2cpu_write_read PASS ===\n");
}

SIMPLE_TEST_MAIN()

#endif  // HSHM_ENABLE_CUDA
