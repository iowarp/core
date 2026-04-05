/**
 * workload_bdev.cc — Block device (bdev) I/O benchmark for CTE GPU bench
 *
 * Extracted from wrp_cte_gpu_bench.cc
 * Measures block device I/O performance:
 *   - Alloc/Free: Benchmark memory allocation and deallocation on bdev
 *   - Read/Write: Benchmark sequential read/write operations through bdev client
 *
 * Both modes run in GPU kernels with per-warp async I/O operations.
 */

#include <cstdint>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>
#include "cte_helpers.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

//==============================================================================
// GPU Kernels
//==============================================================================

/**
 * Kernel: BDEV Alloc + Free loop
 * Each warp allocates and frees the same size block repeatedly.
 * Reports per-warp progress via pinned array.
 */
__global__ void gpu_bdev_alloc_free_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId bdev_pool_id,
    chi::u32 num_blocks,
    chi::u64 alloc_size,
    chi::u32 total_warps,
    chi::u32 iterations,
    int *d_done,
    volatile int *d_progress) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::gpu::IpcManager::GetLaneId();

  if (lane_id == 0 && warp_id < total_warps) {
    d_progress[warp_id] = 1;
    __threadfence_system();
  }
  __syncwarp();

  if (warp_id < total_warps && chi::gpu::IpcManager::IsWarpScheduler()) {
    chimaera::bdev::Client bdev_client(bdev_pool_id);
    auto pool_query = chi::PoolQuery::Local();

    for (chi::u32 iter = 0; iter < iterations; ++iter) {
      d_progress[warp_id] = static_cast<int>(2 + (iter << 8));
      __threadfence_system();

      // Allocate
      auto alloc_task = CHI_IPC->NewTask<chimaera::bdev::AllocateBlocksTask>(
          chi::CreateTaskId(), bdev_client.pool_id_, pool_query, alloc_size);
      if (alloc_task.IsNull()) {
        d_progress[warp_id] = -(1000 + static_cast<int>(iter));
        __threadfence_system();
        break;
      }
      auto alloc_future = CHI_IPC->Send(alloc_task);
      alloc_future.WaitGpu();

      // Free
      auto free_task = CHI_IPC->NewTask<chimaera::bdev::FreeBlocksTask>(
          chi::CreateTaskId(), bdev_client.pool_id_, pool_query, alloc_task->blocks_);
      if (!free_task.IsNull()) {
        auto free_future = CHI_IPC->Send(free_task);
        free_future.WaitGpu();
      }
    }
  }
  __syncwarp();

  if (chi::gpu::IpcManager::IsWarpScheduler() && warp_id < total_warps) {
    d_progress[warp_id] = 10;
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

/**
 * Kernel: BDEV Write + Read loop with pre-allocated blocks.
 * Alloc once → (Write + Read) x iterations → Free once.
 * Reports per-warp write and read clock cycles via pinned arrays.
 */
__global__ void gpu_bdev_read_write_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId bdev_pool_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 warp_bytes,
    chi::u32 total_warps,
    chi::u32 iterations,
    int *d_done,
    volatile int *d_progress,
    long long *d_write_clk,
    long long *d_read_clk) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::gpu::IpcManager::GetLaneId();

  if (lane_id == 0 && warp_id < total_warps) {
    d_progress[warp_id] = 1;
    d_write_clk[warp_id] = 0;
    d_read_clk[warp_id] = 0;
    __threadfence_system();
  }
  __syncwarp();

  if (warp_id < total_warps) {
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * warp_bytes;
    char *my_data = data_ptr.ptr_ + my_offset;

    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      chimaera::bdev::Client bdev_client(bdev_pool_id);
      auto pool_query = chi::PoolQuery::Local();
      auto pool_query_parallel = chi::PoolQuery::Local(32);
      long long write_acc = 0, read_acc = 0;

      // Allocate blocks once
      auto alloc_task = CHI_IPC->NewTask<chimaera::bdev::AllocateBlocksTask>(
          chi::CreateTaskId(), bdev_client.pool_id_, pool_query, warp_bytes);
      if (alloc_task.IsNull()) {
        d_progress[warp_id] = -1000;
        __threadfence_system();
      } else {
        auto alloc_future = CHI_IPC->Send(alloc_task);
        alloc_future.WaitGpu();

        d_progress[warp_id] = 2;
        __threadfence_system();

        // Build ShmPtr for this warp's data slice
        hipc::ShmPtr<> data_shm;
        data_shm.alloc_id_ = data_alloc_id;
        size_t base_off = data_ptr.shm_.off_.load();
        data_shm.off_.exchange(base_off + my_offset);

        // Write + Read loop
        for (chi::u32 iter = 0; iter < iterations; ++iter) {
          d_progress[warp_id] = static_cast<int>(3 + (iter << 8));
          __threadfence_system();

          // Fill write buffer (all lanes participate)
          __syncwarp();
          for (chi::u64 i = lane_id; i < warp_bytes; i += 32) {
            my_data[i] = static_cast<char>((warp_id + iter) & 0xFF);
          }
          __syncwarp();

          // Write (timed)
          long long t0 = clock64();
          auto write_task = CHI_IPC->NewTask<chimaera::bdev::WriteTask>(
              chi::CreateTaskId(), bdev_client.pool_id_,
              pool_query_parallel, alloc_task->blocks_, data_shm, warp_bytes);
          if (write_task.IsNull()) {
            d_progress[warp_id] = -(2000 + static_cast<int>(iter));
            __threadfence_system();
            break;
          }
          auto write_future = CHI_IPC->Send(write_task);
          write_future.WaitGpu();
          long long t1 = clock64();
          write_acc += (t1 - t0);

          // Zero buffer before read
          __syncwarp();
          for (chi::u64 i = lane_id; i < warp_bytes; i += 32) {
            my_data[i] = 0;
          }
          __syncwarp();

          // Read (timed)
          long long t2 = clock64();
          auto read_task = CHI_IPC->NewTask<chimaera::bdev::ReadTask>(
              chi::CreateTaskId(), bdev_client.pool_id_,
              pool_query_parallel, alloc_task->blocks_, data_shm, warp_bytes);
          if (read_task.IsNull()) {
            d_progress[warp_id] = -(3000 + static_cast<int>(iter));
            __threadfence_system();
            break;
          }
          auto read_future = CHI_IPC->Send(read_task);
          read_future.WaitGpu();
          long long t3 = clock64();
          read_acc += (t3 - t2);
        }

        // Free blocks
        auto free_task = CHI_IPC->NewTask<chimaera::bdev::FreeBlocksTask>(
            chi::CreateTaskId(), bdev_client.pool_id_, pool_query, alloc_task->blocks_);
        if (!free_task.IsNull()) {
          auto free_future = CHI_IPC->Send(free_task);
          free_future.WaitGpu();
        }

        d_write_clk[warp_id] = write_acc;
        d_read_clk[warp_id] = read_acc;
        __threadfence_system();
      }
    }
  }

  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler() && warp_id < total_warps) {
    d_progress[warp_id] = 10;
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

//==============================================================================
// Host-only code: CPU-side launchers (guarded by HSHM_IS_HOST)
//==============================================================================
#if HSHM_IS_HOST

// Utility: Poll for kernel completion
static bool PollDone(int *d_done, int total_warps, int64_t timeout_us) {
  int64_t elapsed_us = 0;
  int cur = __atomic_load_n(d_done, __ATOMIC_ACQUIRE);
  while (cur < total_warps && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
    cur = __atomic_load_n(d_done, __ATOMIC_ACQUIRE);
  }
  return cur >= total_warps;
}

/**
 * Launcher: BDEV Alloc/Free benchmark
 *
 * Allocates and frees blocks on the bdev pool repeatedly.
 * Returns 0 on success, negative on failure.
 */
int run_bdev_alloc_free(
    chi::PoolId bdev_pool_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 alloc_size,
    chi::u32 iterations,
    int timeout_sec,
    float *out_elapsed_ms) {
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);
  CHI_CPU_IPC->PauseGpuOrchestrator();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  // Scratch backend
  constexpr size_t kPerWarpScratch = 1 * 1024 * 1024;
  size_t scratch_size = static_cast<size_t>(total_warps) * kPerWarpScratch;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // Heap backend
  constexpr size_t kPerWarpHeap = 1 * 1024 * 1024;
  size_t heap_size = static_cast<size_t>(total_warps) * kPerWarpHeap;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;


  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;
  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int) * total_warps);
  memset((void*)d_progress, 0, sizeof(int) * total_warps);

  if (scratch_backend.data_ != nullptr)
    cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  if (heap_backend.data_ != nullptr)
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  auto wall_start = std::chrono::high_resolution_clock::now();

  gpu_bdev_alloc_free_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, bdev_pool_id, client_blocks,
      alloc_size, total_warps, iterations,
      d_done, d_progress);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "ERROR: bdev_alloc_free kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->ResumeGpuOrchestrator();
  auto *orchestrator = static_cast<chi::gpu::WorkOrchestrator *>(
      CHI_CPU_IPC->GetGpuIpcManager()->gpu_orchestrator_);
  auto *ctrl = orchestrator ? orchestrator->control_ : nullptr;
  if (ctrl) {
    int wait_ms = 0;
    while (ctrl->running_flag == 0 && wait_ms < 5000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ++wait_ms;
    }
  }

  int64_t timeout_us = static_cast<int64_t>(timeout_sec) * 1000000;
  fprintf(stderr, "PollDone: waiting up to %d seconds for %u warps...\n",
          timeout_sec, total_warps);
  fflush(stderr);
  bool completed = PollDone(d_done, static_cast<int>(total_warps), timeout_us);

  auto wall_end = std::chrono::high_resolution_clock::now();
  *out_elapsed_ms = std::chrono::duration<float, std::milli>(
      wall_end - wall_start).count();

  if (!completed) {
    fprintf(stderr, "TIMEOUT: d_done=%d/%u\n",
            __atomic_load_n(d_done, __ATOMIC_ACQUIRE), total_warps);
    for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
      fprintf(stderr, "  warp[%u]: %d\n", i, d_progress[i]);
    }
    fflush(stderr);
    CHI_CPU_IPC->PauseGpuOrchestrator();
    *out_elapsed_ms = 0;
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    hshm::GpuApi::DestroyStream(stream);
    return -4;
  }

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->PauseGpuOrchestrator();
  cudaFreeHost(d_done);
  cudaFreeHost((void*)d_progress);
  hshm::GpuApi::DestroyStream(stream);
  return 0;
}

/**
 * Launcher: BDEV Read/Write benchmark
 *
 * Allocates blocks, then repeatedly writes and reads data.
 * Measures per-warp clock cycles for write and read operations.
 * Returns 0 on success, negative on failure.
 */
int run_bdev_read_write(
    chi::PoolId bdev_pool_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 warp_bytes,
    chi::u32 iterations,
    int timeout_sec,
    float *out_elapsed_ms,
    float *out_write_ms,
    float *out_read_ms) {
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);
  CHI_CPU_IPC->PauseGpuOrchestrator();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;
  chi::u64 total_data_bytes = warp_bytes * total_warps;

  // Data backend for write/read buffers
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  size_t data_backend_size = total_data_bytes + 4 * 1024 * 1024;
  if (!data_backend.shm_init(data_backend_id, data_backend_size, "", 0)) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // Scratch backend
  constexpr size_t kPerWarpScratch = 1 * 1024 * 1024;
  size_t scratch_size = static_cast<size_t>(total_warps) * kPerWarpScratch;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // Heap backend
  constexpr size_t kPerWarpHeap = 1 * 1024 * 1024;
  size_t heap_size = static_cast<size_t>(total_warps) * kPerWarpHeap;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // Allocate data array on device
  hipc::FullPtr<char> *d_array_ptr;
  cudaMallocHost(&d_array_ptr, sizeof(hipc::FullPtr<char>));
  d_array_ptr->SetNull();
  gpu_putblob_alloc_kernel<<<1, 1>>>(
      static_cast<hipc::MemoryBackend &>(data_backend),
      total_data_bytes, d_array_ptr);
  cudaDeviceSynchronize();
  if (d_array_ptr->IsNull()) {
    cudaFreeHost(d_array_ptr);
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -2;
  }
  hipc::FullPtr<char> data_ptr = *d_array_ptr;
  cudaFreeHost(d_array_ptr);

  // Register data backend for ShmPtr resolution
  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(data_backend_id, data_backend.data_,
                                 data_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;


  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;
  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int) * total_warps);
  memset((void*)d_progress, 0, sizeof(int) * total_warps);

  // Per-warp clock cycle accumulators (pinned host memory)
  long long *d_write_clk, *d_read_clk;
  cudaMallocHost(&d_write_clk, sizeof(long long) * total_warps);
  cudaMallocHost(&d_read_clk, sizeof(long long) * total_warps);
  memset(d_write_clk, 0, sizeof(long long) * total_warps);
  memset(d_read_clk, 0, sizeof(long long) * total_warps);

  if (scratch_backend.data_ != nullptr)
    cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  if (heap_backend.data_ != nullptr)
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  auto wall_start = std::chrono::high_resolution_clock::now();

  gpu_bdev_read_write_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, bdev_pool_id, client_blocks,
      data_ptr,
      hipc::AllocatorId(data_backend_id.major_, data_backend_id.minor_),
      warp_bytes, total_warps, iterations,
      d_done, d_progress, d_write_clk, d_read_clk);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "ERROR: bdev_read_write kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    cudaFreeHost(d_write_clk);
    cudaFreeHost(d_read_clk);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->ResumeGpuOrchestrator();
  auto *orchestrator = static_cast<chi::gpu::WorkOrchestrator *>(
      CHI_CPU_IPC->GetGpuIpcManager()->gpu_orchestrator_);
  auto *ctrl = orchestrator ? orchestrator->control_ : nullptr;
  if (ctrl) {
    int wait_ms = 0;
    while (ctrl->running_flag == 0 && wait_ms < 5000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ++wait_ms;
    }
  }

  int64_t timeout_us = static_cast<int64_t>(timeout_sec) * 1000000;
  fprintf(stderr, "PollDone: waiting up to %d seconds for %u warps...\n",
          timeout_sec, total_warps);
  fflush(stderr);
  bool completed = PollDone(d_done, static_cast<int>(total_warps), timeout_us);

  auto wall_end = std::chrono::high_resolution_clock::now();
  *out_elapsed_ms = std::chrono::duration<float, std::milli>(
      wall_end - wall_start).count();

  if (!completed) {
    fprintf(stderr, "TIMEOUT: d_done=%d/%u\n",
            __atomic_load_n(d_done, __ATOMIC_ACQUIRE), total_warps);
    for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
      fprintf(stderr, "  warp[%u]: %d\n", i, d_progress[i]);
    }
    fflush(stderr);
    CHI_CPU_IPC->PauseGpuOrchestrator();
    *out_elapsed_ms = 0;
    *out_write_ms = 0;
    *out_read_ms = 0;
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    cudaFreeHost(d_write_clk);
    cudaFreeHost(d_read_clk);
    hshm::GpuApi::DestroyStream(stream);
    return -4;
  }

  // Convert GPU clock cycles to milliseconds using max across warps.
  // clock64() counts at the SM clock rate.
  int gpu_device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpu_device);
  double clk_rate_khz = static_cast<double>(prop.clockRate);  // kHz

  long long max_write_clk = 0, max_read_clk = 0;
  for (chi::u32 i = 0; i < total_warps; ++i) {
    if (d_write_clk[i] > max_write_clk) max_write_clk = d_write_clk[i];
    if (d_read_clk[i] > max_read_clk) max_read_clk = d_read_clk[i];
  }
  // clk_rate_khz = cycles/ms
  *out_write_ms = static_cast<float>(max_write_clk / clk_rate_khz);
  *out_read_ms = static_cast<float>(max_read_clk / clk_rate_khz);

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->PauseGpuOrchestrator();
  cudaFreeHost(d_done);
  cudaFreeHost((void*)d_progress);
  cudaFreeHost(d_write_clk);
  cudaFreeHost(d_read_clk);
  hshm::GpuApi::DestroyStream(stream);
  return 0;
}

#endif  // HSHM_IS_HOST
