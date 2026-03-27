/**
 * workload_cte_client_overhead.cc — Measure GPU-side cost of AsyncPutBlob
 *
 * This benchmark isolates the client-side overhead of submitting a PutBlob
 * task from a GPU kernel. Each warp calls AsyncPutBlob and times only the
 * submission (not the Wait), reporting per-warp clock cycles via pinned
 * memory. This reveals how expensive the CTE client API is on the GPU.
 *
 * GPU Kernels:
 * 1. gpu_putblob_alloc_kernel — Initialize allocator and allocate array
 * 2. gpu_client_overhead_kernel — Time AsyncPutBlob submission cost
 *
 * Launcher:
 * - run_cte_client_overhead() — Setup, launch, collect per-warp timings
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/logging.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>

#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

//==============================================================================
// GPU Kernels
//==============================================================================

/**
 * Initialize a BuddyAllocator over device memory and allocate
 * a contiguous array of `total_bytes` bytes.  Returns the FullPtr via
 * pinned host memory so the CPU can read it.
 */
__global__ void gpu_putblob_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) {
    d_out_ptr->SetNull();
    return;
  }
  auto result = alloc->AllocateObjs<char>(total_bytes);
  *d_out_ptr = result;
}

/**
 * Measure the GPU-side cost of AsyncPutBlob submission.
 *
 * Each warp:
 *   1. Fills its data slice
 *   2. Calls AsyncPutBlob, timing only the call itself (clock64 delta)
 *   3. Waits for completion (untimed)
 *   4. Accumulates per-warp submit cycles in d_submit_clk[]
 *
 * d_submit_clk[warp_id] = total clock64 cycles spent in AsyncPutBlob calls
 *                          across all iterations for that warp.
 */
__global__ void gpu_client_overhead_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> array_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 warp_bytes,
    chi::u32 total_warps,
    chi::u32 iterations,
    bool to_cpu,
    int *d_done,
    volatile int *d_progress,
    long long *d_submit_clk) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (lane_id == 0 && warp_id < total_warps) {
    d_progress[warp_id] = 1;  // init complete
    d_submit_clk[warp_id] = 0;
    __threadfence_system();
  }
  __syncwarp();

  if (warp_id < total_warps) {
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * warp_bytes;
    char *my_data = array_ptr.ptr_ + my_offset;

    if (lane_id == 0) {
      d_progress[warp_id] = -100;  // entering loop
      __threadfence_system();
    }
    __syncwarp();

    bool alloc_failed = false;
    long long submit_acc = 0;

    for (chi::u32 iter = 0; iter < iterations; ++iter) {
      // All lanes participate in memset
      for (chi::u64 i = lane_id; i < warp_bytes; i += 32) {
        my_data[i] = static_cast<char>((warp_id + iter) & 0xFF);
      }
      __syncwarp();

      // Only lane 0 submits PutBlob
      if (chi::IpcManager::IsWarpScheduler()) {
        if (!alloc_failed) {
          if (warp_id < total_warps) {
            d_progress[warp_id] = static_cast<int>(2 + (iter << 8));
            __threadfence_system();
          }

          wrp_cte::core::Client cte_client(cte_pool_id);

          hipc::ShmPtr<> blob_shm;
          blob_shm.alloc_id_ = data_alloc_id;
          size_t base_off = array_ptr.shm_.off_.load();
          blob_shm.off_.exchange(base_off + my_offset);

          // Build blob name: "w_<warp_id>"
          using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
          char name_buf[32];
          int pos = 0;
          name_buf[pos++] = 'w';
          name_buf[pos++] = '_';
          pos += StrT::NumberToStr(name_buf + pos, 32 - pos, warp_id);
          name_buf[pos] = '\0';

          auto pool_query = to_cpu ? chi::PoolQuery::ToLocalCpu()
                                   : chi::PoolQuery::Local();

          // === TIMED: AsyncPutBlob submission only ===
          long long t0 = clock64();
          auto future = cte_client.AsyncPutBlob(
              tag_id, name_buf,
              (chi::u64)0, warp_bytes,
              blob_shm, -1.0f,
              wrp_cte::core::Context(), (chi::u32)0, pool_query);
          long long t1 = clock64();
          submit_acc += (t1 - t0);

          // Check if allocation succeeded before waiting
          if (future.GetFutureShmPtr().IsNull()) {
            if (warp_id < total_warps) {
              d_progress[warp_id] = -(1000 + static_cast<int>(iter));
              __threadfence_system();
            }
            alloc_failed = true;
          } else {
            // Wait is NOT timed — we only care about submit cost
            future.Wait();
          }
        }
      }
      __syncwarp();
    }

    // Store accumulated submit cycles
    if (chi::IpcManager::IsWarpScheduler() && warp_id < total_warps) {
      d_submit_clk[warp_id] = submit_acc;
      __threadfence_system();
    }
  }

  __syncwarp();
  if (chi::IpcManager::IsWarpScheduler()) {
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

//==============================================================================
// Host-side Launcher
//==============================================================================

#include <hermes_shm/constants/macros.h>

#if HSHM_IS_HOST

#include "workload.h"

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

int run_cte_client_overhead(
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    uint32_t rt_blocks,
    uint32_t rt_threads,
    uint32_t client_blocks,
    uint32_t client_threads,
    uint64_t warp_bytes,
    uint32_t iterations,
    bool to_cpu,
    int timeout_sec,
    float *out_elapsed_ms,
    float *out_avg_submit_us) {
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);
  CHI_IPC->PauseGpuOrchestrator();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;
  chi::u64 total_data_bytes = warp_bytes * total_warps;

  // --- 1. Data backend: device memory for array ---
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  size_t data_backend_size = total_data_bytes + 4 * 1024 * 1024;
  if (!data_backend.shm_init(data_backend_id, data_backend_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 2. Client scratch backend ---
  constexpr size_t kPerWarpScratch = 1 * 1024 * 1024;
  size_t scratch_size = static_cast<size_t>(total_warps) * kPerWarpScratch;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 3. GPU heap backend ---
  constexpr size_t kPerWarpHeap = 1 * 1024 * 1024;
  size_t heap_size = static_cast<size_t>(total_warps) * kPerWarpHeap;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 4. Allocate array ---
  hipc::FullPtr<char> *d_array_ptr;
  cudaMallocHost(&d_array_ptr, sizeof(hipc::FullPtr<char>));
  d_array_ptr->SetNull();

  gpu_putblob_alloc_kernel<<<1, 1>>>(
      static_cast<hipc::MemoryBackend &>(data_backend),
      total_data_bytes, d_array_ptr);
  cudaDeviceSynchronize();

  if (d_array_ptr->IsNull()) {
    cudaFreeHost(d_array_ptr);
    CHI_IPC->ResumeGpuOrchestrator();
    return -2;
  }

  hipc::FullPtr<char> array_ptr = *d_array_ptr;
  cudaFreeHost(d_array_ptr);

  // --- 5. Register data backend ---
  CHI_IPC->RegisterGpuAllocator(data_backend_id, data_backend.data_,
                                 data_backend.data_capacity_);

  // --- 6. Build GPU info and launch kernel ---
  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int) * total_warps);
  memset((void*)d_progress, 0, sizeof(int) * total_warps);

  long long *d_submit_clk;
  cudaMallocHost(&d_submit_clk, sizeof(long long) * total_warps);
  memset(d_submit_clk, 0, sizeof(long long) * total_warps);

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  if (scratch_backend.data_ != nullptr) {
    cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  }
  if (heap_backend.data_ != nullptr) {
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  }

  // Launch client overhead kernel
  gpu_client_overhead_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, client_blocks,
      array_ptr,
      hipc::AllocatorId(data_backend_id.major_, data_backend_id.minor_),
      warp_bytes, total_warps, iterations, to_cpu, d_done, d_progress,
      d_submit_clk);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "ERROR: Client overhead kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    cudaFreeHost(d_submit_clk);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  auto *orchestrator = static_cast<chi::gpu::WorkOrchestrator *>(
      CHI_IPC->gpu_orchestrator_);
  auto *ctrl = orchestrator ? orchestrator->control_ : nullptr;

  CHI_IPC->ResumeGpuOrchestrator();
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
  auto wall_start = std::chrono::high_resolution_clock::now();
  bool completed = PollDone(d_done, static_cast<int>(total_warps), timeout_us);
  auto wall_end = std::chrono::high_resolution_clock::now();

  if (!completed) {
    if (ctrl) {
      fprintf(stderr, "TIMEOUT: d_done=%d/%u running_flag=%d\n",
              __atomic_load_n(d_done, __ATOMIC_ACQUIRE), total_warps,
              ctrl->running_flag);
    }
    fprintf(stderr, "Client warp progress:\n");
    for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
      int p = d_progress[i];
      if (p <= -1000) {
        int fail_iter = -(p + 1000);
        fprintf(stderr, "  warp[%u]: ALLOC_FAIL at iter=%d\n", i, fail_iter);
      } else if (p >= 2) {
        int iter = (p >> 8) & 0xFFFFFF;
        fprintf(stderr, "  warp[%u]: iter=%d (raw=%d)\n", i, iter, p);
      } else {
        fprintf(stderr, "  warp[%u]: %d\n", i, p);
      }
    }
    fflush(stderr);
    CHI_IPC->PauseGpuOrchestrator();
    *out_elapsed_ms = 0;
    *out_avg_submit_us = 0;
    cudaFreeHost(d_done);
    cudaFreeHost((void*)d_progress);
    cudaFreeHost(d_submit_clk);
    hshm::GpuApi::DestroyStream(stream);
    return -4;
  }

  // Wall-clock elapsed
  float gpu_elapsed_ms = std::chrono::duration<float, std::milli>(
      wall_end - wall_start).count();
  *out_elapsed_ms = gpu_elapsed_ms;

  // Convert GPU clock cycles to microseconds
  // clock64() counts at the SM clock rate (clockRate in kHz).
  int gpu_device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpu_device);
  double clk_rate_khz = static_cast<double>(prop.clockRate);  // kHz = cycles/ms

  // Compute average per-call submit cost across all warps
  long long total_submit_clk = 0;
  uint32_t active_warps = 0;
  for (chi::u32 i = 0; i < total_warps; ++i) {
    if (d_submit_clk[i] > 0) {
      total_submit_clk += d_submit_clk[i];
      active_warps++;
    }
  }

  if (active_warps > 0 && iterations > 0) {
    // Average cycles per submit call per warp
    double avg_cycles_per_call = static_cast<double>(total_submit_clk) /
                                  (active_warps * iterations);
    // Convert: cycles / (cycles/ms) = ms; * 1000 = us
    *out_avg_submit_us = static_cast<float>((avg_cycles_per_call / clk_rate_khz) * 1000.0);
  } else {
    *out_avg_submit_us = 0;
  }

  // Print per-warp breakdown
  printf("\n--- Per-warp AsyncPutBlob submit cost ---\n");
  printf("SM clock rate: %.0f kHz (%.1f GHz)\n", clk_rate_khz, clk_rate_khz / 1e6);
  for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
    double warp_cycles = static_cast<double>(d_submit_clk[i]);
    double warp_us_per_call = (warp_cycles / iterations / clk_rate_khz) * 1000.0;
    printf("  warp[%u]: %lld total cycles, %.1f us/call (%u calls)\n",
           i, d_submit_clk[i], warp_us_per_call, iterations);
  }
  printf("  Average: %.1f us/call across %u warps\n", *out_avg_submit_us, active_warps);
  printf("---\n");

  hshm::GpuApi::Synchronize(stream);
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  cudaFreeHost((void*)d_progress);
  cudaFreeHost(d_submit_clk);
  hshm::GpuApi::DestroyStream(stream);

  return 0;
}

#endif  // HSHM_IS_HOST

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
