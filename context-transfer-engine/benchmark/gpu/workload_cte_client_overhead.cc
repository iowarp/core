/**
 * workload_cte_client_overhead.cc — Measure GPU-side cost of PutBlob
 *
 * Each warp submits PutBlob tasks and measures submission latency.
 * Uses thread-0-only Send/WaitGpu pattern (no warp-level primitives).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/logging.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/gpu/work_orchestrator.h>
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
 * Allocate a contiguous array from device memory.
 */
__global__ void gpu_putblob_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

/**
 * Client overhead kernel: each warp does PutBlob and measures submission cost.
 * Thread 0 handles Send/WaitGpu; all lanes cooperate on memset.
 */
__global__ void gpu_client_overhead_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    char *array_base,        // absolute device pointer to data buffer
    chi::u64 warp_bytes,
    chi::u32 total_warps,
    chi::u32 iterations,
    bool to_cpu,
    int *d_done,
    volatile int *d_progress,
    long long *d_submit_clk) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::gpu::IpcManager::GetLaneId();
  auto *ipc = CHI_IPC;
  using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;

  if (warp_id >= total_warps) return;

  char *my_data = array_base + static_cast<chi::u64>(warp_id) * warp_bytes;

  if (lane_id == 0) {
    d_progress[warp_id] = 1;  // init complete
    d_submit_clk[warp_id] = 0;
    __threadfence_system();
  }
  __syncwarp();

  long long submit_acc = 0;

  // Pre-allocate a PutBlobTask (thread 0 only, reuse across iterations)
  hipc::FullPtr<wrp_cte::core::PutBlobTask> put_task;
  if (lane_id == 0) {
    // Use null alloc_id + absolute device address (ToFullPtr reads off_ as raw ptr)
    hipc::ShmPtr<> shm;
    shm.alloc_id_.SetNull();
    shm.off_.exchange(reinterpret_cast<size_t>(my_data));

    auto pool_query = to_cpu ? chi::PoolQuery::ToLocalCpu()
                             : chi::PoolQuery::Local();
    put_task = ipc->NewTask<wrp_cte::core::PutBlobTask>(
        chi::CreateTaskId(), cte_pool_id, pool_query,
        tag_id, "w_0", chi::u64(0), warp_bytes, shm, -1.0f,
        wrp_cte::core::Context(), chi::u32(0));
    if (put_task.IsNull()) {
      d_progress[warp_id] = -1000;
      __threadfence_system();
      atomicAdd_system(d_done, 1);
      return;
    }
  }
  __syncwarp();

  for (chi::u32 iter = 0; iter < iterations; ++iter) {
    // All lanes: fill data
    for (chi::u64 i = lane_id; i < warp_bytes; i += 32) {
      my_data[i] = static_cast<char>((warp_id + iter) & 0xFF);
    }
    __syncwarp();
    __threadfence();

    // Thread 0: update task fields, send, time, wait
    if (lane_id == 0) {
      // Update blob name
      char name[32]; int pos = 0;
      name[pos++] = 'w'; name[pos++] = '_';
      pos += StrT::NumberToStr(name + pos, 32 - pos, warp_id);
      name[pos] = '\0';

      put_task->blob_name_ = chi::priv::string(CHI_PRIV_ALLOC, name);
      put_task->blob_data_.alloc_id_.SetNull();
      put_task->blob_data_.off_.exchange(reinterpret_cast<size_t>(my_data));
      put_task->task_id_ = chi::CreateTaskId();
      put_task->return_code_ = -1;

      long long t0 = clock64();
      auto f = ipc->Send(put_task);
      long long t1 = clock64();
      submit_acc += (t1 - t0);

      f.WaitGpu();

      d_progress[warp_id] = static_cast<int>(2 + (iter << 8));
      __threadfence_system();
    }
    __syncwarp();
  }

  // Store results and signal completion
  if (lane_id == 0) {
    d_submit_clk[warp_id] = submit_acc;
    ipc->DelTask(put_task);
    __threadfence_system();
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
    chi::PoolId bdev_pool_id,
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
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);
  CHI_CPU_IPC->PauseGpuOrchestrator();

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;
  chi::u64 total_data_bytes = warp_bytes * total_warps;

  // --- 1. Data backend: device memory for blob data ---
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  if (!data_backend.shm_init(data_backend_id, total_data_bytes + 16*1024*1024, "", 0)) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 2. Allocate array via GPU kernel ---
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
  hipc::FullPtr<char> array_ptr = *d_array_ptr;
  cudaFreeHost(d_array_ptr);

  // --- 3. Register data backend and get GPU info ---
  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(
      data_backend_id, data_backend.data_, data_backend.data_capacity_);

  // Use the orchestrator's shared allocator (proven working pattern)
  chi::IpcManagerGpu gpu_info =
      CHI_CPU_IPC->GetGpuIpcManager()->CreateGpuAllocator(0, 0);

  // --- 4. Pinned host for results ---
  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  volatile int *d_progress;
  cudaMallocHost((void **)&d_progress, sizeof(int) * total_warps);
  memset((void *)d_progress, 0, sizeof(int) * total_warps);

  long long *d_submit_clk;
  cudaMallocHost(&d_submit_clk, sizeof(long long) * total_warps);
  memset(d_submit_clk, 0, sizeof(long long) * total_warps);

  // --- 5. Launch kernel ---
  void *stream = hshm::GpuApi::CreateStream();
  gpu_client_overhead_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, client_blocks,
      array_ptr.ptr_,  // absolute device pointer
      warp_bytes, total_warps, iterations, to_cpu,
      d_done, d_progress, d_submit_clk);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "ERROR: kernel launch failed: %s\n",
            cudaGetErrorString(launch_err));
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    cudaFreeHost((void *)d_progress);
    cudaFreeHost(d_submit_clk);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->ResumeGpuOrchestrator();

  // --- 6. Poll for completion ---
  int64_t timeout_us = static_cast<int64_t>(timeout_sec) * 1000000;
  fprintf(stderr, "PollDone: waiting up to %d seconds for %u warps...\n",
          timeout_sec, total_warps);
  fflush(stderr);
  auto wall_start = std::chrono::high_resolution_clock::now();
  bool completed = PollDone(d_done, static_cast<int>(total_warps), timeout_us);
  auto wall_end = std::chrono::high_resolution_clock::now();

  if (!completed) {
    fprintf(stderr, "TIMEOUT: d_done=%d/%u\n",
            __atomic_load_n(d_done, __ATOMIC_ACQUIRE), total_warps);
    for (chi::u32 i = 0; i < total_warps && i < 64; ++i) {
      fprintf(stderr, "  warp[%u]: progress=%d\n", i, (int)d_progress[i]);
    }
    fflush(stderr);
    CHI_CPU_IPC->PauseGpuOrchestrator();
    *out_elapsed_ms = 0;
    *out_avg_submit_us = 0;
    cudaFreeHost(d_done);
    cudaFreeHost((void *)d_progress);
    cudaFreeHost(d_submit_clk);
    hshm::GpuApi::DestroyStream(stream);
    return -4;
  }

  // --- 7. Compute results ---
  float gpu_elapsed_ms = std::chrono::duration<float, std::milli>(
      wall_end - wall_start).count();
  *out_elapsed_ms = gpu_elapsed_ms;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  double clk_rate_khz = static_cast<double>(prop.clockRate);

  long long total_submit_clk = 0;
  uint32_t active_warps = 0;
  for (chi::u32 i = 0; i < total_warps; ++i) {
    if (d_submit_clk[i] > 0) {
      total_submit_clk += d_submit_clk[i];
      active_warps++;
    }
  }

  if (active_warps > 0 && iterations > 0) {
    double avg_cycles_per_call = static_cast<double>(total_submit_clk) /
                                  (active_warps * iterations);
    *out_avg_submit_us = static_cast<float>(
        (avg_cycles_per_call / clk_rate_khz) * 1000.0);
  } else {
    *out_avg_submit_us = 0;
  }

  // --- 8. Cleanup ---
  CHI_CPU_IPC->PauseGpuOrchestrator();
  hshm::GpuApi::Synchronize(stream);
  hshm::GpuApi::DestroyStream(stream);
  cudaFreeHost(d_done);
  cudaFreeHost((void *)d_progress);
  cudaFreeHost(d_submit_clk);
  return 0;
}

#endif  // HSHM_IS_HOST
#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
