/**
 * cte_helpers.h — Shared CTE boilerplate for workload benchmarks
 */
#ifndef BENCH_GPU_CTE_HELPERS_H
#define BENCH_GPU_CTE_HELPERS_H

#include <hermes_shm/constants/macros.h>

// Declare the alloc kernel from workload_cte_client_overhead.cc
extern __global__ void gpu_putblob_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr);

#if HSHM_IS_HOST

#include "workload.h"
#include <thread>

/**
 * Shared GPU benchmark context for CTE workloads.
 * Uses the orchestrator's shared allocator (not custom scratch).
 * Host code uses CHI_CPU_IPC (safe in nvcc host pass).
 */
struct CteGpuContext {
  hipc::MemoryBackendId data_id{200, 0};
  hipc::GpuMalloc data_backend;
  hipc::FullPtr<char> array_ptr;
  chi::IpcManagerGpu gpu_info;
  int *d_done = nullptr;
  volatile int *d_progress = nullptr;
  uint32_t total_warps = 0;
  bool valid = false;

  int init(uint64_t data_bytes, uint32_t num_warps) {
    total_warps = num_warps;
    CHI_CPU_IPC->PauseGpuOrchestrator();

    // Data backend: device memory for blob data
    size_t data_size = data_bytes + 16 * 1024 * 1024;
    if (!data_backend.shm_init(data_id, data_size, "", 0)) {
      CHI_CPU_IPC->ResumeGpuOrchestrator(); return -1;
    }

    // Allocate array via GPU kernel
    hipc::FullPtr<char> *d_ptr;
    cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gpu_putblob_alloc_kernel<<<1, 1>>>(
        static_cast<hipc::MemoryBackend &>(data_backend),
        data_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      cudaFreeHost(d_ptr);
      CHI_CPU_IPC->ResumeGpuOrchestrator(); return -2;
    }
    array_ptr = *d_ptr;
    cudaFreeHost(d_ptr);

    // Register data backend
    CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(
        data_id, data_backend.data_, data_backend.data_capacity_);

    // Use the orchestrator's shared allocator (proven working)
    gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->CreateGpuAllocator(0, 0);

    // Pinned host for completion tracking
    cudaMallocHost(&d_done, sizeof(int));
    *d_done = 0;
    cudaMallocHost((void **)&d_progress, sizeof(int) * num_warps);
    memset((void *)d_progress, 0, sizeof(int) * num_warps);

    valid = true;
    return 0;
  }

  bool resume_and_poll(int timeout_sec) {
    CHI_CPU_IPC->ResumeGpuOrchestrator();
    int64_t timeout_us = (int64_t)timeout_sec * 1000000;
    int64_t elapsed_us = 0;
    int cur = __atomic_load_n(d_done, __ATOMIC_ACQUIRE);
    while (cur < (int)total_warps && elapsed_us < timeout_us) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      elapsed_us += 100;
      cur = __atomic_load_n(d_done, __ATOMIC_ACQUIRE);
    }
    return cur >= (int)total_warps;
  }

  void cleanup() {
    if (d_done) { cudaFreeHost(d_done); d_done = nullptr; }
    if (d_progress) { cudaFreeHost((void *)d_progress); d_progress = nullptr; }
  }
};

#endif  // HSHM_IS_HOST
#endif  // BENCH_GPU_CTE_HELPERS_H
