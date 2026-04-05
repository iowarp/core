/**
 * workload_cuda_malloc.cc — CUDA device malloc/free microbenchmark
 *
 * Measures the cost of malloc() and free() called from device code at
 * three allocation sizes: 512 B, 16 KB, 32 MB.
 *
 * The CUDA device heap (used by in-kernel malloc) defaults to 8 MB.
 * For large allocations we must grow it via cudaDeviceSetLimit().
 * Device malloc uses a global free-list with internal locking, so
 * contention grows with thread count — we sweep parallelism to show this.
 */

#include <hermes_shm/constants/macros.h>
#include "bench_common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

// ================================================================
// Kernel: each thread does malloc -> memset -> free, records cycles
// ================================================================

__global__ void cuda_malloc_bench_kernel(
    uint64_t alloc_bytes,
    uint32_t iterations,
    uint64_t *d_malloc_cycles,   // [total_threads * iterations]
    uint64_t *d_free_cycles,     // [total_threads * iterations]
    int *d_fail_count) {

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total_threads = gridDim.x * blockDim.x;

  for (uint32_t iter = 0; iter < iterations; iter++) {
    // --- malloc ---
    uint64_t t0 = clock64();
    void *ptr = malloc(alloc_bytes);
    uint64_t t1 = clock64();

    if (!ptr) {
      atomicAdd(d_fail_count, 1);
      d_malloc_cycles[tid * iterations + iter] = 0;
      d_free_cycles[tid * iterations + iter] = 0;
      continue;
    }

    // Touch every 4K page so the allocator actually commits
    volatile char *p = (volatile char *)ptr;
    for (uint64_t off = 0; off < alloc_bytes; off += 4096) {
      p[off] = (char)(tid ^ iter);
    }

    d_malloc_cycles[tid * iterations + iter] = t1 - t0;

    // --- free ---
    uint64_t t2 = clock64();
    free(ptr);
    uint64_t t3 = clock64();

    d_free_cycles[tid * iterations + iter] = t3 - t2;
  }
}

// ================================================================
// Kernel: parallel malloc without free (measure peak contention)
// ================================================================

__global__ void cuda_malloc_only_kernel(
    uint64_t alloc_bytes,
    uint64_t *d_malloc_cycles,
    void **d_ptrs,
    int *d_fail_count) {

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t t0 = clock64();
  void *ptr = malloc(alloc_bytes);
  uint64_t t1 = clock64();

  d_ptrs[tid] = ptr;
  d_malloc_cycles[tid] = (ptr != nullptr) ? (t1 - t0) : 0;
  if (!ptr) atomicAdd(d_fail_count, 1);
}

__global__ void cuda_free_only_kernel(void **d_ptrs) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (d_ptrs[tid]) free(d_ptrs[tid]);
}

// ================================================================
// Host driver
// ================================================================

struct MallocBenchResult {
  uint64_t alloc_bytes;
  uint32_t threads;
  uint32_t blocks;
  uint32_t iterations;
  double avg_malloc_us;
  double median_malloc_us;
  double p99_malloc_us;
  double avg_free_us;
  double median_free_us;
  double p99_free_us;
  int fail_count;
};

static double cycles_to_us(uint64_t cycles, int clock_khz) {
  return (double)cycles / (double)clock_khz * 1000.0;
}

static void compute_stats(std::vector<uint64_t> &samples, int clock_khz,
                          double *avg, double *median, double *p99) {
  // Remove zeros (failed allocs)
  samples.erase(
      std::remove(samples.begin(), samples.end(), 0ULL), samples.end());
  if (samples.empty()) {
    *avg = *median = *p99 = 0;
    return;
  }
  std::sort(samples.begin(), samples.end());
  uint64_t sum = 0;
  for (auto s : samples) sum += s;
  *avg = cycles_to_us(sum / samples.size(), clock_khz);
  *median = cycles_to_us(samples[samples.size() / 2], clock_khz);
  size_t p99_idx = std::min(samples.size() - 1, (size_t)(samples.size() * 0.99));
  *p99 = cycles_to_us(samples[p99_idx], clock_khz);
}

static MallocBenchResult run_single(uint64_t alloc_bytes, uint32_t blocks,
                                     uint32_t threads_per_block,
                                     uint32_t iterations) {
  uint32_t total_threads = blocks * threads_per_block;
  uint64_t n_samples = (uint64_t)total_threads * iterations;

  // Heap must be set before first kernel launch — caller is responsible
  // for calling set_device_heap() once before the benchmark loop.

  // Allocate output arrays
  uint64_t *d_malloc_cycles, *d_free_cycles;
  int *d_fail_count;
  CUDA_CHECK(cudaMalloc(&d_malloc_cycles, n_samples * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_free_cycles, n_samples * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_fail_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fail_count, 0, sizeof(int)));

  // Warmup
  cuda_malloc_bench_kernel<<<blocks, threads_per_block>>>(
      alloc_bytes, 1, d_malloc_cycles, d_free_cycles, d_fail_count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(d_fail_count, 0, sizeof(int)));

  // Benchmark
  cuda_malloc_bench_kernel<<<blocks, threads_per_block>>>(
      alloc_bytes, iterations, d_malloc_cycles, d_free_cycles, d_fail_count);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  std::vector<uint64_t> h_malloc(n_samples), h_free(n_samples);
  int h_fail = 0;
  CUDA_CHECK(cudaMemcpy(h_malloc.data(), d_malloc_cycles,
                         n_samples * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_free.data(), d_free_cycles,
                         n_samples * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_fail, d_fail_count, sizeof(int),
                         cudaMemcpyDeviceToHost));

  // Get GPU clock rate for cycle->us conversion
  int device = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int clock_khz = prop.clockRate;  // in kHz

  MallocBenchResult res = {};
  res.alloc_bytes = alloc_bytes;
  res.threads = threads_per_block;
  res.blocks = blocks;
  res.iterations = iterations;
  res.fail_count = h_fail;

  compute_stats(h_malloc, clock_khz,
                &res.avg_malloc_us, &res.median_malloc_us, &res.p99_malloc_us);
  compute_stats(h_free, clock_khz,
                &res.avg_free_us, &res.median_free_us, &res.p99_free_us);

  CUDA_CHECK(cudaFree(d_malloc_cycles));
  CUDA_CHECK(cudaFree(d_free_cycles));
  CUDA_CHECK(cudaFree(d_fail_count));

  return res;
}

#if HSHM_IS_HOST
#include <hermes_shm/util/logging.h>

static void set_device_heap_once() {
  // Set the device heap to the maximum we'll need across all test cases.
  // Must be called before any device malloc kernel launch.
  // We use up to half of free GPU memory, capped at 2 GB.
  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t heap_size = std::min((size_t)(free_mem * 0.5),
                              (size_t)(2ULL * 1024 * 1024 * 1024));
  heap_size = std::max(heap_size, (size_t)(64 * 1024 * 1024));
  cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Warning: cudaDeviceSetLimit(%zu) failed: %s\n",
            heap_size, cudaGetErrorString(err));
  } else {
    printf("║  Device heap set to: %zu bytes (%.0f MB)                                         ║\n",
           heap_size, (double)heap_size / (1024.0 * 1024.0));
  }
}

int run_cuda_malloc_bench(uint32_t client_blocks, uint32_t client_threads,
                          uint32_t iterations) {
  printf("\n");
  printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
  printf("║                           CUDA Device malloc/free Microbenchmark                               ║\n");
  printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n");

  // Print device heap info
  size_t heap_size = 0;
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  printf("║  Default device heap size: %zu bytes (%.1f MB)                                      ║\n",
         heap_size, (double)heap_size / (1024.0 * 1024.0));

  int device = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("║  GPU: %-40s  Clock: %d MHz                   ║\n",
         prop.name, prop.clockRate / 1000);

  // Set device heap once before any kernel launches
  set_device_heap_once();

  printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n");

  // Allocation sizes to test
  struct AllocSpec {
    uint64_t bytes;
    const char *label;
  };
  AllocSpec alloc_sizes[] = {
      {512,                   "512 B"},
      {16 * 1024,             "16 KB"},
      {32 * 1024 * 1024,      "32 MB"},
  };

  // Thread counts to sweep (showing parallelism impact)
  struct ParallelSpec {
    uint32_t blocks;
    uint32_t threads;
    const char *label;
  };
  ParallelSpec par_configs[] = {
      {1,  1,   "1 thread"},
      {1,  32,  "1 warp (32t)"},
      {1,  128, "4 warps (128t)"},
      {1,  256, "8 warps (256t)"},
      {4,  256, "32 warps (1024t)"},
  };

  // If user specified custom parallelism, also add that
  bool custom_added = false;
  for (auto &pc : par_configs) {
    if (pc.blocks == client_blocks && pc.threads == client_threads) {
      custom_added = true;
      break;
    }
  }

  printf("║                                                                                                ║\n");
  printf("║  Testing each (alloc_size x parallelism) combination                                           ║\n");
  printf("║  Metrics: avg / median / p99 latency in microseconds                                           ║\n");
  printf("║                                                                                                ║\n");
  printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n");

  for (auto &as : alloc_sizes) {
    printf("║                                                                                                ║\n");
    printf("║  ── Allocation size: %-10s ──────────────────────────────────────────────────────────────── ║\n", as.label);
    printf("║  %-22s │ %12s %12s %12s │ %12s %12s %12s │ %5s ║\n",
           "Parallelism", "malloc avg", "median", "p99",
           "free avg", "median", "p99", "fails");
    printf("║  ──────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┼─────  ║\n");

    for (auto &pc : par_configs) {
      // For 32MB allocs, skip high parallelism (would need >32GB heap)
      if (as.bytes >= 32 * 1024 * 1024 && pc.blocks * pc.threads > 32) {
        // Reduce to feasible thread count
        uint32_t max_threads = 2048ULL * 1024 * 1024 / (as.bytes * 2);
        if (pc.blocks * pc.threads > max_threads) {
          printf("║  %-22s │ %52s │ %5s ║\n",
                 pc.label, "(skipped: would exceed 2GB heap limit)", "—");
          continue;
        }
      }

      auto res = run_single(as.bytes, pc.blocks, pc.threads, iterations);

      printf("║  %-22s │ %9.1f us %9.1f us %9.1f us │ %9.1f us %9.1f us %9.1f us │ %5d ║\n",
             pc.label,
             res.avg_malloc_us, res.median_malloc_us, res.p99_malloc_us,
             res.avg_free_us, res.median_free_us, res.p99_free_us,
             res.fail_count);
    }

    // Run with custom config if different from presets
    if (!custom_added && (client_blocks != 1 || client_threads != 32)) {
      char custom_label[64];
      snprintf(custom_label, sizeof(custom_label), "custom (%ux%u)",
               client_blocks, client_threads);
      auto res = run_single(as.bytes, client_blocks, client_threads, iterations);
      printf("║  %-22s │ %9.1f us %9.1f us %9.1f us │ %9.1f us %9.1f us %9.1f us │ %5d ║\n",
             custom_label,
             res.avg_malloc_us, res.median_malloc_us, res.p99_malloc_us,
             res.avg_free_us, res.median_free_us, res.p99_free_us,
             res.fail_count);
    }
  }

  printf("║                                                                                                ║\n");
  printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
  printf("║  Notes:                                                                                        ║\n");
  printf("║  - Device malloc uses a global free-list with internal locking                                 ║\n");
  printf("║  - Default device heap = 8 MB; enlarged via cudaDeviceSetLimit(cudaLimitMallocHeapSize)        ║\n");
  printf("║  - 32 MB allocs require heap > 8 MB default; we set heap = max(2x total, 64 MB, cap 2 GB)     ║\n");
  printf("║  - High parallelism + large allocs may fail if heap is exhausted                               ║\n");
  printf("║  - Cycles measured with clock64(); converted to us via GPU clock rate                          ║\n");
  printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

  return 0;
}

#endif  // HSHM_IS_HOST
