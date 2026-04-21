/**
 * CUDA Dynamic Parallelism (CDP) launch overhead benchmark.
 *
 * Compiled with NVCC (not Clang CUDA) because CDP requires NVCC's
 * support for <<<>>> syntax inside __global__ functions.
 *
 * Test cases:
 *   cdp      - Device-side child kernel launch + cudaDeviceSynchronize
 *              Reports submit (launch) and sync times separately.
 *   dispatch - Host launches a kernel that memsets pinned host segments,
 *              one per lane. Measures host-side launch+sync overhead
 *              for comparison with CDP.
 *
 * Usage:
 *   bench_cdp [--test cdp|dispatch] [--blocks N] [--threads N] [--tasks N]
 *             [--depth N]
 *
 *   --depth N   Number of kernel launches to issue before synchronizing
 *               (default: 1).  E.g. --depth 4 launches 4 child kernels
 *               back-to-back, then syncs once.
 */

// CUDA_FORCE_CDP1_IF_SUPPORTED is defined via CMakeLists.txt to enable
// cudaDeviceSynchronize in device code (CDP2 removes it by default).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

// ============================================================================
// CDP test: device-side child kernel launch + sync
// ============================================================================

__global__ void cdp_child_kernel() {
  // Trivial child — just exists to be launched and retired
}

__global__ void cdp_parent_kernel(
    unsigned int child_blocks,
    unsigned int child_threads,
    unsigned int total_tasks,
    unsigned int depth,
    long long *d_results) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  long long t_submit = 0, t_sync = 0;
  long long tc;

  // Warmup
  for (unsigned int d = 0; d < depth; ++d)
    cdp_child_kernel<<<child_blocks, child_threads>>>();
  cudaDeviceSynchronize();

  unsigned int total_launches = total_tasks * depth;

  for (unsigned int i = 0; i < total_tasks; ++i) {
    tc = clock64();
    for (unsigned int d = 0; d < depth; ++d)
      cdp_child_kernel<<<child_blocks, child_threads>>>();
    t_submit += clock64() - tc;

    tc = clock64();
    cudaDeviceSynchronize();
    t_sync += clock64() - tc;
  }

  d_results[0] = t_submit;
  d_results[1] = t_sync;

  printf("=== CDP Device-Side Latency (%u rounds x %u depth = %u launches, "
         "%u blocks x %u threads) ===\n",
         total_tasks, depth, total_launches, child_blocks, child_threads);
  printf("  Submit:       %lld total  (%lld/round)  (%lld/launch)\n",
         t_submit, t_submit / total_tasks, t_submit / total_launches);
  printf("  Sync:         %lld total  (%lld/round)\n",
         t_sync, t_sync / total_tasks);
  long long total = t_submit + t_sync;
  printf("  Total:        %lld total  (%lld/round)  (%lld/launch)\n",
         total, total / total_tasks, total / total_launches);
}

static int run_cdp(unsigned int blocks, unsigned int threads,
                   unsigned int tasks, unsigned int depth) {
  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 4096);

  long long *d_results;
  cudaMalloc(&d_results, 2 * sizeof(long long));
  cudaMemset(d_results, 0, 2 * sizeof(long long));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cdp_parent_kernel<<<1, 1>>>(blocks, threads, tasks, depth, d_results);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CDP kernel error: %s\n", cudaGetErrorString(err));
    cudaFree(d_results);
    return 1;
  }

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  unsigned int total_launches = tasks * depth;
  printf("\n  Host wall:    %.3f ms\n", ms);
  printf("  Avg latency:  %.3f us/round  %.3f us/launch (wall)\n",
         (ms * 1000.0f) / tasks, (ms * 1000.0f) / total_launches);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_results);
  return 0;
}

// ============================================================================
// Dispatch test: host launches kernel, each lane memsets its pinned segment
// ============================================================================

/**
 * Each thread zeroes its own segment of pinned host memory.
 * total_lanes = blocks * threads. seg_bytes = per-lane I/O size.
 */
__global__ void dispatch_memset_kernel(
    char *pinned_buf, unsigned int total_lanes, unsigned int seg_bytes) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_lanes) return;

  char *seg = pinned_buf + (size_t)tid * seg_bytes;
  for (unsigned int i = 0; i < seg_bytes; i += 4) {
    *reinterpret_cast<unsigned int *>(seg + i) = 0;
  }
  __threadfence_system();
}

static int run_dispatch(unsigned int blocks, unsigned int threads,
                        unsigned int tasks, unsigned int io_size,
                        unsigned int depth) {
  unsigned int total_lanes = blocks * threads;
  // Round up io_size to multiple of 4 for aligned writes
  io_size = (io_size + 3) & ~3u;
  size_t buf_size = (size_t)total_lanes * io_size;

  // Allocate pinned host memory
  char *h_buf;
  cudaMallocHost(&h_buf, buf_size);
  memset(h_buf, 0xFF, buf_size);

  // Warmup
  for (unsigned int i = 0; i < 10; ++i) {
    for (unsigned int d = 0; d < depth; ++d)
      dispatch_memset_kernel<<<blocks, threads>>>(h_buf, total_lanes, io_size);
    cudaDeviceSynchronize();
  }
  memset(h_buf, 0xFF, buf_size);

  unsigned int total_launches = tasks * depth;

  // === Per-round timing: submit vs sync ===
  float sum_submit_us = 0, sum_sync_us = 0;
  float min_us = 1e9f, max_us = 0;
  for (unsigned int i = 0; i < tasks; ++i) {
    memset(h_buf, 0xFF, buf_size);

    cudaEvent_t s0, s1, s2;
    cudaEventCreate(&s0);
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);

    cudaEventRecord(s0);
    for (unsigned int d = 0; d < depth; ++d)
      dispatch_memset_kernel<<<blocks, threads>>>(h_buf, total_lanes, io_size);
    cudaEventRecord(s1);
    cudaEventSynchronize(s1);
    cudaEventRecord(s2);
    cudaEventSynchronize(s2);

    float submit_ms = 0, sync_ms = 0;
    cudaEventElapsedTime(&submit_ms, s0, s1);
    cudaEventElapsedTime(&sync_ms, s1, s2);
    float submit_us = submit_ms * 1000.0f;
    float sync_us = sync_ms * 1000.0f;
    float total_us = submit_us + sync_us;
    sum_submit_us += submit_us;
    sum_sync_us += sync_us;
    if (total_us < min_us) min_us = total_us;
    if (total_us > max_us) max_us = total_us;

    cudaEventDestroy(s0);
    cudaEventDestroy(s1);
    cudaEventDestroy(s2);
  }

  // === Batch wall time ===
  memset(h_buf, 0xFF, buf_size);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (unsigned int i = 0; i < tasks; ++i) {
    for (unsigned int d = 0; d < depth; ++d)
      dispatch_memset_kernel<<<blocks, threads>>>(h_buf, total_lanes, io_size);
    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float batch_ms = 0;
  cudaEventElapsedTime(&batch_ms, start, stop);

  printf("=== Host Dispatch: memset pinned (%u rounds x %u depth = %u launches, "
         "%u blocks x %u threads) ===\n",
         tasks, depth, total_launches, blocks, threads);
  printf("  Lanes:        %u (each zeroes %uB of pinned host mem)\n",
         total_lanes, io_size);
  printf("  Submit:       avg=%.3f us/round  (%.3f us/launch)\n",
         sum_submit_us / tasks, sum_submit_us / total_launches);
  printf("  Sync:         avg=%.3f us/round\n",
         sum_sync_us / tasks);
  printf("  Per-round:    avg=%.3f us  min=%.3f us  max=%.3f us\n",
         (sum_submit_us + sum_sync_us) / tasks, min_us, max_us);
  printf("  Batch wall:   %.3f ms (%.3f us/round, %.3f us/launch)\n",
         batch_ms, (batch_ms * 1000.0f) / tasks,
         (batch_ms * 1000.0f) / total_launches);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFreeHost(h_buf);
  return 0;
}

// ============================================================================
// Main
// ============================================================================

static void print_help(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --test    T   Test case: 'cdp' or 'dispatch' (default: cdp)\n");
  printf("  --blocks  N   Kernel blocks          (default: 1)\n");
  printf("  --threads N   Kernel threads/block    (default: 32)\n");
  printf("  --tasks   N   Number of iterations    (default: 1000)\n");
  printf("  --io-size N   Per-lane I/O bytes [dispatch] (default: 64)\n");
  printf("  --depth   N   Launches per sync round  (default: 1)\n");
}

int main(int argc, char **argv) {
  unsigned int blocks = 1, threads = 32, tasks = 1000, io_size = 64, depth = 1;
  const char *test = "cdp";

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--test") && i + 1 < argc) {
      test = argv[++i];
    } else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) {
      blocks = (unsigned)atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
      threads = (unsigned)atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--tasks") && i + 1 < argc) {
      tasks = (unsigned)atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--io-size") && i + 1 < argc) {
      io_size = (unsigned)atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--depth") && i + 1 < argc) {
      depth = (unsigned)atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      print_help(argv[0]);
      return 0;
    }
  }

  printf("bench_cdp: test=%s, %u blocks x %u threads, %u iterations, depth=%u\n",
         test, blocks, threads, tasks, depth);

  if (!strcmp(test, "cdp")) {
    return run_cdp(blocks, threads, tasks, depth);
  } else if (!strcmp(test, "dispatch")) {
    return run_dispatch(blocks, threads, tasks, io_size, depth);

  } else {
    fprintf(stderr, "Unknown test '%s'; use 'cdp' or 'dispatch'\n", test);
    return 1;
  }
}
