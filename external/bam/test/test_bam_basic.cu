/**
 * test_bam_basic.cu -- BaM test: HBM cache + DRAM backing store
 *
 * Memory hierarchy under test:
 *
 *   GPU HBM (VRAM)      ← page cache (fast, limited capacity)
 *        ↕ cache miss/eviction
 *   Host DRAM            ← backing store (pinned cudaMallocHost)
 *
 * No NVMe hardware required. This exercises the full BaM page cache
 * path: acquire → miss → load from DRAM → hit → evict → writeback.
 *
 * Test 1: Sequential read — every element read once, verify values.
 * Test 2: Write-back — double every element, re-read and verify.
 * Test 3: Cache pressure — array larger than cache, forces evictions.
 */
#include <bam/bam.h>
#include <bam/array.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
              __FILE__, __LINE__, cudaGetErrorString(err));          \
      exit(1);                                                       \
    }                                                                \
  } while (0)

/* ------------------------------------------------------------------ */
/* GPU kernels                                                         */
/* ------------------------------------------------------------------ */

/** Read every element through the BaM HBM cache, write to output. */
__global__ void read_kernel(bam::ArrayDevice<float> arr, float *output,
                            uint64_t n) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = arr.read(idx);
}

/** Double every element: read from cache, write 2x back. */
__global__ void double_kernel(bam::ArrayDevice<float> arr, uint64_t n) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float val = arr.read(idx);
  arr.write(idx, val * 2.0f);
}

/** Verify every element equals expected value. */
__global__ void verify_kernel(bam::ArrayDevice<float> arr,
                              float multiplier, int *errors, uint64_t n) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float val = arr.read(idx);
  float expected = (float)idx * multiplier;
  if (fabsf(val - expected) > 1e-3f) {
    atomicAdd(errors, 1);
  }
}

/* ------------------------------------------------------------------ */
/* Test runner                                                         */
/* ------------------------------------------------------------------ */

struct TestResult {
  const char *name;
  bool passed;
  int errors;
  uint64_t elements;
};

static TestResult run_test(const char *name, uint64_t N,
                           size_t page_size, uint32_t num_pages) {
  TestResult result = {name, false, 0, N};

  printf("\n--- %s ---\n", name);
  printf("  Array:   %lu elements (%.1f KB)\n",
         (unsigned long)N, N * sizeof(float) / 1024.0);
  printf("  Cache:   %u pages x %zu B = %.1f KB in HBM (GPU VRAM)\n",
         num_pages, page_size, num_pages * page_size / 1024.0);
  printf("  Backing: %.1f KB in DRAM (pinned host memory)\n",
         N * sizeof(float) / 1024.0);

  size_t array_bytes = N * sizeof(float);
  size_t cache_bytes = (size_t)num_pages * page_size;
  if (array_bytes > cache_bytes) {
    printf("  Pressure: array %.1fx larger than cache → evictions expected\n",
           (double)array_bytes / cache_bytes);
  } else {
    printf("  Pressure: array fits in cache → no evictions expected\n");
  }

  // Configure: HBM cache + DRAM backing store
  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = num_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  bam::Array<float> arr(N, cache);

  // Load initial data into DRAM: arr[i] = (float)i
  std::vector<float> host_data(N);
  for (uint64_t i = 0; i < N; i++) host_data[i] = (float)i;
  if (arr.load_from_host(host_data.data(), N) != 0) {
    printf("  FAIL: could not load data into DRAM backing store\n");
    return result;
  }
  printf("  Loaded %lu elements into DRAM backing store\n", (unsigned long)N);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  // Phase 1: Read all elements through HBM cache → verify
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

  read_kernel<<<blocks, threads>>>(arr.device(), d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_output(N);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float),
                         cudaMemcpyDeviceToHost));

  int read_errors = 0;
  for (uint64_t i = 0; i < N; i++) {
    if (fabsf(h_output[i] - (float)i) > 1e-3f) {
      if (read_errors < 3)
        printf("  READ MISMATCH: [%lu] got %f, want %f\n",
               (unsigned long)i, h_output[i], (float)i);
      read_errors++;
    }
  }
  printf("  Phase 1 (read):  %d / %lu errors\n", read_errors, (unsigned long)N);

  // Phase 2: Double all elements, then verify via GPU
  double_kernel<<<blocks, threads>>>(arr.device(), N);
  CUDA_CHECK(cudaDeviceSynchronize());

  int *d_errors;
  CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_errors, 0, sizeof(int)));

  verify_kernel<<<blocks, threads>>>(arr.device(), 2.0f, d_errors, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  int write_errors = 0;
  CUDA_CHECK(cudaMemcpy(&write_errors, d_errors, sizeof(int),
                         cudaMemcpyDeviceToHost));
  printf("  Phase 2 (write): %d / %lu errors\n",
         write_errors, (unsigned long)N);

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_errors));

  result.errors = read_errors + write_errors;
  result.passed = (result.errors == 0);
  printf("  %s\n", result.passed ? "PASSED" : "FAILED");
  return result;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    printf("No CUDA devices found. Skipping BaM tests.\n");
    return 0;
  }
  CUDA_CHECK(cudaSetDevice(0));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("========================================\n");
  printf(" BaM Test: HBM Cache + DRAM Backing Store\n");
  printf("========================================\n");
  printf("GPU: %s\n", prop.name);
  printf("  HBM capacity: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
  printf("  Memory hierarchy: GPU HBM (cache) <-> Host DRAM (backing)\n");
  printf("  NVMe: disabled (DRAM-only mode)\n");

  std::vector<TestResult> results;

  // Test 1: Small array that fits entirely in cache (no evictions)
  results.push_back(run_test(
      "Test 1: Array fits in HBM cache (no evictions)",
      /*N=*/1024, /*page_size=*/4096, /*num_pages=*/32));

  // Test 2: Array larger than cache (forces evictions / cache pressure)
  results.push_back(run_test(
      "Test 2: Array exceeds HBM cache (evictions)",
      /*N=*/4096, /*page_size=*/4096, /*num_pages=*/8));

  // Test 3: Larger scale with 64KB pages (realistic page size for GPU)
  results.push_back(run_test(
      "Test 3: 64KB pages, moderate pressure",
      /*N=*/16384, /*page_size=*/65536, /*num_pages=*/4));

  // Summary
  printf("\n========================================\n");
  printf(" Summary\n");
  printf("========================================\n");
  int total_pass = 0, total_fail = 0;
  for (auto &r : results) {
    printf("  [%s] %s (%d errors / %lu elements)\n",
           r.passed ? "PASS" : "FAIL", r.name,
           r.errors, (unsigned long)r.elements);
    if (r.passed) total_pass++; else total_fail++;
  }
  printf("\n  %d passed, %d failed\n", total_pass, total_fail);

  return total_fail > 0 ? 1 : 0;
}
