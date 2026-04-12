/**
 * bench_bam_vs_baselines.cu -- BaM vs Direct vs cudaMemcpy comparison
 *
 * Standalone GPU microbenchmark comparing BaM page cache performance
 * against direct GPU→DRAM memcpy and cudaMemcpyAsync baselines.
 *
 * Memory hierarchy:
 *   GPU HBM (VRAM)  ←→  Host DRAM (pinned cudaMallocHost)
 *
 * Test modes:
 *   bam_read     — GPU reads through BaM HBM page cache from DRAM
 *   bam_write    — GPU writes through BaM HBM page cache to DRAM
 *   direct_read  — GPU reads directly from pinned DRAM (no cache)
 *   direct_write — GPU writes directly to pinned DRAM (no cache)
 *   cudamemcpy   — cudaMemcpyAsync D2H/H2D (theoretical PCIe max)
 *
 * Usage:
 *   bench_bam_vs_baselines [--warps N] [--io-size BYTES] [--page-size BYTES]
 *                          [--cache-pages N] [--iterations N]
 */
#include <bam/bam.h>
#include <bam/array.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
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

/* ================================================================== */
/* GPU Kernels                                                         */
/* ================================================================== */

/** BaM read: each thread reads 4 bytes through the page cache. */
__global__ void bam_read_kernel(
    bam::ArrayDevice<char> arr,
    uint64_t total_bytes,
    int *d_done) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  uint64_t n_words = total_bytes / 4;
  volatile uint32_t sink = 0;

  for (uint64_t i = tid; i < n_words; i += stride) {
    uint64_t byte_off = i * 4;
    uint64_t page_off = byte_off & ~((uint64_t)arr.cache_state.page_size - 1);
    uint32_t in_page  = (uint32_t)(byte_off & ((uint64_t)arr.cache_state.page_size - 1));

    bool needs_load;
    uint8_t *page = bam::page_cache_acquire(
        const_cast<bam::PageCacheDeviceState &>(arr.cache_state),
        page_off, &needs_load);

    if (needs_load) {
      bam::host_read_page(page, arr.host_base, page_off,
                          arr.cache_state.page_size);
      bam::page_cache_finish_load(
          const_cast<bam::PageCacheDeviceState &>(arr.cache_state), page_off);
    }

    sink += *reinterpret_cast<const uint32_t *>(page + in_page);
  }

  if (sink == 0xDEADBEEF && tid == 0) atomicAdd(d_done, -1);
  atomicAdd(d_done, 1);
  __threadfence_system();
}

/** BaM write: each thread writes 4 bytes through the page cache + writeback. */
__global__ void bam_write_kernel(
    bam::ArrayDevice<char> arr,
    uint64_t total_bytes,
    int *d_done) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  uint64_t n_words = total_bytes / 4;

  for (uint64_t i = tid; i < n_words; i += stride) {
    uint64_t byte_off = i * 4;
    uint64_t page_off = byte_off & ~((uint64_t)arr.cache_state.page_size - 1);
    uint32_t in_page  = (uint32_t)(byte_off & ((uint64_t)arr.cache_state.page_size - 1));

    bool needs_load;
    uint8_t *page = bam::page_cache_acquire(
        const_cast<bam::PageCacheDeviceState &>(arr.cache_state),
        page_off, &needs_load);

    if (needs_load) {
      bam::host_read_page(page, arr.host_base, page_off,
                          arr.cache_state.page_size);
      bam::page_cache_finish_load(
          const_cast<bam::PageCacheDeviceState &>(arr.cache_state), page_off);
    }

    *reinterpret_cast<uint32_t *>(page + in_page) = (uint32_t)(tid + i);

    bam::host_write_page(page, const_cast<uint8_t *>(arr.host_base),
                         page_off, arr.cache_state.page_size);
  }

  atomicAdd(d_done, 1);
  __threadfence_system();
}

/* ================================================================== */
/* Warp-cooperative BaM kernels (inspired by GIDS)                     */
/* ================================================================== */

/**
 * Warp-cooperative BaM read.
 * Each warp processes pages together: lane 0 acquires, all 32 lanes copy.
 * After the page is in HBM, all lanes read their elements from it.
 */
__global__ void bam_warp_read_kernel(
    bam::ArrayDevice<char> arr,
    uint64_t total_bytes,
    int *d_done) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint32_t page_size = arr.cache_state.page_size;
  uint64_t num_pages = (total_bytes + page_size - 1) / page_size;

  volatile uint32_t sink = 0;

  // Each warp processes pages in a strided pattern
  for (uint64_t p = warp_id; p < num_pages; p += num_warps) {
    uint64_t page_off = p * page_size;

    // Warp-cooperative acquire: lane 0 does atomic, broadcasts result
    bool needs_load;
    uint8_t *page = bam::warp_page_cache_acquire(
        arr.cache_state, page_off, &needs_load);

    if (needs_load) {
      // All 32 lanes cooperate on page copy (GIDS-style)
      bam::warp_host_read_page(page, arr.host_base, page_off, page_size);
      bam::warp_page_cache_finish_load(arr.cache_state, page_off);
    }

    // All lanes read their slice of the page
    uint32_t words_per_page = page_size / 4;
    for (uint32_t w = lane_id; w < words_per_page; w += 32) {
      sink += reinterpret_cast<const uint32_t *>(page)[w];
    }
    __syncwarp();
  }

  if (sink == 0xDEADBEEF && lane_id == 0) atomicAdd(d_done, -1);
  if (lane_id == 0) {
    atomicAdd(d_done, 1);
    __threadfence_system();
  }
}

/**
 * Warp-cooperative BaM write.
 * Each warp loads a page cooperatively, all lanes write their slice,
 * then all lanes cooperatively flush the page back to DRAM.
 * No per-element write-through — one flush per page.
 */
__global__ void bam_warp_write_kernel(
    bam::ArrayDevice<char> arr,
    uint64_t total_bytes,
    int *d_done) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint32_t page_size = arr.cache_state.page_size;
  uint64_t num_pages = (total_bytes + page_size - 1) / page_size;

  for (uint64_t p = warp_id; p < num_pages; p += num_warps) {
    uint64_t page_off = p * page_size;

    bool needs_load;
    uint8_t *page = bam::warp_page_cache_acquire(
        arr.cache_state, page_off, &needs_load);

    if (needs_load) {
      bam::warp_host_read_page(page, arr.host_base, page_off, page_size);
      bam::warp_page_cache_finish_load(arr.cache_state, page_off);
    }

    // All lanes write their slice of the page
    uint32_t words_per_page = page_size / 4;
    for (uint32_t w = lane_id; w < words_per_page; w += 32) {
      reinterpret_cast<uint32_t *>(page)[w] = warp_id + w;
    }
    __syncwarp();

    // One cooperative flush per page (not per element)
    bam::warp_host_write_page(page, const_cast<uint8_t *>(arr.host_base),
                              page_off, page_size);
    if (lane_id == 0) {
      bam::page_cache_mark_dirty(arr.cache_state, page_off);
    }
    __syncwarp();
  }

  if (lane_id == 0) {
    atomicAdd(d_done, 1);
    __threadfence_system();
  }
}

/* ================================================================== */
/* Baseline kernels                                                    */
/* ================================================================== */

/** Direct read: GPU reads from pinned host DRAM, no cache. */
__global__ void direct_read_kernel(
    const char *h_src,
    uint64_t total_bytes,
    int *d_done) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  uint64_t n_words = total_bytes / 4;
  const uint32_t *src4 = reinterpret_cast<const uint32_t *>(h_src);

  volatile uint32_t sink = 0;
  for (uint64_t i = tid; i < n_words; i += stride) {
    sink += src4[i];
  }

  if (sink == 0xDEADBEEF && tid == 0) atomicAdd(d_done, -1);
  atomicAdd(d_done, 1);
  __threadfence_system();
}

/** Direct write: GPU writes to pinned host DRAM, no cache. */
__global__ void direct_write_kernel(
    const char *d_src,
    char *h_dst,
    uint64_t total_bytes,
    int *d_done) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  uint64_t n_words = total_bytes / 4;
  const uint32_t *src4 = reinterpret_cast<const uint32_t *>(d_src);
  uint32_t *dst4 = reinterpret_cast<uint32_t *>(h_dst);

  for (uint64_t i = tid; i < n_words; i += stride) {
    dst4[i] = src4[i];
  }

  atomicAdd(d_done, 1);
  __threadfence_system();
}

/* ================================================================== */
/* Benchmark runner                                                    */
/* ================================================================== */

struct BenchResult {
  const char *name;
  double elapsed_ms;
  double bandwidth_gbps;
  uint64_t total_bytes;
};

static void poll_done(int *d_done, int expected) {
  while (__atomic_load_n(d_done, __ATOMIC_ACQUIRE) < expected) {}
}

static BenchResult run_bam_read(uint32_t blocks, uint32_t threads,
                                 uint64_t total_bytes, uint64_t page_size,
                                 uint32_t cache_pages, int iters) {
  BenchResult r = {"bam_read", 0, 0, total_bytes};

  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = cache_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  bam::Array<char> arr(total_bytes, cache);

  std::vector<char> host_data(total_bytes);
  for (uint64_t i = 0; i < total_bytes; i++) host_data[i] = (char)(i & 0xFF);
  arr.load_from_host(host_data.data(), total_bytes);

  int *d_done;
  CUDA_CHECK(cudaMallocHost(&d_done, sizeof(int)));
  uint32_t total_threads = blocks * threads;

  // Warmup
  *d_done = 0;
  bam_read_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    // Reset page cache tags to force misses each iteration
    CUDA_CHECK(cudaMemset(cache.device_state().page_tags, 0xFF,
                           cache_pages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(cache.device_state().page_states, 0,
                           cache_pages * sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    *d_done = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    bam_read_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
    poll_done(d_done, total_threads);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaFreeHost(d_done));
  return r;
}

static BenchResult run_bam_write(uint32_t blocks, uint32_t threads,
                                  uint64_t total_bytes, uint64_t page_size,
                                  uint32_t cache_pages, int iters) {
  BenchResult r = {"bam_write", 0, 0, total_bytes};

  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = cache_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  bam::Array<char> arr(total_bytes, cache);
  std::vector<char> zeros(total_bytes, 0);
  arr.load_from_host(zeros.data(), total_bytes);

  int *d_done;
  CUDA_CHECK(cudaMallocHost(&d_done, sizeof(int)));
  uint32_t total_threads = blocks * threads;

  // Warmup
  *d_done = 0;
  bam_write_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    CUDA_CHECK(cudaMemset(cache.device_state().page_tags, 0xFF,
                           cache_pages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(cache.device_state().page_states, 0,
                           cache_pages * sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    *d_done = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    bam_write_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
    poll_done(d_done, total_threads);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaFreeHost(d_done));
  return r;
}

static BenchResult run_bam_warp_read(uint32_t blocks, uint32_t threads,
                                      uint64_t total_bytes, uint64_t page_size,
                                      uint32_t cache_pages, int iters) {
  BenchResult r = {"bam_warp_read", 0, 0, total_bytes};

  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = cache_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  bam::Array<char> arr(total_bytes, cache);

  std::vector<char> host_data(total_bytes);
  for (uint64_t i = 0; i < total_bytes; i++) host_data[i] = (char)(i & 0xFF);
  arr.load_from_host(host_data.data(), total_bytes);

  int *d_done;
  CUDA_CHECK(cudaMallocHost(&d_done, sizeof(int)));
  uint32_t num_warps = (blocks * threads) / 32;

  // Warmup
  *d_done = 0;
  bam_warp_read_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    CUDA_CHECK(cudaMemset(cache.device_state().page_tags, 0xFF,
                           cache_pages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(cache.device_state().page_states, 0,
                           cache_pages * sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    *d_done = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    bam_warp_read_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
    poll_done(d_done, num_warps);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaFreeHost(d_done));
  return r;
}

static BenchResult run_bam_warp_write(uint32_t blocks, uint32_t threads,
                                       uint64_t total_bytes, uint64_t page_size,
                                       uint32_t cache_pages, int iters) {
  BenchResult r = {"bam_warp_write", 0, 0, total_bytes};

  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = cache_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  bam::Array<char> arr(total_bytes, cache);
  std::vector<char> zeros(total_bytes, 0);
  arr.load_from_host(zeros.data(), total_bytes);

  int *d_done;
  CUDA_CHECK(cudaMallocHost(&d_done, sizeof(int)));
  uint32_t num_warps = (blocks * threads) / 32;

  // Warmup
  *d_done = 0;
  bam_warp_write_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    CUDA_CHECK(cudaMemset(cache.device_state().page_tags, 0xFF,
                           cache_pages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(cache.device_state().page_states, 0,
                           cache_pages * sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    *d_done = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    bam_warp_write_kernel<<<blocks, threads>>>(arr.device(), total_bytes, d_done);
    poll_done(d_done, num_warps);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaFreeHost(d_done));
  return r;
}

static BenchResult run_direct_read(uint32_t blocks, uint32_t threads,
                                    uint64_t total_bytes, int iters) {
  BenchResult r = {"direct_read", 0, 0, total_bytes};

  char *h_src;
  CUDA_CHECK(cudaMallocHost(&h_src, total_bytes));
  memset(h_src, 0xAB, total_bytes);

  int *d_done;
  CUDA_CHECK(cudaMallocHost(&d_done, sizeof(int)));
  uint32_t total_threads = blocks * threads;

  // Warmup
  *d_done = 0;
  direct_read_kernel<<<blocks, threads>>>(h_src, total_bytes, d_done);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    *d_done = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    direct_read_kernel<<<blocks, threads>>>(h_src, total_bytes, d_done);
    poll_done(d_done, total_threads);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaFreeHost(h_src));
  CUDA_CHECK(cudaFreeHost(d_done));
  return r;
}

static BenchResult run_direct_write(uint32_t blocks, uint32_t threads,
                                     uint64_t total_bytes, int iters) {
  BenchResult r = {"direct_write", 0, 0, total_bytes};

  char *d_src;
  CUDA_CHECK(cudaMalloc(&d_src, total_bytes));
  CUDA_CHECK(cudaMemset(d_src, 0xAB, total_bytes));

  char *h_dst;
  CUDA_CHECK(cudaMallocHost(&h_dst, total_bytes));

  int *d_done;
  CUDA_CHECK(cudaMallocHost(&d_done, sizeof(int)));
  uint32_t total_threads = blocks * threads;

  // Warmup
  *d_done = 0;
  direct_write_kernel<<<blocks, threads>>>(d_src, h_dst, total_bytes, d_done);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    *d_done = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    direct_write_kernel<<<blocks, threads>>>(d_src, h_dst, total_bytes, d_done);
    poll_done(d_done, total_threads);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFreeHost(h_dst));
  CUDA_CHECK(cudaFreeHost(d_done));
  return r;
}

static BenchResult run_cudamemcpy_d2h(uint64_t total_bytes, int iters) {
  BenchResult r = {"cudaMemcpy_D2H", 0, 0, total_bytes};

  char *d_src;
  CUDA_CHECK(cudaMalloc(&d_src, total_bytes));
  CUDA_CHECK(cudaMemset(d_src, 0xAB, total_bytes));

  char *h_dst;
  CUDA_CHECK(cudaMallocHost(&h_dst, total_bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Warmup
  CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, total_bytes,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, total_bytes,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(h_dst));
  CUDA_CHECK(cudaFree(d_src));
  return r;
}

static BenchResult run_cudamemcpy_h2d(uint64_t total_bytes, int iters) {
  BenchResult r = {"cudaMemcpy_H2D", 0, 0, total_bytes};

  char *h_src;
  CUDA_CHECK(cudaMallocHost(&h_src, total_bytes));
  memset(h_src, 0xAB, total_bytes);

  char *d_dst;
  CUDA_CHECK(cudaMalloc(&d_dst, total_bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Warmup
  CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, total_bytes,
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double total_ms = 0;
  for (int it = 0; it < iters; it++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, total_bytes,
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / iters;
  r.bandwidth_gbps = (total_bytes / 1e9) / (r.elapsed_ms / 1e3);
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(h_src));
  CUDA_CHECK(cudaFree(d_dst));
  return r;
}

/* ================================================================== */
/* CLI + Main                                                          */
/* ================================================================== */

static uint64_t parse_size(const char *s) {
  double val = atof(s);
  const char *p = s;
  while (*p && (isdigit(*p) || *p == '.')) p++;
  switch (tolower(*p)) {
    case 'k': return (uint64_t)(val * 1024);
    case 'm': return (uint64_t)(val * 1024 * 1024);
    case 'g': return (uint64_t)(val * 1024 * 1024 * 1024);
    default:  return (uint64_t)val;
  }
}

int main(int argc, char **argv) {
  // Defaults: 32 warps, 128KB I/O, 64KB pages, 256 cache pages
  uint32_t warps = 32;
  uint64_t io_size = 128 * 1024;
  uint64_t page_size = 64 * 1024;
  uint32_t cache_pages = 256;
  int iterations = 5;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--warps" && i + 1 < argc) warps = atoi(argv[++i]);
    else if (arg == "--io-size" && i + 1 < argc) io_size = parse_size(argv[++i]);
    else if (arg == "--page-size" && i + 1 < argc) page_size = parse_size(argv[++i]);
    else if (arg == "--cache-pages" && i + 1 < argc) cache_pages = atoi(argv[++i]);
    else if (arg == "--iterations" && i + 1 < argc) iterations = atoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s [--warps N] [--io-size BYTES] [--page-size BYTES]\n"
             "           [--cache-pages N] [--iterations N]\n", argv[0]);
      return 0;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  uint32_t threads_per_block = 256;
  uint32_t total_threads = warps * 32;
  uint32_t blocks = (total_threads + threads_per_block - 1) / threads_per_block;
  uint64_t total_bytes = io_size;

  printf("============================================================\n");
  printf("  BaM vs Baselines: GPU <-> DRAM Microbenchmark\n");
  printf("============================================================\n");
  printf("GPU:           %s\n", prop.name);
  printf("Warps:         %u (%u threads, %u blocks x %u threads)\n",
         warps, total_threads, blocks, threads_per_block);
  printf("I/O size:      %lu bytes (%.1f KB / %.1f MB)\n",
         (unsigned long)total_bytes, total_bytes / 1024.0,
         total_bytes / (1024.0 * 1024.0));
  printf("BaM page size: %lu bytes (%.1f KB)\n",
         (unsigned long)page_size, page_size / 1024.0);
  printf("BaM cache:     %u pages (%.1f KB in HBM)\n",
         cache_pages, (cache_pages * page_size) / 1024.0);
  printf("Iterations:    %d (averaged)\n", iterations);
  printf("------------------------------------------------------------\n\n");

  std::vector<BenchResult> results;

  // ------------------------------------------------------------------
  // BaM: GPU-initiated DRAM↔HBM via software page cache
  // Comparable to CTE putblob_gpu (GPU writes data → runtime moves to DRAM)
  // ------------------------------------------------------------------
  printf("Running bam_warp_read  (DRAM→HBM, warp-coop page cache)...\n");
  results.push_back(run_bam_warp_read(blocks, threads_per_block, total_bytes,
                                       page_size, cache_pages, iterations));

  printf("Running bam_warp_write (HBM→DRAM, warp-coop page cache)...\n");
  results.push_back(run_bam_warp_write(blocks, threads_per_block, total_bytes,
                                        page_size, cache_pages, iterations));

  // ------------------------------------------------------------------
  // CTE-equivalent baselines (no Chimaera runtime needed)
  //   direct_read  ≈ CTE "direct" baseline (GPU reads pinned DRAM)
  //   direct_write ≈ CTE putblob "direct" (GPU writes to pinned DRAM)
  //   cudaMemcpy   ≈ CTE theoretical PCIe max
  // ------------------------------------------------------------------
  printf("Running direct_read    (GPU reads pinned DRAM, CTE baseline)...\n");
  results.push_back(run_direct_read(blocks, threads_per_block, total_bytes,
                                     iterations));

  printf("Running direct_write   (GPU writes pinned DRAM, CTE baseline)...\n");
  results.push_back(run_direct_write(blocks, threads_per_block, total_bytes,
                                      iterations));

  printf("Running cudaMemcpy D2H (PCIe max, CTE ceiling)...\n");
  results.push_back(run_cudamemcpy_d2h(total_bytes, iterations));

  printf("Running cudaMemcpy H2D (PCIe max, CTE ceiling)...\n");
  results.push_back(run_cudamemcpy_h2d(total_bytes, iterations));

  // Old single-thread BaM skipped (too slow at large I/O sizes)

  // Print results table
  printf("\n============================================================\n");
  printf("  Results (I/O = %.1f KB, %u warps)\n",
         total_bytes / 1024.0, warps);
  printf("============================================================\n");
  printf("  BaM = GPU-initiated via HBM page cache\n");
  printf("  CTE-equiv = direct GPU↔DRAM (what CTE baselines measure)\n");
  printf("------------------------------------------------------------\n");
  printf("%-18s  %10s  %10s  %s\n", "Method", "Time (ms)", "BW (GB/s)", "Category");
  printf("%-18s  %10s  %10s  %s\n", "------", "---------", "---------", "--------");

  const char *cats[] = {
    "BaM", "BaM",
    "CTE-equiv", "CTE-equiv", "CTE-equiv", "CTE-equiv",
  };
  for (size_t i = 0; i < results.size(); i++) {
    printf("%-18s  %10.3f  %10.3f  %s\n",
           results[i].name, results[i].elapsed_ms,
           results[i].bandwidth_gbps,
           i < 6 ? cats[i] : "");
  }
  printf("============================================================\n");

  return 0;
}
