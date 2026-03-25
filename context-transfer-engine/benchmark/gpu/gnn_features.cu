/**
 * gnn_features.cu — GPU GNN feature loading benchmark: BaM vs direct DRAM
 *
 * Workload description:
 *   In Graph Neural Network (GNN) training, each mini-batch requires
 *   gathering node feature vectors for the sampled subgraph. Feature
 *   tensors for large graphs (e.g., IGB-260M: 260M nodes × 1024 floats
 *   = ~1 TB) far exceed GPU memory and must be fetched from DRAM or SSD.
 *
 *   This benchmark simulates the feature gather phase:
 *     - A random mini-batch of node indices is sampled
 *     - For each node, its feature vector (embedding) is read
 *     - Features are gathered into a contiguous output tensor
 *
 *   Two approaches compared:
 *     1. BaM: Features in DRAM, accessed through GPU HBM page cache
 *        (warp-cooperative page loading, GIDS-inspired)
 *     2. Direct: Features in pinned DRAM, GPU reads directly over PCIe
 *
 *   The access pattern is random (depends on mini-batch sampling),
 *   which stresses the page cache with irregular access.
 *
 * Based on: GIDS evaluation/homogenous_train (OSDI'24)
 *
 * Usage:
 *   bench_gpu_gnn_features [--num-nodes N] [--emb-dim D] [--batch-size B]
 *                          [--num-batches N] [--page-size B] [--cache-pages N]
 */
#include <bam/bam.h>
#include <bam/page_cache.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

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
/* GPU kernels                                                         */
/* ================================================================== */

/**
 * GNN feature gather via BaM page cache (GIDS-style).
 *
 * Each warp reads one node's feature vector. Lane 0 acquires the cache
 * page(s), all 32 lanes cooperatively copy features to the output tensor.
 *
 * This mirrors GIDS's read_feature_kernel_with_cpu_backing_memory.
 */
__global__ void gnn_gather_bam_kernel(
    bam::PageCacheDeviceState cache_state,
    const uint8_t *host_features_base,  // Feature store in DRAM
    const uint32_t *d_batch_indices,    // Mini-batch node indices
    float *d_output,                    // Output: gathered features
    uint32_t batch_size,
    uint32_t emb_dim) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  uint32_t page_size = cache_state.page_size;
  uint32_t floats_per_page = page_size / sizeof(float);

  for (uint32_t b = warp_id; b < batch_size; b += num_warps) {
    uint32_t node_id = d_batch_indices[b];
    uint64_t feature_byte_off = (uint64_t)node_id * emb_dim * sizeof(float);
    float *out_row = d_output + (uint64_t)b * emb_dim;

    // Process feature vector page by page (all lanes share the same page)
    for (uint32_t f_base = 0; f_base < emb_dim; f_base += floats_per_page) {
      uint64_t page_off = (feature_byte_off + f_base * sizeof(float))
                          & ~((uint64_t)page_size - 1);

      bool needs_load;
      uint8_t *page = bam::warp_page_cache_acquire(
          cache_state, page_off, &needs_load);

      if (needs_load) {
        bam::warp_host_read_page(page, host_features_base, page_off,
                                  page_size);
        bam::warp_page_cache_finish_load(cache_state, page_off);
      }

      // All 32 lanes read their slice of this page
      uint32_t f_end = (f_base + floats_per_page < emb_dim)
                       ? f_base + floats_per_page : emb_dim;
      for (uint32_t f = f_base + lane_id; f < f_end; f += 32) {
        uint64_t byte_off = feature_byte_off + f * sizeof(float);
        uint32_t in_page = (uint32_t)(byte_off & ((uint64_t)page_size - 1));
        out_row[f] = *reinterpret_cast<const float *>(page + in_page);
      }
      __syncwarp();
    }
  }
}

/**
 * GNN feature gather via direct DRAM access (baseline).
 * Each warp reads one node's features directly from pinned host memory.
 */
__global__ void gnn_gather_direct_kernel(
    const float *h_features,           // Feature store in pinned DRAM
    const uint32_t *d_batch_indices,
    float *d_output,
    uint32_t batch_size,
    uint32_t emb_dim) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  for (uint32_t b = warp_id; b < batch_size; b += num_warps) {
    uint32_t node_id = d_batch_indices[b];
    const float *in_row = h_features + (uint64_t)node_id * emb_dim;
    float *out_row = d_output + (uint64_t)b * emb_dim;

    for (uint32_t f = lane_id; f < emb_dim; f += 32) {
      out_row[f] = in_row[f];
    }
    __syncwarp();
  }
}

/* ================================================================== */
/* Benchmark runner                                                    */
/* ================================================================== */

struct GNNResult {
  const char *name;
  double elapsed_ms;
  double throughput_gbps;
  double features_per_sec;
};

static GNNResult run_gnn_bam(uint32_t num_nodes, uint32_t emb_dim,
                              uint32_t batch_size, uint32_t num_batches,
                              uint32_t blocks, uint32_t threads,
                              uint64_t page_size, uint32_t cache_pages,
                              const float *h_features,
                              std::mt19937 &rng) {
  GNNResult r = {"gnn_bam", 0, 0, 0};

  uint64_t feature_bytes = (uint64_t)num_nodes * emb_dim * sizeof(float);

  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = cache_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  uint64_t feature_bytes_aligned = ((feature_bytes + page_size - 1) / page_size) * page_size;
  cache.alloc_host_backing(feature_bytes_aligned);
  memcpy(cache.host_buffer(), h_features, feature_bytes);
  if (feature_bytes_aligned > feature_bytes) {
    memset(cache.host_buffer() + feature_bytes, 0, feature_bytes_aligned - feature_bytes);
  }

  // Batch indices and output on GPU
  uint32_t *d_indices;
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_indices, batch_size * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_output, (uint64_t)batch_size * emb_dim * sizeof(float)));

  std::uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);
  std::vector<uint32_t> h_indices(batch_size);

  // Warmup
  for (uint32_t i = 0; i < batch_size; i++) h_indices[i] = node_dist(rng);
  CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(),
                         batch_size * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
  gnn_gather_bam_kernel<<<blocks, threads>>>(
      cache.device_state(), cache.host_buffer(),
      d_indices, d_output, batch_size, emb_dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  double total_ms = 0;
  for (uint32_t batch = 0; batch < num_batches; batch++) {
    // Random mini-batch each iteration
    for (uint32_t i = 0; i < batch_size; i++) h_indices[i] = node_dist(rng);
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(),
                           batch_size * sizeof(uint32_t),
                           cudaMemcpyHostToDevice));

    // Reset cache for fair comparison
    CUDA_CHECK(cudaMemset(cache.device_state().page_tags, 0xFF,
                           cache_pages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(cache.device_state().page_states, 0,
                           cache_pages * sizeof(uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    gnn_gather_bam_kernel<<<blocks, threads>>>(
        cache.device_state(), cache.host_buffer(),
        d_indices, d_output, batch_size, emb_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / num_batches;
  uint64_t bytes_per_batch = (uint64_t)batch_size * emb_dim * sizeof(float);
  r.throughput_gbps = (bytes_per_batch / 1e9) / (r.elapsed_ms / 1e3);
  r.features_per_sec = batch_size / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_output));
  return r;
}

static GNNResult run_gnn_direct(uint32_t num_nodes, uint32_t emb_dim,
                                 uint32_t batch_size, uint32_t num_batches,
                                 uint32_t blocks, uint32_t threads,
                                 const float *h_features,
                                 std::mt19937 &rng) {
  GNNResult r = {"gnn_direct", 0, 0, 0};

  uint32_t *d_indices;
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_indices, batch_size * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_output, (uint64_t)batch_size * emb_dim * sizeof(float)));

  std::uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);
  std::vector<uint32_t> h_indices(batch_size);

  // Warmup
  for (uint32_t i = 0; i < batch_size; i++) h_indices[i] = node_dist(rng);
  CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(),
                         batch_size * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
  gnn_gather_direct_kernel<<<blocks, threads>>>(
      h_features, d_indices, d_output, batch_size, emb_dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0;
  for (uint32_t batch = 0; batch < num_batches; batch++) {
    for (uint32_t i = 0; i < batch_size; i++) h_indices[i] = node_dist(rng);
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(),
                           batch_size * sizeof(uint32_t),
                           cudaMemcpyHostToDevice));

    auto t0 = std::chrono::high_resolution_clock::now();
    gnn_gather_direct_kernel<<<blocks, threads>>>(
        h_features, d_indices, d_output, batch_size, emb_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  r.elapsed_ms = total_ms / num_batches;
  uint64_t bytes_per_batch = (uint64_t)batch_size * emb_dim * sizeof(float);
  r.throughput_gbps = (bytes_per_batch / 1e9) / (r.elapsed_ms / 1e3);
  r.features_per_sec = batch_size / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_output));
  return r;
}

/* ================================================================== */
/* Main                                                                */
/* ================================================================== */

static uint64_t parse_size(const char *s) {
  double val = atof(s);
  const char *p = s;
  while (*p && (isdigit(*p) || *p == '.')) p++;
  switch (tolower(*p)) {
    case 'k': return (uint64_t)(val * 1024);
    case 'm': return (uint64_t)(val * 1024 * 1024);
    default:  return (uint64_t)val;
  }
}

int main(int argc, char **argv) {
  uint32_t num_nodes = 1000000;   // 1M nodes
  uint32_t emb_dim = 128;         // 128-float embeddings
  uint32_t batch_size = 1024;
  uint32_t num_batches = 20;
  uint64_t page_size = 65536;
  uint32_t cache_pages = 512;
  uint32_t warps = 32;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--num-nodes" && i+1 < argc) num_nodes = atoi(argv[++i]);
    else if (arg == "--emb-dim" && i+1 < argc) emb_dim = atoi(argv[++i]);
    else if (arg == "--batch-size" && i+1 < argc) batch_size = atoi(argv[++i]);
    else if (arg == "--num-batches" && i+1 < argc) num_batches = atoi(argv[++i]);
    else if (arg == "--page-size" && i+1 < argc) page_size = parse_size(argv[++i]);
    else if (arg == "--cache-pages" && i+1 < argc) cache_pages = atoi(argv[++i]);
    else if (arg == "--warps" && i+1 < argc) warps = atoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s [--num-nodes N] [--emb-dim D] [--batch-size B]\n"
             "           [--num-batches N] [--page-size B] [--cache-pages N]\n",
             argv[0]);
      return 0;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  uint64_t feature_bytes = (uint64_t)num_nodes * emb_dim * sizeof(float);

  printf("============================================================\n");
  printf("  GPU GNN Feature Loading: BaM vs Direct DRAM Access\n");
  printf("============================================================\n");
  printf("GPU:           %s\n", prop.name);
  printf("Nodes:         %u\n", num_nodes);
  printf("Embedding dim: %u (%.1f KB/node)\n", emb_dim,
         emb_dim * sizeof(float) / 1024.0);
  printf("Feature store: %.1f MB in DRAM\n", feature_bytes / (1024.0 * 1024.0));
  printf("Batch size:    %u nodes (%.1f MB/batch)\n", batch_size,
         (uint64_t)batch_size * emb_dim * sizeof(float) / (1024.0 * 1024.0));
  printf("Num batches:   %u\n", num_batches);
  printf("Warps:         %u\n", warps);
  printf("BaM cache:     %u x %lu B (%.1f MB)\n",
         cache_pages, (unsigned long)page_size,
         (cache_pages * page_size) / (1024.0 * 1024.0));
  printf("------------------------------------------------------------\n");

  // Generate random features in pinned DRAM
  printf("Generating feature store...\n");
  float *h_features;
  CUDA_CHECK(cudaMallocHost(&h_features, feature_bytes));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> feat_dist(-1.0f, 1.0f);
  for (uint64_t i = 0; i < (uint64_t)num_nodes * emb_dim; i++) {
    h_features[i] = feat_dist(rng);
  }
  printf("  Generated %.1f MB of features\n\n", feature_bytes / (1024.0 * 1024.0));

  uint32_t threads_per_block = 256;
  uint32_t total_threads = warps * 32;
  uint32_t blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  printf("Running GNN feature gather (BaM)...\n");
  GNNResult bam = run_gnn_bam(num_nodes, emb_dim, batch_size, num_batches,
                               blocks, threads_per_block,
                               page_size, cache_pages, h_features, rng);

  printf("Running GNN feature gather (direct)...\n");
  GNNResult direct = run_gnn_direct(num_nodes, emb_dim, batch_size,
                                     num_batches, blocks, threads_per_block,
                                     h_features, rng);

  printf("\n============================================================\n");
  printf("  GNN Feature Gather Results (per mini-batch)\n");
  printf("============================================================\n");
  printf("%-14s  %10s  %10s  %12s\n",
         "Method", "Time (ms)", "BW (GB/s)", "Nodes/sec");
  printf("%-14s  %10s  %10s  %12s\n",
         "------", "---------", "---------", "---------");
  printf("%-14s  %10.3f  %10.3f  %10.2e\n",
         bam.name, bam.elapsed_ms, bam.throughput_gbps, bam.features_per_sec);
  printf("%-14s  %10.3f  %10.3f  %10.2e\n",
         direct.name, direct.elapsed_ms, direct.throughput_gbps,
         direct.features_per_sec);
  printf("%-14s  %10.2fx\n", "BaM speedup",
         direct.elapsed_ms / bam.elapsed_ms);
  printf("============================================================\n");

  CUDA_CHECK(cudaFreeHost(h_features));
  return 0;
}
