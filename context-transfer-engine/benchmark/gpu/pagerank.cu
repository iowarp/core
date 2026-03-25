/**
 * pagerank.cu — GPU PageRank benchmark: BaM page cache vs direct DRAM access
 *
 * Workload description:
 *   PageRank is an iterative graph algorithm that computes vertex importance
 *   scores. Each iteration, every vertex pushes its score to neighbors via
 *   the edge list. The edge list is the dominant data structure — for large
 *   graphs it can be tens of GB, far exceeding GPU HBM capacity.
 *
 *   This benchmark compares two approaches for accessing the edge list:
 *     1. BaM: Edge list in DRAM, accessed through GPU HBM page cache
 *     2. Direct: Edge list in pinned DRAM, GPU reads directly over PCIe
 *
 *   The access pattern is irregular (depends on graph structure), which
 *   is where BaM's caching provides the most benefit — hot edges stay
 *   in HBM across iterations.
 *
 * Graph format:
 *   CSR (Compressed Sparse Row): offsets[] and edges[] arrays.
 *   Synthetic R-MAT graph generated on the host.
 *
 * Based on: BaM benchmarks/pagerank (ASPLOS'23)
 *
 * Usage:
 *   bench_gpu_pagerank [--vertices N] [--avg-degree D] [--iterations N]
 *                      [--page-size B] [--cache-pages N]
 */
#include <bam/bam.h>
#include <bam/page_cache.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

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
/* Synthetic graph generation (R-MAT style)                            */
/* ================================================================== */

struct CSRGraph {
  std::vector<uint64_t> offsets;  // size = num_vertices + 1
  std::vector<uint32_t> edges;   // size = num_edges
  uint32_t num_vertices;
  uint64_t num_edges;
};

static CSRGraph generate_rmat_graph(uint32_t num_vertices, uint32_t avg_degree,
                                     uint64_t seed = 42) {
  CSRGraph g;
  g.num_vertices = num_vertices;

  // Generate random edges with power-law degree distribution
  std::mt19937_64 rng(seed);
  std::vector<std::vector<uint32_t>> adj(num_vertices);

  uint64_t target_edges = (uint64_t)num_vertices * avg_degree;
  // R-MAT parameters: a=0.57, b=0.19, c=0.19, d=0.05
  double a = 0.57, b = 0.19, c = 0.19;

  uint32_t log2n = 0;
  uint32_t n = num_vertices;
  while (n > 1) { n >>= 1; log2n++; }

  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (uint64_t e = 0; e < target_edges; e++) {
    uint32_t u = 0, v = 0;
    for (uint32_t level = 0; level < log2n; level++) {
      double r = dist(rng);
      uint32_t half = 1u << (log2n - level - 1);
      if (r < a) {
        // quadrant (0,0)
      } else if (r < a + b) {
        v += half;  // quadrant (0,1)
      } else if (r < a + b + c) {
        u += half;  // quadrant (1,0)
      } else {
        u += half; v += half;  // quadrant (1,1)
      }
    }
    u %= num_vertices;
    v %= num_vertices;
    if (u != v) {
      adj[u].push_back(v);
    }
  }

  // Build CSR
  g.offsets.resize(num_vertices + 1);
  g.offsets[0] = 0;
  for (uint32_t i = 0; i < num_vertices; i++) {
    // Deduplicate and sort
    std::sort(adj[i].begin(), adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    g.offsets[i + 1] = g.offsets[i] + adj[i].size();
  }

  g.num_edges = g.offsets[num_vertices];
  g.edges.resize(g.num_edges);
  for (uint32_t i = 0; i < num_vertices; i++) {
    std::copy(adj[i].begin(), adj[i].end(),
              g.edges.begin() + g.offsets[i]);
  }

  return g;
}

/* ================================================================== */
/* GPU kernels: PageRank via BaM page cache                            */
/* ================================================================== */

/**
 * PageRank push kernel (warp-per-vertex, BaM page cache for edges).
 *
 * Each warp processes one vertex: reads its outgoing edges through the
 * BaM page cache and atomicAdds its contribution to each neighbor.
 */
__global__ void pagerank_push_bam_kernel(
    const uint64_t *d_offsets,       // CSR offsets (in HBM)
    bam::PageCacheDeviceState cache_state,
    const uint8_t *host_edges_base,  // Edge list in DRAM (backing store)
    const float *d_values,           // Current PR scores
    float *d_residuals,              // Accumulated contributions
    uint32_t num_vertices,
    float alpha) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  for (uint32_t v = warp_id; v < num_vertices; v += num_warps) {
    uint64_t start = d_offsets[v];
    uint64_t end = d_offsets[v + 1];
    uint32_t degree = (uint32_t)(end - start);
    if (degree == 0) continue;

    float contribution = alpha * d_values[v] / degree;

    // Read edges through BaM page cache (per-thread acquire, safe for
    // divergent page offsets across lanes)
    for (uint64_t e = start + lane_id; e < end; e += 32) {
      uint64_t byte_off = e * sizeof(uint32_t);
      uint64_t page_off = byte_off & ~((uint64_t)cache_state.page_size - 1);
      uint32_t in_page = (uint32_t)(byte_off & ((uint64_t)cache_state.page_size - 1));

      bool needs_load;
      uint8_t *page = bam::page_cache_acquire(
          cache_state, page_off, &needs_load);

      if (needs_load) {
        bam::host_read_page(page, host_edges_base, page_off,
                            cache_state.page_size);
        bam::page_cache_finish_load(cache_state, page_off);
      }

      uint32_t neighbor = *reinterpret_cast<const uint32_t *>(page + in_page);
      atomicAdd(&d_residuals[neighbor], contribution);
    }
    __syncwarp();
  }
}

/**
 * PageRank push kernel (warp-per-vertex, direct DRAM access).
 * Baseline: reads edges directly from pinned host memory.
 */
__global__ void pagerank_push_direct_kernel(
    const uint64_t *d_offsets,
    const uint32_t *h_edges,         // Pinned DRAM
    const float *d_values,
    float *d_residuals,
    uint32_t num_vertices,
    float alpha) {
  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t num_warps = (blockDim.x * gridDim.x) / 32;

  for (uint32_t v = warp_id; v < num_vertices; v += num_warps) {
    uint64_t start = d_offsets[v];
    uint64_t end = d_offsets[v + 1];
    uint32_t degree = (uint32_t)(end - start);
    if (degree == 0) continue;

    float contribution = alpha * d_values[v] / degree;

    for (uint64_t e = start + lane_id; e < end; e += 32) {
      uint32_t neighbor = h_edges[e];
      atomicAdd(&d_residuals[neighbor], contribution);
    }
    __syncwarp();
  }
}

/**
 * PageRank update kernel: apply residuals, check convergence.
 */
__global__ void pagerank_update_kernel(
    float *d_values,
    float *d_residuals,
    uint32_t num_vertices,
    float tolerance,
    int *d_active_count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_vertices) return;

  float r = d_residuals[tid];
  if (fabsf(r) > tolerance) {
    d_values[tid] += r;
    atomicAdd(d_active_count, 1);
  }
  d_residuals[tid] = 0.0f;
}

/* ================================================================== */
/* Benchmark runner                                                    */
/* ================================================================== */

struct PRResult {
  const char *name;
  double elapsed_ms;
  double edge_throughput;  // edges/sec
  int iterations;
};

static PRResult run_pagerank_bam(const CSRGraph &g,
                                  uint32_t blocks, uint32_t threads,
                                  int max_iters, float alpha, float tol,
                                  uint64_t page_size, uint32_t cache_pages) {
  PRResult r = {"pagerank_bam", 0, 0, 0};

  // Upload offsets to GPU
  uint64_t *d_offsets;
  CUDA_CHECK(cudaMalloc(&d_offsets, (g.num_vertices + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_offsets, g.offsets.data(),
                         (g.num_vertices + 1) * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));

  // Edge list in BaM page cache (DRAM backing)
  bam::PageCacheConfig config;
  config.page_size = page_size;
  config.num_pages = cache_pages;
  config.num_queues = 0;
  config.queue_depth = 0;
  config.backend = bam::BackendType::kHostMemory;
  config.nvme_dev = nullptr;

  bam::PageCache cache(config);
  uint64_t edge_bytes = g.num_edges * sizeof(uint32_t);
  // Round up to page-aligned size to prevent out-of-bounds page reads
  uint64_t edge_bytes_aligned = ((edge_bytes + page_size - 1) / page_size) * page_size;
  cache.alloc_host_backing(edge_bytes_aligned);
  memcpy(cache.host_buffer(), g.edges.data(), edge_bytes);
  // Zero-fill the padding
  if (edge_bytes_aligned > edge_bytes) {
    memset(cache.host_buffer() + edge_bytes, 0, edge_bytes_aligned - edge_bytes);
  }

  // PR state arrays on GPU
  float *d_values, *d_residuals;
  CUDA_CHECK(cudaMalloc(&d_values, g.num_vertices * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_residuals, g.num_vertices * sizeof(float)));

  // Initialize: value = 1.0, residuals = 0
  std::vector<float> init_vals(g.num_vertices, 1.0f);
  CUDA_CHECK(cudaMemcpy(d_values, init_vals.data(),
                         g.num_vertices * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_residuals, 0, g.num_vertices * sizeof(float)));

  int *d_active;
  CUDA_CHECK(cudaMallocHost(&d_active, sizeof(int)));

  auto t0 = std::chrono::high_resolution_clock::now();

  int iter;
  for (iter = 0; iter < max_iters; iter++) {
    pagerank_push_bam_kernel<<<blocks, threads>>>(
        d_offsets, cache.device_state(),
        cache.host_buffer(),
        d_values, d_residuals, g.num_vertices, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());

    *d_active = 0;
    int update_blocks = (g.num_vertices + 255) / 256;
    pagerank_update_kernel<<<update_blocks, 256>>>(
        d_values, d_residuals, g.num_vertices, tol, d_active);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (*d_active == 0) { iter++; break; }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.iterations = iter;
  r.edge_throughput = (double)g.num_edges * iter / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_residuals));
  CUDA_CHECK(cudaFreeHost(d_active));
  return r;
}

static PRResult run_pagerank_direct(const CSRGraph &g,
                                     uint32_t blocks, uint32_t threads,
                                     int max_iters, float alpha, float tol) {
  PRResult r = {"pagerank_direct", 0, 0, 0};

  uint64_t *d_offsets;
  CUDA_CHECK(cudaMalloc(&d_offsets, (g.num_vertices + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_offsets, g.offsets.data(),
                         (g.num_vertices + 1) * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));

  // Edge list in pinned DRAM (direct access)
  uint32_t *h_edges;
  CUDA_CHECK(cudaMallocHost(&h_edges, g.num_edges * sizeof(uint32_t)));
  memcpy(h_edges, g.edges.data(), g.num_edges * sizeof(uint32_t));

  float *d_values, *d_residuals;
  CUDA_CHECK(cudaMalloc(&d_values, g.num_vertices * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_residuals, g.num_vertices * sizeof(float)));

  std::vector<float> init_vals(g.num_vertices, 1.0f);
  CUDA_CHECK(cudaMemcpy(d_values, init_vals.data(),
                         g.num_vertices * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_residuals, 0, g.num_vertices * sizeof(float)));

  int *d_active;
  CUDA_CHECK(cudaMallocHost(&d_active, sizeof(int)));

  auto t0 = std::chrono::high_resolution_clock::now();

  int iter;
  for (iter = 0; iter < max_iters; iter++) {
    pagerank_push_direct_kernel<<<blocks, threads>>>(
        d_offsets, h_edges,
        d_values, d_residuals, g.num_vertices, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());

    *d_active = 0;
    int update_blocks = (g.num_vertices + 255) / 256;
    pagerank_update_kernel<<<update_blocks, 256>>>(
        d_values, d_residuals, g.num_vertices, tol, d_active);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (*d_active == 0) { iter++; break; }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.iterations = iter;
  r.edge_throughput = (double)g.num_edges * iter / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_residuals));
  CUDA_CHECK(cudaFreeHost(h_edges));
  CUDA_CHECK(cudaFreeHost(d_active));
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
  uint32_t num_vertices = 100000;
  uint32_t avg_degree = 16;
  int max_iters = 20;
  float alpha = 0.85f;
  float tolerance = 0.001f;
  uint64_t page_size = 65536;
  uint32_t cache_pages = 256;
  uint32_t warps = 32;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--vertices" && i+1 < argc) num_vertices = atoi(argv[++i]);
    else if (arg == "--avg-degree" && i+1 < argc) avg_degree = atoi(argv[++i]);
    else if (arg == "--iterations" && i+1 < argc) max_iters = atoi(argv[++i]);
    else if (arg == "--page-size" && i+1 < argc) page_size = parse_size(argv[++i]);
    else if (arg == "--cache-pages" && i+1 < argc) cache_pages = atoi(argv[++i]);
    else if (arg == "--warps" && i+1 < argc) warps = atoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s [--vertices N] [--avg-degree D] [--iterations N]\n"
             "           [--page-size B] [--cache-pages N] [--warps N]\n", argv[0]);
      return 0;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("============================================================\n");
  printf("  GPU PageRank Benchmark: BaM vs Direct DRAM Access\n");
  printf("============================================================\n");
  printf("GPU:          %s\n", prop.name);
  printf("Vertices:     %u\n", num_vertices);
  printf("Avg degree:   %u\n", avg_degree);
  printf("Max iters:    %d\n", max_iters);
  printf("Alpha:        %.2f\n", alpha);
  printf("Tolerance:    %.4f\n", tolerance);
  printf("Warps:        %u\n", warps);
  printf("BaM pages:    %u x %lu B (%.1f MB cache)\n",
         cache_pages, (unsigned long)page_size,
         (cache_pages * page_size) / (1024.0 * 1024.0));
  printf("------------------------------------------------------------\n");

  printf("Generating R-MAT graph...\n");
  auto t_gen = std::chrono::high_resolution_clock::now();
  CSRGraph g = generate_rmat_graph(num_vertices, avg_degree);
  auto t_gen_end = std::chrono::high_resolution_clock::now();
  double gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen).count();

  printf("  Vertices: %u, Edges: %lu (%.1f MB)\n",
         g.num_vertices, (unsigned long)g.num_edges,
         g.num_edges * sizeof(uint32_t) / (1024.0 * 1024.0));
  printf("  Generated in %.1f ms\n\n", gen_ms);

  uint32_t threads_per_block = 256;
  uint32_t total_threads = warps * 32;
  uint32_t blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  printf("Running PageRank (BaM page cache)...\n");
  PRResult bam = run_pagerank_bam(g, blocks, threads_per_block,
                                   max_iters, alpha, tolerance,
                                   page_size, cache_pages);

  printf("Running PageRank (direct DRAM)...\n");
  PRResult direct = run_pagerank_direct(g, blocks, threads_per_block,
                                         max_iters, alpha, tolerance);

  printf("\n============================================================\n");
  printf("  PageRank Results\n");
  printf("============================================================\n");
  printf("%-18s  %10s  %10s  %12s\n",
         "Method", "Time (ms)", "Iters", "Edges/sec");
  printf("%-18s  %10s  %10s  %12s\n",
         "------", "---------", "-----", "---------");
  printf("%-18s  %10.1f  %10d  %10.2e\n",
         bam.name, bam.elapsed_ms, bam.iterations, bam.edge_throughput);
  printf("%-18s  %10.1f  %10d  %10.2e\n",
         direct.name, direct.elapsed_ms, direct.iterations,
         direct.edge_throughput);
  printf("%-18s  %10.2fx\n", "BaM speedup",
         direct.elapsed_ms / bam.elapsed_ms);
  printf("============================================================\n");

  return 0;
}
