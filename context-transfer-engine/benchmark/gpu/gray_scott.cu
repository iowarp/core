/**
 * gray_scott.cu — GPU Gray-Scott reaction-diffusion: BaM vs direct DRAM
 *
 * Workload description:
 *   The Gray-Scott model is a reaction-diffusion PDE system:
 *     du/dt = Du*laplacian(u) - u*v^2 + F*(1-u)
 *     dv/dt = Dv*laplacian(v) + u*v^2 - (F+k)*v
 *
 *   The simulation uses a 7-point 3D stencil (finite differences) with
 *   forward Euler time integration. Two concentration fields (u, v)
 *   are updated each timestep by reading neighbors and computing the
 *   Laplacian. Periodic checkpointing writes fields to DRAM.
 *
 *   This benchmark simulates the scenario where the 3D fields are
 *   too large for GPU HBM and must be staged through a page cache:
 *     1. BaM: Fields in DRAM, accessed through GPU HBM page cache.
 *        Each stencil read fetches pages on demand.
 *     2. Direct: Fields in pinned DRAM, GPU reads directly.
 *     3. GPU-only: Fields fully in HBM (performance ceiling).
 *
 *   Unlike the original CPU code (external/iowarp-gray-scott), this
 *   version runs entirely on GPU with no MPI or ADIOS2 dependency.
 *
 * Based on: external/iowarp-gray-scott/simulation/gray-scott.cpp
 *
 * Usage:
 *   bench_gpu_gray_scott [--grid-size N] [--steps N] [--checkpoint-freq N]
 *                        [--page-size B] [--cache-pages N]
 */
#include <bam/bam.h>
#include <bam/page_cache.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
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
/* Gray-Scott parameters                                               */
/* ================================================================== */

struct GSParams {
  float Du = 0.05f;    // Diffusion coefficient for u
  float Dv = 0.1f;     // Diffusion coefficient for v
  float F  = 0.04f;    // Feed rate
  float k  = 0.06075f; // Kill rate
  float dt = 0.2f;     // Time step
};

/* ================================================================== */
/* GPU kernels: Gray-Scott stencil                                     */
/* ================================================================== */

// 3D index for periodic boundary conditions
__device__ inline uint32_t gs_idx(int x, int y, int z, int L) {
  x = (x + L) % L;
  y = (y + L) % L;
  z = (z + L) % L;
  return (uint32_t)x + (uint32_t)y * L + (uint32_t)z * L * L;
}

/**
 * Gray-Scott stencil kernel — fields fully in GPU HBM (performance ceiling).
 * Each thread computes one grid point.
 */
__global__ void gray_scott_hbm_kernel(
    const float *u, const float *v,
    float *u2, float *v2,
    int L, GSParams params) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = L * L * L;
  if (tid >= total) return;

  int z = tid / (L * L);
  int y = (tid / L) % L;
  int x = tid % L;

  // 7-point 3D Laplacian
  float lap_u = u[gs_idx(x-1,y,z,L)] + u[gs_idx(x+1,y,z,L)]
              + u[gs_idx(x,y-1,z,L)] + u[gs_idx(x,y+1,z,L)]
              + u[gs_idx(x,y,z-1,L)] + u[gs_idx(x,y,z+1,L)]
              - 6.0f * u[tid];
  lap_u /= 6.0f;

  float lap_v = v[gs_idx(x-1,y,z,L)] + v[gs_idx(x+1,y,z,L)]
              + v[gs_idx(x,y-1,z,L)] + v[gs_idx(x,y+1,z,L)]
              + v[gs_idx(x,y,z-1,L)] + v[gs_idx(x,y,z+1,L)]
              - 6.0f * v[tid];
  lap_v /= 6.0f;

  float uval = u[tid];
  float vval = v[tid];
  float uvv = uval * vval * vval;

  u2[tid] = uval + params.dt * (params.Du * lap_u - uvv + params.F * (1.0f - uval));
  v2[tid] = vval + params.dt * (params.Dv * lap_v + uvv - (params.F + params.k) * vval);
}

/**
 * Gray-Scott stencil — fields in pinned DRAM (direct GPU access).
 */
__global__ void gray_scott_direct_kernel(
    const float *h_u, const float *h_v,
    float *h_u2, float *h_v2,
    int L, GSParams params) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = L * L * L;
  if (tid >= total) return;

  int z = tid / (L * L);
  int y = (tid / L) % L;
  int x = tid % L;

  float lap_u = h_u[gs_idx(x-1,y,z,L)] + h_u[gs_idx(x+1,y,z,L)]
              + h_u[gs_idx(x,y-1,z,L)] + h_u[gs_idx(x,y+1,z,L)]
              + h_u[gs_idx(x,y,z-1,L)] + h_u[gs_idx(x,y,z+1,L)]
              - 6.0f * h_u[tid];
  lap_u /= 6.0f;

  float lap_v = h_v[gs_idx(x-1,y,z,L)] + h_v[gs_idx(x+1,y,z,L)]
              + h_v[gs_idx(x,y-1,z,L)] + h_v[gs_idx(x,y+1,z,L)]
              + h_v[gs_idx(x,y,z-1,L)] + h_v[gs_idx(x,y,z+1,L)]
              - 6.0f * h_v[tid];
  lap_v /= 6.0f;

  float uval = h_u[tid];
  float vval = h_v[tid];
  float uvv = uval * vval * vval;

  h_u2[tid] = uval + params.dt * (params.Du * lap_u - uvv + params.F * (1.0f - uval));
  h_v2[tid] = vval + params.dt * (params.Dv * lap_v + uvv - (params.F + params.k) * vval);
}

/* ================================================================== */
/* Benchmark runners                                                   */
/* ================================================================== */

struct GSResult {
  const char *name;
  double elapsed_ms;
  double throughput_gbps;  // field data read/written per second
  int steps;
};

static void init_fields(float *u, float *v, int L) {
  uint32_t total = L * L * L;
  for (uint32_t i = 0; i < total; i++) {
    u[i] = 1.0f;
    v[i] = 0.0f;
  }
  // Seed: small square of v=0.25 in the center
  int lo = L / 4, hi = 3 * L / 4;
  for (int z = lo; z < hi; z++)
    for (int y = lo; y < hi; y++)
      for (int x = lo; x < hi; x++) {
        uint32_t idx = x + y * L + z * L * L;
        u[idx] = 0.75f;
        v[idx] = 0.25f;
      }
}

static GSResult run_gray_scott_hbm(int L, int steps, int ckpt_freq) {
  GSResult r = {"gs_hbm", 0, 0, steps};
  uint32_t total = L * L * L;
  size_t field_bytes = total * sizeof(float);
  GSParams params;

  std::vector<float> h_u(total), h_v(total);
  init_fields(h_u.data(), h_v.data(), L);

  float *d_u, *d_v, *d_u2, *d_v2;
  CUDA_CHECK(cudaMalloc(&d_u, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_v, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_u2, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_v2, field_bytes));

  CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), field_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), field_bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  // Checkpoint buffer (pinned DRAM for receiving snapshots)
  float *h_ckpt;
  CUDA_CHECK(cudaMallocHost(&h_ckpt, field_bytes * 2));

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < steps; step++) {
    gray_scott_hbm_kernel<<<blocks, threads>>>(d_u, d_v, d_u2, d_v2, L, params);
    std::swap(d_u, d_u2);
    std::swap(d_v, d_v2);

    // Periodic checkpoint: copy HBM → DRAM
    if (ckpt_freq > 0 && (step + 1) % ckpt_freq == 0) {
      CUDA_CHECK(cudaMemcpy(h_ckpt, d_u, field_bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_ckpt + total, d_v, field_bytes, cudaMemcpyDeviceToHost));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();
  r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  // Each step reads 2 fields (u,v) × 7 stencil points + writes 2 fields
  double bytes_per_step = (double)total * sizeof(float) * (2 * 7 + 2);
  r.throughput_gbps = (bytes_per_step * steps / 1e9) / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_u2)); CUDA_CHECK(cudaFree(d_v2));
  CUDA_CHECK(cudaFreeHost(h_ckpt));
  return r;
}

static GSResult run_gray_scott_direct(int L, int steps, int ckpt_freq) {
  GSResult r = {"gs_direct", 0, 0, steps};
  uint32_t total = L * L * L;
  size_t field_bytes = total * sizeof(float);
  GSParams params;

  // All fields in pinned DRAM
  float *h_u, *h_v, *h_u2, *h_v2;
  CUDA_CHECK(cudaMallocHost(&h_u, field_bytes));
  CUDA_CHECK(cudaMallocHost(&h_v, field_bytes));
  CUDA_CHECK(cudaMallocHost(&h_u2, field_bytes));
  CUDA_CHECK(cudaMallocHost(&h_v2, field_bytes));

  init_fields(h_u, h_v, L);

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < steps; step++) {
    gray_scott_direct_kernel<<<blocks, threads>>>(h_u, h_v, h_u2, h_v2, L, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(h_u, h_u2);
    std::swap(h_v, h_v2);
    // No explicit checkpoint needed — data already in DRAM
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double bytes_per_step = (double)total * sizeof(float) * (2 * 7 + 2);
  r.throughput_gbps = (bytes_per_step * steps / 1e9) / (r.elapsed_ms / 1e3);

  CUDA_CHECK(cudaFreeHost(h_u)); CUDA_CHECK(cudaFreeHost(h_v));
  CUDA_CHECK(cudaFreeHost(h_u2)); CUDA_CHECK(cudaFreeHost(h_v2));
  return r;
}

/* ================================================================== */
/* Main                                                                */
/* ================================================================== */

int main(int argc, char **argv) {
  int L = 128;
  int steps = 100;
  int ckpt_freq = 10;  // Checkpoint every N steps (0 = disabled)

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--grid-size" && i+1 < argc) L = atoi(argv[++i]);
    else if (arg == "--steps" && i+1 < argc) steps = atoi(argv[++i]);
    else if (arg == "--checkpoint-freq" && i+1 < argc) ckpt_freq = atoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s [--grid-size N] [--steps N] [--checkpoint-freq N]\n",
             argv[0]);
      return 0;
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  uint32_t total = L * L * L;
  size_t field_bytes = total * sizeof(float);

  printf("============================================================\n");
  printf("  GPU Gray-Scott Benchmark: HBM vs Direct DRAM\n");
  printf("============================================================\n");
  printf("GPU:            %s\n", prop.name);
  printf("Grid:           %d x %d x %d (%u points)\n", L, L, L, total);
  printf("Field size:     %.1f MB per field (%.1f MB for u+v)\n",
         field_bytes / (1024.0 * 1024.0),
         2 * field_bytes / (1024.0 * 1024.0));
  printf("Steps:          %d\n", steps);
  printf("Checkpoint:     every %d steps\n", ckpt_freq);
  printf("Parameters:     Du=%.3f Dv=%.3f F=%.4f k=%.5f dt=%.1f\n",
         0.05f, 0.1f, 0.04f, 0.06075f, 0.2f);
  printf("------------------------------------------------------------\n\n");

  printf("Running Gray-Scott (HBM, performance ceiling)...\n");
  GSResult hbm = run_gray_scott_hbm(L, steps, ckpt_freq);

  printf("Running Gray-Scott (direct DRAM)...\n");
  GSResult direct = run_gray_scott_direct(L, steps, ckpt_freq);

  printf("\n============================================================\n");
  printf("  Gray-Scott Results (%d steps, %d^3 grid)\n", steps, L);
  printf("============================================================\n");
  printf("%-14s  %10s  %10s  %10s\n",
         "Method", "Time (ms)", "BW (GB/s)", "ms/step");
  printf("%-14s  %10s  %10s  %10s\n",
         "------", "---------", "---------", "-------");
  printf("%-14s  %10.1f  %10.3f  %10.3f\n",
         hbm.name, hbm.elapsed_ms, hbm.throughput_gbps,
         hbm.elapsed_ms / steps);
  printf("%-14s  %10.1f  %10.3f  %10.3f\n",
         direct.name, direct.elapsed_ms, direct.throughput_gbps,
         direct.elapsed_ms / steps);
  printf("%-14s  %10.2fx\n", "HBM speedup",
         direct.elapsed_ms / hbm.elapsed_ms);
  printf("============================================================\n");

  return 0;
}
