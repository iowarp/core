/**
 * workload_gray_scott.cc — Gray-Scott stencil for CTE GPU bench
 *
 * CTE mode: Combined kernel. Each warp owns a slice of grid points.
 *   Lane 0 calls AsyncGetBlob to load u,v slices (with halo for neighbors).
 *   All 32 lanes compute 7-point stencil (the science).
 *   Lane 0 calls AsyncPutBlob to write updated u2,v2.
 *
 * BaM mode: Uses bam::ArrayDevice<float>::read() for u,v access.
 *   Transparent page cache — no raw page_cache_acquire.
 *
 * HBM/Direct: Standard compute kernels.
 */

#include <cstdint>
#include <cmath>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/array.cuh>
#endif

struct GSParams { float Du=0.05f, Dv=0.1f, F=0.04f, k=0.06075f, dt=0.2f; };

__device__ inline uint32_t gs_idx(int x, int y, int z, int L) {
  return (uint32_t)(((x+L)%L) + ((y+L)%L)*L + ((z+L)%L)*L*L);
}

// --- HBM/Direct compute kernels ---

__global__ void gs_stencil_hbm(const float *u, const float *v,
                                float *u2, float *v2, int L, GSParams p) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint32_t)(L*L*L)) return;
  int z=tid/(L*L), y=(tid/L)%L, x=tid%L;
  float lu = u[gs_idx(x-1,y,z,L)]+u[gs_idx(x+1,y,z,L)]
           +u[gs_idx(x,y-1,z,L)]+u[gs_idx(x,y+1,z,L)]
           +u[gs_idx(x,y,z-1,L)]+u[gs_idx(x,y,z+1,L)]-6.0f*u[tid];
  lu/=6.0f;
  float lv = v[gs_idx(x-1,y,z,L)]+v[gs_idx(x+1,y,z,L)]
           +v[gs_idx(x,y-1,z,L)]+v[gs_idx(x,y+1,z,L)]
           +v[gs_idx(x,y,z-1,L)]+v[gs_idx(x,y,z+1,L)]-6.0f*v[tid];
  lv/=6.0f;
  float uv=u[tid], vv=v[tid], uvv=uv*vv*vv;
  u2[tid]=uv+p.dt*(p.Du*lu-uvv+p.F*(1.0f-uv));
  v2[tid]=vv+p.dt*(p.Dv*lv+uvv-(p.F+p.k)*vv);
}

__global__ void gs_stencil_direct(const float *h_u, const float *h_v,
                                   float *h_u2, float *h_v2, int L, GSParams p) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint32_t)(L*L*L)) return;
  int z=tid/(L*L), y=(tid/L)%L, x=tid%L;
  float lu = h_u[gs_idx(x-1,y,z,L)]+h_u[gs_idx(x+1,y,z,L)]
           +h_u[gs_idx(x,y-1,z,L)]+h_u[gs_idx(x,y+1,z,L)]
           +h_u[gs_idx(x,y,z-1,L)]+h_u[gs_idx(x,y,z+1,L)]-6.0f*h_u[tid];
  lu/=6.0f;
  float lv = h_v[gs_idx(x-1,y,z,L)]+h_v[gs_idx(x+1,y,z,L)]
           +h_v[gs_idx(x,y-1,z,L)]+h_v[gs_idx(x,y+1,z,L)]
           +h_v[gs_idx(x,y,z-1,L)]+h_v[gs_idx(x,y,z+1,L)]-6.0f*h_v[tid];
  lv/=6.0f;
  float uv=h_u[tid], vv=h_v[tid], uvv=uv*vv*vv;
  h_u2[tid]=uv+p.dt*(p.Du*lu-uvv+p.F*(1.0f-uv));
  h_v2[tid]=vv+p.dt*(p.Dv*lv+uvv-(p.F+p.k)*vv);
}

#ifdef WRP_CORE_ENABLE_BAM
/**
 * BaM Gray-Scott stencil: reads u/v through bam::ArrayDevice<float>.
 */
__global__ void gs_stencil_bam(bam::ArrayDevice<float> u_arr,
                                bam::ArrayDevice<float> v_arr,
                                float *u2, float *v2, int L, GSParams p) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint32_t)(L*L*L)) return;
  int z=tid/(L*L), y=(tid/L)%L, x=tid%L;

  float lu = u_arr.read(gs_idx(x-1,y,z,L)) + u_arr.read(gs_idx(x+1,y,z,L))
           + u_arr.read(gs_idx(x,y-1,z,L)) + u_arr.read(gs_idx(x,y+1,z,L))
           + u_arr.read(gs_idx(x,y,z-1,L)) + u_arr.read(gs_idx(x,y,z+1,L))
           - 6.0f * u_arr.read(tid);
  lu/=6.0f;

  float lv = v_arr.read(gs_idx(x-1,y,z,L)) + v_arr.read(gs_idx(x+1,y,z,L))
           + v_arr.read(gs_idx(x,y-1,z,L)) + v_arr.read(gs_idx(x,y+1,z,L))
           + v_arr.read(gs_idx(x,y,z-1,L)) + v_arr.read(gs_idx(x,y,z+1,L))
           - 6.0f * v_arr.read(tid);
  lv/=6.0f;

  float uv = u_arr.read(tid);
  float vv = v_arr.read(tid);
  float uvv = uv * vv * vv;
  u2[tid] = uv + p.dt * (p.Du * lu - uvv + p.F * (1.0f - uv));
  v2[tid] = vv + p.dt * (p.Dv * lv + uvv - (p.F + p.k) * vv);
}
#endif

/**
 * Combined CTE Gray-Scott kernel: I/O + stencil in one kernel.
 *
 * Each warp owns a slice of grid points. Per step:
 *   1. Lane 0: AsyncGetBlob loads u,v fields for this warp's slice
 *   2. All 32 lanes: compute 7-point stencil (the science)
 *   3. Lane 0: AsyncPutBlob writes u2,v2
 *
 * Data layout in data backend:
 *   [warp_0 u,v | warp_1 u,v | ... | warp_N u,v | warp_0 u2,v2 | ... | warp_N u2,v2]
 */
__global__ void gs_cte_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    // Data
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    // Per-warp partition info (pinned host arrays)
    const uint64_t *warp_field_offsets,  // byte offset for each warp's u,v
    const uint64_t *warp_field_bytes,    // byte count for each warp's fields
    const uint32_t *warp_point_start,    // first point index for each warp
    const uint32_t *warp_point_end,      // last+1 point index for each warp
    chi::u32 total_warps,
    chi::u32 grid_size,
    int *d_done) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    uint32_t point_start = warp_point_start[warp_id];
    uint32_t point_end = warp_point_end[warp_id];
    chi::u64 my_field_offset = warp_field_offsets[warp_id];
    chi::u64 my_field_bytes = warp_field_bytes[warp_id];
    char *my_data = data_ptr.ptr_ + my_field_offset;

    // Build blob name: "gs_u_w<warp_id>" and "gs_v_w<warp_id>"
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
    char name_u[32], name_v[32];
    int pos_u = 0, pos_v = 0;
    const char *pfx = "gs_u_w";
    while (*pfx) name_u[pos_u++] = *pfx++;
    pos_u += StrT::NumberToStr(name_u + pos_u, 32 - pos_u, warp_id);
    name_u[pos_u] = '\0';

    pfx = "gs_v_w";
    while (*pfx) name_v[pos_v++] = *pfx++;
    pos_v += StrT::NumberToStr(name_v + pos_v, 32 - pos_v, warp_id);
    name_v[pos_v] = '\0';

    bool alloc_failed = false;

    // === I/O: GetBlob — load this warp's u,v fields from CTE ===
    if (chi::IpcManager::IsWarpScheduler() && my_field_bytes > 0) {
      wrp_cte::core::Client cte_client(cte_pool_id);
      uint64_t half_bytes = my_field_bytes / 2;

      // Get u field
      hipc::ShmPtr<> shm_u;
      shm_u.alloc_id_ = data_alloc_id;
      shm_u.off_.exchange(data_ptr.shm_.off_.load() + my_field_offset);
      auto get_u = cte_client.AsyncGetBlob(
          tag_id, name_u, (chi::u64)0, half_bytes,
          (chi::u32)0, shm_u, chi::PoolQuery::Local());
      if (!get_u.GetFutureShmPtr().IsNull()) {
        get_u.Wait();
      } else {
        alloc_failed = true;
      }

      // Get v field
      if (!alloc_failed) {
        hipc::ShmPtr<> shm_v;
        shm_v.alloc_id_ = data_alloc_id;
        shm_v.off_.exchange(data_ptr.shm_.off_.load() + my_field_offset + half_bytes);
        auto get_v = cte_client.AsyncGetBlob(
            tag_id, name_v, (chi::u64)0, half_bytes,
            (chi::u32)0, shm_v, chi::PoolQuery::Local());
        if (!get_v.GetFutureShmPtr().IsNull()) {
          get_v.Wait();
        } else {
          alloc_failed = true;
        }
      }
    }
    __syncwarp();

    // === COMPUTE: All 32 lanes compute stencil (THE SCIENCE) ===
    if (!alloc_failed) {
      uint64_t total_bytes = my_field_bytes / 2;
      const float *my_u = reinterpret_cast<const float *>(my_data);
      const float *my_v = reinterpret_cast<const float *>(my_data + total_bytes);
      float *my_u2 = reinterpret_cast<float *>(my_data + 2*total_bytes);
      float *my_v2 = reinterpret_cast<float *>(my_data + 3*total_bytes);

      GSParams params;
      int L = grid_size;

      for (uint32_t tidx = point_start + lane_id; tidx < point_end; tidx += 32) {
        int z=tidx/(L*L), y=(tidx/L)%L, x=tidx%L;
        float lu = my_u[gs_idx(x-1,y,z,L)]+my_u[gs_idx(x+1,y,z,L)]
                 +my_u[gs_idx(x,y-1,z,L)]+my_u[gs_idx(x,y+1,z,L)]
                 +my_u[gs_idx(x,y,z-1,L)]+my_u[gs_idx(x,y,z+1,L)]-6.0f*my_u[tidx];
        lu/=6.0f;
        float lv = my_v[gs_idx(x-1,y,z,L)]+my_v[gs_idx(x+1,y,z,L)]
                 +my_v[gs_idx(x,y-1,z,L)]+my_v[gs_idx(x,y+1,z,L)]
                 +my_v[gs_idx(x,y,z-1,L)]+my_v[gs_idx(x,y,z+1,L)]-6.0f*my_v[tidx];
        lv/=6.0f;
        float uv=my_u[tidx], vv=my_v[tidx], uvv=uv*vv*vv;
        my_u2[tidx]=uv+params.dt*(params.Du*lu-uvv+params.F*(1.0f-uv));
        my_v2[tidx]=vv+params.dt*(params.Dv*lv+uvv-(params.F+params.k)*vv);
      }
      __syncwarp();
    }

    // === I/O: PutBlob — write back updated u2,v2 ===
    if (chi::IpcManager::IsWarpScheduler() && !alloc_failed && my_field_bytes > 0) {
      wrp_cte::core::Client cte_client(cte_pool_id);
      uint64_t half_bytes = my_field_bytes / 2;

      // Put u2
      hipc::ShmPtr<> shm_u2;
      shm_u2.alloc_id_ = data_alloc_id;
      shm_u2.off_.exchange(data_ptr.shm_.off_.load() + my_field_offset + 2*half_bytes);
      auto put_u2 = cte_client.AsyncPutBlob(
          tag_id, name_u, (chi::u64)0, half_bytes,
          shm_u2, -1.0f, wrp_cte::core::Context(), (chi::u32)0,
          chi::PoolQuery::Local());
      if (!put_u2.GetFutureShmPtr().IsNull()) {
        put_u2.Wait();
      }

      // Put v2
      hipc::ShmPtr<> shm_v2;
      shm_v2.alloc_id_ = data_alloc_id;
      shm_v2.off_.exchange(data_ptr.shm_.off_.load() + my_field_offset + 3*half_bytes);
      auto put_v2 = cte_client.AsyncPutBlob(
          tag_id, name_v, (chi::u64)0, half_bytes,
          shm_v2, -1.0f, wrp_cte::core::Context(), (chi::u32)0,
          chi::PoolQuery::Local());
      if (!put_v2.GetFutureShmPtr().IsNull()) {
        put_v2.Wait();
      }
    }
    __syncwarp();
  }

  // Signal completion
  if (chi::IpcManager::IsWarpScheduler()) {
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

// Alloc kernel
__global__ void gs_cte_alloc_kernel(
    hipc::MemoryBackend data_backend, chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x!=0||blockIdx.x!=0) return;
  using AllocT=hipc::PrivateBuddyAllocator;
  auto *alloc=data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr=alloc->AllocateObjs<char>(total_bytes);
}

// ================================================================
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache_host.h>
#endif
#include <vector>
#include <cstring>

static void gs_init(float *u, float *v, int L) {
  uint32_t total = L*L*L;
  for (uint32_t i = 0; i < total; i++) { u[i]=1.0f; v[i]=0.0f; }
  int lo=L/4, hi=3*L/4;
  for (int z=lo;z<hi;z++) for (int y=lo;y<hi;y++) for (int x=lo;x<hi;x++) {
    uint32_t idx = x+y*L+z*L*L; u[idx]=0.75f; v[idx]=0.25f;
  }
}

int run_workload_gray_scott(const WorkloadConfig &cfg, const char *mode,
                            WorkloadResult *result) {
  // Scale grid_size so that per-warp field data ≈ warp_bytes.
  // Each warp stores 4 fields (u, v, u2, v2) × (L^3/warps) floats.
  // Per-warp bytes = (L^3 / warps) * 4 * sizeof(float) ≈ warp_bytes.
  // Solve for L: L^3 = warps * warp_bytes / (4 * sizeof(float))
  uint32_t est_warps = (cfg.client_blocks * cfg.client_threads) / 32;
  if (est_warps == 0) est_warps = 1;
  int L = cfg.param_grid_size;
  if (cfg.warp_bytes > 0) {
    // Per-warp data = L^3/warps * 4 fields * 4 bytes = warp_bytes
    // L^3 = warps * warp_bytes / 16
    double target_total = (double)est_warps * cfg.warp_bytes / (4.0 * sizeof(float));
    int target_L = (int)cbrt(target_total);
    if (target_L < 4) target_L = 4;
    // Cap to avoid OOM: total field data = L^3 * 4 * 4 bytes, keep under 32 MB
    while ((double)target_L * target_L * target_L * 16.0 > 32.0 * 1024 * 1024 && target_L > 4) {
      target_L--;
    }
    L = target_L;
  }
  int steps = cfg.param_steps;
  uint32_t total = (uint32_t)L*L*L;
  size_t fb = total * sizeof(float);
  GSParams params;
  int threads = 256, blocks = (total+threads-1)/threads;
  std::string m(mode);

  HIPRINT("  Gray-Scott: L={} ({} points), {} steps", L, total, steps);

  if (m == "cte") {
    // ======== CTE: Combined kernel with multi-warp I/O + compute ========
    uint32_t total_warps = (cfg.client_blocks * cfg.client_threads) / 32;
    if (total_warps == 0) total_warps = 1;

    // Partition grid points among warps
    uint32_t points_per_warp = (total + total_warps - 1) / total_warps;
    std::vector<uint32_t> h_point_start(total_warps), h_point_end(total_warps);
    std::vector<uint64_t> h_field_offsets(total_warps), h_field_bytes(total_warps);

    uint64_t total_field_bytes = 0;
    for (uint32_t w = 0; w < total_warps; w++) {
      h_point_start[w] = w * points_per_warp;
      h_point_end[w] = std::min((w + 1) * points_per_warp, total);
      uint32_t batch_size = h_point_end[w] - h_point_start[w];
      h_field_offsets[w] = total_field_bytes;
      h_field_bytes[w] = (uint64_t)batch_size * 4 * sizeof(float);
      total_field_bytes += h_field_bytes[w];
    }

    HIPRINT("  CTE: {} warps, {} points/warp, {:.1f} MB fields/step",
            total_warps, points_per_warp, total_field_bytes/(1024.0*1024.0));

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    // Data backend
    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, total_field_bytes + 4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, (size_t)total_warps*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, (size_t)total_warps*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gs_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),
                                  total_field_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError,"GS CTE alloc failed"); cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator();
      return -2;
    }
    hipc::FullPtr<char> array_ptr = *d_ptr; cudaFreeHost(d_ptr);
    hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
    CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_, data_backend.data_capacity_);

    chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;
    int *d_done; cudaMallocHost(&d_done, sizeof(int)); *d_done = 0;
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Copy partition info to pinned memory
    uint64_t *h_fo, *h_fb; uint32_t *h_ps, *h_pe;
    cudaMallocHost(&h_fo, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_fb, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_ps, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_pe, total_warps*sizeof(uint32_t));
    memcpy(h_fo, h_field_offsets.data(), total_warps*sizeof(uint64_t));
    memcpy(h_fb, h_field_bytes.data(), total_warps*sizeof(uint64_t));
    memcpy(h_ps, h_point_start.data(), total_warps*sizeof(uint32_t));
    memcpy(h_pe, h_point_end.data(), total_warps*sizeof(uint32_t));

    // Seed per-warp u,v fields
    std::vector<float> hu(total), hv(total);
    gs_init(hu.data(), hv.data(), L);

    char *h_all_fields = (char*)malloc(total_field_bytes);
    for (uint32_t w = 0; w < total_warps; w++) {
      uint32_t point_start = h_point_start[w];
      uint32_t batch_size = h_point_end[w] - point_start;
      uint64_t half_bytes = h_field_bytes[w] / 4;

      // Copy u for this warp's batch
      for (uint32_t i = 0; i < batch_size; i++) {
        float *dst = (float*)(h_all_fields + h_field_offsets[w]) + i;
        float *src = hu.data() + (point_start + i);
        *dst = *src;
      }
      // Copy v for this warp's batch
      for (uint32_t i = 0; i < batch_size; i++) {
        float *dst = (float*)(h_all_fields + h_field_offsets[w] + half_bytes) + i;
        float *src = hv.data() + (point_start + i);
        *dst = *src;
      }
    }
    cudaMemcpy(array_ptr.ptr_, h_all_fields, total_field_bytes, cudaMemcpyHostToDevice);
    free(h_all_fields);
    cudaDeviceSynchronize();

    // Seed each warp's blobs via CTE client
    {
      wrp_cte::core::Client cte_client(cfg.cte_pool_id);
      for (uint32_t w = 0; w < total_warps; w++) {
        // Seed u blob
        char bname_u[32];
        int pos = 0;
        const char *pfx = "gs_u_w";
        while (*pfx) bname_u[pos++] = *pfx++;
        pos += std::to_string(w).copy(bname_u + pos, 32 - pos);
        bname_u[pos] = '\0';

        uint64_t half_bytes = h_field_bytes[w] / 4;
        hipc::ShmPtr<> shm_u;
        shm_u.alloc_id_ = data_alloc_id;
        shm_u.off_.exchange(array_ptr.shm_.off_.load() + h_field_offsets[w]);
        auto f_u = cte_client.AsyncPutBlob(cfg.tag_id, bname_u,
            (chi::u64)0, half_bytes, shm_u, -1.0f,
            wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
        f_u.Wait();

        // Seed v blob
        char bname_v[32];
        pos = 0;
        pfx = "gs_v_w";
        while (*pfx) bname_v[pos++] = *pfx++;
        pos += std::to_string(w).copy(bname_v + pos, 32 - pos);
        bname_v[pos] = '\0';

        hipc::ShmPtr<> shm_v;
        shm_v.alloc_id_ = data_alloc_id;
        shm_v.off_.exchange(array_ptr.shm_.off_.load() + h_field_offsets[w] + half_bytes);
        auto f_v = cte_client.AsyncPutBlob(cfg.tag_id, bname_v,
            (chi::u64)0, half_bytes, shm_v, -1.0f,
            wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
        f_v.Wait();
      }
      HIPRINT("  CTE: Seeded {} per-warp u,v blobs", total_warps);
    }

    // Clear data backend before GetBlob
    cudaMemset(array_ptr.ptr_, 0, total_field_bytes);
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Run combined CTE kernel for each step
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; step++) {
      *d_done = 0;
      if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      void *stream = hshm::GpuApi::CreateStream();
      gs_cte_kernel<<<cfg.client_blocks, cfg.client_threads, 0,
                      static_cast<cudaStream_t>(stream)>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.client_blocks,
          array_ptr, data_alloc_id,
          h_fo, h_fb, h_ps, h_pe,
          total_warps, L, d_done);

      CHI_IPC->ResumeGpuOrchestrator();
      auto *orch = static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
      auto *ctrl = orch ? orch->control_ : nullptr;
      if(ctrl){int w=0;while(ctrl->running_flag==0&&w<5000){std::this_thread::sleep_for(std::chrono::milliseconds(1));++w;}}
      int64_t tus=(int64_t)cfg.timeout_sec*1000000,el=0;
      while(__atomic_load_n(d_done,__ATOMIC_ACQUIRE)<(int)total_warps&&el<tus){
        std::this_thread::sleep_for(std::chrono::microseconds(100));el+=100;}
      CHI_IPC->PauseGpuOrchestrator();
      hshm::GpuApi::Synchronize(stream);
      hshm::GpuApi::DestroyStream(stream);

      if(__atomic_load_n(d_done,__ATOMIC_ACQUIRE)<(int)total_warps){
        HLOG(kError,"GS CTE step {} timed out",step); break;}
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps = (double)total * sizeof(float) * (2*7+2);
    result->bandwidth_gbps = (bps*steps/1e9) / (result->elapsed_ms/1e3);
    result->primary_metric = result->elapsed_ms / steps;
    result->metric_name = "ms/step";

    cudaFreeHost(d_done); cudaFreeHost(h_fo); cudaFreeHost(h_fb);
    cudaFreeHost(h_ps); cudaFreeHost(h_pe);
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m == "bam") {
    uint64_t fb_aligned = ((fb+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    uint32_t total_pages = (uint32_t)(fb_aligned / cfg.bam_page_size);
    uint64_t hbm_bytes = cfg.GetHbmBytes();
    uint32_t cache_pages = (hbm_bytes > 0)
        ? std::max(1u, (uint32_t)(hbm_bytes / cfg.bam_page_size))
        : total_pages;
    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cache_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;

    bam::PageCache cache_u(pcfg), cache_v(pcfg);
    bam::Array<float> u_arr(total, cache_u);
    bam::Array<float> v_arr(total, cache_v);

    std::vector<float> hu(total), hv(total);
    gs_init(hu.data(), hv.data(), L);
    u_arr.load_from_host(hu.data(), total);
    v_arr.load_from_host(hv.data(), total);

    float *d_u2, *d_v2;
    cudaMalloc(&d_u2, fb); cudaMalloc(&d_v2, fb);

    HIPRINT("  BaM HBM cache: {} / {} pages ({}%) x {} B = {:.1f} MB per field",
            cache_pages, total_pages,
            cache_pages * 100 / std::max(1u, total_pages), cfg.bam_page_size,
            (double)cache_pages*cfg.bam_page_size/(1024.0*1024.0));

    auto t0=std::chrono::high_resolution_clock::now();
    for(int s=0;s<steps;s++){
      gs_stencil_bam<<<blocks,threads>>>(u_arr.device(), v_arr.device(),
                                          d_u2, d_v2, L, params);
      cudaDeviceSynchronize();
      cudaMemcpy(hu.data(), d_u2, fb, cudaMemcpyDeviceToHost);
      cudaMemcpy(hv.data(), d_v2, fb, cudaMemcpyDeviceToHost);
      u_arr.load_from_host(hu.data(), total);
      v_arr.load_from_host(hv.data(), total);
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps = (double)total * sizeof(float) * (2*7+2);
    result->bandwidth_gbps = (bps*steps/1e9) / (result->elapsed_ms/1e3);
    result->primary_metric = result->elapsed_ms / steps;
    result->metric_name = "ms/step";
    cudaFree(d_u2); cudaFree(d_v2);
  }
#endif

  else if (m=="hbm") {
    float *d_u,*d_v,*d_u2,*d_v2;
    cudaMalloc(&d_u,fb); cudaMalloc(&d_v,fb); cudaMalloc(&d_u2,fb); cudaMalloc(&d_v2,fb);
    std::vector<float> hu(total), hv(total); gs_init(hu.data(),hv.data(),L);
    cudaMemcpy(d_u,hu.data(),fb,cudaMemcpyHostToDevice); cudaMemcpy(d_v,hv.data(),fb,cudaMemcpyHostToDevice);
    auto t0=std::chrono::high_resolution_clock::now();
    for (int s=0;s<steps;s++) {
      gs_stencil_hbm<<<blocks,threads>>>(d_u,d_v,d_u2,d_v2,L,params);
      std::swap(d_u,d_u2); std::swap(d_v,d_v2);
    }
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps = (double)total * sizeof(float) * (2*7+2);
    result->bandwidth_gbps = (bps*steps/1e9) / (result->elapsed_ms/1e3);
    result->primary_metric = result->elapsed_ms / steps;
    result->metric_name = "ms/step";
    cudaFree(d_u);cudaFree(d_v);cudaFree(d_u2);cudaFree(d_v2);
  } else if (m=="direct") {
    float *h_u,*h_v,*h_u2,*h_v2;
    cudaMallocHost(&h_u,fb);cudaMallocHost(&h_v,fb);cudaMallocHost(&h_u2,fb);cudaMallocHost(&h_v2,fb);
    gs_init(h_u,h_v,L);
    auto t0=std::chrono::high_resolution_clock::now();
    for (int s=0;s<steps;s++) {
      gs_stencil_direct<<<blocks,threads>>>(h_u,h_v,h_u2,h_v2,L,params); cudaDeviceSynchronize();
      std::swap(h_u,h_u2); std::swap(h_v,h_v2);
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps = (double)total * sizeof(float) * (2*7+2);
    result->bandwidth_gbps = (bps*steps/1e9) / (result->elapsed_ms/1e3);
    result->primary_metric = result->elapsed_ms / steps;
    result->metric_name = "ms/step";
    cudaFreeHost(h_u);cudaFreeHost(h_v);cudaFreeHost(h_u2);cudaFreeHost(h_v2);
  } else { HLOG(kError, "gray_scott: unknown mode '{}'", mode); return -1; }
  return 0;
}

#endif  // HSHM_IS_HOST
