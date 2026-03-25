/**
 * workload_gray_scott.cc — Gray-Scott stencil simulation for CTE GPU bench
 *
 * CTE mode: Stencil compute in HBM, checkpoints via AsyncPutBlob.
 * BaM mode: Stencil reads u/v through BaM page cache from DRAM.
 * HBM mode: Stencil + cudaMemcpy checkpoint.
 * Direct mode: All data in pinned DRAM.
 */

// GPU kernels (visible in both host and device passes)
#include <cstdint>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache.cuh>
#include <bam/types.h>
#endif

struct GSParams { float Du=0.05f, Dv=0.1f, F=0.04f, k=0.06075f, dt=0.2f; };

__device__ inline uint32_t gs_idx(int x, int y, int z, int L) {
  return (uint32_t)(((x+L)%L) + ((y+L)%L)*L + ((z+L)%L)*L*L);
}

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
 * BaM stencil: reads u/v through the page cache from DRAM.
 * Per-thread page_cache_acquire (stencil neighbors may span pages).
 * Writes go to HBM output arrays (double-buffered in HBM).
 */
__device__ inline float gs_bam_read(bam::PageCacheDeviceState &cache,
                                     const uint8_t *host_base,
                                     uint32_t idx) {
  uint64_t byte_off = (uint64_t)idx * sizeof(float);
  uint64_t page_off = byte_off & ~((uint64_t)cache.page_size - 1);
  uint32_t in_page = (uint32_t)(byte_off & ((uint64_t)cache.page_size - 1));

  bool needs_load;
  uint8_t *page = bam::page_cache_acquire(cache, page_off, &needs_load);
  if (needs_load) {
    bam::host_read_page(page, host_base, page_off, cache.page_size);
    bam::page_cache_finish_load(cache, page_off);
  }
  return *reinterpret_cast<const float *>(page + in_page);
}

__global__ void gs_stencil_bam(bam::PageCacheDeviceState u_cache,
                                const uint8_t *u_host,
                                bam::PageCacheDeviceState v_cache,
                                const uint8_t *v_host,
                                float *u2, float *v2, int L, GSParams p) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint32_t)(L*L*L)) return;
  int z=tid/(L*L), y=(tid/L)%L, x=tid%L;

  float lu = gs_bam_read(u_cache, u_host, gs_idx(x-1,y,z,L))
           + gs_bam_read(u_cache, u_host, gs_idx(x+1,y,z,L))
           + gs_bam_read(u_cache, u_host, gs_idx(x,y-1,z,L))
           + gs_bam_read(u_cache, u_host, gs_idx(x,y+1,z,L))
           + gs_bam_read(u_cache, u_host, gs_idx(x,y,z-1,L))
           + gs_bam_read(u_cache, u_host, gs_idx(x,y,z+1,L))
           - 6.0f * gs_bam_read(u_cache, u_host, tid);
  lu /= 6.0f;

  float lv = gs_bam_read(v_cache, v_host, gs_idx(x-1,y,z,L))
           + gs_bam_read(v_cache, v_host, gs_idx(x+1,y,z,L))
           + gs_bam_read(v_cache, v_host, gs_idx(x,y-1,z,L))
           + gs_bam_read(v_cache, v_host, gs_idx(x,y+1,z,L))
           + gs_bam_read(v_cache, v_host, gs_idx(x,y,z-1,L))
           + gs_bam_read(v_cache, v_host, gs_idx(x,y,z+1,L))
           - 6.0f * gs_bam_read(v_cache, v_host, tid);
  lv /= 6.0f;

  float uv = gs_bam_read(u_cache, u_host, tid);
  float vv = gs_bam_read(v_cache, v_host, tid);
  float uvv = uv * vv * vv;
  u2[tid] = uv + p.dt * (p.Du * lu - uvv + p.F * (1.0f - uv));
  v2[tid] = vv + p.dt * (p.Dv * lv + uvv - (p.F + p.k) * vv);
}
#endif  // WRP_CORE_ENABLE_BAM

// CTE checkpoint kernel
__global__ void gs_cte_ckpt_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId data_alloc_id,
    chi::u64 data_bytes, chi::u32 step_num, int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  if (warp_id == 0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client cte_client(cte_pool_id);
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
    char name[32]; int pos = 0;
    const char *pfx = "gs_"; while (*pfx) name[pos++] = *pfx++;
    pos += StrT::NumberToStr(name + pos, 32 - pos, step_num);
    name[pos] = '\0';
    hipc::ShmPtr<> blob_shm;
    blob_shm.alloc_id_ = data_alloc_id;
    blob_shm.off_.exchange(data_ptr.shm_.off_.load());
    auto future = cte_client.AsyncPutBlob(tag_id, name, (chi::u64)0, data_bytes,
        blob_shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
    if (!future.GetFutureShmPtr().IsNull()) future.Wait();
    atomicAdd_system(d_done, 1); __threadfence_system();
  }
}

__global__ void gs_cte_alloc_kernel(hipc::MemoryBackend data_backend,
    chi::u64 total_bytes, hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

/** GPU-initiated GetBlob: load u and v fields from CTE. */
__global__ void gs_cte_getblob_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId alloc_id,
    chi::u64 field_bytes, const char *u_name, const char *v_name,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  if (chi::IpcManager::GetWarpId()==0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client c(pool_id);
    // GetBlob u field (first half of data buffer)
    hipc::ShmPtr<> u_shm; u_shm.alloc_id_=alloc_id;
    u_shm.off_.exchange(data_ptr.shm_.off_.load());
    auto f1=c.AsyncGetBlob(tag_id, u_name, (chi::u64)0, field_bytes,
        (chi::u32)0, u_shm, chi::PoolQuery::Local());
    if (!f1.GetFutureShmPtr().IsNull()) f1.Wait();

    // GetBlob v field (second half)
    hipc::ShmPtr<> v_shm; v_shm.alloc_id_=alloc_id;
    v_shm.off_.exchange(data_ptr.shm_.off_.load() + field_bytes);
    auto f2=c.AsyncGetBlob(tag_id, v_name, (chi::u64)0, field_bytes,
        (chi::u32)0, v_shm, chi::PoolQuery::Local());
    if (!f2.GetFutureShmPtr().IsNull()) f2.Wait();

    atomicAdd_system(d_done, 1); __threadfence_system();
  }
}

/** GPU-initiated PutBlob: write u2 and v2 fields to CTE. */
__global__ void gs_cte_putblob_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId alloc_id,
    chi::u64 field_bytes, const char *u_name, const char *v_name,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  if (chi::IpcManager::GetWarpId()==0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client c(pool_id);
    // PutBlob u2 (third quarter of buffer)
    hipc::ShmPtr<> u_shm; u_shm.alloc_id_=alloc_id;
    u_shm.off_.exchange(data_ptr.shm_.off_.load() + 2*field_bytes);
    auto f1=c.AsyncPutBlob(tag_id, u_name, (chi::u64)0, field_bytes,
        u_shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
    if (!f1.GetFutureShmPtr().IsNull()) f1.Wait();

    // PutBlob v2 (fourth quarter)
    hipc::ShmPtr<> v_shm; v_shm.alloc_id_=alloc_id;
    v_shm.off_.exchange(data_ptr.shm_.off_.load() + 3*field_bytes);
    auto f2=c.AsyncPutBlob(tag_id, v_name, (chi::u64)0, field_bytes,
        v_shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
    if (!f2.GetFutureShmPtr().IsNull()) f2.Wait();

    atomicAdd_system(d_done, 1); __threadfence_system();
  }
}

// Host-only code
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#include <vector>
#include <cstring>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache_host.h>
#endif

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
  int L = cfg.param_grid_size;
  int steps = cfg.param_steps;
  int ckpt = cfg.param_checkpoint_freq;
  uint32_t total = L*L*L;
  size_t fb = total * sizeof(float);
  GSParams params;
  int threads = 256, blocks = (total+threads-1)/threads;
  std::string m(mode);

#ifdef WRP_CORE_ENABLE_BAM
  if (m == "bam") {
    // ---- BaM mode: stencil reads through HBM page cache from DRAM ----
    // Output arrays in HBM (double buffered)
    float *d_u2, *d_v2;
    cudaMalloc(&d_u2, fb); cudaMalloc(&d_v2, fb);

    // Page cache sized to match CTE HBM tier (bam_cache_pages * page_size)
    bam::PageCacheConfig u_cfg, v_cfg;
    u_cfg.page_size = cfg.bam_page_size;
    u_cfg.num_pages = cfg.bam_cache_pages;
    u_cfg.num_queues = 0; u_cfg.queue_depth = 0;
    u_cfg.backend = bam::BackendType::kHostMemory;
    u_cfg.nvme_dev = nullptr;
    v_cfg = u_cfg;

    bam::PageCache u_cache(u_cfg), v_cache(v_cfg);

    // Backing store: page-aligned DRAM
    uint64_t fb_aligned = ((fb + cfg.bam_page_size - 1) / cfg.bam_page_size) * cfg.bam_page_size;
    u_cache.alloc_host_backing(fb_aligned);
    v_cache.alloc_host_backing(fb_aligned);

    // Initialize fields
    std::vector<float> hu(total), hv(total);
    gs_init(hu.data(), hv.data(), L);
    memcpy(u_cache.host_buffer(), hu.data(), fb);
    memcpy(v_cache.host_buffer(), hv.data(), fb);

    HIPRINT("  BaM cache: {} pages x {} B = {:.1f} MB per field",
            cfg.bam_cache_pages, cfg.bam_page_size,
            (double)cfg.bam_cache_pages * cfg.bam_page_size / (1024.0 * 1024.0));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
      gs_stencil_bam<<<blocks, threads>>>(
          u_cache.device_state(), u_cache.host_buffer(),
          v_cache.device_state(), v_cache.host_buffer(),
          d_u2, d_v2, L, params);
      cudaDeviceSynchronize();

      // Copy output back to DRAM backing store for next iteration
      cudaMemcpy(u_cache.host_buffer(), d_u2, fb, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_cache.host_buffer(), d_v2, fb, cudaMemcpyDeviceToHost);

      // Reset cache tags so next iteration re-fetches from updated DRAM
      cudaMemset(u_cache.device_state().page_tags, 0xFF,
                 cfg.bam_cache_pages * sizeof(uint64_t));
      cudaMemset(u_cache.device_state().page_states, 0,
                 cfg.bam_cache_pages * sizeof(uint32_t));
      cudaMemset(v_cache.device_state().page_tags, 0xFF,
                 cfg.bam_cache_pages * sizeof(uint64_t));
      cudaMemset(v_cache.device_state().page_states, 0,
                 cfg.bam_cache_pages * sizeof(uint32_t));
      cudaDeviceSynchronize();
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double bps = (double)total * sizeof(float) * (2*7+2);
    result->bandwidth_gbps = (bps*steps/1e9) / (result->elapsed_ms/1e3);
    result->primary_metric = result->elapsed_ms / steps;
    result->metric_name = "ms/step";
    cudaFree(d_u2); cudaFree(d_v2);
    return 0;
  }
#endif  // WRP_CORE_ENABLE_BAM

  if (m == "cte") {
    // ======== CTE: GetBlob u,v → stencil → PutBlob u2,v2 per step ========
    // Data backend layout: [u | v | u2 | v2] = 4*fb bytes
    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, 4*fb+4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, 1*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, 1*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gs_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend), 4*fb, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) { cudaFreeHost(d_ptr); CHI_IPC->ResumeGpuOrchestrator(); return -2; }
    hipc::FullPtr<char> array_ptr = *d_ptr; cudaFreeHost(d_ptr);
    hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
    CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_, data_backend.data_capacity_);
    chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;
    int *d_done; cudaMallocHost(&d_done, sizeof(int));
    if (scratch_backend.data_) cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
    if (heap_backend.data_) cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Blob names
    char *h_u_name, *h_v_name;
    cudaMallocHost(&h_u_name, 32); strcpy(h_u_name, "gs_u");
    cudaMallocHost(&h_v_name, 32); strcpy(h_v_name, "gs_v");

    // Initialize and seed u,v into CTE
    std::vector<float> hu(total), hv(total);
    gs_init(hu.data(), hv.data(), L);
    cudaMemcpy(array_ptr.ptr_, hu.data(), fb, cudaMemcpyHostToDevice);
    cudaMemcpy(array_ptr.ptr_ + fb, hv.data(), fb, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto gs_cycle = [&]() -> bool {
      CHI_IPC->ResumeGpuOrchestrator();
      auto *o=static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
      auto *c=o?o->control_:nullptr;
      if(c){int w=0;while(c->running_flag==0&&w<5000){std::this_thread::sleep_for(std::chrono::milliseconds(1));++w;}}
      int64_t tus=(int64_t)cfg.timeout_sec*1000000,el=0;
      while(__atomic_load_n(d_done,__ATOMIC_ACQUIRE)<1&&el<tus){std::this_thread::sleep_for(std::chrono::microseconds(100));el+=100;}
      bool ok=__atomic_load_n(d_done,__ATOMIC_ACQUIRE)>=1;
      CHI_IPC->PauseGpuOrchestrator();
      return ok;
    };

    auto reset_scratch = [&]() {
      if(scratch_backend.data_)cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();
    };

    // Seed: PutBlob initial u,v
    HIPRINT("  CTE: Seeding u,v via AsyncPutBlob...");
    reset_scratch(); *d_done=0;
    // Use gs_cte_putblob_kernel to write u (at offset 0) and v (at offset fb)
    // But putblob writes from offsets 2*fb and 3*fb. Copy to those positions first.
    cudaMemcpy(array_ptr.ptr_+2*fb, hu.data(), fb, cudaMemcpyHostToDevice);
    cudaMemcpy(array_ptr.ptr_+3*fb, hv.data(), fb, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gs_cte_putblob_kernel<<<cfg.rt_blocks, cfg.rt_threads>>>(
        gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.rt_blocks,
        array_ptr, data_alloc_id, fb, h_u_name, h_v_name, d_done);
    if (!gs_cycle()) { HLOG(kError, "GS CTE seed timed out"); goto gs_cleanup; }
    HIPRINT("  CTE: Seeded.");

    // Timed loop
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int s = 0; s < steps; s++) {
        // GetBlob u,v into data backend (first 2*fb bytes)
        reset_scratch(); *d_done=0;
        gs_cte_getblob_kernel<<<cfg.rt_blocks, cfg.rt_threads>>>(
            gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.rt_blocks,
            array_ptr, data_alloc_id, fb, h_u_name, h_v_name, d_done);
        if (!gs_cycle()) { HLOG(kError, "GS CTE GetBlob step {} timed out", s); break; }

        // Compute stencil: u,v at offsets 0,fb → u2,v2 at offsets 2*fb,3*fb
        float *d_u = reinterpret_cast<float*>(array_ptr.ptr_);
        float *d_v = reinterpret_cast<float*>(array_ptr.ptr_ + fb);
        float *d_u2 = reinterpret_cast<float*>(array_ptr.ptr_ + 2*fb);
        float *d_v2 = reinterpret_cast<float*>(array_ptr.ptr_ + 3*fb);
        gs_stencil_hbm<<<blocks,threads>>>(d_u, d_v, d_u2, d_v2, L, params);
        cudaDeviceSynchronize();

        // PutBlob u2,v2 (from offsets 2*fb,3*fb) → these become the new u,v for next step
        reset_scratch(); *d_done=0;
        gs_cte_putblob_kernel<<<cfg.rt_blocks, cfg.rt_threads>>>(
            gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.rt_blocks,
            array_ptr, data_alloc_id, fb, h_u_name, h_v_name, d_done);
        if (!gs_cycle()) { HLOG(kError, "GS CTE PutBlob step {} timed out", s); break; }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
      double bps = (double)total*sizeof(float)*(2*7+2);
      result->bandwidth_gbps = (bps*steps/1e9)/(result->elapsed_ms/1e3);
      result->primary_metric = result->elapsed_ms/steps;
      result->metric_name = "ms/step";
    }
gs_cleanup:
    cudaFreeHost(d_done); cudaFreeHost(h_u_name); cudaFreeHost(h_v_name);
    return 0;

  } else if (m == "hbm") {
    float *d_u,*d_v,*d_u2,*d_v2;
    cudaMalloc(&d_u,fb); cudaMalloc(&d_v,fb); cudaMalloc(&d_u2,fb); cudaMalloc(&d_v2,fb);
    std::vector<float> hu(total), hv(total); gs_init(hu.data(),hv.data(),L);
    cudaMemcpy(d_u,hu.data(),fb,cudaMemcpyHostToDevice); cudaMemcpy(d_v,hv.data(),fb,cudaMemcpyHostToDevice);
    float *h_ckpt; cudaMallocHost(&h_ckpt, fb*2);
    auto t0=std::chrono::high_resolution_clock::now();
    for (int s=0;s<steps;s++) {
      gs_stencil_hbm<<<blocks,threads>>>(d_u,d_v,d_u2,d_v2,L,params);
      std::swap(d_u,d_u2); std::swap(d_v,d_v2);
      if (ckpt>0 && (s+1)%ckpt==0) { cudaMemcpy(h_ckpt,d_u,fb,cudaMemcpyDeviceToHost); cudaMemcpy(h_ckpt+total,d_v,fb,cudaMemcpyDeviceToHost); }
    }
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->bandwidth_gbps=((double)total*sizeof(float)*(2*7+2)*steps/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=result->elapsed_ms/steps; result->metric_name="ms/step";
    cudaFree(d_u);cudaFree(d_v);cudaFree(d_u2);cudaFree(d_v2);cudaFreeHost(h_ckpt);

  } else if (m == "direct") {
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
    result->bandwidth_gbps=((double)total*sizeof(float)*(2*7+2)*steps/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=result->elapsed_ms/steps; result->metric_name="ms/step";
    cudaFreeHost(h_u);cudaFreeHost(h_v);cudaFreeHost(h_u2);cudaFreeHost(h_v2);
  } else { HLOG(kError, "gray_scott: unknown mode '{}'", mode); return -1; }
  return 0;
}

#endif  // HSHM_IS_HOST
