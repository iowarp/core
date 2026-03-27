/**
 * workload_pagerank.cc — PageRank for CTE GPU bench
 *
 * CTE mode: Single combined kernel. Each warp owns a vertex range.
 *   Lane 0 calls AsyncGetBlob to load edges for its vertices.
 *   All 32 lanes compute the push (the science).
 *   Lane 0 calls AsyncPutBlob to write back updated values.
 *
 * BaM mode: Uses bam::ArrayDevice<uint32_t>::read() for edge access.
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

// --- HBM/Direct compute kernels (unchanged) ---

__global__ void pr_push_hbm(const uint64_t *offsets, const uint32_t *edges,
                             const float *values, float *residuals,
                             uint32_t nv, float alpha) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t v=wid;v<nv;v+=(blockDim.x*gridDim.x)/32) {
    uint64_t s=offsets[v],e=offsets[v+1]; uint32_t deg=(uint32_t)(e-s);
    if (!deg) continue; float c=alpha*values[v]/deg;
    for (uint64_t i=s+lid;i<e;i+=32) atomicAdd(&residuals[edges[i]],c);
    __syncwarp();
  }
}

__global__ void pr_push_direct(const uint64_t *offsets, const uint32_t *h_edges,
                                const float *values, float *residuals,
                                uint32_t nv, float alpha) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t v=wid;v<nv;v+=(blockDim.x*gridDim.x)/32) {
    uint64_t s=offsets[v],e=offsets[v+1]; uint32_t deg=(uint32_t)(e-s);
    if (!deg) continue; float c=alpha*values[v]/deg;
    for (uint64_t i=s+lid;i<e;i+=32) atomicAdd(&residuals[h_edges[i]],c);
    __syncwarp();
  }
}

#ifdef WRP_CORE_ENABLE_BAM
/**
 * BaM PageRank push: reads edges through bam::ArrayDevice<uint32_t>.
 * Each read() call transparently goes through the HBM page cache.
 */
__global__ void pr_push_bam(const uint64_t *offsets,
                             bam::ArrayDevice<uint32_t> edges,
                             const float *values, float *residuals,
                             uint32_t nv, float alpha) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t v=wid;v<nv;v+=(blockDim.x*gridDim.x)/32) {
    uint64_t s=offsets[v],e=offsets[v+1]; uint32_t deg=(uint32_t)(e-s);
    if (!deg) continue; float c=alpha*values[v]/deg;
    for (uint64_t i=s+lid;i<e;i+=32) {
      uint32_t neighbor = edges.read(i);
      atomicAdd(&residuals[neighbor],c);
    }
    __syncwarp();
  }
}
#endif

__global__ void pr_update(float *values, float *residuals,
                           uint32_t nv, float tol, int *active) {
  uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x;
  if (tid>=nv) return;
  float r=residuals[tid];
  if (fabsf(r)>tol) { values[tid]+=r; atomicAdd(active,1); }
  residuals[tid]=0.0f;
}

/**
 * Combined CTE PageRank kernel: I/O + compute in one kernel.
 *
 * Each warp owns a range of vertices. Per iteration:
 *   1. Lane 0: AsyncGetBlob loads this warp's edge chunk
 *   2. All 32 lanes: push residuals to neighbors (the science)
 *   3. Lane 0: AsyncPutBlob writes back updated data
 *
 * Data layout in data backend:
 *   [warp_0 edges | warp_1 edges | ... | warp_N edges]
 *   Each warp's slice = edges for vertices [v_start, v_end)
 */
__global__ void pr_cte_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    // Data
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    const uint64_t *d_offsets,        // CSR offsets (in HBM)
    const float *d_values,            // PR scores (in HBM)
    float *d_residuals,               // Residuals (in HBM, atomics)
    // Per-warp partition info (pinned host arrays)
    const uint64_t *warp_edge_offsets, // byte offset into data_ptr for each warp's edges
    const uint64_t *warp_edge_bytes,   // byte count for each warp's edges
    const uint32_t *warp_v_start,      // first vertex for each warp
    const uint32_t *warp_v_end,        // last+1 vertex for each warp
    chi::u32 total_warps,
    float alpha,
    int *d_done) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    uint32_t v_start = warp_v_start[warp_id];
    uint32_t v_end = warp_v_end[warp_id];
    chi::u64 my_edge_offset = warp_edge_offsets[warp_id];
    chi::u64 my_edge_bytes = warp_edge_bytes[warp_id];
    char *my_data = data_ptr.ptr_ + my_edge_offset;

    // Build blob name: "pr_w<warp_id>"
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
    char name_buf[32];
    int pos = 0;
    const char *pfx = "pr_w";
    while (*pfx) name_buf[pos++] = *pfx++;
    pos += StrT::NumberToStr(name_buf + pos, 32 - pos, warp_id);
    name_buf[pos] = '\0';

    bool alloc_failed = false;

    // === I/O: GetBlob — load this warp's edges from CTE ===
    if (chi::IpcManager::IsWarpScheduler() && my_edge_bytes > 0) {
      wrp_cte::core::Client cte_client(cte_pool_id);
      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(data_ptr.shm_.off_.load() + my_edge_offset);

      auto get_future = cte_client.AsyncGetBlob(
          tag_id, name_buf, (chi::u64)0, my_edge_bytes,
          (chi::u32)0, shm, chi::PoolQuery::Local());
      if (!get_future.GetFutureShmPtr().IsNull()) {
        get_future.Wait();
      } else {
        alloc_failed = true;
      }
    }
    __syncwarp();

    // === COMPUTE: All 32 lanes push residuals (THE SCIENCE) ===
    if (!alloc_failed) {
      uint64_t edge_base = d_offsets[v_start]; // global edge index of first edge in this warp's range
      const uint32_t *my_edges = reinterpret_cast<const uint32_t *>(my_data);

      for (uint32_t v = v_start; v < v_end; v++) {
        uint64_t s = d_offsets[v] - edge_base;
        uint64_t e = d_offsets[v + 1] - edge_base;
        uint32_t deg = (uint32_t)(e - s);
        if (deg == 0) continue;
        float contribution = alpha * d_values[v] / deg;

        for (uint64_t i = s + lane_id; i < e; i += 32) {
          uint32_t neighbor = my_edges[i];
          atomicAdd(&d_residuals[neighbor], contribution);
        }
        __syncwarp();
      }
    }

    // === I/O: PutBlob — write back edges (round-trip for benchmarking) ===
    if (chi::IpcManager::IsWarpScheduler() && !alloc_failed && my_edge_bytes > 0) {
      wrp_cte::core::Client cte_client(cte_pool_id);
      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(data_ptr.shm_.off_.load() + my_edge_offset);

      auto put_future = cte_client.AsyncPutBlob(
          tag_id, name_buf, (chi::u64)0, my_edge_bytes,
          shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0,
          chi::PoolQuery::Local());
      if (!put_future.GetFutureShmPtr().IsNull()) {
        put_future.Wait();
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
__global__ void pr_cte_alloc_kernel(
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
#include <random>
#include <algorithm>
#include <cstring>

struct CSRGraph { std::vector<uint64_t> offsets; std::vector<uint32_t> edges; uint32_t nv; uint64_t ne; };

static CSRGraph gen_rmat(uint32_t nv, uint32_t ad, uint64_t seed=42) {
  CSRGraph g; g.nv=nv;
  std::mt19937_64 rng(seed); std::vector<std::vector<uint32_t>> adj(nv);
  uint64_t target=(uint64_t)nv*ad; double a=0.57,b=0.19,c=0.19;
  uint32_t log2n=0; for(uint32_t n=nv;n>1;n>>=1) log2n++;
  std::uniform_real_distribution<double> dist(0.0,1.0);
  for(uint64_t e=0;e<target;e++){
    uint32_t u=0,v=0;
    for(uint32_t l=0;l<log2n;l++){double r=dist(rng);uint32_t h=1u<<(log2n-l-1);
      if(r<a){}else if(r<a+b)v+=h;else if(r<a+b+c)u+=h;else{u+=h;v+=h;}}
    u%=nv;v%=nv;if(u!=v)adj[u].push_back(v);
  }
  g.offsets.resize(nv+1);g.offsets[0]=0;
  for(uint32_t i=0;i<nv;i++){std::sort(adj[i].begin(),adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(),adj[i].end()),adj[i].end());
    g.offsets[i+1]=g.offsets[i]+adj[i].size();}
  g.ne=g.offsets[nv];g.edges.resize(g.ne);
  for(uint32_t i=0;i<nv;i++)std::copy(adj[i].begin(),adj[i].end(),g.edges.begin()+g.offsets[i]);
  return g;
}

int run_workload_pagerank(const WorkloadConfig &cfg, const char *mode,
                          WorkloadResult *result) {
  uint32_t nv=cfg.param_vertices; int iters=cfg.iterations>0?cfg.iterations:10;
  float alpha=0.85f, tol=0.001f; std::string m(mode);
  HIPRINT("  Generating R-MAT graph: {} verts, avg deg {}", nv, cfg.param_avg_degree);
  CSRGraph g=gen_rmat(nv,cfg.param_avg_degree);
  HIPRINT("  Graph: {} edges ({:.1f} MB)", g.ne, g.ne*4/(1024.0*1024.0));

  uint64_t *d_off; cudaMalloc(&d_off,(nv+1)*sizeof(uint64_t));
  cudaMemcpy(d_off,g.offsets.data(),(nv+1)*sizeof(uint64_t),cudaMemcpyHostToDevice);
  float *d_vals,*d_res; cudaMalloc(&d_vals,nv*sizeof(float)); cudaMalloc(&d_res,nv*sizeof(float));
  std::vector<float> ones(nv,1.0f);
  cudaMemcpy(d_vals,ones.data(),nv*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemset(d_res,0,nv*sizeof(float));
  int *d_active; cudaMallocHost(&d_active,sizeof(int));
  uint32_t threads=256, comp_blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!comp_blocks)comp_blocks=1; int ub=(nv+255)/256;

  if (m == "cte") {
    // ======== CTE: Combined kernel with multi-warp I/O + compute ========
    uint32_t total_warps = (cfg.client_blocks * cfg.client_threads) / 32;
    if (total_warps == 0) total_warps = 1;

    // Partition vertices among warps
    uint32_t verts_per_warp = (nv + total_warps - 1) / total_warps;
    std::vector<uint32_t> h_v_start(total_warps), h_v_end(total_warps);
    std::vector<uint64_t> h_edge_offsets(total_warps), h_edge_bytes(total_warps);

    uint64_t total_edge_bytes = 0;
    for (uint32_t w = 0; w < total_warps; w++) {
      h_v_start[w] = w * verts_per_warp;
      h_v_end[w] = std::min((w + 1) * verts_per_warp, nv);
      uint64_t first_edge = g.offsets[h_v_start[w]];
      uint64_t last_edge = g.offsets[h_v_end[w]];
      h_edge_offsets[w] = first_edge * sizeof(uint32_t);
      h_edge_bytes[w] = (last_edge - first_edge) * sizeof(uint32_t);
      total_edge_bytes += h_edge_bytes[w];
    }

    HIPRINT("  CTE: {} warps, {} verts/warp, {:.1f} MB total edges",
            total_warps, verts_per_warp, total_edge_bytes/(1024.0*1024.0));

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    // Data backend
    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, total_edge_bytes + 4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, (size_t)total_warps*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, (size_t)total_warps*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    pr_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),
                                  total_edge_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError,"PR CTE alloc failed"); cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator();
      cudaFree(d_off);cudaFree(d_vals);cudaFree(d_res);cudaFreeHost(d_active);
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
    uint64_t *h_eo, *h_eb; uint32_t *h_vs, *h_ve;
    cudaMallocHost(&h_eo, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_eb, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_vs, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_ve, total_warps*sizeof(uint32_t));
    memcpy(h_eo, h_edge_offsets.data(), total_warps*sizeof(uint64_t));
    memcpy(h_eb, h_edge_bytes.data(), total_warps*sizeof(uint64_t));
    memcpy(h_vs, h_v_start.data(), total_warps*sizeof(uint32_t));
    memcpy(h_ve, h_v_end.data(), total_warps*sizeof(uint32_t));

    // Seed per-warp edge blobs: copy edges into data backend, then PutBlob
    // For simplicity, copy all edges into data backend contiguously
    cudaMemcpy(array_ptr.ptr_, g.edges.data(), g.ne*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Seed each warp's blob via a simple PutBlob kernel
    // (reusing the same combined kernel for initial seeding — warp 0 seeds all)
    // Actually, use the host-side CTE client for seeding
    {
      wrp_cte::core::Client cte_client(cfg.cte_pool_id);
      using StrT = std::string;
      for (uint32_t w = 0; w < total_warps; w++) {
        std::string bname = "pr_w" + std::to_string(w);
        // Build ShmPtr pointing to this warp's edge data in the data backend
        hipc::ShmPtr<> shm;
        shm.alloc_id_ = data_alloc_id;
        shm.off_.exchange(array_ptr.shm_.off_.load() + h_edge_offsets[w]);
        auto f = cte_client.AsyncPutBlob(cfg.tag_id, bname,
            (chi::u64)0, h_edge_bytes[w], shm, -1.0f,
            wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
        f.Wait();
      }
      HIPRINT("  CTE: Seeded {} per-warp edge blobs", total_warps);
    }

    // Clear data backend before GetBlob
    cudaMemset(array_ptr.ptr_, 0, total_edge_bytes);
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Run combined CTE kernel for each PR iteration
    auto t0 = std::chrono::high_resolution_clock::now();
    int iter;
    for (iter = 0; iter < iters; iter++) {
      *d_done = 0;
      if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      void *stream = hshm::GpuApi::CreateStream();
      pr_cte_kernel<<<cfg.client_blocks, cfg.client_threads, 0,
                       static_cast<cudaStream_t>(stream)>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.client_blocks,
          array_ptr, data_alloc_id,
          d_off, d_vals, d_res,
          h_eo, h_eb, h_vs, h_ve,
          total_warps, alpha, d_done);

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
        HLOG(kError,"PR CTE iter {} timed out",iter); break;}

      // Update step (standard kernel)
      *d_active=0;
      pr_update<<<ub,256>>>(d_vals,d_res,nv,tol,d_active);
      cudaDeviceSynchronize();
      if (*d_active==0) { iter++; break; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric = (double)g.ne*iter/(result->elapsed_ms/1e3);
    result->metric_name = "edges/sec";
    result->bandwidth_gbps = (g.ne*4.0*iter/1e9)/(result->elapsed_ms/1e3);

    cudaFreeHost(d_done); cudaFreeHost(h_eo); cudaFreeHost(h_eb);
    cudaFreeHost(h_vs); cudaFreeHost(h_ve);
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m == "bam") {
    // ======== BaM: Uses bam::ArrayDevice<uint32_t>::read() ========
    uint64_t edge_bytes = g.ne * sizeof(uint32_t);
    uint64_t eb_aligned = ((edge_bytes+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    uint32_t total_pages = (uint32_t)(eb_aligned / cfg.bam_page_size);
    uint32_t cache_pages = std::max(1u, total_pages * cfg.hbm_cache_pct / 100);

    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cache_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;

    bam::PageCache cache(pcfg);
    bam::Array<uint32_t> edges(g.ne, cache);
    edges.load_from_host(g.edges.data(), g.ne);

    HIPRINT("  BaM HBM cache: {} / {} pages ({}%) x {} B = {:.1f} MB",
            cache_pages, total_pages, cfg.hbm_cache_pct, cfg.bam_page_size,
            (double)cache_pages*cfg.bam_page_size/(1024.0*1024.0));

    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      pr_push_bam<<<comp_blocks,threads>>>(d_off, edges.device(),
                                            d_vals, d_res, nv, alpha);
      cudaDeviceSynchronize();
      *d_active=0; pr_update<<<ub,256>>>(d_vals,d_res,nv,tol,d_active);
      cudaDeviceSynchronize(); if(*d_active==0){iter++;break;}
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)g.ne*iter/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(g.ne*4.0*iter/1e9)/(result->elapsed_ms/1e3);
  }
#endif

  else if (m=="hbm") {
    uint32_t *d_edges; cudaMalloc(&d_edges,g.ne*sizeof(uint32_t));
    cudaMemcpy(d_edges,g.edges.data(),g.ne*sizeof(uint32_t),cudaMemcpyHostToDevice);
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      pr_push_hbm<<<comp_blocks,threads>>>(d_off,d_edges,d_vals,d_res,nv,alpha);
      cudaDeviceSynchronize();
      *d_active=0; pr_update<<<ub,256>>>(d_vals,d_res,nv,tol,d_active);
      cudaDeviceSynchronize(); if(*d_active==0){iter++;break;}
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)g.ne*iter/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec"; result->bandwidth_gbps=(g.ne*4.0*iter/1e9)/(result->elapsed_ms/1e3);
    cudaFree(d_edges);

  } else if (m=="direct") {
    uint32_t *h_edges; cudaMallocHost(&h_edges,g.ne*sizeof(uint32_t));
    memcpy(h_edges,g.edges.data(),g.ne*sizeof(uint32_t));
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      pr_push_direct<<<comp_blocks,threads>>>(d_off,h_edges,d_vals,d_res,nv,alpha);
      cudaDeviceSynchronize();
      *d_active=0; pr_update<<<ub,256>>>(d_vals,d_res,nv,tol,d_active);
      cudaDeviceSynchronize(); if(*d_active==0){iter++;break;}
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)g.ne*iter/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec"; result->bandwidth_gbps=(g.ne*4.0*iter/1e9)/(result->elapsed_ms/1e3);
    cudaFreeHost(h_edges);

  } else {
    HLOG(kError,"pagerank: unknown mode '{}'",mode);
    cudaFree(d_off);cudaFree(d_vals);cudaFree(d_res);cudaFreeHost(d_active);
    return -1;
  }

  cudaFree(d_off);cudaFree(d_vals);cudaFree(d_res);cudaFreeHost(d_active);
  return 0;
}

#endif  // HSHM_IS_HOST
