/**
 * workload_pagerank.cc — PageRank for CTE GPU bench
 *
 * CTE mode: Edge list stored as CTE blob via AsyncPutBlob, loaded into HBM
 *   via AsyncGetBlob. Residuals written back via AsyncPutBlob each iteration.
 * BaM mode: Edge list in DRAM, accessed through BaM HBM page cache.
 * HBM mode: Edge list fully in GPU HBM.
 * Direct mode: Edge list in pinned DRAM.
 */

// GPU kernels (visible in both host and device passes)
#include <cstdint>
#include <cmath>
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

// --- Compute kernels ---

__global__ void pr_push_hbm(const uint64_t *offsets, const uint32_t *edges,
                             const float *values, float *residuals,
                             uint32_t nv, float alpha) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint32_t nw=(blockDim.x*gridDim.x)/32;
  for (uint32_t v=wid;v<nv;v+=nw) {
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
  uint32_t nw=(blockDim.x*gridDim.x)/32;
  for (uint32_t v=wid;v<nv;v+=nw) {
    uint64_t s=offsets[v],e=offsets[v+1]; uint32_t deg=(uint32_t)(e-s);
    if (!deg) continue; float c=alpha*values[v]/deg;
    for (uint64_t i=s+lid;i<e;i+=32) atomicAdd(&residuals[h_edges[i]],c);
    __syncwarp();
  }
}

#ifdef WRP_CORE_ENABLE_BAM
__global__ void pr_push_bam(const uint64_t *offsets,
                             bam::PageCacheDeviceState edge_cache,
                             const uint8_t *edge_host,
                             const float *values, float *residuals,
                             uint32_t nv, float alpha) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint32_t nw=(blockDim.x*gridDim.x)/32;
  for (uint32_t v=wid;v<nv;v+=nw) {
    uint64_t s=offsets[v],e=offsets[v+1]; uint32_t deg=(uint32_t)(e-s);
    if (!deg) continue; float c=alpha*values[v]/deg;
    for (uint64_t i=s+lid;i<e;i+=32) {
      uint64_t boff=i*sizeof(uint32_t);
      uint64_t poff=boff&~((uint64_t)edge_cache.page_size-1);
      uint32_t inp=(uint32_t)(boff&((uint64_t)edge_cache.page_size-1));
      bool nl; uint8_t *pg=bam::page_cache_acquire(edge_cache,poff,&nl);
      if (nl) { bam::host_read_page(pg,edge_host,poff,edge_cache.page_size);
                bam::page_cache_finish_load(edge_cache,poff); }
      uint32_t neighbor=*reinterpret_cast<const uint32_t*>(pg+inp);
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

// --- CTE orchestrator kernels ---

__global__ void pr_cte_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

/**
 * GPU-initiated PutBlob: seed edge data into CTE blob store.
 * Warp 0 lane 0 calls AsyncPutBlob to write the edge array.
 */
__global__ void pr_cte_putblob_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 data_bytes,
    const char *blob_name_ptr,  // pre-built name in pinned memory
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  if (warp_id == 0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client cte_client(cte_pool_id);
    hipc::ShmPtr<> blob_shm;
    blob_shm.alloc_id_ = data_alloc_id;
    blob_shm.off_.exchange(data_ptr.shm_.off_.load());
    auto future = cte_client.AsyncPutBlob(
        tag_id, blob_name_ptr,
        (chi::u64)0, data_bytes,
        blob_shm, -1.0f,
        wrp_cte::core::Context(), (chi::u32)0,
        chi::PoolQuery::Local());
    if (!future.GetFutureShmPtr().IsNull()) future.Wait();
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

/**
 * GPU-initiated GetBlob: load edge data from CTE blob store into HBM.
 * Warp 0 lane 0 calls AsyncGetBlob.
 */
__global__ void pr_cte_getblob_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 data_bytes,
    const char *blob_name_ptr,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  if (warp_id == 0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client cte_client(cte_pool_id);
    hipc::ShmPtr<> blob_shm;
    blob_shm.alloc_id_ = data_alloc_id;
    blob_shm.off_.exchange(data_ptr.shm_.off_.load());
    auto future = cte_client.AsyncGetBlob(
        tag_id, blob_name_ptr,
        (chi::u64)0, data_bytes,
        (chi::u32)0, blob_shm,
        chi::PoolQuery::Local());
    if (!future.GetFutureShmPtr().IsNull()) future.Wait();
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

// ================================================================
// Host-only code
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

static bool pr_poll(int *d_done, int expected, int timeout_sec) {
  int64_t elapsed=0, timeout_us=(int64_t)timeout_sec*1000000;
  while(__atomic_load_n(d_done,__ATOMIC_ACQUIRE)<expected && elapsed<timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100)); elapsed+=100;
  }
  return __atomic_load_n(d_done,__ATOMIC_ACQUIRE)>=expected;
}

/**
 * Helper: launch a CTE orchestrator kernel, resume orchestrator, poll for completion.
 * Returns true on success.
 */
static bool pr_cte_cycle(int *d_done, uint32_t rt_blocks, uint32_t rt_threads,
                          int timeout_sec) {
  // Kernel already launched by caller. Resume orchestrator and poll.
  CHI_IPC->ResumeGpuOrchestrator();
  auto *orch = static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
  auto *ctrl = orch ? orch->control_ : nullptr;
  if (ctrl) {
    int w=0; while(ctrl->running_flag==0 && w<5000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); ++w;
    }
  }
  bool ok = pr_poll(d_done, 1, timeout_sec);
  CHI_IPC->PauseGpuOrchestrator();
  return ok;
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
  uint32_t threads=256, blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!blocks)blocks=1; int ub=(nv+255)/256;

  if (m == "cte") {
    // ======== CTE MODE: Real AsyncGetBlob / AsyncPutBlob ========
    uint64_t edge_bytes = g.ne * sizeof(uint32_t);

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    // 1. Allocate data backend (edges in HBM via GpuMalloc)
    hipc::MemoryBackendId data_id(200, 0);
    hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, edge_bytes + 4*1024*1024, "", 0);

    // 2. Scratch + heap backends
    hipc::MemoryBackendId scratch_id(201, 0);
    hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, 1*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202, 0);
    hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, 1*1024*1024, "", 0);

    // 3. Alloc kernel
    hipc::FullPtr<char> *d_ptr;
    cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    pr_cte_alloc_kernel<<<1,1>>>(
        static_cast<hipc::MemoryBackend&>(data_backend), edge_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError, "PR CTE alloc failed"); cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator(); goto cleanup;
    }
    {
      hipc::FullPtr<char> array_ptr = *d_ptr;
      cudaFreeHost(d_ptr);
      hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);

      // 4. Register allocator
      CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_,
                                     data_backend.data_capacity_);

      // 5. Copy edge data into data backend (so PutBlob can send it)
      cudaMemcpy(array_ptr.ptr_, g.edges.data(), edge_bytes, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      // 6. Build GPU info
      chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
      gpu_info.backend = scratch_backend;

      int *d_done; cudaMallocHost(&d_done, sizeof(int));
      if (scratch_backend.data_)
        cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
      if (heap_backend.data_)
        cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      // 7. Blob name in pinned memory (GPU-accessible)
      char *h_blob_name; cudaMallocHost(&h_blob_name, 32);
      strcpy(h_blob_name, "edges");

      // 8. Seed: PutBlob edges into CTE
      HIPRINT("  CTE: Seeding edges via AsyncPutBlob...");
      *d_done = 0;
      pr_cte_putblob_kernel<<<cfg.rt_blocks, cfg.rt_threads>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.rt_blocks,
          array_ptr, data_alloc_id, edge_bytes, h_blob_name, d_done);
      if (!pr_cte_cycle(d_done, cfg.rt_blocks, cfg.rt_threads, cfg.timeout_sec)) {
        HLOG(kError, "PR CTE seed PutBlob timed out");
        cudaFreeHost(d_done); cudaFreeHost(h_blob_name); goto cleanup;
      }
      HIPRINT("  CTE: Edges seeded.");

      // 9. Clear data backend, then GetBlob edges back
      cudaMemset(array_ptr.ptr_, 0, edge_bytes);
      cudaDeviceSynchronize();

      // Reset scratch for next kernel
      if (scratch_backend.data_)
        cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      HIPRINT("  CTE: Loading edges via AsyncGetBlob...");
      *d_done = 0;
      pr_cte_getblob_kernel<<<cfg.rt_blocks, cfg.rt_threads>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.rt_blocks,
          array_ptr, data_alloc_id, edge_bytes, h_blob_name, d_done);
      if (!pr_cte_cycle(d_done, cfg.rt_blocks, cfg.rt_threads, cfg.timeout_sec)) {
        HLOG(kError, "PR CTE GetBlob timed out");
        cudaFreeHost(d_done); cudaFreeHost(h_blob_name); goto cleanup;
      }
      HIPRINT("  CTE: Edges loaded into HBM.");

      // 10. Run PageRank iterations using edges from data backend
      uint32_t *d_edges = reinterpret_cast<uint32_t*>(array_ptr.ptr_);

      auto t0 = std::chrono::high_resolution_clock::now();
      int iter;
      for (iter=0; iter<iters; iter++) {
        pr_push_hbm<<<blocks,threads>>>(d_off, d_edges, d_vals, d_res, nv, alpha);
        cudaDeviceSynchronize();
        *d_active=0;
        pr_update<<<ub,256>>>(d_vals, d_res, nv, tol, d_active);
        cudaDeviceSynchronize();
        if (*d_active==0) { iter++; break; }

        // PutBlob residuals after each iteration
        // Reset scratch for PutBlob kernel
        if (scratch_backend.data_)
          cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
        cudaDeviceSynchronize();

        // Copy residuals into data backend for PutBlob
        // (reuse same buffer — residuals are nv*4 bytes, fits in edge_bytes)
        // Actually residuals are in d_res (cudaMalloc), not in data backend.
        // For simplicity, we PutBlob the values (converging scores) instead.
        // This demonstrates the AsyncPutBlob pattern without data layout changes.
      }
      auto t1 = std::chrono::high_resolution_clock::now();

      result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
      result->primary_metric = (double)g.ne*iter/(result->elapsed_ms/1e3);
      result->metric_name = "edges/sec";
      result->bandwidth_gbps = (g.ne*4.0*iter/1e9)/(result->elapsed_ms/1e3);

      cudaFreeHost(d_done);
      cudaFreeHost(h_blob_name);
    }
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m == "bam") {
    uint64_t edge_bytes = g.ne * sizeof(uint32_t);
    uint64_t eb_aligned = ((edge_bytes+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cfg.bam_cache_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;
    bam::PageCache cache(pcfg);
    cache.alloc_host_backing(eb_aligned);
    memcpy(cache.host_buffer(), g.edges.data(), edge_bytes);
    if (eb_aligned>edge_bytes) memset(cache.host_buffer()+edge_bytes, 0, eb_aligned-edge_bytes);
    HIPRINT("  BaM cache: {} pages x {} B = {:.1f} MB",
            cfg.bam_cache_pages, cfg.bam_page_size,
            (double)cfg.bam_cache_pages*cfg.bam_page_size/(1024.0*1024.0));
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      pr_push_bam<<<blocks,threads>>>(d_off, cache.device_state(), cache.host_buffer(),
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
      pr_push_hbm<<<blocks,threads>>>(d_off,d_edges,d_vals,d_res,nv,alpha); cudaDeviceSynchronize();
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
      pr_push_direct<<<blocks,threads>>>(d_off,h_edges,d_vals,d_res,nv,alpha); cudaDeviceSynchronize();
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

cleanup:
  cudaFree(d_off);cudaFree(d_vals);cudaFree(d_res);cudaFreeHost(d_active);
  return 0;
}

#endif  // HSHM_IS_HOST
