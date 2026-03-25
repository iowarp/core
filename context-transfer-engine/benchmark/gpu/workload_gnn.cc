/**
 * workload_gnn.cc — GNN feature loading for CTE GPU bench
 *
 * CTE mode: Feature table stored as CTE blob via AsyncPutBlob (seed),
 *   loaded into HBM via AsyncGetBlob. Gather from HBM.
 * BaM mode: Features in DRAM, read through BaM HBM page cache.
 * HBM/Direct modes: standard.
 */

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

// --- Compute kernels ---

__global__ void gnn_gather_hbm(const float *features, const uint32_t *indices,
                                float *output, uint32_t bs, uint32_t ed) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t b=wid;b<bs;b+=(blockDim.x*gridDim.x)/32) {
    const float *in=features+(uint64_t)indices[b]*ed;
    float *out=output+(uint64_t)b*ed;
    for (uint32_t f=lid;f<ed;f+=32) out[f]=in[f]; __syncwarp();
  }
}

__global__ void gnn_gather_direct(const float *h_features, const uint32_t *indices,
                                   float *output, uint32_t bs, uint32_t ed) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  for (uint32_t b=wid;b<bs;b+=(blockDim.x*gridDim.x)/32) {
    const float *in=h_features+(uint64_t)indices[b]*ed;
    float *out=output+(uint64_t)b*ed;
    for (uint32_t f=lid;f<ed;f+=32) out[f]=in[f]; __syncwarp();
  }
}

#ifdef WRP_CORE_ENABLE_BAM
__global__ void gnn_gather_bam(bam::PageCacheDeviceState cache,
                                const uint8_t *host_base,
                                const uint32_t *indices, float *output,
                                uint32_t bs, uint32_t ed) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint32_t page_size=cache.page_size, fps=page_size/sizeof(float);
  for (uint32_t b=wid;b<bs;b+=(blockDim.x*gridDim.x)/32) {
    uint32_t node=indices[b]; uint64_t foff=(uint64_t)node*ed*sizeof(float);
    float *out=output+(uint64_t)b*ed;
    for (uint32_t fb=0;fb<ed;fb+=fps) {
      uint64_t poff=(foff+fb*sizeof(float))&~((uint64_t)page_size-1);
      bool nl; uint8_t *pg=bam::warp_page_cache_acquire(cache,poff,&nl);
      if (nl) { bam::warp_host_read_page(pg,host_base,poff,page_size);
                bam::warp_page_cache_finish_load(cache,poff); }
      uint32_t fe=(fb+fps<ed)?fb+fps:ed;
      for (uint32_t f=fb+lid;f<fe;f+=32) {
        uint64_t bo=foff+f*sizeof(float);
        uint32_t ip=(uint32_t)(bo&((uint64_t)page_size-1));
        out[f]=*reinterpret_cast<const float*>(pg+ip);
      }
      __syncwarp();
    }
  }
}
#endif

// --- CTE orchestrator kernels ---

__global__ void gnn_cte_alloc_kernel(
    hipc::MemoryBackend data_backend, chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x!=0||blockIdx.x!=0) return;
  using AllocT=hipc::PrivateBuddyAllocator;
  auto *alloc=data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr=alloc->AllocateObjs<char>(total_bytes);
}

__global__ void gnn_cte_putblob_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId data_alloc_id,
    chi::u64 data_bytes, const char *blob_name, int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  if (chi::IpcManager::GetWarpId()==0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client c(cte_pool_id);
    hipc::ShmPtr<> shm; shm.alloc_id_=data_alloc_id;
    shm.off_.exchange(data_ptr.shm_.off_.load());
    auto f=c.AsyncPutBlob(tag_id,blob_name,(chi::u64)0,data_bytes,shm,-1.0f,
        wrp_cte::core::Context(),(chi::u32)0,chi::PoolQuery::Local());
    if (!f.GetFutureShmPtr().IsNull()) f.Wait();
    atomicAdd_system(d_done,1); __threadfence_system();
  }
}

__global__ void gnn_cte_getblob_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId data_alloc_id,
    chi::u64 data_bytes, const char *blob_name, int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  if (chi::IpcManager::GetWarpId()==0 && chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client c(cte_pool_id);
    hipc::ShmPtr<> shm; shm.alloc_id_=data_alloc_id;
    shm.off_.exchange(data_ptr.shm_.off_.load());
    auto f=c.AsyncGetBlob(tag_id,blob_name,(chi::u64)0,data_bytes,
        (chi::u32)0,shm,chi::PoolQuery::Local());
    if (!f.GetFutureShmPtr().IsNull()) f.Wait();
    atomicAdd_system(d_done,1); __threadfence_system();
  }
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
#include <cstring>

static bool gnn_poll(int *d, int exp, int ts) {
  int64_t el=0,tu=(int64_t)ts*1000000;
  while(__atomic_load_n(d,__ATOMIC_ACQUIRE)<exp&&el<tu){
    std::this_thread::sleep_for(std::chrono::microseconds(100));el+=100;}
  return __atomic_load_n(d,__ATOMIC_ACQUIRE)>=exp;
}

static bool gnn_cte_cycle(int *d_done, uint32_t rtb, uint32_t rtt, int ts) {
  CHI_IPC->ResumeGpuOrchestrator();
  auto *o=static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
  auto *c=o?o->control_:nullptr;
  if(c){int w=0;while(c->running_flag==0&&w<5000){std::this_thread::sleep_for(std::chrono::milliseconds(1));++w;}}
  bool ok=gnn_poll(d_done,1,ts);
  CHI_IPC->PauseGpuOrchestrator();
  return ok;
}

int run_workload_gnn(const WorkloadConfig &cfg, const char *mode, WorkloadResult *result) {
  uint32_t nn=cfg.param_num_nodes, ed=cfg.param_emb_dim, bs=cfg.param_batch_size;
  uint32_t nb=cfg.iterations>0?cfg.iterations:10;
  std::string m(mode); uint64_t feat_bytes=(uint64_t)nn*ed*sizeof(float);
  uint32_t threads=256, blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!blocks)blocks=1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f,1.0f);
  std::uniform_int_distribution<uint32_t> ndist(0,nn-1);
  uint32_t *d_idx; float *d_out;
  cudaMalloc(&d_idx,bs*sizeof(uint32_t)); cudaMalloc(&d_out,(uint64_t)bs*ed*sizeof(float));
  std::vector<uint32_t> h_idx(bs);

  if (m == "cte") {
    // ======== CTE MODE: Real AsyncPutBlob (seed) + AsyncGetBlob (load) ========
    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, feat_bytes+4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, 1*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, 1*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr,sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gnn_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),feat_bytes,d_ptr);
    cudaDeviceSynchronize();
    if(d_ptr->IsNull()){HLOG(kError,"GNN CTE alloc failed");cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator(); goto cleanup;}
    {
      hipc::FullPtr<char> array_ptr=*d_ptr; cudaFreeHost(d_ptr);
      hipc::AllocatorId data_alloc_id(data_id.major_,data_id.minor_);
      CHI_IPC->RegisterGpuAllocator(data_id,data_backend.data_,data_backend.data_capacity_);

      chi::IpcManagerGpu gpu_info=CHI_IPC->GetClientGpuInfo(0);
      gpu_info.backend=scratch_backend;
      int *d_done; cudaMallocHost(&d_done,sizeof(int));
      if(scratch_backend.data_)cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      if(heap_backend.data_)cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));

      // Generate features and copy into data backend for seeding
      HIPRINT("  Generating {} MB of features...", feat_bytes/(1024*1024));
      std::vector<float> h_feat((uint64_t)nn*ed);
      for(auto &x:h_feat)x=fdist(rng);
      cudaMemcpy(array_ptr.ptr_,h_feat.data(),feat_bytes,cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      char *h_name; cudaMallocHost(&h_name,32); strcpy(h_name,"features");

      // Seed: PutBlob features into CTE
      HIPRINT("  CTE: Seeding features via AsyncPutBlob...");
      *d_done=0;
      gnn_cte_putblob_kernel<<<cfg.rt_blocks,cfg.rt_threads>>>(
          gpu_info,cfg.cte_pool_id,cfg.tag_id,cfg.rt_blocks,
          array_ptr,data_alloc_id,feat_bytes,h_name,d_done);
      if(!gnn_cte_cycle(d_done,cfg.rt_blocks,cfg.rt_threads,cfg.timeout_sec)){
        HLOG(kError,"GNN CTE seed timed out");cudaFreeHost(d_done);cudaFreeHost(h_name);goto cleanup;}
      HIPRINT("  CTE: Features seeded.");

      // Clear and GetBlob back
      cudaMemset(array_ptr.ptr_,0,feat_bytes); cudaDeviceSynchronize();
      if(scratch_backend.data_)cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      HIPRINT("  CTE: Loading features via AsyncGetBlob...");
      *d_done=0;
      gnn_cte_getblob_kernel<<<cfg.rt_blocks,cfg.rt_threads>>>(
          gpu_info,cfg.cte_pool_id,cfg.tag_id,cfg.rt_blocks,
          array_ptr,data_alloc_id,feat_bytes,h_name,d_done);
      if(!gnn_cte_cycle(d_done,cfg.rt_blocks,cfg.rt_threads,cfg.timeout_sec)){
        HLOG(kError,"GNN CTE GetBlob timed out");cudaFreeHost(d_done);cudaFreeHost(h_name);goto cleanup;}
      HIPRINT("  CTE: Features loaded into HBM.");

      float *d_feat=reinterpret_cast<float*>(array_ptr.ptr_);

      // Warmup
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      gnn_gather_hbm<<<blocks,threads>>>(d_feat,d_idx,d_out,bs,ed);cudaDeviceSynchronize();

      double total_ms=0;
      for(uint32_t b=0;b<nb;b++){
        for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
        cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
        auto t0=std::chrono::high_resolution_clock::now();
        gnn_gather_hbm<<<blocks,threads>>>(d_feat,d_idx,d_out,bs,ed);cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
      }
      result->elapsed_ms=total_ms/nb;
      result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
      result->primary_metric=bs/(result->elapsed_ms/1e3);
      result->metric_name="nodes/sec";
      cudaFreeHost(d_done); cudaFreeHost(h_name);
    }
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m=="bam") {
    uint64_t fb_al=((feat_bytes+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    bam::PageCacheConfig pcfg; pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cfg.bam_cache_pages;
    pcfg.num_queues=0;pcfg.queue_depth=0;pcfg.backend=bam::BackendType::kHostMemory;pcfg.nvme_dev=nullptr;
    bam::PageCache cache(pcfg); cache.alloc_host_backing(fb_al);
    for(uint64_t i=0;i<(uint64_t)nn*ed;i++)reinterpret_cast<float*>(cache.host_buffer())[i]=fdist(rng);
    HIPRINT("  BaM cache: {} pages x {} B = {:.1f} MB",cfg.bam_cache_pages,cfg.bam_page_size,
            (double)cfg.bam_cache_pages*cfg.bam_page_size/(1024.0*1024.0));
    for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
    cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
    gnn_gather_bam<<<blocks,threads>>>(cache.device_state(),cache.host_buffer(),d_idx,d_out,bs,ed);
    cudaDeviceSynchronize();
    double total_ms=0;
    for(uint32_t b=0;b<nb;b++){
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemset(cache.device_state().page_tags,0xFF,cfg.bam_cache_pages*sizeof(uint64_t));
      cudaMemset(cache.device_state().page_states,0,cfg.bam_cache_pages*sizeof(uint32_t));
      cudaDeviceSynchronize();
      auto t0=std::chrono::high_resolution_clock::now();
      gnn_gather_bam<<<blocks,threads>>>(cache.device_state(),cache.host_buffer(),d_idx,d_out,bs,ed);
      cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    result->elapsed_ms=total_ms/nb;
    result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=bs/(result->elapsed_ms/1e3); result->metric_name="nodes/sec";
  }
#endif

  else if (m=="hbm") {
    float *d_feat; cudaMalloc(&d_feat,feat_bytes);
    std::vector<float> h_feat((uint64_t)nn*ed); for(auto &x:h_feat)x=fdist(rng);
    cudaMemcpy(d_feat,h_feat.data(),feat_bytes,cudaMemcpyHostToDevice);
    for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
    cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
    gnn_gather_hbm<<<blocks,threads>>>(d_feat,d_idx,d_out,bs,ed);cudaDeviceSynchronize();
    double total_ms=0;
    for(uint32_t b=0;b<nb;b++){
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      auto t0=std::chrono::high_resolution_clock::now();
      gnn_gather_hbm<<<blocks,threads>>>(d_feat,d_idx,d_out,bs,ed);cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    result->elapsed_ms=total_ms/nb;result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=bs/(result->elapsed_ms/1e3);result->metric_name="nodes/sec";
    cudaFree(d_feat);

  } else if (m=="direct") {
    float *h_feat; cudaMallocHost(&h_feat,feat_bytes);
    for(uint64_t i=0;i<(uint64_t)nn*ed;i++)h_feat[i]=fdist(rng);
    for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
    cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
    gnn_gather_direct<<<blocks,threads>>>(h_feat,d_idx,d_out,bs,ed);cudaDeviceSynchronize();
    double total_ms=0;
    for(uint32_t b=0;b<nb;b++){
      for(uint32_t i=0;i<bs;i++)h_idx[i]=ndist(rng);
      cudaMemcpy(d_idx,h_idx.data(),bs*sizeof(uint32_t),cudaMemcpyHostToDevice);
      auto t0=std::chrono::high_resolution_clock::now();
      gnn_gather_direct<<<blocks,threads>>>(h_feat,d_idx,d_out,bs,ed);cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      total_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    result->elapsed_ms=total_ms/nb;result->bandwidth_gbps=((uint64_t)bs*ed*4/1e9)/(result->elapsed_ms/1e3);
    result->primary_metric=bs/(result->elapsed_ms/1e3);result->metric_name="nodes/sec";
    cudaFreeHost(h_feat);
  } else { HLOG(kError,"gnn: unknown mode '{}'",mode); cudaFree(d_idx);cudaFree(d_out); return -1; }

cleanup:
  cudaFree(d_idx);cudaFree(d_out);
  return 0;
}

#endif
