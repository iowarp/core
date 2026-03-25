/**
 * workload_llm_kvcache.cc — LLM KV cache offloading for CTE GPU bench
 *
 * CTE mode: Per-layer KV stored as CTE blobs. Each decode token:
 *   AsyncGetBlob to load layer KV → attention → AsyncPutBlob updated KV.
 * BaM mode: KV cache in DRAM, attention reads through BaM page cache.
 * HBM/Direct modes: standard.
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
#include <bam/page_cache.cuh>
#include <bam/types.h>
#endif

// --- Compute kernels ---

__global__ void llm_attn_hbm(const float *kv, const float *q, float *out,
                              uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    const float*K=kv+(uint64_t)h*kvs; const float*V=kv+(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){float d=0;for(uint32_t i=lid;i<hd;i+=32)d+=Q[i]*K[s*hd+i];
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);if(d>mx){mx=d;bp=s;}}
    for(uint32_t i=lid;i<hd;i+=32)O[i]=V[bp*hd+i]; __syncwarp();
  }
}

__global__ void llm_attn_direct(const float *h_kv, const float *q, float *out,
                                 uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    const float*K=h_kv+(uint64_t)h*kvs; const float*V=h_kv+(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){float d=0;for(uint32_t i=lid;i<hd;i+=32)d+=Q[i]*K[s*hd+i];
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);if(d>mx){mx=d;bp=s;}}
    for(uint32_t i=lid;i<hd;i+=32)O[i]=V[bp*hd+i]; __syncwarp();
  }
}

#ifdef WRP_CORE_ENABLE_BAM
__device__ inline float llm_bam_read(bam::PageCacheDeviceState &cache,
                                      const uint8_t *host, uint64_t idx) {
  uint64_t boff=idx*sizeof(float); uint64_t poff=boff&~((uint64_t)cache.page_size-1);
  uint32_t inp=(uint32_t)(boff&((uint64_t)cache.page_size-1));
  bool nl; uint8_t *pg=bam::page_cache_acquire(cache,poff,&nl);
  if(nl){bam::host_read_page(pg,host,poff,cache.page_size);bam::page_cache_finish_load(cache,poff);}
  return *reinterpret_cast<const float*>(pg+inp);
}

__global__ void llm_attn_bam(bam::PageCacheDeviceState kv_cache,
                              const uint8_t *kv_host, const float *q, float *out,
                              uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    uint64_t kb=(uint64_t)h*kvs, vb=(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){float d=0;
      for(uint32_t i=lid;i<hd;i+=32)d+=Q[i]*llm_bam_read(kv_cache,kv_host,kb+s*hd+i);
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);if(d>mx){mx=d;bp=s;}}
    for(uint32_t i=lid;i<hd;i+=32)O[i]=llm_bam_read(kv_cache,kv_host,vb+bp*hd+i);
    __syncwarp();
  }
}
#endif

__global__ void llm_kv_wb_hbm(float *kv, const float *nk, const float *nv,
                               uint32_t nh, uint32_t sl, uint32_t hd, uint32_t pos) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for(uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32){
    for(uint32_t i=lid;i<hd;i+=32){kv[h*kvs+pos*hd+i]=nk[h*hd+i];kv[nh*kvs+h*kvs+pos*hd+i]=nv[h*hd+i];}
    __syncwarp();}
}

__global__ void llm_kv_wb_direct(float *h_kv, const float *nk, const float *nv,
                                  uint32_t nh, uint32_t sl, uint32_t hd, uint32_t pos) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for(uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32){
    for(uint32_t i=lid;i<hd;i+=32){h_kv[h*kvs+pos*hd+i]=nk[h*hd+i];h_kv[nh*kvs+h*kvs+pos*hd+i]=nv[h*hd+i];}
    __syncwarp();}
}

// --- CTE orchestrator kernels ---

__global__ void llm_cte_alloc_kernel(
    hipc::MemoryBackend data_backend, chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if(threadIdx.x!=0||blockIdx.x!=0)return;
  using AllocT=hipc::PrivateBuddyAllocator;
  auto *alloc=data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if(!alloc){d_out_ptr->SetNull();return;}
  *d_out_ptr=alloc->AllocateObjs<char>(total_bytes);
}

/**
 * GPU-initiated PutBlob/GetBlob for KV cache layers.
 * Blob name is passed via pinned memory (pre-built by host).
 */
__global__ void llm_cte_putblob_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId alloc_id,
    chi::u64 data_bytes, const char *blob_name, int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  if(chi::IpcManager::GetWarpId()==0 && chi::IpcManager::IsWarpScheduler()){
    wrp_cte::core::Client c(pool_id);
    hipc::ShmPtr<> shm; shm.alloc_id_=alloc_id;
    shm.off_.exchange(data_ptr.shm_.off_.load());
    auto f=c.AsyncPutBlob(tag_id,blob_name,(chi::u64)0,data_bytes,shm,-1.0f,
        wrp_cte::core::Context(),(chi::u32)0,chi::PoolQuery::Local());
    if(!f.GetFutureShmPtr().IsNull())f.Wait();
    atomicAdd_system(d_done,1); __threadfence_system();
  }
}

__global__ void llm_cte_getblob_kernel(
    chi::IpcManagerGpu gpu_info, chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id, chi::u32 num_blocks,
    hipc::FullPtr<char> data_ptr, hipc::AllocatorId alloc_id,
    chi::u64 data_bytes, const char *blob_name, int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);
  if(chi::IpcManager::GetWarpId()==0 && chi::IpcManager::IsWarpScheduler()){
    wrp_cte::core::Client c(pool_id);
    hipc::ShmPtr<> shm; shm.alloc_id_=alloc_id;
    shm.off_.exchange(data_ptr.shm_.off_.load());
    auto f=c.AsyncGetBlob(tag_id,blob_name,(chi::u64)0,data_bytes,
        (chi::u32)0,shm,chi::PoolQuery::Local());
    if(!f.GetFutureShmPtr().IsNull())f.Wait();
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
#include <cstring>

static bool llm_poll(int *d,int e,int ts){
  int64_t el=0,tu=(int64_t)ts*1000000;
  while(__atomic_load_n(d,__ATOMIC_ACQUIRE)<e&&el<tu){
    std::this_thread::sleep_for(std::chrono::microseconds(100));el+=100;}
  return __atomic_load_n(d,__ATOMIC_ACQUIRE)>=e;
}

static bool llm_cte_cycle(int *d_done, int ts) {
  CHI_IPC->ResumeGpuOrchestrator();
  auto *o=static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
  auto *c=o?o->control_:nullptr;
  if(c){int w=0;while(c->running_flag==0&&w<5000){std::this_thread::sleep_for(std::chrono::milliseconds(1));++w;}}
  bool ok=llm_poll(d_done,1,ts);
  CHI_IPC->PauseGpuOrchestrator();
  return ok;
}

int run_workload_llm_kvcache(const WorkloadConfig &cfg, const char *mode, WorkloadResult *result) {
  uint32_t nl=cfg.param_num_layers,nh=cfg.param_num_heads,hd=cfg.param_head_dim,sl=cfg.param_seq_len;
  uint32_t dt=cfg.param_decode_tokens; std::string m(mode);
  uint64_t kvpl=2ULL*nh*sl*hd;  // floats per layer
  uint64_t kvbl=kvpl*sizeof(float);  // bytes per layer
  uint64_t kvbt=(uint64_t)nl*kvbl;   // total bytes
  uint64_t qof=(uint64_t)nh*hd;

  float *d_q,*d_o,*d_nk,*d_nv;
  cudaMalloc(&d_q,qof*4);cudaMalloc(&d_o,qof*4);cudaMalloc(&d_nk,qof*4);cudaMalloc(&d_nv,qof*4);
  std::vector<float> hq(qof,0.1f);
  cudaMemcpy(d_q,hq.data(),qof*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nk,hq.data(),qof*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_nv,hq.data(),qof*4,cudaMemcpyHostToDevice);
  uint32_t threads=256,blocks=(cfg.client_blocks*cfg.client_threads+threads-1)/threads;
  if(!blocks)blocks=1;

  if (m=="cte") {
    // ======== CTE MODE: Per-layer GetBlob/PutBlob per token ========
    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    // Data backend: one layer's KV at a time in HBM
    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, kvbl+4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, 1*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, 1*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr,sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    llm_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),kvbl,d_ptr);
    cudaDeviceSynchronize();
    if(d_ptr->IsNull()){HLOG(kError,"LLM CTE alloc failed");cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator();goto cleanup;}
    {
      hipc::FullPtr<char> array_ptr=*d_ptr; cudaFreeHost(d_ptr);
      hipc::AllocatorId data_alloc_id(data_id.major_,data_id.minor_);
      CHI_IPC->RegisterGpuAllocator(data_id,data_backend.data_,data_backend.data_capacity_);

      chi::IpcManagerGpu gpu_info=CHI_IPC->GetClientGpuInfo(0);
      gpu_info.backend=scratch_backend;
      int *d_done; cudaMallocHost(&d_done,sizeof(int));

      // Blob names for each layer (pinned memory, GPU-accessible)
      char **h_names; cudaMallocHost(&h_names, nl*sizeof(char*));
      for(uint32_t l=0;l<nl;l++){
        cudaMallocHost(&h_names[l],32);
        snprintf(h_names[l],32,"kv_l%u",l);
      }

      // Timing variables declared before any goto
      std::chrono::high_resolution_clock::time_point t0, t1;

      // Seed: PutBlob zero-initialized KV for each layer
      HIPRINT("  CTE: Seeding {} layers of KV ({:.1f} MB each)...", nl, kvbl/(1024.0*1024.0));
      cudaMemset(array_ptr.ptr_, 0, kvbl); cudaDeviceSynchronize();
      for(uint32_t l=0;l<nl;l++){
        if(scratch_backend.data_)cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
        if(heap_backend.data_)cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
        cudaDeviceSynchronize();
        *d_done=0;
        llm_cte_putblob_kernel<<<cfg.rt_blocks,cfg.rt_threads>>>(
            gpu_info,cfg.cte_pool_id,cfg.tag_id,cfg.rt_blocks,
            array_ptr,data_alloc_id,kvbl,h_names[l],d_done);
        if(!llm_cte_cycle(d_done,cfg.timeout_sec)){
          HLOG(kError,"LLM CTE seed PutBlob layer {} timed out",l); goto llm_cleanup;}
      }
      HIPRINT("  CTE: All layers seeded.");

      // Decode loop
      t0=std::chrono::high_resolution_clock::now();
      for(uint32_t t=0;t<dt;t++){
        for(uint32_t l=0;l<nl;l++){
          // GetBlob: load layer l's KV into HBM
          if(scratch_backend.data_)cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
          cudaDeviceSynchronize();
          *d_done=0;
          llm_cte_getblob_kernel<<<cfg.rt_blocks,cfg.rt_threads>>>(
              gpu_info,cfg.cte_pool_id,cfg.tag_id,cfg.rt_blocks,
              array_ptr,data_alloc_id,kvbl,h_names[l],d_done);
          if(!llm_cte_cycle(d_done,cfg.timeout_sec)){
            HLOG(kError,"LLM CTE GetBlob layer {} token {} timed out",l,t); goto llm_cleanup;}

          // Compute attention on HBM KV
          float *lkv=reinterpret_cast<float*>(array_ptr.ptr_);
          llm_attn_hbm<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
          llm_kv_wb_hbm<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);
          cudaDeviceSynchronize();

          // PutBlob: write updated KV back to CTE
          if(scratch_backend.data_)cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
          cudaDeviceSynchronize();
          *d_done=0;
          llm_cte_putblob_kernel<<<cfg.rt_blocks,cfg.rt_threads>>>(
              gpu_info,cfg.cte_pool_id,cfg.tag_id,cfg.rt_blocks,
              array_ptr,data_alloc_id,kvbl,h_names[l],d_done);
          if(!llm_cte_cycle(d_done,cfg.timeout_sec)){
            HLOG(kError,"LLM CTE PutBlob layer {} token {} timed out",l,t); goto llm_cleanup;}
        }
      }
      t1=std::chrono::high_resolution_clock::now();
      result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
      result->primary_metric=dt/(result->elapsed_ms/1e3);
      result->metric_name="tokens/sec";
      result->bandwidth_gbps=((uint64_t)nl*(kvbl+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);

llm_cleanup:
      for(uint32_t l=0;l<nl;l++) cudaFreeHost(h_names[l]);
      cudaFreeHost(h_names); cudaFreeHost(d_done);
    }
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m=="bam") {
    uint64_t kvb_al=((kvbt+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    bam::PageCacheConfig pcfg; pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cfg.bam_cache_pages;
    pcfg.num_queues=0;pcfg.queue_depth=0;pcfg.backend=bam::BackendType::kHostMemory;pcfg.nvme_dev=nullptr;
    bam::PageCache cache(pcfg); cache.alloc_host_backing(kvb_al);
    memset(cache.host_buffer(),0,kvb_al);
    HIPRINT("  BaM cache: {} pages x {} B = {:.1f} MB (KV total {:.1f} MB)",
            cfg.bam_cache_pages,cfg.bam_page_size,
            (double)cfg.bam_cache_pages*cfg.bam_page_size/(1024.0*1024.0),kvbt/(1024.0*1024.0));
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){
      for(uint32_t l=0;l<nl;l++){
        llm_attn_bam<<<blocks,threads>>>(cache.device_state(),
            cache.host_buffer()+l*kvbl, d_q,d_o,nh,sl,hd);
        llm_kv_wb_direct<<<blocks,threads>>>(
            reinterpret_cast<float*>(cache.host_buffer()+l*kvbl),
            d_nk,d_nv,nh,sl,hd,t);
      }
      cudaDeviceSynchronize();
      cudaMemset(cache.device_state().page_tags,0xFF,cfg.bam_cache_pages*sizeof(uint64_t));
      cudaMemset(cache.device_state().page_states,0,cfg.bam_cache_pages*sizeof(uint32_t));
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvbl+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
  }
#endif

  else if (m=="hbm") {
    float *d_kv; cudaMalloc(&d_kv,kvbt); cudaMemset(d_kv,0,kvbt);
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){for(uint32_t l=0;l<nl;l++){
      float*lkv=d_kv+(uint64_t)l*kvpl;
      llm_attn_hbm<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
      llm_kv_wb_hbm<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);}}
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvbl+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFree(d_kv);
  } else if (m=="direct") {
    float *h_kv; cudaMallocHost(&h_kv,kvbt); memset(h_kv,0,kvbt);
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){for(uint32_t l=0;l<nl;l++){
      float*lkv=h_kv+(uint64_t)l*kvpl;
      llm_attn_direct<<<blocks,threads>>>(lkv,d_q,d_o,nh,sl,hd);
      llm_kv_wb_direct<<<blocks,threads>>>(lkv,d_nk,d_nv,nh,sl,hd,t);
    }cudaDeviceSynchronize();}
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=dt/(result->elapsed_ms/1e3);result->metric_name="tokens/sec";
    result->bandwidth_gbps=((uint64_t)nl*(kvbl+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);
    cudaFreeHost(h_kv);
  } else { HLOG(kError,"llm_kvcache: unknown mode '{}'",mode); goto cleanup; }

cleanup:
  cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
  return 0;
}

#endif
