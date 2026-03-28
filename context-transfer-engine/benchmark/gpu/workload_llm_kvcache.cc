/**
 * workload_llm_kvcache.cc — LLM KV cache for CTE GPU bench
 *
 * CTE mode: Combined kernel. Each warp owns a batch of attention heads.
 *   For each layer: Lane 0 calls AsyncGetBlob to load KV for its heads.
 *   All 32 lanes compute attention (Q·K^T, argmax, V lookup).
 *   Lane 0 calls AsyncPutBlob to write updated KV after writeback.
 *
 * BaM mode: Uses bam::ArrayDevice<float>::read() for KV access.
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

// --- HBM/Direct compute kernels ---

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
/**
 * BaM LLM attention: reads KV through bam::ArrayDevice<float>.
 */
__global__ void llm_attn_bam(bam::ArrayDevice<float> kv_cache,
                              const float *q, float *out,
                              uint32_t nh, uint32_t sl, uint32_t hd) {
  uint32_t wid=(blockIdx.x*blockDim.x+threadIdx.x)/32, lid=threadIdx.x&31;
  uint64_t kvs=(uint64_t)sl*hd;
  for (uint32_t h=wid;h<nh;h+=(blockDim.x*gridDim.x)/32) {
    uint64_t kb=(uint64_t)h*kvs, vb=(uint64_t)nh*kvs+(uint64_t)h*kvs;
    const float*Q=q+(uint64_t)h*hd; float*O=out+(uint64_t)h*hd;
    float mx=-1e30f; uint32_t bp=0;
    for(uint32_t s=0;s<sl;s++){float d=0;
      for(uint32_t i=lid;i<hd;i+=32)d+=Q[i]*kv_cache.read(kb+s*hd+i);
      for(int o=16;o>0;o>>=1)d+=__shfl_down_sync(0xFFFFFFFF,d,o);
      d=__shfl_sync(0xFFFFFFFF,d,0);d/=sqrtf((float)hd);if(d>mx){mx=d;bp=s;}}
    for(uint32_t i=lid;i<hd;i+=32)O[i]=kv_cache.read(vb+bp*hd+i);
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

/**
 * Combined CTE LLM kernel: I/O + attention + writeback in one kernel.
 *
 * Each warp owns a batch of attention heads. For each layer:
 *   1. Lane 0: AsyncGetBlob loads KV for this warp's heads
 *   2. All 32 lanes: compute attention (the science)
 *   3. Lane 0: AsyncPutBlob writes updated KV after writeback
 *
 * Data layout in data backend:
 *   [layer_0 KV | layer_1 KV | ... | layer_N KV]
 *   Each layer's KV = [K heads | V heads] contiguously
 */
__global__ void llm_cte_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    // Data
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    const float *d_q,                  // Query (in HBM)
    float *d_o,                        // Output (in HBM)
    const float *d_nk, const float *d_nv, // New K, V (in HBM)
    // Per-layer per-warp partition info (pinned host arrays)
    const uint64_t *warp_kv_offsets,   // byte offset into data_ptr for each warp's KV per layer
    const uint64_t *warp_kv_bytes,     // byte count for each warp's KV per layer
    const uint32_t *warp_head_start,   // first head for each warp
    const uint32_t *warp_head_end,     // last+1 head for each warp
    chi::u32 total_warps,
    chi::u32 num_layers, chi::u32 num_heads, chi::u32 seq_len, chi::u32 head_dim,
    chi::u32 decode_token_pos,
    int *d_done) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    uint32_t head_start = warp_head_start[warp_id];
    uint32_t head_end = warp_head_end[warp_id];
    chi::u64 my_kv_offset = warp_kv_offsets[warp_id];
    chi::u64 my_kv_bytes = warp_kv_bytes[warp_id];
    char *my_data = data_ptr.ptr_ + my_kv_offset;

    // Iterate over layers
    for (uint32_t layer = 0; layer < num_layers; layer++) {
      // Build blob name: "kv_l<layer>_w<warp_id>"
      using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
      char name_buf[32];
      int pos = 0;
      const char *pfx = "kv_l";
      while (*pfx) name_buf[pos++] = *pfx++;
      pos += StrT::NumberToStr(name_buf + pos, 32 - pos, layer);
      name_buf[pos++] = '_';
      pfx = "w";
      while (*pfx) name_buf[pos++] = *pfx++;
      pos += StrT::NumberToStr(name_buf + pos, 32 - pos, warp_id);
      name_buf[pos] = '\0';

      bool alloc_failed = false;

      // === I/O: GetBlob — load this warp's KV for this layer from CTE ===
      if (chi::IpcManager::IsWarpScheduler() && my_kv_bytes > 0) {
        wrp_cte::core::Client cte_client(cte_pool_id);
        hipc::ShmPtr<> shm;
        shm.alloc_id_ = data_alloc_id;
        shm.off_.exchange(data_ptr.shm_.off_.load() + my_kv_offset);

        auto get_future = cte_client.AsyncGetBlob(
            tag_id, name_buf, (chi::u64)0, my_kv_bytes,
            (chi::u32)0, shm, chi::PoolQuery::Local());
        if (!get_future.GetFutureShmPtr().IsNull()) {
          get_future.Wait();
        } else {
          alloc_failed = true;
        }
      }
      __syncwarp();

      // === COMPUTE: All 32 lanes compute attention (THE SCIENCE) ===
      if (!alloc_failed) {
        float *my_kv = reinterpret_cast<float *>(my_data);
        uint64_t kvs = (uint64_t)seq_len * head_dim;

        for (uint32_t head = head_start + lane_id; head < head_end; head += 32) {
          const float *K = my_kv + (head - head_start) * kvs;
          const float *V = my_kv + (head_end - head_start) * kvs + (head - head_start) * kvs;
          const float *Q = d_q + head * head_dim;
          float *O = d_o + head * head_dim;

          float mx = -1e30f;
          uint32_t bp = 0;
          for (uint32_t s = 0; s < seq_len; s++) {
            float d = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) {
              d += Q[i] * K[s * head_dim + i];
            }
            d /= sqrtf((float)head_dim);
            if (d > mx) { mx = d; bp = s; }
          }
          for (uint32_t i = 0; i < head_dim; i++) {
            O[i] = V[bp * head_dim + i];
          }
        }
        __syncwarp();

        // === Writeback: Update KV with new tokens ===
        for (uint32_t head = head_start + lane_id; head < head_end; head += 32) {
          float *K = my_kv + (head - head_start) * kvs;
          float *V = my_kv + (head_end - head_start) * kvs + (head - head_start) * kvs;
          const float *NK = d_nk + head * head_dim;
          const float *NV = d_nv + head * head_dim;

          for (uint32_t i = 0; i < head_dim; i++) {
            K[decode_token_pos * head_dim + i] = NK[i];
            V[decode_token_pos * head_dim + i] = NV[i];
          }
        }
        __syncwarp();
      }

      // === I/O: PutBlob — write back updated KV ===
      if (chi::IpcManager::IsWarpScheduler() && !alloc_failed && my_kv_bytes > 0) {
        wrp_cte::core::Client cte_client(cte_pool_id);
        hipc::ShmPtr<> shm;
        shm.alloc_id_ = data_alloc_id;
        shm.off_.exchange(data_ptr.shm_.off_.load() + my_kv_offset);

        auto put_future = cte_client.AsyncPutBlob(
            tag_id, name_buf, (chi::u64)0, my_kv_bytes,
            shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0,
            chi::PoolQuery::Local());
        if (!put_future.GetFutureShmPtr().IsNull()) {
          put_future.Wait();
        }
      }
      __syncwarp();
    }
  }

  // Signal completion
  if (chi::IpcManager::IsWarpScheduler()) {
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

// Alloc kernel
__global__ void llm_cte_alloc_kernel(
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

int run_workload_llm_kvcache(const WorkloadConfig &cfg, const char *mode, WorkloadResult *result) {
  uint32_t nl=cfg.param_num_layers, nh=cfg.param_num_heads, hd=cfg.param_head_dim;
  uint32_t dt=cfg.param_decode_tokens; std::string m(mode);

  // Scale seq_len so that per-head KV size ≈ warp_bytes
  // KV per head per layer = 2 * seq_len * head_dim * sizeof(float)
  uint32_t sl = cfg.param_seq_len;
  if (cfg.warp_bytes > 0) {
    uint32_t target_sl = (uint32_t)(cfg.warp_bytes / (2 * hd * sizeof(float)));
    if (target_sl > 0) sl = target_sl;
  }

  HIPRINT("  LLM params: {} layers, {} heads, {} head_dim, {} seq_len, {} decode_tokens",
          nl, nh, hd, sl, dt);

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
    // ======== CTE: Combined kernel with multi-warp I/O + compute ========
    uint32_t total_warps = (cfg.client_blocks * cfg.client_threads) / 32;
    if (total_warps == 0) total_warps = 1;
    // Cap active warps to number of heads (no point having idle warps)
    // Also must cap client_blocks to match, since CHIMAERA_GPU_CLIENT_INIT
    // runs on all launched blocks.
    uint32_t active_warps = std::min(total_warps, nh);
    uint32_t active_client_blocks = (active_warps * 32 + cfg.client_threads - 1) / cfg.client_threads;
    if (active_client_blocks == 0) active_client_blocks = 1;
    total_warps = active_warps;

    // Partition heads among warps
    uint32_t heads_per_warp = (nh + total_warps - 1) / total_warps;
    std::vector<uint32_t> h_head_start(total_warps), h_head_end(total_warps);
    std::vector<uint64_t> h_kv_offsets(total_warps), h_kv_bytes(total_warps);

    uint64_t total_kv_bytes = 0;
    for (uint32_t w = 0; w < total_warps; w++) {
      h_head_start[w] = w * heads_per_warp;
      h_head_end[w] = std::min((w + 1) * heads_per_warp, nh);
      uint32_t batch_heads = h_head_end[w] - h_head_start[w];
      h_kv_offsets[w] = total_kv_bytes;
      h_kv_bytes[w] = (uint64_t)batch_heads * 2 * sl * hd * sizeof(float);
      total_kv_bytes += h_kv_bytes[w];
    }

    HIPRINT("  CTE: {} warps, {} heads/warp, {:.1f} MB KV/layer, {} layers",
            total_warps, heads_per_warp, total_kv_bytes/(1024.0*1024.0), nl);

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    // Data backend
    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, total_kv_bytes + 4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, (size_t)total_warps*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, (size_t)total_warps*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    llm_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),
                                   total_kv_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError,"LLM CTE alloc failed"); cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator();
      cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
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
    uint64_t *h_ko, *h_kb; uint32_t *h_hs, *h_he;
    cudaMallocHost(&h_ko, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_kb, total_warps*sizeof(uint64_t));
    cudaMallocHost(&h_hs, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_he, total_warps*sizeof(uint32_t));
    memcpy(h_ko, h_kv_offsets.data(), total_warps*sizeof(uint64_t));
    memcpy(h_kb, h_kv_bytes.data(), total_warps*sizeof(uint64_t));
    memcpy(h_hs, h_head_start.data(), total_warps*sizeof(uint32_t));
    memcpy(h_he, h_head_end.data(), total_warps*sizeof(uint32_t));

    // Seed per-warp KV blobs (zero-initialized for all layers)
    char *h_all_kv = (char*)malloc(total_kv_bytes);
    memset(h_all_kv, 0, total_kv_bytes);
    cudaMemcpy(array_ptr.ptr_, h_all_kv, total_kv_bytes, cudaMemcpyHostToDevice);
    free(h_all_kv);
    cudaDeviceSynchronize();

    // Seed each warp's blobs via CTE client
    {
      wrp_cte::core::Client cte_client(cfg.cte_pool_id);
      for (uint32_t layer = 0; layer < nl; layer++) {
        for (uint32_t w = 0; w < total_warps; w++) {
          char bname[32];
          int pos = 0;
          const char *pfx = "kv_l";
          while (*pfx) bname[pos++] = *pfx++;
          pos += std::to_string(layer).copy(bname + pos, 32 - pos);
          bname[pos++] = '_';
          pfx = "w";
          while (*pfx) bname[pos++] = *pfx++;
          pos += std::to_string(w).copy(bname + pos, 32 - pos);
          bname[pos] = '\0';

          hipc::ShmPtr<> shm;
          shm.alloc_id_ = data_alloc_id;
          shm.off_.exchange(array_ptr.shm_.off_.load() + h_kv_offsets[w]);
          auto f = cte_client.AsyncPutBlob(cfg.tag_id, bname,
              (chi::u64)0, h_kv_bytes[w], shm, -1.0f,
              wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
          f.Wait();
        }
      }
      HIPRINT("  CTE: Seeded {} layers x {} warps KV blobs", nl, total_warps);
    }

    // Clear data backend before GetBlob
    cudaMemset(array_ptr.ptr_, 0, total_kv_bytes);
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Run combined CTE kernel for each decode token
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t token = 0; token < dt; token++) {
      *d_done = 0;
      if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      void *stream = hshm::GpuApi::CreateStream();
      llm_cte_kernel<<<active_client_blocks, cfg.client_threads, 0,
                       static_cast<cudaStream_t>(stream)>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, active_client_blocks,
          array_ptr, data_alloc_id,
          d_q, d_o, d_nk, d_nv,
          h_ko, h_kb, h_hs, h_he,
          total_warps, nl, nh, sl, hd, token, d_done);

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
        HLOG(kError,"LLM CTE token {} timed out",token); break;}
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric = dt/(result->elapsed_ms/1e3);
    result->metric_name = "tokens/sec";
    result->bandwidth_gbps = ((uint64_t)nl*(kvbl+2*nh*hd*4)*dt/1e9)/(result->elapsed_ms/1e3);

    cudaFreeHost(d_done); cudaFreeHost(h_ko); cudaFreeHost(h_kb);
    cudaFreeHost(h_hs); cudaFreeHost(h_he);
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m=="bam") {
    uint64_t kvb_al=((kvbt+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    uint32_t total_pages=(uint32_t)(kvb_al/cfg.bam_page_size);
    uint64_t hbm_bytes = cfg.GetHbmBytes();
    uint32_t cache_pages = (hbm_bytes > 0)
        ? std::max(1u, (uint32_t)(hbm_bytes / cfg.bam_page_size))
        : total_pages;
    bam::PageCacheConfig pcfg; pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cache_pages;
    pcfg.num_queues=0;pcfg.queue_depth=0;pcfg.backend=bam::BackendType::kHostMemory;pcfg.nvme_dev=nullptr;
    bam::PageCache cache(pcfg);
    bam::Array<float> kv_array(nl*kvpl, cache);
    std::vector<float> h_kv(nl*kvpl, 0.0f);
    kv_array.load_from_host(h_kv.data(), nl*kvpl);
    HIPRINT("  BaM HBM cache: {} / {} pages ({}%) x {} B = {:.1f} MB",
            cache_pages, total_pages,
            cache_pages * 100 / std::max(1u, total_pages), cfg.bam_page_size,
            (double)cache_pages*cfg.bam_page_size/(1024.0*1024.0));
    auto t0=std::chrono::high_resolution_clock::now();
    for(uint32_t t=0;t<dt;t++){
      for(uint32_t l=0;l<nl;l++){
        llm_attn_bam<<<blocks,threads>>>(kv_array.device(), d_q,d_o,nh,sl,hd);
        llm_kv_wb_direct<<<blocks,threads>>>(
            reinterpret_cast<float*>(cache.host_buffer())+l*kvpl,
            d_nk,d_nv,nh,sl,hd,t);
      }
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
  } else { HLOG(kError,"llm_kvcache: unknown mode '{}'",mode);
    cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
    return -1; }

  cudaFree(d_q);cudaFree(d_o);cudaFree(d_nk);cudaFree(d_nv);
  return 0;
}

#endif  // HSHM_IS_HOST
