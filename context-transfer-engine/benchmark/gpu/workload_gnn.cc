/**
 * workload_gnn.cc — Graph Neural Network for CTE GPU bench
 *
 * CTE mode: Combined kernel. Each warp owns a batch of nodes.
 *   Lane 0 calls AsyncGetBlob to load feature vectors for its batch.
 *   All 32 lanes gather and aggregate features (the science).
 *   No PutBlob (read-only workload).
 *
 * BaM mode: Uses bam::ArrayDevice<float>::read() for feature access.
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

__global__ void gnn_gather_hbm(const float *features,
                                const uint32_t *adj_list,
                                const uint64_t *adj_offsets,
                                float *output,
                                uint32_t num_nodes, uint32_t emb_dim,
                                uint32_t batch_size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t node_id = tid; node_id < batch_size; node_id += gridDim.x * blockDim.x) {
    uint64_t nbr_start = adj_offsets[node_id];
    uint64_t nbr_end = adj_offsets[node_id + 1];

    for (uint32_t f = 0; f < emb_dim; f++) {
      float sum = features[node_id * emb_dim + f];
      for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
        uint32_t nbr = adj_list[nbr_idx];
        sum += features[nbr * emb_dim + f];
      }
      output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
    }
  }
}

__global__ void gnn_gather_direct(const float *h_features,
                                   const uint32_t *h_adj_list,
                                   const uint64_t *adj_offsets,
                                   float *output,
                                   uint32_t num_nodes, uint32_t emb_dim,
                                   uint32_t batch_size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t node_id = tid; node_id < batch_size; node_id += gridDim.x * blockDim.x) {
    uint64_t nbr_start = adj_offsets[node_id];
    uint64_t nbr_end = adj_offsets[node_id + 1];

    for (uint32_t f = 0; f < emb_dim; f++) {
      float sum = h_features[node_id * emb_dim + f];
      for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
        uint32_t nbr = h_adj_list[nbr_idx];
        sum += h_features[nbr * emb_dim + f];
      }
      output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
    }
  }
}

#ifdef WRP_CORE_ENABLE_BAM
/**
 * BaM GNN gather: reads features through bam::ArrayDevice<float>.
 * Each read() call transparently goes through the HBM page cache.
 */
__global__ void gnn_gather_bam(bam::ArrayDevice<float> features,
                                const uint32_t *adj_list,
                                const uint64_t *adj_offsets,
                                float *output,
                                uint32_t num_nodes, uint32_t emb_dim,
                                uint32_t batch_size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t node_id = tid; node_id < batch_size; node_id += gridDim.x * blockDim.x) {
    uint64_t nbr_start = adj_offsets[node_id];
    uint64_t nbr_end = adj_offsets[node_id + 1];

    for (uint32_t f = 0; f < emb_dim; f++) {
      float sum = features.read(node_id * emb_dim + f);
      for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
        uint32_t nbr = adj_list[nbr_idx];
        sum += features.read(nbr * emb_dim + f);
      }
      output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
    }
  }
}
#endif

/**
 * Combined CTE GNN kernel: chunked I/O + gather in one kernel.
 *
 * The feature table is split into chunks of ~warp_bytes. Each chunk is a
 * separate CTE blob ("gnn_f0", "gnn_f1", ...). Each warp loads only the
 * chunk(s) covering its own node partition into the shared HBM buffer,
 * then waits for all warps to finish loading (via d_load_done atomic
 * counter), then computes the gather reading neighbor features from the
 * fully-populated shared buffer.
 *
 * Data layout in shared buffer (same as feature table):
 *   [chunk_0 | chunk_1 | ... | chunk_N]
 *   chunk_i = features for nodes [chunk_i_start, chunk_i_end)
 *
 * Each warp loads chunks that overlap [node_start, node_end).
 */
__global__ void gnn_cte_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    // Data
    hipc::FullPtr<char> data_ptr,
    hipc::AllocatorId data_alloc_id,
    const uint32_t *d_adj_list,         // Adjacency list (in HBM)
    const uint64_t *d_adj_offsets,      // Adjacency offsets (in HBM)
    float *d_output,                    // Output features (in HBM)
    // Per-chunk info (pinned host arrays, indexed by chunk_id)
    const uint64_t *chunk_byte_offsets, // byte offset into data_ptr for chunk
    const uint64_t *chunk_byte_sizes,   // byte count for chunk
    const uint32_t *chunk_node_start,   // first node in chunk
    const uint32_t *chunk_node_end,     // last+1 node in chunk
    chi::u32 num_chunks,
    // Per-warp info
    const uint32_t *warp_node_start,    // first node for this warp
    const uint32_t *warp_node_end,      // last+1 node for this warp
    const uint32_t *warp_first_chunk,   // first chunk this warp must load
    const uint32_t *warp_last_chunk,    // last+1 chunk this warp must load
    chi::u32 total_warps,
    chi::u32 emb_dim,
    int *d_load_done,                   // atomic counter for load barrier
    int *d_done) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    uint32_t my_node_start = warp_node_start[warp_id];
    uint32_t my_node_end = warp_node_end[warp_id];
    uint32_t my_first_chunk = warp_first_chunk[warp_id];
    uint32_t my_last_chunk = warp_last_chunk[warp_id];
    bool alloc_failed = false;

    // === PHASE 1: I/O — load only this warp's chunks from CTE ===
    if (chi::IpcManager::IsWarpScheduler()) {
      wrp_cte::core::Client cte_client(cte_pool_id);

      for (uint32_t c = my_first_chunk; c < my_last_chunk; c++) {
        uint64_t c_offset = chunk_byte_offsets[c];
        uint64_t c_size = chunk_byte_sizes[c];
        if (c_size == 0) continue;

        // Build blob name: "gnn_f<chunk_id>"
        using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
        char name_buf[32];
        int pos = 0;
        const char *pfx = "gnn_f";
        while (*pfx) name_buf[pos++] = *pfx++;
        pos += StrT::NumberToStr(name_buf + pos, 32 - pos, c);
        name_buf[pos] = '\0';

        hipc::ShmPtr<> shm;
        shm.alloc_id_ = data_alloc_id;
        shm.off_.exchange(data_ptr.shm_.off_.load() + c_offset);

        auto get_future = cte_client.AsyncGetBlob(
            tag_id, name_buf, (chi::u64)0, c_size,
            (chi::u32)0, shm, chi::PoolQuery::Local());
        if (!get_future.GetFutureShmPtr().IsNull()) {
          get_future.Wait();
        } else {
          alloc_failed = true;
          break;
        }
      }
    }
    __syncwarp();

    // Signal that this warp's load is done
    if (chi::IpcManager::IsWarpScheduler()) {
      atomicAdd_system(d_load_done, 1);
      __threadfence_system();
    }

    // === BARRIER: spin until all warps have finished loading ===
    if (chi::IpcManager::IsWarpScheduler()) {
      while (atomicAdd(d_load_done, 0) < (int)total_warps) {
        __threadfence_system();
      }
    }
    __syncwarp();

    // === PHASE 2: COMPUTE — gather features from shared HBM buffer ===
    if (!alloc_failed) {
      const float *all_features = reinterpret_cast<const float *>(data_ptr.ptr_);

      for (uint32_t node_id = my_node_start + lane_id; node_id < my_node_end; node_id += 32) {
        uint64_t nbr_start = d_adj_offsets[node_id];
        uint64_t nbr_end = d_adj_offsets[node_id + 1];

        for (uint32_t f = 0; f < emb_dim; f++) {
          float sum = all_features[node_id * emb_dim + f];
          for (uint64_t nbr_idx = nbr_start; nbr_idx < nbr_end; nbr_idx++) {
            uint32_t nbr = d_adj_list[nbr_idx];
            sum += all_features[nbr * emb_dim + f];
          }
          d_output[node_id * emb_dim + f] = sum / (1.0f + (float)(nbr_end - nbr_start));
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
__global__ void gnn_cte_alloc_kernel(
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

static std::vector<float> gen_features(uint32_t num_nodes, uint32_t emb_dim, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> features(num_nodes * emb_dim);
  for (size_t i = 0; i < features.size(); i++) {
    features[i] = dist(rng);
  }
  return features;
}

static std::pair<std::vector<uint32_t>, std::vector<uint64_t>> gen_adjacency(
    uint32_t num_nodes, uint32_t avg_neighbors, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::vector<std::vector<uint32_t>> adj(num_nodes);
  std::uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);

  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t deg = std::max(1u, avg_neighbors);
    for (uint32_t j = 0; j < deg; j++) {
      uint32_t nbr = node_dist(rng);
      if (nbr != i) adj[i].push_back(nbr);
    }
    std::sort(adj[i].begin(), adj[i].end());
    adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
  }

  std::vector<uint64_t> offsets(num_nodes + 1);
  offsets[0] = 0;
  for (uint32_t i = 0; i < num_nodes; i++) {
    offsets[i + 1] = offsets[i] + adj[i].size();
  }

  std::vector<uint32_t> adj_list(offsets[num_nodes]);
  for (uint32_t i = 0; i < num_nodes; i++) {
    std::copy(adj[i].begin(), adj[i].end(), adj_list.begin() + offsets[i]);
  }

  return {adj_list, offsets};
}

int run_workload_gnn(const WorkloadConfig &cfg, const char *mode,
                     WorkloadResult *result) {
  uint32_t num_nodes = cfg.param_num_nodes;
  uint32_t emb_dim = cfg.param_emb_dim;
  int iters = cfg.iterations > 0 ? cfg.iterations : 10;
  std::string m(mode);

  HIPRINT("  Generating GNN graph: {} nodes, {} dims, avg nbrs {}",
          num_nodes, emb_dim, cfg.param_avg_degree);
  auto features = gen_features(num_nodes, emb_dim);
  auto [adj_list, adj_offsets] = gen_adjacency(num_nodes, cfg.param_avg_degree);
  uint64_t adj_bytes = (uint64_t)adj_list.size() * sizeof(uint32_t);
  HIPRINT("  GNN: {} edges ({:.1f} MB)", adj_list.size(), adj_bytes/(1024.0*1024.0));

  uint64_t *d_offsets; cudaMalloc(&d_offsets, (num_nodes+1)*sizeof(uint64_t));
  cudaMemcpy(d_offsets, adj_offsets.data(), (num_nodes+1)*sizeof(uint64_t), cudaMemcpyHostToDevice);
  uint32_t *d_adj; cudaMalloc(&d_adj, adj_bytes);
  cudaMemcpy(d_adj, adj_list.data(), adj_bytes, cudaMemcpyHostToDevice);
  float *d_output; cudaMalloc(&d_output, num_nodes*emb_dim*sizeof(float));
  cudaMemset(d_output, 0, num_nodes*emb_dim*sizeof(float));

  uint32_t threads = 256, comp_blocks = (cfg.client_blocks * cfg.client_threads + threads - 1) / threads;
  if (!comp_blocks) comp_blocks = 1;

  if (m == "cte") {
    // ======== CTE: Chunked I/O — each warp loads only its partition ========
    //
    // Feature table split into chunks of ~warp_bytes. Each chunk is a
    // separate CTE blob. Each warp loads only the chunks covering its
    // node partition, then all warps read neighbor features from the
    // shared HBM buffer (populated collectively by all warps).
    uint32_t total_warps = (cfg.client_blocks * cfg.client_threads) / 32;
    if (total_warps == 0) total_warps = 1;
    uint64_t feat_bytes = (uint64_t)num_nodes * emb_dim * sizeof(float);
    uint64_t bytes_per_node = (uint64_t)emb_dim * sizeof(float);

    // Chunk the feature table into blobs of ~warp_bytes
    uint64_t chunk_size = cfg.warp_bytes;
    if (chunk_size < bytes_per_node) chunk_size = bytes_per_node; // at least 1 node per chunk
    uint32_t nodes_per_chunk = (uint32_t)(chunk_size / bytes_per_node);
    if (nodes_per_chunk == 0) nodes_per_chunk = 1;
    uint32_t num_chunks = (num_nodes + nodes_per_chunk - 1) / nodes_per_chunk;

    // Build per-chunk info
    std::vector<uint64_t> h_chunk_offsets(num_chunks);
    std::vector<uint64_t> h_chunk_sizes(num_chunks);
    std::vector<uint32_t> h_chunk_node_start(num_chunks);
    std::vector<uint32_t> h_chunk_node_end(num_chunks);
    for (uint32_t c = 0; c < num_chunks; c++) {
      h_chunk_node_start[c] = c * nodes_per_chunk;
      h_chunk_node_end[c] = std::min((c + 1) * nodes_per_chunk, num_nodes);
      h_chunk_offsets[c] = (uint64_t)h_chunk_node_start[c] * bytes_per_node;
      h_chunk_sizes[c] = (uint64_t)(h_chunk_node_end[c] - h_chunk_node_start[c]) * bytes_per_node;
    }

    // Partition nodes among warps
    uint32_t nodes_per_warp = (num_nodes + total_warps - 1) / total_warps;
    std::vector<uint32_t> h_warp_node_start(total_warps), h_warp_node_end(total_warps);
    std::vector<uint32_t> h_warp_first_chunk(total_warps), h_warp_last_chunk(total_warps);
    for (uint32_t w = 0; w < total_warps; w++) {
      h_warp_node_start[w] = w * nodes_per_warp;
      h_warp_node_end[w] = std::min((w + 1) * nodes_per_warp, num_nodes);
      // Which chunks does this warp need? Its node range maps to contiguous chunks.
      h_warp_first_chunk[w] = h_warp_node_start[w] / nodes_per_chunk;
      h_warp_last_chunk[w] = (h_warp_node_end[w] + nodes_per_chunk - 1) / nodes_per_chunk;
      if (h_warp_last_chunk[w] > num_chunks) h_warp_last_chunk[w] = num_chunks;
    }

    uint64_t total_warp_io = 0;
    for (uint32_t w = 0; w < total_warps; w++) {
      for (uint32_t c = h_warp_first_chunk[w]; c < h_warp_last_chunk[w]; c++)
        total_warp_io += h_chunk_sizes[c];
    }

    HIPRINT("  CTE: {} warps, {} chunks of ~{} KB, {:.1f} MB features total, "
            "{:.1f} MB I/O per iteration",
            total_warps, num_chunks, chunk_size / 1024,
            feat_bytes / (1024.0 * 1024.0),
            total_warp_io / (1024.0 * 1024.0));

    CHI_IPC->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CHI_IPC->PauseGpuOrchestrator();

    hipc::MemoryBackendId data_id(200,0); hipc::GpuMalloc data_backend;
    data_backend.shm_init(data_id, feat_bytes + 4*1024*1024, "", 0);
    hipc::MemoryBackendId scratch_id(201,0); hipc::GpuMalloc scratch_backend;
    scratch_backend.shm_init(scratch_id, (size_t)total_warps*1024*1024, "", 0);
    hipc::MemoryBackendId heap_id(202,0); hipc::GpuMalloc heap_backend;
    heap_backend.shm_init(heap_id, (size_t)total_warps*1024*1024, "", 0);

    hipc::FullPtr<char> *d_ptr; cudaMallocHost(&d_ptr, sizeof(hipc::FullPtr<char>));
    d_ptr->SetNull();
    gnn_cte_alloc_kernel<<<1,1>>>(static_cast<hipc::MemoryBackend&>(data_backend),
                                   feat_bytes, d_ptr);
    cudaDeviceSynchronize();
    if (d_ptr->IsNull()) {
      HLOG(kError,"GNN CTE alloc failed"); cudaFreeHost(d_ptr);
      CHI_IPC->ResumeGpuOrchestrator();
      cudaFree(d_offsets);cudaFree(d_adj);cudaFree(d_output);
      return -2;
    }
    hipc::FullPtr<char> array_ptr = *d_ptr; cudaFreeHost(d_ptr);
    hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
    CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_, data_backend.data_capacity_);

    chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
    gpu_info.backend = scratch_backend;
    int *d_done; cudaMallocHost(&d_done, sizeof(int)); *d_done = 0;
    int *d_load_done; cudaMallocHost(&d_load_done, sizeof(int)); *d_load_done = 0;
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Copy per-chunk info to pinned memory
    uint64_t *h_co, *h_cs; uint32_t *h_cns, *h_cne;
    cudaMallocHost(&h_co, num_chunks*sizeof(uint64_t));
    cudaMallocHost(&h_cs, num_chunks*sizeof(uint64_t));
    cudaMallocHost(&h_cns, num_chunks*sizeof(uint32_t));
    cudaMallocHost(&h_cne, num_chunks*sizeof(uint32_t));
    memcpy(h_co, h_chunk_offsets.data(), num_chunks*sizeof(uint64_t));
    memcpy(h_cs, h_chunk_sizes.data(), num_chunks*sizeof(uint64_t));
    memcpy(h_cns, h_chunk_node_start.data(), num_chunks*sizeof(uint32_t));
    memcpy(h_cne, h_chunk_node_end.data(), num_chunks*sizeof(uint32_t));

    // Copy per-warp info to pinned memory
    uint32_t *h_wns, *h_wne, *h_wfc, *h_wlc;
    cudaMallocHost(&h_wns, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_wne, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_wfc, total_warps*sizeof(uint32_t));
    cudaMallocHost(&h_wlc, total_warps*sizeof(uint32_t));
    memcpy(h_wns, h_warp_node_start.data(), total_warps*sizeof(uint32_t));
    memcpy(h_wne, h_warp_node_end.data(), total_warps*sizeof(uint32_t));
    memcpy(h_wfc, h_warp_first_chunk.data(), total_warps*sizeof(uint32_t));
    memcpy(h_wlc, h_warp_last_chunk.data(), total_warps*sizeof(uint32_t));

    // Seed per-chunk blobs via host CTE client
    cudaMemcpy(array_ptr.ptr_, features.data(), feat_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
      wrp_cte::core::Client cte_client(cfg.cte_pool_id);
      for (uint32_t c = 0; c < num_chunks; c++) {
        std::string bname = "gnn_f" + std::to_string(c);
        hipc::ShmPtr<> shm;
        shm.alloc_id_ = data_alloc_id;
        shm.off_.exchange(array_ptr.shm_.off_.load() + h_chunk_offsets[c]);
        auto f = cte_client.AsyncPutBlob(cfg.tag_id, bname,
            (chi::u64)0, h_chunk_sizes[c], shm, -1.0f,
            wrp_cte::core::Context(), (chi::u32)0, chi::PoolQuery::Local());
        f.Wait();
      }
      HIPRINT("  CTE: Seeded {} feature chunk blobs ({:.1f} MB total)",
              num_chunks, feat_bytes / (1024.0 * 1024.0));
    }

    // Clear data backend before GetBlob
    cudaMemset(array_ptr.ptr_, 0, feat_bytes);
    if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    if(heap_backend.data_) cudaMemset(heap_backend.data_,0,sizeof(hipc::PartitionedAllocator));
    cudaDeviceSynchronize();

    // Run combined CTE kernel for each iteration
    auto t0 = std::chrono::high_resolution_clock::now();
    int iter;
    for (iter = 0; iter < iters; iter++) {
      *d_done = 0;
      *d_load_done = 0;
      if(scratch_backend.data_) cudaMemset(scratch_backend.data_,0,sizeof(hipc::PartitionedAllocator));
      cudaDeviceSynchronize();

      void *stream = hshm::GpuApi::CreateStream();
      gnn_cte_kernel<<<cfg.client_blocks, cfg.client_threads, 0,
                       static_cast<cudaStream_t>(stream)>>>(
          gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.client_blocks,
          array_ptr, data_alloc_id,
          d_adj, d_offsets, d_output,
          h_co, h_cs, h_cns, h_cne, num_chunks,
          h_wns, h_wne, h_wfc, h_wlc,
          total_warps, emb_dim, d_load_done, d_done);

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
        HLOG(kError,"GNN CTE iter {} timed out",iter); break;}
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    result->elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric = (double)adj_list.size()*iter/(result->elapsed_ms/1e3);
    result->metric_name = "edges/sec";
    result->bandwidth_gbps = (total_warp_io*(double)iter)/(1e9*(result->elapsed_ms/1e3));

    cudaFreeHost(d_done); cudaFreeHost(d_load_done);
    cudaFreeHost(h_co); cudaFreeHost(h_cs); cudaFreeHost(h_cns); cudaFreeHost(h_cne);
    cudaFreeHost(h_wns); cudaFreeHost(h_wne); cudaFreeHost(h_wfc); cudaFreeHost(h_wlc);
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m == "bam") {
    // ======== BaM: Uses bam::ArrayDevice<float>::read() ========
    uint64_t feature_bytes = num_nodes * emb_dim * sizeof(float);
    uint64_t fb_aligned = ((feature_bytes+cfg.bam_page_size-1)/cfg.bam_page_size)*cfg.bam_page_size;
    uint32_t total_pages = (uint32_t)(fb_aligned / cfg.bam_page_size);
    uint32_t cache_pages = std::max(1u, total_pages * cfg.hbm_cache_pct / 100);

    bam::PageCacheConfig pcfg;
    pcfg.page_size=cfg.bam_page_size; pcfg.num_pages=cache_pages;
    pcfg.num_queues=0; pcfg.queue_depth=0;
    pcfg.backend=bam::BackendType::kHostMemory; pcfg.nvme_dev=nullptr;

    bam::PageCache cache(pcfg);
    bam::Array<float> feat_array(num_nodes * emb_dim, cache);
    feat_array.load_from_host(features.data(), num_nodes * emb_dim);

    HIPRINT("  BaM HBM cache: {} / {} pages ({}%) x {} B = {:.1f} MB",
            cache_pages, total_pages, cfg.hbm_cache_pct, cfg.bam_page_size,
            (double)cache_pages*cfg.bam_page_size/(1024.0*1024.0));

    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      gnn_gather_bam<<<comp_blocks,threads>>>(feat_array.device(),
                                               d_adj, d_offsets, d_output,
                                               num_nodes, emb_dim, num_nodes);
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));
  }
#endif

  else if (m=="hbm") {
    float *d_features; cudaMalloc(&d_features, num_nodes*emb_dim*sizeof(float));
    cudaMemcpy(d_features, features.data(), num_nodes*emb_dim*sizeof(float), cudaMemcpyHostToDevice);
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      gnn_gather_hbm<<<comp_blocks,threads>>>(d_features, d_adj, d_offsets, d_output,
                                               num_nodes, emb_dim, num_nodes);
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));
    cudaFree(d_features);
  }

  else if (m=="direct") {
    float *h_features; cudaMallocHost(&h_features, num_nodes*emb_dim*sizeof(float));
    memcpy(h_features, features.data(), num_nodes*emb_dim*sizeof(float));
    auto t0=std::chrono::high_resolution_clock::now(); int iter;
    for(iter=0;iter<iters;iter++){
      gnn_gather_direct<<<comp_blocks,threads>>>(h_features, d_adj, d_offsets, d_output,
                                                  num_nodes, emb_dim, num_nodes);
      cudaDeviceSynchronize();
    }
    auto t1=std::chrono::high_resolution_clock::now();
    result->elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result->primary_metric=(double)adj_list.size()*iters/(result->elapsed_ms/1e3);
    result->metric_name="edges/sec";
    result->bandwidth_gbps=(adj_list.size()*4.0*iters + num_nodes*emb_dim*4.0*iters)/(1e9*(result->elapsed_ms/1e3));
    cudaFreeHost(h_features);
  }

  else {
    HLOG(kError,"gnn: unknown mode '{}'",mode);
    cudaFree(d_offsets);cudaFree(d_adj);cudaFree(d_output);
    return -1;
  }

  cudaFree(d_offsets);cudaFree(d_adj);cudaFree(d_output);
  return 0;
}

#endif  // HSHM_IS_HOST
