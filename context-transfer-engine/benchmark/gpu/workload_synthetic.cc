/**
 * workload_synthetic.cc — Synthetic I/O benchmark for CTE GPU bench
 *
 * Measures raw put/get throughput across 4 modes: cte, bam, direct, hbm.
 * No science — pure I/O. Each warp performs sequential or random put/get
 * operations on a fixed-size buffer per iteration.
 *
 * CTE mode: AsyncPutBlob + AsyncGetBlob round-trip per iteration
 * BaM mode: warp_page_cache_acquire for write/read through HBM cache
 * Direct: memcpy to/from pinned DRAM
 * HBM: memcpy within device memory
 */

#include <cstdint>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

#ifdef WRP_CORE_ENABLE_BAM
#include <bam/array.cuh>
#endif

// ================================================================
// HBM kernel: device memcpy, no I/O
// ================================================================

__global__ void synthetic_hbm_kernel(
    char *write_buffer,
    char *read_buffer,
    uint64_t warp_bytes,
    uint32_t total_warps,
    uint32_t iterations,
    bool validate,
    int *d_errors) {

  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;

  if (warp_id >= total_warps) return;

  // Each warp works on its own slice
  char *my_write = write_buffer + warp_id * warp_bytes;
  char *my_read = read_buffer + warp_id * warp_bytes;

  for (uint32_t iter = 0; iter < iterations; iter++) {
    // Fill write buffer (all lanes cooperate to fill)
    for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
      if (i + 8 <= warp_bytes) {
        *(uint64_t*)(my_write + i) = (((uint64_t)warp_id << 32) | iter);
      }
    }
    __syncwarp();

    // Memcpy: write -> read (simulate put/get)
    for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
      if (i + 8 <= warp_bytes) {
        *(uint64_t*)(my_read + i) = *(uint64_t*)(my_write + i);
      }
    }
    __syncwarp();

    // Optional validation
    if (validate) {
      for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
        if (i + 8 <= warp_bytes) {
          uint64_t expected = (((uint64_t)warp_id << 32) | iter);
          uint64_t got = *(uint64_t*)(my_read + i);
          if (got != expected) {
            atomicAdd(d_errors, 1);
          }
        }
      }
      __syncwarp();
    }
  }
}

// ================================================================
// Direct kernel: pinned DRAM memcpy
// ================================================================

__global__ void synthetic_direct_kernel(
    char *h_write_buffer,  // pinned host memory
    char *h_read_buffer,   // pinned host memory
    uint64_t warp_bytes,
    uint32_t total_warps,
    uint32_t iterations,
    bool validate,
    int *d_errors) {

  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;

  if (warp_id >= total_warps) return;

  // Each warp works on its own slice of pinned memory
  char *my_write = h_write_buffer + warp_id * warp_bytes;
  char *my_read = h_read_buffer + warp_id * warp_bytes;

  for (uint32_t iter = 0; iter < iterations; iter++) {
    // Fill write buffer
    for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
      if (i + 8 <= warp_bytes) {
        *(uint64_t*)(my_write + i) = (((uint64_t)warp_id << 32) | iter);
      }
    }
    __syncwarp();

    // Memcpy: write -> read (PCIe + host buffer coherency)
    for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
      if (i + 8 <= warp_bytes) {
        *(uint64_t*)(my_read + i) = *(uint64_t*)(my_write + i);
      }
    }
    __syncwarp();

    // Optional validation
    if (validate) {
      for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
        if (i + 8 <= warp_bytes) {
          uint64_t expected = (((uint64_t)warp_id << 32) | iter);
          uint64_t got = *(uint64_t*)(my_read + i);
          if (got != expected) {
            atomicAdd(d_errors, 1);
          }
        }
      }
      __syncwarp();
    }
  }
}

// ================================================================
// BaM kernel: warp-cooperative page cache (GIDS-style)
//
// Matches GIDS bam_ptr pattern: acquire page once, access elements via raw
// pointer, only mark dirty — NO per-page DRAM flush during iteration.
// Writeback is deferred (dirty pages flushed on eviction or at end).
//
// Each warp iterates over 64KB pages:
//   1. Lane 0 calls warp_page_cache_acquire (1 atomicCAS)
//   2. If cache miss: all 32 lanes cooperatively load page from DRAM
//   3. All 32 lanes read/write the raw HBM page pointer directly
//   4. Mark dirty (no flush) — like GIDS bam_ptr::operator[]
//
// This avoids per-element atomic overhead AND per-page PCIe flushes.
// ================================================================

#ifdef WRP_CORE_ENABLE_BAM
__global__ void synthetic_bam_kernel(
    bam::ArrayDevice<char> write_array,
    bam::ArrayDevice<char> read_array,
    uint64_t warp_bytes,
    uint32_t total_warps,
    uint32_t iterations,
    bool validate,
    int *d_errors) {

  uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  uint32_t lane_id = threadIdx.x & 31;

  if (warp_id >= total_warps) return;

  uint32_t page_size = write_array.cache_state.page_size;
  uint64_t warp_base = (uint64_t)warp_id * warp_bytes;

  for (uint32_t iter = 0; iter < iterations; iter++) {
    uint32_t pat_lo = iter;
    uint32_t pat_hi = warp_id;

    // === WRITE: iterate over pages in this warp's slice ===
    for (uint64_t page_off = warp_base; page_off < warp_base + warp_bytes;
         page_off += page_size) {
      uint64_t bytes_left = (warp_base + warp_bytes) - page_off;
      uint32_t this_page = (bytes_left < page_size) ? (uint32_t)bytes_left : page_size;
      uint32_t this_uint4s = this_page / sizeof(uint4);

      // Acquire page — 1 atomicCAS per page, broadcast to all lanes
      bool needs_load;
      uint8_t *page = bam::warp_page_cache_acquire(
          write_array.cache_state, page_off, &needs_load);

      if (needs_load) {
        bam::warp_host_read_page(page, write_array.host_base,
                                  page_off, page_size);
        bam::warp_page_cache_finish_load(write_array.cache_state, page_off);
      }

      // All 32 lanes write pattern into HBM cached page (coalesced uint4)
      uint4 pat4;
      pat4.x = pat_lo; pat4.y = pat_hi; pat4.z = pat_lo; pat4.w = pat_hi;
      for (uint32_t i = lane_id; i < this_uint4s; i += 32) {
        reinterpret_cast<uint4 *>(page)[i] = pat4;
      }
      __syncwarp();

      // Mark dirty — NO flush to DRAM (deferred, like GIDS bam_ptr)
      if (lane_id == 0) {
        bam::page_cache_mark_dirty(write_array.cache_state, page_off);
      }
      __syncwarp();
    }

    // === READ: iterate over pages, copy from write cache to read cache ===
    for (uint64_t page_off = warp_base; page_off < warp_base + warp_bytes;
         page_off += page_size) {
      uint64_t bytes_left = (warp_base + warp_bytes) - page_off;
      uint32_t this_page = (bytes_left < page_size) ? (uint32_t)bytes_left : page_size;
      uint32_t this_uint4s = this_page / sizeof(uint4);

      // Acquire write_array page (cache hit — just wrote it)
      bool needs_load_w;
      uint8_t *w_page = bam::warp_page_cache_acquire(
          write_array.cache_state, page_off, &needs_load_w);
      if (needs_load_w) {
        bam::warp_host_read_page(w_page, write_array.host_base,
                                  page_off, page_size);
        bam::warp_page_cache_finish_load(write_array.cache_state, page_off);
      }

      // Acquire read_array page
      bool needs_load_r;
      uint8_t *r_page = bam::warp_page_cache_acquire(
          read_array.cache_state, page_off, &needs_load_r);
      if (needs_load_r) {
        bam::warp_host_read_page(r_page, read_array.host_base,
                                  page_off, page_size);
        bam::warp_page_cache_finish_load(read_array.cache_state, page_off);
      }

      // All 32 lanes copy HBM→HBM (coalesced uint4)
      for (uint32_t i = lane_id; i < this_uint4s; i += 32) {
        reinterpret_cast<uint4 *>(r_page)[i] =
            reinterpret_cast<const uint4 *>(w_page)[i];
      }
      __syncwarp();

      // Mark dirty — no flush
      if (lane_id == 0) {
        bam::page_cache_mark_dirty(read_array.cache_state, page_off);
      }
      __syncwarp();
    }

    // Optional validation (reads from HBM cache, no DRAM round-trip)
    if (validate) {
      for (uint64_t page_off = warp_base; page_off < warp_base + warp_bytes;
           page_off += page_size) {
        uint64_t bytes_left = (warp_base + warp_bytes) - page_off;
        uint32_t this_page = (bytes_left < page_size) ? (uint32_t)bytes_left : page_size;
        uint32_t this_uint4s = this_page / sizeof(uint4);

        bool needs_load;
        uint8_t *r_page = bam::warp_page_cache_acquire(
            read_array.cache_state, page_off, &needs_load);
        if (needs_load) {
          bam::warp_host_read_page(r_page, read_array.host_base,
                                    page_off, page_size);
          bam::warp_page_cache_finish_load(read_array.cache_state, page_off);
        }

        uint4 pat4;
        pat4.x = pat_lo; pat4.y = pat_hi; pat4.z = pat_lo; pat4.w = pat_hi;
        for (uint32_t i = lane_id; i < this_uint4s; i += 32) {
          uint4 got = reinterpret_cast<const uint4 *>(r_page)[i];
          if (got.x != pat4.x || got.y != pat4.y ||
              got.z != pat4.z || got.w != pat4.w) {
            atomicAdd(d_errors, 1);
          }
        }
        __syncwarp();
      }
    }
  }
}
#endif

// ================================================================
// CTE kernel: AsyncPutBlob + AsyncGetBlob round-trip per iteration
// ================================================================

/**
 * LCG random number generator for random access pattern.
 * seed = seed * 6364136223846793005 + 1442695040888963407
 */
__device__ static inline uint64_t lcg_next(uint64_t &seed) {
  seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return seed >> 33;
}

__global__ void synthetic_cte_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    // Data — absolute device pointers
    char *write_base,
    char *read_base,
    uint64_t warp_bytes,
    uint32_t total_warps,
    uint32_t iterations,
    bool to_cpu,
    bool validate,
    int *d_done,
    int *d_errors) {

  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::gpu::IpcManager::GetLaneId();

  if (warp_id >= total_warps) return;

  char *my_write = write_base + warp_id * warp_bytes;
  char *my_read = read_base + warp_id * warp_bytes;

  uint64_t lcg_seed = warp_id + 1;  // LCG seed (non-zero)
  chi::PoolQuery pool_query = to_cpu ? chi::PoolQuery::ToLocalCpu() : chi::PoolQuery::Local();

  // Build blob name: "syn_w<warp_id>"
  using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
  char name_buf[32];
  int pos = 0;
  const char *pfx = "syn_w";
  while (*pfx) name_buf[pos++] = *pfx++;
  pos += StrT::NumberToStr(name_buf + pos, 32 - pos, warp_id);
  name_buf[pos] = '\0';

  bool alloc_failed = false;

  for (uint32_t iter = 0; iter < iterations; iter++) {
    // Determine offset (sequential or random)
    uint64_t offset = 0;
    if (chi::gpu::IpcManager::GetLaneId() == 0) {
      offset = (lcg_next(lcg_seed) % total_warps) * warp_bytes;
    }

    // === I/O: PutBlob — send data to CTE ===
    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      // Fill buffer with pattern
      uint64_t pattern = (((uint64_t)warp_id << 32) | iter);
      for (uint64_t i = 0; i < warp_bytes; i += 8) {
        if (i + 8 <= warp_bytes) {
          *(uint64_t*)(my_write + i) = pattern;
        }
      }

      // Use null alloc_id + absolute device address
      hipc::ShmPtr<> shm;
      shm.alloc_id_.SetNull();
      shm.off_.exchange(reinterpret_cast<size_t>(my_write));

      auto put_task = CHI_IPC->NewTask<wrp_cte::core::PutBlobTask>(
          chi::CreateTaskId(), cte_pool_id, pool_query,
          tag_id, name_buf, offset, warp_bytes,
          shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0);
      auto put_future = CHI_IPC->Send(put_task);
      put_future.WaitGpu();
    }
    __syncwarp();

    // === I/O: GetBlob — retrieve data from CTE ===
    if (chi::gpu::IpcManager::IsWarpScheduler() && !alloc_failed) {
      hipc::ShmPtr<> shm;
      shm.alloc_id_.SetNull();
      shm.off_.exchange(reinterpret_cast<size_t>(my_read));

      auto get_task = CHI_IPC->NewTask<wrp_cte::core::GetBlobTask>(
          chi::CreateTaskId(), cte_pool_id, pool_query,
          tag_id, name_buf, offset, warp_bytes,
          (chi::u32)0, shm);
      auto get_future = CHI_IPC->Send(get_task);
      get_future.WaitGpu();
    }
    __syncwarp();

    // === Optional validation ===
    if (validate && !alloc_failed) {
      uint64_t expected = (((uint64_t)warp_id << 32) | iter);
      for (uint64_t i = lane_id * 8; i < warp_bytes; i += 32 * 8) {
        if (i + 8 <= warp_bytes) {
          uint64_t got = *(uint64_t*)(my_read + i);
          if (got != expected) {
            atomicAdd(d_errors, 1);
          }
        }
      }
      __syncwarp();
    }
  }

  // Signal completion
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    atomicAdd_system(d_done, 1);
    __threadfence_system();
  }
}

// ================================================================
// Alloc kernel (used by CTE mode)
// ================================================================

__global__ void synthetic_cte_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

// ================================================================
#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "workload.h"
#include "cte_helpers.h"
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#ifdef WRP_CORE_ENABLE_BAM
#include <bam/page_cache_host.h>
#endif
#include <cstring>
#include <chrono>
#include <thread>

int run_workload_synthetic(const WorkloadConfig &cfg, const char *mode,
                           WorkloadResult *result) {
  std::string m(mode);

  // Compute grid dimensions
  uint32_t total_warps = (cfg.client_blocks * cfg.client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  // Compute warp_bytes if not set
  uint64_t warp_bytes = cfg.warp_bytes;
  if (warp_bytes == 0) {
    warp_bytes = 4096;  // default
  }

  uint64_t total_bytes = (uint64_t)total_warps * warp_bytes;
  uint32_t iters = cfg.iterations > 0 ? cfg.iterations : 10;

  HIPRINT("  Synthetic I/O: {} warps, {} KB/warp, {} iterations",
          total_warps, warp_bytes / 1024, iters);

  if (m == "cte") {
    // ======== CTE: AsyncPutBlob + AsyncGetBlob per iteration ========

    CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(cfg.rt_blocks, cfg.rt_threads);
    CteGpuContext ctx;
    if (ctx.init(total_bytes * 2, total_warps) != 0) {
      HLOG(kError, "synthetic CTE init failed");
      return -1;
    }

    // Seed data in CTE via host client
    {
      wrp_cte::core::Client cte_client(cfg.cte_pool_id);
      std::vector<char> host_data(total_bytes);
      memset(host_data.data(), 0xAA, total_bytes);

      // Upload all warp blobs
      for (uint32_t w = 0; w < total_warps; w++) {
        using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;
        char name_buf[32];
        int pos = 0;
        const char *pfx = "syn_w";
        while (*pfx) name_buf[pos++] = *pfx++;
        pos += StrT::NumberToStr(name_buf + pos, 32 - pos, w);
        name_buf[pos] = '\0';

        hipc::ShmPtr<> shm;
        shm.alloc_id_ = hipc::AllocatorId(ctx.data_id.major_, ctx.data_id.minor_);
        shm.off_.exchange(ctx.array_ptr.shm_.off_.load() + w * warp_bytes);

        cudaMemcpy(ctx.array_ptr.ptr_ + w * warp_bytes,
                   host_data.data() + (w % 1) * warp_bytes,
                   warp_bytes, cudaMemcpyHostToDevice);

        auto f = cte_client.AsyncPutBlob(
            cfg.tag_id, name_buf, (chi::u64)0, warp_bytes,
            shm, -1.0f, wrp_cte::core::Context(), (chi::u32)0,
            cfg.routing == "to_cpu" ? chi::PoolQuery::ToLocalCpu() : chi::PoolQuery::Local());
        f.Wait();
      }
      HIPRINT("  CTE: Seeded {} warp blobs ({:.1f} MB)",
              total_warps, total_bytes / (1024.0 * 1024.0));
    }

    // Read buffer is second half of the allocated region
    char *read_base = ctx.array_ptr.ptr_ + total_bytes;

    int *d_errors;
    cudaMallocHost(&d_errors, sizeof(int));
    *d_errors = 0;

    // Run synthetic kernel
    auto t0 = std::chrono::high_resolution_clock::now();

    void *stream = hshm::GpuApi::CreateStream();
    PrintKernelInfo("synthetic_cte_kernel",
                    (const void *)synthetic_cte_kernel,
                    cfg.client_blocks, cfg.client_threads);
    synthetic_cte_kernel<<<cfg.client_blocks, cfg.client_threads, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        ctx.gpu_info, cfg.cte_pool_id, cfg.tag_id, cfg.client_blocks,
        ctx.array_ptr.ptr_, read_base,
        warp_bytes, total_warps, iters,
        cfg.routing == "to_cpu", cfg.validate, ctx.d_done, d_errors);

    if (!ctx.resume_and_poll(cfg.timeout_sec)) {
      HLOG(kError, "synthetic CTE timed out");
      CHI_CPU_IPC->PauseGpuOrchestrator();
      hshm::GpuApi::Synchronize(stream);
      hshm::GpuApi::DestroyStream(stream);
      cudaFreeHost(d_errors);
      ctx.cleanup();
      return -2;
    }

    // Pause orchestrator before synchronizing client stream
    CHI_CPU_IPC->PauseGpuOrchestrator();
    hshm::GpuApi::Synchronize(stream);
    hshm::GpuApi::DestroyStream(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cfg.validate && *d_errors > 0) {
      HLOG(kWarning, "synthetic CTE validation: {} errors", *d_errors);
    }

    // Metrics: (put + get) per iteration per warp
    result->bandwidth_gbps = (total_bytes * 2.0 * iters) /
                             (result->elapsed_ms / 1e3) / 1e9;
    result->primary_metric = (total_warps * iters * 1000.0) / result->elapsed_ms;
    result->metric_name = "putgets/sec";

    cudaFreeHost(d_errors);
    ctx.cleanup();
  }

#ifdef WRP_CORE_ENABLE_BAM
  else if (m == "bam") {
    // ======== BaM: warp-cooperative page cache (GIDS-style) ========

    uint64_t fb_aligned = ((total_bytes + cfg.bam_page_size - 1) / cfg.bam_page_size) * cfg.bam_page_size;
    uint32_t total_pages = (uint32_t)(fb_aligned / cfg.bam_page_size);
    uint64_t hbm_bytes = cfg.GetHbmBytes();
    uint32_t cache_pages = (hbm_bytes > 0)
        ? std::max(1u, (uint32_t)(hbm_bytes / cfg.bam_page_size))
        : total_pages;

    bam::PageCacheConfig pcfg;
    pcfg.page_size = cfg.bam_page_size;
    pcfg.num_pages = cache_pages;
    pcfg.num_queues = 0;
    pcfg.queue_depth = 0;
    pcfg.backend = bam::BackendType::kHostMemory;
    pcfg.nvme_dev = nullptr;

    bam::PageCache cache(pcfg);
    bam::Array<char> write_array(total_bytes, cache);
    bam::Array<char> read_array(total_bytes, cache);

    // Initialize arrays
    std::vector<char> init_data(total_bytes, 0);
    write_array.load_from_host(init_data.data(), init_data.size());
    read_array.load_from_host(init_data.data(), init_data.size());

    HIPRINT("  BaM HBM cache: {} / {} pages ({}%) x {} B = {:.1f} MB",
            cache_pages, total_pages,
            cache_pages * 100 / std::max(1u, total_pages), cfg.bam_page_size,
            (double)cache_pages * cfg.bam_page_size / (1024.0 * 1024.0));

    int *d_errors;
    cudaMallocHost(&d_errors, sizeof(int));
    *d_errors = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    PrintKernelInfo("synthetic_bam_kernel",
                    (const void *)synthetic_bam_kernel,
                    cfg.client_blocks, cfg.client_threads);
    synthetic_bam_kernel<<<cfg.client_blocks, cfg.client_threads>>>(
        write_array.device(), read_array.device(),
        warp_bytes, total_warps, iters, cfg.validate, d_errors);

    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cfg.validate && *d_errors > 0) {
      HLOG(kWarning, "synthetic BaM validation: {} errors", *d_errors);
    }

    result->bandwidth_gbps = (total_bytes * 2.0 * iters) /
                             (result->elapsed_ms / 1e3) / 1e9;
    result->primary_metric = (total_warps * iters * 1000.0) / result->elapsed_ms;
    result->metric_name = "putgets/sec";

    cudaFreeHost(d_errors);
  }
#endif

  else if (m == "hbm") {
    // ======== HBM: device memcpy only ========

    char *d_write, *d_read;
    cudaMalloc(&d_write, total_bytes);
    cudaMalloc(&d_read, total_bytes);
    cudaMemset(d_write, 0, total_bytes);
    cudaMemset(d_read, 0, total_bytes);

    int *d_errors;
    cudaMallocHost(&d_errors, sizeof(int));
    *d_errors = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    PrintKernelInfo("synthetic_hbm_kernel",
                    (const void *)synthetic_hbm_kernel,
                    cfg.client_blocks, cfg.client_threads);
    synthetic_hbm_kernel<<<cfg.client_blocks, cfg.client_threads>>>(
        d_write, d_read, warp_bytes, total_warps, iters, cfg.validate, d_errors);

    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cfg.validate && *d_errors > 0) {
      HLOG(kWarning, "synthetic HBM validation: {} errors", *d_errors);
    }

    result->bandwidth_gbps = (total_bytes * 2.0 * iters) /
                             (result->elapsed_ms / 1e3) / 1e9;
    result->primary_metric = (total_warps * iters * 1000.0) / result->elapsed_ms;
    result->metric_name = "putgets/sec";

    cudaFreeHost(d_errors);
    cudaFree(d_write);
    cudaFree(d_read);
  }

  else if (m == "direct") {
    // ======== Direct: pinned DRAM memcpy ========

    char *h_write, *h_read;
    cudaMallocHost(&h_write, total_bytes);
    cudaMallocHost(&h_read, total_bytes);
    memset(h_write, 0, total_bytes);
    memset(h_read, 0, total_bytes);

    int *d_errors;
    cudaMallocHost(&d_errors, sizeof(int));
    *d_errors = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    PrintKernelInfo("synthetic_direct_kernel",
                    (const void *)synthetic_direct_kernel,
                    cfg.client_blocks, cfg.client_threads);
    synthetic_direct_kernel<<<cfg.client_blocks, cfg.client_threads>>>(
        h_write, h_read, warp_bytes, total_warps, iters, cfg.validate, d_errors);

    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    result->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cfg.validate && *d_errors > 0) {
      HLOG(kWarning, "synthetic Direct validation: {} errors", *d_errors);
    }

    result->bandwidth_gbps = (total_bytes * 2.0 * iters) /
                             (result->elapsed_ms / 1e3) / 1e9;
    result->primary_metric = (total_warps * iters * 1000.0) / result->elapsed_ms;
    result->metric_name = "putgets/sec";

    cudaFreeHost(d_errors);
    cudaFreeHost(h_write);
    cudaFreeHost(h_read);
  }

  else {
    HLOG(kError, "synthetic: unknown mode '{}'", mode);
    return -1;
  }

  return 0;
}

#endif  // HSHM_IS_HOST
