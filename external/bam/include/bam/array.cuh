/**
 * bam/array.cuh -- bam::Array<T> for transparent storage-backed GPU arrays
 *
 * Memory hierarchy:
 *   GPU HBM (VRAM) ← page cache (fast, limited)
 *       ↕ cache miss / writeback
 *   Host DRAM       ← backing store (pinned cudaMallocHost)
 *       ↕ (optional, future)
 *   NVMe SSD        ← persistent storage
 *
 * Usage from a GPU kernel:
 *
 *   __global__ void my_kernel(bam::ArrayDevice<float> arr) {
 *       float val = arr.read(threadIdx.x);
 *       arr.write(threadIdx.x, val * 2.0f);
 *   }
 *
 * Host-side setup:
 *
 *   bam::Array<float> arr(num_elements, cache);
 *   my_kernel<<<grid, block>>>(arr.device());
 */
#ifndef BAM_ARRAY_CUH
#define BAM_ARRAY_CUH

#include <bam/types.h>
#include <bam/page_cache.cuh>
#include <cuda_runtime.h>
#include <cstddef>

namespace bam {

/* ------------------------------------------------------------------ */
/* Device-side array accessor                                          */
/* ------------------------------------------------------------------ */

template <typename T>
struct ArrayDevice {
  PageCacheDeviceState cache_state;
  QueuePairDevice      qp;           // NVMe queue (unused if DRAM fallback)
  uint64_t            *buf_bus_addrs; // Per-cache-page bus addresses (NVMe)
  const uint8_t       *host_base;    // Pinned DRAM base (host-memory mode)
  uint64_t             num_elements;
  uint64_t             total_bytes;
  BackendType          backend;

  /**
   * Read element at logical index.
   *
   * 1. Compute page-aligned offset and in-page offset.
   * 2. Acquire cache page (may be a hit or miss).
   * 3. On miss: load entire page from DRAM into HBM cache.
   * 4. Read the element from the HBM cache page.
   */
  __device__ T read(uint64_t idx) const {
    uint64_t byte_off = idx * sizeof(T);
    uint64_t page_off = byte_off & ~((uint64_t)cache_state.page_size - 1);
    uint32_t in_page  = (uint32_t)(byte_off & ((uint64_t)cache_state.page_size - 1));

    bool needs_load;
    uint8_t *page = page_cache_acquire(
        const_cast<PageCacheDeviceState &>(cache_state), page_off, &needs_load);

    if (needs_load) {
      if (backend == BackendType::kNvme) {
        uint32_t slot = (uint32_t)((page_off >> cache_state.page_shift) % cache_state.num_pages);
        nvme_read_page(const_cast<QueuePairDevice &>(qp),
                       buf_bus_addrs[slot], page_off, cache_state.page_size);
      } else {
        host_read_page(page, host_base, page_off, cache_state.page_size);
      }
      page_cache_finish_load(
          const_cast<PageCacheDeviceState &>(cache_state), page_off);
    }

    T val = *reinterpret_cast<const T *>(page + in_page);
    return val;
  }

  /**
   * Write element at logical index.
   *
   * Same as read but also writes back and marks dirty.
   */
  __device__ void write(uint64_t idx, T val) {
    uint64_t byte_off = idx * sizeof(T);
    uint64_t page_off = byte_off & ~((uint64_t)cache_state.page_size - 1);
    uint32_t in_page  = (uint32_t)(byte_off & ((uint64_t)cache_state.page_size - 1));

    bool needs_load;
    uint8_t *page = page_cache_acquire(
        const_cast<PageCacheDeviceState &>(cache_state), page_off, &needs_load);

    if (needs_load) {
      // Read-before-write for partial-page updates
      if (backend == BackendType::kNvme) {
        uint32_t slot = (uint32_t)((page_off >> cache_state.page_shift) % cache_state.num_pages);
        nvme_read_page(const_cast<QueuePairDevice &>(qp),
                       buf_bus_addrs[slot], page_off, cache_state.page_size);
      } else {
        host_read_page(page, host_base, page_off, cache_state.page_size);
      }
      page_cache_finish_load(
          const_cast<PageCacheDeviceState &>(cache_state), page_off);
    }

    *reinterpret_cast<T *>(page + in_page) = val;

    // Write through to DRAM backing store for durability
    if (backend == BackendType::kHostMemory) {
      host_write_page(page, const_cast<uint8_t *>(host_base),
                      page_off, cache_state.page_size);
    }

    page_cache_mark_dirty(
        const_cast<PageCacheDeviceState &>(cache_state), page_off);
  }
};

/* ------------------------------------------------------------------ */
/* Host-side array manager                                             */
/* ------------------------------------------------------------------ */

class PageCache;  // Forward declaration

template <typename T>
class Array {
 public:
  Array(uint64_t num_elements, PageCache &cache);
  ~Array();

  ArrayDevice<T> device() const { return dev_; }
  uint64_t total_bytes() const { return num_elements_ * sizeof(T); }
  uint64_t size() const { return num_elements_; }

  /**
   * Pre-populate the DRAM backing store with host data.
   */
  int load_from_host(const T *host_data, uint64_t count);

 private:
  uint64_t num_elements_;
  ArrayDevice<T> dev_;
  PageCache *cache_;
};

}  // namespace bam

#endif  // BAM_ARRAY_CUH
