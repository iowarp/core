#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// IOWarp GPU Virtual Memory Manager
#include "wrp_cte/uvm/gpu_vmm.h"

namespace wrp_llm {
namespace weights {

/**
 * IOWarp Weight Manager for llama.cpp (FlexGen-style offloading)
 *
 * Uses IOWarp's GpuVirtualMemoryManager to create a single large CUDA
 * virtual address space (512 GB by default) backed by:
 *   1. GPU HBM     — pages that are currently mapped / in active use
 *   2. CPU DRAM    — pages evicted from GPU (GpuVmm host backing store)
 *   3. NVMe SSD    — pages evicted from CPU DRAM (via CTE when use_cte=true)
 *
 * During inference, layers are processed sequentially.  WeightManager
 * pre-loads the next layer's pages while the current layer computes,
 * then evicts the previous layer's pages — providing an overlapped
 * compute/transfer pipeline analogous to FlexGen.
 *
 * Integration with llama.cpp:
 *   - The ggml_iowarp_backend (see ggml_iowarp_backend.h) allocates model
 *     weight tensors inside this virtual address space.
 *   - Before each transformer layer, PrepareLayer(layer_idx) is called to
 *     guarantee all pages are mapped.
 *   - After each layer, ReleaseLayer(layer_idx) evicts pages back to host.
 */
class WeightManager {
 public:
  struct Config {
    /** Total CUDA virtual address space reserved (bytes). 512 GB default. */
    size_t va_size_bytes = 512ULL * 1024 * 1024 * 1024;

    /** CUDA virtual memory page granularity (bytes). 2 MB default. */
    size_t page_size = 2ULL * 1024 * 1024;

    /**
     * Number of layers to prefetch ahead.
     * E.g. prefetch_window=2 means while layer N executes, layers N+1
     * and N+2 are being loaded asynchronously.
     */
    size_t prefetch_window = 2;

    /** CUDA device ordinal. */
    int device = 0;

    /**
     * If true, the GpuVmm backs evicted CPU pages into the IOWarp CTE
     * (enabling SSD overflow for very large models).
     * If false, backing store is plain host RAM only.
     */
    bool use_cte = true;

    /** CTE tag name used by GpuVmm when use_cte=true. */
    std::string cte_tag_name = "llama_weights";
  };

  WeightManager();
  explicit WeightManager(const Config& cfg);
  ~WeightManager();

  WeightManager(const WeightManager&) = delete;
  WeightManager& operator=(const WeightManager&) = delete;

  /**
   * Initialize the GpuVmm and (optionally) connect to CTE.
   * Must be called before RegisterLayer() or PrepareLayer().
   */
  bool Init();

  /** Release all GPU virtual memory and CTE resources. */
  void Shutdown();

  /**
   * Register the page range for a model layer.
   * Called once per layer during model loading, after all tensors for
   * that layer have been placed at virtual addresses inside the Vmm.
   *
   * @param layer_idx   Zero-based transformer layer index.
   * @param page_start  First page index occupied by this layer's tensors.
   * @param page_count  Number of pages the layer occupies.
   */
  void RegisterLayer(int layer_idx, size_t page_start, size_t page_count);

  /**
   * Ensure all pages for layer_idx are physically mapped on the GPU.
   * Blocks until the pages are ready.
   * Also kicks off async prefetch for the next `prefetch_window` layers.
   *
   * Call this just before llama.cpp's graph executor computes layer_idx.
   */
  void PrepareLayer(int layer_idx);

  /**
   * Evict layer_idx's pages back to host (or CTE if use_cte=true).
   * Non-blocking: launches async D2H transfers and returns immediately.
   *
   * Call this after the layer's computation is complete.
   */
  void ReleaseLayer(int layer_idx);

  /**
   * Returns the base virtual address of the Vmm allocation.
   * Tensors are placed at offsets within this range by the ggml backend.
   */
  void* BaseAddress() const;

  /** Return true if Init() completed successfully. */
  bool IsReady() const;

  /** Size of each page in bytes. */
  size_t PageSize() const;

  /** GpuVmm instance (direct access for the ggml backend). */
  wrp_cte::uvm::GpuVirtualMemoryManager& Vmm();

 private:
  struct LayerRange {
    int layer_idx;
    size_t page_start;
    size_t page_count;
  };

  Config cfg_;
  wrp_cte::uvm::GpuVirtualMemoryManager vmm_;
  std::vector<LayerRange> layer_ranges_;
  bool ready_ = false;
};

}  // namespace weights
}  // namespace wrp_llm
