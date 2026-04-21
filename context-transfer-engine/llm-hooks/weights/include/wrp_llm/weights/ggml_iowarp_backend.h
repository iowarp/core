#pragma once

#include "wrp_llm/weights/weight_manager.h"

// ggml backend abstraction headers (part of llama.cpp)
#include "ggml-backend.h"

// Forward-declare ggml_context so callers don't need to include ggml.h
struct ggml_context;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// IOWarp ggml backend
//
// Implements a custom ggml_backend_buffer_type that allocates weight tensors
// inside a GpuVirtualMemoryManager (GpuVmm) address space instead of using
// plain CUDA cudaMalloc.
//
// Benefits over the standard CUDA backend:
//   - The total virtual address space is 512 GB, so a model that does not
//     fit in physical HBM can still be mapped.
//   - Pages that are not currently in use are evicted to CPU DRAM (or NVMe
//     SSD via CTE), reducing peak GPU memory pressure.
//   - The WeightManager pipelines layer prefetch with compute, so the GPU
//     is rarely stalled waiting for weights to arrive.
//
// Usage (in llama-model.cpp, inside load_tensors()):
//
//   // Create and initialise once per model load:
//   static auto* weight_mgr = new wrp_llm::weights::WeightManager();
//   weight_mgr->Init();
//
//   // Obtain the buffer type and pass it to llama.cpp's allocator:
//   ggml_backend_buffer_type_t buft =
//       ggml_backend_iowarp_buffer_type(weight_mgr);
//
//   // Then use buft wherever a gpu buft_list entry would appear:
//   //   ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
// ---------------------------------------------------------------------------

/**
 * Return the ggml_backend_buffer_type backed by the supplied WeightManager.
 * The returned pointer is valid for the lifetime of weight_mgr.
 */
ggml_backend_buffer_type_t ggml_backend_iowarp_buffer_type(
    wrp_llm::weights::WeightManager* weight_mgr);

/**
 * Return a ggml_backend that can execute graphs whose tensors live in the
 * IOWarp buffer type above.  For now this is a thin wrapper: graph execution
 * is delegated to the standard CUDA backend, but tensor storage and paging
 * go through the WeightManager.
 */
ggml_backend_t ggml_backend_iowarp_init(
    wrp_llm::weights::WeightManager* weight_mgr);

/**
 * Auto-register layer page ranges from already-loaded model tensors.
 *
 * Call this once after load_all_data() / set_tensor() has been called for all
 * model weight tensors.  The function scans every tensor whose data pointer
 * falls inside the WeightManager's GpuVmm virtual-address range, extracts the
 * layer index from the tensor name (pattern "blk.N." → layer N), and calls
 * WeightManager::RegisterLayer() with the corresponding page range.
 *
 * @param weight_mgr  Initialised WeightManager (Init() must have been called).
 * @param ctx         The ggml_context that owns the model weight tensors.
 */
void ggml_backend_iowarp_auto_register_layers(
    wrp_llm::weights::WeightManager * weight_mgr,
    struct ggml_context              * ctx);

/** Name string returned by the backend for logging. */
#define GGML_BACKEND_IOWARP_NAME "IOWarp-GpuVmm"

#ifdef __cplusplus
}
#endif
