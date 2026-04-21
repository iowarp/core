#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wrp_llm {
namespace kvcache {

/**
 * IOWarp KV Cache Manager for llama.cpp
 *
 * Provides tiered storage for LLM KV cache blocks using IOWarp's
 * Context Transfer Engine (CTE).  Two capabilities:
 *
 *  1. KV Cache Eviction Offload
 *     When llama.cpp evicts a sequence's KV block (via seq_rm), instead
 *     of discarding it we store it in CTE.  Future requests that share the
 *     same prefix can restore the block, skipping expensive recomputation.
 *
 *     Storage tiers (CTE blob score):
 *       GPU HBM  (score ≈ 0.0)  ← active / just evicted
 *       CPU DRAM (score ≈ 0.4)  ← warm  (default eviction target)
 *       NVMe SSD (score ≈ 0.8)  ← cold  (demoted after timeout)
 *
 *  2. Prefix Hash Lookup (MOONCAKE-style)
 *     Before a prefill, call LookupPrefix().  If a cached block exists
 *     for a matching token prefix, the data is restored into the KV
 *     cache buffer and the matched length is returned so llama.cpp can
 *     skip computing those tokens.
 *
 * Usage (from llama.cpp integration):
 *
 *   KVCacheManager mgr;
 *   mgr.Init();   // once at startup
 *
 *   // Before prefill:
 *   size_t matched = 0;
 *   if (mgr.LookupPrefix(tokens, kv_ptr, kv_size, true, matched))
 *       batch_start = matched;  // skip already-cached tokens
 *
 *   // After seq_rm eviction:
 *   mgr.OnEvict(evicted_tokens, kv_ptr, kv_size, true);
 */
class KVCacheManager {
 public:
  struct Config {
    /** CTE tag name for all KV cache blobs. */
    std::string tag_name = "llama_kvcache";

    /** Score for newly evicted blocks → CPU DRAM. */
    float evicted_score = 0.4f;

    /** Score for cold blocks → NVMe SSD (applied after cold_timeout_s). */
    float cold_score = 0.8f;

    /** How long (seconds) a block stays at evicted_score before demotion. */
    double cold_timeout_s = 60.0;

    /**
     * Maximum in-process hash index entries.
     * Once full, the oldest entries are dropped from the index
     * (the data remains in CTE, but prefix lookup won't find them
     * until a full CTE scan is implemented).
     */
    size_t max_index_entries = 65536;
  };

  KVCacheManager();
  explicit KVCacheManager(const Config& cfg);
  ~KVCacheManager();

  // Non-copyable, movable.
  KVCacheManager(const KVCacheManager&) = delete;
  KVCacheManager& operator=(const KVCacheManager&) = delete;
  KVCacheManager(KVCacheManager&&) noexcept;
  KVCacheManager& operator=(KVCacheManager&&) noexcept;

  /**
   * Connect to IOWarp CTE and create/open the KV cache tag.
   * Must be called before any other method.
   * @return true on success.
   */
  bool Init();

  /** Release CTE resources. Safe to call even if Init() was never called. */
  void Shutdown();

  // ---------------------------------------------------------------------------
  // KV Block Storage
  // ---------------------------------------------------------------------------

  /**
   * Store a KV cache block for a given token prefix.
   *
   * @param tokens   Token sequence this block covers (used as cache key).
   * @param kv_data  Pointer to the raw KV data bytes.
   * @param size     Size in bytes of kv_data.
   * @param on_gpu   If true, kv_data is a device pointer; a D2H copy is
   *                 performed internally before handing to CTE.
   * @return true on success.
   */
  bool StoreBlock(const std::vector<int32_t>& tokens,
                  const void* kv_data, size_t size, bool on_gpu = true);

  /**
   * Look up the longest cached prefix matching the supplied token sequence.
   *
   * @param tokens      Full input token sequence.
   * @param dst         Destination buffer to write restored KV data into.
   * @param dst_size    Capacity of dst in bytes.
   * @param on_gpu      If true, dst is a device pointer; an H2D copy is
   *                    performed internally after reading from CTE.
   * @param matched_len OUT: number of tokens whose KV data was restored.
   *                    0 if no prefix was found.
   * @return true if any prefix was matched and data restored.
   */
  bool LookupPrefix(const std::vector<int32_t>& tokens,
                    void* dst, size_t dst_size, bool on_gpu,
                    size_t& matched_len);

  /**
   * Notify that a KV block has been evicted by llama.cpp.
   * Saves kv_data to CTE at evicted_score (CPU DRAM tier).
   * Pass kv_data=nullptr to just remove the entry from the index.
   */
  void OnEvict(const std::vector<int32_t>& tokens,
               const void* kv_data, size_t size, bool on_gpu = true);

  // ---------------------------------------------------------------------------
  // Raw-key API
  //
  // Used by llama_context::iowarp_kvcache_save/restore, which already holds
  // serialised KV bytes and just needs CTE as a key-value store.
  // The key is used directly as the CTE blob name (no hashing applied).
  // ---------------------------------------------------------------------------

  /** Store an opaque blob under an arbitrary string key. */
  bool StoreBlockRaw(const std::string& key, const void* data, size_t size);

  /**
   * Retrieve a blob by its raw string key.
   * Populates out_data on success; returns false if the key is not found.
   */
  bool LookupRaw(const std::string& key, std::vector<uint8_t>& out_data);

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  /**
   * Compute a stable 64-bit FNV-1a hash of a token sequence and
   * return it as a fixed-width hex string (blob name in CTE).
   */
  static std::string HashTokens(const std::vector<int32_t>& tokens);

  /** Return true if Init() completed successfully. */
  bool IsReady() const;

 private:
  struct Impl;
  Config cfg_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvcache
}  // namespace wrp_llm
