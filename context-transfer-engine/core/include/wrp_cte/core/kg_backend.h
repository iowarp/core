#ifndef WRPCTE_KG_BACKEND_H_
#define WRPCTE_KG_BACKEND_H_

#include <wrp_cte/core/core_tasks.h>
#include <hermes_shm/data_structures/priv/knowledge_graph/knowledge_graph.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace wrp_cte::core {

/// Result from a backend search: tag identifier + confidence score
using KGSearchResult = hshm::priv::SearchResult<TagId>;

/**
 * Abstract interface for knowledge graph backends.
 * Each backend implements Add/Remove/Search over (TagId, text) pairs.
 * The CTE runtime calls these through the factory-created instance.
 */
class KGBackend {
 public:
  virtual ~KGBackend() = default;

  /** Human-readable name (e.g. "bm25", "elasticsearch", "neo4j") */
  virtual std::string Name() const = 0;

  /** Initialize the backend with a config string (YAML/JSON from compose).
   *  Called once at CTE runtime startup. */
  virtual void Init(const std::string &config) = 0;

  /** Shut down connections, release resources. */
  virtual void Destroy() = 0;

  /** Add or update an entry. Idempotent on the same tag_id. */
  virtual void Add(const TagId &tag_id, const std::string &text) = 0;

  /** Remove an entry by tag_id. */
  virtual void Remove(const TagId &tag_id) = 0;

  /** Search for entries matching a natural-language query.
   *  Returns up to top_k results sorted by descending confidence score. */
  virtual std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) = 0;

  /** Number of entries currently stored. */
  virtual size_t Size() const = 0;

  /** Remove all entries. */
  virtual void Clear() = 0;

  // --- Distributed IDF sync (optional, only BM25 uses these) ---

  /** Set global IDF statistics collected from all nodes.
   *  When set, Search() uses global N and df for IDF calculation. */
  virtual void SetGlobalIdf(
      size_t global_n,
      std::unordered_map<std::string, size_t> global_df,
      float global_avg_dl) {}

  /** Get the number of documents in this local partition. */
  virtual size_t GetLocalN() const { return 0; }

  /** Get the total term count across all local documents. */
  virtual size_t GetLocalTotalTerms() const { return 0; }

  /** Get per-term document frequency for this local partition. */
  virtual std::unordered_map<std::string, size_t> GetLocalDf() const {
    return {};
  }
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_H_
