#ifndef WRPCTE_KG_BACKEND_BM25_H_
#define WRPCTE_KG_BACKEND_BM25_H_

#include <wrp_cte/core/kg_backend.h>
#include <hermes_shm/data_structures/priv/knowledge_graph/knowledge_graph.h>

namespace wrp_cte::core {

/**
 * BM25 backend — wraps the existing KnowledgeGraph implementation.
 * Zero external dependencies. In-memory inverted index with BM25 scoring.
 * Supports distributed IDF sync for partitioned KG across nodes.
 */
class BM25Backend : public KGBackend {
 public:
  std::string Name() const override { return "bm25"; }

  void Init(const std::string &config) override {
    // BM25 needs no initialization — the inverted index is built on Add()
    (void)config;
  }

  void Destroy() override {
    kg_.Clear();
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    kg_.Add(tag_id, text);
  }

  void Remove(const TagId &tag_id) override {
    kg_.Remove(tag_id);
  }

  std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) override {
    return kg_.Search(query, top_k);
  }

  size_t Size() const override {
    return kg_.GetLocalN();
  }

  void Clear() override {
    kg_.Clear();
  }

  // --- Distributed IDF sync ---

  void SetGlobalIdf(
      size_t global_n,
      std::unordered_map<std::string, size_t> global_df,
      float global_avg_dl) override {
    kg_.SetGlobalIdf(global_n, std::move(global_df), global_avg_dl);
  }

  size_t GetLocalN() const override {
    return kg_.GetLocalN();
  }

  size_t GetLocalTotalTerms() const override {
    return kg_.GetLocalTotalTerms();
  }

  std::unordered_map<std::string, size_t> GetLocalDf() const override {
    return kg_.GetLocalDf();
  }

 private:
  hshm::priv::KnowledgeGraph<TagId, hshm::hash<TagId>> kg_;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_BM25_H_
