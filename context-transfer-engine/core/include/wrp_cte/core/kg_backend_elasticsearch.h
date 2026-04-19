#ifndef WRPCTE_KG_BACKEND_ELASTICSEARCH_H_
#define WRPCTE_KG_BACKEND_ELASTICSEARCH_H_

#include <wrp_cte/core/embedding_client.h>
#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace wrp_cte::core {

/**
 * Elasticsearch backend — full-text (BM25) and/or dense-vector (kNN) search.
 *
 * Elasticsearch 8.4+ supports dense_vector fields with HNSW indexing and a
 * top-level `knn` query that returns nearest neighbours by cosine similarity.
 * This backend can operate in three modes:
 *
 *   keyword  - classic BM25 `match` query over the `text` field only.
 *   vector   - kNN on a `dense_vector` embedding generated from the text.
 *   both     - writes both fields; the runtime chooses which query path to use
 *              (for now each Search() issues both and returns vector-mode
 *              results with keyword-mode as a fallback when no embedder is
 *              available).
 *
 * Config string format:
 *   "host:port/index_name [mode] [embedding_endpoint]"
 *   examples:
 *     "localhost:9200/cte_bench"
 *     "localhost:9200/cte_bench vector"
 *     "localhost:9200/cte_bench both http://localhost:8090/v1/embeddings"
 *
 * Env var overrides (shared with other backends):
 *   CTE_EMBEDDING_ENDPOINT, CTE_EMBEDDING_MODEL
 */
class ElasticsearchBackend : public KGBackend {
 public:
  enum class Mode { kKeyword, kVector, kBoth };

  std::string Name() const override { return "elasticsearch"; }

  void Init(const std::string &config) override {
    endpoint_ = "localhost";
    port_ = 9200;
    index_ = "cte_kg_bench";
    mode_ = Mode::kKeyword;
    embedding_dim_ = 384;

    std::string emb_endpoint;

    // Parse config: "host:port/index [mode] [embedding_endpoint]"
    if (!config.empty()) {
      std::istringstream iss(config);
      std::string address;
      iss >> address;
      auto slash = address.find('/');
      std::string host_port =
          (slash != std::string::npos) ? address.substr(0, slash) : address;
      if (slash != std::string::npos) index_ = address.substr(slash + 1);
      auto colon = host_port.find(':');
      if (colon != std::string::npos) {
        endpoint_ = host_port.substr(0, colon);
        port_ = std::stoi(host_port.substr(colon + 1));
      } else if (!host_port.empty()) {
        endpoint_ = host_port;
      }

      std::string tok;
      if (iss >> tok) {
        if (tok == "vector") mode_ = Mode::kVector;
        else if (tok == "both") mode_ = Mode::kBoth;
        else if (tok == "keyword") mode_ = Mode::kKeyword;
        else emb_endpoint = tok;  // unknown token: treat as endpoint
      }
      if (iss >> tok) emb_endpoint = tok;
    }

    embedder_.Configure(emb_endpoint);

    try {
      client_ = std::make_unique<httplib::Client>(endpoint_, port_);
      client_->set_connection_timeout(5);
      client_->set_read_timeout(10);

      // Auto-detect embedding dimension from the configured endpoint.
      if (UseVector() && embedder_.Configured()) {
        auto test = embedder_.Embed("test");
        if (!test.empty()) embedding_dim_ = static_cast<int>(test.size());
      }

      auto check = client_->Get("/" + index_);
      if (!check || check->status == 404) {
        client_->Put("/" + index_, BuildMapping().dump(), "application/json");
      }
      size_ = 0;
    } catch (...) {
      client_.reset();
      size_ = 0;
    }
  }

  void Destroy() override {
    if (client_) {
      client_->Delete("/" + index_);
      client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    if (!client_) return;
    std::string doc_id =
        std::to_string(tag_id.major_) + "_" + std::to_string(tag_id.minor_);
    nlohmann::json doc = {{"text", text},
                          {"tag_major", tag_id.major_},
                          {"tag_minor", tag_id.minor_}};

    // Include embedding when the mode uses vectors and we can generate one.
    if (UseVector() && embedder_.Configured()) {
      auto emb = embedder_.Embed(text);
      if (!emb.empty()) {
        doc["embedding"] = emb;
      }
    }

    auto res = client_->Put("/" + index_ + "/_doc/" + doc_id, doc.dump(),
                            "application/json");
    if (res && res->status >= 200 && res->status < 300) size_++;
  }

  void Remove(const TagId &tag_id) override {
    if (!client_) return;
    std::string doc_id =
        std::to_string(tag_id.major_) + "_" + std::to_string(tag_id.minor_);
    auto res = client_->Delete("/" + index_ + "/_doc/" + doc_id);
    if (res && (res->status == 200 || res->status == 404)) {
      if (size_ > 0) size_--;
    }
  }

  std::vector<KGSearchResult> Search(const std::string &query,
                                     int top_k) override {
    std::vector<KGSearchResult> results;
    if (!client_) return results;

    // Make recent writes searchable.
    client_->Post("/" + index_ + "/_refresh", "", "application/json");

    // Prefer vector search when configured.
    if (UseVector() && embedder_.Configured()) {
      auto q_emb = embedder_.Embed(query);
      if (!q_emb.empty()) {
        nlohmann::json body = {
            {"knn",
             {{"field", "embedding"},
              {"query_vector", q_emb},
              {"k", top_k},
              {"num_candidates", std::max(top_k * 10, 50)}}},
            {"size", top_k},
            {"_source", nlohmann::json::array({"tag_major", "tag_minor"})}};
        auto res = client_->Post("/" + index_ + "/_search", body.dump(),
                                 "application/json");
        if (res && res->status == 200) {
          auto parsed = nlohmann::json::parse(res->body, nullptr, false);
          if (!parsed.is_discarded()) {
            for (const auto &hit : parsed["hits"]["hits"]) {
              TagId tid;
              tid.major_ = hit["_source"]["tag_major"].get<chi::u32>();
              tid.minor_ = hit["_source"]["tag_minor"].get<chi::u32>();
              float score = hit["_score"].get<float>();
              results.push_back({tid, score});
            }
            if (!results.empty() || mode_ == Mode::kVector) return results;
          }
        }
      }
      if (mode_ == Mode::kVector) return results;  // strict vector mode
    }

    // Keyword (BM25) path — used in kKeyword or as fallback in kBoth.
    nlohmann::json body = {{"query", {{"match", {{"text", query}}}}},
                           {"size", top_k}};
    auto res = client_->Post("/" + index_ + "/_search", body.dump(),
                             "application/json");
    if (!res || res->status != 200) return results;

    auto parsed = nlohmann::json::parse(res->body, nullptr, false);
    if (parsed.is_discarded()) return results;

    for (const auto &hit : parsed["hits"]["hits"]) {
      TagId tid;
      tid.major_ = hit["_source"]["tag_major"].get<chi::u32>();
      tid.minor_ = hit["_source"]["tag_minor"].get<chi::u32>();
      float score = hit["_score"].get<float>();
      results.push_back({tid, score});
    }
    return results;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    if (!client_) return;
    client_->Delete("/" + index_);
    client_->Put("/" + index_, BuildMapping().dump(), "application/json");
    size_ = 0;
  }

 private:
  bool UseVector() const { return mode_ == Mode::kVector || mode_ == Mode::kBoth; }

  nlohmann::json BuildMapping() const {
    nlohmann::json properties = {
        {"text", {{"type", "text"}, {"analyzer", "standard"}}},
        {"tag_major", {{"type", "long"}}},
        {"tag_minor", {{"type", "long"}}}};
    if (UseVector()) {
      properties["embedding"] = {{"type", "dense_vector"},
                                 {"dims", embedding_dim_},
                                 {"index", true},
                                 {"similarity", "cosine"}};
    }
    return nlohmann::json{
        {"settings", {{"number_of_shards", 1}, {"number_of_replicas", 0}}},
        {"mappings", {{"properties", properties}}}};
  }

  std::string endpoint_;
  int port_;
  std::string index_;
  Mode mode_;
  int embedding_dim_;
  EmbeddingClient embedder_;
  std::unique_ptr<httplib::Client> client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_ELASTICSEARCH_H_
