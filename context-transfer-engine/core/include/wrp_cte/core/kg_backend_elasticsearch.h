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
#include <unordered_map>
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
 *   both     - runs both queries in parallel and fuses with Reciprocal Rank
 *              Fusion (k=60). A single-list score boost on exact-phrase
 *              matches (via the `text.keyword` sub-field) is added to each
 *              BM25 result.
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

    // Over-fetch so RRF has enough candidates to work with.
    const int fetch_k = std::max(top_k * 3, 10);

    std::vector<KGSearchResult> vec_hits;
    std::vector<KGSearchResult> kw_hits;

    if (mode_ == Mode::kVector || mode_ == Mode::kBoth) {
      if (embedder_.Configured()) {
        auto q_emb = embedder_.Embed(query);
        if (!q_emb.empty()) {
          nlohmann::json body = {
              {"knn",
               {{"field", "embedding"},
                {"query_vector", q_emb},
                {"k", fetch_k},
                {"num_candidates", std::max(fetch_k * 4, 50)}}},
              {"size", fetch_k},
              {"_source", nlohmann::json::array({"tag_major", "tag_minor"})}};
          auto res = client_->Post("/" + index_ + "/_search", body.dump(),
                                   "application/json");
          ParseHits(res, vec_hits);
        }
      }
      if (mode_ == Mode::kVector) {
        if (static_cast<int>(vec_hits.size()) > top_k) vec_hits.resize(top_k);
        return vec_hits;
      }
    }

    if (mode_ == Mode::kKeyword || mode_ == Mode::kBoth) {
      // Use a bool-should that combines the analyzed text field with the
      // `text.keyword` sub-field. `text.keyword` gives exact-string hits a
      // boost without affecting the BM25 tokenization path.
      nlohmann::json body = {
          {"query",
           {{"bool",
             {{"should", nlohmann::json::array(
                            {{{"match", {{"text", query}}}},
                             {{"term", {{"text.keyword",
                                          {{"value", query},
                                           {"boost", 2.0}}}}}}})}}}}},
          {"size", fetch_k},
          {"_source", nlohmann::json::array({"tag_major", "tag_minor"})}};
      auto res = client_->Post("/" + index_ + "/_search", body.dump(),
                               "application/json");
      ParseHits(res, kw_hits);

      if (mode_ == Mode::kKeyword) {
        if (static_cast<int>(kw_hits.size()) > top_k) kw_hits.resize(top_k);
        return kw_hits;
      }
    }

    // Mode::kBoth — fuse via Reciprocal Rank Fusion (k=60).
    std::unordered_map<uint64_t, float> rrf;
    auto accumulate = [&rrf](const std::vector<KGSearchResult> &lst) {
      constexpr float kRrfK = 60.0f;
      for (size_t i = 0; i < lst.size(); ++i) {
        uint64_t key = (static_cast<uint64_t>(lst[i].key.major_) << 32) |
                       lst[i].key.minor_;
        rrf[key] += 1.0f / (kRrfK + static_cast<float>(i + 1));
      }
    };
    accumulate(vec_hits);
    accumulate(kw_hits);

    results.reserve(rrf.size());
    for (const auto &kv : rrf) {
      TagId tid;
      tid.major_ = static_cast<chi::u32>(kv.first >> 32);
      tid.minor_ = static_cast<chi::u32>(kv.first & 0xFFFFFFFFu);
      results.push_back({tid, kv.second});
    }
    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) { return a.score > b.score; });
    if (static_cast<int>(results.size()) > top_k) results.resize(top_k);
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
  bool UseVector() const {
    return mode_ == Mode::kVector || mode_ == Mode::kBoth;
  }

  void ParseHits(const httplib::Result &res,
                 std::vector<KGSearchResult> &out) {
    if (!res || res->status != 200) return;
    auto parsed = nlohmann::json::parse(res->body, nullptr, false);
    if (parsed.is_discarded()) return;
    if (!parsed.contains("hits") || !parsed["hits"].contains("hits")) return;
    for (const auto &hit : parsed["hits"]["hits"]) {
      TagId tid;
      tid.major_ = hit["_source"]["tag_major"].get<chi::u32>();
      tid.minor_ = hit["_source"]["tag_minor"].get<chi::u32>();
      float score = hit["_score"].get<float>();
      out.push_back({tid, score});
    }
  }

  nlohmann::json BuildMapping() const {
    // Pattern analyzer splits on any non-alphanumeric run (including
    // underscore/slash/dot/dash). Matches the BM25 backend's tokenization —
    // so "kg_backend_bm25" tokenizes to [kg, backend, bm25] and a query for
    // "bm25" correctly finds it. A `keyword` sub-field preserves the raw
    // string for exact-match boosting.
    nlohmann::json properties = {
        {"text",
         {{"type", "text"},
          {"analyzer", "acropolis_analyzer"},
          {"fields",
           {{"keyword",
             {{"type", "keyword"},
              {"ignore_above", 8192}}}}}}},
        {"tag_major", {{"type", "long"}}},
        {"tag_minor", {{"type", "long"}}}};
    if (UseVector()) {
      properties["embedding"] = {{"type", "dense_vector"},
                                 {"dims", embedding_dim_},
                                 {"index", true},
                                 {"similarity", "cosine"}};
    }
    return nlohmann::json{
        {"settings",
         {{"number_of_shards", 1},
          {"number_of_replicas", 0},
          {"analysis",
           {{"analyzer",
             {{"acropolis_analyzer",
               {{"type", "pattern"},
                {"pattern", "[^a-zA-Z0-9]+"},
                {"lowercase", true}}}}}}}}},
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
