#ifndef WRPCTE_KG_BACKEND_QDRANT_H_
#define WRPCTE_KG_BACKEND_QDRANT_H_

#include <wrp_cte/core/embedding_client.h>
#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace wrp_cte::core {

/**
 * Qdrant backend — vector database using embedding-based semantic search.
 * Requires an embedding endpoint (OpenAI-compatible) to convert text to vectors.
 *
 * Config string format:
 *   "host:port/collection [embedding_endpoint]"
 *
 * Env var overrides (see EmbeddingClient):
 *   CTE_EMBEDDING_ENDPOINT, CTE_EMBEDDING_MODEL (preferred)
 *   QDRANT_EMBEDDING_ENDPOINT, QDRANT_EMBEDDING_MODEL (legacy)
 */
class QdrantBackend : public KGBackend {
 public:
  std::string Name() const override { return "qdrant"; }

  void Init(const std::string &config) override {
    qdrant_host_ = "localhost";
    qdrant_port_ = 6333;
    collection_ = "cte_kg_bench";
    embedding_dim_ = 384;

    std::string emb_endpoint;
    // Parse config: "host:port/collection embedding_endpoint"
    if (!config.empty()) {
      auto space = config.find(' ');
      std::string qdrant_part =
          (space != std::string::npos) ? config.substr(0, space) : config;
      if (space != std::string::npos) {
        emb_endpoint = config.substr(space + 1);
      }

      auto colon = qdrant_part.find(':');
      if (colon != std::string::npos) {
        qdrant_host_ = qdrant_part.substr(0, colon);
        std::string rest = qdrant_part.substr(colon + 1);
        auto slash = rest.find('/');
        if (slash != std::string::npos) {
          qdrant_port_ = std::stoi(rest.substr(0, slash));
          collection_ = rest.substr(slash + 1);
        } else {
          qdrant_port_ = std::stoi(rest);
        }
      }
    }

    embedder_.Configure(emb_endpoint);

    try {
      qdrant_client_ =
          std::make_unique<httplib::Client>(qdrant_host_, qdrant_port_);
      qdrant_client_->set_connection_timeout(5);
      qdrant_client_->set_read_timeout(30);

      // Auto-detect embedding dimension with a test embedding
      if (embedder_.Configured()) {
        auto test_emb = embedder_.Embed("test");
        if (!test_emb.empty()) {
          embedding_dim_ = static_cast<int>(test_emb.size());
        }
      }

      // Check if collection exists; create only if not
      auto check = qdrant_client_->Get("/collections/" + collection_);
      if (!check || check->status == 404 ||
          check->body.find("\"status\":\"ok\"") == std::string::npos) {
        nlohmann::json create_body = {
            {"vectors", {{"size", embedding_dim_}, {"distance", "Cosine"}}}};
        qdrant_client_->Put("/collections/" + collection_, create_body.dump(),
                            "application/json");
      }
    } catch (...) {
      qdrant_client_.reset();
    }
    size_ = 0;
  }

  void Destroy() override {
    if (qdrant_client_) {
      qdrant_client_->Delete("/collections/" + collection_);
      qdrant_client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    auto embedding = embedder_.Embed(text);
    if (embedding.empty()) return;

    uint64_t point_id =
        (static_cast<uint64_t>(tag_id.major_) << 32) | tag_id.minor_;

    nlohmann::json body = {
        {"points",
         {{{"id", point_id},
           {"vector", embedding},
           {"payload",
            {{"tag_major", tag_id.major_},
             {"tag_minor", tag_id.minor_},
             {"text", text}}}}}}};
    qdrant_client_->Put("/collections/" + collection_ + "/points", body.dump(),
                        "application/json");
    size_++;
  }

  void Remove(const TagId &tag_id) override {
    uint64_t point_id =
        (static_cast<uint64_t>(tag_id.major_) << 32) | tag_id.minor_;
    nlohmann::json body = {{"points", {point_id}}};
    qdrant_client_->Post("/collections/" + collection_ + "/points/delete",
                         body.dump(), "application/json");
    if (size_ > 0) size_--;
  }

  std::vector<KGSearchResult> Search(const std::string &query,
                                     int top_k) override {
    std::vector<KGSearchResult> results;
    auto embedding = embedder_.Embed(query);
    if (embedding.empty()) return results;

    nlohmann::json body = {
        {"vector", embedding}, {"limit", top_k}, {"with_payload", true}};
    auto res = qdrant_client_->Post(
        "/collections/" + collection_ + "/points/search", body.dump(),
        "application/json");

    if (!res || res->status != 200) return results;

    auto resp = nlohmann::json::parse(res->body, nullptr, false);
    if (resp.is_discarded()) return results;

    for (const auto &hit : resp["result"]) {
      TagId tid;
      tid.major_ = hit["payload"]["tag_major"].get<chi::u32>();
      tid.minor_ = hit["payload"]["tag_minor"].get<chi::u32>();
      float score = hit["score"].get<float>();
      results.push_back({tid, score});
    }
    return results;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    if (qdrant_client_) {
      qdrant_client_->Delete("/collections/" + collection_);
      nlohmann::json create_body = {
          {"vectors", {{"size", embedding_dim_}, {"distance", "Cosine"}}}};
      qdrant_client_->Put("/collections/" + collection_, create_body.dump(),
                          "application/json");
      size_ = 0;
    }
  }

 private:
  std::string qdrant_host_;
  int qdrant_port_;
  std::string collection_;
  int embedding_dim_;
  EmbeddingClient embedder_;
  std::unique_ptr<httplib::Client> qdrant_client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_QDRANT_H_
