#ifndef WRPCTE_KG_BACKEND_QDRANT_H_
#define WRPCTE_KG_BACKEND_QDRANT_H_

#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <functional>

namespace wrp_cte::core {

/**
 * Qdrant backend — vector database using embedding-based semantic search.
 * Requires an embedding endpoint (OpenAI-compatible) to convert text to vectors.
 * Measures the cost of semantic precision vs keyword matching.
 *
 * Set env vars:
 *   QDRANT_EMBEDDING_ENDPOINT=http://localhost:8081/v1
 *   QDRANT_EMBEDDING_MODEL=qwen2.5-3b
 */
class QdrantBackend : public KGBackend {
 public:
  std::string Name() const override { return "qdrant"; }

  void Init(const std::string &config) override {
    qdrant_host_ = "localhost";
    qdrant_port_ = 6333;
    collection_ = "cte_kg_bench";
    embedding_dim_ = 1024;

    // Parse qdrant host:port from config
    if (!config.empty()) {
      auto colon = config.find(':');
      if (colon != std::string::npos) {
        qdrant_host_ = config.substr(0, colon);
        qdrant_port_ = std::stoi(config.substr(colon + 1));
      }
    }

    // Embedding endpoint from env
    const char *emb_endpoint = std::getenv("QDRANT_EMBEDDING_ENDPOINT");
    const char *emb_model = std::getenv("QDRANT_EMBEDDING_MODEL");
    if (emb_endpoint) embedding_endpoint_ = emb_endpoint;
    if (emb_model) embedding_model_ = emb_model;

    qdrant_client_ = std::make_unique<httplib::Client>(qdrant_host_, qdrant_port_);
    qdrant_client_->set_connection_timeout(5);
    qdrant_client_->set_read_timeout(30);

    // Auto-detect embedding dimension with a test embedding
    if (!embedding_endpoint_.empty()) {
      auto test_emb = GetEmbedding("test");
      if (!test_emb.empty()) {
        embedding_dim_ = static_cast<int>(test_emb.size());
      }
    }

    // Delete collection if exists, create fresh
    qdrant_client_->Delete("/collections/" + collection_);
    nlohmann::json create_body = {
        {"vectors", {{"size", embedding_dim_}, {"distance", "Cosine"}}}
    };
    qdrant_client_->Put("/collections/" + collection_,
                        create_body.dump(), "application/json");
    size_ = 0;
  }

  void Destroy() override {
    if (qdrant_client_) {
      qdrant_client_->Delete("/collections/" + collection_);
      qdrant_client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    auto embedding = GetEmbedding(text);
    if (embedding.empty()) return;

    // Use a deterministic point ID from tag_id
    uint64_t point_id = (static_cast<uint64_t>(tag_id.major_) << 32) |
                        tag_id.minor_;

    nlohmann::json body = {
        {"points", {{
            {"id", point_id},
            {"vector", embedding},
            {"payload", {
                {"tag_major", tag_id.major_},
                {"tag_minor", tag_id.minor_},
                {"text", text}
            }}
        }}}
    };
    qdrant_client_->Put("/collections/" + collection_ + "/points",
                        body.dump(), "application/json");
    size_++;
  }

  void Remove(const TagId &tag_id) override {
    uint64_t point_id = (static_cast<uint64_t>(tag_id.major_) << 32) |
                        tag_id.minor_;
    nlohmann::json body = {{"points", {point_id}}};
    qdrant_client_->Post("/collections/" + collection_ + "/points/delete",
                         body.dump(), "application/json");
    if (size_ > 0) size_--;
  }

  std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) override {
    std::vector<KGSearchResult> results;

    auto embedding = GetEmbedding(query);
    if (embedding.empty()) return results;

    nlohmann::json body = {
        {"vector", embedding},
        {"limit", top_k},
        {"with_payload", true}
    };
    auto res = qdrant_client_->Post(
        "/collections/" + collection_ + "/points/search",
        body.dump(), "application/json");

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
          {"vectors", {{"size", embedding_dim_}, {"distance", "Cosine"}}}
      };
      qdrant_client_->Put("/collections/" + collection_,
                          create_body.dump(), "application/json");
      size_ = 0;
    }
  }

 private:
  std::vector<float> GetEmbedding(const std::string &text) {
    if (embedding_endpoint_.empty()) return {};

    // Parse host:port from endpoint URL
    // e.g., "http://localhost:8081/v1"
    std::string host = "localhost";
    int port = 8081;
    std::string path = "/v1/embeddings";

    auto proto_end = embedding_endpoint_.find("://");
    std::string rest = (proto_end != std::string::npos)
        ? embedding_endpoint_.substr(proto_end + 3)
        : embedding_endpoint_;
    auto slash = rest.find('/');
    if (slash != std::string::npos) {
      path = rest.substr(slash) + "/embeddings";
      rest = rest.substr(0, slash);
    }
    auto colon = rest.find(':');
    if (colon != std::string::npos) {
      host = rest.substr(0, colon);
      port = std::stoi(rest.substr(colon + 1));
    } else {
      host = rest;
    }

    httplib::Client emb_client(host, port);
    emb_client.set_connection_timeout(10);
    emb_client.set_read_timeout(60);

    nlohmann::json body = {
        {"model", embedding_model_},
        {"input", text}
    };
    auto res = emb_client.Post(path, body.dump(), "application/json");
    if (!res || res->status != 200) return {};

    auto resp = nlohmann::json::parse(res->body, nullptr, false);
    if (resp.is_discarded()) return {};

    try {
      return resp["data"][0]["embedding"].get<std::vector<float>>();
    } catch (...) {
      return {};
    }
  }

  std::string qdrant_host_;
  int qdrant_port_;
  std::string collection_;
  int embedding_dim_;
  std::string embedding_endpoint_;
  std::string embedding_model_;
  std::unique_ptr<httplib::Client> qdrant_client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_QDRANT_H_
