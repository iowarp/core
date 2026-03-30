#ifndef WRPCTE_KG_BACKEND_ZEP_H_
#define WRPCTE_KG_BACKEND_ZEP_H_

#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace wrp_cte::core {

/**
 * Zep backend — open-source memory system for AI agents.
 * Uses Zep's REST API for document storage and semantic search.
 * Requires a running Zep server (with postgres + nlp + optional LLM).
 *
 * Docker: docker compose -f zep-docker-compose.yml up
 * Docs: https://docs.getzep.com
 */
class ZepBackend : public KGBackend {
 public:
  std::string Name() const override { return "zep"; }

  void Init(const std::string &config) override {
    endpoint_ = "localhost";
    port_ = 8198;
    collection_ = "cte_kg_bench";
    api_key_ = "cte_bench_test_secret";

    if (!config.empty()) {
      // Parse "host:port" or "host:port/collection"
      auto slash = config.find('/');
      std::string host_port = (slash != std::string::npos)
          ? config.substr(0, slash) : config;
      if (slash != std::string::npos) {
        collection_ = config.substr(slash + 1);
      }
      auto colon = host_port.find(':');
      if (colon != std::string::npos) {
        endpoint_ = host_port.substr(0, colon);
        port_ = std::stoi(host_port.substr(colon + 1));
      } else if (!host_port.empty()) {
        endpoint_ = host_port;
      }
    }

    // Read API key from env if set
    const char *key = std::getenv("ZEP_API_SECRET");
    if (key) api_key_ = key;

    client_ = std::make_unique<httplib::Client>(endpoint_, port_);
    client_->set_connection_timeout(5);
    client_->set_read_timeout(30);

    // Delete collection if exists, then create
    httplib::Headers headers = {{"Authorization", "Bearer " + api_key_}};
    client_->Delete("/api/v2/collections/" + collection_, headers);

    nlohmann::json create_body = {
        {"name", collection_},
        {"description", "CTE KG benchmark collection"},
        {"embedding_dimensions", 384},
        {"is_auto_embedded", true}
    };
    client_->Post("/api/v2/collections", headers,
                  create_body.dump(), "application/json");
    size_ = 0;
  }

  void Destroy() override {
    if (client_) {
      httplib::Headers headers = {{"Authorization", "Bearer " + api_key_}};
      client_->Delete("/api/v2/collections/" + collection_, headers);
      client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    httplib::Headers headers = {{"Authorization", "Bearer " + api_key_}};

    std::string doc_id = std::to_string(tag_id.major_) + "_" +
                         std::to_string(tag_id.minor_);

    nlohmann::json docs = {{
        {"document_id", doc_id},
        {"content", text},
        {"metadata", {
            {"tag_major", tag_id.major_},
            {"tag_minor", tag_id.minor_}
        }}
    }};

    auto res = client_->Post(
        "/api/v2/collections/" + collection_ + "/documents",
        headers, docs.dump(), "application/json");
    if (res && res->status >= 200 && res->status < 300) {
      size_++;
    }
  }

  void Remove(const TagId &tag_id) override {
    httplib::Headers headers = {{"Authorization", "Bearer " + api_key_}};
    std::string doc_id = std::to_string(tag_id.major_) + "_" +
                         std::to_string(tag_id.minor_);
    client_->Delete(
        "/api/v2/collections/" + collection_ + "/documents/" + doc_id,
        headers);
    if (size_ > 0) size_--;
  }

  std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) override {
    std::vector<KGSearchResult> results;
    httplib::Headers headers = {{"Authorization", "Bearer " + api_key_}};

    nlohmann::json search_body = {
        {"text", query},
        {"search_type", "similarity"},
        {"limit", top_k}
    };
    auto res = client_->Post(
        "/api/v2/collections/" + collection_ + "/search",
        headers, search_body.dump(), "application/json");

    if (!res || res->status != 200) return results;

    auto body = nlohmann::json::parse(res->body, nullptr, false);
    if (body.is_discarded() || !body.is_array()) return results;

    for (const auto &hit : body) {
      TagId tid;
      float score = 0;
      try {
        if (hit.contains("metadata")) {
          tid.major_ = hit["metadata"]["tag_major"].get<chi::u32>();
          tid.minor_ = hit["metadata"]["tag_minor"].get<chi::u32>();
        }
        if (hit.contains("score")) {
          score = hit["score"].get<float>();
        }
      } catch (...) { continue; }
      results.push_back({tid, score});
    }
    return results;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    if (client_) {
      httplib::Headers headers = {{"Authorization", "Bearer " + api_key_}};
      client_->Delete("/api/v2/collections/" + collection_, headers);
      nlohmann::json create_body = {
          {"name", collection_},
          {"description", "CTE KG benchmark collection"},
          {"embedding_dimensions", 384},
          {"is_auto_embedded", true}
      };
      client_->Post("/api/v2/collections", headers,
                    create_body.dump(), "application/json");
      size_ = 0;
    }
  }

 private:
  std::string endpoint_;
  int port_;
  std::string collection_;
  std::string api_key_;
  std::unique_ptr<httplib::Client> client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_ZEP_H_
