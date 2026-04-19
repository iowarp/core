#ifndef WRPCTE_EMBEDDING_CLIENT_H_
#define WRPCTE_EMBEDDING_CLIENT_H_

/**
 * EmbeddingClient — shared HTTP client for OpenAI-compatible /embeddings
 * endpoints.
 *
 * Used by backends that need to turn text into a vector (Qdrant, Elasticsearch
 * in vector mode). Also used by Acropolis's L3 indexing-depth executor to
 * generate an embedding from a file's text summary.
 *
 * Environment variables (override any config-supplied endpoint):
 *   CTE_EMBEDDING_ENDPOINT  — full URL, e.g. "http://localhost:8090/v1/embeddings"
 *   CTE_EMBEDDING_MODEL     — model name, e.g. "qwen2.5-3b"
 *
 * Legacy vars still respected for backward compatibility:
 *   QDRANT_EMBEDDING_ENDPOINT / QDRANT_EMBEDDING_MODEL
 */

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <string>
#include <vector>

namespace wrp_cte::core {

class EmbeddingClient {
 public:
  EmbeddingClient() = default;

  /** Configure from an endpoint URL and optional model name. Env vars override. */
  void Configure(const std::string &endpoint, const std::string &model = "") {
    endpoint_ = endpoint;
    model_    = model;

    if (const char *e = std::getenv("CTE_EMBEDDING_ENDPOINT")) endpoint_ = e;
    else if (const char *e2 = std::getenv("QDRANT_EMBEDDING_ENDPOINT")) endpoint_ = e2;

    if (const char *m = std::getenv("CTE_EMBEDDING_MODEL")) model_ = m;
    else if (const char *m2 = std::getenv("QDRANT_EMBEDDING_MODEL")) model_ = m2;
  }

  bool Configured() const { return !endpoint_.empty(); }

  const std::string &Endpoint() const { return endpoint_; }
  const std::string &Model() const { return model_; }

  /**
   * Turn a text into an embedding vector. Returns empty vector on failure
   * (backends/callers should degrade gracefully to non-semantic search).
   */
  std::vector<float> Embed(const std::string &text) const {
    if (endpoint_.empty()) return {};

    std::string host = "localhost";
    int port = 8090;
    std::string path = "/v1/embeddings";

    // Parse "http://host:port/path"
    auto proto_end = endpoint_.find("://");
    std::string rest =
        (proto_end != std::string::npos) ? endpoint_.substr(proto_end + 3)
                                          : endpoint_;
    auto slash = rest.find('/');
    if (slash != std::string::npos) {
      path = rest.substr(slash);
      rest = rest.substr(0, slash);
    }
    auto colon = rest.find(':');
    if (colon != std::string::npos) {
      host = rest.substr(0, colon);
      port = std::stoi(rest.substr(colon + 1));
    } else {
      host = rest;
    }

    httplib::Client cli(host, port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(60);

    nlohmann::json body = {{"model", model_}, {"input", text}};
    auto res = cli.Post(path, body.dump(), "application/json");
    if (!res || res->status != 200) return {};

    auto resp = nlohmann::json::parse(res->body, nullptr, false);
    if (resp.is_discarded()) return {};

    try {
      return resp["data"][0]["embedding"].get<std::vector<float>>();
    } catch (...) {
      return {};
    }
  }

 private:
  std::string endpoint_;
  std::string model_;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_EMBEDDING_CLIENT_H_
