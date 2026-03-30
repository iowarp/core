#ifndef WRPCTE_KG_BACKEND_ELASTICSEARCH_H_
#define WRPCTE_KG_BACKEND_ELASTICSEARCH_H_

#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace wrp_cte::core {

/**
 * Elasticsearch backend — uses ES's built-in BM25 for text search.
 * Fairest comparison to CTE's native BM25 (same algorithm, different transport).
 */
class ElasticsearchBackend : public KGBackend {
 public:
  std::string Name() const override { return "elasticsearch"; }

  void Init(const std::string &config) override {
    // Parse config: "host:port/index_name" or use defaults
    endpoint_ = "localhost";
    port_ = 9200;
    index_ = "cte_kg_bench";

    if (!config.empty()) {
      // Simple parsing: "host:port/index"
      auto slash = config.find('/');
      std::string host_port = (slash != std::string::npos)
          ? config.substr(0, slash) : config;
      if (slash != std::string::npos) {
        index_ = config.substr(slash + 1);
      }
      auto colon = host_port.find(':');
      if (colon != std::string::npos) {
        endpoint_ = host_port.substr(0, colon);
        port_ = std::stoi(host_port.substr(colon + 1));
      } else if (!host_port.empty()) {
        endpoint_ = host_port;
      }
    }

    client_ = std::make_unique<httplib::Client>(endpoint_, port_);
    client_->set_connection_timeout(5);
    client_->set_read_timeout(10);

    // Delete index if exists, then create fresh
    client_->Delete("/" + index_);

    nlohmann::json settings = {
        {"settings", {{"number_of_shards", 1}, {"number_of_replicas", 0}}},
        {"mappings", {{"properties", {
            {"text", {{"type", "text"}, {"analyzer", "standard"}}},
            {"tag_major", {{"type", "long"}}},
            {"tag_minor", {{"type", "long"}}}
        }}}}
    };
    client_->Put("/" + index_, settings.dump(), "application/json");
    size_ = 0;
  }

  void Destroy() override {
    if (client_) {
      client_->Delete("/" + index_);
      client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    std::string doc_id = std::to_string(tag_id.major_) + "_" +
                         std::to_string(tag_id.minor_);
    nlohmann::json doc = {
        {"text", text},
        {"tag_major", tag_id.major_},
        {"tag_minor", tag_id.minor_}
    };
    auto res = client_->Put("/" + index_ + "/_doc/" + doc_id,
                            doc.dump(), "application/json");
    if (res && res->status >= 200 && res->status < 300) {
      size_++;
    }
  }

  void Remove(const TagId &tag_id) override {
    std::string doc_id = std::to_string(tag_id.major_) + "_" +
                         std::to_string(tag_id.minor_);
    auto res = client_->Delete("/" + index_ + "/_doc/" + doc_id);
    if (res && (res->status == 200 || res->status == 404)) {
      if (size_ > 0) size_--;
    }
  }

  std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) override {
    // Refresh index to make recent writes searchable
    client_->Post("/" + index_ + "/_refresh", "", "application/json");

    nlohmann::json search_body = {
        {"query", {{"match", {{"text", query}}}}},
        {"size", top_k}
    };
    auto res = client_->Post("/" + index_ + "/_search",
                             search_body.dump(), "application/json");

    std::vector<KGSearchResult> results;
    if (!res || res->status != 200) return results;

    auto body = nlohmann::json::parse(res->body, nullptr, false);
    if (body.is_discarded()) return results;

    for (const auto &hit : body["hits"]["hits"]) {
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
    if (client_) {
      client_->Delete("/" + index_);
      // Recreate empty index
      nlohmann::json settings = {
          {"settings", {{"number_of_shards", 1}, {"number_of_replicas", 0}}},
          {"mappings", {{"properties", {
              {"text", {{"type", "text"}, {"analyzer", "standard"}}},
              {"tag_major", {{"type", "long"}}},
              {"tag_minor", {{"type", "long"}}}
          }}}}
      };
      client_->Put("/" + index_, settings.dump(), "application/json");
      size_ = 0;
    }
  }

 private:
  std::string endpoint_;
  int port_;
  std::string index_;
  std::unique_ptr<httplib::Client> client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_ELASTICSEARCH_H_
