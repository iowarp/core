#ifndef WRPCTE_KG_BACKEND_NEO4J_H_
#define WRPCTE_KG_BACKEND_NEO4J_H_

#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace wrp_cte::core {

/**
 * Neo4j backend — true graph database with Lucene full-text search.
 * Uses HTTP REST API (transactional endpoint).
 */
class Neo4jBackend : public KGBackend {
 public:
  std::string Name() const override { return "neo4j"; }

  void Init(const std::string &config) override {
    endpoint_ = "localhost";
    port_ = 7474;

    if (!config.empty()) {
      auto colon = config.find(':');
      if (colon != std::string::npos) {
        endpoint_ = config.substr(0, colon);
        port_ = std::stoi(config.substr(colon + 1));
      } else {
        endpoint_ = config;
      }
    }

    client_ = std::make_unique<httplib::Client>(endpoint_, port_);
    client_->set_connection_timeout(5);
    client_->set_read_timeout(30);

    // Clear existing data and create full-text index
    RunCypher("MATCH (d:Dataset) DETACH DELETE d");
    RunCypher("DROP INDEX cte_kg_idx IF EXISTS");
    RunCypher(
        "CREATE FULLTEXT INDEX cte_kg_idx IF NOT EXISTS "
        "FOR (d:Dataset) ON EACH [d.text]");
    size_ = 0;
  }

  void Destroy() override {
    if (client_) {
      RunCypher("MATCH (d:Dataset) DETACH DELETE d");
      RunCypher("DROP INDEX cte_kg_idx IF EXISTS");
      client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    nlohmann::json params = {
        {"major", tag_id.major_},
        {"minor", tag_id.minor_},
        {"text", text}
    };
    RunCypher(
        "MERGE (d:Dataset {tag_major: $major, tag_minor: $minor}) "
        "SET d.text = $text",
        params);
    size_++;
  }

  void Remove(const TagId &tag_id) override {
    nlohmann::json params = {
        {"major", tag_id.major_},
        {"minor", tag_id.minor_}
    };
    RunCypher(
        "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
        "DETACH DELETE d",
        params);
    if (size_ > 0) size_--;
  }

  std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) override {
    nlohmann::json params = {
        {"query", query},
        {"top_k", top_k}
    };
    auto response = RunCypher(
        "CALL db.index.fulltext.queryNodes('cte_kg_idx', $query) "
        "YIELD node, score "
        "RETURN node.tag_major AS major, node.tag_minor AS minor, score "
        "LIMIT $top_k",
        params);

    std::vector<KGSearchResult> results;
    try {
      auto body = nlohmann::json::parse(response, nullptr, false);
      if (body.is_discarded()) return results;

      for (const auto &result : body["results"]) {
        for (const auto &row : result["data"]) {
          TagId tid;
          tid.major_ = row["row"][0].get<chi::u32>();
          tid.minor_ = row["row"][1].get<chi::u32>();
          float score = row["row"][2].get<float>();
          results.push_back({tid, score});
        }
      }
    } catch (...) {}
    return results;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    RunCypher("MATCH (d:Dataset) DETACH DELETE d");
    size_ = 0;
  }

 private:
  std::string RunCypher(const std::string &cypher,
                        const nlohmann::json &params = {}) {
    nlohmann::json body = {
        {"statements", {{
            {"statement", cypher},
            {"parameters", params}
        }}}
    };
    auto res = client_->Post("/db/neo4j/tx/commit",
                             body.dump(), "application/json");
    if (res && res->status == 200) {
      return res->body;
    }
    return "{}";
  }

  std::string endpoint_;
  int port_;
  std::unique_ptr<httplib::Client> client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_NEO4J_H_
