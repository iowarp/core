#ifndef WRPCTE_KG_BACKEND_NEO4J_H_
#define WRPCTE_KG_BACKEND_NEO4J_H_

#include <wrp_cte/core/embedding_client.h>
#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace wrp_cte::core {

/**
 * Neo4j backend — native fulltext index + optional native vector index.
 *
 * Maps directly onto Neo4j 5.x built-ins:
 *   (:Dataset {tag_major, tag_minor, text, embedding})
 *   FULLTEXT INDEX cte_kg_idx  ON d.text       (configurable Lucene analyzer)
 *   VECTOR   INDEX cte_kg_vec  ON d.embedding  (cosine, created when embedder
 *                                               is configured)
 *
 * Config string:
 *   "host:port [analyzer=<name>] [http://embedding-endpoint]"
 *
 * Analyzer choice matters: Lucene's default `standard` keeps
 * underscore-joined identifiers as single tokens, so a query for "bm25"
 * will not match a doc containing "kg_backend_bm25". The default here is
 * `simple`, which splits on any non-letter.
 *
 * Search path:
 *   - If an embedder is configured, run fulltext + vector and fuse with
 *     Reciprocal Rank Fusion (k=60).
 *   - Otherwise run fulltext only.
 *
 * Domain-specific entity extraction (e.g. `(\w+)_particles=N`) is explicitly
 * NOT done here — it was dragging recall down on generic text. A caller that
 * wants a graph overlay can run its own extractor on top.
 */
class Neo4jBackend : public KGBackend {
 public:
  std::string Name() const override { return "neo4j"; }

  void Init(const std::string &config) override {
    endpoint_ = "localhost";
    port_ = 7474;
    analyzer_ = "simple";
    embedding_dim_ = 384;

    std::string emb_endpoint;

    if (!config.empty()) {
      std::istringstream iss(config);
      std::string tok;
      bool first = true;
      while (iss >> tok) {
        if (first) {
          auto colon = tok.find(':');
          if (colon != std::string::npos) {
            endpoint_ = tok.substr(0, colon);
            port_ = std::stoi(tok.substr(colon + 1));
          } else {
            endpoint_ = tok;
          }
          first = false;
          continue;
        }
        if (tok.rfind("analyzer=", 0) == 0) {
          analyzer_ = tok.substr(9);
        } else {
          emb_endpoint = tok;
        }
      }
    }

    embedder_.Configure(emb_endpoint);

    try {
      client_ = std::make_unique<httplib::Client>(endpoint_, port_);
      client_->set_connection_timeout(5);
      client_->set_read_timeout(30);

      if (embedder_.Configured()) {
        auto test = embedder_.Embed("test");
        if (!test.empty()) embedding_dim_ = static_cast<int>(test.size());
      }

      RunCypher("MATCH (n:Dataset) DETACH DELETE n");

      RunCypher(
          "CREATE INDEX dataset_tag IF NOT EXISTS "
          "FOR (d:Dataset) ON (d.tag_major, d.tag_minor)");

      {
        nlohmann::json p = {{"analyzer", analyzer_}};
        RunCypher(
            "CREATE FULLTEXT INDEX cte_kg_idx IF NOT EXISTS "
            "FOR (d:Dataset) ON EACH [d.text] "
            "OPTIONS { indexConfig: { `fulltext.analyzer`: $analyzer } }",
            p);
      }

      if (embedder_.Configured()) {
        nlohmann::json p = {{"dims", embedding_dim_}};
        RunCypher(
            "CREATE VECTOR INDEX cte_kg_vec IF NOT EXISTS "
            "FOR (d:Dataset) ON (d.embedding) "
            "OPTIONS { indexConfig: { "
            "  `vector.dimensions`: toInteger($dims), "
            "  `vector.similarity_function`: 'cosine' } }",
            p);
      }

      size_ = 0;
    } catch (...) {
      client_.reset();
      size_ = 0;
    }
  }

  void Destroy() override {
    if (client_) {
      RunCypher("MATCH (n:Dataset) DETACH DELETE n");
      RunCypher("DROP INDEX cte_kg_idx IF EXISTS");
      RunCypher("DROP INDEX cte_kg_vec IF EXISTS");
      RunCypher("DROP INDEX dataset_tag IF EXISTS");
      client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    if (!client_) return;
    nlohmann::json params = {
        {"major", tag_id.major_},
        {"minor", tag_id.minor_},
        {"text", text}};

    if (embedder_.Configured()) {
      auto emb = embedder_.Embed(text);
      if (!emb.empty()) {
        params["embedding"] = emb;
        RunCypher(
            "MERGE (d:Dataset {tag_major: $major, tag_minor: $minor}) "
            "SET d.text = $text, d.embedding = $embedding",
            params);
        size_++;
        return;
      }
    }
    RunCypher(
        "MERGE (d:Dataset {tag_major: $major, tag_minor: $minor}) "
        "SET d.text = $text",
        params);
    size_++;
  }

  void Remove(const TagId &tag_id) override {
    if (!client_) return;
    nlohmann::json params = {
        {"major", tag_id.major_}, {"minor", tag_id.minor_}};
    RunCypher(
        "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
        "DETACH DELETE d",
        params);
    if (size_ > 0) size_--;
  }

  std::vector<KGSearchResult> Search(const std::string &query,
                                     int top_k) override {
    std::vector<KGSearchResult> out;
    if (!client_) return out;

    std::vector<KGSearchResult> ft;
    {
      std::string lq = SanitizeLucene(query);
      if (!lq.empty()) {
        nlohmann::json p = {{"q", lq}, {"k", top_k * 3}};
        auto resp = RunCypher(
            "CALL db.index.fulltext.queryNodes('cte_kg_idx', $q) "
            "YIELD node, score "
            "WHERE node:Dataset "
            "RETURN node.tag_major AS major, node.tag_minor AS minor, score "
            "LIMIT toInteger($k)",
            p);
        ParseResults(resp, ft);
      }
    }

    std::vector<KGSearchResult> vec;
    if (embedder_.Configured()) {
      auto qv = embedder_.Embed(query);
      if (!qv.empty()) {
        nlohmann::json p = {{"k", top_k * 3}, {"vec", qv}};
        auto resp = RunCypher(
            "CALL db.index.vector.queryNodes("
            "  'cte_kg_vec', toInteger($k), $vec) "
            "YIELD node, score "
            "WHERE node:Dataset "
            "RETURN node.tag_major AS major, node.tag_minor AS minor, score",
            p);
        ParseResults(resp, vec);
      }
    }

    // Reciprocal Rank Fusion (k=60) across the two ranked lists.
    std::unordered_map<uint64_t, float> rrf;
    auto accumulate = [&rrf](const std::vector<KGSearchResult> &lst) {
      constexpr float kRrfK = 60.0f;
      for (size_t i = 0; i < lst.size(); ++i) {
        uint64_t key = (static_cast<uint64_t>(lst[i].key.major_) << 32) |
                       lst[i].key.minor_;
        rrf[key] += 1.0f / (kRrfK + static_cast<float>(i + 1));
      }
    };
    accumulate(ft);
    accumulate(vec);

    out.reserve(rrf.size());
    for (const auto &kv : rrf) {
      TagId tid;
      tid.major_ = static_cast<chi::u32>(kv.first >> 32);
      tid.minor_ = static_cast<chi::u32>(kv.first & 0xFFFFFFFFu);
      out.push_back({tid, kv.second});
    }
    std::sort(out.begin(), out.end(),
              [](const auto &a, const auto &b) { return a.score > b.score; });
    if (static_cast<int>(out.size()) > top_k) out.resize(top_k);
    return out;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    if (!client_) return;
    RunCypher("MATCH (n:Dataset) DETACH DELETE n");
    size_ = 0;
  }

 private:
  void ParseResults(const std::string &response,
                    std::vector<KGSearchResult> &results) {
    auto body = nlohmann::json::parse(response, nullptr, false);
    if (body.is_discarded()) return;
    if (!body.contains("results")) return;
    for (const auto &result : body["results"]) {
      if (!result.contains("data")) continue;
      for (const auto &row : result["data"]) {
        const auto &r = row["row"];
        if (!r.is_array() || r.size() < 3) continue;
        TagId tid;
        tid.major_ = r[0].get<chi::u32>();
        tid.minor_ = r[1].get<chi::u32>();
        float score = r[2].get<float>();
        results.push_back({tid, score});
      }
    }
  }

  // Strip Lucene reserved characters so raw user input can't break the parser.
  // Keeps alphanumerics, underscores, whitespace; drops the rest.
  static std::string SanitizeLucene(const std::string &in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
      if (std::isalnum(static_cast<unsigned char>(c)) ||
          c == '_' || c == ' ' || c == '\t') {
        out.push_back(c);
      } else {
        out.push_back(' ');
      }
    }
    return out;
  }

  std::string RunCypher(const std::string &cypher,
                        const nlohmann::json &params = nlohmann::json()) {
    if (!client_) return "{}";
    nlohmann::json stmt = {{"statement", cypher}};
    if (!params.is_null()) stmt["parameters"] = params;
    nlohmann::json body = {{"statements", nlohmann::json::array({stmt})}};
    auto res = client_->Post("/db/neo4j/tx/commit",
                             body.dump(), "application/json");
    if (res && res->status == 200) return res->body;
    return "{}";
  }

  std::string endpoint_;
  int port_;
  std::string analyzer_;
  int embedding_dim_;
  EmbeddingClient embedder_;
  std::unique_ptr<httplib::Client> client_;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_NEO4J_H_
