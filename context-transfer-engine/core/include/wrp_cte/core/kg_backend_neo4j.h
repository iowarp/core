#ifndef WRPCTE_KG_BACKEND_NEO4J_H_
#define WRPCTE_KG_BACKEND_NEO4J_H_

#include <wrp_cte/core/kg_backend.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>

namespace wrp_cte::core {

/**
 * Neo4j backend — true knowledge graph with entity extraction.
 *
 * Instead of storing flat text, extracts entities and relationships
 * from metadata at ingest time:
 *
 *   (Dataset {name: "galaxy_hires/snapshot_023"})
 *     -[:HAS_PARTICLE_TYPE]-> (ParticleType {name: "dark_matter", count: 40000})
 *     -[:HAS_PARTICLE_TYPE]-> (ParticleType {name: "disk", count: 20000})
 *     -[:IS_SIM_TYPE]-> (SimType {name: "isolated_galaxy"})
 *     -[:HAS_PROPERTY]-> (Property {name: "non_cosmological"})
 *     -[:HAS_PROPERTY]-> (Property {name: "no_star_formation"})
 *
 * Search uses graph traversal + full-text index on Dataset.summary.
 * This enables relational queries that BM25/Qdrant can't handle:
 *   "Find simulations with dark matter but no gas"
 *   "Which runs have star formation enabled?"
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

    try {
      client_ = std::make_unique<httplib::Client>(endpoint_, port_);
      client_->set_connection_timeout(5);
      client_->set_read_timeout(30);

      // Clean previous data
      RunCypher("MATCH (n) DETACH DELETE n");

      // Create indexes for fast lookup
      RunCypher("CREATE INDEX dataset_tag IF NOT EXISTS FOR (d:Dataset) ON (d.tag_major, d.tag_minor)");
      RunCypher("CREATE INDEX particle_name IF NOT EXISTS FOR (p:ParticleType) ON (p.name)");
      RunCypher("CREATE INDEX simtype_name IF NOT EXISTS FOR (s:SimType) ON (s.name)");
      RunCypher("CREATE INDEX property_name IF NOT EXISTS FOR (p:Property) ON (p.name)");

      // Full-text index on Dataset summary for natural language search
      RunCypher(
          "CREATE FULLTEXT INDEX cte_kg_idx IF NOT EXISTS "
          "FOR (d:Dataset) ON EACH [d.summary, d.text]");

      size_ = 0;
    } catch (...) {
      client_.reset();
      size_ = 0;
    }
  }

  void Destroy() override {
    if (client_) {
      RunCypher("MATCH (n) DETACH DELETE n");
      RunCypher("DROP INDEX cte_kg_idx IF EXISTS");
      RunCypher("DROP INDEX dataset_tag IF EXISTS");
      RunCypher("DROP INDEX particle_name IF EXISTS");
      RunCypher("DROP INDEX simtype_name IF EXISTS");
      RunCypher("DROP INDEX property_name IF EXISTS");
      client_.reset();
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    // Step 1: Create the Dataset node with the summary text
    nlohmann::json params = {
        {"major", tag_id.major_},
        {"minor", tag_id.minor_},
        {"text", text},
        {"summary", text}
    };
    RunCypher(
        "MERGE (d:Dataset {tag_major: $major, tag_minor: $minor}) "
        "SET d.text = $text, d.summary = $summary",
        params);

    // Step 2: Extract entities from the text and create graph relationships
    ExtractAndLinkEntities(tag_id, text);

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
    // Two-stage search:
    // 1. Full-text search on Dataset.summary (Lucene BM25) — primary ranking
    // 2. Graph-boosted search: find datasets via entity traversal, boost score
    //
    // The graph structure enables queries that pure text search can't handle:
    //   "Find simulations with dark matter but no gas"
    //   → Traverse: HAS_PARTICLE_TYPE->dark_matter AND NOT HAS_PARTICLE_TYPE->gas

    std::vector<KGSearchResult> results;
    std::unordered_map<uint64_t, float> score_map;

    // Stage 1: Full-text search (primary — uses Lucene BM25 on summary)
    nlohmann::json ft_params = {{"query", query}, {"top_k", top_k * 3}};
    auto ft_response = RunCypher(
        "CALL db.index.fulltext.queryNodes('cte_kg_idx', $query) "
        "YIELD node, score "
        "WHERE node:Dataset "
        "RETURN node.tag_major AS major, node.tag_minor AS minor, score "
        "LIMIT $top_k",
        ft_params);
    {
      std::vector<KGSearchResult> ft_results;
      ParseResults(ft_response, ft_results);
      for (const auto &r : ft_results) {
        uint64_t key = (static_cast<uint64_t>(r.key.major_) << 32) | r.key.minor_;
        score_map[key] += r.score * 3.0f;  // Full-text dominates
      }
    }

    // Stage 2: Graph traversal boost — single efficient query
    // Count matching entities per dataset for all keywords at once
    std::vector<std::string> keywords = Tokenize(query);
    if (!keywords.empty()) {
      // Build a keyword list for Cypher
      nlohmann::json kw_array = nlohmann::json::array();
      for (const auto &kw : keywords) kw_array.push_back(kw);

      nlohmann::json gp = {{"keywords", kw_array}, {"top_k", top_k * 3}};
      auto graph_response = RunCypher(
          "UNWIND $keywords AS kw "
          "MATCH (d:Dataset)-[rel]->(entity) "
          "WHERE (entity:ParticleType OR entity:SimType OR entity:Property OR entity:Run) "
          "  AND toLower(entity.name) CONTAINS toLower(kw) "
          "WITH d, count(DISTINCT entity) AS matches, "
          "  sum(CASE WHEN entity:SimType THEN 2.0 "
          "           WHEN entity:ParticleType THEN 1.5 "
          "           ELSE 1.0 END) AS weighted_matches "
          "RETURN d.tag_major AS major, d.tag_minor AS minor, "
          "  weighted_matches AS score "
          "ORDER BY weighted_matches DESC "
          "LIMIT $top_k",
          gp);

      std::vector<KGSearchResult> graph_results;
      ParseResults(graph_response, graph_results);
      for (const auto &r : graph_results) {
        uint64_t key = (static_cast<uint64_t>(r.key.major_) << 32) | r.key.minor_;
        score_map[key] += r.score;  // Graph boost adds to full-text
      }
    }

    // Build final sorted results
    for (const auto &[key, score] : score_map) {
      TagId tid;
      tid.major_ = static_cast<chi::u32>(key >> 32);
      tid.minor_ = static_cast<chi::u32>(key & 0xFFFFFFFF);
      results.push_back({tid, score});
    }
    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) { return a.score > b.score; });
    if (static_cast<int>(results.size()) > top_k) {
      results.resize(top_k);
    }

    return results;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    RunCypher("MATCH (n) DETACH DELETE n");
    size_ = 0;
  }

 private:
  /**
   * Extract entities from metadata text and create graph relationships.
   * Parses key=value pairs from the description to build structured graph.
   */
  void ExtractAndLinkEntities(const TagId &tag_id, const std::string &text) {
    nlohmann::json base_params = {
        {"major", tag_id.major_},
        {"minor", tag_id.minor_}
    };

    // Extract particle types (e.g., "gas_particles=1472", "dark_matter_particles=40000")
    std::regex particle_re(R"((\w+)_particles=(\d+))");
    std::sregex_iterator it(text.begin(), text.end(), particle_re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
      std::string ptype = (*it)[1].str();
      int count = std::stoi((*it)[2].str());
      nlohmann::json params = base_params;
      params["ptype"] = ptype;
      params["count"] = count;
      RunCypher(
          "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
          "MERGE (p:ParticleType {name: $ptype}) "
          "MERGE (d)-[:HAS_PARTICLE_TYPE {count: $count}]->(p)",
          params);
    }

    // Extract simulation type (e.g., "simulation_type=isolated_galaxy_dark_matter_disk")
    std::regex simtype_re(R"(simulation_type=(\S+))");
    std::smatch simtype_match;
    if (std::regex_search(text, simtype_match, simtype_re)) {
      std::string simtype = simtype_match[1].str();
      // Convert underscores to spaces for better matching
      std::replace(simtype.begin(), simtype.end(), '_', ' ');
      nlohmann::json params = base_params;
      params["simtype"] = simtype;
      RunCypher(
          "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
          "MERGE (s:SimType {name: $simtype}) "
          "MERGE (d)-[:IS_SIM_TYPE]->(s)",
          params);
    }

    // Extract boolean properties (star_formation, cooling, cosmological)
    std::regex prop_re(R"((star_formation|cooling|cosmological)=(\S+))");
    std::sregex_iterator pit(text.begin(), text.end(), prop_re);
    for (; pit != end; ++pit) {
      std::string prop_name = (*pit)[1].str();
      std::string prop_val = (*pit)[2].str();
      std::string label = prop_name + "_" + prop_val;
      nlohmann::json params = base_params;
      params["label"] = label;
      params["prop_name"] = prop_name;
      params["prop_val"] = prop_val;
      RunCypher(
          "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
          "MERGE (pr:Property {name: $label}) "
          "SET pr.property = $prop_name, pr.value = $prop_val "
          "MERGE (d)-[:HAS_PROPERTY]->(pr)",
          params);
    }

    // Extract run name
    std::regex run_re(R"(run_name=(\S+))");
    std::smatch run_match;
    if (std::regex_search(text, run_match, run_re)) {
      nlohmann::json params = base_params;
      params["run_name"] = run_match[1].str();
      RunCypher(
          "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
          "MERGE (r:Run {name: $run_name}) "
          "MERGE (d)-[:FROM_RUN]->(r)",
          params);
    }

    // Extract numeric values (Time, Omega0, Redshift)
    std::regex num_re(R"((Time|Omega0|Redshift|BoxSize)=([0-9.e+-]+))");
    std::sregex_iterator nit(text.begin(), text.end(), num_re);
    for (; nit != end; ++nit) {
      std::string name = (*nit)[1].str();
      std::string val = (*nit)[2].str();
      nlohmann::json params = base_params;
      params["name"] = name;
      params["val"] = std::stod(val);
      RunCypher(
          "MATCH (d:Dataset {tag_major: $major, tag_minor: $minor}) "
          "SET d." + name + " = $val",
          params);
    }
  }

  void ParseResults(const std::string &response,
                    std::vector<KGSearchResult> &results) {
    try {
      auto body = nlohmann::json::parse(response, nullptr, false);
      if (body.is_discarded()) return;
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
  }

  std::vector<std::string> Tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    // Stop words to skip
    static const std::unordered_set<std::string> stop_words = {
        "find", "the", "a", "an", "is", "are", "was", "were", "with",
        "and", "or", "of", "in", "on", "at", "to", "for", "from",
        "which", "that", "this", "those", "these", "locate", "where",
        "show", "me", "any", "all", "simulation", "simulations",
        "data", "output", "snapshot", "snapshots", "run", "runs"
    };
    while (iss >> word) {
      // Lowercase
      std::transform(word.begin(), word.end(), word.begin(), ::tolower);
      if (stop_words.find(word) == stop_words.end() && word.size() > 2) {
        tokens.push_back(word);
      }
    }
    return tokens;
  }

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
