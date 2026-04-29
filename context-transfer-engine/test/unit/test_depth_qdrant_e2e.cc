/*
 * Acropolis — E2E: DepthController + CTE runtime + Qdrant backend.
 *
 * Boots an embedded Chimaera + CTE runtime via CHI_SERVER_CONF pointing at a
 * compose YAML that configures the Qdrant backend + embedding endpoint, sets
 * an xattr on a file to force L1 (metadata + embedding), sends
 * UpdateKnowledgeGraph, and verifies that SemanticQuery returns the tag with
 * a cosine-similarity score.
 *
 * Requires:
 *   - Qdrant reachable at $QDRANT_URL (default host.docker.internal:6333/<col>)
 *   - Mock OpenAI-compatible embedding server at $EMB_URL
 */

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_runtime.h>
#include <wrp_cte/core/core_tasks.h>

#include <sys/xattr.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "simple_test.h"

namespace fs = std::filesystem;
using namespace wrp_cte::core;

namespace {

// Write a compose YAML that points CTE at Qdrant and an embedding endpoint.
// Returns the path to the written file.
std::string WriteComposeYaml() {
  const char *qurl = std::getenv("QDRANT_URL");
  const char *emb  = std::getenv("EMB_URL");
  std::string qdrant =
      qurl ? qurl : "host.docker.internal:6333/acropolis_runtime_e2e";
  std::string embed = emb ? emb : "http://127.0.0.1:9999/v1/embeddings";

  std::string path = "/tmp/acropolis_runtime_compose.yaml";
  std::ofstream f(path);
  f << "runtime:\n"
       "  num_threads: 4\n"
       "  queue_depth: 256\n"
       "  conf_dir: /tmp/cte_qdrant_e2e\n"
       "\n"
       "compose:\n"
       "  - mod_name: wrp_cte_core\n"
       "    pool_name: cte_qdrant_e2e_pool\n"
       "    pool_id: \"4731.0\"\n"
       "    pool_query: local\n"
       "    knowledge_graph:\n"
       "      backend: qdrant\n"
       "      config: \""
    << qdrant << " " << embed
    << "\"\n"
       "    embedding:\n"
       "      endpoint: \""
    << embed
    << "\"\n"
       "      model: \"mock\"\n"
       "    indexing_depth:\n"
       "      default: 0\n"
       "      formats:\n"
       "        h5: 1\n"
       "        hdf5: 1\n";
  return path;
}

struct RuntimeFx {
  chi::PoolId pool_id = chi::PoolId(4731, 0);
  std::unique_ptr<Client> client;
  static inline bool g_inited = false;

  RuntimeFx() {
    std::string yaml = WriteComposeYaml();
    ::setenv("CHI_SERVER_CONF", yaml.c_str(), 1);

    // Also set embedder env in case it's consulted directly.
    const char *emb = std::getenv("EMB_URL");
    ::setenv("CTE_EMBEDDING_ENDPOINT",
             emb ? emb : "http://127.0.0.1:9999/v1/embeddings", 1);
    ::setenv("CTE_EMBEDDING_MODEL", "mock", 1);

    if (!g_inited) {
      bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(ok);
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      g_inited = true;
    }
    // The runtime auto-created the pool via compose, so just wrap it.
    client = std::make_unique<Client>(pool_id);
  }
};

}  // namespace

TEST_CASE("Runtime+Qdrant E2E: L1 embedding round-trip", "[depth][qdrant][e2e]") {
  RuntimeFx fx;

  auto tmp = fs::temp_directory_path() / "acropolis_runtime_qdrant.h5";
  { std::ofstream(tmp) << "x"; }
  // Format default in compose YAML is L1 for .h5; xattr forces L1 explicitly.
  // (L1 is where embeddings start being generated.)
  const char *val = "1";
  (void)::setxattr(tmp.c_str(), "user.acropolis.depth", val, 1, 0);

  std::string tag_name = tmp.string();
  auto got = fx.client->AsyncGetOrCreateTag(tag_name);
  got.Wait();
  REQUIRE(got->GetReturnCode() == 0);
  TagId tag_id = got->tag_id_;

  std::string caller = "unique_acropolis_runtime_token_alpha_beta_gamma";
  auto upd = fx.client->AsyncUpdateKnowledgeGraph(tag_id, tag_name, caller);
  upd.Wait();
  REQUIRE(upd->GetReturnCode() == 0);
  INFO("UpdateKnowledgeGraph at L1 succeeded (embedding generated)");

  // Self-query with the same text — Qdrant returns the point it just stored.
  auto sq = fx.client->AsyncSemanticQuery(caller, /*top_k=*/5,
                                          chi::PoolQuery::Local());
  sq.Wait();
  REQUIRE(sq->GetReturnCode() == 0);
  REQUIRE(!sq->result_tags_.empty());

  bool found = false;
  float score = -1.0f;
  for (size_t i = 0; i < sq->result_tags_.size(); ++i) {
    const auto &t = sq->result_tags_[i];
    if (t.major_ == tag_id.major_ && t.minor_ == tag_id.minor_) {
      found = true;
      score = sq->result_scores_[i];
      INFO("  hit score=" << score);
      break;
    }
  }
  REQUIRE(found);

  // Cosine similarity is in [-1, 1]; BM25 scores can easily exceed 2+.
  // The stored L1 payload = L0 + format extractor output + caller summary,
  // which is NOT identical to the query text alone, so we don't expect a
  // perfect 1.0. Our mock SHA256-based embedder produces nearly-random
  // vectors for different inputs, so the sign of the score is meaningless —
  // all we can check is that the score is in cosine range (not a BM25-style
  // unbounded score). That's enough to prove the vector path ran.
  INFO("  asserting cosine-range score in [-1.05, 1.05]");
  REQUIRE(score <= 1.05f);
  REQUIRE(score >= -1.05f);

  fs::remove(tmp);
}

SIMPLE_TEST_MAIN()
