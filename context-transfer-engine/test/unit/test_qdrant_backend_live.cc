/*
 * Acropolis — live Qdrant backend test.
 *
 * Instantiates QdrantBackend directly (no CTE runtime) against a running
 * Qdrant server and a mock OpenAI-compatible embedding endpoint. Verifies
 * that:
 *   1. Init() creates a collection with the correct embedding dim.
 *   2. Add() stores a document with its embedding.
 *   3. Search() retrieves the top-1 match for its own text.
 *   4. Remove() decrements the stored count.
 *
 * Configure via env vars before running:
 *   QDRANT_URL   (default "host.docker.internal:6333/acropolis_e2e")
 *   EMB_URL      (default "http://127.0.0.1:9999/v1/embeddings")
 */

#include <wrp_cte/core/kg_backend_qdrant.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "simple_test.h"

using wrp_cte::core::QdrantBackend;
using wrp_cte::core::TagId;

namespace {

TagId MakeTag(uint32_t major, uint32_t minor) {
  TagId t;
  t.major_ = major;
  t.minor_ = minor;
  return t;
}

std::string QdrantCfg() {
  const char *q = std::getenv("QDRANT_URL");
  const char *e = std::getenv("EMB_URL");
  std::string base = q ? q : "host.docker.internal:6333/acropolis_e2e";
  std::string emb  = e ? e : "http://127.0.0.1:9999/v1/embeddings";
  return base + " " + emb;
}

}  // namespace

TEST_CASE("Qdrant live: Init creates collection", "[qdrant][live]") {
  ::unsetenv("CTE_EMBEDDING_ENDPOINT");
  ::unsetenv("QDRANT_EMBEDDING_ENDPOINT");
  QdrantBackend q;
  q.Init(QdrantCfg());
  REQUIRE(q.Size() == 0);
  q.Clear();
  REQUIRE(q.Size() == 0);
}

TEST_CASE("Qdrant live: Add then Search returns same tag", "[qdrant][live]") {
  QdrantBackend q;
  q.Init(QdrantCfg());
  q.Clear();

  TagId tag = MakeTag(42, 7);
  std::string text = "temperature field on a 3D grid, units Kelvin";
  q.Add(tag, text);

  // Small pause: Qdrant is immediate after write but we give it a beat to
  // be safe against network timing on slow hosts.
  auto results = q.Search(text, /*top_k=*/5);
  REQUIRE(!results.empty());

  bool found = false;
  for (const auto &r : results) {
    if (r.key.major_ == tag.major_ && r.key.minor_ == tag.minor_) {
      found = true;
      INFO("  self-query score=" << r.score);
      break;
    }
  }
  REQUIRE(found);
}

TEST_CASE("Qdrant live: Remove decreases size", "[qdrant][live]") {
  QdrantBackend q;
  q.Init(QdrantCfg());
  q.Clear();

  q.Add(MakeTag(1, 0), "alpha");
  q.Add(MakeTag(2, 0), "beta");
  q.Add(MakeTag(3, 0), "gamma");
  REQUIRE(q.Size() == 3);

  q.Remove(MakeTag(2, 0));
  REQUIRE(q.Size() == 2);
}

TEST_CASE("Qdrant live: exact-text self-match is top for each doc",
          "[qdrant][live]") {
  // Our mock embedder hashes input to a vector, so only exact-text matches
  // round-trip cleanly. This test proves that every document can be retrieved
  // by its own text as the top hit (which is the minimum correctness bar for
  // the Add/Search plumbing). Semantic ranking requires a real embedder and
  // is out of scope for this backend-integration test.
  QdrantBackend q;
  q.Init(QdrantCfg());
  q.Clear();

  struct Entry { uint32_t id; std::string text; };
  std::vector<Entry> docs = {
      {100, "ocean salinity and temperature profiles"},
      {200, "cold dark matter particle counts cosmological sim"},
      {300, "galaxy cluster mass function z=0"},
  };
  for (const auto &d : docs) q.Add(MakeTag(d.id, 0), d.text);

  for (const auto &d : docs) {
    auto r = q.Search(d.text, 3);
    REQUIRE(!r.empty());
    INFO("  query='" << d.text.substr(0, 30) << "...' top=" << r[0].key.major_
         << " score=" << r[0].score);
    REQUIRE(r[0].key.major_ == d.id);
  }
}

SIMPLE_TEST_MAIN()
