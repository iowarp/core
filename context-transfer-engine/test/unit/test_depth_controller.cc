/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

/**
 * DEPTH CONTROLLER UNIT TESTS
 *
 * Pure unit tests for Acropolis's adaptive indexing depth controller.
 * No CTE runtime required — exercises the header-only class directly.
 */

#include <wrp_cte/core/depth_controller.h>
#include <wrp_cte/core/core_tasks.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/xattr.h>
#endif

#include "simple_test.h"

namespace fs = std::filesystem;
using wrp_cte::core::DepthController;
using wrp_cte::core::DepthDefaults;
using wrp_cte::core::EmbeddingClient;
using wrp_cte::core::IndexDepth;
using wrp_cte::core::IndexPayload;
using wrp_cte::core::TagId;

namespace {

TagId MakeTag(uint32_t major, uint32_t minor) {
  TagId t;
  t.major_ = major;
  t.minor_ = minor;
  return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// Enum + helpers
// ---------------------------------------------------------------------------

TEST_CASE("IndexDepth values are 0..4", "[depth][enum]") {
  REQUIRE(static_cast<int>(IndexDepth::kNameOnly)      == 0);
  REQUIRE(static_cast<int>(IndexDepth::kStatMeta)      == 1);
  REQUIRE(static_cast<int>(IndexDepth::kFormatExtract) == 2);
  REQUIRE(static_cast<int>(IndexDepth::kEmbedding)     == 3);
  REQUIRE(static_cast<int>(IndexDepth::kDeepContent)   == 4);
}

TEST_CASE("IndexDepthName returns human-readable labels", "[depth][enum]") {
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kNameOnly))    == "L0-name");
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kStatMeta))    == "L1-stat");
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kFormatExtract))== "L2-format");
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kEmbedding))   == "L3-embed");
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kDeepContent)) == "L4-deep");
}

// ---------------------------------------------------------------------------
// Policy resolution
// ---------------------------------------------------------------------------

TEST_CASE("Policy: explicit target wins over everything", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.global_default = IndexDepth::kNameOnly;
  dd.per_format["hdf5"] = IndexDepth::kFormatExtract;
  ctrl.SetDefaults(dd);

  // Despite the hdf5 default being L2, the explicit target overrides it
  auto r = ctrl.ResolvePolicy("/tmp/x.hdf5", std::make_optional(IndexDepth::kEmbedding));
  REQUIRE(r == IndexDepth::kEmbedding);
}

TEST_CASE("Policy: format-specific default applied by extension", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.global_default = IndexDepth::kNameOnly;
  dd.per_format["hdf5"] = IndexDepth::kFormatExtract;
  dd.per_format["mp4"]  = IndexDepth::kNameOnly;
  ctrl.SetDefaults(dd);

  REQUIRE(ctrl.ResolvePolicy("/scratch/sim.hdf5", std::nullopt) == IndexDepth::kFormatExtract);
  REQUIRE(ctrl.ResolvePolicy("/scratch/video.mp4", std::nullopt) == IndexDepth::kNameOnly);
}

TEST_CASE("Policy: global default when no format match", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.global_default = IndexDepth::kStatMeta;
  ctrl.SetDefaults(dd);

  REQUIRE(ctrl.ResolvePolicy("/scratch/unknown.xyz", std::nullopt) == IndexDepth::kStatMeta);
  // No extension at all
  REQUIRE(ctrl.ResolvePolicy("/scratch/noext", std::nullopt) == IndexDepth::kStatMeta);
}

TEST_CASE("Policy: extension lookup is case-insensitive", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.per_format["hdf5"] = IndexDepth::kFormatExtract;
  ctrl.SetDefaults(dd);

  REQUIRE(ctrl.ResolvePolicy("/tmp/sim.HDF5", std::nullopt) == IndexDepth::kFormatExtract);
  REQUIRE(ctrl.ResolvePolicy("/tmp/sim.Hdf5", std::nullopt) == IndexDepth::kFormatExtract);
}

#ifdef __linux__
TEST_CASE("Policy: file-level xattr overrides format default", "[depth][policy][xattr]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.per_format["hdf5"] = IndexDepth::kStatMeta;  // would be L1 by default
  ctrl.SetDefaults(dd);

  // Create a temporary file and set the xattr
  auto tmp = fs::temp_directory_path() / "acropolis_xattr_test.hdf5";
  { std::ofstream(tmp) << "hello"; }
  const char *val = "3";
  int rc = ::setxattr(tmp.c_str(), "user.acropolis.depth", val, 1, 0);
  if (rc != 0) {
    // xattr may be unsupported on this filesystem (tmpfs/overlayfs sometimes).
    // Skip gracefully.
    INFO("setxattr unsupported on this filesystem; skipping xattr override test");
    fs::remove(tmp);
    return;
  }

  REQUIRE(ctrl.ResolvePolicy(tmp.string(), std::nullopt) == IndexDepth::kEmbedding);
  fs::remove(tmp);
}
#endif

// ---------------------------------------------------------------------------
// Level executor behaviour (additive)
// ---------------------------------------------------------------------------

TEST_CASE("Index L0 produces name metadata only", "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/data.h5", data, 4096,
                      IndexDepth::kNameOnly);
  REQUIRE(p.depth_achieved == IndexDepth::kNameOnly);
  REQUIRE(p.text.find("path=/scratch/data.h5") != std::string::npos);
  REQUIRE(p.text.find("size=4096") != std::string::npos);
  REQUIRE(p.text.find("ext=h5") != std::string::npos);
  REQUIRE(p.embedding.empty());  // no embedding at L0
}

TEST_CASE("Index L1 adds format sniff on top of L0", "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/data.h5", data, 4096,
                      IndexDepth::kStatMeta);
  REQUIRE(p.depth_achieved == IndexDepth::kStatMeta);
  REQUIRE(p.text.find("path=") != std::string::npos);       // L0 still present
  REQUIRE(p.text.find("format=h5") != std::string::npos);   // L1 content
  REQUIRE(p.embedding.empty());
}

TEST_CASE("Index L2 records content kind for scientific formats",
          "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data;

  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/s.hdf5", data, 0,
                      IndexDepth::kFormatExtract);
  REQUIRE(p.depth_achieved == IndexDepth::kFormatExtract);
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);

  p = ctrl.Index(MakeTag(2, 0), "/data/x.nc", data, 0,
                 IndexDepth::kFormatExtract);
  REQUIRE(p.text.find("content_kind=netcdf_scientific") != std::string::npos);

  p = ctrl.Index(MakeTag(3, 0), "/data/x.parquet", data, 0,
                 IndexDepth::kFormatExtract);
  REQUIRE(p.text.find("content_kind=parquet_columnar") != std::string::npos);

  p = ctrl.Index(MakeTag(4, 0), "/data/x.csv", data, 0,
                 IndexDepth::kFormatExtract);
  REQUIRE(p.text.find("content_kind=tabular_text") != std::string::npos);
}

TEST_CASE("Index L3 without embedder still returns text without embedding",
          "[depth][levels][embed]") {
  DepthController ctrl;
  // Intentionally NOT calling SetEmbedder — L3 should degrade gracefully.
  std::vector<char> data;
  auto p = ctrl.Index(MakeTag(7, 0), "/scratch/experiment.h5", data, 1024,
                      IndexDepth::kEmbedding);
  REQUIRE(p.depth_achieved == IndexDepth::kEmbedding);
  REQUIRE(!p.text.empty());
  REQUIRE(p.embedding.empty());  // no endpoint -> empty vector (not an error)
}

TEST_CASE("Index L4 adds byte-level statistics", "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data(256);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>(i);
  }
  auto p = ctrl.Index(MakeTag(42, 0), "/tmp/bytes.bin", data, data.size(),
                      IndexDepth::kDeepContent);
  REQUIRE(p.depth_achieved == IndexDepth::kDeepContent);
  REQUIRE(p.text.find("byte_mean=") != std::string::npos);
  REQUIRE(p.text.find("byte_var=")  != std::string::npos);
}

TEST_CASE("Index L4 on empty data stays safe", "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 0), "/tmp/x.bin", empty, 0,
                      IndexDepth::kDeepContent);
  REQUIRE(p.depth_achieved == IndexDepth::kDeepContent);
  // No byte stats appended when there is no data — should not crash.
}

// ---------------------------------------------------------------------------
// Magic-byte fallback when extension is missing
// ---------------------------------------------------------------------------

TEST_CASE("L1 magic-byte fallback detects HDF5", "[depth][levels][magic]") {
  DepthController ctrl;
  std::vector<char> hdf5_header = {
      (char)0x89, 'H', 'D', 'F', '\r', '\n', (char)0x1A, '\n', 'X', 'X'};
  // filename has no extension, so format sniff should come from magic bytes
  auto p = ctrl.Index(MakeTag(9, 0), "/scratch/noext", hdf5_header, 8,
                      IndexDepth::kStatMeta);
  REQUIRE(p.text.find("format=hdf5") != std::string::npos);
}

TEST_CASE("L1 magic-byte fallback detects Parquet", "[depth][levels][magic]") {
  DepthController ctrl;
  std::vector<char> pq = {'P', 'A', 'R', '1', 'x', 'y'};
  auto p = ctrl.Index(MakeTag(9, 1), "/scratch/noext", pq, 4,
                      IndexDepth::kStatMeta);
  REQUIRE(p.text.find("format=parquet") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Depth is monotonic (lower target = less text)
// ---------------------------------------------------------------------------

TEST_CASE("Higher depth never emits less text than lower depth",
          "[depth][monotonic]") {
  DepthController ctrl;
  std::vector<char> data(128, 'A');

  auto p0 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kNameOnly);
  auto p1 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kStatMeta);
  auto p2 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kFormatExtract);
  auto p4 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kDeepContent);

  REQUIRE(p1.text.size() >= p0.text.size());
  REQUIRE(p2.text.size() >= p1.text.size());
  REQUIRE(p4.text.size() >= p2.text.size());
}

// ---------------------------------------------------------------------------
// TagInfo carries the depth field
// ---------------------------------------------------------------------------

TEST_CASE("TagInfo default index_depth is L0", "[depth][taginfo]") {
  wrp_cte::core::TagInfo info("my_tag", MakeTag(1, 1));
  REQUIRE(info.index_depth_ == IndexDepth::kNameOnly);
}

TEST_CASE("TagInfo depth field round-trips through copy", "[depth][taginfo]") {
  wrp_cte::core::TagInfo info("my_tag", MakeTag(1, 1));
  info.index_depth_ = IndexDepth::kEmbedding;

  wrp_cte::core::TagInfo copy = info;
  REQUIRE(copy.index_depth_ == IndexDepth::kEmbedding);

  wrp_cte::core::TagInfo assigned;
  assigned = info;
  REQUIRE(assigned.index_depth_ == IndexDepth::kEmbedding);
}

// ---------------------------------------------------------------------------
// EmbeddingClient — configuration only (no live HTTP call)
// ---------------------------------------------------------------------------

TEST_CASE("EmbeddingClient.Configured false when no endpoint", "[embedder]") {
  EmbeddingClient ec;
  REQUIRE_FALSE(ec.Configured());
}

TEST_CASE("EmbeddingClient.Configure honours explicit endpoint", "[embedder]") {
  // Unset any env overrides for this test
  ::unsetenv("CTE_EMBEDDING_ENDPOINT");
  ::unsetenv("QDRANT_EMBEDDING_ENDPOINT");

  EmbeddingClient ec;
  ec.Configure("http://localhost:8090/v1/embeddings", "qwen2.5-3b");
  REQUIRE(ec.Configured());
  REQUIRE(ec.Endpoint() == "http://localhost:8090/v1/embeddings");
  REQUIRE(ec.Model()    == "qwen2.5-3b");
}

TEST_CASE("EmbeddingClient.Embed returns empty when no endpoint", "[embedder]") {
  EmbeddingClient ec;
  auto v = ec.Embed("hello world");
  REQUIRE(v.empty());
}

// ---------------------------------------------------------------------------
// L2 extractor registration (pluggable per-format extractors)
// ---------------------------------------------------------------------------

TEST_CASE("L2 registered extractor is called for matching extension",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  REQUIRE_FALSE(ctrl.HasL2Extractor("h5"));

  int call_count = 0;
  std::string seen_path;
  ctrl.RegisterL2Extractor("h5",
      [&](const std::string &path) -> std::string {
        ++call_count;
        seen_path = path;
        return "fake_summary=extracted dataset_count=3";
      });

  REQUIRE(ctrl.HasL2Extractor("h5"));

  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/x.h5", empty, 0,
                      IndexDepth::kFormatExtract);
  REQUIRE(call_count == 1);
  REQUIRE(seen_path == "/scratch/x.h5");
  REQUIRE(p.text.find("fake_summary=extracted") != std::string::npos);
  // When a custom extractor runs, the default content_kind tag is suppressed.
  REQUIRE(p.text.find("content_kind=hdf5_scientific") == std::string::npos);
}

TEST_CASE("L2 unregistered extension falls back to content_kind tag",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  // register for hdf5 only; h5 is separate
  ctrl.RegisterL2Extractor("hdf5", [](const std::string &) {
    return "rich=hdf5_only";
  });

  std::vector<char> empty;
  auto p_h5 = ctrl.Index(MakeTag(1, 0), "/x.h5", empty, 0,
                         IndexDepth::kFormatExtract);
  REQUIRE(p_h5.text.find("content_kind=hdf5_scientific") != std::string::npos);
  REQUIRE(p_h5.text.find("rich=hdf5_only") == std::string::npos);

  auto p_hdf5 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", empty, 0,
                           IndexDepth::kFormatExtract);
  REQUIRE(p_hdf5.text.find("rich=hdf5_only") != std::string::npos);
}

TEST_CASE("L2 extractor returning empty string falls back to default tag",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  ctrl.RegisterL2Extractor("h5", [](const std::string &) {
    return std::string();  // extractor couldn't open the file
  });

  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/broken.h5", empty, 0,
                      IndexDepth::kFormatExtract);
  // Fallback kicks in when extractor returns empty
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);
}

TEST_CASE("L2 Unregister removes the extractor",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  ctrl.RegisterL2Extractor("h5", [](const std::string &) { return "x=1"; });
  REQUIRE(ctrl.HasL2Extractor("h5"));
  ctrl.UnregisterL2Extractor("h5");
  REQUIRE_FALSE(ctrl.HasL2Extractor("h5"));

  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/x.h5", empty, 0,
                      IndexDepth::kFormatExtract);
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);
}

SIMPLE_TEST_MAIN()
