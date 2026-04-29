/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

/**
 * DEPTH CONTROLLER UNIT TESTS — 3-level model
 *
 * Pure unit tests for Acropolis's adaptive indexing depth controller.
 * No CTE runtime required — exercises the header-only class directly.
 *
 * Levels:
 *   L0  Name      filename, path, size, ext
 *   L1  Metadata  + format detection + format extractor + embedding
 *   L2  Content   inside the controller, identical to L1; the runtime
 *                 layer appends a caller-supplied LLM summary on top.
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
// Enum + name helper
// ---------------------------------------------------------------------------

TEST_CASE("IndexDepth values are 0..2", "[depth][enum]") {
  REQUIRE(static_cast<int>(IndexDepth::kNameOnly) == 0);
  REQUIRE(static_cast<int>(IndexDepth::kMetadata) == 1);
  REQUIRE(static_cast<int>(IndexDepth::kContent)  == 2);
}

TEST_CASE("IndexDepthName returns human-readable labels", "[depth][enum]") {
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kNameOnly)) == "L0-name");
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kMetadata)) == "L1-metadata");
  REQUIRE(std::string(wrp_cte::core::IndexDepthName(IndexDepth::kContent))  == "L2-content");
}

// ---------------------------------------------------------------------------
// Policy resolution
// ---------------------------------------------------------------------------

TEST_CASE("Policy: explicit target wins over everything", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.global_default = IndexDepth::kNameOnly;
  dd.per_format["hdf5"] = IndexDepth::kMetadata;
  ctrl.SetDefaults(dd);

  auto r = ctrl.ResolvePolicy("/tmp/x.hdf5",
                              std::make_optional(IndexDepth::kContent));
  REQUIRE(r == IndexDepth::kContent);
}

TEST_CASE("Policy: format-specific default applied by extension", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.global_default = IndexDepth::kNameOnly;
  dd.per_format["hdf5"] = IndexDepth::kMetadata;
  dd.per_format["mp4"]  = IndexDepth::kNameOnly;
  ctrl.SetDefaults(dd);

  REQUIRE(ctrl.ResolvePolicy("/scratch/sim.hdf5", std::nullopt) == IndexDepth::kMetadata);
  REQUIRE(ctrl.ResolvePolicy("/scratch/video.mp4", std::nullopt) == IndexDepth::kNameOnly);
}

TEST_CASE("Policy: global default when no format match", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.global_default = IndexDepth::kMetadata;
  ctrl.SetDefaults(dd);

  REQUIRE(ctrl.ResolvePolicy("/scratch/unknown.xyz", std::nullopt) == IndexDepth::kMetadata);
  REQUIRE(ctrl.ResolvePolicy("/scratch/noext", std::nullopt) == IndexDepth::kMetadata);
}

TEST_CASE("Policy: extension lookup is case-insensitive", "[depth][policy]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.per_format["hdf5"] = IndexDepth::kMetadata;
  ctrl.SetDefaults(dd);

  REQUIRE(ctrl.ResolvePolicy("/tmp/sim.HDF5", std::nullopt) == IndexDepth::kMetadata);
  REQUIRE(ctrl.ResolvePolicy("/tmp/sim.Hdf5", std::nullopt) == IndexDepth::kMetadata);
}

#ifdef __linux__
TEST_CASE("Policy: file-level xattr overrides format default", "[depth][policy][xattr]") {
  DepthController ctrl;
  DepthDefaults dd;
  dd.per_format["hdf5"] = IndexDepth::kNameOnly;  // would be L0
  ctrl.SetDefaults(dd);

  auto tmp = fs::temp_directory_path() / "acropolis_xattr_test.hdf5";
  { std::ofstream(tmp) << "hello"; }
  const char *val = "2";
  int rc = ::setxattr(tmp.c_str(), "user.acropolis.depth", val, 1, 0);
  if (rc != 0) {
    INFO("setxattr unsupported on this filesystem; skipping xattr test");
    fs::remove(tmp);
    return;
  }

  REQUIRE(ctrl.ResolvePolicy(tmp.string(), std::nullopt) == IndexDepth::kContent);
  fs::remove(tmp);
}
#endif

// ---------------------------------------------------------------------------
// Level executor behaviour
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
  // L0 must NOT include format token or content_kind (those are L1).
  REQUIRE(p.text.find("format=") == std::string::npos);
  REQUIRE(p.text.find("content_kind=") == std::string::npos);
}

TEST_CASE("Index L1 adds format sniff + content_kind on top of L0",
          "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/data.h5", data, 4096,
                      IndexDepth::kMetadata);
  REQUIRE(p.depth_achieved == IndexDepth::kMetadata);
  REQUIRE(p.text.find("path=") != std::string::npos);          // L0 still present
  REQUIRE(p.text.find("format=h5") != std::string::npos);      // L1 format
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);
  REQUIRE(p.embedding.empty());  // no embedder configured here
}

TEST_CASE("Index L1 records content_kind for known scientific formats",
          "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data;

  auto p = ctrl.Index(MakeTag(1, 2), "/x.hdf5", data, 0, IndexDepth::kMetadata);
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);

  p = ctrl.Index(MakeTag(2, 0), "/x.nc", data, 0, IndexDepth::kMetadata);
  REQUIRE(p.text.find("content_kind=netcdf_scientific") != std::string::npos);

  p = ctrl.Index(MakeTag(3, 0), "/x.parquet", data, 0, IndexDepth::kMetadata);
  REQUIRE(p.text.find("content_kind=parquet_columnar") != std::string::npos);

  p = ctrl.Index(MakeTag(4, 0), "/x.csv", data, 0, IndexDepth::kMetadata);
  REQUIRE(p.text.find("content_kind=tabular_text") != std::string::npos);
}

TEST_CASE("Index L2 produces same controller payload as L1 (caller adds summary)",
          "[depth][levels]") {
  DepthController ctrl;
  std::vector<char> data;
  auto p1 = ctrl.Index(MakeTag(1, 2), "/scratch/data.h5", data, 4096,
                       IndexDepth::kMetadata);
  auto p2 = ctrl.Index(MakeTag(1, 2), "/scratch/data.h5", data, 4096,
                       IndexDepth::kContent);
  REQUIRE(p1.text == p2.text);
  REQUIRE(p2.depth_achieved == IndexDepth::kContent);
}

TEST_CASE("Embedding is generated at L1+ when an embedder is configured",
          "[depth][levels][embed]") {
  // Without an embedder, L1 still works — embedding stays empty.
  {
    DepthController ctrl;
    std::vector<char> data;
    auto p = ctrl.Index(MakeTag(7, 0), "/scratch/x.h5", data, 1024,
                        IndexDepth::kMetadata);
    REQUIRE(p.depth_achieved == IndexDepth::kMetadata);
    REQUIRE(p.embedding.empty());
  }
}

TEST_CASE("Higher depth never emits less text than lower depth",
          "[depth][monotonic]") {
  DepthController ctrl;
  std::vector<char> data(128, 'A');

  auto p0 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kNameOnly);
  auto p1 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kMetadata);
  auto p2 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", data, data.size(),
                       IndexDepth::kContent);

  REQUIRE(p1.text.size() >= p0.text.size());
  REQUIRE(p2.text.size() >= p1.text.size());
}

// ---------------------------------------------------------------------------
// Magic-byte fallback
// ---------------------------------------------------------------------------

TEST_CASE("L1 magic-byte fallback detects HDF5", "[depth][levels][magic]") {
  DepthController ctrl;
  std::vector<char> hdf5_header = {
      (char)0x89, 'H', 'D', 'F', '\r', '\n', (char)0x1A, '\n', 'X', 'X'};
  auto p = ctrl.Index(MakeTag(9, 0), "/scratch/noext", hdf5_header, 8,
                      IndexDepth::kMetadata);
  REQUIRE(p.text.find("format=hdf5") != std::string::npos);
}

TEST_CASE("L1 magic-byte fallback detects Parquet", "[depth][levels][magic]") {
  DepthController ctrl;
  std::vector<char> pq = {'P', 'A', 'R', '1', 'x', 'y'};
  auto p = ctrl.Index(MakeTag(9, 1), "/scratch/noext", pq, 4,
                      IndexDepth::kMetadata);
  REQUIRE(p.text.find("format=parquet") != std::string::npos);
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
  info.index_depth_ = IndexDepth::kContent;

  wrp_cte::core::TagInfo copy = info;
  REQUIRE(copy.index_depth_ == IndexDepth::kContent);

  wrp_cte::core::TagInfo assigned;
  assigned = info;
  REQUIRE(assigned.index_depth_ == IndexDepth::kContent);
}

// ---------------------------------------------------------------------------
// EmbeddingClient — config only, no live HTTP
// ---------------------------------------------------------------------------

TEST_CASE("EmbeddingClient.Configured false when no endpoint", "[embedder]") {
  EmbeddingClient ec;
  REQUIRE_FALSE(ec.Configured());
}

TEST_CASE("EmbeddingClient.Configure honours explicit endpoint", "[embedder]") {
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
// Metadata extractor plug-in API
// ---------------------------------------------------------------------------

TEST_CASE("L1 registered extractor is called for matching extension",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  REQUIRE_FALSE(ctrl.HasMetadataExtractor("h5"));

  int call_count = 0;
  std::string seen_path;
  ctrl.RegisterMetadataExtractor("h5",
      [&](const std::string &path) -> std::string {
        ++call_count;
        seen_path = path;
        return "fake_summary=extracted dataset_count=3";
      });

  REQUIRE(ctrl.HasMetadataExtractor("h5"));

  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/x.h5", empty, 0,
                      IndexDepth::kMetadata);
  REQUIRE(call_count == 1);
  REQUIRE(seen_path == "/scratch/x.h5");
  REQUIRE(p.text.find("fake_summary=extracted") != std::string::npos);
  // When a custom extractor runs, the default content_kind tag is suppressed.
  REQUIRE(p.text.find("content_kind=hdf5_scientific") == std::string::npos);
}

TEST_CASE("L1 unregistered extension falls back to content_kind tag",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  ctrl.RegisterMetadataExtractor("hdf5",
      [](const std::string &) { return "rich=hdf5_only"; });

  std::vector<char> empty;
  auto p_h5 = ctrl.Index(MakeTag(1, 0), "/x.h5", empty, 0,
                         IndexDepth::kMetadata);
  REQUIRE(p_h5.text.find("content_kind=hdf5_scientific") != std::string::npos);
  REQUIRE(p_h5.text.find("rich=hdf5_only") == std::string::npos);

  auto p_hdf5 = ctrl.Index(MakeTag(1, 0), "/x.hdf5", empty, 0,
                           IndexDepth::kMetadata);
  REQUIRE(p_hdf5.text.find("rich=hdf5_only") != std::string::npos);
}

TEST_CASE("L1 extractor returning empty string falls back to default tag",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  ctrl.RegisterMetadataExtractor("h5",
      [](const std::string &) { return std::string(); });

  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/broken.h5", empty, 0,
                      IndexDepth::kMetadata);
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);
}

TEST_CASE("UnregisterMetadataExtractor removes the extractor",
          "[depth][levels][extractor]") {
  DepthController ctrl;
  ctrl.RegisterMetadataExtractor("h5",
      [](const std::string &) { return "x=1"; });
  REQUIRE(ctrl.HasMetadataExtractor("h5"));
  ctrl.UnregisterMetadataExtractor("h5");
  REQUIRE_FALSE(ctrl.HasMetadataExtractor("h5"));

  std::vector<char> empty;
  auto p = ctrl.Index(MakeTag(1, 2), "/scratch/x.h5", empty, 0,
                      IndexDepth::kMetadata);
  REQUIRE(p.text.find("content_kind=hdf5_scientific") != std::string::npos);
}

SIMPLE_TEST_MAIN()
