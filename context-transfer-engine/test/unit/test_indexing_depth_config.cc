/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

/**
 * INDEXING DEPTH CONFIG PARSING TESTS
 *
 * Exercises Config::LoadFromString for the new `indexing_depth:` and
 * `embedding:` sections added by Acropolis.
 *
 * No CTE runtime required — operates on Config object directly.
 */

#include <wrp_cte/core/core_config.h>

#include <string>

#include "simple_test.h"

using wrp_cte::core::Config;

TEST_CASE("Config: indexing_depth default parsed", "[config][depth]") {
  const char *yaml =
      "indexing_depth:\n"
      "  default: 2\n";
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  REQUIRE(cfg.depth_default_ == 2);
}

TEST_CASE("Config: indexing_depth per-format map parsed", "[config][depth]") {
  const char *yaml =
      "indexing_depth:\n"
      "  default: 0\n"
      "  formats:\n"
      "    h5: 2\n"
      "    nc: 2\n"
      "    csv: 1\n"
      "    mp4: 0\n";
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  REQUIRE(cfg.depth_default_ == 0);
  REQUIRE(cfg.depth_per_format_.size() == 4);
  REQUIRE(cfg.depth_per_format_["h5"]  == 2);
  REQUIRE(cfg.depth_per_format_["nc"]  == 2);
  REQUIRE(cfg.depth_per_format_["csv"] == 1);
  REQUIRE(cfg.depth_per_format_["mp4"] == 0);
}

TEST_CASE("Config: indexing_depth out-of-range default clamps to 0",
          "[config][depth][validation]") {
  const char *yaml =
      "indexing_depth:\n"
      "  default: 7\n";
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  REQUIRE(cfg.depth_default_ == 0);
}

TEST_CASE("Config: indexing_depth out-of-range format level is skipped",
          "[config][depth][validation]") {
  const char *yaml =
      "indexing_depth:\n"
      "  formats:\n"
      "    h5: 99\n"
      "    nc: 2\n"
      "    bad: -1\n";
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  // Only the valid "nc" entry should survive
  REQUIRE(cfg.depth_per_format_.count("h5")  == 0);
  REQUIRE(cfg.depth_per_format_.count("bad") == 0);
  REQUIRE(cfg.depth_per_format_["nc"] == 2);
}

TEST_CASE("Config: embedding section parsed", "[config][embed]") {
  const char *yaml =
      "embedding:\n"
      "  endpoint: \"http://localhost:8090/v1/embeddings\"\n"
      "  model: \"qwen2.5-3b\"\n";
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  REQUIRE(cfg.embedding_endpoint_ == "http://localhost:8090/v1/embeddings");
  REQUIRE(cfg.embedding_model_    == "qwen2.5-3b");
}

TEST_CASE("Config: missing indexing_depth leaves defaults", "[config][depth]") {
  const char *yaml =
      "runtime_type: core\n";  // no indexing_depth section
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  REQUIRE(cfg.depth_default_ == 0);
  REQUIRE(cfg.depth_per_format_.empty());
  REQUIRE(cfg.embedding_endpoint_.empty());
}

TEST_CASE("Config: full Acropolis compose section parses cleanly",
          "[config][depth][full]") {
  const char *yaml =
      "indexing_depth:\n"
      "  default: 0\n"
      "  formats:\n"
      "    h5: 2\n"
      "    hdf5: 2\n"
      "    nc: 2\n"
      "    parquet: 2\n"
      "    csv: 1\n"
      "    mp4: 0\n"
      "\n"
      "embedding:\n"
      "  endpoint: \"http://localhost:8090/v1/embeddings\"\n"
      "  model: \"qwen2.5-3b\"\n"
      "\n"
      "knowledge_graph:\n"
      "  backend: bm25\n";
  Config cfg;
  REQUIRE(cfg.LoadFromString(yaml));
  REQUIRE(cfg.depth_default_ == 0);
  REQUIRE(cfg.depth_per_format_.size() == 6);
  REQUIRE(cfg.embedding_endpoint_ == "http://localhost:8090/v1/embeddings");
  REQUIRE(cfg.kg_backend_ == "bm25");
}

SIMPLE_TEST_MAIN()
