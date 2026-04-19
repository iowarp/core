/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

/**
 * END-TO-END TEST: Depth controller through the live CTE runtime
 *
 * Boots an embedded Chimaera runtime with the BM25 KG backend, creates a
 * temporary file, sets its user.acropolis.depth xattr to L2, sends an
 * UpdateKnowledgeGraph task through the client, and verifies:
 *
 *   1. The runtime accepts the task.
 *   2. A subsequent SemanticQuery returns the tag.
 *   3. The query matches on depth-controller-generated text fragments
 *      (content_kind=hdf5_scientific for L2).
 *
 * Requires: WRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON at build time.
 */

#include <chimaera/admin/admin_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_runtime.h>
#include <wrp_cte/core/core_tasks.h>

#include <sys/xattr.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "simple_test.h"

namespace fs = std::filesystem;
using namespace wrp_cte::core;

namespace {

class RuntimeFixture {
 public:
  chi::PoolId cte_pool_id_ = chi::PoolId(4711, 0);
  std::unique_ptr<Client> client_;
  std::string storage_path_ = "/tmp/cte_depth_e2e_storage.dat";
  static inline bool g_inited = false;

  RuntimeFixture() {
    // Ensure no embedding endpoint is set — L3 should degrade to "no vector".
    ::unsetenv("CTE_EMBEDDING_ENDPOINT");
    ::unsetenv("QDRANT_EMBEDDING_ENDPOINT");

    if (fs::exists(storage_path_)) fs::remove(storage_path_);

    if (!g_inited) {
      bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(ok);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      g_inited = true;
    }

    client_ = std::make_unique<Client>(cte_pool_id_);

    // Create a CTE pool with BM25 KG backend and L0 default, L2 for HDF5.
    CreateParams params;
    params.config_.kg_backend_      = "bm25";
    params.config_.depth_default_   = 0;
    params.config_.depth_per_format_["h5"]   = 2;
    params.config_.depth_per_format_["hdf5"] = 2;

    auto task = client_->AsyncCreate(chi::PoolQuery::Dynamic(),
                                     "cte_depth_e2e_pool", cte_pool_id_, params);
    task.Wait();
    REQUIRE(task->GetReturnCode() == 0);
    INFO("Created CTE pool with BM25 KG + indexing_depth config");
  }

  ~RuntimeFixture() {
    client_.reset();
    if (fs::exists(storage_path_)) fs::remove(storage_path_);
  }
};

}  // namespace

TEST_CASE("E2E: DepthController runs inside UpdateKnowledgeGraph",
          "[depth][runtime][e2e]") {
  RuntimeFixture fx;

  // 1. Create a temp file so xattrs can bind to it, and set L2 via xattr.
  auto tmp = fs::temp_directory_path() / "acropolis_e2e_hdf5_file.h5";
  { std::ofstream(tmp) << "dummy"; }

  // xattr may fail on tmpfs — we still exercise the format-default path.
  const char *val = "2";
  int rc = ::setxattr(tmp.c_str(), "user.acropolis.depth", val, 1, 0);
  if (rc != 0) {
    INFO("setxattr unsupported, falling back to YAML format default");
  }

  // 2. Create a tag named after the file path.
  std::string tag_name = tmp.string();
  auto get_or_create = fx.client_->AsyncGetOrCreateTag(tag_name);
  get_or_create.Wait();
  REQUIRE(get_or_create->GetReturnCode() == 0);
  TagId tag_id = get_or_create->tag_id_;
  INFO("Tag id = " << tag_id.major_ << "." << tag_id.minor_);

  // 3. Send an UpdateKnowledgeGraph task. The handler will:
  //    - Read xattr / format default from the path
  //    - Run level executors 0..2 additively
  //    - Append the caller-supplied summary
  //    - Store the combined payload in the BM25 backend
  std::string caller_summary =
      "atmospheric temperature and pressure fields 3D grid units Kelvin";
  auto upd = fx.client_->AsyncUpdateKnowledgeGraph(tag_id, tag_name,
                                                    caller_summary);
  upd.Wait();
  REQUIRE(upd->GetReturnCode() == 0);

  // 4. Verify via SemanticQuery — BM25 should score the tag highly for any
  //    of the depth-controller-generated text fragments, *or* the caller
  //    summary text. We also verify a negative case.
  auto positive_term = [&](const std::string &q) -> bool {
    auto sq = fx.client_->AsyncSemanticQuery(q, /*top_k=*/5,
                                             chi::PoolQuery::Local());
    sq.Wait();
    if (sq->GetReturnCode() != 0) return false;
    for (size_t i = 0; i < sq->result_tags_.size(); ++i) {
      const auto &tid = sq->result_tags_[i];
      if (tid.major_ == tag_id.major_ && tid.minor_ == tag_id.minor_) {
        INFO("  hit for q='" << q << "' score=" << sq->result_scores_[i]);
        return true;
      }
    }
    return false;
  };

  // Caller-summary token
  REQUIRE(positive_term("temperature"));

  // L1 format-sniff content (only when xattr override is honoured OR when
  // H5/HDF5 format default kicks in)
  SECTION("L1 format token is searchable") {
    REQUIRE(positive_term("h5"));
  }

  // L2 content kind — only if L2 actually ran
  SECTION("L2 content_kind token is searchable") {
    REQUIRE(positive_term("hdf5_scientific"));
  }

  // L0 size / path token
  SECTION("L0 path token is searchable") {
    REQUIRE(positive_term("acropolis_e2e_hdf5_file"));
  }

  fs::remove(tmp);
}

TEST_CASE("E2E: L0 default applied for unknown extensions",
          "[depth][runtime][e2e]") {
  RuntimeFixture fx;

  auto tmp = fs::temp_directory_path() / "acropolis_e2e_plain.txt";
  { std::ofstream(tmp) << "hello"; }

  std::string tag_name = tmp.string();
  auto get_or_create = fx.client_->AsyncGetOrCreateTag(tag_name);
  get_or_create.Wait();
  TagId tag_id = get_or_create->tag_id_;

  std::string caller_summary = "document about galaxy evolution";
  auto upd = fx.client_->AsyncUpdateKnowledgeGraph(tag_id, tag_name,
                                                    caller_summary);
  upd.Wait();
  REQUIRE(upd->GetReturnCode() == 0);

  // Caller summary is always present.
  auto sq = fx.client_->AsyncSemanticQuery("galaxy", 5, chi::PoolQuery::Local());
  sq.Wait();
  bool found = false;
  for (const auto &tid : sq->result_tags_) {
    if (tid.major_ == tag_id.major_ && tid.minor_ == tag_id.minor_) {
      found = true;
      break;
    }
  }
  REQUIRE(found);

  // "hdf5_scientific" should NOT appear — this tag was only indexed at L0,
  // so the content-kind string was never emitted.
  sq = fx.client_->AsyncSemanticQuery("hdf5_scientific", 5,
                                      chi::PoolQuery::Local());
  sq.Wait();
  for (const auto &tid : sq->result_tags_) {
    // We don't assert size==0 because other tests in the same fixture may
    // have registered HDF5 tags. But this particular tag must not be among
    // the results for content_kind=hdf5_scientific if it was a .txt file.
    REQUIRE_FALSE(tid.major_ == tag_id.major_ && tid.minor_ == tag_id.minor_);
  }

  fs::remove(tmp);
}

SIMPLE_TEST_MAIN()
