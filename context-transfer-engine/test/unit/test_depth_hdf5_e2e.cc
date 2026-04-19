/*
 * Acropolis — L2 HDF5 extractor E2E test.
 *
 * Creates a real HDF5 file with named datasets and attributes, ingests it
 * through the CTE runtime with the built-in Hdf5Summary extractor registered,
 * and verifies that a SemanticQuery for a dataset name returns the tag.
 *
 * Proves that:
 *   - The Hdf5Summary extractor opens a real file and walks its group tree
 *   - The extractor output replaces the default content_kind=... tag
 *   - The rich text is searchable through the BM25 backend
 */

#include <chimaera/admin/admin_client.h>
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_runtime.h>
#include <wrp_cte/core/core_tasks.h>

#include <hdf5.h>
#include <sys/xattr.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "simple_test.h"

namespace fs = std::filesystem;
using namespace wrp_cte::core;

namespace {

// Create an HDF5 file with a known group/dataset/attribute structure so we
// can assert the extractor pulled out specific names.
void WriteSampleHdf5(const std::string &path) {
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t atm_group = H5Gcreate2(file, "/atmosphere",
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[3] = {360, 180, 50};
  hid_t space = H5Screate_simple(3, dims, nullptr);
  hid_t ds_temp = H5Dcreate2(atm_group, "temperature_kelvin", H5T_NATIVE_FLOAT,
                             space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t ds_pres = H5Dcreate2(atm_group, "pressure_pascal", H5T_NATIVE_DOUBLE,
                             space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(ds_temp);
  H5Dclose(ds_pres);
  H5Sclose(space);

  // String attribute on the atmosphere group
  hid_t scalar = H5Screate(H5S_SCALAR);
  hid_t atype  = H5Tcopy(H5T_C_S1);
  const char *attr_value = "cosmological_baseline";
  H5Tset_size(atype, strlen(attr_value) + 1);  // include null terminator
  hid_t attr = H5Acreate2(atm_group, "simulation_type", atype, scalar,
                          H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, atype, attr_value);
  H5Aclose(attr);
  H5Tclose(atype);
  H5Sclose(scalar);

  H5Gclose(atm_group);
  H5Fclose(file);
}

struct RuntimeFx {
  chi::PoolId pool_id = chi::PoolId(4741, 0);
  std::unique_ptr<Client> client;
  static inline bool g_inited = false;

  RuntimeFx() {
    ::unsetenv("CTE_EMBEDDING_ENDPOINT");
    ::unsetenv("QDRANT_EMBEDDING_ENDPOINT");

    if (!g_inited) {
      bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(ok);
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      g_inited = true;
    }
    client = std::make_unique<Client>(pool_id);

    CreateParams params;
    params.config_.kg_backend_ = "bm25";
    // Force L2 for any .h5 so the extractor is invoked
    params.config_.depth_per_format_["h5"]   = 2;
    params.config_.depth_per_format_["hdf5"] = 2;

    auto t = client->AsyncCreate(chi::PoolQuery::Dynamic(),
                                 "cte_hdf5_e2e_pool", pool_id, params);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
  }
};

bool QueryHit(Client &c, const TagId &tag, const std::string &q) {
  auto sq = c.AsyncSemanticQuery(q, 5, chi::PoolQuery::Local());
  sq.Wait();
  if (sq->GetReturnCode() != 0) return false;
  for (const auto &t : sq->result_tags_) {
    if (t.major_ == tag.major_ && t.minor_ == tag.minor_) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("E2E: Hdf5 L2 extractor walks dataset tree and indexes names",
          "[depth][hdf5][e2e]") {
  RuntimeFx fx;

  auto tmp = fs::temp_directory_path() / "acropolis_e2e_atm.h5";
  if (fs::exists(tmp)) fs::remove(tmp);
  WriteSampleHdf5(tmp.string());

  // Set xattr to L2 so we definitely invoke the extractor.
  const char *v = "2";
  int xattr_rc = ::setxattr(tmp.c_str(), "user.acropolis.depth", v, 1, 0);
  INFO("setxattr rc=" << xattr_rc);

  auto g = fx.client->AsyncGetOrCreateTag(tmp.string());
  g.Wait();
  TagId tag_id = g->tag_id_;

  // NOTE: pass no caller summary, so the ONLY content in the index comes
  // from the DepthController (L0 + L1 + L2 HDF5 extractor).
  auto upd = fx.client->AsyncUpdateKnowledgeGraph(tag_id, tmp.string(), "");
  upd.Wait();
  REQUIRE(upd->GetReturnCode() == 0);

  // Collect which queries hit so we can report them all (not short-circuit).
  struct Case { const char *q; bool must_hit; };
  Case cases[] = {
    // L0 token — the filename must always be indexed
    {"acropolis",            true},
    // L1 token — format sniff
    {"h5",                   true},
    // L2 default tag — should fire even if the extractor returned empty
    {"hdf5_scientific",      true},
    // L2 extractor-produced tokens (split by BM25 on '_' / '/')
    {"atmosphere",           true},
    {"temperature",          true},
    {"kelvin",               true},
    {"pressure",             true},
    {"pascal",               true},
    {"cosmological",         true},
    {"baseline",             true},
  };
  int passed = 0, failed = 0;
  for (const auto &c : cases) {
    bool hit = QueryHit(*fx.client, tag_id, c.q);
    std::cout << "    query='" << c.q << "' hit=" << (hit ? "YES" : "no ")
              << "\n";
    if (c.must_hit) {
      if (hit) ++passed; else ++failed;
    }
  }
  INFO("passed=" << passed << " failed=" << failed);
  REQUIRE(failed == 0);

  fs::remove(tmp);
}

SIMPLE_TEST_MAIN()
