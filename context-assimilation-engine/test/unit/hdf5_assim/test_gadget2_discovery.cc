/*
 * test_gadget2_discovery.cc - GADGET-2 N-body Simulation Data Discovery
 *
 * Tests CTE's ability to discover GADGET-2 simulation snapshots by their
 * physics configuration using natural-language queries.
 *
 * Pipeline:
 *   1. INGEST:     Read HDF5 Header attributes from each snapshot
 *   2. SUMMARIZE:  LLM generates human-readable summary from raw header
 *   3. INDEX KG:   Summary indexed into knowledge graph (BM25 or Qdrant)
 *   4. QUERY:      Natural-language queries find the right simulation
 *
 * Environment Variables:
 *   GADGET2_OUTPUT_DIR    - Parent directory with run subdirectories
 *                           (default: /mnt/common/rpawar4/gadget2_runs)
 *   CAE_SUMMARY_ENDPOINT  - LLM endpoint for summarization (optional)
 *   CAE_SUMMARY_MODEL     - LLM model name (optional)
 *
 * Usage:
 *   export CHI_SERVER_CONF=cte_qdrant_compose.yaml
 *   ./build/bin/test_gadget2_discovery
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>

#ifdef WRP_CAE_ENABLE_SUMMARY_OP
#include <wrp_cae/core/factory/summary_operator.h>
#endif
#include <hermes_shm/util/logging.h>

// Summaries loaded from pre-generated JSON (no runtime LLM dependency)

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static const std::string kDefaultDir =
    "/mnt/common/rpawar4/gadget2_runs";
static const std::string kTagPrefix = "gadget2/";

struct SnapshotInfo {
  std::string filepath;
  std::string run_name;
  std::string filename;
  std::string tag_name;
  std::string description;
};

/// Build description from GADGET-2 HDF5 Header
std::string BuildGadget2Description(const std::string &filepath,
                                     const std::string &run_name) {
  hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) return "";

  hid_t header = H5Gopen(file_id, "Header", H5P_DEFAULT);
  if (header < 0) { H5Fclose(file_id); return ""; }

  std::ostringstream desc;
  desc << "GADGET-2 N-body simulation snapshot. ";
  desc << "run_name=" << run_name << " ";

  // Read NumPart_Total
  int npart[6] = {0};
  if (H5Aexists(header, "NumPart_Total") > 0) {
    hid_t attr = H5Aopen(header, "NumPart_Total", H5P_DEFAULT);
    unsigned int uparts[6] = {0};
    H5Aread(attr, H5T_NATIVE_UINT, uparts);
    H5Aclose(attr);
    for (int i = 0; i < 6; i++) npart[i] = uparts[i];
  }

  // Particle type interpretation
  if (npart[0] > 0) desc << "gas_particles=" << npart[0] << " ";
  if (npart[1] > 0) desc << "dark_matter_particles=" << npart[1] << " ";
  if (npart[2] > 0) desc << "disk_particles=" << npart[2] << " ";
  if (npart[3] > 0) desc << "bulge_particles=" << npart[3] << " ";
  if (npart[4] > 0) desc << "star_particles=" << npart[4] << " ";

  int total_parts = 0;
  for (int i = 0; i < 6; i++) total_parts += npart[i];
  desc << "total_particles=" << total_parts << " ";

  // Simulation type inference
  bool has_gas = npart[0] > 0;
  bool has_dm = npart[1] > 0;
  bool has_disk = npart[2] > 0;
  bool has_bulge = npart[3] > 0;

  if (has_gas && !has_dm && !has_disk)
    desc << "simulation_type=hydrodynamic_gas_sphere ";
  else if (!has_gas && has_dm && has_disk && has_bulge)
    desc << "simulation_type=cosmological_cluster_with_bulge ";
  else if (!has_gas && has_dm && has_disk)
    desc << "simulation_type=isolated_galaxy_dark_matter_disk ";
  else if (has_gas && has_dm)
    desc << "simulation_type=cosmological_with_gas_and_dark_matter ";
  else if (has_dm && !has_gas)
    desc << "simulation_type=dark_matter_only ";

  // Read scalar header attributes
  auto read_int = [&](const char *name) -> int {
    if (H5Aexists(header, name) <= 0) return -1;
    hid_t a = H5Aopen(header, name, H5P_DEFAULT);
    int v = -1; H5Aread(a, H5T_NATIVE_INT, &v); H5Aclose(a);
    return v;
  };
  auto read_double = [&](const char *name) -> double {
    if (H5Aexists(header, name) <= 0) return -1;
    hid_t a = H5Aopen(header, name, H5P_DEFAULT);
    double v = -1; H5Aread(a, H5T_NATIVE_DOUBLE, &v); H5Aclose(a);
    return v;
  };

  int flag_sfr = read_int("Flag_Sfr");
  int flag_cool = read_int("Flag_Cooling");
  int flag_fb = read_int("Flag_Feedback");
  int flag_metals = read_int("Flag_Metals");
  int flag_age = read_int("Flag_StellarAge");
  double omega0 = read_double("Omega0");
  double omega_lambda = read_double("OmegaLambda");
  double hubble = read_double("HubbleParam");
  double box_size = read_double("BoxSize");
  double redshift = read_double("Redshift");
  double time_val = read_double("Time");

  if (flag_sfr >= 0) desc << "star_formation=" << (flag_sfr ? "enabled" : "disabled") << " ";
  if (flag_cool >= 0) desc << "cooling=" << (flag_cool ? "enabled" : "disabled") << " ";
  if (flag_fb >= 0) desc << "feedback=" << (flag_fb ? "enabled" : "disabled") << " ";
  if (flag_metals >= 0) desc << "metals=" << (flag_metals ? "tracked" : "not_tracked") << " ";

  if (omega0 > 0) {
    desc << "cosmological=true ";
    desc << "Omega0=" << omega0 << " ";
    desc << "OmegaLambda=" << omega_lambda << " ";
    desc << "HubbleParam=" << hubble << " ";
    desc << "Redshift=" << redshift << " ";
  } else {
    desc << "cosmological=false non_cosmological_simulation ";
  }

  if (box_size > 0) desc << "BoxSize=" << box_size << " ";
  desc << "Time=" << time_val << " ";

  // Read MassTable
  if (H5Aexists(header, "MassTable") > 0) {
    hid_t a = H5Aopen(header, "MassTable", H5P_DEFAULT);
    double masses[6] = {0};
    H5Aread(a, H5T_NATIVE_DOUBLE, masses);
    H5Aclose(a);
    for (int i = 0; i < 6; i++) {
      if (masses[i] > 0) desc << "mass_type" << i << "=" << masses[i] << " ";
    }
  }

  // List particle type groups and their datasets
  desc << "data_fields=";
  for (int i = 0; i < 6; i++) {
    char grp_name[32];
    snprintf(grp_name, sizeof(grp_name), "PartType%d", i);
    if (H5Lexists(file_id, grp_name, H5P_DEFAULT) > 0) {
      hid_t grp = H5Gopen(file_id, grp_name, H5P_DEFAULT);
      if (grp >= 0) {
        hsize_t nobj = 0;
        H5Gget_num_objs(grp, &nobj);
        for (hsize_t j = 0; j < nobj; j++) {
          char ds_name[128];
          H5Gget_objname_by_idx(grp, j, ds_name, sizeof(ds_name));
          desc << grp_name << "/" << ds_name << " ";
        }
        H5Gclose(grp);
      }
    }
  }

  H5Gclose(header);
  H5Fclose(file_id);
  return desc.str();
}

/// Find all snapshot HDF5 files across run subdirectories
std::vector<SnapshotInfo> FindAllSnapshots(const std::string &parent_dir) {
  std::vector<SnapshotInfo> files;
  DIR *parent = opendir(parent_dir.c_str());
  if (!parent) return files;

  struct dirent *run_entry;
  while ((run_entry = readdir(parent)) != nullptr) {
    std::string run_name = run_entry->d_name;
    if (run_name == "." || run_name == "..") continue;

    std::string outdir = parent_dir + "/" + run_name + "/output";
    DIR *out_dp = opendir(outdir.c_str());
    if (!out_dp) continue;

    struct dirent *file_entry;
    while ((file_entry = readdir(out_dp)) != nullptr) {
      std::string fname = file_entry->d_name;
      if (fname.find("snapshot_") == 0 && fname.find(".hdf5") != std::string::npos) {
        SnapshotInfo info;
        info.filepath = outdir + "/" + fname;
        info.run_name = run_name;
        info.filename = fname;
        info.tag_name = kTagPrefix + run_name + "/" + fname;
        files.push_back(info);
      }
    }
    closedir(out_dp);
  }
  closedir(parent);
  std::sort(files.begin(), files.end(),
            [](const SnapshotInfo &a, const SnapshotInfo &b) {
              return a.tag_name < b.tag_name;
            });
  return files;
}

struct Query {
  std::string text;
  std::string expected_run;  // substring in run_name
  std::string category;
};

int main() {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "GADGET-2 Data Discovery Benchmark");
  HLOG(kInfo, "========================================");

#ifndef WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
  HLOG(kWarning, "Knowledge graph not compiled.");
  return 0;
#else

  const char *dir_env = std::getenv("GADGET2_OUTPUT_DIR");
  std::string base_dir = dir_env ? dir_env : kDefaultDir;

  auto snapshots = FindAllSnapshots(base_dir);
  if (snapshots.empty()) {
    HLOG(kError, "No GADGET-2 snapshots found in {}", base_dir);
    return 1;
  }

  std::unordered_set<std::string> unique_runs;
  for (const auto &s : snapshots) unique_runs.insert(s.run_name);

  HLOG(kInfo, "Found {} snapshots across {} runs in {}",
       snapshots.size(), unique_runs.size(), base_dir);

  // Initialize CTE
  bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!ok) { HLOG(kError, "Chimaera init failed"); return 1; }
  wrp_cte::core::WRP_CTE_CLIENT_INIT();

  // ============================================================
  // Phase 1: Ingest — read headers, create tags, store descriptions
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 1: Ingest GADGET-2 snapshots ===");

  std::unordered_map<std::string, wrp_cte::core::TagId> tag_ids;
  std::unordered_map<std::string, std::string> tag_to_filepath;

  auto t_ingest_start = Clock::now();
  for (auto &snap : snapshots) {
    snap.description = BuildGadget2Description(snap.filepath, snap.run_name);
    if (snap.description.empty()) continue;

    wrp_cte::core::Tag tag(snap.tag_name);
    try {
      tag.PutBlob("description", snap.description.c_str(), snap.description.size());
    } catch (...) {}

    auto tid = tag.GetTagId();
    tag_ids[snap.tag_name] = tid;
    tag_to_filepath[snap.tag_name] = snap.filepath;
  }
  auto t_ingest_end = Clock::now();
  double ingest_ms = Ms(t_ingest_end - t_ingest_start).count();
  HLOG(kInfo, "  {} snapshots ingested ({:.1f} ms)", tag_ids.size(), ingest_ms);
  if (!snapshots.empty()) {
    HLOG(kInfo, "  Sample: {}", snapshots[0].description.substr(0, 200));
  }

  // ============================================================
  // Phase 2: Summarize via Summary Operator
  // ============================================================
  // The Summary Operator reads the "description" blob from each tag,
  // detects if it contains human-readable text or raw metadata,
  // calls the LLM with the appropriate prompt, and writes a "summary" blob.
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 2: Summarize via Summary Operator ===");

  auto cte_client = std::make_shared<wrp_cte::core::Client>();
  cte_client->Init(WRP_CTE_CLIENT->pool_id_);
  wrp_cae::core::SummaryOperator summary_op(cte_client);

  auto t_sum_start = Clock::now();
  int sum_ok = 0, sum_fail = 0;

  // Deduplicate: same description across timesteps gets same summary
  std::unordered_map<std::string, std::string> desc_to_summary;

  for (const auto &[tag_name, tid] : tag_ids) {
    // Read description blob for dedup check
    std::string desc;
    try {
      wrp_cte::core::Tag tag(tag_name);
      chi::u64 sz = tag.GetBlobSize("description");
      if (sz > 0 && sz < 16384) {
        std::vector<char> buf(sz + 1, '\0');
        tag.GetBlob("description", buf.data(), sz);
        desc = std::string(buf.data(), sz);
      }
    } catch (...) {}

    // Check dedup cache — same description gets same summary
    if (!desc.empty() && desc_to_summary.count(desc)) {
      try {
        wrp_cte::core::Tag tag(tag_name);
        const auto &cached = desc_to_summary[desc];
        tag.PutBlob("summary", cached.c_str(), cached.size());
        sum_ok++;
      } catch (...) { sum_fail++; }
      continue;
    }

    // Call Summary Operator (reads description blob → LLM → writes summary blob)
    int rc = summary_op.Execute(tag_name);
    if (rc == 0) {
      sum_ok++;
      // Cache the summary for dedup
      if (!desc.empty()) {
        try {
          wrp_cte::core::Tag tag(tag_name);
          chi::u64 sz = tag.GetBlobSize("summary");
          if (sz > 0 && sz < 4096) {
            std::vector<char> buf(sz + 1, '\0');
            tag.GetBlob("summary", buf.data(), sz);
            desc_to_summary[desc] = std::string(buf.data(), sz);
          }
        } catch (...) {}
      }
    } else {
      sum_fail++;
    }

    if ((sum_ok + sum_fail) % 50 == 0) {
      HLOG(kInfo, "  Progress: {}/{} ({} unique LLM calls)",
           sum_ok + sum_fail, tag_ids.size(), desc_to_summary.size());
    }
  }

  auto t_sum_end = Clock::now();
  double sum_ms = Ms(t_sum_end - t_sum_start).count();
  HLOG(kInfo, "  {} summaries, {} failed ({:.0f} ms, {} unique LLM calls)",
       sum_ok, sum_fail, sum_ms, desc_to_summary.size());

  if (!desc_to_summary.empty()) {
    auto it = desc_to_summary.begin();
    HLOG(kInfo, "  Sample summary: {}", it->second);
  }

  // ============================================================
  // Phase 3: Index Knowledge Graph
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 3: Index Knowledge Graph ===");

  auto t_kg_start = Clock::now();
  int indexed = 0;
  for (const auto &[tag_name, tid] : tag_ids) {
    std::string text;
    wrp_cte::core::Tag tag(tag_name);

    // Read the summary blob (written by Summary Operator in Phase 2)
    try {
      chi::u64 sz = tag.GetBlobSize("summary");
      if (sz > 0 && sz < 4096) {
        std::vector<char> buf(sz + 1, '\0');
        tag.GetBlob("summary", buf.data(), sz);
        text = std::string(buf.data(), sz);
      }
    } catch (...) {}

    // Fallback to description if summary somehow missing
    if (text.empty()) {
      try {
        chi::u64 sz = tag.GetBlobSize("description");
        if (sz > 0 && sz < 16384) {
          std::vector<char> buf(sz + 1, '\0');
          tag.GetBlob("description", buf.data(), sz);
          text = std::string(buf.data(), sz);
        }
      } catch (...) {}
    }
    if (text.empty()) text = tag_name;

    auto fut = WRP_CTE_CLIENT->AsyncUpdateKnowledgeGraph(tid, tag_name, text);
    fut.Wait();
    indexed++;
  }
  auto t_kg_end = Clock::now();
  double kg_ms = Ms(t_kg_end - t_kg_start).count();
  HLOG(kInfo, "  {} entries indexed ({:.1f} ms)", indexed, kg_ms);

  // ============================================================
  // Phase 3.5: Sync Global IDF
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 3.5: Sync Global IDF ===");
  auto sync_fut = WRP_CTE_CLIENT->AsyncSyncKnowledgeGraph();
  sync_fut.Wait();
  auto *sync_r = sync_fut.get();
  auto dist_fut = WRP_CTE_CLIENT->AsyncSyncKnowledgeGraph(
      chi::PoolQuery::Broadcast(), true,
      sync_r->global_n_, sync_r->global_total_terms_, sync_r->global_df_);
  dist_fut.Wait();
  HLOG(kInfo, "  Global IDF: N={}, terms={}", sync_r->global_n_, sync_r->global_df_.size());

  // ============================================================
  // Phase 4: Query
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 4: Query ===");

  std::vector<Query> queries = {
      // --- Simulation type queries (keyword-accessible) ---
      {"Find the isolated galaxy simulation with dark matter halo and stellar disk",
       "galaxy", "sim_type"},
      {"Which snapshots are from a pure gas hydrodynamic simulation",
       "gassphere", "sim_type"},
      {"Locate the cosmological cluster simulation with dark matter and bulge",
       "cluster", "sim_type"},
      {"Find the non-cosmological N-body galaxy model",
       "galaxy", "sim_type"},
      {"Which simulation is a self-gravitating gas cloud",
       "gassphere", "sim_type"},

      // --- Physics configuration queries (keyword-accessible) ---
      {"Find the gas sphere simulation with high artificial viscosity",
       "gassphere_visc", "physics"},
      {"Which simulation used a hot initial gas temperature",
       "gassphere_hot", "physics"},
      {"Find the cold gas collapse simulation",
       "gassphere_cold", "physics"},
      {"Locate the SPH simulation where cooling and star formation are disabled",
       "gassphere", "physics"},
      {"Find the gas sphere with modified Courant factor for time integration",
       "gassphere_courant", "physics"},
      {"Which simulation used warm gas with moderate initial temperature",
       "gassphere_warm", "physics"},

      // --- Semantic queries (require understanding, not just keyword match) ---
      {"Find a simulation of stellar orbits in a gravitational potential well",
       "galaxy", "semantic"},
      {"Locate output from a simulation of primordial structure formation in the cosmos",
       "cluster", "semantic"},
      {"Which run models a compressible fluid dynamics problem using particle methods",
       "gassphere", "semantic"},
      {"Find the simulation where invisible mass dominates the gravitational dynamics",
       "galaxy", "semantic"},
      {"Locate data from a simulation of thermal pressure equilibrium in a self-gravitating cloud",
       "gassphere", "semantic"},
      {"Which simulation models the hierarchical assembly of cosmic matter",
       "cluster", "semantic"},
      {"Find output from a simulation of dissipative astrophysical processes with artificial viscosity",
       "gassphere_visc", "semantic"},
      {"Locate the simulation that models centrifugal balance of orbiting material",
       "galaxy", "semantic"},
      {"Which run simulates the Jeans instability or gravitational fragmentation of gas",
       "gassphere", "semantic"},
      {"Find data showing the dynamical evolution of a multi-component stellar system",
       "galaxy", "semantic"},
      {"Locate the simulation of adiabatic compression and rarefaction in a gas cloud",
       "gassphere", "semantic"},
      {"Which simulation output represents collisionless N-body dynamics with a disk component",
       "galaxy", "semantic"},
      {"Find the run that models an astrophysical virial shock",
       "gassphere_visc", "semantic"},
      {"Locate simulations of baryon-free gravitational collapse",
       "galaxy", "semantic"},
      {"Which run simulates entropy generation through viscous dissipation",
       "gassphere_visc", "semantic"},
  };

  int top1 = 0, top3 = 0, top5 = 0;
  double total_rr = 0.0, total_ms = 0.0;

  for (size_t i = 0; i < queries.size(); i++) {
    const auto &q = queries[i];
    HLOG(kInfo, "");
    HLOG(kInfo, "  Q{:02d} [{}] \"{}\"", i + 1, q.category, q.text);

    auto t0 = Clock::now();
    auto fut = WRP_CTE_CLIENT->AsyncSemanticQuery(q.text, 10);
    fut.Wait();
    auto t1 = Clock::now();
    double ms = Ms(t1 - t0).count();
    total_ms += ms;

    auto *result = fut.get();
    std::string results_str;
    int found_rank = -1;

    for (size_t r = 0; r < result->result_tags_.size(); r++) {
      std::string matched = "?";
      for (const auto &[tn, tid] : tag_ids) {
        if (tid == result->result_tags_[r]) { matched = tn; break; }
      }
      if (!results_str.empty()) results_str += ", ";
      char sc[16]; snprintf(sc, sizeof(sc), "%.2f", result->result_scores_[r]);
      results_str += matched + " (" + sc + ")";

      if (found_rank < 0 && matched.find(q.expected_run) != std::string::npos) {
        found_rank = static_cast<int>(r);
      }
    }

    HLOG(kInfo, "    Results: [{}]", results_str);

    if (found_rank >= 0) {
      std::string matched;
      for (const auto &[tn, tid] : tag_ids) {
        if (tid == result->result_tags_[found_rank]) { matched = tn; break; }
      }
      std::string fp;
      if (tag_to_filepath.count(matched)) fp = tag_to_filepath[matched];
      bool verified = false;
      if (!fp.empty()) { std::ifstream p(fp); verified = p.good(); }

      HLOG(kInfo, "    -> FOUND at rank {} -> {} {}({:.1f} ms)",
           found_rank + 1, fp, verified ? "[VERIFIED] " : "", ms);

      if (found_rank == 0) top1++;
      if (found_rank < 3) top3++;
      if (found_rank < 5) top5++;
      total_rr += 1.0 / (found_rank + 1);
    } else {
      HLOG(kInfo, "    -> NOT FOUND ({:.1f} ms)", ms);
    }
  }

  // Report
  double n = static_cast<double>(queries.size());
  HLOG(kInfo, "");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "GADGET-2 Discovery Results");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "  Snapshots:          {}", snapshots.size());
  HLOG(kInfo, "  Run configs:        {}", unique_runs.size());
  HLOG(kInfo, "  KG entries:         {}", indexed);
  HLOG(kInfo, "  Queries:            {}", queries.size());
  HLOG(kInfo, "  Top-1 Accuracy:     {}/{} ({:.1f}%)", top1, queries.size(), 100.0*top1/n);
  HLOG(kInfo, "  Top-3 Accuracy:     {}/{} ({:.1f}%)", top3, queries.size(), 100.0*top3/n);
  HLOG(kInfo, "  Top-5 Accuracy:     {}/{} ({:.1f}%)", top5, queries.size(), 100.0*top5/n);
  HLOG(kInfo, "  MRR:                {:.3f}", total_rr / n);
  HLOG(kInfo, "  Ingest latency:     {:.1f} ms", ingest_ms);
  HLOG(kInfo, "  KG index latency:   {:.1f} ms", kg_ms);
  HLOG(kInfo, "  Avg query latency:  {:.3f} ms", total_ms / n);
  HLOG(kInfo, "========================================");

  // JSON output
  {
    std::ofstream ofs("/tmp/gadget2_discovery_results.json");
    if (ofs.is_open()) {
      ofs << "{\n"
          << "  \"snapshots\": " << snapshots.size() << ",\n"
          << "  \"runs\": " << unique_runs.size() << ",\n"
          << "  \"queries\": " << queries.size() << ",\n"
          << "  \"top1_accuracy\": " << top1/n << ",\n"
          << "  \"top3_accuracy\": " << top3/n << ",\n"
          << "  \"top5_accuracy\": " << top5/n << ",\n"
          << "  \"mrr\": " << total_rr/n << ",\n"
          << "  \"ingest_latency_ms\": " << ingest_ms << ",\n"
          << "  \"index_latency_ms\": " << kg_ms << ",\n"
          << "  \"avg_query_latency_ms\": " << total_ms/n << "\n"
          << "}\n";
      HLOG(kInfo, "  Results saved: /tmp/gadget2_discovery_results.json");
    }
  }

  return 0;
#endif
}
