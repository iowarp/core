/*
 * bench_hdf5_discovery.cc - HDF5 Dataset Discovery Benchmark
 *
 * Tests CTE Knowledge Graph's ability to locate datasets by natural language:
 *   1. INGEST:   Store dataset descriptions as CTE tags
 *   2. INDEX KG: Feed descriptions into the knowledge graph (BM25)
 *   3. QUERY:    Natural language queries -> SemanticQuery -> check accuracy
 *   4. REPORT:   Top-k accuracy, MRR, latency
 *
 * Reads datasets and queries from a manifest.json file.
 * Generate with: python3 gen_hdf5_datasets.py [NUM] [OUTPUT_DIR]
 * Then set HDF5_BENCH_DIR to the output directory.
 *
 * Requirements:
 * - Built with -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON
 * - CTE runtime running (CHI_SERVER_CONF set)
 * - HDF5_BENCH_DIR env var pointing to manifest.json directory
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

#include <nlohmann/json.hpp>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;
using json = nlohmann::json;

// ============================================================
// Dataset and query structures
// ============================================================
struct DatasetInfo {
  std::string tag_name;
  std::string filename;
  std::string description;
};

struct Query {
  std::string text;
  std::string expected_tag;
  std::string category;
};

static const std::string kTagPrefix = "hdf5_bench/";

// ============================================================
// Fallback hardcoded data (used when no manifest.json found)
// ============================================================
static const std::vector<DatasetInfo> kFallbackDatasets = {
    {"weather_forecast", "weather_forecast.h5",
     "Global weather forecast data including temperature, humidity, wind speed, "
     "and atmospheric pressure measurements from 500 weather stations across "
     "North America recorded hourly"},
    {"particle_physics", "particle_physics.h5",
     "High-energy particle collision events from the Large Hadron Collider "
     "including particle trajectories, energy deposits, and detector hit "
     "patterns for proton-proton collisions at 13 TeV"},
    {"ocean_temperature", "ocean_temperature.h5",
     "Sea surface temperature measurements from NOAA satellite observations "
     "covering the Pacific Ocean basin with 0.25 degree spatial resolution "
     "daily averages"},
    {"genome_sequences", "genome_sequences.h5",
     "Human genome variant call data from whole genome sequencing including "
     "SNP positions, allele frequencies, and quality scores across "
     "chromosome 1-22"},
    {"stock_market", "stock_market.h5",
     "Historical stock market trading data including open, high, low, close "
     "prices and volume for S&P 500 companies from 2010 to 2024 at minute "
     "resolution"},
    {"brain_imaging", "brain_imaging.h5",
     "Functional MRI brain scan data from cognitive neuroscience experiments "
     "showing blood oxygen level dependent signals across cortical regions "
     "during memory tasks"},
    {"climate_model", "climate_model.h5",
     "Earth system climate model output with global CO2 concentration, "
     "radiative forcing, and temperature anomaly projections under RCP 8.5 "
     "scenario from 2020 to 2100"},
    {"seismic_waves", "seismic_waves.h5",
     "Earthquake seismograph recordings from the Pacific Ring of Fire "
     "including P-wave and S-wave arrival times, magnitude estimates, and "
     "focal mechanism solutions"},
    {"protein_structure", "protein_structure.h5",
     "3D protein structure coordinates from X-ray crystallography including "
     "atomic positions, B-factors, and secondary structure annotations for "
     "enzyme active sites"},
    {"satellite_imagery", "satellite_imagery.h5",
     "Multispectral satellite imagery from Landsat 8 with red, green, blue, "
     "near-infrared, and shortwave infrared bands at 30 meter ground "
     "resolution"},
    {"wind_turbine", "wind_turbine.h5",
     "Wind turbine performance metrics including power output, rotor speed, "
     "blade pitch angle, nacelle temperature, and vibration sensor data from "
     "an offshore wind farm"},
    {"drug_trials", "drug_trials.h5",
     "Clinical drug trial results with patient demographics, dosage levels, "
     "biomarker measurements, adverse events, and efficacy endpoints for a "
     "phase III cardiovascular study"},
    {"traffic_flow", "traffic_flow.h5",
     "Urban traffic flow sensor data from highway loop detectors measuring "
     "vehicle count, average speed, and lane occupancy at 5-minute intervals "
     "across 200 intersections"},
    {"soil_moisture", "soil_moisture.h5",
     "Agricultural soil moisture measurements from wireless sensor networks "
     "at multiple depths including volumetric water content, soil temperature, "
     "and electrical conductivity"},
    {"audio_speech", "audio_speech.h5",
     "Speech recognition training data with mel-frequency cepstral "
     "coefficients, phoneme labels, speaker embeddings, and acoustic features "
     "extracted from 1000 hours of English speech"},
    {"galaxy_survey", "galaxy_survey.h5",
     "Astronomical galaxy survey catalog with redshift measurements, "
     "photometric magnitudes, morphological classifications, and stellar mass "
     "estimates for 2 million galaxies"},
    {"battery_cycling", "battery_cycling.h5",
     "Lithium-ion battery charge-discharge cycling data including voltage, "
     "current, capacity, impedance spectroscopy, and temperature measurements "
     "over 500 cycles"},
    {"air_quality", "air_quality.h5",
     "Urban air quality monitoring station data with particulate matter PM2.5 "
     "and PM10, ozone, nitrogen dioxide, sulfur dioxide concentrations, and "
     "meteorological conditions"},
    {"neural_network", "neural_network.h5",
     "Deep neural network training checkpoint with model weights, gradient "
     "statistics, optimizer state, learning rate schedule, and per-layer "
     "activation distributions"},
    {"dna_methylation", "dna_methylation.h5",
     "Epigenetic DNA methylation array data from Illumina EPIC BeadChip with "
     "beta values, detection p-values, and probe annotations across 850,000 "
     "CpG sites"},
};

static const std::vector<Query> kFallbackQueries = {
    {"Find the weather forecast dataset", "weather_forecast", "exact"},
    {"Locate particle collision data from CERN", "particle_physics", "synonym"},
    {"Where is the ocean temperature data?", "ocean_temperature", "exact"},
    {"Show me the genomics sequencing results", "genome_sequences", "synonym"},
    {"Find stock trading historical prices", "stock_market", "synonym"},
    {"Locate the fMRI brain scan data", "brain_imaging", "synonym"},
    {"Find climate change projection data", "climate_model", "synonym"},
    {"Where are the earthquake recordings?", "seismic_waves", "synonym"},
    {"Locate protein crystallography coordinates", "protein_structure", "synonym"},
    {"Find the Landsat satellite images", "satellite_imagery", "synonym"},
    {"Show wind farm power generation data", "wind_turbine", "synonym"},
    {"Find the cardiovascular drug trial results", "drug_trials", "synonym"},
    {"Locate highway traffic sensor measurements", "traffic_flow", "synonym"},
    {"Where is the agricultural soil sensor data?", "soil_moisture", "synonym"},
    {"Find the speech recognition features dataset", "audio_speech", "synonym"},
    {"Locate the galaxy redshift catalog", "galaxy_survey", "synonym"},
    {"Find battery charge-discharge test data", "battery_cycling", "synonym"},
    {"Where is the PM2.5 air pollution data?", "air_quality", "synonym"},
    {"Find the neural network model weights", "neural_network", "synonym"},
    {"Locate the DNA epigenetic methylation data", "dna_methylation", "synonym"},
};

// ============================================================
// Main benchmark
// ============================================================
int main() {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "HDF5 Dataset Discovery Benchmark");
  HLOG(kInfo, "========================================");

#ifndef WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
  HLOG(kWarning, "Knowledge graph not compiled. "
                  "Rebuild with -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON");
  return 0;
#else

  // Load datasets and queries from manifest.json or fallback
  std::vector<DatasetInfo> datasets;
  std::vector<Query> queries;

  const char* bench_dir_env = std::getenv("HDF5_BENCH_DIR");
  std::string bench_dir = bench_dir_env ? bench_dir_env : "";
  std::string manifest_path = bench_dir.empty() ? ""
      : bench_dir + "/manifest.json";

  if (!manifest_path.empty()) {
    std::ifstream ifs(manifest_path);
    if (ifs.is_open()) {
      json manifest = json::parse(ifs);
      for (const auto& ds : manifest["datasets"]) {
        datasets.push_back({
            ds["tag_name"].get<std::string>(),
            ds.value("filename", ""),
            ds["description"].get<std::string>()});
      }
      for (const auto& q : manifest["queries"]) {
        queries.push_back({
            q["text"].get<std::string>(),
            q["expected_tag"].get<std::string>(),
            q.value("category", "synonym")});
      }
      HLOG(kInfo, "Loaded manifest: {} datasets, {} queries from {}",
           datasets.size(), queries.size(), manifest_path);
    } else {
      HLOG(kWarning, "Cannot open manifest: {} — using fallback data",
           manifest_path);
    }
  }

  if (datasets.empty()) {
    HLOG(kInfo, "Using built-in 20 datasets (set HDF5_BENCH_DIR for more)");
    datasets = kFallbackDatasets;
    queries = kFallbackQueries;
  }

  HLOG(kInfo, "Datasets: {}, Queries: {}", datasets.size(), queries.size());

  // Initialize CTE
  bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!ok) {
    HLOG(kError, "Failed to initialize Chimaera");
    return 1;
  }
  wrp_cte::core::WRP_CTE_CLIENT_INIT();

  // Map tag_name -> TagId for lookup
  std::unordered_map<std::string, wrp_cte::core::TagId> tag_ids;

  // ============================================================
  // Phase 1: Ingest to CTE (store descriptions as blobs)
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 1: Ingest to CTE ===");

  auto t_ingest_start = Clock::now();
  for (const auto& ds : datasets) {
    std::string full_tag = kTagPrefix + ds.tag_name;

    wrp_cte::core::Tag tag(full_tag);

    // Store description as a blob (best-effort — not required for KG benchmark)
    try {
      tag.PutBlob("description", ds.description.c_str(), ds.description.size());
    } catch (const std::exception& e) {
      HLOG(kWarning, "  PutBlob skipped for {}: {}", full_tag, e.what());
    }

    auto tid = tag.GetTagId();
    tag_ids[ds.tag_name] = tid;
    HLOG(kInfo, "  {}: desc={} bytes, tag_id={}.{}", full_tag,
         ds.description.size(), tid.major_, tid.minor_);
  }
  auto t_ingest_end = Clock::now();
  double ingest_ms = Ms(t_ingest_end - t_ingest_start).count();
  HLOG(kInfo, "  Total ingest: {:.1f} ms", ingest_ms);

  // ============================================================
  // Phase 2: Index Knowledge Graph
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 2: Index Knowledge Graph ===");

  auto t_kg_start = Clock::now();
  for (const auto& ds : datasets) {
    auto tag_id = tag_ids[ds.tag_name];
    std::string full_tag = kTagPrefix + ds.tag_name;
    auto fut = WRP_CTE_CLIENT->AsyncUpdateKnowledgeGraph(
        tag_id, full_tag, ds.description);
    fut.Wait();
  }
  auto t_kg_end = Clock::now();
  double kg_ms = Ms(t_kg_end - t_kg_start).count();
  HLOG(kInfo, "  {} descriptions indexed ({:.1f} ms)", datasets.size(), kg_ms);

  // ============================================================
  // Phase 2.5: Sync Global IDF
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 2.5: Sync Global IDF ===");

  auto sync_fut = WRP_CTE_CLIENT->AsyncSyncKnowledgeGraph();
  sync_fut.Wait();
  auto* sync_result = sync_fut.get();

  auto dist_fut = WRP_CTE_CLIENT->AsyncSyncKnowledgeGraph(
      chi::PoolQuery::Broadcast(), true,
      sync_result->global_n_, sync_result->global_total_terms_,
      sync_result->global_df_);
  dist_fut.Wait();
  HLOG(kInfo, "  Global IDF synced: N={}, unique_terms={}",
       sync_result->global_n_, sync_result->global_df_.size());

  // ============================================================
  // Phase 3: Query
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 3: Query ({} questions) ===", queries.size());

  int top1_correct = 0;
  int top3_correct = 0;
  int top5_correct = 0;
  double total_rr = 0.0;  // Sum of reciprocal ranks
  double total_query_ms = 0.0;

  for (size_t i = 0; i < queries.size(); i++) {
    const auto& q = queries[i];
    HLOG(kInfo, "");
    HLOG(kInfo, "  Q{:02d} [{}] \"{}\"", i + 1, q.category, q.text);

    // SemanticQuery
    auto t0 = Clock::now();
    auto fut = WRP_CTE_CLIENT->AsyncSemanticQuery(q.text, 5);
    fut.Wait();
    auto t1 = Clock::now();
    double query_ms = Ms(t1 - t0).count();
    total_query_ms += query_ms;

    auto* result = fut.get();
    std::vector<wrp_cte::core::TagId> result_tags = result->result_tags_;
    std::vector<float> result_scores = result->result_scores_;

    // Find expected TagId
    wrp_cte::core::TagId expected_tid;
    if (tag_ids.count(q.expected_tag)) {
      expected_tid = tag_ids[q.expected_tag];
    }

    // Log results
    std::string results_str;
    int found_rank = -1;
    for (size_t r = 0; r < result_tags.size(); r++) {
      // Find dataset name for this TagId
      std::string name = "?";
      for (const auto& [tname, tid] : tag_ids) {
        if (tid == result_tags[r]) {
          name = tname;
          break;
        }
      }
      if (!results_str.empty()) results_str += ", ";
      char score_str[16];
      snprintf(score_str, sizeof(score_str), "%.2f", result_scores[r]);
      results_str += name + " (" + score_str + ")";

      if (result_tags[r] == expected_tid && found_rank < 0) {
        found_rank = static_cast<int>(r);
      }
    }

    HLOG(kInfo, "    Results: [{}]", results_str);

    if (found_rank >= 0) {
      HLOG(kInfo, "    Expected: {} -> FOUND at rank {} ({:.1f} ms)",
           q.expected_tag, found_rank + 1, query_ms);
      if (found_rank == 0) top1_correct++;
      if (found_rank < 3) top3_correct++;
      if (found_rank < 5) top5_correct++;
      total_rr += 1.0 / (found_rank + 1);
    } else {
      HLOG(kInfo, "    Expected: {} -> NOT FOUND ({:.1f} ms)",
           q.expected_tag, query_ms);
    }
  }

  // ============================================================
  // Phase 4: Report
  // ============================================================
  double n = static_cast<double>(queries.size());
  double mrr = total_rr / n;

  HLOG(kInfo, "");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Results (CTE)");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "  Top-1 Accuracy: {}/{} ({:.1f}%)",
       top1_correct, queries.size(),
       100.0 * top1_correct / n);
  HLOG(kInfo, "  Top-3 Accuracy: {}/{} ({:.1f}%)",
       top3_correct, queries.size(),
       100.0 * top3_correct / n);
  HLOG(kInfo, "  Top-5 Accuracy: {}/{} ({:.1f}%)",
       top5_correct, queries.size(),
       100.0 * top5_correct / n);
  HLOG(kInfo, "  Mean Reciprocal Rank: {:.3f}", mrr);
  HLOG(kInfo, "  Ingest latency:      {:.1f} ms", ingest_ms);
  HLOG(kInfo, "  KG index latency:    {:.1f} ms", kg_ms);
  HLOG(kInfo, "  Avg query latency:   {:.3f} ms", total_query_ms / n);
  HLOG(kInfo, "  Total queries:       {}", queries.size());
  HLOG(kInfo, "========================================");

  // Write results JSON
  {
    std::string out_path = "/tmp/cte_bench_results.json";
    std::ofstream ofs(out_path);
    if (ofs.is_open()) {
      ofs << "{\n"
          << "  \"provider\": \"cte\",\n"
          << "  \"datasets\": " << datasets.size() << ",\n"
          << "  \"queries\": " << queries.size() << ",\n"
          << "  \"top1_accuracy\": " << top1_correct / n << ",\n"
          << "  \"top3_accuracy\": " << top3_correct / n << ",\n"
          << "  \"top5_accuracy\": " << top5_correct / n << ",\n"
          << "  \"mrr\": " << mrr << ",\n"
          << "  \"ingest_latency_ms\": " << ingest_ms << ",\n"
          << "  \"index_latency_ms\": " << kg_ms << ",\n"
          << "  \"avg_query_latency_ms\": " << total_query_ms / n << "\n"
          << "}\n";
      HLOG(kInfo, "\n  Results saved: {}", out_path);
    }
  }

  return 0;
#endif  // WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
}
