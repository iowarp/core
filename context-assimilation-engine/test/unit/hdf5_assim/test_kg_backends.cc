/**
 * test_kg_backends.cc — Standalone test for all KG backend implementations.
 *
 * Tests each backend with the same 20 datasets and 20 queries to compare
 * accuracy and latency. Does NOT require the CTE runtime — tests backends
 * directly via the KGBackend interface.
 *
 * Usage:
 *   ./test_kg_backends [backend_name] [config]
 *
 * Examples:
 *   ./test_kg_backends bm25
 *   ./test_kg_backends elasticsearch localhost:9200/cte_kg_bench
 *   ./test_kg_backends neo4j localhost:7474
 *   ./test_kg_backends qdrant localhost:6333
 *   ./test_kg_backends all       # Run all available backends
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <nlohmann/json.hpp>

#ifdef WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
#include <wrp_cte/core/kg_backend.h>
#include <wrp_cte/core/kg_backend_factory.h>
#include <wrp_cte/core/core_tasks.h>
#endif

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;
using json = nlohmann::json;

struct DatasetInfo {
  std::string tag_name;
  std::string description;
};

struct Query {
  std::string text;
  std::string expected_tag;
};

// Same 20 datasets from the CTE benchmark
static const std::vector<DatasetInfo> kDatasets = {
    {"weather_forecast", "temperature humidity wind speed atmospheric pressure stations weather stations recordings hourly North America"},
    {"particle_physics", "particle collision trajectories energy deposits detector proton-proton 13 TeV LHC"},
    {"ocean_temperature", "sea surface temperature satellite NOAA Pacific Ocean 0.25 degree spatial resolution daily"},
    {"genome_sequences", "genome variant sequencing SNP allele frequencies quality scores chromosome"},
    {"stock_market", "stock market trading open high low close prices volume S&P 500 minute resolution"},
    {"brain_imaging", "fMRI brain scan blood oxygen cortical regions memory tasks cognitive neuroscience"},
    {"climate_model", "climate model CO2 concentration radiative forcing temperature anomaly RCP 8.5 projections"},
    {"seismic_waves", "earthquake seismograph P-wave S-wave arrival times magnitude focal mechanism Pacific Ring Fire"},
    {"protein_structure", "protein structure X-ray crystallography atomic positions B-factors secondary structure enzyme active sites"},
    {"satellite_imagery", "satellite imagery Landsat 8 multispectral red green blue near-infrared 30 meter resolution"},
    {"wind_turbine", "wind turbine power output rotor speed blade pitch nacelle temperature vibration offshore wind farm"},
    {"drug_trials", "drug trial patient demographics dosage biomarker adverse events efficacy cardiovascular phase III"},
    {"traffic_flow", "traffic flow highway loop detectors vehicle count speed lane occupancy 5-minute intervals intersections"},
    {"soil_moisture", "soil moisture wireless sensor volumetric water content soil temperature electrical conductivity agricultural"},
    {"audio_speech", "speech recognition mel-frequency cepstral coefficients phoneme labels speaker embeddings acoustic features English"},
    {"galaxy_survey", "galaxy survey redshift photometric magnitudes morphological classifications stellar mass estimates catalog"},
    {"battery_cycling", "lithium-ion battery charge-discharge voltage current capacity impedance spectroscopy temperature cycling"},
    {"air_quality", "air quality PM2.5 PM10 ozone nitrogen dioxide sulfur dioxide meteorological urban monitoring station"},
    {"neural_network", "neural network training checkpoint model weights gradient statistics optimizer state learning rate activations"},
    {"dna_methylation", "DNA methylation Illumina EPIC BeadChip beta values detection p-values probe annotations CpG sites epigenetic"},
};

static const std::vector<Query> kQueries = {
    {"Find the weather forecast dataset", "weather_forecast"},
    {"Locate particle collision data from CERN", "particle_physics"},
    {"Where is the ocean temperature data?", "ocean_temperature"},
    {"Show me the genomics sequencing results", "genome_sequences"},
    {"Find stock trading historical prices", "stock_market"},
    {"Locate the fMRI brain scan data", "brain_imaging"},
    {"Find climate change projection data", "climate_model"},
    {"Where are the earthquake recordings?", "seismic_waves"},
    {"Locate protein crystallography coordinates", "protein_structure"},
    {"Find the Landsat satellite images", "satellite_imagery"},
    {"Show wind farm power generation data", "wind_turbine"},
    {"Find the cardiovascular drug trial results", "drug_trials"},
    {"Locate highway traffic sensor measurements", "traffic_flow"},
    {"Where is the agricultural soil sensor data?", "soil_moisture"},
    {"Find the speech recognition features dataset", "audio_speech"},
    {"Locate the galaxy redshift catalog", "galaxy_survey"},
    {"Find battery charge-discharge test data", "battery_cycling"},
    {"Where is the PM2.5 air pollution data?", "air_quality"},
    {"Find the neural network model weights", "neural_network"},
    {"Locate the DNA epigenetic methylation data", "dna_methylation"},
};

#ifdef WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
void RunBackendTest(const std::string &backend_name,
                    const std::string &config) {
  using namespace wrp_cte::core;

  std::cout << "\n========================================\n";
  std::cout << "Backend: " << backend_name << "\n";
  std::cout << "Config:  " << (config.empty() ? "(default)" : config) << "\n";
  std::cout << "========================================\n";

  auto backend = CreateKGBackend(backend_name);
  if (!backend) {
    std::cout << "  SKIP: Backend not available\n";
    return;
  }

  // Init
  auto t0 = Clock::now();
  backend->Init(config);
  auto t1 = Clock::now();
  std::cout << "  Init: " << Ms(t1 - t0).count() << " ms\n";

  // Assign fake TagIds (sequential)
  std::unordered_map<std::string, TagId> tag_ids;
  for (size_t i = 0; i < kDatasets.size(); i++) {
    TagId tid;
    tid.major_ = 512;
    tid.minor_ = static_cast<chi::u32>(i + 1);
    tag_ids[kDatasets[i].tag_name] = tid;
  }

  // Index
  auto t_idx_start = Clock::now();
  for (const auto &ds : kDatasets) {
    backend->Add(tag_ids[ds.tag_name], ds.description);
  }
  auto t_idx_end = Clock::now();
  double index_ms = Ms(t_idx_end - t_idx_start).count();
  std::cout << "  Index: " << kDatasets.size() << " entries in "
            << index_ms << " ms\n";

  // Query
  int top1 = 0, top3 = 0, top5 = 0;
  double total_rr = 0;
  double total_query_ms = 0;

  for (const auto &q : kQueries) {
    auto tq0 = Clock::now();
    auto results = backend->Search(q.text, 5);
    auto tq1 = Clock::now();
    double qms = Ms(tq1 - tq0).count();
    total_query_ms += qms;

    TagId expected = tag_ids.count(q.expected_tag)
        ? tag_ids[q.expected_tag] : TagId();

    int found_rank = -1;
    for (size_t r = 0; r < results.size(); r++) {
      if (results[r].key == expected) {
        found_rank = static_cast<int>(r);
        break;
      }
    }

    if (found_rank >= 0) {
      if (found_rank == 0) top1++;
      if (found_rank < 3) top3++;
      if (found_rank < 5) top5++;
      total_rr += 1.0 / (found_rank + 1);
    }
  }

  double n = static_cast<double>(kQueries.size());
  double mrr = total_rr / n;

  std::cout << "\n  Results:\n";
  std::cout << "    Top-1: " << top1 << "/" << kQueries.size()
            << " (" << 100.0 * top1 / n << "%)\n";
  std::cout << "    Top-3: " << top3 << "/" << kQueries.size()
            << " (" << 100.0 * top3 / n << "%)\n";
  std::cout << "    Top-5: " << top5 << "/" << kQueries.size()
            << " (" << 100.0 * top5 / n << "%)\n";
  std::cout << "    MRR:   " << mrr << "\n";
  std::cout << "    Avg query: " << total_query_ms / n << " ms\n";
  std::cout << "    Index time: " << index_ms << " ms\n";

  // Write JSON result
  json result = {
      {"backend", backend_name},
      {"num_datasets", kDatasets.size()},
      {"num_queries", kQueries.size()},
      {"index_time_ms", index_ms},
      {"results", {
          {"top1_correct", top1},
          {"top3_correct", top3},
          {"top5_correct", top5},
          {"mrr", mrr},
          {"avg_query_ms", total_query_ms / n}
      }},
      {"config", config}
  };
  std::string out_path = "/tmp/cte_kg_" + backend_name + "_results.json";
  std::ofstream ofs(out_path);
  if (ofs.is_open()) {
    ofs << result.dump(2) << "\n";
    std::cout << "    Saved: " << out_path << "\n";
  }

  // Cleanup
  backend->Destroy();
}
#endif

int main(int argc, char *argv[]) {
#ifndef WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
  std::cerr << "Knowledge graph not compiled. "
            << "Rebuild with -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON\n";
  return 1;
#else
  std::string backend = (argc > 1) ? argv[1] : "bm25";
  std::string config = (argc > 2) ? argv[2] : "";

  if (backend == "all") {
    RunBackendTest("bm25", "");
#ifdef WRP_CTE_KG_ELASTICSEARCH
    RunBackendTest("elasticsearch", "localhost:9200/cte_kg_bench");
#endif
#ifdef WRP_CTE_KG_NEO4J
    RunBackendTest("neo4j", "localhost:7474");
#endif
#ifdef WRP_CTE_KG_QDRANT
    RunBackendTest("qdrant", "localhost:6333");
#endif
  } else {
    RunBackendTest(backend, config);
  }

  return 0;
#endif
}
