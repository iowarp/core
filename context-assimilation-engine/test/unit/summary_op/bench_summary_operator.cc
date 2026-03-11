/*
 * bench_summary_operator.cc - Performance benchmark for SummaryOperator
 *
 * Measures:
 * 1. CTE blob I/O baseline (PutBlob + GetBlob without LLM)
 * 2. Full SummaryOperator pipeline (blob read + LLM call + blob write)
 * 3. Breakdown: ReadDescriptionBlob, CallLlm, WriteSummaryBlob
 *
 * Requirements:
 * - CAE_SUMMARY_ENDPOINT and CAE_SUMMARY_MODEL must be set
 * - A running inference server at the endpoint
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <chimaera/chimaera.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/core_client.h>

#ifdef WRP_CAE_ENABLE_SUMMARY_OP
#include <wrp_cae/core/factory/summary_operator.h>
#endif

#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static const int NUM_ITERATIONS = 5;

// Sample descriptions of varying complexity
static const std::vector<std::string> kDescriptions = {
    "binary<size=1048576, offset=0>",
    "tensor<float32, 256, 128>",
    "binary<size=4294967296, offset=1048576>",
    "tensor<float64, 1024, 1024, 3>",
    "binary<size=512, offset=0, checksum=a1b2c3d4>",
};

int main(int argc, char* argv[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Summary Operator Performance Benchmark");
  HLOG(kInfo, "========================================");

#ifndef WRP_CAE_ENABLE_SUMMARY_OP
  HLOG(kWarning, "Summary operator not compiled in. "
                  "Rebuild with -DWRP_CAE_ENABLE_SUMMARY_OP=ON");
  return 0;
#else

  const char* endpoint_env = std::getenv("CAE_SUMMARY_ENDPOINT");
  const char* model_env = std::getenv("CAE_SUMMARY_MODEL");

  if (!endpoint_env || std::strlen(endpoint_env) == 0 ||
      !model_env || std::strlen(model_env) == 0) {
    HLOG(kWarning, "CAE_SUMMARY_ENDPOINT and/or CAE_SUMMARY_MODEL not set");
    return 0;
  }

  HLOG(kInfo, "Endpoint: {}", endpoint_env);
  HLOG(kInfo, "Model: {}", model_env);
  HLOG(kInfo, "Iterations per description: {}", NUM_ITERATIONS);
  HLOG(kInfo, "Number of descriptions: {}", kDescriptions.size());

  try {
    // Initialize Chimaera + CTE
    HLOG(kInfo, "Initializing runtime...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      HLOG(kError, "Failed to initialize Chimaera");
      return 1;
    }
    wrp_cte::core::WRP_CTE_CLIENT_INIT();

    // ================================================================
    // Benchmark 1: CTE Blob I/O Baseline (no LLM)
    // ================================================================
    HLOG(kInfo, "");
    HLOG(kInfo, "--- Benchmark 1: CTE Blob I/O Baseline ---");

    double total_put_ms = 0.0;
    double total_get_ms = 0.0;
    int blob_ops = 0;

    for (size_t d = 0; d < kDescriptions.size(); d++) {
      const auto& desc = kDescriptions[d];
      std::string tag_name = "bench_blob_io_" + std::to_string(d);

      for (int i = 0; i < NUM_ITERATIONS; i++) {
        // PutBlob timing
        auto t0 = Clock::now();
        {
          wrp_cte::core::Tag tag(tag_name);
          tag.PutBlob("description", desc.c_str(), desc.size());
        }
        auto t1 = Clock::now();
        double put_ms = Ms(t1 - t0).count();
        total_put_ms += put_ms;

        // GetBlob timing
        auto t2 = Clock::now();
        {
          wrp_cte::core::Tag tag(tag_name);
          chi::u64 sz = tag.GetBlobSize("description");
          std::vector<char> buf(sz);
          tag.GetBlob("description", buf.data(), sz);
        }
        auto t3 = Clock::now();
        double get_ms = Ms(t3 - t2).count();
        total_get_ms += get_ms;

        blob_ops++;
      }
    }

    double avg_put = total_put_ms / blob_ops;
    double avg_get = total_get_ms / blob_ops;
    HLOG(kInfo, "CTE PutBlob avg: {:.3f} ms ({} ops)", avg_put, blob_ops);
    HLOG(kInfo, "CTE GetBlob avg: {:.3f} ms ({} ops)", avg_get, blob_ops);
    HLOG(kInfo, "CTE round-trip avg: {:.3f} ms", avg_put + avg_get);

    // ================================================================
    // Benchmark 2: Full SummaryOperator Pipeline
    // ================================================================
    HLOG(kInfo, "");
    HLOG(kInfo, "--- Benchmark 2: Full SummaryOperator.Execute() ---");

    auto cte_client = std::shared_ptr<wrp_cte::core::Client>(
        WRP_CTE_CLIENT, [](wrp_cte::core::Client*) {});
    wrp_cae::core::SummaryOperator op(cte_client);

    double total_execute_ms = 0.0;
    int execute_ops = 0;
    double min_execute = 1e9, max_execute = 0.0;

    for (size_t d = 0; d < kDescriptions.size(); d++) {
      const auto& desc = kDescriptions[d];
      std::string tag_name = "bench_summary_" + std::to_string(d);

      // Pre-populate description blob
      {
        wrp_cte::core::Tag tag(tag_name);
        tag.PutBlob("description", desc.c_str(), desc.size());
      }

      HLOG(kInfo, "Description[{}]: '{}'", d, desc);

      for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto t0 = Clock::now();
        int rc = op.Execute(tag_name);
        auto t1 = Clock::now();
        double exec_ms = Ms(t1 - t0).count();

        if (rc == 0) {
          total_execute_ms += exec_ms;
          execute_ops++;
          if (exec_ms < min_execute) min_execute = exec_ms;
          if (exec_ms > max_execute) max_execute = exec_ms;

          // Read back the summary
          wrp_cte::core::Tag tag(tag_name);
          chi::u64 sz = tag.GetBlobSize("summary");
          std::vector<char> buf(sz);
          tag.GetBlob("summary", buf.data(), sz);
          std::string summary(buf.data(), sz);
          HLOG(kInfo, "  iter={}: {:.1f} ms -> '{}'", i, exec_ms, summary);
        } else {
          HLOG(kError, "  iter={}: FAILED (rc={})", i, rc);
        }
      }
    }

    if (execute_ops > 0) {
      double avg_execute = total_execute_ms / execute_ops;
      HLOG(kInfo, "");
      HLOG(kInfo, "========================================");
      HLOG(kInfo, "RESULTS SUMMARY");
      HLOG(kInfo, "========================================");
      HLOG(kInfo, "CTE blob I/O (PutBlob):     {:.3f} ms avg", avg_put);
      HLOG(kInfo, "CTE blob I/O (GetBlob):     {:.3f} ms avg", avg_get);
      HLOG(kInfo, "CTE round-trip:             {:.3f} ms avg", avg_put + avg_get);
      HLOG(kInfo, "");
      HLOG(kInfo, "SummaryOperator.Execute():  {:.1f} ms avg", avg_execute);
      HLOG(kInfo, "SummaryOperator min:        {:.1f} ms", min_execute);
      HLOG(kInfo, "SummaryOperator max:        {:.1f} ms", max_execute);
      HLOG(kInfo, "LLM overhead:               {:.1f} ms ({:.1f}x blob I/O)",
           avg_execute - (avg_put + avg_get),
           avg_execute / (avg_put + avg_get));
      HLOG(kInfo, "Successful runs:            {}/{}", execute_ops,
           (int)(kDescriptions.size() * NUM_ITERATIONS));
      HLOG(kInfo, "========================================");
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception: {}", e.what());
    return 1;
  }

  return 0;
#endif
}
