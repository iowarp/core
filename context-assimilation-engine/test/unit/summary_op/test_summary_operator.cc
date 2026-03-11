/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * test_summary_operator.cc - Unit test for SummaryOperator
 *
 * This test validates the SummaryOperator by:
 * 1. Creating a CTE tag with a mock "description" blob
 * 2. Running the SummaryOperator to summarize it via LLM
 * 3. Verifying the "summary" blob was written
 *
 * Requirements:
 * - CAE_SUMMARY_ENDPOINT must be set (e.g., http://localhost:8080/v1)
 * - CAE_SUMMARY_MODEL must be set (e.g., gemma, qwen)
 * - A running inference server at the endpoint
 *
 * The test will SKIP if the environment variables are not set.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Chimaera and CAE headers
#include <chimaera/chimaera.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/core_client.h>

#ifdef WRP_CAE_ENABLE_SUMMARY_OP
#include <wrp_cae/core/factory/summary_operator.h>
#endif

// CTE headers
#include <wrp_cte/core/core_client.h>

// Logging
#include <hermes_shm/util/logging.h>

const std::string kTestTagName = "test_summary_op_tag";

int main(int argc, char* argv[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Summary Operator Unit Test");
  HLOG(kInfo, "========================================");

#ifndef WRP_CAE_ENABLE_SUMMARY_OP
  HLOG(kWarning, "Summary operator not compiled in. "
                  "Rebuild with -DWRP_CAE_ENABLE_SUMMARY_OP=ON");
  HLOG(kInfo, "TEST SKIPPED");
  return 0;
#else

  // Check if inference server is configured
  const char* endpoint_env = std::getenv("CAE_SUMMARY_ENDPOINT");
  const char* model_env = std::getenv("CAE_SUMMARY_MODEL");

  if (!endpoint_env || std::strlen(endpoint_env) == 0 ||
      !model_env || std::strlen(model_env) == 0) {
    HLOG(kWarning, "CAE_SUMMARY_ENDPOINT and/or CAE_SUMMARY_MODEL not set");
    HLOG(kWarning, "Set these to run the summary operator test:");
    HLOG(kWarning, "  export CAE_SUMMARY_ENDPOINT=http://localhost:8080/v1");
    HLOG(kWarning, "  export CAE_SUMMARY_MODEL=gemma");
    HLOG(kInfo, "TEST SKIPPED");
    return 0;
  }

  HLOG(kInfo, "Endpoint: {}", endpoint_env);
  HLOG(kInfo, "Model: {}", model_env);

  int exit_code = 0;

  try {
    // Step 1: Initialize Chimaera
    HLOG(kInfo, "[STEP 1] Initializing Chimaera...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      HLOG(kError, "Failed to initialize Chimaera");
      return 1;
    }
    HLOG(kSuccess, "Chimaera initialized");

    // Step 2: Initialize CTE client
    HLOG(kInfo, "[STEP 2] Connecting to CTE...");
    wrp_cte::core::WRP_CTE_CLIENT_INIT();
    HLOG(kSuccess, "CTE client initialized");

    // Step 3: Create a tag with a mock "description" blob
    HLOG(kInfo, "[STEP 3] Creating tag with description blob...");
    {
      wrp_cte::core::Tag tag(kTestTagName);
      std::string description = "binary<size=1048576, offset=0>";
      tag.PutBlob("description", description.c_str(), description.size());
      HLOG(kSuccess, "Description blob stored: '{}'", description);

      // Verify it was written
      chi::u64 desc_size = tag.GetBlobSize("description");
      HLOG(kInfo, "Description blob size: {} bytes", desc_size);
      if (desc_size != description.size()) {
        HLOG(kError, "Description blob size mismatch: expected {}, got {}",
             description.size(), desc_size);
        return 1;
      }
    }

    // Step 4: Run the SummaryOperator
    HLOG(kInfo, "[STEP 4] Running SummaryOperator...");
    auto cte_client = std::shared_ptr<wrp_cte::core::Client>(
        WRP_CTE_CLIENT, [](wrp_cte::core::Client*) {});
    wrp_cae::core::SummaryOperator op(cte_client);

    int rc = op.Execute(kTestTagName);
    if (rc != 0) {
      HLOG(kError, "SummaryOperator::Execute failed with code {}", rc);
      exit_code = 1;
    } else {
      HLOG(kSuccess, "SummaryOperator executed successfully");
    }

    // Step 5: Verify the summary blob
    HLOG(kInfo, "[STEP 5] Verifying summary blob...");
    {
      wrp_cte::core::Tag tag(kTestTagName);
      chi::u64 summary_size = tag.GetBlobSize("summary");

      if (summary_size == 0) {
        HLOG(kError, "Summary blob not found or empty");
        exit_code = 1;
      } else {
        std::vector<char> summary_data(summary_size);
        tag.GetBlob("summary", summary_data.data(), summary_size);
        std::string summary(summary_data.data(), summary_size);
        HLOG(kSuccess, "Summary blob ({} bytes): '{}'", summary_size, summary);
      }
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception caught: {}", e.what());
    exit_code = 1;
  }

  // Print result
  HLOG(kInfo, "========================================");
  if (exit_code == 0) {
    HLOG(kSuccess, "TEST PASSED");
  } else {
    HLOG(kError, "TEST FAILED");
  }
  HLOG(kInfo, "========================================");

  return exit_code;
#endif  // WRP_CAE_ENABLE_SUMMARY_OP
}
