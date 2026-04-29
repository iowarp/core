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

#include <chimaera/chimaera.h>
#include <wrp_cae/core/factory/summary_operator.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef WRP_CAE_ENABLE_SUMMARY_OP
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#endif

// Include wrp_cte headers after closing any wrp_cae namespace to avoid Method
// namespace collision
#include <wrp_cte/core/core_client.h>

namespace wrp_cae::core {

SummaryOperator::SummaryOperator(
    std::shared_ptr<wrp_cte::core::Client> cte_client)
    : cte_client_(cte_client) {
  const char* endpoint_env = std::getenv("CAE_SUMMARY_ENDPOINT");
  if (endpoint_env && std::strlen(endpoint_env) > 0) {
    endpoint_ = endpoint_env;
  }
  const char* model_env = std::getenv("CAE_SUMMARY_MODEL");
  if (model_env && std::strlen(model_env) > 0) {
    model_ = model_env;
  }
}

int SummaryOperator::Execute(const std::string& tag_name) {
  HLOG(kInfo, "SummaryOperator::Execute ENTRY: tag='{}'", tag_name);

  // Validate configuration
  if (endpoint_.empty() || model_.empty()) {
    HLOG(kError,
         "SummaryOperator: CAE_SUMMARY_ENDPOINT and CAE_SUMMARY_MODEL "
         "must be set");
    return -1;
  }

  // Step 1: Read the description blob
  std::string description = ReadDescriptionBlob(tag_name);
  if (description.empty()) {
    HLOG(kError, "SummaryOperator: Failed to read description blob from tag '{}'",
         tag_name);
    return -3;
  }
  HLOG(kInfo, "SummaryOperator: Read description: '{}'", description);

  // Step 2: Call LLM to summarize
  // Check if the description contains a human-written description field.
  // If yes, use the summarization prompt. If no (only raw metadata like
  // key=value pairs, numeric fields), use the interpretation prompt.
  bool has_description_text =
      description.find("description:") != std::string::npos ||
      description.find("description=") != std::string::npos ||
      description.find("long_name:") != std::string::npos;
  std::string summary = CallLlm(description, has_description_text);
  if (summary.empty()) {
    HLOG(kError, "SummaryOperator: LLM call failed for tag '{}'", tag_name);
    return -4;
  }
  HLOG(kInfo, "SummaryOperator: Generated summary: '{}'", summary);

  // Step 3: Write summary blob
  int rc = WriteSummaryBlob(tag_name, summary);
  if (rc != 0) {
    HLOG(kError, "SummaryOperator: Failed to write summary blob to tag '{}'",
         tag_name);
    return rc;
  }

  HLOG(kInfo, "SummaryOperator::Execute EXIT: Success for tag '{}'", tag_name);
  return 0;
}

std::string SummaryOperator::ReadDescriptionBlob(const std::string& tag_name) {
  try {
    wrp_cte::core::Tag tag(tag_name);

    // Get the size of the description blob
    chi::u64 blob_size = tag.GetBlobSize("description");
    if (blob_size == 0) {
      HLOG(kError, "SummaryOperator: 'description' blob not found or empty in "
                    "tag '{}'",
           tag_name);
      return "";
    }

    // Read the blob data
    std::vector<char> buffer(blob_size);
    tag.GetBlob("description", buffer.data(), blob_size);

    return std::string(buffer.data(), blob_size);
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Exception reading description blob: {}",
         e.what());
    return "";
  }
}

#ifdef WRP_CAE_ENABLE_SUMMARY_OP

// libcurl write callback
static size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb,
                                 std::string* output) {
  size_t total_size = size * nmemb;
  output->append(static_cast<char*>(contents), total_size);
  return total_size;
}

std::string SummaryOperator::CallLlm(const std::string& description,
                                      bool has_description_text) {
  // Two prompts:
  // 1. Description available: summarize the human-readable text
  // 2. Raw metadata only: interpret the metadata and generate a description
  // Env-var override lets callers swap the default HPC-dataset prompt for a
  // different one (e.g. code search needs keyword-dense multi-sentence
  // summaries rather than 4-8 word labels).
  std::string system_prompt;
  int max_tokens = 64;
  if (const char* p = std::getenv("CAE_SUMMARY_SYSTEM_PROMPT")) {
    system_prompt = p;
  } else if (has_description_text) {
    system_prompt =
        "You are a scientific data analyst. Given a dataset description from "
        "a simulation output file, summarize it in exactly 4 to 8 words. "
        "Keep domain-specific terms. Return ONLY the summary, nothing else.";
  } else {
    system_prompt =
        "You are a scientific data analyst for HPC simulations. Given raw "
        "metadata from a simulation output file, write a concise 4-8 word "
        "searchable description. Identify the simulation type and key "
        "properties. Translate numeric codes and flags to their scientific "
        "meaning. Return ONLY the description, nothing else.";
  }
  if (const char* mt = std::getenv("CAE_SUMMARY_MAX_TOKENS")) {
    int v = std::atoi(mt);
    if (v > 0 && v <= 4096) max_tokens = v;
  }

  nlohmann::json request_body;
  request_body["model"] = model_;
  request_body["messages"] = nlohmann::json::array({
      {{"role", "system"}, {"content", system_prompt}},
      {{"role", "user"}, {"content", description}},
  });
  request_body["max_tokens"] = max_tokens;
  request_body["temperature"] = 0.0;

  std::string payload = request_body.dump();
  std::string url = endpoint_ + "/chat/completions";

  HLOG(kDebug, "SummaryOperator: POST {} payload={}", url, payload);

  // Initialize curl
  CURL* curl = curl_easy_init();
  if (!curl) {
    HLOG(kError, "SummaryOperator: Failed to initialize libcurl");
    return "";
  }

  std::string response_body;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

  CURLcode res = curl_easy_perform(curl);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    HLOG(kError, "SummaryOperator: curl request failed: {}",
         curl_easy_strerror(res));
    return "";
  }

  HLOG(kDebug, "SummaryOperator: LLM response: {}", response_body);

  // Parse the response
  try {
    nlohmann::json response = nlohmann::json::parse(response_body);
    if (response.contains("choices") && !response["choices"].empty()) {
      auto& message = response["choices"][0]["message"];
      // Try "content" first, fall back to "reasoning_content" (Qwen3 thinking mode)
      std::string result;
      if (message.contains("content") && !message["content"].is_null()) {
        result = message["content"].get<std::string>();
      }
      if (result.empty() && message.contains("reasoning_content") &&
          !message["reasoning_content"].is_null()) {
        HLOG(kDebug, "SummaryOperator: Using reasoning_content (thinking mode)");
        result = message["reasoning_content"].get<std::string>();
      }
      if (!result.empty()) {
        return result;
      }
      HLOG(kError, "SummaryOperator: Both content and reasoning_content are empty");
      return "";
    }
    HLOG(kError, "SummaryOperator: No 'choices' in LLM response");
    return "";
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Failed to parse LLM response: {}",
         e.what());
    return "";
  }
}

#else  // !WRP_CAE_ENABLE_SUMMARY_OP

std::string SummaryOperator::CallLlm(const std::string& description,
                                      bool has_description_text) {
  HLOG(kError,
       "SummaryOperator: Summary operator not compiled in. "
       "Rebuild with -DWRP_CAE_ENABLE_SUMMARY_OP=ON");
  return "";
}

#endif  // WRP_CAE_ENABLE_SUMMARY_OP

int SummaryOperator::WriteSummaryBlob(const std::string& tag_name,
                                      const std::string& summary) {
  try {
    wrp_cte::core::Tag tag(tag_name);
    tag.PutBlob("summary", summary.c_str(), summary.size());
    HLOG(kDebug, "SummaryOperator: Wrote 'summary' blob ({} bytes) to tag '{}'",
         summary.size(), tag_name);
    return 0;
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Exception writing summary blob: {}",
         e.what());
    return -5;
  }
}

}  // namespace wrp_cae::core
