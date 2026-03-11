/*
 * bench_cte_memory.cc - CTE Agent Memory Benchmark
 *
 * Compares two CTE-based approaches for long-term agent memory:
 *
 *   1. CTE Raw:     Store sessions -> Retrieve ALL raw sessions -> LLM answers
 *   2. CTE+Summary: Store sessions -> Summarize via LLM -> Store summaries
 *                   -> Retrieve summaries -> LLM answers
 *
 * Both approaches use CTE for storage. Measures accuracy, latency, and
 * context reduction.
 *
 * Requirements:
 * - CAE_SUMMARY_ENDPOINT and CAE_SUMMARY_MODEL must be set
 * - A running inference server at the endpoint
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

#ifdef WRP_CAE_ENABLE_SUMMARY_OP
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#endif

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// ============================================================
// Test data: multi-session agent conversations
// ============================================================
struct Session {
  std::string id;
  std::string date;
  std::string content;  // Full conversation text
};

struct Question {
  std::string text;
  std::string expected;
  std::string category;
};

static const std::vector<Session> kSessions = {
    {"s1", "2024-06-10",
     "User: I need to set up a Kubernetes cluster for our new microservices "
     "project.\n"
     "Agent: What cloud provider are you using?\n"
     "User: AWS. We have 3 services: auth-service on port 8080, data-pipeline "
     "on port 9090, and frontend on port 3000.\n"
     "Agent: I will configure EKS with those three deployments.\n"
     "User: Use t3.large instances. Budget is $2000/month max."},

    {"s2", "2024-06-15",
     "User: The auth-service keeps crashing. Getting OOM errors.\n"
     "Agent: What memory limit did you set?\n"
     "User: 256MB. But it needs more during peak login hours, 9-10 AM.\n"
     "Agent: I will increase it to 512MB and add horizontal pod autoscaling. "
     "Also recommend adding Redis cache.\n"
     "User: Good idea. Add Redis with 1GB memory limit."},

    {"s3", "2024-06-22",
     "User: We need to add monitoring. What do you recommend?\n"
     "Agent: Prometheus + Grafana. I can set up dashboards for CPU, memory, "
     "and request latency.\n"
     "User: Set up alerts if any pod restarts more than 3 times in 5 "
     "minutes.\n"
     "Agent: I will also add PagerDuty integration. What is the on-call "
     "rotation?\n"
     "User: Team lead Sarah gets alerts Mon-Thu, I handle Fri-Sun. My "
     "PagerDuty is user-mike-ops."},

    {"s4", "2024-07-01",
     "User: The data-pipeline processed 1 million records yesterday without "
     "issues!\n"
     "Agent: The autoscaling is working well. Peak was 3 replicas.\n"
     "User: I want to add a new service: recommendation-engine. It needs GPU "
     "support.\n"
     "Agent: For GPU on EKS, you need p3.2xlarge instances, about $3/hour.\n"
     "User: Only run it during business hours 8AM-6PM ET to save costs. Use "
     "a CronJob to scale."},

    {"s5", "2024-07-10",
     "User: Security audit found we need to encrypt all inter-service "
     "communication.\n"
     "Agent: I will set up Istio service mesh with mTLS.\n"
     "User: Also the auditor wants all container images scanned. We use "
     "ECR.\n"
     "Agent: I will enable ECR image scanning and add an admission "
     "controller to block images with critical CVEs.\n"
     "User: The compliance deadline is August 15th."},

    {"s6", "2024-07-18",
     "User: The recommendation-engine GPU costs are higher than expected. "
     "$4500 last week.\n"
     "Agent: The CronJob scale-down is not working - pods stay up overnight.\n"
     "User: Can we use spot instances for the GPU nodes?\n"
     "Agent: Yes, p3.2xlarge spot instances are about 70% cheaper.\n"
     "User: Do it. Also Sarah is leaving - remove her from PagerDuty. New "
     "team lead is James, pagerduty ID james-infra."},
};

static const std::vector<Question> kQuestions = {
    {"What port does the auth-service run on?", "8080", "fact-recall"},
    {"What changes were made to the auth-service after it started crashing?",
     "Memory limit increased from 256MB to 512MB, horizontal pod autoscaling "
     "added, Redis cache with 1GB limit added",
     "cross-session"},
    {"Who is currently on-call for weekday alerts via PagerDuty?",
     "James (pagerduty ID james-infra), replacing Sarah who left",
     "temporal"},
    {"What is the monthly cloud budget and what cost issues have come up?",
     "Budget is $2000/month. GPU costs were $4500/week due to CronJob not "
     "scaling down. Fix: spot instances (70% cheaper)",
     "quantitative"},
    {"What database does the data-pipeline service use?",
     "Not specified in the conversations", "unanswerable"},
    {"List all services running in the cluster.",
     "auth-service (port 8080, 512MB, HPA, Redis), data-pipeline (port 9090, "
     "1M records, 3 replicas), frontend (port 3000), recommendation-engine "
     "(GPU, p3.2xlarge spot, 8AM-6PM ET)",
     "aggregation"},
    {"What security measures need to be in place and by when?",
     "By August 15th: Istio mTLS, ECR image scanning, admission controller "
     "blocking critical CVEs",
     "deadline"},
};

#ifdef WRP_CAE_ENABLE_SUMMARY_OP

// ============================================================
// LLM call helper (reuses curl pattern from SummaryOperator)
// ============================================================
static size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb,
                                 std::string* output) {
  output->append(static_cast<char*>(contents), size * nmemb);
  return size * nmemb;
}

static std::string CallLlm(const std::string& endpoint,
                            const std::string& model,
                            const std::string& system_prompt,
                            const std::string& user_prompt,
                            int max_tokens = 256) {
  nlohmann::json body;
  body["model"] = model;
  body["messages"] = nlohmann::json::array();
  if (!system_prompt.empty()) {
    body["messages"].push_back({{"role", "system"}, {"content", system_prompt}});
  }
  body["messages"].push_back({{"role", "user"}, {"content", user_prompt}});
  body["max_tokens"] = max_tokens;
  body["temperature"] = 0.0;

  std::string payload = body.dump();
  std::string url = endpoint + "/chat/completions";

  CURL* curl = curl_easy_init();
  if (!curl) return "";

  std::string response_body;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

  CURLcode res = curl_easy_perform(curl);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    HLOG(kError, "LLM curl error: {}", curl_easy_strerror(res));
    return "";
  }

  try {
    auto json = nlohmann::json::parse(response_body);
    if (json.contains("choices") && !json["choices"].empty()) {
      auto& msg = json["choices"][0]["message"];
      std::string result;
      if (msg.contains("content") && !msg["content"].is_null())
        result = msg["content"].get<std::string>();
      if (result.empty() && msg.contains("reasoning_content") &&
          !msg["reasoning_content"].is_null())
        result = msg["reasoning_content"].get<std::string>();
      return result;
    }
  } catch (...) {}

  HLOG(kError, "Failed to parse LLM response");
  return "";
}

// Judge an answer: returns true if correct
static bool JudgeAnswer(const std::string& endpoint,
                         const std::string& model,
                         const Question& q,
                         const std::string& answer) {
  std::string prompt =
      "Question: " + q.text + "\n\nExpected answer: " + q.expected +
      "\n\nActual answer: " + answer + "\n\nVerdict:";
  std::string sys =
      "You are a judge. Compare the actual answer to the expected answer. "
      "Reply with exactly one word: CORRECT if the actual answer captures the "
      "key facts, or INCORRECT if it misses key facts or is wrong.";

  std::string verdict = CallLlm(endpoint, model, sys, prompt, 16);
  // Check if verdict contains "CORRECT" (case-insensitive)
  for (auto& c : verdict) c = toupper(c);
  return verdict.find("CORRECT") != std::string::npos &&
         verdict.find("INCORRECT") == std::string::npos;
}

#endif  // WRP_CAE_ENABLE_SUMMARY_OP

int main() {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "CTE Agent Memory Benchmark");
  HLOG(kInfo, "========================================");

#ifndef WRP_CAE_ENABLE_SUMMARY_OP
  HLOG(kWarning, "Summary operator not compiled. "
                  "Rebuild with -DWRP_CAE_ENABLE_SUMMARY_OP=ON");
  return 0;
#else

  const char* endpoint_env = std::getenv("CAE_SUMMARY_ENDPOINT");
  const char* model_env = std::getenv("CAE_SUMMARY_MODEL");
  if (!endpoint_env || !model_env ||
      std::strlen(endpoint_env) == 0 || std::strlen(model_env) == 0) {
    HLOG(kWarning, "Set CAE_SUMMARY_ENDPOINT and CAE_SUMMARY_MODEL");
    return 0;
  }

  std::string endpoint(endpoint_env);
  std::string model(model_env);
  HLOG(kInfo, "Endpoint: {}", endpoint);
  HLOG(kInfo, "Model: {}", model);
  HLOG(kInfo, "Sessions: {}, Questions: {}", kSessions.size(),
       kQuestions.size());

  // Initialize CTE
  bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!ok) {
    HLOG(kError, "Failed to initialize Chimaera");
    return 1;
  }
  wrp_cte::core::WRP_CTE_CLIENT_INIT();

  // ============================================================
  // Phase 1: Ingest all sessions into CTE
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "--- Phase 1: Ingesting sessions into CTE ---");

  auto t_ingest_start = Clock::now();
  const std::string tag_name = "bench_agent_memory";
  {
    wrp_cte::core::Tag tag(tag_name);
    for (const auto& sess : kSessions) {
      std::string blob_data = sess.date + "\n" + sess.content;
      tag.PutBlob(sess.id, blob_data.c_str(), blob_data.size());
      HLOG(kInfo, "  Stored session {} ({} bytes)", sess.id,
           blob_data.size());
    }
  }
  auto t_ingest_end = Clock::now();
  double ingest_ms = Ms(t_ingest_end - t_ingest_start).count();
  HLOG(kInfo, "  Ingest total: {:.1f} ms", ingest_ms);

  // ============================================================
  // Phase 2: CTE Raw — retrieve all, answer questions
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "APPROACH 1: CTE Raw Retrieval");
  HLOG(kInfo, "========================================");

  // Build full context from CTE blobs
  std::string raw_context;
  auto t_raw_retrieve_start = Clock::now();
  {
    wrp_cte::core::Tag tag(tag_name);
    for (const auto& sess : kSessions) {
      chi::u64 sz = tag.GetBlobSize(sess.id);
      std::vector<char> buf(sz);
      tag.GetBlob(sess.id, buf.data(), sz);
      raw_context += "[CTE blob: " + sess.id + "]\n";
      raw_context += std::string(buf.data(), sz) + "\n\n";
    }
  }
  auto t_raw_retrieve_end = Clock::now();
  double raw_retrieve_ms =
      Ms(t_raw_retrieve_end - t_raw_retrieve_start).count();

  HLOG(kInfo, "  Retrieved {} sessions from CTE ({:.1f} ms, {} chars)",
       kSessions.size(), raw_retrieve_ms, raw_context.size());

  // Answer questions with raw context
  int raw_correct = 0;
  double raw_total_answer_ms = 0.0;
  std::vector<std::string> raw_answers;
  std::vector<bool> raw_verdicts;

  for (size_t i = 0; i < kQuestions.size(); i++) {
    const auto& q = kQuestions[i];
    std::string prompt =
        "You are an AI assistant with access to stored conversation history "
        "retrieved from CTE. Answer the question using the retrieved "
        "sessions.\n\n"
        "Question: " +
        q.text +
        "\n\nRetrieved sessions from CTE:\n" + raw_context +
        "\nInstructions:\n"
        "- Answer concisely based ONLY on the retrieved context\n"
        "- If the context does not contain enough information, say so\n\n"
        "Answer:";

    auto t0 = Clock::now();
    std::string answer = CallLlm(endpoint, model, "", prompt);
    auto t1 = Clock::now();
    double ans_ms = Ms(t1 - t0).count();
    raw_total_answer_ms += ans_ms;
    raw_answers.push_back(answer);

    bool correct = JudgeAnswer(endpoint, model, q, answer);
    raw_verdicts.push_back(correct);
    if (correct) raw_correct++;

    HLOG(kInfo, "  Q{} [{}]: {} ({:.0f} ms) -> {}",
         i, q.category, correct ? "CORRECT" : "INCORRECT", ans_ms,
         answer.substr(0, 80));
  }

  // ============================================================
  // Phase 3: CTE+Summary — summarize, store, answer
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "APPROACH 2: CTE + Summary Module");
  HLOG(kInfo, "========================================");

  // Summarize each session
  HLOG(kInfo, "  --- Summarizing sessions ---");
  double total_summary_ms = 0.0;
  const std::string summary_tag = "bench_agent_memory_summaries";
  {
    wrp_cte::core::Tag stag(summary_tag);
    wrp_cte::core::Tag raw_tag(tag_name);

    for (const auto& sess : kSessions) {
      // Read raw session from CTE
      chi::u64 sz = raw_tag.GetBlobSize(sess.id);
      std::vector<char> buf(sz);
      raw_tag.GetBlob(sess.id, buf.data(), sz);
      std::string session_text(buf.data(), sz);

      // Summarize via LLM
      auto t0 = Clock::now();
      std::string summary = CallLlm(
          endpoint, model,
          "Summarize this conversation in 2-3 sentences. Capture ALL key "
          "facts: names, numbers, ports, dates, decisions, and action items. "
          "Return only the summary.",
          session_text, 150);
      auto t1 = Clock::now();
      double sum_ms = Ms(t1 - t0).count();
      total_summary_ms += sum_ms;

      // Store summary in CTE
      std::string summary_blob = sess.date + "\n" + summary;
      stag.PutBlob("summary_" + sess.id, summary_blob.c_str(),
                    summary_blob.size());

      HLOG(kInfo, "  {} ({:.0f} ms): {}", sess.id, sum_ms,
           summary.substr(0, 100));
    }
  }
  HLOG(kInfo, "  Total summarization: {:.0f} ms", total_summary_ms);

  // Build summary context from CTE
  std::string summary_context;
  auto t_sum_retrieve_start = Clock::now();
  {
    wrp_cte::core::Tag stag(summary_tag);
    for (const auto& sess : kSessions) {
      std::string blob_name = "summary_" + sess.id;
      chi::u64 sz = stag.GetBlobSize(blob_name);
      std::vector<char> buf(sz);
      stag.GetBlob(blob_name, buf.data(), sz);
      summary_context += "[CTE blob: " + blob_name + "]\n";
      summary_context += std::string(buf.data(), sz) + "\n\n";
    }
  }
  auto t_sum_retrieve_end = Clock::now();
  double sum_retrieve_ms =
      Ms(t_sum_retrieve_end - t_sum_retrieve_start).count();

  HLOG(kInfo, "  Retrieved {} summaries from CTE ({:.1f} ms, {} chars)",
       kSessions.size(), sum_retrieve_ms, summary_context.size());

  // Answer questions with summary context
  int sum_correct = 0;
  double sum_total_answer_ms = 0.0;
  std::vector<std::string> sum_answers;
  std::vector<bool> sum_verdicts;

  for (size_t i = 0; i < kQuestions.size(); i++) {
    const auto& q = kQuestions[i];
    std::string prompt =
        "You are an AI assistant with access to summarized conversation "
        "history from CTE. Answer the question using the retrieved "
        "summaries.\n\n"
        "Question: " +
        q.text +
        "\n\nRetrieved summaries from CTE:\n" + summary_context +
        "\nInstructions:\n"
        "- Answer concisely based ONLY on the provided summaries\n"
        "- If the summaries do not contain enough information, say so\n\n"
        "Answer:";

    auto t0 = Clock::now();
    std::string answer = CallLlm(endpoint, model, "", prompt);
    auto t1 = Clock::now();
    double ans_ms = Ms(t1 - t0).count();
    sum_total_answer_ms += ans_ms;
    sum_answers.push_back(answer);

    bool correct = JudgeAnswer(endpoint, model, q, answer);
    sum_verdicts.push_back(correct);
    if (correct) sum_correct++;

    HLOG(kInfo, "  Q{} [{}]: {} ({:.0f} ms) -> {}",
         i, q.category, correct ? "CORRECT" : "INCORRECT", ans_ms,
         answer.substr(0, 80));
  }

  // ============================================================
  // Results
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "RESULTS: CTE Agent Memory Benchmark");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Model: {}", model);
  HLOG(kInfo, "Sessions: {}, Questions: {}", kSessions.size(),
       kQuestions.size());
  HLOG(kInfo, "");

  HLOG(kInfo, "Accuracy:");
  HLOG(kInfo, "  CTE Raw:       {}/{}", raw_correct, kQuestions.size());
  HLOG(kInfo, "  CTE+Summary:   {}/{}", sum_correct, kQuestions.size());
  HLOG(kInfo, "");

  double raw_avg = raw_total_answer_ms / kQuestions.size();
  double sum_avg = sum_total_answer_ms / kQuestions.size();

  HLOG(kInfo, "Latency:");
  HLOG(kInfo, "  CTE ingest:              {:.1f} ms", ingest_ms);
  HLOG(kInfo, "  CTE retrieve (raw):      {:.1f} ms", raw_retrieve_ms);
  HLOG(kInfo, "  CTE retrieve (summary):  {:.1f} ms", sum_retrieve_ms);
  HLOG(kInfo, "  Summarization (1-time):  {:.0f} ms", total_summary_ms);
  HLOG(kInfo, "  Answer total (raw):      {:.0f} ms", raw_total_answer_ms);
  HLOG(kInfo, "  Answer total (summary):  {:.0f} ms", sum_total_answer_ms);
  HLOG(kInfo, "  Answer avg (raw):        {:.0f} ms", raw_avg);
  HLOG(kInfo, "  Answer avg (summary):    {:.0f} ms", sum_avg);
  HLOG(kInfo, "");

  HLOG(kInfo, "Context size:");
  HLOG(kInfo, "  Raw context:     {} chars", raw_context.size());
  HLOG(kInfo, "  Summary context: {} chars", summary_context.size());
  if (raw_context.size() > 0) {
    double reduction =
        100.0 * (1.0 - (double)summary_context.size() / raw_context.size());
    HLOG(kInfo, "  Reduction:       {:.0f}%", reduction);
  }
  HLOG(kInfo, "");

  if (sum_avg > 0) {
    HLOG(kInfo, "Per-query speedup: {:.1f}x", raw_avg / sum_avg);
  }

  HLOG(kInfo, "");
  HLOG(kInfo, "Per-question breakdown:");
  HLOG(kInfo, "  Q# | Category        | CTE Raw  | CTE+Sum  | Raw      | "
              "Summary");
  HLOG(kInfo, "  ---|-----------------|----------|----------|----------|-"
              "--------");
  for (size_t i = 0; i < kQuestions.size(); i++) {
    HLOG(kInfo, "  Q{} | {:<15} | {:>6.0f}ms | {:>6.0f}ms | {:<8} | {}",
         i, kQuestions[i].category, 0.0, 0.0,
         raw_verdicts[i] ? "CORRECT" : "WRONG",
         sum_verdicts[i] ? "CORRECT" : "WRONG");
  }

  HLOG(kInfo, "========================================");

  // ============================================================
  // Write results to JSON file
  // ============================================================
  double context_reduction = 0.0;
  if (raw_context.size() > 0) {
    context_reduction =
        100.0 * (1.0 - (double)summary_context.size() / raw_context.size());
  }
  double speedup = (sum_avg > 0) ? raw_avg / sum_avg : 0.0;

  nlohmann::json results;
  results["benchmark"] = "CTE Agent Memory Benchmark";
  results["model"] = model;
  results["endpoint"] = endpoint;
  results["sessions"] = kSessions.size();
  results["questions"] = kQuestions.size();

  results["accuracy"]["cte_raw"] = raw_correct;
  results["accuracy"]["cte_summary"] = sum_correct;
  results["accuracy"]["total"] = (int)kQuestions.size();

  results["latency_ms"]["cte_ingest"] = ingest_ms;
  results["latency_ms"]["cte_retrieve_raw"] = raw_retrieve_ms;
  results["latency_ms"]["cte_retrieve_summary"] = sum_retrieve_ms;
  results["latency_ms"]["summarization_total"] = total_summary_ms;
  results["latency_ms"]["answer_total_raw"] = raw_total_answer_ms;
  results["latency_ms"]["answer_total_summary"] = sum_total_answer_ms;
  results["latency_ms"]["answer_avg_raw"] = raw_avg;
  results["latency_ms"]["answer_avg_summary"] = sum_avg;

  results["context"]["raw_chars"] = raw_context.size();
  results["context"]["summary_chars"] = summary_context.size();
  results["context"]["reduction_pct"] = context_reduction;
  results["context"]["speedup"] = speedup;

  for (size_t i = 0; i < kQuestions.size(); i++) {
    nlohmann::json qr;
    qr["question"] = kQuestions[i].text;
    qr["category"] = kQuestions[i].category;
    qr["expected"] = kQuestions[i].expected;
    qr["raw_answer"] = raw_answers[i];
    qr["summary_answer"] = sum_answers[i];
    qr["raw_correct"] = raw_verdicts[i];
    qr["summary_correct"] = sum_verdicts[i];
    results["per_question"].push_back(qr);
  }

  // Write to file next to the binary or in current dir
  std::string out_path = "cte_memory_bench_results.json";
  std::ofstream ofs(out_path);
  if (ofs.is_open()) {
    ofs << results.dump(2) << std::endl;
    ofs.close();
    HLOG(kInfo, "Results written to: {}", out_path);
  } else {
    HLOG(kWarning, "Could not write results file: {}", out_path);
    // Print JSON to stdout as fallback
    std::cout << results.dump(2) << std::endl;
  }

  return 0;
#endif
}
