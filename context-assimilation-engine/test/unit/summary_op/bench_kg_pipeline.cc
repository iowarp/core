/*
 * bench_kg_pipeline.cc - Knowledge Graph Pipeline Benchmark
 *
 * Full end-to-end demo of the CTE knowledge graph pipeline:
 *   1. INGEST:    Store conversation sessions as CTE blobs (one tag per session)
 *   2. SUMMARIZE: LLM generates 2-3 sentence summaries
 *   3. UPDATE KG: Feed summaries into CTE knowledge graph (BM25 index)
 *   4. QUERY:     SemanticQuery to find relevant sessions, LLM answers
 *   5. REPORT:    Accuracy, latency, retrieval quality
 *
 * Requirements:
 * - CAE_SUMMARY_ENDPOINT and CAE_SUMMARY_MODEL must be set
 * - A running inference server at the endpoint
 * - Built with -DWRP_CAE_ENABLE_SUMMARY_OP=ON -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// ============================================================
// Test data: multi-session agent conversations
// ============================================================
struct Session {
  std::string id;
  std::string date;
  std::string content;
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

// ============================================================
// LLM call helpers
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
  for (auto& c : verdict) c = toupper(c);
  return verdict.find("CORRECT") != std::string::npos &&
         verdict.find("INCORRECT") == std::string::npos;
}

// ============================================================
// Main benchmark
// ============================================================
int main() {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Knowledge Graph Pipeline Benchmark");
  HLOG(kInfo, "========================================");

#ifndef WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
  HLOG(kWarning, "Knowledge graph not compiled. "
                  "Rebuild with -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON");
  return 0;
#else

  // Check environment
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

  // Map session ID → TagId for later lookup
  const std::string tag_prefix = "kg_bench_";
  std::unordered_map<std::string, wrp_cte::core::TagId> session_tag_ids;

  // ============================================================
  // Phase 1: Ingest — one tag per session
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 1: Ingest ({} sessions) ===", kSessions.size());

  auto t_ingest_start = Clock::now();
  for (const auto& sess : kSessions) {
    std::string tag_name = tag_prefix + sess.id;
    std::string blob_data = sess.date + "\n" + sess.content;

    wrp_cte::core::Tag tag(tag_name);
    tag.PutBlob("raw", blob_data.c_str(), blob_data.size());

    session_tag_ids[sess.id] = tag.GetTagId();

    HLOG(kInfo, "  {}: {} bytes", tag_name, blob_data.size());
  }
  auto t_ingest_end = Clock::now();
  double ingest_ms = Ms(t_ingest_end - t_ingest_start).count();
  HLOG(kInfo, "  Total ingest: {:.1f} ms", ingest_ms);

  // ============================================================
  // Phase 2: Summarize each session via LLM
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 2: Summarize via LLM ===");

  std::unordered_map<std::string, std::string> summaries;
  double total_summary_ms = 0.0;

  for (const auto& sess : kSessions) {
    std::string tag_name = tag_prefix + sess.id;
    wrp_cte::core::Tag tag(tag_name);

    // Read raw blob back
    chi::u64 sz = tag.GetBlobSize("raw");
    std::vector<char> buf(sz);
    tag.GetBlob("raw", buf.data(), sz);
    std::string session_text(buf.data(), sz);

    // Call LLM to summarize
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

    // Store summary blob in CTE
    tag.PutBlob("summary", summary.c_str(), summary.size());
    summaries[sess.id] = summary;

    HLOG(kInfo, "  {} ({:.0f} ms): {}", sess.id, sum_ms,
         summary.substr(0, 100));
  }
  HLOG(kInfo, "  Total summarize: {:.1f} sec", total_summary_ms / 1000.0);

  // ============================================================
  // Phase 3: Update Knowledge Graph
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 3: Update Knowledge Graph ===");

  auto t_kg_start = Clock::now();
  for (const auto& sess : kSessions) {
    auto tag_id = session_tag_ids[sess.id];
    const auto& summary = summaries[sess.id];

    auto fut = WRP_CTE_CLIENT->AsyncUpdateKnowledgeGraph(tag_id, summary);
    fut.Wait();
    HLOG(kInfo, "  Added {} to KG", sess.id);
  }
  auto t_kg_end = Clock::now();
  double kg_ms = Ms(t_kg_end - t_kg_start).count();
  HLOG(kInfo, "  {} entries added in {:.1f} ms", kSessions.size(), kg_ms);

  // ============================================================
  // Phase 4: Query + Answer
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "=== Phase 4: Query + Answer ({} questions) ===",
       kQuestions.size());

  int correct_count = 0;
  double total_search_ms = 0.0;
  double total_answer_ms = 0.0;
  int total_retrieved = 0;

  for (size_t i = 0; i < kQuestions.size(); i++) {
    const auto& q = kQuestions[i];
    HLOG(kInfo, "");
    HLOG(kInfo, "  Q{} [{}] \"{}\"", i, q.category, q.text);

    // Semantic query
    auto t_search_start = Clock::now();
    auto fut = WRP_CTE_CLIENT->AsyncSemanticQuery(q.text, 3);
    fut.Wait();
    auto *task_result = fut.get();
    std::vector<wrp_cte::core::TagId> result_tags = task_result->result_tags_;
    std::vector<float> result_scores = task_result->result_scores_;
    auto t_search_end = Clock::now();
    double search_ms = Ms(t_search_end - t_search_start).count();
    total_search_ms += search_ms;

    // Log retrieved sessions with scores
    std::string retrieved_str;
    for (size_t r = 0; r < result_tags.size(); ++r) {
      // Find which session this TagId belongs to
      for (const auto& [sid, stored_tid] : session_tag_ids) {
        if (stored_tid == result_tags[r]) {
          if (!retrieved_str.empty()) retrieved_str += ", ";
          retrieved_str += sid + " (" +
              std::to_string(result_scores[r]).substr(0, 4) + ")";
          break;
        }
      }
    }
    total_retrieved += result_tags.size();
    HLOG(kInfo, "    KG retrieved: [{}] ({:.1f} ms)", retrieved_str, search_ms);

    // Build context from retrieved sessions
    std::string context;
    for (const auto& tid : result_tags) {
      for (const auto& [sid, stored_tid] : session_tag_ids) {
        if (stored_tid == tid) {
          std::string tag_name = tag_prefix + sid;
          wrp_cte::core::Tag tag(tag_name);
          chi::u64 sz = tag.GetBlobSize("raw");
          std::vector<char> buf(sz);
          tag.GetBlob("raw", buf.data(), sz);
          context += "[Session " + sid + "]\n";
          context += std::string(buf.data(), sz) + "\n\n";
          break;
        }
      }
    }

    // LLM answer
    std::string prompt =
        "You are an AI assistant. Answer the question using ONLY the "
        "retrieved conversation sessions below.\n\n"
        "Question: " + q.text +
        "\n\nRetrieved sessions:\n" + context +
        "\nInstructions:\n"
        "- Answer concisely based ONLY on the retrieved context\n"
        "- If the context does not contain enough information, say so\n\n"
        "Answer:";

    auto t_ans_start = Clock::now();
    std::string answer = CallLlm(endpoint, model, "", prompt);
    auto t_ans_end = Clock::now();
    double ans_ms = Ms(t_ans_end - t_ans_start).count();
    total_answer_ms += ans_ms;

    // Judge
    bool is_correct = JudgeAnswer(endpoint, model, q, answer);
    if (is_correct) correct_count++;

    HLOG(kInfo, "    Answer: {} ({:.0f} ms)",
         answer.substr(0, 120), ans_ms);
    HLOG(kInfo, "    Verdict: {}", is_correct ? "CORRECT" : "INCORRECT");
  }

  // ============================================================
  // Phase 5: Report
  // ============================================================
  HLOG(kInfo, "");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Results");
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "  Accuracy: {}/{} ({:.1f}%)",
       correct_count, kQuestions.size(),
       100.0 * correct_count / kQuestions.size());
  HLOG(kInfo, "  Ingest latency:     {:.1f} ms total", ingest_ms);
  HLOG(kInfo, "  Summarize latency:  {:.1f} sec total",
       total_summary_ms / 1000.0);
  HLOG(kInfo, "  KG update latency:  {:.1f} ms total", kg_ms);
  HLOG(kInfo, "  Avg KG search:      {:.1f} ms",
       total_search_ms / kQuestions.size());
  HLOG(kInfo, "  Avg LLM answer:     {:.1f} sec",
       total_answer_ms / 1000.0 / kQuestions.size());
  HLOG(kInfo, "  Avg sessions retrieved: {:.1f}",
       (double)total_retrieved / kQuestions.size());
  HLOG(kInfo, "========================================");

  return 0;
#endif  // WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
}
