/*
 * Acropolis — minimal "agent scans a repo" benchmark.
 *
 * Boots an embedded CTE runtime, walks a directory of text files, ingests
 * each one as a CTE tag at a chosen indexing depth, then runs a fixed set
 * of natural-language queries via SemanticQuery (and optionally a
 * tool-calling LLM agent) and measures precision + agent token usage.
 *
 * Depth is set globally in the compose YAML (`indexing_depth: default:`)
 * that this binary writes per run. That mirrors the production path —
 * operators declare policy in YAML, not per-file.
 *
 * Usage:
 *   bench_repo_scan <repo_dir> [level]
 *     repo_dir  directory to scan (default: /workspace/context-transfer-engine)
 *     level     0|1|2 = name | metadata | content (default: 0)
 */

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_runtime.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/core/content_transfer_engine.h>

#ifdef ACROPOLIS_BENCH_USE_SUMMARY
#include <wrp_cae/core/factory/summary_operator.h>
#endif

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace wrp_cte::core;

namespace {

bool IsTextFile(const fs::path &p) {
  static const std::unordered_set<std::string> kTextExt = {
      ".cc", ".cpp", ".cxx", ".c",  ".h",   ".hpp", ".py", ".md",
      ".txt", ".yaml", ".yml", ".json", ".sh", ".cmake",
  };
  std::string ext = p.extension().string();
  for (auto &c : ext) c = std::tolower(static_cast<unsigned char>(c));
  if (kTextExt.count(ext)) return true;
  // Files named CMakeLists.txt have ext=".txt"; others without ext are skipped
  return false;
}

bool ShouldSkipDir(const fs::path &p) {
  std::string n = p.filename().string();
  return n == ".git" || n == "build" || n.rfind("build_", 0) == 0 ||
         n == "node_modules" || n == "__pycache__" || n == ".cache";
}

}  // namespace

int main(int argc, char **argv) {
  std::string repo_dir =
      (argc > 1) ? argv[1] : "/workspace/context-transfer-engine";
  int level = (argc > 2) ? std::atoi(argv[2]) : 0;
  if (level < 0 || level > 2) {
    std::cerr << "level must be 0, 1, or 2\n";
    return 1;
  }

  // Backend selection via env var (default: bm25).
  // Qdrant requires QDRANT_URL (host:port/collection) + an embedding endpoint
  // (CTE_EMBEDDING_ENDPOINT + CTE_EMBEDDING_MODEL).
  std::string backend = "bm25";
  if (const char *b = std::getenv("ACROPOLIS_BENCH_BACKEND")) backend = b;

  std::cout << "=== Acropolis repo scan ===\n"
            << "repo:     " << repo_dir << "\n"
            << "level:    " << level << " ("
            << (level == 0 ? "name" : level == 1 ? "metadata" : "content")
            << ")\n"
            << "backend:  " << backend << "\n\n";

  // --- Compose YAML (only path that propagates kg_backend config to runtime) ---
  chi::PoolId pool_id(static_cast<chi::u32>(8000 + level), 0);
  chi::PoolId bdev_pool_id(static_cast<chi::u32>(8500 + level), 0);
  std::string compose_path = "/tmp/acropolis_bench_compose_" +
                              std::to_string(level) + "_" + backend + ".yaml";
  {
    std::ofstream y(compose_path);
    y << "runtime:\n"
         "  num_threads: 4\n"
         "  queue_depth: 256\n"
         "  conf_dir: /tmp/acropolis_bench_" << level << "_" << backend << "\n"
         "\n"
         "compose:\n"
         "  - mod_name: chimaera_bdev\n"
         "    pool_name: bench_ram_" << level << "_" << backend << "\n"
         "    pool_id: \"" << bdev_pool_id.major_ << "." << bdev_pool_id.minor_ << "\"\n"
         "    pool_query: local\n"
         "    capacity: 512MB\n"
         "  - mod_name: wrp_cte_core\n"
         "    pool_name: bench_cte_" << level << "_" << backend << "\n"
         "    pool_id: \"" << pool_id.major_ << "." << pool_id.minor_ << "\"\n"
         "    pool_query: local\n"
         "    storage:\n"
         "      - path: /tmp/acropolis_bench_" << level << "_" << backend << "/ram\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 512MB\n"
         "    indexing_depth:\n"
         "      default: " << level << "\n";
    if (backend == "qdrant") {
      const char *qurl = std::getenv("QDRANT_URL");
      std::string q = qurl ? qurl : "host.docker.internal:6333/acropolis_bench";
      const char *emb = std::getenv("CTE_EMBEDDING_ENDPOINT");
      std::string e = emb ? emb : "http://host.docker.internal:11434/v1/embeddings";
      size_t slash = q.find('/');
      std::string host_port = (slash != std::string::npos) ? q.substr(0, slash) : q;
      std::string col = (slash != std::string::npos)
                            ? q.substr(slash + 1) + "_L" + std::to_string(level)
                            : std::string("acropolis_bench_L") + std::to_string(level);
      y << "    knowledge_graph:\n"
           "      backend: qdrant\n"
           "      config: \"" << host_port << "/" << col << " " << e << "\"\n"
           "    embedding:\n"
           "      endpoint: \"" << e << "\"\n";
      const char *emb_model = std::getenv("CTE_EMBEDDING_MODEL");
      if (emb_model) y << "      model: \"" << emb_model << "\"\n";
    } else if (backend == "elasticsearch") {
      const char *eurl = std::getenv("ES_URL");
      std::string base = eurl ? eurl
                               : std::string("host.docker.internal:9200/acropolis_bench");
      size_t slash = base.find('/');
      std::string host_port = (slash != std::string::npos) ? base.substr(0, slash) : base;
      std::string index = (slash != std::string::npos)
                              ? base.substr(slash + 1) + "_l" + std::to_string(level)
                              : std::string("acropolis_bench_l") + std::to_string(level);
      const char *mode = std::getenv("ES_MODE");   // keyword|vector|both
      std::string m = mode ? mode : "keyword";
      std::string cfg = host_port + "/" + index + " " + m;
      if (m == "vector" || m == "both") {
        const char *emb = std::getenv("CTE_EMBEDDING_ENDPOINT");
        std::string e = emb ? emb : "http://host.docker.internal:11434/v1/embeddings";
        cfg += " " + e;
        y << "    knowledge_graph:\n"
             "      backend: elasticsearch\n"
             "      config: \"" << cfg << "\"\n"
             "    embedding:\n"
             "      endpoint: \"" << e << "\"\n";
        const char *emb_model = std::getenv("CTE_EMBEDDING_MODEL");
        if (emb_model) y << "      model: \"" << emb_model << "\"\n";
      } else {
        y << "    knowledge_graph:\n"
             "      backend: elasticsearch\n"
             "      config: \"" << cfg << "\"\n";
      }
    } else if (backend == "neo4j") {
      const char *nurl = std::getenv("NEO4J_URL");
      std::string n = nurl ? nurl : "host.docker.internal:7474";
      const char *analyzer = std::getenv("NEO4J_ANALYZER");
      std::string cfg = n;
      cfg += " analyzer=";
      cfg += analyzer ? analyzer : "simple";
      const char *nmode = std::getenv("NEO4J_MODE");  // keyword|hybrid
      bool want_vec = !nmode || std::string(nmode) == "hybrid";
      if (want_vec) {
        const char *emb = std::getenv("CTE_EMBEDDING_ENDPOINT");
        std::string e = emb ? emb : "http://host.docker.internal:11434/v1/embeddings";
        cfg += " ";
        cfg += e;
        y << "    knowledge_graph:\n"
             "      backend: neo4j\n"
             "      config: \"" << cfg << "\"\n"
             "    embedding:\n"
             "      endpoint: \"" << e << "\"\n";
        const char *emb_model = std::getenv("CTE_EMBEDDING_MODEL");
        if (emb_model) y << "      model: \"" << emb_model << "\"\n";
      } else {
        y << "    knowledge_graph:\n"
             "      backend: neo4j\n"
             "      config: \"" << cfg << "\"\n";
      }
    } else {
      y << "    knowledge_graph:\n"
           "      backend: bm25\n";
    }
  }
  ::setenv("CHI_SERVER_CONF", compose_path.c_str(), 1);

  // --- Boot embedded runtime (reads CHI_SERVER_CONF and auto-creates pools) ---
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    std::cerr << "CHIMAERA_INIT failed\n";
    return 2;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(600));

  if (!WRP_CTE_CLIENT_INIT()) {
    std::cerr << "WRP_CTE_CLIENT_INIT failed\n";
    return 2;
  }

  Client client(pool_id);
  WRP_CTE_CLIENT->Init(pool_id);

  // --- Collect file list ---
  // Depth is set globally in the compose YAML emitted above
  // (`indexing_depth: default: <level>`). No per-file xattr needed.
  std::vector<std::string> files;
  for (auto it = fs::recursive_directory_iterator(
           repo_dir, fs::directory_options::skip_permission_denied);
       it != fs::recursive_directory_iterator(); ++it) {
    if (it->is_directory() && ShouldSkipDir(it->path())) {
      it.disable_recursion_pending();
      continue;
    }
    if (!it->is_regular_file()) continue;
    if (!IsTextFile(it->path())) continue;
    files.push_back(it->path().string());
  }
  std::cout << "Discovered " << files.size() << " files\n";

  // --- Optional L2: pre-generate LLM summaries via CAE SummaryOperator ---
  // Only when level==2 AND CAE_SUMMARY_ENDPOINT is set AND we were built
  // with ACROPOLIS_BENCH_USE_SUMMARY.
  std::unordered_map<std::string, std::string> path_to_summary;
  bool l2_summary_enabled = false;
#ifdef ACROPOLIS_BENCH_USE_SUMMARY
  if (level == 2 && std::getenv("CAE_SUMMARY_ENDPOINT") &&
      std::getenv("CAE_SUMMARY_MODEL")) {
    l2_summary_enabled = true;
    std::cout << "\nL2: generating LLM summaries via CAE SummaryOperator ("
              << std::getenv("CAE_SUMMARY_MODEL") << " @ "
              << std::getenv("CAE_SUMMARY_ENDPOINT") << ")\n";

    // --- Disk cache: summaries are a function of (model, repo, file set), not
    //     backend. Reuse across cells to save ~30 min per matrix run.
    std::string cache_path;
    {
      std::string model = std::getenv("CAE_SUMMARY_MODEL");
      std::vector<std::string> sorted_files = files;
      std::sort(sorted_files.begin(), sorted_files.end());
      size_t h = std::hash<std::string>{}(model) ^
                 std::hash<std::string>{}(repo_dir);
      for (const auto &p : sorted_files) {
        h ^= std::hash<std::string>{}(p) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%zx", h);
      cache_path = std::string("/tmp/acropolis_summary_cache_") + buf + ".json";
    }
    bool loaded_from_cache = false;
    {
      std::ifstream cf(cache_path);
      if (cf) {
        std::string data((std::istreambuf_iterator<char>(cf)),
                         std::istreambuf_iterator<char>());
        auto j = nlohmann::json::parse(data, nullptr, false);
        if (!j.is_discarded() && j.is_object()) {
          for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.value().is_string()) {
              path_to_summary[it.key()] = it.value().get<std::string>();
            }
          }
          std::cout << "  cache hit: " << path_to_summary.size()
                    << " summaries loaded from " << cache_path << "\n";
          loaded_from_cache = true;
        }
      }
    }
    if (!loaded_from_cache) {
    // Worker pool — N threads each holding a SummaryOperator. The CTE client
    // is shared via the global WRP_CTE_CLIENT singleton.
    int num_workers = 8;
    if (const char *w = std::getenv("ACROPOLIS_BENCH_WORKERS")) {
      num_workers = std::max(1, std::atoi(w));
    }
    std::cout << "  workers: " << num_workers << "\n";

    std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::queue<std::string> work;
    bool done = false;
    std::atomic<int> processed{0};
    std::mutex result_mu;
    auto t_sum_start = std::chrono::steady_clock::now();

    for (auto &p : files) work.push(p);
    auto total_work = files.size();

    auto worker = [&]() {
      // Each worker shares the global CTE client and creates its own
      // SummaryOperator (stateless apart from the env-var endpoint/model).
      auto cte_shared = std::shared_ptr<wrp_cte::core::Client>(
          WRP_CTE_CLIENT, [](wrp_cte::core::Client *) {});
      wrp_cae::core::SummaryOperator op(cte_shared);

      while (true) {
        std::string p;
        {
          std::unique_lock<std::mutex> lk(queue_mu);
          queue_cv.wait(lk, [&]() { return done || !work.empty(); });
          if (work.empty()) return;
          p = work.front();
          work.pop();
        }

        // 1. Read file content (cap at 4 KB for prompt-size control)
        std::ifstream f(p, std::ios::binary);
        if (!f) { ++processed; continue; }
        std::string buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        if (buf.empty()) { ++processed; continue; }
        if (buf.size() > 4096) buf.resize(4096);
        // Prefix with the filename so the LLM has context
        std::string desc = "FILE: " + p + "\n\n" + buf;

        int last_rc = 99;
        std::string last_err;
        try {
          wrp_cte::core::Tag tag(p);
          tag.PutBlob("description", desc.c_str(), desc.size());
          last_rc = op.Execute(p);
          if (last_rc == 0) {
            chi::u64 sz = tag.GetBlobSize("summary");
            if (sz > 0 && sz < 4096) {
              std::vector<char> sbuf(sz);
              tag.GetBlob("summary", sbuf.data(), sz);
              std::string s(sbuf.data(), sz);
              std::lock_guard<std::mutex> lk(result_mu);
              path_to_summary[p] = std::move(s);
            }
          }
        } catch (const std::exception &e) {
          last_err = e.what();
        } catch (...) {
          last_err = "unknown exception";
        }
        if (processed.load() < 3 && (last_rc != 0 || !last_err.empty())) {
          std::lock_guard<std::mutex> lk(result_mu);
          std::cerr << "  [debug] " << p << " rc=" << last_rc
                    << " err='" << last_err << "'\n";
        }
        int n = ++processed;
        if (n % 50 == 0 || n == static_cast<int>(total_work)) {
          double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t_sum_start)
                               .count();
          std::cout << "    " << n << "/" << total_work << " summarized ("
                    << elapsed << " s, " << path_to_summary.size()
                    << " success)\r" << std::flush;
        }
      }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < num_workers; ++i) pool.emplace_back(worker);
    {
      std::unique_lock<std::mutex> lk(queue_mu);
      done = true;
    }
    queue_cv.notify_all();
    for (auto &t : pool) t.join();

    double sum_sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_sum_start)
                         .count();
    std::cout << "\n  L2 summaries: " << path_to_summary.size() << "/"
              << files.size() << " in " << sum_sec << " s ("
              << (sum_sec / std::max<size_t>(1, path_to_summary.size()))
              << " s/file avg)\n";

    // Persist cache for the next cell to reuse.
    if (!path_to_summary.empty()) {
      nlohmann::json j(path_to_summary);
      std::ofstream cf(cache_path);
      if (cf) {
        cf << j.dump();
        std::cout << "  cached " << path_to_summary.size()
                  << " summaries to " << cache_path << "\n";
      }
    }
    }  // if (!loaded_from_cache)
  } else if (level == 2) {
    std::cout << "\nL2 mode: SummaryOperator NOT enabled "
                 "(set CAE_SUMMARY_ENDPOINT + CAE_SUMMARY_MODEL); "
                 "L2 will fall back to L1 payload.\n";
  }
#else
  if (level == 2) {
    std::cout << "\nL2 mode: benchmark not built with ACROPOLIS_BENCH_USE_SUMMARY; "
                 "L2 will fall back to L1 payload.\n";
  }
#endif

  // --- Ingest all files: GetOrCreateTag + UpdateKnowledgeGraph ---
  std::cout << "\nIngesting (level=" << level << ", L2 summary="
            << (l2_summary_enabled ? "yes" : "no") << ")...\n";
  std::unordered_map<uint64_t, std::string> tag_to_path;
  auto t0 = std::chrono::steady_clock::now();
  size_t ingested = 0;
  for (const auto &path_str : files) {
    auto got = client.AsyncGetOrCreateTag(path_str);
    got.Wait();
    if (got->GetReturnCode() != 0) continue;
    TagId tag = got->tag_id_;

    std::string summary;
    auto sit = path_to_summary.find(path_str);
    if (sit != path_to_summary.end()) summary = sit->second;

    auto upd = client.AsyncUpdateKnowledgeGraph(tag, path_str, summary);
    upd.Wait();

    uint64_t key = (static_cast<uint64_t>(tag.major_) << 32) | tag.minor_;
    tag_to_path[key] = path_str;
    ++ingested;
  }
  auto t1 = std::chrono::steady_clock::now();
  double ingest_sec = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "Ingested " << ingested << " files in " << ingest_sec << " s\n\n";

  // --- Queries with ground-truth (substring match against the file path) ---
  // Natural-language "locate X" questions. Three difficulty tiers:
  //   (A) Easy   — query and target filename share a keyword (e.g.
  //                "Qdrant" → kg_backend_qdrant.h).
  //   (B) Medium — filename is suggestive but not a direct keyword match.
  //   (C) Hard   — filename gives no hint; only file content or the L2
  //                summary carries the relevant vocabulary.
  struct Query { std::string text; std::string expected_substring; };
  std::vector<Query> queries = {
      // (A) Easy
      {"Find the Qdrant vector backend implementation",
       "kg_backend_qdrant.h"},
      {"Locate the Elasticsearch full-text search backend",
       "kg_backend_elasticsearch.h"},
      {"Show me the Neo4j knowledge graph backend",
       "kg_backend_neo4j.h"},
      {"Where is the HDF5 metadata extractor?",
       "hdf5_summary.h"},
      {"Find the CLI tool that sets indexing depth on files",
       "set_depth.cc"},
      // (B) Medium
      {"Locate the unit test that exercises GPU submission on an actual GPU",
       "test_gpu_submission_gpu.cc"},
      {"Find the code that streams per-layer GGML weights through GpuVMM",
       "ggml_iowarp_backend.cc"},
      {"Where is the KV-cache manager that interacts with llama.cpp?",
       "kvcache_manager.cc"},
      {"Find the end-to-end script that tests KV cache restore on GPU",
       "run_e2e_gpu_test.sh"},
      {"Locate the benchmark that scans a repo with an LLM agent loop",
       "bench_repo_scan.cc"},
      // (C) Hard — filename gives no hint; content/summary carries the terms.
      {"Find the file implementing BM25 scoring with distributed IDF synchronization",
       "kg_backend_bm25.h"},
      {"Where is the hybrid retrieval that uses reciprocal rank fusion?",
       "kg_backend_elasticsearch.h"},
      {"Locate the depth controller that resolves xattr inheritance across directories",
       "depth_controller.h"},
      {"Find the OpenAI-compatible HTTP embeddings client shared across backends",
       "embedding_client.h"},
      {"Where is the unit test validating indexing-depth configuration parsing?",
       "test_indexing_depth_config.cc"},
  };

  int top1 = 0, top5 = 0;
  double total_q_ms = 0.0;
  for (const auto &q_pair : queries) {
    const auto &q = q_pair.text;
    const auto &expected = q_pair.expected_substring;
    auto t_q = std::chrono::steady_clock::now();
    auto sq = client.AsyncSemanticQuery(q, /*top_k=*/5, chi::PoolQuery::Local());
    sq.Wait();
    auto t_q_end = std::chrono::steady_clock::now();
    double q_ms =
        std::chrono::duration<double, std::milli>(t_q_end - t_q).count();
    total_q_ms += q_ms;

    bool hit_top1 = false, hit_top5 = false;
    int rank_of_correct = -1;
    for (size_t i = 0; i < sq->result_tags_.size(); ++i) {
      auto &tid = sq->result_tags_[i];
      uint64_t key = (static_cast<uint64_t>(tid.major_) << 32) | tid.minor_;
      auto pit = tag_to_path.find(key);
      if (pit == tag_to_path.end()) continue;
      if (pit->second.find(expected) != std::string::npos) {
        hit_top5 = true;
        if (i == 0) hit_top1 = true;
        rank_of_correct = static_cast<int>(i + 1);
        break;
      }
    }
    if (hit_top1) ++top1;
    if (hit_top5) ++top5;

    std::cout << "Q: \"" << q << "\"   (" << q_ms << " ms)\n"
              << "   expected: ..." << expected
              << "   rank: "
              << (rank_of_correct > 0 ? std::to_string(rank_of_correct)
                                       : std::string("(not in top 5)"))
              << "\n";
    for (size_t i = 0; i < sq->result_tags_.size(); ++i) {
      auto &tid = sq->result_tags_[i];
      uint64_t key = (static_cast<uint64_t>(tid.major_) << 32) | tid.minor_;
      auto pit = tag_to_path.find(key);
      std::string path =
          (pit != tag_to_path.end()) ? pit->second
                                     : (std::to_string(tid.major_) + "." +
                                        std::to_string(tid.minor_));
      if (path.rfind(repo_dir, 0) == 0) {
        path = path.substr(repo_dir.size());
        if (!path.empty() && path[0] == '/') path = path.substr(1);
      }
      bool is_match = path.find(expected) != std::string::npos;
      std::cout << "    " << (i + 1) << (is_match ? " * " : "   ")
                << "score=" << sq->result_scores_[i] << "  " << path << "\n";
    }
    std::cout << "\n";
  }

  // --- Real-agent loop (gated by ACROPOLIS_BENCH_AGENT=1) ---
  //
  // Tool-use agent: the model sees the user's query and two tools —
  //   semantic_query(query, k)  — runs SemanticQuery through CTE, returns
  //                                path+score and (at L2 only) the stored
  //                                summary.
  //   read_file(path, max_bytes) — escape hatch when the summary / path
  //                                alone isn't enough.
  // The agent decides when to call each. We accumulate prompt+completion
  // tokens across all round-trips and count how many times each tool fires.
  //
  // At L2 the summary is returned inline, so the agent should usually stop
  // after one semantic_query. At L0/L1 it has to fall into read_file.
  //
  // Required env: CAE_SUMMARY_ENDPOINT (e.g. http://host.docker.internal:11434/v1)
  //               CAE_SUMMARY_MODEL    (e.g. qwen2.5:3b)
  size_t agent_real_prompt = 0, agent_real_completion = 0;
  double agent_real_wall_s = 0.0;
  int agent_real_queries = 0;
  int agent_real_sem_calls = 0, agent_real_read_calls = 0;
  int agent_real_correct = 0;  // answer text mentions expected substring
  const char *agent_flag = std::getenv("ACROPOLIS_BENCH_AGENT");
  if (agent_flag && std::string(agent_flag) == "1" &&
      std::getenv("CAE_SUMMARY_ENDPOINT") &&
      std::getenv("CAE_SUMMARY_MODEL")) {
    std::string ep = std::getenv("CAE_SUMMARY_ENDPOINT");
    // Agent model can be swapped independently from the summary model so we
    // can A/B the same summary cache against different agent LLMs.
    std::string model;
    if (const char *am = std::getenv("ACROPOLIS_BENCH_AGENT_MODEL")) {
      model = am;
    } else {
      model = std::getenv("CAE_SUMMARY_MODEL");
    }

    // Parse "http://host:port[/base]" into host/port/base.
    std::string host = "localhost";
    int port = 11434;
    std::string base = "/v1";
    {
      auto p_end = ep.find("://");
      std::string rest =
          (p_end != std::string::npos) ? ep.substr(p_end + 3) : ep;
      auto sl = rest.find('/');
      if (sl != std::string::npos) { base = rest.substr(sl); rest = rest.substr(0, sl); }
      auto co = rest.find(':');
      if (co != std::string::npos) {
        host = rest.substr(0, co);
        port = std::stoi(rest.substr(co + 1));
      } else { host = rest; }
    }
    if (!base.empty() && base.back() == '/') base.pop_back();

    httplib::Client cli(host, port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(180);

    // Agent mode selection:
    //   tool_use   (default) — semantic_query + read_file
    //   no_search            — read_file only, agent must guess paths
    //   list_files           — list_files + read_file (file-system navigation)
    std::string agent_mode = "tool_use";
    if (const char *m = std::getenv("ACROPOLIS_BENCH_AGENT_MODE")) {
      agent_mode = m;
    }
    const bool no_search_mode = (agent_mode == "no_search");
    const bool list_files_mode = (agent_mode == "list_files");

    // Tool schemas (OpenAI function-calling format).
    nlohmann::json semantic_tool =
        {{"type", "function"},
         {"function",
          {{"name", "semantic_query"},
           {"description",
            "Search the codebase index. Returns up to k hits; each hit has "
            "{path, score} and (when the index depth is L2) a 'summary' "
            "field pre-generated by an LLM. Call this first."},
           {"parameters",
            {{"type", "object"},
             {"properties",
              {{"query", {{"type", "string"}}},
               {"k", {{"type", "integer"}}}}},
             {"required", nlohmann::json::array({"query"})}}}}}};
    nlohmann::json read_file_tool =
        {{"type", "function"},
         {"function",
          {{"name", "read_file"},
           {"description",
            no_search_mode
                ? "Read the contents of a file at an absolute path. This is "
                  "the ONLY tool available. Guess paths based on the query "
                  "and your prior knowledge of common codebase layouts."
                : "Read the contents of a file. Use only for paths returned "
                  "by a previous semantic_query call. Returns up to max_bytes."},
           {"parameters",
            {{"type", "object"},
             {"properties",
              {{"path", {{"type", "string"}}},
               {"max_bytes", {{"type", "integer"}}}}},
             {"required", nlohmann::json::array({"path"})}}}}}};
    nlohmann::json list_files_tool =
        {{"type", "function"},
         {"function",
          {{"name", "list_files"},
           {"description",
            "List up to max_entries file paths under a directory. "
            "Use this to navigate the repo when semantic_query is unavailable. "
            "Use dir='/' or dir='' to list from the repo root. "
            "recursive=true to walk subdirectories."},
           {"parameters",
            {{"type", "object"},
             {"properties",
              {{"dir", {{"type", "string"}}},
               {"recursive", {{"type", "boolean"}}},
               {"max_entries", {{"type", "integer"}}}}},
             {"required", nlohmann::json::array({"dir"})}}}}}};
    nlohmann::json tools;
    if (no_search_mode) {
      tools = nlohmann::json::array({read_file_tool});
    } else if (list_files_mode) {
      tools = nlohmann::json::array({list_files_tool, read_file_tool});
    } else {
      tools = nlohmann::json::array({semantic_tool, read_file_tool});
    }

    std::cout << "\n=== Agent loop (mode=" << agent_mode
              << " model=" << model
              << " endpoint=" << host << ":" << port << base << ") ===\n";

    // no_search / list_files modes need more iterations — the agent navigates.
    const int kMaxIters = (no_search_mode || list_files_mode) ? 16 : 8;
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      const auto &q = queries[qi].text;

      nlohmann::json messages = nlohmann::json::array();
      std::string system_content;
      if (no_search_mode) {
        system_content =
            "You are a code-search assistant with NO search index access. "
            "The only tool is read_file(path, max_bytes). You must guess "
            "plausible absolute paths based on the query and common "
            "codebase conventions (e.g. /workspace/<project>/core/..., "
            "include/..., test/...), then read files and answer. "
            "RULES: "
            "(1) Try read_file on up to 6 candidate paths you think are "
            "    most likely. Expect many read_file calls to fail with "
            "    'cannot open file' — that's fine, try the next guess. "
            "(2) Once you find content that matches the query intent, "
            "    answer in one sentence naming that file. "
            "(3) If you can't find anything after 6 attempts, say so.";
      } else if (list_files_mode) {
        system_content =
            "You are a code-search assistant. You MUST answer in English. "
            "Tools available: list_files(dir, recursive, max_entries) and "
            "read_file(path, max_bytes). The repo root is /workspace. "
            "REQUIRED STEPS (follow in order, do not skip):\n"
            "  STEP 1: call list_files(dir='/workspace', recursive=true, "
            "          max_entries=2000).\n"
            "  STEP 2: from the returned paths, pick EXACTLY ONE path whose "
            "          filename best matches the query.\n"
            "  STEP 3: call read_file(path=<that path>, max_bytes=4096) to "
            "          verify the content matches.\n"
            "  STEP 4: answer in ONE sentence, and you MUST include the full "
            "          absolute path of the file in your answer (e.g. "
            "          '/workspace/path/to/file.h').\n"
            "Do NOT answer until STEP 3 has completed. Do NOT summarize the "
            "file list. Do NOT answer in any language other than English.";
      } else {
        system_content =
            "You are a code-search assistant. For each user query, call "
            "semantic_query ONCE to fetch ranked hits. Inspect the result: "
            "every hit has {path, score}, and at index-depth L2 each hit "
            "ALSO has a trusted LLM-generated 'summary' field. "
            "RULES: "
            "(1) If the hits include a 'summary' field, you already have "
            "    enough context — DO NOT call read_file. Pick the best "
            "    hit and answer in one sentence. "
            "(2) Only call read_file when NO summary is present and the "
            "    filename alone is insufficient. "
            "(3) Call read_file at most ONCE per query, on the top-scoring "
            "    hit. Do not invent paths.";
      }
      messages.push_back(
          {{"role", "system"}, {"content", system_content}});
      messages.push_back(
          {{"role", "user"}, {"content", "Query: " + q}});

      size_t q_prompt = 0, q_comp = 0;
      int q_sem = 0, q_read = 0;
      auto t0q = std::chrono::steady_clock::now();
      bool stopped_cleanly = false;
      std::string q_answer;

      for (int iter = 0; iter < kMaxIters; ++iter) {
        nlohmann::json body = {{"model", model},
                               {"messages", messages},
                               {"tools", tools},
                               {"temperature", 0.0},
                               {"max_tokens", 512}};
        auto res = cli.Post(base + "/chat/completions", body.dump(),
                            "application/json");
        if (!res || res->status != 200) {
          std::cout << "  [q#" << (qi + 1) << " iter " << iter
                    << "] HTTP " << (res ? res->status : -1) << "\n";
          break;
        }
        auto parsed = nlohmann::json::parse(res->body, nullptr, false);
        if (parsed.is_discarded()) break;
        if (parsed.contains("usage")) {
          q_prompt += parsed["usage"].value("prompt_tokens", (size_t)0);
          q_comp += parsed["usage"].value("completion_tokens", (size_t)0);
        }
        if (!parsed.contains("choices") || parsed["choices"].empty()) break;
        auto &choice = parsed["choices"][0];
        if (!choice.contains("message")) break;
        auto msg = choice["message"];
        messages.push_back(msg);

        bool has_tool_calls = msg.contains("tool_calls") &&
                              msg["tool_calls"].is_array() &&
                              !msg["tool_calls"].empty();
        if (!has_tool_calls) {
          if (msg.contains("content") && msg["content"].is_string()) {
            q_answer = msg["content"].get<std::string>();
          }
          stopped_cleanly = true;
          break;
        }

        // Helper: tolerantly coerce a JSON value to an int with a default.
        auto int_or = [](const nlohmann::json &j, const std::string &key,
                         int fallback) -> int {
          if (!j.contains(key)) return fallback;
          const auto &v = j.at(key);
          if (v.is_number_integer()) return v.get<int>();
          if (v.is_number_float()) return static_cast<int>(v.get<double>());
          if (v.is_string()) {
            try { return std::stoi(v.get<std::string>()); }
            catch (...) { return fallback; }
          }
          return fallback;
        };

        for (auto &tc : msg["tool_calls"]) {
          std::string tc_id = tc.value("id", "");
          if (!tc.contains("function")) continue;
          std::string fn_name = tc["function"].value("name", "");
          std::string args_str = tc["function"].value("arguments", "{}");
          auto args = nlohmann::json::parse(args_str, nullptr, false);
          if (args.is_discarded() || !args.is_object())
            args = nlohmann::json::object();

          std::string tool_result = "{}";
          try {
          if (fn_name == "semantic_query") {
            ++q_sem;
            std::string q_arg =
                args.contains("query") && args["query"].is_string()
                    ? args["query"].get<std::string>()
                    : std::string();
            int k_arg = int_or(args, "k", 5);
            if (k_arg < 1) k_arg = 1;
            if (k_arg > 20) k_arg = 20;
            auto sq2 = client.AsyncSemanticQuery(
                q_arg, k_arg, chi::PoolQuery::Local());
            sq2.Wait();
            nlohmann::json results = nlohmann::json::array();
            for (size_t i = 0; i < sq2->result_tags_.size(); ++i) {
              auto &tid = sq2->result_tags_[i];
              uint64_t key =
                  (static_cast<uint64_t>(tid.major_) << 32) | tid.minor_;
              auto pit = tag_to_path.find(key);
              if (pit == tag_to_path.end()) continue;
              nlohmann::json hit = {{"path", pit->second},
                                    {"score", sq2->result_scores_[i]}};
              if (level == 2) {
                auto sit = path_to_summary.find(pit->second);
                if (sit != path_to_summary.end()) {
                  std::string s = sit->second;
                  if (s.size() > 1500) s.resize(1500);
                  hit["summary"] = s;
                }
              }
              results.push_back(hit);
            }
            tool_result = results.dump();
          } else if (fn_name == "read_file") {
            ++q_read;
            std::string p_arg =
                args.contains("path") && args["path"].is_string()
                    ? args["path"].get<std::string>()
                    : std::string();
            if (std::getenv("ACROPOLIS_BENCH_TRACE_TOOLS")) {
              std::cout << "    [tool] read_file(" << p_arg << ")\n";
            }
            int mb = int_or(args, "max_bytes", 4096);
            if (mb <= 0) mb = 4096;
            if (mb > 16 * 1024) mb = 16 * 1024;
            size_t max_bytes = static_cast<size_t>(mb);
            std::ifstream f(p_arg, std::ios::binary);
            if (f) {
              std::string content((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
              if (content.size() > max_bytes) content.resize(max_bytes);
              nlohmann::json out = {{"path", p_arg}, {"content", content}};
              tool_result = out.dump();
            } else {
              tool_result = R"({"error":"cannot open file"})";
            }
          } else if (fn_name == "list_files") {
            ++q_read;  // count as a tool call for bookkeeping
            if (std::getenv("ACROPOLIS_BENCH_TRACE_TOOLS")) {
              std::cout << "    [tool] list_files(" << args.dump() << ")\n";
            }
            std::string d_arg =
                args.contains("dir") && args["dir"].is_string()
                    ? args["dir"].get<std::string>()
                    : std::string("/workspace");
            if (d_arg.empty() || d_arg == "/") d_arg = repo_dir;
            bool rec = args.contains("recursive") &&
                       args["recursive"].is_boolean() &&
                       args["recursive"].get<bool>();
            int me = int_or(args, "max_entries", 2000);
            if (me < 1) me = 200;
            if (me > 4000) me = 4000;
            nlohmann::json out = nlohmann::json::array();
            try {
              if (rec) {
                for (auto it = fs::recursive_directory_iterator(
                         d_arg, fs::directory_options::skip_permission_denied);
                     it != fs::recursive_directory_iterator(); ++it) {
                  if (it->is_directory() && ShouldSkipDir(it->path())) {
                    it.disable_recursion_pending();
                    continue;
                  }
                  if (!it->is_regular_file()) continue;
                  out.push_back(it->path().string());
                  if (static_cast<int>(out.size()) >= me) break;
                }
              } else {
                for (auto &e : fs::directory_iterator(d_arg)) {
                  out.push_back(e.path().string());
                  if (static_cast<int>(out.size()) >= me) break;
                }
              }
              tool_result = out.dump();
            } catch (const std::exception &e) {
              tool_result = nlohmann::json(
                                {{"error", std::string("list_files: ") + e.what()}})
                                .dump();
            }
          } else {
            tool_result = R"({"error":"unknown tool"})";
          }
          } catch (const std::exception &e) {
            tool_result = nlohmann::json(
                              {{"error", std::string("tool exception: ") + e.what()}})
                              .dump();
          }

          messages.push_back({{"role", "tool"},
                              {"tool_call_id", tc_id},
                              {"content", tool_result}});
        }
      }

      auto t1q = std::chrono::steady_clock::now();
      agent_real_wall_s +=
          std::chrono::duration<double>(t1q - t0q).count();
      agent_real_prompt += q_prompt;
      agent_real_completion += q_comp;
      agent_real_sem_calls += q_sem;
      agent_real_read_calls += q_read;
      if (stopped_cleanly) ++agent_real_queries;

      // Grade agent correctness: expected substring in the answer text.
      const std::string &expected = queries[qi].expected_substring;
      bool agent_hit = !q_answer.empty() &&
                       q_answer.find(expected) != std::string::npos;
      if (agent_hit) ++agent_real_correct;

      // Collapse multi-line answer to single line for log readability.
      std::string preview = q_answer;
      for (auto &c : preview) if (c == '\n' || c == '\r') c = ' ';
      if (preview.size() > 160) preview.resize(160);

      std::cout << "  q#" << (qi + 1) << ": "
                << "sem_q=" << q_sem << " read=" << q_read
                << " prompt=" << q_prompt << " comp=" << q_comp
                << (agent_hit ? " HIT" : " miss")
                << (stopped_cleanly ? "" : " (capped iters)")
                << "\n    answer: \"" << preview << "\"\n";
    }
    std::cout << "agent queries succeeded: " << agent_real_queries
              << "/" << queries.size() << "\n";
  }

  // --- Summary ---
  std::cout << "=== Summary (level=" << level << ") ===\n"
            << "files ingested:    " << ingested << "\n"
            << "ingest time:       " << ingest_sec << " s\n"
            << "queries:           " << queries.size() << "\n"
            << "precision@1:       " << top1 << "/" << queries.size()
            << " (" << (100.0 * top1 / queries.size()) << "%)\n"
            << "precision@5:       " << top5 << "/" << queries.size()
            << " (" << (100.0 * top5 / queries.size()) << "%)\n"
            << "avg query latency: " << (total_q_ms / queries.size())
            << " ms\n";

  if (agent_real_queries > 0) {
    std::cout << "agent_real@5:      prompt=" << agent_real_prompt
              << " completion=" << agent_real_completion
              << " total=" << (agent_real_prompt + agent_real_completion)
              << " wall=" << agent_real_wall_s << " s"
              << " sem_q=" << agent_real_sem_calls
              << " read=" << agent_real_read_calls
              << " correct=" << agent_real_correct
              << "/" << queries.size()
              << " (n=" << agent_real_queries << ")\n";
  }

  return 0;
}
