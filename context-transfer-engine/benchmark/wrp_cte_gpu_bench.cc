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
 * CTE GPU Benchmark — Main driver
 *
 * CLI entry point for GPU benchmark suite. Parses arguments and dispatches
 * to workload functions defined in gpu/workload_*.cc files.
 *
 * Supports multiple modes:
 *   putblob     -- GPU client -> CTE via GPU->CPU path (ToLocalCpu)
 *   putblob_gpu -- GPU client -> CTE via GPU-local path (Local)
 *   putget_gpu  -- GPU client -> CTE PutBlob + GetBlob round-trip
 *   direct      -- GPU kernel writes directly to pinned host memory (baseline)
 *   cudamemcpy  -- cudaMemcpyAsync baseline (theoretical PCIe max)
 *   managed     -- CUDA managed memory write + prefetch to host
 *   bam_read    -- BaM page cache: GPU reads from DRAM through HBM cache
 *   bam_write   -- BaM page cache: GPU writes to DRAM through HBM cache
 *   bdev_alloc_free  -- Block device alloc/free throughput
 *   bdev_read_write  -- Block device read/write throughput
 *   alloc_test       -- Multi-block allocator stress test
 *   pagerank         -- PageRank workload (graph algorithm)
 *   gnn              -- GNN feature gather workload
 *   gray_scott       -- Gray-Scott stencil simulation workload
 *   llm_kvcache      -- LLM KV cache offloading workload
 *   synthetic        -- Synthetic I/O workload
 *
 * Usage:
 *   wrp_cte_gpu_bench [options]
 */

#include <hermes_shm/constants/macros.h>
#if HSHM_IS_HOST

#include "gpu/workload.h"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cstdlib>
#include <string>
#include <thread>

using namespace std::chrono_literals;

//==============================================================================
// CLI and main()
//==============================================================================

enum class TestCase { kClientOverhead, kClientOverheadCpu,
  kWorkloadPageRank, kWorkloadGNN, kWorkloadGrayScott, kWorkloadLLMKVCache, kWorkloadSynthetic,
  kCudaMalloc };

struct BenchConfig {
  TestCase test_case = TestCase::kClientOverhead;
  chi::u32 rt_blocks = 1;
  chi::u32 rt_threads = 32;
  chi::u32 client_blocks = 1;
  chi::u32 client_threads = 32;
  chi::u64 warp_bytes = 128 * 1024;  // per-warp I/O size
  chi::u32 iterations = 16;          // iterations per warp
  int timeout_sec = 60;              // PollDone timeout in seconds
  // I/O options
  bool validate = false;
  std::string io_pattern = "sequential";  // "sequential" or "random"
  std::string routing = "local";          // "local" or "to_cpu"
  // Workload mode: hbm, direct, cte, bam
  std::string workload_mode = "hbm";
  // CTE storage targets (populated via --target flags)
  std::vector<TargetSpec> targets;
  // Workload-specific parameters
  chi::u32 param_vertices = 100000;
  chi::u32 param_avg_degree = 16;
  chi::u32 param_num_nodes = 500000;
  chi::u32 param_emb_dim = 128;
  chi::u32 param_batch_size = 1024;
  chi::u32 param_grid_size = 128;
  chi::u32 param_steps = 100;
  chi::u32 param_checkpoint_freq = 10;
  chi::u32 param_num_layers = 12;
  chi::u32 param_num_heads = 12;
  chi::u32 param_head_dim = 64;
  chi::u32 param_seq_len = 2048;
  chi::u32 param_decode_tokens = 32;
};

namespace {

chi::u64 ParseSize(const std::string &s) {
  double val = 0.0;
  chi::u64 mult = 1;
  std::string num;
  char suffix = 0;
  for (char c : s) {
    if (std::isdigit(c) || c == '.') num += c;
    else { suffix = std::tolower(c); break; }
  }
  if (num.empty()) return 0;
  val = std::stod(num);
  switch (suffix) {
    case 'k': mult = 1024; break;
    case 'm': mult = 1024 * 1024; break;
    case 'g': mult = 1024ULL * 1024 * 1024; break;
    default: break;
  }
  return static_cast<chi::u64>(val * mult);
}

void PrintUsage(const char *prog) {
  HIPRINT("Usage: {} [options]", prog);
  HIPRINT("Options:");
  HIPRINT("  --test-case <case>     client_overhead, client_overhead_cpu, cuda_malloc,");
  HIPRINT("                         pagerank, gnn, gray_scott, llm_kvcache, synthetic");
  HIPRINT("  --workload-mode <m>    For workloads: hbm, direct, bam, or cte (default: hbm)");
  HIPRINT("  --rt-blocks <N>        GPU runtime orchestrator blocks (default: 1)");
  HIPRINT("  --rt-threads <N>       GPU runtime threads/block (default: 32)");
  HIPRINT("  --client-blocks <N>    GPU client kernel blocks (default: 1)");
  HIPRINT("  --client-threads <N>   GPU client kernel threads/block (default: 32)");
  HIPRINT("  --io-size <bytes>      Per-warp I/O size (default: 128K, supports k/m/g suffixes)");
  HIPRINT("  --iterations <N>       Iterations per warp (default: 16)");
  HIPRINT("  --timeout <seconds>    PollDone timeout in seconds (default: 60)");
  HIPRINT("  --validate             Enable data validation after reads");
  HIPRINT("  --io-pattern <p>       I/O pattern: sequential or random (default: sequential)");
  HIPRINT("  --routing <r>          Task routing: local or to_cpu (default: local)");
  HIPRINT("  --target <type:size>   CTE storage target (repeatable). type: hbm, pinned, ram");
  HIPRINT("                         Example: --target hbm:256m --target pinned:256m");
  HIPRINT("                         If omitted, defaults to one target based on --hbm-cache");
  HIPRINT("  --help, -h             Show this help");
}

/** Parse "type:size" into a TargetSpec. Returns true on success. */
bool ParseTarget(const std::string &spec, TargetSpec &out) {
  auto colon = spec.find(':');
  if (colon == std::string::npos || colon == 0) return false;
  std::string type_str = spec.substr(0, colon);
  std::string size_str = spec.substr(colon + 1);
  if (type_str == "hbm") {
    out.bdev_type = chimaera::bdev::BdevType::kHbm;
  } else if (type_str == "pinned") {
    out.bdev_type = chimaera::bdev::BdevType::kPinned;
  } else if (type_str == "ram") {
    out.bdev_type = chimaera::bdev::BdevType::kRam;
  } else {
    return false;
  }
  out.label = type_str;
  out.size_bytes = ParseSize(size_str);
  return out.size_bytes > 0;
}

bool ParseArgs(int argc, char **argv, BenchConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--test-case" && i + 1 < argc) {
      std::string tc = argv[++i];
      if (tc == "client_overhead") cfg.test_case = TestCase::kClientOverhead;
      else if (tc == "client_overhead_cpu") cfg.test_case = TestCase::kClientOverheadCpu;
      else if (tc == "pagerank") cfg.test_case = TestCase::kWorkloadPageRank;
      else if (tc == "gnn") cfg.test_case = TestCase::kWorkloadGNN;
      else if (tc == "gray_scott") cfg.test_case = TestCase::kWorkloadGrayScott;
      else if (tc == "llm_kvcache") cfg.test_case = TestCase::kWorkloadLLMKVCache;
      else if (tc == "synthetic") cfg.test_case = TestCase::kWorkloadSynthetic;
      else if (tc == "cuda_malloc") cfg.test_case = TestCase::kCudaMalloc;
      else {
        HLOG(kError, "Unknown test case '{}'", tc);
        return false;
      }
    } else if (arg == "--rt-blocks" && i + 1 < argc) {
      cfg.rt_blocks = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--rt-threads" && i + 1 < argc) {
      cfg.rt_threads = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--client-blocks" && i + 1 < argc) {
      cfg.client_blocks = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--client-threads" && i + 1 < argc) {
      cfg.client_threads = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--io-size" && i + 1 < argc) {
      cfg.warp_bytes = ParseSize(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      cfg.iterations = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--timeout" && i + 1 < argc) {
      cfg.timeout_sec = static_cast<int>(std::stol(argv[++i]));
    } else if (arg == "--workload-mode" && i + 1 < argc) {
      cfg.workload_mode = argv[++i];
    } else if (arg == "--validate") {
      cfg.validate = true;
    } else if (arg == "--io-pattern" && i + 1 < argc) {
      cfg.io_pattern = argv[++i];
    } else if (arg == "--routing" && i + 1 < argc) {
      cfg.routing = argv[++i];
    } else if (arg == "--vertices" && i + 1 < argc) {
      cfg.param_vertices = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--avg-degree" && i + 1 < argc) {
      cfg.param_avg_degree = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--num-nodes" && i + 1 < argc) {
      cfg.param_num_nodes = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--emb-dim" && i + 1 < argc) {
      cfg.param_emb_dim = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--batch-size" && i + 1 < argc) {
      cfg.param_batch_size = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--grid-size" && i + 1 < argc) {
      cfg.param_grid_size = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--steps" && i + 1 < argc) {
      cfg.param_steps = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--checkpoint-freq" && i + 1 < argc) {
      cfg.param_checkpoint_freq = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--num-layers" && i + 1 < argc) {
      cfg.param_num_layers = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--num-heads" && i + 1 < argc) {
      cfg.param_num_heads = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--head-dim" && i + 1 < argc) {
      cfg.param_head_dim = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--seq-len" && i + 1 < argc) {
      cfg.param_seq_len = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--decode-tokens" && i + 1 < argc) {
      cfg.param_decode_tokens = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--target" && i + 1 < argc) {
      TargetSpec ts;
      if (!ParseTarget(argv[++i], ts)) {
        HLOG(kError, "Invalid --target '{}'. Format: type:size (hbm:256m)", argv[i]);
        return false;
      }
      cfg.targets.push_back(ts);
    } else {
      HLOG(kError, "Unknown option: {}", arg);
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  BenchConfig cfg;
  if (!ParseArgs(argc, argv, cfg)) return 1;

  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    HLOG(kError, "No GPUs available");
    return 1;
  }

  // gpu_putblob_kernel uses ~145 registers/thread.  With 65536 regs/block
  // on most GPUs, max threads/block ≈ 448 → cap at 256 (8 warps) and
  // redistribute excess warps across multiple blocks.
  {
    constexpr chi::u32 kMaxClientThreads = 256;
    chi::u32 total_client_threads = cfg.client_blocks * cfg.client_threads;
    if (cfg.client_threads > kMaxClientThreads && total_client_threads > kMaxClientThreads) {
      cfg.client_blocks = (total_client_threads + kMaxClientThreads - 1) / kMaxClientThreads;
      cfg.client_threads = kMaxClientThreads;
    }
  }

  // cuda_malloc benchmark: standalone, no Chimaera runtime needed
  if (cfg.test_case == TestCase::kCudaMalloc) {
    return run_cuda_malloc_bench(cfg.client_blocks, cfg.client_threads,
                                 cfg.iterations);
  }

  // Load GPU config if CHI_SERVER_CONF is not already set
  if (!std::getenv("CHI_SERVER_CONF")) {
    std::string config_dir = std::string(__FILE__);
    config_dir = config_dir.substr(0, config_dir.rfind('/'));
    std::string gpu_config = config_dir + "/cte_config_gpu.yaml";
    setenv("CHI_SERVER_CONF", gpu_config.c_str(), 1);
    HLOG(kInfo, "Using GPU benchmark config: {}", gpu_config);
  }

  // Set GPU orchestrator dimensions from benchmark parameters so the heap
  // partition count matches the warp count from the start.
  setenv("CHI_GPU_BLOCKS", std::to_string(cfg.rt_blocks).c_str(), 1);
  setenv("CHI_GPU_THREADS", std::to_string(cfg.rt_threads).c_str(), 1);

  // Initialize Chimaera runtime
  HLOG(kInfo, "Initializing Chimaera runtime...");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }
  std::this_thread::sleep_for(500ms);

  const char *tc_name = (cfg.test_case == TestCase::kClientOverhead) ? "client_overhead" :
                         (cfg.test_case == TestCase::kClientOverheadCpu) ? "client_overhead_cpu" :
                         (cfg.test_case == TestCase::kWorkloadPageRank) ? "pagerank" :
                         (cfg.test_case == TestCase::kWorkloadGNN) ? "gnn" :
                         (cfg.test_case == TestCase::kWorkloadGrayScott) ? "gray_scott" :
                         (cfg.test_case == TestCase::kWorkloadLLMKVCache) ? "llm_kvcache" :
                         (cfg.test_case == TestCase::kWorkloadSynthetic) ? "synthetic" :
                         (cfg.test_case == TestCase::kCudaMalloc) ? "cuda_malloc" :
                         "unknown";

  HIPRINT("\n=== CTE GPU Benchmark ===");
  HIPRINT("Test case:           {}", tc_name);
  HIPRINT("RT blocks:           {}", cfg.rt_blocks);
  HIPRINT("RT threads/block:    {}", cfg.rt_threads);
  HIPRINT("Client blocks:       {}", cfg.client_blocks);
  HIPRINT("Client threads/block:{}", cfg.client_threads);
  HIPRINT("Per-warp I/O size:   {} bytes ({} KB)", cfg.warp_bytes,
          cfg.warp_bytes / 1024);
  HIPRINT("Iterations:          {}", cfg.iterations);
  HIPRINT("Timeout:             {} seconds", cfg.timeout_sec);

  // Compute total I/O: per-warp size x warps x iterations
  chi::u32 client_warps = (cfg.client_blocks * cfg.client_threads) / 32;
  if (client_warps == 0) client_warps = 1;
  chi::u64 total_bytes = cfg.warp_bytes * client_warps * cfg.iterations;
  HIPRINT("Total I/O size:      {} bytes ({} MB)", total_bytes,
          total_bytes / (1024 * 1024));

  float elapsed_ms = 0;
  int rc = 0;

  // --- Workload benchmarks (PageRank, GNN, Gray-Scott, LLM KV cache) ---
  if (cfg.test_case == TestCase::kWorkloadPageRank ||
      cfg.test_case == TestCase::kWorkloadGNN ||
      cfg.test_case == TestCase::kWorkloadGrayScott ||
      cfg.test_case == TestCase::kWorkloadLLMKVCache ||
      cfg.test_case == TestCase::kWorkloadSynthetic) {
    WorkloadConfig wcfg;
    wcfg.rt_blocks = cfg.rt_blocks;
    wcfg.rt_threads = cfg.rt_threads;
    wcfg.client_blocks = cfg.client_blocks;
    wcfg.client_threads = cfg.client_threads;
    wcfg.iterations = cfg.iterations;
    wcfg.timeout_sec = cfg.timeout_sec;
    wcfg.targets = cfg.targets;
    wcfg.param_vertices = cfg.param_vertices;
    wcfg.param_avg_degree = cfg.param_avg_degree;
    wcfg.param_num_nodes = cfg.param_num_nodes;
    wcfg.param_emb_dim = cfg.param_emb_dim;
    wcfg.param_batch_size = cfg.param_batch_size;
    wcfg.param_grid_size = cfg.param_grid_size;
    wcfg.param_steps = cfg.param_steps;
    wcfg.param_checkpoint_freq = cfg.param_checkpoint_freq;
    wcfg.param_num_layers = cfg.param_num_layers;
    wcfg.param_num_heads = cfg.param_num_heads;
    wcfg.param_head_dim = cfg.param_head_dim;
    wcfg.param_seq_len = cfg.param_seq_len;
    wcfg.param_decode_tokens = cfg.param_decode_tokens;
    wcfg.warp_bytes = cfg.warp_bytes;
    wcfg.validate = cfg.validate;
    wcfg.routing = cfg.routing;
    wcfg.io_pattern = (cfg.io_pattern == "random") ? IoPattern::kRandom
                                                    : IoPattern::kSequential;

    // For CTE mode: use the same pool setup as putblob_gpu
    if (cfg.workload_mode == "cte") {

      chi::PoolId gpu_pool_id(wrp_cte::core::kCtePoolId.major_ + 1,
                               wrp_cte::core::kCtePoolId.minor_);
      wrp_cte::core::Client cte_client(gpu_pool_id);
      wrp_cte::core::CreateParams params;
      auto create_task = cte_client.AsyncCreate(
          chi::PoolQuery::Dynamic(),
          "cte_workload_pool", gpu_pool_id, params);

      create_task.Wait();

      if (create_task->GetReturnCode() != 0) {
        HLOG(kError, "Failed to create CTE workload pool: {}",
             create_task->GetReturnCode());
        return 1;
      }
      std::this_thread::sleep_for(200ms);

      // Build target list: use --target specs, or default to 256MB HBM
      std::vector<TargetSpec> targets = cfg.targets;
      if (targets.empty()) {
        TargetSpec ts;
        ts.bdev_type = chimaera::bdev::BdevType::kHbm;
        ts.label = "hbm";
        ts.size_bytes = 256ULL * 1024 * 1024;
        targets.push_back(ts);
      }

      // Register each storage target (CPU + GPU)
      chi::PoolId bdev_pool_id(800, 0);
      for (size_t ti = 0; ti < targets.size(); ti++) {
        const auto &ts = targets[ti];
        std::string target_name = ts.label + "::workload_target_" + std::to_string(ti);
        HIPRINT("  Registering target: {} ({} bytes)", target_name, ts.size_bytes);

        auto reg_task = cte_client.AsyncRegisterTarget(
            target_name, ts.bdev_type, ts.size_bytes,
            chi::PoolQuery::Local(), bdev_pool_id);
        reg_task.Wait();
        if (reg_task->GetReturnCode() != 0) {
          HLOG(kError, "Failed to register target {}: {}",
               target_name, reg_task->GetReturnCode());
          return 1;
        }
        std::this_thread::sleep_for(200ms);

        auto gpu_reg_task = cte_client.AsyncRegisterTarget(
            target_name, ts.bdev_type, ts.size_bytes,
            chi::PoolQuery::Local(), bdev_pool_id,
            chi::PoolQuery::LocalGpuBcast());
        gpu_reg_task.Wait();
        if (gpu_reg_task->GetReturnCode() != 0) {
          HLOG(kError, "Failed to register GPU target {}: {}",
               target_name, gpu_reg_task->GetReturnCode());
          return 1;
        }
        std::this_thread::sleep_for(200ms);
      }

      auto tag_task = cte_client.AsyncGetOrCreateTag(
          "workload_tag", wrp_cte::core::TagId::GetNull(),
          chi::PoolQuery::Local());
      tag_task.Wait();
      if (tag_task->GetReturnCode() != 0) {
        HLOG(kError, "Failed to create tag: {}", tag_task->GetReturnCode());
        return 1;
      }
      wcfg.tag_id = tag_task->tag_id_;

      auto gpu_tag_task = cte_client.AsyncGetOrCreateTag(
          "workload_tag", wcfg.tag_id, chi::PoolQuery::LocalGpuBcast());
      gpu_tag_task.Wait();
      std::this_thread::sleep_for(200ms);

      wcfg.cte_pool_id = cte_client.pool_id_;
      HIPRINT("CTE pool: {}.{}, tag: {}.{}",
              wcfg.cte_pool_id.major_, wcfg.cte_pool_id.minor_,
              wcfg.tag_id.major_, wcfg.tag_id.minor_);
    } else {
      // Pause GPU orchestrator for non-CTE workload modes
      CHI_IPC->PauseGpuOrchestrator();
    }

    WorkloadResult wresult = {};
    const char *wmode = cfg.workload_mode.c_str();

    HIPRINT("Workload mode:       {}", wmode);

    if (cfg.test_case == TestCase::kWorkloadPageRank) {
      HIPRINT("Workload:            PageRank (graph algorithm)");
      rc = run_workload_pagerank(wcfg, wmode, &wresult);
    } else if (cfg.test_case == TestCase::kWorkloadGNN) {
      HIPRINT("Workload:            GNN feature gather");
      rc = run_workload_gnn(wcfg, wmode, &wresult);
    } else if (cfg.test_case == TestCase::kWorkloadGrayScott) {
      HIPRINT("Workload:            Gray-Scott stencil simulation");
      rc = run_workload_gray_scott(wcfg, wmode, &wresult);
    } else if (cfg.test_case == TestCase::kWorkloadLLMKVCache) {
      HIPRINT("Workload:            LLM KV cache offloading");
      rc = run_workload_llm_kvcache(wcfg, wmode, &wresult);
    } else if (cfg.test_case == TestCase::kWorkloadSynthetic) {
      HIPRINT("Workload:            Synthetic I/O");
      rc = run_workload_synthetic(wcfg, wmode, &wresult);
    }

    if (rc != 0) {
      HLOG(kError, "Workload failed with error: {}", rc);
      return 1;
    }

    printf("\n=== %s (%s) Results ===\n", tc_name, wmode);
    printf("Elapsed:             %.3f ms\n", wresult.elapsed_ms);
    printf("Bandwidth:           %.3f GB/s\n", wresult.bandwidth_gbps);
    if (wresult.metric_name) {
      printf("%-20s %.2e\n", wresult.metric_name, wresult.primary_metric);
    }
    printf("=========================\n");
    return 0;

  } else {
    // CTE client overhead benchmark — measure AsyncPutBlob submission cost
    bool to_cpu = (cfg.test_case == TestCase::kClientOverheadCpu);

    // Use the compose-created CTE pool directly (avoids creating a second pool
    // which triggers nested coroutine issues with FlushData).
    chi::PoolId gpu_pool_id = wrp_cte::core::kCtePoolId;
    wrp_cte::core::Client cte_client(gpu_pool_id);
    HIPRINT("Using compose CTE pool: ({},{})", gpu_pool_id.major_, gpu_pool_id.minor_);

    // Compose creates CTE before the GPU orchestrator launches, so the CTE
    // GPU container is never registered. Register it now manually.
    {
      bool did_pause = CHI_IPC->PauseGpuOrchestrator();
      if (did_pause) {
        void *gpu_cte = CHI_IPC->AllocGpuContainer(gpu_pool_id, 0, "wrp_cte_core");
        if (gpu_cte) {
          CHI_IPC->RegisterGpuOrchestratorContainer(gpu_pool_id, gpu_cte);
          HIPRINT("Registered CTE GPU container for pool ({},{})",
                  gpu_pool_id.major_, gpu_pool_id.minor_);
        }
        CHI_IPC->ResumeGpuOrchestrator();
      }
    }
    std::this_thread::sleep_for(200ms);

    // Use compose's ram bdev pool (301,0) as the GPU target backing store.
    chi::PoolId bdev_pool_id(301, 0);  // ram::chi_default_bdev from compose
    chi::u64 target_size = 64ULL * 1024 * 1024;
    if (!cfg.targets.empty()) target_size = cfg.targets[0].size_bytes;

    // Register target on GPU via SendCpuToGpu (POD copy path)
    HIPRINT("Registering GPU target via SendCpuToGpu...");
    {
      auto *ipc = CHI_IPC;
      auto reg_task = ipc->NewTask<wrp_cte::core::RegisterTargetTask>(
          chi::CreateTaskId(), gpu_pool_id,
          chi::PoolQuery::LocalGpuBcast(),
          "gpu_bench_target",
          chimaera::bdev::BdevType::kRam,
          target_size,
          chi::PoolQuery::Local(), bdev_pool_id);
      auto reg_future = ipc->Send(reg_task);
      printf("[BENCH] Sent RegisterTarget, waiting...\n"); fflush(stdout);
      bool ok = reg_future.Wait(5.0f);
      printf("[BENCH] Wait returned: ok=%d rc=%d\n", ok, (int)reg_task->return_code_); fflush(stdout);
      if (!ok || reg_task->return_code_ != 0) {
        HLOG(kError, "GPU RegisterTarget failed: ok={} rc={}",
             ok, (int)reg_task->return_code_);
        return 1;
      }
      HIPRINT("GPU RegisterTarget OK");
    }

    // Create tag on GPU via SendCpuToGpu
    wrp_cte::core::TagId tag_id;
    HIPRINT("Creating GPU tag via SendCpuToGpu...");
    {
      auto *ipc = CHI_IPC;
      auto tag_task = ipc->NewTask<wrp_cte::core::GetOrCreateTagTask<wrp_cte::core::CreateParams>>(
          chi::CreateTaskId(), gpu_pool_id,
          chi::PoolQuery::LocalGpuBcast(),
          "gpu_bench_tag",
          wrp_cte::core::TagId::GetNull());
      auto tag_future = ipc->Send(tag_task);
      bool ok = tag_future.Wait(10.0f);
      if (!ok || tag_task->return_code_ != 0) {
        HLOG(kError, "GPU GetOrCreateTag failed: ok={} rc={}",
             ok, (int)tag_task->return_code_);
        return 1;
      }
      tag_id = tag_task->tag_id_;
      HIPRINT("GPU GetOrCreateTag OK, tag_id=({},{})", tag_id.major_, tag_id.minor_);
    }
    std::this_thread::sleep_for(200ms);
    std::this_thread::sleep_for(200ms);

    HIPRINT("Pool ID: {}.{}", gpu_pool_id.major_, gpu_pool_id.minor_);
    HIPRINT("Tag ID:  {}.{}", tag_id.major_, tag_id.minor_);
    HIPRINT("Routing: {}", to_cpu ? "ToLocalCpu" : "Local");

    float avg_submit_us = 0;
    rc = run_cte_client_overhead(
        cte_client.pool_id_, tag_id,
        cfg.rt_blocks, cfg.rt_threads,
        cfg.client_blocks, cfg.client_threads,
        cfg.warp_bytes, cfg.iterations, to_cpu,
        cfg.timeout_sec, &elapsed_ms, &avg_submit_us);

    if (rc != 0) {
      HLOG(kError, "Benchmark failed with error: {}", rc);
      return 1;
    }

    printf("\n=== %s Results ===\n", tc_name);
    printf("Wall elapsed:        %.3f ms\n", static_cast<double>(elapsed_ms));
    printf("Avg submit cost:     %.1f us/call\n", static_cast<double>(avg_submit_us));
    printf("=========================\n");
    return 0;
  }

  return 0;
}

#endif  // HSHM_IS_HOST
