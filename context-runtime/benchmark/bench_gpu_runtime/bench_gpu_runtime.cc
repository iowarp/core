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
 * GPU Runtime Latency Benchmark — CPU driver
 *
 * Initializes the Chimaera runtime in server mode, creates a MOD_NAME pool,
 * then delegates to the GPU benchmark wrapper (run_gpu_bench_latency) to
 * launch a GPU client kernel against the GPU work orchestrator.
 *
 * Benchmark parameters:
 *   --test-case <case>        Only "latency" is accepted (default: latency)
 *   --rt-blocks <N>           GPU runtime (orchestrator) blocks (default: 1)
 *   --rt-threads <N>          GPU runtime threads per block (default: 32)
 *   --client-blocks <N>       GPU client kernel blocks (default: 1)
 *   --client-threads <N>      GPU client kernel threads per block (default: 32)
 *   --batch-size <N>          Tasks per batch per GPU thread (default: 1)
 *   --total-tasks <N>         Total tasks per GPU thread (default: 100)
 */

#include <chimaera/chimaera.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

using namespace std::chrono_literals;

// Forward declaration of GPU benchmark wrapper (defined in bench_gpu_runtime_gpu.cc)
// BENCH_GPU_KERNELS_COMPILED is set by CMake when building with CUDA/ROCm support.
// The CPU source is compiled with HSHM_ENABLE_CUDA=0 to suppress __device__ annotations,
// so we use this separate flag to detect whether GPU kernels are actually linked in.
#if BENCH_GPU_KERNELS_COMPILED
extern "C" int run_gpu_bench_latency(chi::PoolId pool_id,
                                      chi::u32 method_id,
                                      chi::u32 rt_blocks,
                                      chi::u32 rt_threads,
                                      chi::u32 client_blocks,
                                      chi::u32 client_threads,
                                      chi::u32 batch_size,
                                      chi::u32 total_tasks,
                                      float *out_elapsed_ms);
extern "C" int run_gpu_bench_coroutine(chi::PoolId pool_id,
                                        chi::u32 rt_blocks,
                                        chi::u32 rt_threads,
                                        chi::u32 client_blocks,
                                        chi::u32 client_threads,
                                        chi::u32 total_tasks,
                                        chi::u32 subtasks,
                                        float *out_elapsed_ms);
extern "C" int run_gpu_bench_alloc(chi::PoolId pool_id,
                                    chi::u32 client_blocks,
                                    chi::u32 client_threads,
                                    chi::u32 total_tasks,
                                    float *out_elapsed_ms);
extern "C" int run_gpu_bench_serde(chi::PoolId pool_id,
                                    chi::u32 client_blocks,
                                    chi::u32 client_threads,
                                    chi::u32 total_tasks,
                                    float *out_elapsed_ms);
extern "C" int run_gpu_bench_alloc_serde(chi::PoolId pool_id,
                                          chi::u32 client_blocks,
                                          chi::u32 client_threads,
                                          chi::u32 total_tasks,
                                          float *out_elapsed_ms);
extern "C" int run_gpu_bench_string_alloc(chi::u32 total_tasks,
                                           float *out_elapsed_ms);
extern "C" int run_gpu_bench_putblob(chi::PoolId cte_pool_id,
                                      wrp_cte::core::TagId tag_id,
                                      chi::u32 rt_blocks,
                                      chi::u32 rt_threads,
                                      chi::u32 client_blocks,
                                      chi::u32 client_threads,
                                      chi::u64 total_bytes,
                                      bool to_cpu,
                                      float *out_elapsed_ms);
#else
extern "C" __attribute__((weak)) int run_gpu_bench_latency(
    chi::PoolId, chi::u32, chi::u32, chi::u32, chi::u32, chi::u32,
    chi::u32, chi::u32, float *) {
  return -200;  // No GPU support compiled
}
extern "C" __attribute__((weak)) int run_gpu_bench_coroutine(
    chi::PoolId, chi::u32, chi::u32, chi::u32, chi::u32,
    chi::u32, chi::u32, float *) {
  return -200;  // No GPU support compiled
}
extern "C" __attribute__((weak)) int run_gpu_bench_alloc(
    chi::PoolId, chi::u32, chi::u32,
    chi::u32, float *) {
  return -200;  // No GPU support compiled
}
extern "C" __attribute__((weak)) int run_gpu_bench_serde(
    chi::PoolId, chi::u32, chi::u32,
    chi::u32, float *) {
  return -200;  // No GPU support compiled
}
extern "C" __attribute__((weak)) int run_gpu_bench_alloc_serde(
    chi::PoolId, chi::u32, chi::u32,
    chi::u32, float *) {
  return -200;  // No GPU support compiled
}
extern "C" __attribute__((weak)) int run_gpu_bench_string_alloc(
    chi::u32, float *) {
  return -200;  // No GPU support compiled
}
extern "C" __attribute__((weak)) int run_gpu_bench_putblob(
    chi::PoolId, wrp_cte::core::TagId, chi::u32, chi::u32,
    chi::u32, chi::u32, chi::u64, bool, float *) {
  return -200;  // No GPU support compiled
}
#endif

/** Supported benchmark test cases */
enum class TestCase { kLatency, kCoroutine, kAlloc, kAllocSerde, kSerde, kStringAlloc, kPutBlob, kPutBlobGpu };

/**
 * Configuration for the GPU runtime benchmark.
 * All fields have defaults matching the spec (latency, 1 rt block, 32 threads).
 */
struct BenchmarkConfig {
  TestCase test_case = TestCase::kLatency;  /**< Benchmark mode */
  chi::u32 rt_blocks = 1;       /**< GPU work orchestrator block count */
  chi::u32 rt_threads = 32;     /**< GPU work orchestrator threads per block */
  chi::u32 client_blocks = 1;   /**< GPU client kernel block count */
  chi::u32 client_threads = 32; /**< GPU client kernel threads per block */
  chi::u32 batch_size = 1;      /**< Tasks per batch per GPU thread */
  chi::u32 total_tasks = 100;   /**< Total tasks per GPU thread */
  chi::u32 subtasks = 1;        /**< Subtasks per coroutine task (coroutine test) */
  chi::u64 total_bytes = 64 * 1024 * 1024;  /**< Total I/O size in bytes (putblob test) */
};

/**
 * Print usage information and exit.
 *
 * @param prog Program name (argv[0])
 */
static void PrintHelp(const char *prog) {
  HIPRINT("Usage: {} [options]", prog);
  HIPRINT("Options:");
  HIPRINT("  --test-case <case>     Test case: 'latency', 'coroutine', 'alloc', 'alloc_serde', 'serde', 'string_alloc', 'putblob', or 'putblob_gpu' (default: latency)");
  HIPRINT("  --rt-blocks <N>        GPU runtime orchestrator blocks (default: 1)");
  HIPRINT("  --rt-threads <N>       GPU runtime orchestrator threads/block (default: 32)");
  HIPRINT("  --client-blocks <N>    GPU client kernel blocks (default: 1)");
  HIPRINT("  --client-threads <N>   GPU client kernel threads/block (default: 32)");
  HIPRINT("  --batch-size <N>       Tasks per batch per GPU thread (default: 1)");
  HIPRINT("  --total-tasks <N>      Total tasks per GPU thread (default: 100)");
  HIPRINT("  --subtasks <N>         Subtasks per coroutine task (default: 1)");
  HIPRINT("  --io-size <bytes>      Total I/O size in bytes (putblob test, default: 67108864)");
  HIPRINT("  --help, -h             Show this help");
}

/**
 * Parse command-line arguments into BenchmarkConfig.
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @param cfg  Output config
 * @return true on success, false on error or --help
 */
static bool ParseArgs(int argc, char **argv, BenchmarkConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--help" || arg == "-h")) {
      PrintHelp(argv[0]);
      return false;
    } else if (arg == "--test-case" && i + 1 < argc) {
      std::string tc = argv[++i];
      if (tc == "latency") {
        cfg.test_case = TestCase::kLatency;
      } else if (tc == "coroutine") {
        cfg.test_case = TestCase::kCoroutine;
      } else if (tc == "alloc") {
        cfg.test_case = TestCase::kAlloc;
      } else if (tc == "alloc_serde") {
        cfg.test_case = TestCase::kAllocSerde;
      } else if (tc == "serde") {
        cfg.test_case = TestCase::kSerde;
      } else if (tc == "string_alloc") {
        cfg.test_case = TestCase::kStringAlloc;
      } else if (tc == "putblob") {
        cfg.test_case = TestCase::kPutBlob;
      } else if (tc == "putblob_gpu") {
        cfg.test_case = TestCase::kPutBlobGpu;
      } else {
        HLOG(kError, "Unknown test case '{}'; use 'latency', 'coroutine', 'alloc', 'alloc_serde', 'serde', 'string_alloc', 'putblob', or 'putblob_gpu'", tc);
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
    } else if (arg == "--batch-size" && i + 1 < argc) {
      cfg.batch_size = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--total-tasks" && i + 1 < argc) {
      cfg.total_tasks = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--subtasks" && i + 1 < argc) {
      cfg.subtasks = static_cast<chi::u32>(std::stoul(argv[++i]));
    } else if (arg == "--io-size" && i + 1 < argc) {
      cfg.total_bytes = static_cast<chi::u64>(std::stoull(argv[++i]));
    } else {
      HLOG(kError, "Unknown argument: {}", arg);
      return false;
    }
  }
  return true;
}

/**
 * Create the MOD_NAME pool used by the benchmark.
 *
 * @param pool_id  Desired pool ID
 * @return true on success
 */
static bool CreateBenchPool(const chi::PoolId &pool_id) {
  chimaera::MOD_NAME::Client client(pool_id);
  auto task = client.AsyncCreate(chi::PoolQuery::Dynamic(),
                                  "gpu_bench_pool", pool_id);
  task.Wait();
  if (task->return_code_ != 0) {
    HLOG(kError, "Failed to create MOD_NAME pool (rc={})", task->return_code_);
    return false;
  }
  return true;
}

/**
 * Print benchmark results including throughput and per-task latency.
 *
 * @param cfg         Benchmark configuration
 * @param elapsed_ms  Total elapsed time in ms
 */
static void PrintResults(const BenchmarkConfig &cfg, float elapsed_ms) {
  chi::u64 num_warps = (static_cast<chi::u64>(cfg.client_blocks) *
                         static_cast<chi::u64>(cfg.client_threads)) / 32;
  if (num_warps == 0) num_warps = 1;
  chi::u64 total_ops = num_warps * static_cast<chi::u64>(cfg.total_tasks);
  double throughput = (total_ops * 1000.0) / elapsed_ms;   // tasks/sec
  double latency_us = (elapsed_ms * 1000.0) / cfg.total_tasks; // us per task per warp

  const char *tc_name = (cfg.test_case == TestCase::kSerde) ? "serde" :
                         (cfg.test_case == TestCase::kAllocSerde) ? "alloc_serde" :
                         (cfg.test_case == TestCase::kAlloc) ? "alloc" :
                         (cfg.test_case == TestCase::kStringAlloc) ? "string_alloc" :
                         (cfg.test_case == TestCase::kCoroutine) ? "coroutine" :
                         (cfg.test_case == TestCase::kPutBlobGpu) ? "putblob_gpu" :
                         (cfg.test_case == TestCase::kPutBlob) ? "putblob" : "latency";
  HIPRINT("\n=== GPU Runtime Benchmark Results ===");
  HIPRINT("Test case:           {}", tc_name);
  HIPRINT("RT blocks:           {}", cfg.rt_blocks);
  HIPRINT("RT threads/block:    {}", cfg.rt_threads);
  HIPRINT("Client blocks:       {}", cfg.client_blocks);
  HIPRINT("Client threads/block:{}", cfg.client_threads);
  HIPRINT("Batch size:          {}", cfg.batch_size);
  HIPRINT("Total tasks/warp:    {}", cfg.total_tasks);
  if (cfg.test_case == TestCase::kCoroutine) {
    HIPRINT("Subtasks/task:       {}", cfg.subtasks);
  }
  if (cfg.test_case == TestCase::kPutBlob || cfg.test_case == TestCase::kPutBlobGpu) {
    HIPRINT("Total I/O size:      {} bytes ({} MB)", cfg.total_bytes,
            cfg.total_bytes / (1024 * 1024));
    double bw_gbps = (cfg.total_bytes / 1e9) / (elapsed_ms / 1e3);
    printf("Bandwidth:           %.3f GB/s\n", bw_gbps);
  }
  HIPRINT("GPU client warps:    {}", num_warps);
  HIPRINT("Total task ops:      {}", total_ops);
  printf("Elapsed time:        %.3f ms\n", elapsed_ms);
  printf("Throughput:          %.0f tasks/sec\n", throughput);
  printf("Avg latency:         %.3f us/task/warp\n", latency_us);
}

/**
 * Run the GPU runtime latency benchmark end-to-end.
 *
 * Initializes Chimaera, creates a MOD_NAME pool, then calls into
 * the GPU kernel wrapper to time the full GPU client→runtime round-trip.
 *
 * @param cfg  Benchmark configuration
 * @return 0 on success, non-zero on failure
 */
static int RunBenchmark(const BenchmarkConfig &cfg) {
#if !BENCH_GPU_KERNELS_COMPILED
  HLOG(kError, "GPU support not compiled. Rebuild with HSHM_ENABLE_CUDA=1.");
  return 1;
#endif

  // Initialize Chimaera in client mode with runtime
  // (matches CTE benchmark pattern — kClient + env CHI_WITH_RUNTIME=1)
  HIPRINT("Initializing Chimaera runtime...");
  setenv("CHI_WITH_RUNTIME", "1", 1);
  // Set GPU orchestrator dimensions so heap partitions match warp count.
  setenv("CHI_GPU_BLOCKS", std::to_string(cfg.rt_blocks).c_str(), 1);
  setenv("CHI_GPU_THREADS", std::to_string(cfg.rt_threads).c_str(), 1);
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient)) {
    HLOG(kError, "Failed to initialize Chimaera");
    return 1;
  }
  const chi::PoolId pool_id(9000, 0);
  float elapsed_ms = 0.0f;
  int rc;

  if (cfg.test_case == TestCase::kStringAlloc) {
    // String alloc test doesn't need Chimaera runtime at all
    CHI_IPC->PauseGpuOrchestrator();
    rc = run_gpu_bench_string_alloc(cfg.total_tasks, &elapsed_ms);
  } else if (cfg.test_case == TestCase::kSerde ||
             cfg.test_case == TestCase::kAllocSerde ||
             cfg.test_case == TestCase::kAlloc) {
    // Client-only tests: kill the orchestrator immediately, no pool needed
    CHI_IPC->PauseGpuOrchestrator();
    if (cfg.test_case == TestCase::kSerde) {
      rc = run_gpu_bench_serde(pool_id,
                                cfg.client_blocks, cfg.client_threads,
                                cfg.total_tasks, &elapsed_ms);
    } else if (cfg.test_case == TestCase::kAllocSerde) {
      rc = run_gpu_bench_alloc_serde(pool_id,
                                      cfg.client_blocks, cfg.client_threads,
                                      cfg.total_tasks, &elapsed_ms);
    } else {
      rc = run_gpu_bench_alloc(pool_id,
                                cfg.client_blocks, cfg.client_threads,
                                cfg.total_tasks, &elapsed_ms);
    }
  } else if (cfg.test_case == TestCase::kPutBlob ||
             cfg.test_case == TestCase::kPutBlobGpu) {
    // Use the cte_main pool from compose config (PoolId 512,0)
    std::this_thread::sleep_for(500ms);

    const chi::PoolId cte_pool_id(512, 0);
    wrp_cte::core::Client cte_client(cte_pool_id);

    // Create tag using the compose pool
    auto tag_task = cte_client.AsyncGetOrCreateTag("gpu_bench_tag");
    tag_task.Wait();
    if (tag_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create CTE tag (rc={})", tag_task->GetReturnCode());
      chi::CHIMAERA_FINALIZE();
      return 1;
    }
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    bool to_cpu = (cfg.test_case == TestCase::kPutBlob);
    std::this_thread::sleep_for(200ms);
    rc = run_gpu_bench_putblob(cte_pool_id, tag_id,
                                cfg.rt_blocks, cfg.rt_threads,
                                cfg.client_blocks, cfg.client_threads,
                                cfg.total_bytes, to_cpu, &elapsed_ms);
  } else {
    // Runtime tests need pool + orchestrator stabilization
    std::this_thread::sleep_for(500ms);
    if (!CreateBenchPool(pool_id)) {
      chi::CHIMAERA_FINALIZE();
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    if (cfg.test_case == TestCase::kCoroutine) {
      rc = run_gpu_bench_coroutine(pool_id,
                                    cfg.rt_blocks, cfg.rt_threads,
                                    cfg.client_blocks, cfg.client_threads,
                                    cfg.total_tasks, cfg.subtasks,
                                    &elapsed_ms);
    } else {
      const chi::u32 method_id = chimaera::MOD_NAME::Method::kGpuSubmit;
      rc = run_gpu_bench_latency(pool_id, method_id,
                                  cfg.rt_blocks, cfg.rt_threads,
                                  cfg.client_blocks, cfg.client_threads,
                                  cfg.batch_size, cfg.total_tasks,
                                  &elapsed_ms);
    }
  }
  chi::CHIMAERA_FINALIZE();

  if (rc != 0) {
    HLOG(kError, "GPU benchmark failed with code {}", rc);
    return 1;
  }

  PrintResults(cfg, elapsed_ms);
  return 0;
}

/**
 * Benchmark entry point.
 *
 * Parses arguments and dispatches to RunBenchmark.
 */
int main(int argc, char **argv) {
  BenchmarkConfig cfg;
  if (!ParseArgs(argc, argv, cfg)) {
    return 1;
  }

  HIPRINT("=== Chimaera GPU Runtime Benchmark ===");
  HIPRINT("RT blocks={}, RT threads={}, client blocks={} (1 thread/block)",
          cfg.rt_blocks, cfg.rt_threads, cfg.client_blocks);
  HIPRINT("Total tasks/block={}", cfg.total_tasks);

  return RunBenchmark(cfg);
}
