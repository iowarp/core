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
 * CTE GPU Benchmark — CPU driver
 *
 * Measures the performance of GPU-initiated PutBlob operations through CTE.
 * Supports three modes:
 *   putblob     — GPU client → CTE via GPU→CPU path (ToLocalCpu)
 *   putblob_gpu — GPU client → CTE via GPU-local path (Local)
 *   direct      — GPU kernel writes directly to pinned host memory (baseline)
 *
 * Usage:
 *   wrp_cte_gpu_bench [options]
 *
 * Options:
 *   --test-case <case>       putblob, putblob_gpu, or direct (default: putblob)
 *   --rt-blocks <N>          GPU runtime orchestrator blocks (default: 1)
 *   --rt-threads <N>         GPU runtime threads per block (default: 32)
 *   --client-blocks <N>      GPU client kernel blocks (default: 1)
 *   --client-threads <N>     GPU client kernel threads per block (default: 32)
 *   --io-size <bytes>        Total I/O size (default: 64M, supports k/m/g)
 */

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <hermes_shm/util/logging.h>
#include <hermes_shm/util/gpu_api.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

using namespace std::chrono_literals;

extern "C" int run_cte_gpu_bench_putblob(
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    bool to_cpu,
    float *out_elapsed_ms);

extern "C" int run_cte_gpu_bench_direct(
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    float *out_elapsed_ms);

extern "C" int run_cte_gpu_bench_cudamemcpy(
    chi::u64 total_bytes,
    float *out_elapsed_ms);

extern "C" int run_cte_gpu_bench_managed(
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    float *out_write_ms,
    float *out_prefetch_ms,
    float *out_total_ms);

enum class TestCase { kPutBlob, kPutBlobGpu, kDirect, kCudaMemcpy, kManaged };

struct BenchConfig {
  TestCase test_case = TestCase::kPutBlob;
  chi::u32 rt_blocks = 1;
  chi::u32 rt_threads = 32;
  chi::u32 client_blocks = 1;
  chi::u32 client_threads = 32;
  chi::u64 total_bytes = 64 * 1024 * 1024;
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
  HIPRINT("  --test-case <case>     putblob, putblob_gpu, direct, cudamemcpy, or managed (default: putblob)");
  HIPRINT("  --rt-blocks <N>        GPU runtime orchestrator blocks (default: 1)");
  HIPRINT("  --rt-threads <N>       GPU runtime threads/block (default: 32)");
  HIPRINT("  --client-blocks <N>    GPU client kernel blocks (default: 1)");
  HIPRINT("  --client-threads <N>   GPU client kernel threads/block (default: 32)");
  HIPRINT("  --io-size <bytes>      Total I/O size (default: 64M, supports k/m/g suffixes)");
  HIPRINT("  --help, -h             Show this help");
}

bool ParseArgs(int argc, char **argv, BenchConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--test-case" && i + 1 < argc) {
      std::string tc = argv[++i];
      if (tc == "putblob") cfg.test_case = TestCase::kPutBlob;
      else if (tc == "putblob_gpu") cfg.test_case = TestCase::kPutBlobGpu;
      else if (tc == "direct") cfg.test_case = TestCase::kDirect;
      else if (tc == "cudamemcpy") cfg.test_case = TestCase::kCudaMemcpy;
      else if (tc == "managed") cfg.test_case = TestCase::kManaged;
      else {
        HLOG(kError, "Unknown test case '{}'; use putblob, putblob_gpu, direct, cudamemcpy, or managed", tc);
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
      cfg.total_bytes = ParseSize(argv[++i]);
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

  // Load GPU config if CHI_SERVER_CONF is not already set
  if (!std::getenv("CHI_SERVER_CONF")) {
    std::string config_dir = std::string(__FILE__);
    config_dir = config_dir.substr(0, config_dir.rfind('/'));
    std::string gpu_config = config_dir + "/cte_config_gpu.yaml";
    setenv("CHI_SERVER_CONF", gpu_config.c_str(), 1);
    HLOG(kInfo, "Using GPU benchmark config: {}", gpu_config);
  }

  // Initialize Chimaera runtime
  HLOG(kInfo, "Initializing Chimaera runtime...");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }
  std::this_thread::sleep_for(500ms);

  const char *tc_name = (cfg.test_case == TestCase::kPutBlob) ? "putblob" :
                         (cfg.test_case == TestCase::kPutBlobGpu) ? "putblob_gpu" :
                         (cfg.test_case == TestCase::kCudaMemcpy) ? "cudamemcpy" :
                         (cfg.test_case == TestCase::kManaged) ? "managed" :
                         "direct";

  HIPRINT("\n=== CTE GPU Benchmark ===");
  HIPRINT("Test case:           {}", tc_name);
  HIPRINT("RT blocks:           {}", cfg.rt_blocks);
  HIPRINT("RT threads/block:    {}", cfg.rt_threads);
  HIPRINT("Client blocks:       {}", cfg.client_blocks);
  HIPRINT("Client threads/block:{}", cfg.client_threads);
  HIPRINT("Total I/O size:      {} bytes ({} MB)", cfg.total_bytes,
          cfg.total_bytes / (1024 * 1024));

  float elapsed_ms = 0;
  int rc = 0;

  if (cfg.test_case == TestCase::kManaged) {
    // Managed memory: GPU write + prefetch to host
    float write_ms = 0, prefetch_ms = 0, total_ms = 0;
    rc = run_cte_gpu_bench_managed(cfg.client_blocks, cfg.client_threads,
                                    cfg.total_bytes,
                                    &write_ms, &prefetch_ms, &total_ms);
    if (rc != 0) {
      HLOG(kError, "Benchmark failed with error: {}", rc);
      return 1;
    }
    double bytes = cfg.total_bytes;
    printf("\n=== %s Results ===\n", tc_name);
    printf("GPU write:           %.3f ms  (%.3f GB/s)\n",
           static_cast<double>(write_ms), (bytes / 1e9) / (write_ms / 1e3));
    printf("Prefetch to host:    %.3f ms  (%.3f GB/s)\n",
           static_cast<double>(prefetch_ms), (bytes / 1e9) / (prefetch_ms / 1e3));
    printf("Total:               %.3f ms  (%.3f GB/s)\n",
           static_cast<double>(total_ms), (bytes / 1e9) / (total_ms / 1e3));
    printf("=========================\n");
    return 0;
  } else if (cfg.test_case == TestCase::kCudaMemcpy) {
    // cudaMemcpy baseline — theoretical PCIe maximum
    rc = run_cte_gpu_bench_cudamemcpy(cfg.total_bytes, &elapsed_ms);
  } else if (cfg.test_case == TestCase::kDirect) {
    // Direct kernel memcpy baseline — no CTE, no serialization
    rc = run_cte_gpu_bench_direct(cfg.client_blocks, cfg.client_threads,
                                   cfg.total_bytes, &elapsed_ms);
  } else {
    // CTE putblob path — need pool, target, and tag setup
    // Create a separate CTE pool
    chi::PoolId gpu_pool_id(wrp_cte::core::kCtePoolId.major_ + 1,
                             wrp_cte::core::kCtePoolId.minor_);
    wrp_cte::core::Client cte_client(gpu_pool_id);
    wrp_cte::core::CreateParams params;
    auto create_task = cte_client.AsyncCreate(
        chi::PoolQuery::Dynamic(),
        "cte_gpu_bench_pool", gpu_pool_id, params);
    create_task.Wait();
    if (create_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create CTE GPU pool: {}", create_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    // Register pinned-memory bdev target (CPU side)
    chi::PoolId bdev_pool_id(800, 0);
    auto reg_task = cte_client.AsyncRegisterTarget(
        "pinned::cte_gpu_bench_target",
        chimaera::bdev::BdevType::kPinned,
        256ULL * 1024 * 1024,
        chi::PoolQuery::Local(),
        bdev_pool_id);
    reg_task.Wait();
    if (reg_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to register target (CPU): {}", reg_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    // Register target on GPU side
    auto gpu_reg_task = cte_client.AsyncRegisterTarget(
        "pinned::cte_gpu_bench_target",
        chimaera::bdev::BdevType::kPinned,
        256ULL * 1024 * 1024,
        chi::PoolQuery::Local(),
        bdev_pool_id,
        chi::PoolQuery::LocalGpuBcast());
    gpu_reg_task.Wait();
    if (gpu_reg_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to register target (GPU): {}", gpu_reg_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    // Create tag (CPU side)
    auto tag_task = cte_client.AsyncGetOrCreateTag(
        "gpu_bench_tag", wrp_cte::core::TagId::GetNull(),
        chi::PoolQuery::Local());
    tag_task.Wait();
    if (tag_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create tag");
      return 1;
    }
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    // Create tag on GPU side
    auto gpu_tag_task = cte_client.AsyncGetOrCreateTag(
        "gpu_bench_tag", tag_id, chi::PoolQuery::LocalGpuBcast());
    gpu_tag_task.Wait();
    if (gpu_tag_task->GetReturnCode() != 0) {
      HLOG(kError, "Failed to create tag on GPU: {}", gpu_tag_task->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    HIPRINT("Pool ID: {}.{}", gpu_pool_id.major_, gpu_pool_id.minor_);
    HIPRINT("Tag ID:  {}.{}", tag_id.major_, tag_id.minor_);

    bool to_cpu = (cfg.test_case == TestCase::kPutBlob);
    fprintf(stderr, "DBG: entering run_cte_gpu_bench_putblob\n"); fflush(stderr);
    rc = run_cte_gpu_bench_putblob(
        cte_client.pool_id_, tag_id,
        cfg.rt_blocks, cfg.rt_threads,
        cfg.client_blocks, cfg.client_threads,
        cfg.total_bytes, to_cpu, &elapsed_ms);
  }

  if (rc != 0) {
    HLOG(kError, "Benchmark failed with error: {}", rc);
    return 1;
  }

  double bw_gbps = (cfg.total_bytes / 1e9) / (elapsed_ms / 1e3);

  printf("\n=== %s Results ===\n", tc_name);
  printf("Elapsed:             %.3f ms\n", static_cast<double>(elapsed_ms));
  printf("Bandwidth:           %.3f GB/s\n", bw_gbps);
  printf("=========================\n");

  return 0;
}
