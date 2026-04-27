/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * wrp_sycl_bench — Driver for the SYCL workload benchmark suite.
 *
 * Mirrors the wrp_cte_gpu_bench --test-case dispatch convention so the
 * two suites can be invoked the same way.
 *
 * Usage:
 *   wrp_sycl_bench --test-case <name> [--iterations N] [--threads N]
 *                  [--io-size SIZE]
 *
 * Where <name> is one of:
 *   usm_bandwidth          GpuApi::Memcpy bandwidth sweep
 *   atomic_throughput      HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE ops/sec
 *   orchestrator_lifecycle WorkOrchestrator::Launch + Finalize avg latency
 *   container_alloc        gpu::AllocGpuContainerHost throughput per chimod
 *   all                    run every workload back-to-back
 */

#include "bench_common_sycl.h"
#include "workload_sycl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace wrp_sycl_bench;

namespace {

void print_usage(const char *progname) {
  std::fprintf(stderr,
      "Usage: %s --test-case <name> [options]\n"
      "  --test-case <name>   one of: usm_bandwidth, atomic_throughput,\n"
      "                       orchestrator_lifecycle, container_alloc,\n"
      "                       cte_client_overhead, bdev_client, all\n"
      "  --iterations N       per-workload iteration count (default 100)\n"
      "  --threads N          kernel work-item count (atomic_throughput, default 1024)\n"
      "  --io-size SIZE       single-size override for usm_bandwidth (e.g. 16M)\n"
      "  --timeout-sec N      soft timeout for long workloads (default 60)\n"
      "  --help               show this message\n",
      progname);
}

}  // namespace

int main(int argc, char **argv) {
  std::string test_case = "all";
  BenchConfig cfg{};

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next_or_die = [&](const char *flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (a == "--test-case")             test_case = next_or_die("--test-case");
    else if (a == "--iterations")        cfg.iterations =
        static_cast<uint32_t>(std::atoi(next_or_die("--iterations").c_str()));
    else if (a == "--threads")           cfg.threads =
        static_cast<uint32_t>(std::atoi(next_or_die("--threads").c_str()));
    else if (a == "--io-size")           cfg.io_size_bytes =
        parse_size(next_or_die("--io-size").c_str());
    else if (a == "--timeout-sec")       cfg.timeout_sec =
        std::atoi(next_or_die("--timeout-sec").c_str());
    else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      print_usage(argv[0]);
      return 2;
    }
  }

  sycl::queue q = make_bench_queue();

  int rc = 0;
  bool ran_any = false;
  auto run_if = [&](const std::string &name, auto fn) {
    if (test_case == "all" || test_case == name) {
      ran_any = true;
      rc |= fn(q, cfg);
    }
  };
  run_if("usm_bandwidth",          run_workload_usm_bandwidth);
  run_if("atomic_throughput",      run_workload_atomic_throughput);
  run_if("orchestrator_lifecycle", run_workload_orchestrator_lifecycle);
  run_if("container_alloc",        run_workload_container_alloc);
  run_if("cte_client_overhead",    run_workload_cte_client_overhead);
  run_if("bdev_client",            run_workload_bdev_client);

  if (!ran_any) {
    std::fprintf(stderr, "unknown --test-case: %s\n", test_case.c_str());
    print_usage(argv[0]);
    return 2;
  }
  return rc;
}
