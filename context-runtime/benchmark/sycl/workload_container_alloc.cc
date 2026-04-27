/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * workload_container_alloc — measures gpu::AllocGpuContainerHost throughput
 * per chimod.
 *
 * Each chimod has its own GpuRuntime subclass. AllocGpuContainerHost:
 *   - Looks up the module ID by name.
 *   - sycl::malloc_device(GetGpuModuleSize(module_id)) for the container.
 *   - Submits a single_task that placement-news the right GpuRuntime
 *     (autogen-emitted constructor populates the function-pointer table).
 *   - Returns the device pointer.
 *
 * After the first call, the SYCL runtime caches the JIT'd kernel — so the
 * second-and-later calls see only the cost of submit + USM allocation +
 * the placement-new kernel itself. We report:
 *   - cold first-call latency (JIT cost included).
 *   - warm steady-state latency (avg over `iterations - 1` calls).
 */

#include "workload_sycl.h"

#include <chimaera/types.h>
#include <hermes_shm/util/gpu_api.h>

// Pull in the autogen header that provides AllocGpuContainerHost.
#include "autogen/gpu_work_orchestrator_modules.h"

#include <cstdio>

namespace wrp_sycl_bench {

namespace {

void run_one_chimod(const char *chimod_name, const BenchConfig &cfg) {
  // Cold call (first allocation pays JIT + module-load cost).
  WallTimer cold;
  cold.Start();
  chi::PoolId pid;
  pid.major_ = 0xC0DE0000u;
  pid.minor_ = 1;
  void *first =
      chi::gpu::AllocGpuContainerHost(pid, /*container_id=*/1u, chimod_name);
  double cold_ms = cold.StopMs();
  if (!first) {
    std::fprintf(stderr,
                 "[container_alloc] AllocGpuContainerHost(%s) returned nullptr\n",
                 chimod_name);
    return;
  }
  hshm::GpuApi::Free(first);

  // Warm steady-state.
  uint32_t warm_iters = cfg.iterations > 1 ? cfg.iterations - 1 : 1;
  WallTimer warm;
  warm.Start();
  for (uint32_t i = 0; i < warm_iters; ++i) {
    pid.minor_ = i + 2;
    void *p =
        chi::gpu::AllocGpuContainerHost(pid, /*container_id=*/i + 2,
                                        chimod_name);
    if (!p) {
      std::fprintf(stderr,
                   "[container_alloc] iter %u failed for %s\n", i, chimod_name);
      return;
    }
    hshm::GpuApi::Free(p);
  }
  double warm_ms = warm.StopMs();
  double per_call_ms = warm_ms / warm_iters;
  double calls_sec = 1000.0 / per_call_ms;

  // Two rows: cold + warm. mode column carries the chimod name.
  char mode_cold[40];
  char mode_warm[40];
  std::snprintf(mode_cold, sizeof(mode_cold), "%s/cold", chimod_name);
  std::snprintf(mode_warm, sizeof(mode_warm), "%s/warm", chimod_name);

  BenchResult cold_r{"container_alloc", mode_cold,
                     cold_ms, 1000.0 / cold_ms, "alloc/s", 0.0};
  BenchResult warm_r{"container_alloc", mode_warm,
                     warm_ms, calls_sec, "alloc/s", 0.0};
  print_result(cold_r);
  print_result(warm_r);
}

}  // namespace

int run_workload_container_alloc(sycl::queue & /*q*/, const BenchConfig &cfg) {
  std::printf("\n[workload container_alloc] iterations=%u (1 cold + %u warm)\n",
              cfg.iterations,
              cfg.iterations > 1 ? cfg.iterations - 1 : 1);
  std::printf("  %-22s %-10s %10s   %10s   %12s %s\n",
              "workload", "mode", "elapsed_ms", "bandwidth", "ops/sec", "");

  // Run for each registered chimod that has a GpuRuntime.
  run_one_chimod("chimaera_admin", cfg);
  run_one_chimod("chimaera_bdev", cfg);
  run_one_chimod("wrp_cte_core", cfg);
  return 0;
}

}  // namespace wrp_sycl_bench
