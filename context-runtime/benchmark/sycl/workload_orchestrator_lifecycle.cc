/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * workload_orchestrator_lifecycle — measures end-to-end cost of a
 * WorkOrchestrator::Launch + Finalize cycle on the SYCL backend.
 *
 * Each iteration:
 *   - Launch:    allocates pinned host control + device PoolManager +
 *                host-USM IpcManagerGpuInfo + sycl::queue, submits the
 *                persistent single_task kernel.
 *   - Wait briefly for running_flag = 1 (so the kernel actually came up).
 *   - Finalize:  signals exit_flag, waits the queue, frees all resources.
 *
 * Reports avg ms per cycle. This is the dominant per-pool startup cost
 * for any chimod that brings up its own GPU runtime.
 */

#include "workload_sycl.h"

#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/gpu/gpu_info.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace wrp_sycl_bench {

int run_workload_orchestrator_lifecycle(sycl::queue & /*q*/,
                                         const BenchConfig &cfg) {
  std::printf("\n[workload orchestrator_lifecycle] iterations=%u\n",
              cfg.iterations);
  std::printf("  %-22s %-10s %10s   %10s   %12s %s\n",
              "workload", "mode", "elapsed_ms", "bandwidth", "ops/sec", "");

  // Warm-up cycle — first launch pays JIT + device init costs that we
  // don't want to amortize into the steady-state number.
  {
    chi::gpu::WorkOrchestrator orch;
    chi::IpcManagerGpuInfo gi{};
    orch.Launch(gi, 1, 1, nullptr);
    using namespace std::chrono_literals;
    for (int i = 0; i < 200 && orch.control_->running_flag == 0; ++i) {
      std::this_thread::sleep_for(2ms);
    }
    orch.Finalize();
  }

  WallTimer total;
  total.Start();
  for (uint32_t i = 0; i < cfg.iterations; ++i) {
    chi::gpu::WorkOrchestrator orch;
    chi::IpcManagerGpuInfo gi{};
    if (!orch.Launch(gi, 1, 1, nullptr)) {
      std::fprintf(stderr,
                   "[orchestrator_lifecycle] Launch failed on iter %u\n", i);
      return 1;
    }
    using namespace std::chrono_literals;
    for (int spin = 0; spin < 500 && orch.control_->running_flag == 0;
         ++spin) {
      std::this_thread::sleep_for(1ms);
    }
    orch.Finalize();
  }
  double ms = total.StopMs();

  double per_cycle_ms = ms / cfg.iterations;
  double cycles_sec = 1000.0 / per_cycle_ms;
  BenchResult r{"orch_lifecycle", "launch+fin",
                ms, cycles_sec, "cycles/s", 0.0};
  print_result(r);
  std::printf("  per-cycle latency: %.3f ms\n", per_cycle_ms);

  return 0;
}

}  // namespace wrp_sycl_bench
