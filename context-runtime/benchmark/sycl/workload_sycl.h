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
 * workload_sycl.h — Workload entry points for the SYCL benchmark suite.
 *
 * Each workload returns 0 on success and writes a BenchResult per
 * sub-configuration to stdout via print_result. The driver
 * (wrp_sycl_bench.cc) dispatches based on a --test-case flag.
 */
#ifndef WRP_SYCL_BENCH_WORKLOAD_H
#define WRP_SYCL_BENCH_WORKLOAD_H

#include "bench_common_sycl.h"

#include <cstdint>
#include <vector>

namespace wrp_sycl_bench {

/** Per-run config — knobs each workload may consult. */
struct BenchConfig {
  uint32_t iterations = 100;
  uint32_t threads = 1024;       // total work-items for kernel-based benches
  uint64_t io_size_bytes = 0;    // 0 → workload picks a default sweep
  int timeout_sec = 60;
};

/** USM bandwidth: GpuApi::Memcpy throughput (host->dev, dev->host, dev->dev). */
int run_workload_usm_bandwidth(sycl::queue &q, const BenchConfig &cfg);

/** Atomic throughput: HSHM_DEVICE_ATOMIC_ADD ops/sec under contention. */
int run_workload_atomic_throughput(sycl::queue &q, const BenchConfig &cfg);

/** Orchestrator lifecycle: WorkOrchestrator::Launch + Finalize avg latency. */
int run_workload_orchestrator_lifecycle(sycl::queue &q, const BenchConfig &cfg);

/** Container alloc: gpu::AllocGpuContainerHost ops/sec per chimod. */
int run_workload_container_alloc(sycl::queue &q, const BenchConfig &cfg);

/** CTE client overhead (Phase 10): GPU-initiated CHI_IPC dereference cost.
 *  Microbench analogue of the CUDA workload_cte_client_overhead.cc — the
 *  full Send/WaitGpu round trip needs a real CTE pool + tag; this exercises
 *  just the kernel-scope CHI_IPC binding path. */
int run_workload_cte_client_overhead(sycl::queue &q, const BenchConfig &cfg);

/** bdev client (Phase 10): same shape as cte_client_overhead — measures
 *  GPU-initiated CHI_IPC binding and verifies the kernel-local pointer
 *  shadows the global nullptr fallback. */
int run_workload_bdev_client(sycl::queue &q, const BenchConfig &cfg);

}  // namespace wrp_sycl_bench

#endif  // WRP_SYCL_BENCH_WORKLOAD_H
