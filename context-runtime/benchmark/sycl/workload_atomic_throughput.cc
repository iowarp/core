/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * workload_atomic_throughput — measures HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE
 * ops/sec under maximum contention.
 *
 * Submits an nd_range parallel_for of `threads` work-items; every work-item
 * does `iterations` fetch_add operations on a single shared u32 counter.
 * Since every work-item targets the same address, this measures the
 * device-scope atomic primitive's contended throughput — which is what
 * matters for chimod paths like CTE core's tag-id allocator and bdev's
 * heap bump pointer.
 *
 * Reports total atomic ops, elapsed time, and ops/sec.
 */

#include "workload_sycl.h"

#include <hermes_shm/util/gpu_intrinsics.h>

#include <cstdio>

namespace wrp_sycl_bench {

namespace {

// SYCL requires unique kernel-name types per submission.
class chi_sycl_bench_atomic_warmup_kernel;
class chi_sycl_bench_atomic_timed_kernel;

}  // namespace

int run_workload_atomic_throughput(sycl::queue &q, const BenchConfig &cfg) {
  std::printf("\n[workload atomic_throughput] threads=%u iterations=%u\n",
              cfg.threads, cfg.iterations);
  std::printf("  %-22s %-10s %10s   %10s   %12s %s\n",
              "workload", "mode", "elapsed_ms", "bandwidth", "ops/sec", "");

  // Shared USM is convenient: host seeds the counter, kernel atomically
  // updates it, host reads back without a separate memcpy.
  uint32_t *counter = sycl::malloc_shared<uint32_t>(1, q);
  if (!counter) {
    std::fprintf(stderr, "[atomic_throughput] malloc_shared failed\n");
    return 1;
  }
  *counter = 0;
  q.wait();

  const uint32_t iters = cfg.iterations;
  const size_t threads = cfg.threads;

  // Warm-up — primes JIT cache and any first-launch overhead.
  q.parallel_for<chi_sycl_bench_atomic_warmup_kernel>(
       sycl::range<1>(threads),
       [=](sycl::id<1>) {
         HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE(counter, 1u);
       })
      .wait_and_throw();
  *counter = 0;
  q.wait();

  WallTimer t;
  t.Start();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<chi_sycl_bench_atomic_timed_kernel>(
        sycl::range<1>(threads),
        [=](sycl::id<1>) {
          for (uint32_t i = 0; i < iters; ++i) {
            HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE(counter, 1u);
          }
        });
  }).wait_and_throw();
  double ms = t.StopMs();

  uint64_t total_ops =
      static_cast<uint64_t>(threads) * static_cast<uint64_t>(iters);
  double ops_sec = static_cast<double>(total_ops) / (ms / 1000.0);
  uint32_t observed = *counter;

  // Sanity check: the counter must equal total_ops modulo wrap.
  uint32_t expected = static_cast<uint32_t>(total_ops);
  if (observed != expected) {
    std::fprintf(stderr,
                 "[atomic_throughput] counter mismatch: got %u, expected %u "
                 "(possible silent atomic failure)\n",
                 observed, expected);
  }

  BenchResult r{"atomic_throughput", "atomic_add",
                ms, ops_sec, "atomic_add/s", 0.0};
  print_result(r);

  sycl::free(counter, q);
  return (observed == expected) ? 0 : 1;
}

}  // namespace wrp_sycl_bench
