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
 * bench_common_sycl.h — Shared helpers for SYCL workload benchmarks.
 *
 * Mirrors context-transfer-engine/benchmark/gpu/bench_common.h but uses
 * sycl::queue / sycl::malloc_* directly instead of the CUDA runtime.
 * All workloads in this directory link against the SYCL companion library
 * (chimaera_cxx_gpu) and the shared sycl::queue exposed by the driver.
 */
#ifndef BENCH_SYCL_COMMON_H
#define BENCH_SYCL_COMMON_H

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>

namespace wrp_sycl_bench {

/** One row of benchmark output, formatted to match the CUDA bench style. */
struct BenchResult {
  const char *workload;
  const char *mode;
  double elapsed_ms;
  double primary_metric;          // workload-specific (ops/sec, GB/s, ...)
  const char *metric_name;
  double bandwidth_gbps;          // 0 if not applicable
};

inline void print_result(const BenchResult &r) {
  std::printf("  %-22s %-10s %10.3f ms  %10.3f GB/s  %12.3e %s\n",
              r.workload, r.mode, r.elapsed_ms, r.bandwidth_gbps,
              r.primary_metric, r.metric_name);
}

/** Parse "1k", "16M", "2G" — case-insensitive size suffix. */
inline uint64_t parse_size(const char *s) {
  if (!s || !*s) return 0;
  double val = std::atof(s);
  const char *p = s;
  while (*p && (std::isdigit((unsigned char)*p) || *p == '.')) ++p;
  switch (std::tolower((unsigned char)*p)) {
    case 'k': return static_cast<uint64_t>(val * 1024.0);
    case 'm': return static_cast<uint64_t>(val * 1024.0 * 1024.0);
    case 'g': return static_cast<uint64_t>(val * 1024.0 * 1024.0 * 1024.0);
    default:  return static_cast<uint64_t>(val);
  }
}

/** Stopwatch returning elapsed milliseconds at Stop(). */
class WallTimer {
 public:
  void Start() { t0_ = std::chrono::steady_clock::now(); }
  double StopMs() {
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0_).count();
  }
 private:
  std::chrono::steady_clock::time_point t0_;
};

/**
 * Pick a SYCL device. Order of preference:
 *   1. The first GPU device sycl::gpu_selector_v finds.
 *   2. The first available device (CPU fallback) — useful in CI.
 * The selected device's name is printed once so benchmark output is
 * self-describing.
 */
inline sycl::queue make_bench_queue() {
  sycl::queue q;
  try {
    q = sycl::queue(sycl::gpu_selector_v,
                    sycl::property::queue::in_order{});
  } catch (const sycl::exception &e) {
    std::fprintf(stderr,
                 "[bench] no SYCL GPU available (%s); falling back to default\n",
                 e.what());
    q = sycl::queue(sycl::default_selector_v,
                    sycl::property::queue::in_order{});
  }
  std::printf("[bench] device = %s [%s]\n",
              q.get_device().get_info<sycl::info::device::name>().c_str(),
              q.get_device()
                  .get_platform()
                  .get_info<sycl::info::platform::name>()
                  .c_str());
  return q;
}

}  // namespace wrp_sycl_bench

#endif  // BENCH_SYCL_COMMON_H
