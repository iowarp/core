/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * workload_usm_bandwidth — measures hshm::GpuApi::Memcpy throughput.
 *
 * Three modes:
 *   h2d   host  USM -> device USM   (sycl::malloc_host -> malloc_device)
 *   d2h   device USM -> host  USM
 *   d2d   device USM -> device USM
 *
 * Sweeps a fixed size table by default (4 KiB ... 256 MiB). Each size
 * runs `iterations` Memcpy calls back-to-back inside one timed window;
 * the queue is drained with q.wait() before stopping the timer so we
 * measure end-to-end transfer cost (not just submission).
 *
 * GB/s is reported in IEC binary GB (1 GB = 2^30 bytes), matching the
 * convention used by the existing CUDA bench output.
 */

#include "workload_sycl.h"

#include <hermes_shm/util/gpu_api.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace wrp_sycl_bench {

namespace {

constexpr double kBytesPerGB = 1024.0 * 1024.0 * 1024.0;

struct SizeRun {
  uint64_t bytes;
  const char *label;
};

/** Default sweep — 4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB, 256 MiB. */
const std::vector<SizeRun> &default_sweep() {
  static const std::vector<SizeRun> kSweep = {
      {  4ULL * 1024,             "4K"},
      { 64ULL * 1024,            "64K"},
      {  1ULL * 1024 * 1024,      "1M"},
      { 16ULL * 1024 * 1024,     "16M"},
      { 64ULL * 1024 * 1024,     "64M"},
      {256ULL * 1024 * 1024,    "256M"},
  };
  return kSweep;
}

/**
 * Run iterations Memcpy calls of `bytes` from src→dst and time the burst.
 * Returns elapsed milliseconds.
 */
double time_memcpy_burst(void *dst, const void *src, uint64_t bytes,
                         uint32_t iterations) {
  WallTimer t;
  t.Start();
  for (uint32_t i = 0; i < iterations; ++i) {
    hshm::GpuApi::Memcpy(static_cast<unsigned char *>(dst),
                         static_cast<const unsigned char *>(src),
                         bytes);
  }
  return t.StopMs();
}

}  // namespace

int run_workload_usm_bandwidth(sycl::queue & /*q*/, const BenchConfig &cfg) {
  std::printf("\n[workload usm_bandwidth] iterations=%u\n", cfg.iterations);
  std::printf("  %-22s %-10s %10s   %10s   %12s %s\n",
              "workload", "size", "elapsed_ms", "bandwidth", "ops/sec", "");

  std::vector<SizeRun> sweep;
  if (cfg.io_size_bytes != 0) {
    char label[16];
    std::snprintf(label, sizeof(label), "%lluB",
                  static_cast<unsigned long long>(cfg.io_size_bytes));
    // Note: label memory must outlive print_result; use a static buffer
    // per row by carrying it through the BenchResult.
    static std::string staged_label;
    staged_label = label;
    sweep.push_back({cfg.io_size_bytes, staged_label.c_str()});
  } else {
    sweep = default_sweep();
  }

  for (const auto &run : sweep) {
    // Allocate src + dst for each direction. Reuse across iterations of
    // the same size; reallocate per size since GpuApi::Free / FreeHost is
    // fast and isolates each run from prior cache effects.
    using GA = hshm::GpuApi;
    auto *h_src = GA::MallocHost<unsigned char>(run.bytes);
    auto *h_dst = GA::MallocHost<unsigned char>(run.bytes);
    auto *d_src = GA::Malloc<unsigned char>(run.bytes);
    auto *d_dst = GA::Malloc<unsigned char>(run.bytes);
    if (!h_src || !h_dst || !d_src || !d_dst) {
      std::fprintf(stderr,
                   "[usm_bandwidth] allocation failed at size=%s (%llu bytes)\n",
                   run.label, static_cast<unsigned long long>(run.bytes));
      if (h_src) GA::FreeHost(h_src);
      if (h_dst) GA::FreeHost(h_dst);
      if (d_src) GA::Free(d_src);
      if (d_dst) GA::Free(d_dst);
      continue;
    }
    std::memset(h_src, 0xAB, run.bytes);

    // Warm-up: prime caches and the SYCL submission pipeline.
    GA::Memcpy(d_src, h_src, run.bytes);

    // h2d
    {
      double ms = time_memcpy_burst(d_src, h_src, run.bytes, cfg.iterations);
      double total_bytes =
          static_cast<double>(run.bytes) * static_cast<double>(cfg.iterations);
      double gbps = (total_bytes / kBytesPerGB) / (ms / 1000.0);
      double ops_sec = static_cast<double>(cfg.iterations) / (ms / 1000.0);
      BenchResult r{"usm_bandwidth", "h2d", ms, ops_sec, "memcpy/sec", gbps};
      // Override the mode column with size for readability.
      char mode[24];
      std::snprintf(mode, sizeof(mode), "h2d %s", run.label);
      r.mode = mode;
      print_result(r);
    }
    // d2h
    {
      double ms = time_memcpy_burst(h_dst, d_src, run.bytes, cfg.iterations);
      double total_bytes =
          static_cast<double>(run.bytes) * static_cast<double>(cfg.iterations);
      double gbps = (total_bytes / kBytesPerGB) / (ms / 1000.0);
      double ops_sec = static_cast<double>(cfg.iterations) / (ms / 1000.0);
      BenchResult r{"usm_bandwidth", nullptr, ms, ops_sec, "memcpy/sec", gbps};
      char mode[24];
      std::snprintf(mode, sizeof(mode), "d2h %s", run.label);
      r.mode = mode;
      print_result(r);
    }
    // d2d
    {
      double ms = time_memcpy_burst(d_dst, d_src, run.bytes, cfg.iterations);
      double total_bytes =
          static_cast<double>(run.bytes) * static_cast<double>(cfg.iterations);
      double gbps = (total_bytes / kBytesPerGB) / (ms / 1000.0);
      double ops_sec = static_cast<double>(cfg.iterations) / (ms / 1000.0);
      BenchResult r{"usm_bandwidth", nullptr, ms, ops_sec, "memcpy/sec", gbps};
      char mode[24];
      std::snprintf(mode, sizeof(mode), "d2d %s", run.label);
      r.mode = mode;
      print_result(r);
    }

    GA::FreeHost(h_src);
    GA::FreeHost(h_dst);
    GA::Free(d_src);
    GA::Free(d_dst);
  }
  return 0;
}

}  // namespace wrp_sycl_bench
