/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * workload_bdev_sycl — SYCL port of
 * context-transfer-engine/benchmark/gpu/workload_bdev.cc.
 *
 * The CUDA original launches kernels that submit AllocateBlocks /
 * FreeBlocks / Write / Read tasks to bdev from device code via
 * CHI_IPC->NewTask + Send + WaitGpu, and times the round trip. Like
 * workload_cte_client_overhead, that requires a registered bdev pool
 * and a running orchestrator.
 *
 * This SYCL port is the same shape: it allocates a host-USM IpcManager
 * and a host-USM IpcManagerGpuInfo, captures both pointers in a kernel,
 * and measures GPU-initiated CHI_IPC dereference cost. The integration
 * path that submits real bdev tasks lives in the integration harness.
 *
 * The point of this microbench in the SYCL suite is to validate that
 * Phase 10's CHI_IPC plumbing reaches a member of the IpcManager from
 * a SYCL kernel — the same plumbing that the full bdev workload
 * depends on.
 */

#include "workload_sycl.h"

#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/util/gpu_intrinsics.h>

#include <chimaera/types.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/gpu/gpu_ipc_manager.h>

#include <cstdio>

namespace wrp_sycl_bench {

namespace {

class chi_sycl_bench_bdev_client_kernel;

}  // namespace

int run_workload_bdev_client(sycl::queue &q, const BenchConfig &cfg) {
  std::printf("\n[workload bdev_client] threads=%u iterations=%u\n",
              cfg.threads, cfg.iterations);
  std::printf("  %-22s %-10s %10s   %10s   %12s %s\n",
              "workload", "mode", "elapsed_ms", "bandwidth", "ops/sec", "");

  auto *ipc_storage = hshm::GpuApi::MallocHost<chi::gpu::IpcManager>(
      sizeof(chi::gpu::IpcManager));
  if (!ipc_storage) {
    std::fprintf(stderr, "[bdev_client] MallocHost for IpcManager failed\n");
    return 1;
  }
  new (ipc_storage) chi::gpu::IpcManager();

  auto *gpu_info_storage = hshm::GpuApi::MallocHost<chi::IpcManagerGpuInfo>(
      sizeof(chi::IpcManagerGpuInfo));
  if (!gpu_info_storage) {
    ipc_storage->~IpcManager();
    hshm::GpuApi::FreeHost(ipc_storage);
    std::fprintf(stderr, "[bdev_client] MallocHost for gpu_info failed\n");
    return 1;
  }
  new (gpu_info_storage) chi::IpcManagerGpuInfo();

  // Track in shared USM so the kernel can publish a "non-null IpcManager
  // observed" flag — sanity check that the binding actually flowed.
  uint32_t *seen = sycl::malloc_shared<uint32_t>(1, q);
  if (!seen) {
    ipc_storage->~IpcManager();
    hshm::GpuApi::FreeHost(ipc_storage);
    gpu_info_storage->~IpcManagerGpuInfo();
    hshm::GpuApi::FreeHost(gpu_info_storage);
    return 1;
  }
  *seen = 0;

  const uint32_t threads = cfg.threads;
  const uint32_t iterations = cfg.iterations;
  auto *ipc_ptr = ipc_storage;
  auto *gpu_info_ptr = gpu_info_storage;

  WallTimer t;
  t.Start();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<chi_sycl_bench_bdev_client_kernel>(
        sycl::range<1>(threads),
        [=](sycl::id<1>) {
#if HSHM_IS_DEVICE_PASS
          CHIMAERA_GPU_CLIENT_INIT(*gpu_info_ptr, /*num_blocks=*/1, ipc_ptr);

          uint32_t local_seen = 0;
          for (uint32_t i = 0; i < iterations; ++i) {
            auto *ipc = CHI_IPC;
            // ipc must be the kernel-captured pointer, NOT the global
            // namespace fallback nullptr.
            if (ipc != nullptr) ++local_seen;
          }
          if (local_seen == iterations) {
            HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE(seen, 1u);
          }
          (void)g_ipc_manager;
#else
          (void)gpu_info_ptr; (void)ipc_ptr; (void)iterations; (void)seen;
#endif
        });
  }).wait_and_throw();
  double ms = t.StopMs();

  uint64_t total_ops =
      static_cast<uint64_t>(threads) * static_cast<uint64_t>(iterations);
  double ops_sec = static_cast<double>(total_ops) / (ms / 1000.0);

  // Verify every work-item saw a non-null CHI_IPC — i.e. the kernel-
  // local binding shadowed the global-namespace nullptr fallback.
  if (*seen != threads) {
    std::fprintf(stderr,
                 "[bdev_client] CHI_IPC binding broken: seen=%u threads=%u\n",
                 *seen, threads);
    sycl::free(seen, q);
    ipc_storage->~IpcManager();
    hshm::GpuApi::FreeHost(ipc_storage);
    gpu_info_storage->~IpcManagerGpuInfo();
    hshm::GpuApi::FreeHost(gpu_info_storage);
    return 2;
  }

  BenchResult r{"bdev_client", "ipc_resolve",
                ms, ops_sec, "CHI_IPC/s", 0.0};
  print_result(r);

  sycl::free(seen, q);
  ipc_storage->~IpcManager();
  hshm::GpuApi::FreeHost(ipc_storage);
  gpu_info_storage->~IpcManagerGpuInfo();
  hshm::GpuApi::FreeHost(gpu_info_storage);
  return 0;
}

}  // namespace wrp_sycl_bench
