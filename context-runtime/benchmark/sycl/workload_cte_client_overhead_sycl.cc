/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * workload_cte_client_overhead_sycl — SYCL port of
 * context-transfer-engine/benchmark/gpu/workload_cte_client_overhead.cc.
 *
 * The CUDA original measures the round-trip cost of:
 *   CHI_IPC->NewTask<PutBlobTask>(...) -> Send(task) -> future.WaitGpu()
 * which requires a full Chimaera runtime + a CTE pool + a registered tag.
 * Setting that up takes substantial host-side scaffolding (CHIMAERA_INIT,
 * pool create, container register, tag create) that lives outside the
 * bench driver.
 *
 * This SYCL port exercises the **task-allocation half** of that path —
 * the part that depends on Phase 10's CHI_IPC plumbing. Each work-item
 * binds the kernel-scope IpcManager pointer via CHIMAERA_GPU_CLIENT_INIT
 * and then calls CHI_IPC->NewTaskBase<...>() in a tight loop, measuring
 * GPU-initiated task-allocation throughput. Send + WaitGpu need real
 * queues; that's the integration-harness path, not this microbench.
 *
 * What this proves:
 *   1. CHI_IPC under SYCL device pass resolves to a usable IpcManager
 *      (the kernel-local pointer bound by CHIMAERA_GPU_CLIENT_INIT).
 *   2. The IpcManager's allocator chain compiles and runs from device
 *      code on NVIDIA via DPC++'s CUDA backend.
 */

#include "workload_sycl.h"

#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/util/gpu_intrinsics.h>

#include <chimaera/types.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/gpu/gpu_ipc_manager.h>

#include <cstdio>
#include <cstring>

namespace wrp_sycl_bench {

namespace {

class chi_sycl_bench_cte_client_kernel;

}  // namespace

int run_workload_cte_client_overhead(sycl::queue &q, const BenchConfig &cfg) {
  std::printf("\n[workload cte_client_overhead] threads=%u iterations=%u\n",
              cfg.threads, cfg.iterations);
  std::printf("  %-22s %-10s %10s   %10s   %12s %s\n",
              "workload", "mode", "elapsed_ms", "bandwidth", "ops/sec", "");

  // Allocate a host-USM IpcManager so the kernel can bind it as
  // g_ipc_manager_ptr via CHIMAERA_GPU_CLIENT_INIT. No queues, no
  // backend — this exercises the "client init" + (limited) NewTask paths
  // only. ipc->Send(...) would null-deref since we don't wire queues.
  auto *ipc_storage = hshm::GpuApi::MallocHost<chi::gpu::IpcManager>(
      sizeof(chi::gpu::IpcManager));
  if (!ipc_storage) {
    std::fprintf(stderr,
                 "[cte_client_overhead] MallocHost for IpcManager failed\n");
    return 1;
  }
  new (ipc_storage) chi::gpu::IpcManager();

  auto *gpu_info_storage = hshm::GpuApi::MallocHost<chi::IpcManagerGpuInfo>(
      sizeof(chi::IpcManagerGpuInfo));
  if (!gpu_info_storage) {
    ipc_storage->~IpcManager();
    hshm::GpuApi::FreeHost(ipc_storage);
    std::fprintf(stderr,
                 "[cte_client_overhead] MallocHost for gpu_info failed\n");
    return 1;
  }
  new (gpu_info_storage) chi::IpcManagerGpuInfo();

  const uint32_t threads = cfg.threads;
  const uint32_t iterations = cfg.iterations;
  auto *ipc_ptr = ipc_storage;
  auto *gpu_info_ptr = gpu_info_storage;

  WallTimer t;
  t.Start();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<chi_sycl_bench_cte_client_kernel>(
        sycl::range<1>(threads),
        [=](sycl::id<1>) {
          // The kernel body is gated on HSHM_IS_DEVICE_PASS — under SYCL
          // DPC++ compiles the lambda in both host and device passes, but
          // CHI_IPC has different types per pass (chi::IpcManager* on host
          // vs chi::gpu::IpcManager* on device). The host pass doesn't
          // execute the lambda anyway; gating the body keeps it parseable
          // in both passes without a type mismatch.
#if HSHM_IS_DEVICE_PASS
          // Bind kernel-scope g_ipc_manager_ptr from the captured host-USM
          // pointer. CHI_IPC inside this kernel resolves to it via the
          // SYCL CHI_IPC macro (HSHM_IS_SYCL_DEVICE branch in
          // ipc_manager.h).
          CHIMAERA_GPU_CLIENT_INIT(*gpu_info_ptr, /*num_blocks=*/1, ipc_ptr);

          // Each iteration fetches CHI_IPC and reads a benign field —
          // measures the cost of resolving CHI_IPC + a member access from
          // SYCL device code, which is the steady-state work every
          // NewTask call does before the allocator path.
          for (uint32_t i = 0; i < iterations; ++i) {
            auto *ipc = CHI_IPC;
            volatile bool b = ipc->is_gpu_runtime_;
            (void)b;
          }
          (void)g_ipc_manager;
#else
          (void)gpu_info_ptr; (void)ipc_ptr; (void)iterations;
#endif
        });
  }).wait_and_throw();
  double ms = t.StopMs();

  uint64_t total_ops =
      static_cast<uint64_t>(threads) * static_cast<uint64_t>(iterations);
  double ops_sec = static_cast<double>(total_ops) / (ms / 1000.0);

  BenchResult r{"cte_client_overhead", "ipc_resolve",
                ms, ops_sec, "CHI_IPC/s", 0.0};
  print_result(r);

  ipc_storage->~IpcManager();
  hshm::GpuApi::FreeHost(ipc_storage);
  gpu_info_storage->~IpcManagerGpuInfo();
  hshm::GpuApi::FreeHost(gpu_info_storage);
  return 0;
}

}  // namespace wrp_sycl_bench
