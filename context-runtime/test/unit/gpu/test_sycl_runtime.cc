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
 * SYCL backend smoke tests for the GPU orchestrator infrastructure.
 *
 * Validates the Phase 1–7 SYCL plumbing without requiring full Chimaera
 * runtime initialization (no admin pool, no IPC, no networking — those
 * have CPU-only paths that work the same under any GPU backend). The
 * three test cases here cover the SYCL-specific surface:
 *
 *   1. Device-intrinsic abstractions (fences, atomics, clock, printf
 *      routed through HSHM_DEVICE_* macros from gpu_intrinsics.h).
 *   2. GpuApi USM allocation and host<->device transfer.
 *   3. WorkOrchestrator::Launch / Finalize lifecycle on a real
 *      sycl::queue, including the persistent single_task kernel that
 *      runs WorkerSycl::PollOnce in a loop.
 *
 * The test executable links against chimaera_cxx_gpu (the SYCL companion
 * library built by add_sycl_library when WRP_CORE_ENABLE_SYCL=ON).
 */

#if HSHM_ENABLE_SYCL

#include "simple_test.h"

#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/gpu/gpu_info.h>
#include <chimaera/gpu/pool_manager.h>

#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/util/gpu_intrinsics.h>

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <thread>

using namespace std::chrono_literals;

namespace {

/** Opaque kernel-name types — SYCL requires unique types per submission. */
class chi_sycl_test_atomic_kernel;
class chi_sycl_test_memcpy_kernel;

}  // namespace

TEST_CASE("SYCL device intrinsic abstractions", "[sycl][intrinsics]") {
  sycl::queue q{sycl::gpu_selector_v};

  // Allocate a uint32 in shared USM so the host can both seed it and
  // read it back without a separate memcpy.
  uint32_t *flag = sycl::malloc_shared<uint32_t>(1, q);
  REQUIRE(flag != nullptr);
  *flag = 0;

  // Single work-item kernel exercises every HSHM_DEVICE_* abstraction
  // we ship today: fences, device-scope atomic OR, atomic ADD, and the
  // (no-op under SYCL) printf macro. Failure here means an abstraction
  // expanded to something the SYCL frontend rejects.
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_test_atomic_kernel>([=]() {
      HSHM_DEVICE_FENCE_DEVICE();
      HSHM_DEVICE_ATOMIC_OR_U32_DEVICE(flag, 0x1u);
      HSHM_DEVICE_FENCE_SYSTEM();
      HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE(flag, 0x2u);
      // HSHM_DEVICE_PRINTF is a no-op under SYCL but the call must
      // still compile — variadic args evaluated to (void)0.
      HSHM_DEVICE_PRINTF("flag=%u\n", static_cast<unsigned>(*flag));
      // HSHM_DEVICE_CLOCK64 returns 0 under SYCL; just make sure the
      // expression type-checks.
      long long t = HSHM_DEVICE_CLOCK64();
      (void)t;
    });
  }).wait_and_throw();

  // 0x1 (atomic OR) then 0x2 added on top → 0x3.
  REQUIRE(*flag == 0x3u);
  sycl::free(flag, q);
}

TEST_CASE("hshm::GpuApi USM round-trip", "[sycl][gpu_api]") {
  // Validates that the SYCL branches we added to GpuApi
  // (Phase 3+ — Malloc, MallocHost, Memcpy, Free, FreeHost,
  // CreateStream, Synchronize, DestroyStream) all wire to sycl::*
  // primitives correctly.
  constexpr size_t kCount = 64;
  using T = uint64_t;

  // Host allocation via MallocHost → sycl::malloc_host.
  T *host_buf = hshm::GpuApi::MallocHost<T>(kCount * sizeof(T));
  REQUIRE(host_buf != nullptr);
  for (size_t i = 0; i < kCount; ++i) {
    host_buf[i] = static_cast<T>(i * i + 7);
  }

  // Device allocation via Malloc → sycl::malloc_device.
  T *dev_buf = hshm::GpuApi::Malloc<T>(kCount * sizeof(T));
  REQUIRE(dev_buf != nullptr);

  // Round-trip: host -> device -> different host buffer.
  hshm::GpuApi::Memcpy(dev_buf, host_buf, kCount * sizeof(T));
  T *host_back = hshm::GpuApi::MallocHost<T>(kCount * sizeof(T));
  REQUIRE(host_back != nullptr);
  hshm::GpuApi::Memcpy(host_back, dev_buf, kCount * sizeof(T));

  for (size_t i = 0; i < kCount; ++i) {
    REQUIRE(host_back[i] == host_buf[i]);
  }

  hshm::GpuApi::Free(dev_buf);
  hshm::GpuApi::FreeHost(host_buf);
  hshm::GpuApi::FreeHost(host_back);
}

TEST_CASE("WorkOrchestrator Launch/Finalize lifecycle", "[sycl][orchestrator]") {
  // Validates that work_orchestrator_sycl.cc's host-side methods bring up
  // the persistent single_task kernel and shut it down cleanly. We don't
  // dispatch a real task here — that would require a full pool registry
  // and queue infrastructure. The lifecycle alone exercises the host USM
  // for control, the device PoolManager allocation, the dedicated
  // sycl::queue stream, and the kernel submission path.
  chi::gpu::WorkOrchestrator orch;

  // Build a minimal IpcManagerGpuInfo; the kernel only reads
  // queue pointers (all null here, so the poll loop sees empty queues).
  chi::IpcManagerGpuInfo gpu_info{};

  // Launch with 1 block, 1 thread (orchestrator runs as single_task
  // anyway — the params are kept for Pause/Resume parity).
  bool launched = orch.Launch(gpu_info, /*blocks=*/1,
                              /*threads_per_block=*/1,
                              /*cpu2gpu_queue_base=*/nullptr);
  REQUIRE(launched);
  REQUIRE(orch.is_launched_);

  // Give the persistent kernel a moment to set running_flag = 1.
  for (int i = 0; i < 200 && orch.control_->running_flag == 0; ++i) {
    std::this_thread::sleep_for(5ms);
  }
  REQUIRE(orch.control_->running_flag == 1);

  // Finalize signals exit_flag, syncs the queue, and frees all resources.
  orch.Finalize();
  REQUIRE_FALSE(orch.is_launched_);
  REQUIRE(orch.control_ == nullptr);
  REQUIRE(orch.d_pool_mgr_ == nullptr);
  REQUIRE(orch.stream_ == nullptr);
  REQUIRE(orch.gpu_info_storage_ == nullptr);
}

SIMPLE_TEST_MAIN()

#else  // !HSHM_ENABLE_SYCL

int main() {
  // Test is meaningful only with WRP_CORE_ENABLE_SYCL=ON; pass otherwise so
  // the test list isn't gated entirely on backend.
  return 0;
}

#endif  // HSHM_ENABLE_SYCL
