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
 * Per-chimod SYCL smoke tests.
 *
 * Each test calls gpu::AllocGpuContainerHost for one chimod, which:
 *   1. Looks up the module ID via GetGpuModuleId(name).
 *   2. Sizes the device USM region via GetGpuModuleSize(module_id).
 *   3. Submits a single_task that placement-news the concrete GpuRuntime
 *      subclass and runs the autogen-emitted constructor (which sets up
 *      the entire function-pointer dispatch table — RunImpl,
 *      AllocTaskImpl, LoadTaskTmpl, SaveTaskTmpl, DestroyTaskImpl, ...).
 *   4. Returns the device USM pointer to the constructed container.
 *
 * Validating that the call returns non-null + that the container's
 * function pointers are populated proves that:
 *   - The autogen kernel's SYCL branch compiled and JITed correctly
 *     against the chimod's full GpuRuntime layout.
 *   - HSHM_INDIRECTLY_CALLABLE made every *Impl address-takeable under
 *     DPC++.
 *   - HSHM_DEVICE_EXTERN-tagged HSHM_GPU_FUN methods linked across TUs.
 *   - The chimod's transitive includes (chimaera/types.h, ipc_manager.h,
 *     hermes_shm/util/{logging,error,gpu_intrinsics}.h, etc.) are all
 *     SYCL-device-pass clean.
 */

#if HSHM_ENABLE_SYCL

#include "simple_test.h"

#include <chimaera/types.h>
#include <hermes_shm/util/gpu_api.h>

// Pull the autogen header that defines AllocGpuContainerHost +
// GetGpuModuleId + GetGpuModuleSize.
#include "autogen/gpu_work_orchestrator_modules.h"

#include <cstring>

namespace {

/**
 * Read back the bytes of a placement-newed device GpuRuntime into a host
 * staging buffer so we can sanity-check the function-pointer table fields.
 * The base chi::gpu::Container layout starts with `pool_id_`, `container_id_`,
 * and the `RunFn run_` pointer — so a non-null run_ field means the autogen
 * constructor reached past pool_id assignment and successfully set up the
 * dispatch table on device.
 */
void ReadbackContainer(void *d_obj, size_t size,
                       std::vector<unsigned char> &out) {
  out.assign(size, 0);
  hshm::GpuApi::Memcpy(out.data(), reinterpret_cast<unsigned char *>(d_obj),
                       size);
}

/** Locate the run_ function-pointer field in a chi::gpu::Container layout. */
chi::gpu::Container::RunFn ExtractRunFn(const std::vector<unsigned char> &buf) {
  // chi::gpu::Container layout (from container.h:120-134):
  //   PoolId pool_id_;       // sizeof(PoolId) bytes
  //   u32 container_id_;     // 4 bytes
  //   RunFn run_;            // first function pointer in the table
  // PoolId is 16 bytes (two u64s, see types.h).
  constexpr size_t kRunFnOffset =
      sizeof(chi::PoolId) + sizeof(chi::u32) +
      // padding — RunFn pointer is 8-byte aligned on x86_64; offset_of would
      // be ideal but PoolId+u32 is already 20 bytes -> next 8-aligned is 24.
      ((sizeof(chi::PoolId) + sizeof(chi::u32)) % alignof(void *) == 0
           ? 0
           : alignof(void *) -
                 ((sizeof(chi::PoolId) + sizeof(chi::u32)) % alignof(void *)));
  REQUIRE(buf.size() >= kRunFnOffset + sizeof(void *));
  chi::gpu::Container::RunFn run_fn = nullptr;
  std::memcpy(&run_fn, buf.data() + kRunFnOffset, sizeof(run_fn));
  return run_fn;
}

/**
 * Per-chimod allocation test body. Calls AllocGpuContainerHost, validates
 * the returned device pointer, reads the dispatch table back, asserts run_
 * was populated, and frees the device USM.
 */
void RunChimodAllocTest(const char *chimod_name,
                        chi::u32 expected_container_id) {
  INFO("chimod = " << chimod_name);

  // Sanity-check the module ID + size lookup tables.
  chi::u32 module_id = chi::gpu::GetGpuModuleId(chimod_name);
  REQUIRE(module_id != 0xFFFFFFFFu);
  size_t obj_size = chi::gpu::GetGpuModuleSize(module_id);
  REQUIRE(obj_size > 0);
  REQUIRE(obj_size >= sizeof(chi::gpu::Container));

  // Build a deterministic PoolId so we can confirm the autogen ctor ran
  // Init(*pid, cid) — pool_id_ should match what we passed in.
  chi::PoolId pool_id;
  pool_id.major_ = 0xC0DE0000u + module_id;
  pool_id.minor_ = expected_container_id;

  void *d_obj = chi::gpu::AllocGpuContainerHost(pool_id, expected_container_id,
                                                chimod_name);
  REQUIRE(d_obj != nullptr);

  // Inspect the constructed object on host.
  std::vector<unsigned char> staging;
  ReadbackContainer(d_obj, obj_size, staging);

  // Verify Init() populated pool_id_ and container_id_ at the expected
  // offsets (chi::gpu::Container is the base class; layout begins with
  // pool_id_ then container_id_).
  chi::PoolId obs_pool_id;
  chi::u32 obs_container_id = 0;
  std::memcpy(&obs_pool_id, staging.data(), sizeof(obs_pool_id));
  std::memcpy(&obs_container_id, staging.data() + sizeof(obs_pool_id),
              sizeof(obs_container_id));
  REQUIRE(obs_pool_id.major_ == pool_id.major_);
  REQUIRE(obs_pool_id.minor_ == pool_id.minor_);
  REQUIRE(obs_container_id == expected_container_id);

  // Verify the RunFn slot in the function-pointer table is non-null —
  // i.e. the autogen ctor reached the `run_ = &RunImpl;` assignment and
  // DPC++ was able to take the address of the indirectly-callable wrapper.
  chi::gpu::Container::RunFn run_fn = ExtractRunFn(staging);
  REQUIRE(run_fn != nullptr);

  // Free the device USM region. The matching ~GpuRuntime() runs on host
  // (placement-newed dtor) but the underlying allocation was sycl::malloc_device.
  hshm::GpuApi::Free(d_obj);
}

}  // namespace

TEST_CASE("chimod admin: GpuRuntime allocation", "[sycl][chimod][admin]") {
  RunChimodAllocTest("chimaera_admin", /*container_id=*/1u);
}

TEST_CASE("chimod bdev: GpuRuntime allocation", "[sycl][chimod][bdev]") {
  RunChimodAllocTest("chimaera_bdev", /*container_id=*/2u);
}

TEST_CASE("chimod CTE core: GpuRuntime allocation", "[sycl][chimod][cte_core]") {
  RunChimodAllocTest("wrp_cte_core", /*container_id=*/3u);
}

TEST_CASE("GetGpuModuleId rejects unknown chimod", "[sycl][chimod]") {
  REQUIRE(chi::gpu::GetGpuModuleId("not_a_real_chimod") == 0xFFFFFFFFu);
}

SIMPLE_TEST_MAIN()

#else  // !HSHM_ENABLE_SYCL

int main() {
  return 0;
}

#endif  // HSHM_ENABLE_SYCL
