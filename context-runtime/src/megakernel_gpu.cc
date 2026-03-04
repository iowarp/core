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
 * GPU Megakernel implementation
 *
 * A persistent GPU kernel that polls CPU->GPU and GPU->GPU queues for tasks,
 * dispatches them to GPU-side containers via gpu::Worker / gpu::PoolManager,
 * and communicates results back through FutureShm completion flags.
 *
 * IMPORTANT: Device function pointers are NOT valid across CUDA shared
 * libraries. The companion _runtime_gpu.so libraries allocate GPU containers,
 * but the function pointers they set on those containers point to device
 * functions in the companion library's CUDA module — which are not callable
 * from the megakernel's persistent kernel (different CUDA module).
 *
 * Solution: This file includes all GPU container class definitions and
 * defines its own dispatch wrappers. After container allocation, a fixup
 * kernel overwrites the container's function pointers with addresses from
 * THIS compilation unit's CUDA module.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/ipc_manager.h"
#include "chimaera/megakernel.h"
#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/gpu_worker.h"
#include "chimaera/config_manager.h"

// Include GPU container class definitions so their methods compile in
// this CUDA module's device code context
#include "chimaera/admin/admin_gpu_runtime.h"
#include "chimaera/MOD_NAME/MOD_NAME_gpu_runtime.h"

namespace chi {

//==============================================================================
// Megakernel-local dispatch wrappers
//
// These __device__ functions are compiled in this CUDA module's device code
// context. When called from the megakernel's persistent kernel, they resolve
// to valid device addresses (same module). The fixup kernel below overwrites
// container function pointers with these addresses after allocation.
//==============================================================================

namespace mk_admin {
__device__ void dispatch_run(
    gpu::Container *self, u32 method,
    hipc::FullPtr<Task> task_ptr, gpu::GpuRunContext &rctx) {
  static_cast<chimaera::admin::GpuRuntime *>(self)->RunImpl(
      method, task_ptr, rctx);
}
__device__ hipc::FullPtr<Task> dispatch_alloc_load(
    gpu::Container *self, u32 method,
    LocalLoadTaskArchive &archive) {
  return static_cast<chimaera::admin::GpuRuntime *>(self)->AllocLoadImpl(
      method, archive);
}
__device__ void dispatch_save(
    gpu::Container *self, u32 method,
    LocalSaveTaskArchive &archive, const hipc::FullPtr<Task> &task) {
  static_cast<chimaera::admin::GpuRuntime *>(self)->SaveImpl(
      method, archive, task);
}
}  // namespace mk_admin

namespace mk_mod_name {
__device__ void dispatch_run(
    gpu::Container *self, u32 method,
    hipc::FullPtr<Task> task_ptr, gpu::GpuRunContext &rctx) {
  static_cast<chimaera::MOD_NAME::GpuRuntime *>(self)->RunImpl(
      method, task_ptr, rctx);
}
__device__ hipc::FullPtr<Task> dispatch_alloc_load(
    gpu::Container *self, u32 method,
    LocalLoadTaskArchive &archive) {
  return static_cast<chimaera::MOD_NAME::GpuRuntime *>(self)->AllocLoadImpl(
      method, archive);
}
__device__ void dispatch_save(
    gpu::Container *self, u32 method,
    LocalSaveTaskArchive &archive, const hipc::FullPtr<Task> &task) {
  static_cast<chimaera::MOD_NAME::GpuRuntime *>(self)->SaveImpl(
      method, archive, task);
}
}  // namespace mk_mod_name

/**
 * Module ID enum for megakernel dispatch table.
 * Must match the values used by FixupContainerDispatch.
 */
enum GpuModuleId : u32 {
  kGpuModAdmin = 0,
  kGpuModModName = 1,
};

/**
 * Fixup kernel: overwrites a container's function pointer dispatch table
 * with addresses from this CUDA module's device code context.
 *
 * Must be called after companion library allocates the container but before
 * the megakernel dispatches tasks to it.
 *
 * @param container Device pointer to the gpu::Container
 * @param module_id GpuModuleId identifying which dispatch wrappers to use
 */
__global__ void _mk_fixup_dispatch(gpu::Container *container,
                                    u32 module_id) {
  switch (module_id) {
    case kGpuModAdmin:
      container->run_fn_ = mk_admin::dispatch_run;
      container->alloc_load_fn_ = mk_admin::dispatch_alloc_load;
      container->save_fn_ = mk_admin::dispatch_save;
      break;
    case kGpuModModName:
      container->run_fn_ = mk_mod_name::dispatch_run;
      container->alloc_load_fn_ = mk_mod_name::dispatch_alloc_load;
      container->save_fn_ = mk_mod_name::dispatch_save;
      break;
    default:
      // Unknown module — leave function pointers as-is
      break;
  }
}

/**
 * GPU Megakernel - persistent kernel for GPU task execution.
 */
__global__ void chimaera_megakernel(gpu::PoolManager *pool_mgr,
                                     MegakernelControl *control,
                                     IpcManagerGpuInfo gpu_info,
                                     u32 num_blocks) {
  // All threads: initialize per-block IpcManager (ArenaAllocators)
  CHIMAERA_MEGAKERNEL_INIT(gpu_info, num_blocks);

  // Only block 0, thread 0 runs the worker loop
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    control->running_flag = 1;

    gpu::Worker worker;
    worker.Init(0,
                gpu_info.to_gpu_queue,
                gpu_info.gpu_to_gpu_queue,
                pool_mgr,
                gpu_info.queue_backend_base);

    // Poll until exit signal
    while (!control->exit_flag) {
      worker.PollOnce();
    }

    worker.Finalize();
  }

  // Other blocks/threads: wait for exit signal
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    while (!control->exit_flag) {
      // Spin-wait
    }
  }
}

//==============================================================================
// MegakernelLauncher out-of-class method implementations
//==============================================================================

/**
 * Launch the persistent megakernel on the GPU.
 */
bool MegakernelLauncher::Launch(const IpcManagerGpuInfo &gpu_info, u32 blocks,
                                 u32 threads_per_block) {
  if (is_launched_) {
    return true;
  }

  // Allocate MegakernelControl in pinned host memory
  MegakernelControl *pinned_control = nullptr;
  cudaMallocHost(&pinned_control, sizeof(MegakernelControl));
  control_ = pinned_control;
  if (!control_) {
    HLOG(kError, "MegakernelLauncher: Failed to allocate control structure");
    return false;
  }
  control_->exit_flag = 0;
  control_->running_flag = 0;

  // Allocate gpu::PoolManager on device
  gpu::PoolManager *d_pm = hshm::GpuApi::Malloc<gpu::PoolManager>(
      sizeof(gpu::PoolManager));
  if (!d_pm) {
    HLOG(kError, "MegakernelLauncher: Failed to allocate GPU PoolManager");
    cudaFreeHost(control_);
    control_ = nullptr;
    return false;
  }
  d_pool_mgr_ = d_pm;

  // Initialize PoolManager on device (zero-initialize)
  gpu::PoolManager host_pm;
  hshm::GpuApi::Memcpy(d_pm, &host_pm, sizeof(gpu::PoolManager));

  // Increase GPU stack size for deep template call chains
  cudaDeviceSetLimit(cudaLimitStackSize, 131072);

  // Create dedicated stream
  stream_ = hshm::GpuApi::CreateStream();

  // Save launch parameters for Pause/Resume
  blocks_ = blocks;
  threads_per_block_ = threads_per_block;

  // Launch persistent megakernel
  HLOG(kInfo, "Launching megakernel with {} blocks, {} threads/block",
       blocks, threads_per_block);
  chimaera_megakernel<<<blocks, threads_per_block, 0,
      static_cast<cudaStream_t>(stream_)>>>(
      d_pm, control_, gpu_info, blocks);

  is_launched_ = true;
  HLOG(kInfo, "Megakernel launched successfully");
  return true;
}

/** Stop the megakernel and free resources. */
void MegakernelLauncher::Finalize() {
  if (!is_launched_) {
    return;
  }

  if (control_) {
    control_->exit_flag = 1;
  }

  if (stream_) {
    hshm::GpuApi::Synchronize(stream_);
    hshm::GpuApi::DestroyStream(stream_);
    stream_ = nullptr;
  } else {
    hshm::GpuApi::Synchronize();
  }

  if (d_pool_mgr_) {
    hshm::GpuApi::Free(d_pool_mgr_);
    d_pool_mgr_ = nullptr;
  }
  if (control_) {
    cudaFreeHost(control_);
    control_ = nullptr;
  }

  is_launched_ = false;
  HLOG(kInfo, "Megakernel finalized");
}

/** Pause the megakernel. */
void MegakernelLauncher::Pause() {
  if (!is_launched_) {
    return;
  }
  control_->exit_flag = 1;
  hshm::GpuApi::Synchronize(stream_);
  is_launched_ = false;
  HLOG(kInfo, "Megakernel paused");
}

/** Resume a paused megakernel. */
void MegakernelLauncher::Resume(const IpcManagerGpuInfo &gpu_info) {
  if (is_launched_) {
    return;
  }
  control_->exit_flag = 0;
  control_->running_flag = 0;

  auto *d_pm = static_cast<gpu::PoolManager *>(d_pool_mgr_);
  chimaera_megakernel<<<blocks_, threads_per_block_, 0,
      static_cast<cudaStream_t>(stream_)>>>(
      d_pm, control_, gpu_info, blocks_);

  is_launched_ = true;
  HLOG(kInfo, "Megakernel resumed with {} blocks, {} threads/block",
       blocks_, threads_per_block_);
}

/**
 * Register a GPU container with the device-side PoolManager, then fix up
 * the container's function pointer dispatch table with megakernel-local
 * device addresses.
 */
void MegakernelLauncher::RegisterGpuContainer(const PoolId &pool_id,
                                               void *gpu_container_ptr,
                                               const std::string &chimod_name) {
  if (!d_pool_mgr_ || !gpu_container_ptr) {
    return;
  }

  // Step 1: Register container with PoolManager
  auto *d_pm = static_cast<gpu::PoolManager *>(d_pool_mgr_);
  gpu::PoolManager host_pm;
  hshm::GpuApi::Memcpy(&host_pm, d_pm, sizeof(gpu::PoolManager));
  host_pm.RegisterContainer(pool_id,
                             static_cast<gpu::Container *>(gpu_container_ptr));
  hshm::GpuApi::Memcpy(d_pm, &host_pm, sizeof(gpu::PoolManager));

  // Step 2: Determine module ID from ChiMod name
  u32 module_id = 0xFFFFFFFF;  // Unknown
  if (chimod_name == "chimaera_admin") {
    module_id = kGpuModAdmin;
  } else if (chimod_name == "chimaera_MOD_NAME") {
    module_id = kGpuModModName;
  }

  // Step 3: Fix up function pointers with megakernel-local device addresses
  if (module_id != 0xFFFFFFFF) {
    auto *container = static_cast<gpu::Container *>(gpu_container_ptr);
    void *fixup_stream = hshm::GpuApi::CreateStream();
    _mk_fixup_dispatch<<<1, 1, 0,
        static_cast<cudaStream_t>(fixup_stream)>>>(container, module_id);
    hshm::GpuApi::Synchronize(fixup_stream);
    hshm::GpuApi::DestroyStream(fixup_stream);
    HLOG(kInfo, "Fixed up dispatch for pool {} (module={})",
         pool_id, chimod_name);
  } else {
    HLOG(kWarning, "Unknown GPU module: {} — dispatch not fixed up",
         chimod_name);
  }

  HLOG(kInfo, "Registered GPU container for pool {}", pool_id);
}

}  // namespace chi

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
