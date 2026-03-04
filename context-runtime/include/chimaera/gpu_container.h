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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_

#include "chimaera/types.h"
#include "chimaera/pool_query.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"

namespace chi {
namespace gpu {

/**
 * Lightweight RunContext for GPU-side task execution.
 * Unlike the host RunContext, this has no STL members, no coroutines,
 * and is fully GPU-compatible.
 */
struct GpuRunContext {
  u32 block_id_;
  u32 thread_id_;

  HSHM_GPU_FUN GpuRunContext() : block_id_(0), thread_id_(0) {}
  HSHM_GPU_FUN GpuRunContext(u32 block_id, u32 thread_id)
      : block_id_(block_id), thread_id_(thread_id) {}
};

class Container;

/**
 * Function pointer types for GPU container method dispatch.
 *
 * These replace virtual functions to avoid cross-library vtable
 * resolution issues with CUDA shared libraries. Each GPU companion
 * library (.so) captures device function addresses in its allocation
 * kernel, which are globally valid device addresses.
 */
using RunFn = void (*)(Container *, u32, hipc::FullPtr<Task>,
                        GpuRunContext &);
using AllocLoadFn = hipc::FullPtr<Task> (*)(Container *, u32,
                                             LocalLoadTaskArchive &);
using SaveFn = void (*)(Container *, u32, LocalSaveTaskArchive &,
                         const hipc::FullPtr<Task> &);

/**
 * GPU-side container base class
 *
 * Uses function pointer dispatch instead of virtual functions to support
 * cross-library dispatch from the megakernel. Each concrete container's
 * CHI_TASK_GPU_CC macro generates __device__ wrapper functions and sets
 * the function pointers during GPU-side allocation.
 *
 * Concrete containers implement RunImpl, AllocLoadImpl, SaveImpl methods
 * which are called through the function pointer dispatch table.
 */
class Container {
 public:
  PoolId pool_id_;
  u32 container_id_;
  HSHM_DEFAULT_ALLOC_GPU_T *gpu_alloc_ = nullptr;  /**< Set by worker before dispatch */

  /** Function pointer dispatch table (set by CHI_TASK_GPU_CC macro) */
  RunFn run_fn_ = nullptr;
  AllocLoadFn alloc_load_fn_ = nullptr;
  SaveFn save_fn_ = nullptr;

  HSHM_GPU_FUN Container() : container_id_(0), gpu_alloc_(nullptr) {}
  HSHM_GPU_FUN ~Container() = default;

  /**
   * Initialize the GPU container
   * @param pool_id Pool identifier
   * @param container_id Container ID (typically node_id)
   */
  HSHM_GPU_FUN void Init(const PoolId &pool_id, u32 container_id) {
    pool_id_ = pool_id;
    container_id_ = container_id;
  }

  /**
   * Execute a task method on the GPU via function pointer dispatch.
   * @param method Method ID to execute
   * @param task_ptr Full pointer to the task
   * @param rctx GPU run context
   */
  HSHM_GPU_FUN void Run(u32 method, hipc::FullPtr<Task> task_ptr,
                         GpuRunContext &rctx) {
    run_fn_(this, method, task_ptr, rctx);
  }

  /**
   * Allocate and deserialize a task via function pointer dispatch.
   * @param method Method ID identifying the task type
   * @param archive LocalLoadTaskArchive containing serialized input
   * @return FullPtr to the deserialized task, or null on failure
   */
  HSHM_GPU_FUN hipc::FullPtr<Task> LocalAllocLoadTask(
      u32 method, LocalLoadTaskArchive &archive) {
    return alloc_load_fn_(this, method, archive);
  }

  /**
   * Serialize task output via function pointer dispatch.
   * @param method Method ID identifying the task type
   * @param archive LocalSaveTaskArchive to write output into
   * @param task FullPtr to the completed task
   */
  HSHM_GPU_FUN void LocalSaveTask(
      u32 method, LocalSaveTaskArchive &archive,
      const hipc::FullPtr<Task> &task) {
    save_fn_(this, method, archive, task);
  }

  /**
   * Get remaining work for load balancing
   * @return Amount of work remaining (0 = idle)
   */
  HSHM_GPU_FUN u64 GetWorkRemaining() const { return 0; }
};

}  // namespace gpu
}  // namespace chi

/**
 * CHI_TASK_GPU_CC macro - Generates GPU container allocation/construction kernels
 * and function pointer dispatch wrappers.
 *
 * Generates:
 * - __device__ dispatch wrappers that static_cast to the concrete type
 * - Device kernel to allocate container and set function pointers
 * - Device kernel to allocate + Init container with function pointers
 * - Host functions callable from CPU to create GPU containers
 *
 * The concrete class T must implement:
 * - HSHM_GPU_FUN void RunImpl(u32 method, FullPtr<Task> task, GpuRunContext &rctx)
 * - HSHM_GPU_FUN FullPtr<Task> AllocLoadImpl(u32 method, LocalLoadTaskArchive &ar)
 * - HSHM_GPU_FUN void SaveImpl(u32 method, LocalSaveTaskArchive &ar, const FullPtr<Task> &task)
 *
 * @param T Fully-qualified GPU container class name
 */
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#define CHI_TASK_GPU_CC(T)                                                     \
  /* Device-side dispatch wrappers with addresses valid in this library */      \
  __device__ void _chimod_dispatch_run(                                        \
      chi::gpu::Container *self, chi::u32 method,                              \
      hipc::FullPtr<chi::Task> task_ptr, chi::gpu::GpuRunContext &rctx) {      \
    static_cast<T *>(self)->RunImpl(method, task_ptr, rctx);                   \
  }                                                                            \
  __device__ hipc::FullPtr<chi::Task> _chimod_dispatch_alloc_load(             \
      chi::gpu::Container *self, chi::u32 method,                              \
      chi::LocalLoadTaskArchive &archive) {                                    \
    return static_cast<T *>(self)->AllocLoadImpl(method, archive);             \
  }                                                                            \
  __device__ void _chimod_dispatch_save(                                       \
      chi::gpu::Container *self, chi::u32 method,                              \
      chi::LocalSaveTaskArchive &archive,                                      \
      const hipc::FullPtr<chi::Task> &task) {                                  \
    static_cast<T *>(self)->SaveImpl(method, archive, task);                   \
  }                                                                            \
                                                                               \
  __global__ void _chimod_gpu_alloc_kernel(T **out) {                          \
    auto *obj = new T();                                                       \
    obj->run_fn_ = _chimod_dispatch_run;                                       \
    obj->alloc_load_fn_ = _chimod_dispatch_alloc_load;                         \
    obj->save_fn_ = _chimod_dispatch_save;                                     \
    *out = obj;                                                                \
  }                                                                            \
                                                                               \
  __global__ void _chimod_gpu_new_kernel(T **out, const chi::PoolId *pid,      \
                                          chi::u32 cid) {                      \
    T *obj = new T();                                                          \
    obj->run_fn_ = _chimod_dispatch_run;                                       \
    obj->alloc_load_fn_ = _chimod_dispatch_alloc_load;                         \
    obj->save_fn_ = _chimod_dispatch_save;                                     \
    obj->Init(*pid, cid);                                                      \
    *out = obj;                                                                \
  }                                                                            \
                                                                               \
  extern "C" void *alloc_chimod_gpu() {                                        \
    void *stream = hshm::GpuApi::CreateStream();                               \
    T **d_out = hshm::GpuApi::Malloc<T *>(sizeof(T *));                        \
    _chimod_gpu_alloc_kernel<<<1, 1, 0,                                        \
        static_cast<cudaStream_t>(stream)>>>(d_out);                           \
    hshm::GpuApi::Synchronize(stream);                                         \
    T *h_ptr = nullptr;                                                        \
    hshm::GpuApi::Memcpy(&h_ptr, d_out, sizeof(T *));                         \
    hshm::GpuApi::Free(d_out);                                                \
    hshm::GpuApi::DestroyStream(stream);                                       \
    return static_cast<void *>(h_ptr);                                         \
  }                                                                            \
                                                                               \
  extern "C" void *new_chimod_gpu(const chi::PoolId *pool_id, chi::u32 cid) { \
    void *stream = hshm::GpuApi::CreateStream();                               \
    T **d_out = hshm::GpuApi::Malloc<T *>(sizeof(T *));                        \
    chi::PoolId *d_pid =                                                       \
        hshm::GpuApi::Malloc<chi::PoolId>(sizeof(chi::PoolId));                \
    hshm::GpuApi::Memcpy(d_pid, pool_id, sizeof(chi::PoolId));                \
    _chimod_gpu_new_kernel<<<1, 1, 0,                                          \
        static_cast<cudaStream_t>(stream)>>>(d_out, d_pid, cid);              \
    hshm::GpuApi::Synchronize(stream);                                         \
    T *h_ptr = nullptr;                                                        \
    hshm::GpuApi::Memcpy(&h_ptr, d_out, sizeof(T *));                         \
    hshm::GpuApi::Free(d_out);                                                \
    hshm::GpuApi::Free(d_pid);                                                \
    hshm::GpuApi::DestroyStream(stream);                                       \
    return static_cast<void *>(h_ptr);                                         \
  }

#else
#define CHI_TASK_GPU_CC(T)
#endif

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_CONTAINER_H_
