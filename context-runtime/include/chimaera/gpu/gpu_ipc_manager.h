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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_IPC_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_IPC_MANAGER_H_

#include "chimaera/types.h"
#include "chimaera/task.h"
#include "chimaera/gpu/gpu_info.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/ipc/ipc_gpu2gpu.h"
#include "chimaera/ipc/ipc_cpu2gpu.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"

namespace chi {
namespace gpu {

/**
 * GPU IPC infrastructure manager.
 *
 * Owns all GPU-specific state: device-side queue pointers, allocators,
 * and host-side backend lifecycle. On GPU device, CHI_IPC resolves to
 * gpu::IpcManager* via GetBlockIpcManager(). On CPU host,
 * chi::IpcManager delegates GPU operations through gpu_ipc_.
 *
 * Methods are HSHM_CROSS_FUN when called from client code that compiles
 * for both host and device (e.g. NewTask, Send, DelTask). Methods that
 * use CUDA builtins and are only called from device code are HSHM_GPU_FUN.
 */
class IpcManager {
 public:
  // ================================================================
  // GPU thread/warp topology utilities
  // ================================================================

#if HSHM_IS_GPU_COMPILER
  /** Get linear GPU thread ID for 1D/2D/3D blocks */
  static HSHM_GPU_FUN inline int GetGpuThreadId() {
    return threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
  }
  /** Get total threads in block */
  static HSHM_GPU_FUN inline int GetGpuNumThreads() {
    return blockDim.x * blockDim.y * blockDim.z;
  }
  /** Get total number of warps in the grid */
  static HSHM_GPU_FUN inline u32 GetNumWarps() {
    return (gridDim.x * blockDim.x) / 32;
  }
#endif
  /** Get the global warp ID within the grid */
  static HSHM_CROSS_FUN inline u32 GetWarpId() {
#if HSHM_IS_GPU
    return (blockIdx.x * blockDim.x + threadIdx.x) / 32;
#else
    return 0;
#endif
  }
  /** Get the lane ID (0-31) within the warp */
  static HSHM_CROSS_FUN inline u32 GetLaneId() {
#if HSHM_IS_GPU
    return threadIdx.x % 32;
#else
    return 0;
#endif
  }
  /** Whether this thread is the warp scheduler (lane 0) */
  static HSHM_CROSS_FUN inline bool IsWarpScheduler() {
    return GetLaneId() == 0;
  }

  // ================================================================
  // GPU allocator and initialization methods
  // ================================================================

  /**
   * Initialize GPU client/orchestrator state from IpcManagerGpuInfo.
   */
  HSHM_GPU_FUN
  void ClientInitGpu(const IpcManagerGpuInfo &gpu_info, int num_threads,
                     int num_blocks = 1) {
    gpu_info_ = gpu_info;
    is_gpu_runtime_ = false;
    u32 num_warps = num_threads / 32;
    if (num_warps < 1) num_warps = 1;
    if (gpu_info_.backend.data_ != nullptr) {
      auto *alloc = reinterpret_cast<hipc::RoundRobinAllocator *>(
          gpu_info_.backend.data_);
      if (alloc->heap_ready_.load() == 1) {
        // Reattach to existing allocator
        gpu_alloc_ = alloc;
      } else {
        // First-time initialization
        InitHeapAllocator(gpu_info_.backend, num_warps, &gpu_alloc_);
      }
    }
  }

  /**
   * Initialize a RoundRobinAllocator from a MemoryBackend.
   */
  HSHM_GPU_FUN
  static void InitHeapAllocator(const hipc::MemoryBackend &backend,
                                int num_threads,
                                hipc::RoundRobinAllocator **alloc_out) {
    char *base = backend.data_;
    size_t data_capacity = backend.data_capacity_;
    auto *alloc = reinterpret_cast<hipc::RoundRobinAllocator *>(base);
    new (alloc) hipc::RoundRobinAllocator();
    hipc::MemoryBackend sub_backend;
    sub_backend.data_ = base;
    sub_backend.data_capacity_ = data_capacity;
    sub_backend.id_ = backend.id_;
    size_t overhead = sizeof(hipc::RoundRobinAllocator);
    size_t thread_unit = (data_capacity - overhead) / num_threads;
    alloc->shm_init(sub_backend, 0, num_threads, thread_unit);
    alloc->MarkReady();
    *alloc_out = alloc;
  }

  HSHM_GPU_FUN int ClaimPartition() {
    return gpu_alloc_ ? gpu_alloc_->ClaimPartition() : 0;
  }

  HSHM_GPU_FUN hipc::BuddyAllocator *GetPartitionAlloc(int partition_id) {
    return gpu_alloc_ ? gpu_alloc_->GetAllocator(partition_id) : nullptr;
  }

  HSHM_CROSS_FUN CHI_TASK_ALLOC_T *GetMainAllocator() {
#if HSHM_IS_GPU
    return static_cast<CHI_TASK_ALLOC_T *>(static_cast<void *>(gpu_alloc_));
#else
    return nullptr;
#endif
  }

  template <typename TaskT>
  HSHM_CROSS_FUN static FutureShm *GetTaskFutureShm(TaskT *task) {
    return reinterpret_cast<FutureShm *>(
        reinterpret_cast<char *>(task) + sizeof(TaskT));
  }

  template <typename T, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<T> NewObj(Args &&...args) {
    hipc::FullPtr<char> buffer = AllocateBuffer(sizeof(T));
    if (buffer.IsNull()) return hipc::FullPtr<T>();
    new (buffer.ptr_) T(std::forward<Args>(args)...);
    return buffer.Cast<T>();
  }

  HSHM_CROSS_FUN hipc::FullPtr<char> AllocateBuffer(size_t size) {
#if HSHM_IS_GPU
    if (gpu_alloc_ != nullptr) {
      return gpu_alloc_->AllocateObjs<char>(size);
    }
#endif
    (void)size;
    return hipc::FullPtr<char>::GetNull();
  }

  HSHM_CROSS_FUN void FreeBuffer(hipc::FullPtr<char> buffer_ptr) {
#if HSHM_IS_GPU
    if (buffer_ptr.IsNull()) return;
    if (gpu_alloc_ && gpu_alloc_->ContainsPtr(buffer_ptr.ptr_)) {
      gpu_alloc_->Free(buffer_ptr);
    }
#else
    (void)buffer_ptr;
#endif
  }

  HSHM_CROSS_FUN void FreeBuffer(hipc::ShmPtr<char> buffer_ptr) {
    if (buffer_ptr.IsNull()) return;
    hipc::FullPtr<char> full_ptr(ToFullPtr<char>(buffer_ptr));
    FreeBuffer(full_ptr);
  }

  HSHM_CROSS_FUN hipc::Arena<hipc::RoundRobinAllocator> PushArena(
      size_t size) {
#if HSHM_IS_GPU
    if (gpu_alloc_ != nullptr) {
      return gpu_alloc_->PushArena(size);
    }
#endif
    (void)size;
    return hipc::Arena<hipc::RoundRobinAllocator>();
  }

  HSHM_GPU_FUN hipc::FullPtr<char> AllocateDeviceData(size_t size) {
    if (gpu_alloc_ != nullptr) {
      return gpu_alloc_->AllocateObjs<char>(size);
    }
    return hipc::FullPtr<char>::GetNull();
  }

  // ================================================================
  // Task allocation / deletion (CROSS_FUN: called from client code)
  // ================================================================

  template <typename TaskT, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<TaskT> NewTaskBase(size_t append_size,
                                                    Args &&...args) {
#if HSHM_IS_GPU
    if (!IsWarpScheduler()) return hipc::FullPtr<TaskT>();
    if (!gpu_alloc_) return hipc::FullPtr<TaskT>();
    size_t total = sizeof(TaskT) + append_size;
    auto fp = gpu_alloc_->template AllocateObjs<char>(total);
    if (fp.IsNull()) return hipc::FullPtr<TaskT>();
    new (fp.ptr_) TaskT(std::forward<Args>(args)...);
    return fp.template Cast<TaskT>();
#else
    (void)append_size;
    return hipc::FullPtr<TaskT>();
#endif
  }

  template <typename TaskT, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<TaskT> NewTask(Args &&...args) {
#if HSHM_IS_GPU
    if (!IsWarpScheduler()) return hipc::FullPtr<TaskT>();
    size_t append = sizeof(FutureShm);
    auto result = NewTaskBase<TaskT>(append, std::forward<Args>(args)...);
    if (!result.IsNull()) {
      result->pod_size_ = static_cast<u32>(sizeof(TaskT));
      char *base = reinterpret_cast<char *>(result.ptr_);
      new (base + sizeof(TaskT)) FutureShm();
    }
    return result;
#else
    return hipc::FullPtr<TaskT>();
#endif
  }

  template <typename TaskT, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<TaskT> NewTaskExec(size_t stack_size,
                                                    Args &&...args) {
    (void)stack_size;
    return NewTask<TaskT>(std::forward<Args>(args)...);
  }

  template <typename TaskT>
  HSHM_CROSS_FUN void DelTask(hipc::FullPtr<TaskT> task_ptr) {
#if HSHM_IS_GPU
    if (task_ptr.IsNull()) return;
    task_ptr.ptr_->~TaskT();
    if (gpu_alloc_) gpu_alloc_->Free(task_ptr.template Cast<char>());
#else
    (void)task_ptr;
#endif
  }

  template <typename T>
  HSHM_CROSS_FUN void DelObj(hipc::FullPtr<T> obj_ptr) {
#if HSHM_IS_GPU
    if (obj_ptr.IsNull()) return;
    obj_ptr.ptr_->~T();
    FreeBuffer(obj_ptr.template Cast<char>());
#else
    (void)obj_ptr;
#endif
  }

  // ================================================================
  // GPU-only allocator helpers
  // ================================================================

#if HSHM_IS_GPU_COMPILER
  HSHM_GPU_FUN HSHM_DEFAULT_ALLOC_GPU_T *FindGpuAlloc(
      const hipc::AllocatorId &id) {
    if (gpu_alloc_ && gpu_alloc_->GetId() == id) {
      return static_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
          static_cast<void *>(gpu_alloc_));
    }
    return gpu_alloc_ ? static_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
                            static_cast<void *>(gpu_alloc_))
                      : nullptr;
  }

  static HSHM_GPU_FUN __noinline__ IpcManager *GetBlockIpcManager() {
    __shared__ char s_ipc_bytes[sizeof(IpcManager)];
    return reinterpret_cast<IpcManager *>(s_ipc_bytes);
  }
#endif  // HSHM_IS_GPU_COMPILER

  // ================================================================
  // ShmPtr resolution (CROSS_FUN: called from client code)
  // ================================================================

  template <typename T>
  HSHM_CROSS_FUN hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
#if HSHM_IS_GPU
    if (shm_ptr.IsNull()) return hipc::FullPtr<T>();
    if (shm_ptr.alloc_id_ == hipc::AllocatorId::GetNull()) {
      T *raw_ptr = reinterpret_cast<T *>(shm_ptr.off_.load());
      return hipc::FullPtr<T>(raw_ptr);
    }
    auto *alloc = FindGpuAlloc(shm_ptr.alloc_id_);
    if (!alloc) return hipc::FullPtr<T>();
    return hipc::FullPtr<T>(alloc, shm_ptr);
#else
    (void)shm_ptr;
    return hipc::FullPtr<T>();
#endif
  }

  template <typename T>
  HSHM_CROSS_FUN hipc::FullPtr<T> ToFullPtr(T *ptr) {
#if HSHM_IS_GPU
    if (!ptr) return hipc::FullPtr<T>();
    hipc::ShmPtr<T> shm;
    if (gpu_alloc_) {
      shm.alloc_id_ = gpu_alloc_->GetId();
    } else {
      shm.alloc_id_.SetNull();
    }
    shm.off_ = reinterpret_cast<size_t>(ptr);
    return hipc::FullPtr<T>(shm, ptr);
#else
    if (!ptr) return hipc::FullPtr<T>();
    return hipc::FullPtr<T>(ptr);
#endif
  }

  // ================================================================
  // GPU Send/Recv — dispatches to IpcGpu2Gpu / IpcGpu2Self
  // ================================================================

#if HSHM_IS_GPU_COMPILER
  HSHM_GPU_FUN RouteResult RouteTask(const hipc::FullPtr<Task> &task_ptr) {
    RoutingMode mode = task_ptr->pool_query_.GetRoutingMode();
    if (mode == RoutingMode::ToLocalCpu) return RouteResult::Network;
    return RouteResult::Local;
  }
#endif  // HSHM_IS_GPU_COMPILER

  /** GPU-side Send: dispatches to IpcGpu2Self or IpcGpu2Gpu. */
  template <typename TaskT>
  HSHM_CROSS_FUN Future<TaskT> Send(const hipc::FullPtr<TaskT> &task_ptr) {
#if HSHM_IS_GPU
    if (is_gpu_runtime_) {
      return IpcGpu2Self::ClientSend(this, task_ptr);
    }
    return IpcGpu2Gpu::ClientSend(this, task_ptr);
#else
    (void)task_ptr;
    return Future<TaskT>();
#endif
  }

  /** GPU-side Recv: delegates to IpcGpu2Gpu::ClientRecv. */
  template <typename TaskT>
  HSHM_CROSS_FUN void Recv(Future<TaskT> &future, TaskT *task_ptr) {
#if HSHM_IS_GPU
    IpcGpu2Gpu::ClientRecv(this, future, task_ptr);
#else
    (void)future; (void)task_ptr;
#endif
  }

  // ================================================================
  // Host-side transport (CPU-side orchestrator delegates here)
  // ================================================================


  hipc::FullPtr<Task> RecvRuntime(Future<Task> &future, Container *container,
                                  u32 method_id,
                                  hshm::lbm::Transport *recv_transport);

  void SendRuntime(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                   Container *container);

  template <typename TaskT>
  bool Recv(Future<TaskT> &future, float max_sec = 0);

  // ================================================================
  // Host-side GPU infrastructure (CPU-managed)
  // ================================================================

  void RegisterGpuAllocator(const hipc::MemoryBackendId &id,
                            char *data, size_t capacity);

  /**
   * Create a GpuMalloc backend for client kernel use and return the
   * populated IpcManagerGpuInfo. Internally calls RegisterGpuAllocator
   * so the host can resolve ShmPtrs produced by the client kernel.
   *
   * @param gpu_memory_size Size of the device-memory backend (bytes)
   * @param gpu_id Target GPU device
   * @return IpcManagerGpuInfo ready to pass to a GPU kernel
   */
  IpcManagerGpuInfo CreateGpuAllocator(size_t gpu_memory_size,
                                       u32 gpu_id = 0);

  void ServerInitGpuQueues();

  u64 GetCpu2GpuQueueOffset(u32 gpu_id) const;
  u64 GetGpu2CpuQueueOffset(u32 gpu_id) const;
  u64 GetGpu2GpuQueueOffset(u32 gpu_id) const;
  u64 GetCpu2GpuBackendSize(u32 gpu_id) const;
  u64 GetGpu2CpuBackendSize(u32 gpu_id) const;
  void GetGpu2GpuIpcHandle(u32 gpu_id, char *out_bytes) const;

  bool RegisterGpuMemoryFromClient(const hipc::MemoryBackendId &backend_id,
                                   const hshm::GpuIpcMemHandle &ipc_handle,
                                   size_t data_capacity);

  bool ClientInitGpuQueues(u32 num_gpus, const u64 *cpu2gpu_offsets,
                           const u64 *gpu2cpu_offsets,
                           const u64 *gpu2gpu_offsets, const u64 *cpu2gpu_sizes,
                           const u64 *gpu2cpu_sizes, u32 queue_depth,
                           const char gpu2gpu_ipc_handles[][64]);

  IpcManagerGpuInfo GetGpuInfo(u32 gpu_id = 0) const;

  IpcManagerGpuInfo GetClientGpuInfo(u32 gpu_id = 0) const {
    return GetGpuInfo(gpu_id);
  }

  void RegisterGpuOrchestratorContainer(const PoolId &pool_id,
                                        void *gpu_container_ptr);

  // ================================================================
  // Orchestrator lifecycle management
  // ================================================================

  bool PauseGpuOrchestrator();
  void ResumeGpuOrchestrator();
  void SetGpuOrchestratorBlocks(u32 blocks, u32 threads_per_block = 0);
  void PrintGpuOrchestratorProfile();
  void RebuildGpu2GpuQueue(u32 gpu_id, u32 new_lanes);
  void RebuildInternalQueue(u32 gpu_id, u32 new_lanes);

  // ================================================================
  // Fields
  // ================================================================

  IpcManagerGpuInfo gpu_info_;
  hipc::RoundRobinAllocator *gpu_alloc_ = nullptr;
  bool is_gpu_runtime_ = false;

#if HSHM_IS_HOST
  struct GpuDeviceInfo {
    std::unique_ptr<hipc::GpuMalloc> gpu2gpu_queue_backend;
    std::unique_ptr<hipc::GpuMalloc> internal_queue_backend;
    std::unique_ptr<hipc::GpuShmMmap> gpu2cpu_queue_backend;
    std::unique_ptr<hipc::GpuShmMmap> cpu2gpu_queue_backend;
    std::unique_ptr<hipc::GpuShmMmap> gpu2cpu_copy_backend;
    std::unique_ptr<hipc::GpuShmMmap> cpu2gpu_copy_backend;
    std::unique_ptr<hipc::GpuShmMmap> gpu_orchestrator_backend;
    hipc::FullPtr<::chi::GpuTaskQueue> gpu2gpu_queue;
    hipc::FullPtr<::chi::GpuTaskQueue> internal_queue;
    hipc::FullPtr<::chi::GpuTaskQueue> gpu2cpu_queue;
    hipc::FullPtr<::chi::GpuTaskQueue> cpu2gpu_queue;
    std::unique_ptr<hipc::GpuShmMmap> client_gpu2cpu_backend;
    std::unique_ptr<hipc::GpuShmMmap> client_cpu2gpu_backend;
  };

  std::vector<GpuDeviceInfo> gpu_devices_;
  std::vector<std::unique_ptr<hipc::GpuMalloc>> client_gpu_data_backends_;
  std::vector<std::unique_ptr<hipc::GpuMalloc>> client_alloc_backends_;

  struct GpuAllocInfo {
    hipc::AllocatorId alloc_id;
    char *base = nullptr;
    size_t capacity = 0;
  };
  std::unordered_map<u64, GpuAllocInfo> gpu_alloc_map_;
  IpcManagerGpuInfo gpu_orchestrator_info_;
  void *gpu_orchestrator_ = nullptr;
#endif  // HSHM_IS_HOST

  IpcManager() = default;
  ~IpcManager() = default;
};

}  // namespace gpu
}  // namespace chi

// Transport class template implementations (need full IpcManager definition)
#include "chimaera/ipc/ipc_gpu2gpu_impl.h"
#include "chimaera/ipc/ipc_cpu2gpu_impl.h"

// ================================================================
// GPU kernel initialization macros
// ================================================================

#define CHIMAERA_GPU_INIT(gpu_info)                                           \
  chi::gpu::IpcManager *g_ipc_manager_ptr =                                   \
      chi::gpu::IpcManager::GetBlockIpcManager();                             \
  int thread_id = chi::gpu::IpcManager::GetGpuThreadId();                     \
  int num_threads = chi::gpu::IpcManager::GetGpuNumThreads();                 \
  if (thread_id == 0) {                                                       \
    g_ipc_manager_ptr->ClientInitGpu(gpu_info, num_threads);                  \
  }                                                                           \
  __syncthreads();                                                            \
  chi::gpu::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

#define CHI_CLIENT_GPU_INIT(gpu_info) CHIMAERA_GPU_INIT(gpu_info)

#define CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks)                         \
  chi::gpu::IpcManager *g_ipc_manager_ptr =                                    \
      chi::gpu::IpcManager::GetBlockIpcManager();                              \
  int thread_id = chi::gpu::IpcManager::GetGpuThreadId();                      \
  int num_threads = chi::gpu::IpcManager::GetGpuNumThreads();                  \
  if (thread_id == 0) {                                                        \
    g_ipc_manager_ptr->ClientInitGpu(gpu_info, num_threads, num_blocks);       \
  }                                                                            \
  __syncthreads();                                                             \
  chi::gpu::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

#define CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks)                   \
  chi::gpu::IpcManager *g_ipc_manager_ptr =                                    \
      chi::gpu::IpcManager::GetBlockIpcManager();                              \
  int thread_id = chi::gpu::IpcManager::GetGpuThreadId();                      \
  int num_threads = chi::gpu::IpcManager::GetGpuNumThreads();                  \
  if (thread_id == 0) {                                                        \
    g_ipc_manager_ptr->ClientInitGpu(gpu_info, num_threads, num_blocks);       \
    g_ipc_manager_ptr->is_gpu_runtime_ = true;                                 \
  }                                                                            \
  __syncthreads();                                                             \
  chi::gpu::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

#define CHIMAERA_GPU_SUBTASK_INIT(gpu_info, num_blocks)                        \
  chi::gpu::IpcManager *g_ipc_manager_ptr =                                    \
      chi::gpu::IpcManager::GetBlockIpcManager();                              \
  int thread_id = chi::gpu::IpcManager::GetGpuThreadId();                      \
  int num_threads = chi::gpu::IpcManager::GetGpuNumThreads();                  \
  __shared__ int s_partition_id;                                               \
  if (thread_id == 0) {                                                        \
    g_ipc_manager_ptr->ClientInitGpu(gpu_info, num_threads, num_blocks);       \
    g_ipc_manager_ptr->is_gpu_runtime_ = true;                                 \
    s_partition_id = g_ipc_manager_ptr->ClaimPartition();                       \
  }                                                                            \
  __syncthreads();                                                             \
  chi::gpu::IpcManager &g_ipc_manager = *g_ipc_manager_ptr;                   \
  (void)s_partition_id

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_IPC_MANAGER_H_
