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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hermes_shm/data_structures/priv/array_vector.h"
#include "chimaera/chimaera_manager.h"
#include "chimaera/corwlock.h"
#include "chimaera/scheduler/scheduler.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "hermes_shm/lightbeam/transport_factory_impl.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#endif

namespace chi {

// Forward declarations — full definitions in local_task_archives.h,
// included after CHI_IPC is defined at the bottom of this header.
template <typename BufferT> class LocalSaveTaskArchive;
template <typename BufferT> class LocalLoadTaskArchive;
using DefaultSaveArchive = LocalSaveTaskArchive<chi::priv::vector<char>>;
using DefaultLoadArchive = LocalLoadTaskArchive<chi::priv::vector<char>>;
enum class LocalMsgType : uint8_t;

/**
 * IPC transport mode for client-to-runtime communication
 */
enum class IpcMode : u32 {
  kTcp = 0,  ///< ZMQ tcp:// (default, always available)
  kIpc = 1,  ///< ZMQ ipc:// (Unix Domain Socket)
  kShm = 2,  ///< Shared memory (existing behavior)
};

/**
 * Network queue priority levels for send operations
 */
enum class NetQueuePriority : u32 {
  kSendIn = 0,   ///< Priority 0: SendIn operations (sending task inputs)
  kSendOut = 1,  ///< Priority 1: SendOut operations (sending task outputs)
  kClientSendTcp = 2,  ///< Priority 2: Client response via TCP
  kClientSendIpc = 3,  ///< Priority 3: Client response via IPC
};

/**
 * Network queue for storing Future<SendTask> objects
 * One lane with two priorities (SendIn and SendOut)
 */
using NetQueue = hipc::multi_mpsc_ring_buffer<Future<Task>, CHI_QUEUE_ALLOC_T>;

/**
 * Typedef for worker queue type to simplify usage
 */
using WorkQueue = chi::ipc::mpsc_ring_buffer<hipc::ShmPtr<TaskLane>>;

/**
 * Metadata for client <-> server communication via lightbeam
 * Compatible with lightbeam Send/RecvMetadata via duck typing
 * (has send, recv, send_bulks, recv_bulks fields)
 */
struct ClientTaskMeta {
  std::vector<hshm::lbm::Bulk> send;
  std::vector<hshm::lbm::Bulk> recv;
  size_t send_bulks = 0;
  size_t recv_bulks = 0;
  std::vector<char> wire_data;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(send, recv, send_bulks, recv_bulks, wire_data);
  }
};

/**
 * Information about a per-process shared memory segment
 * Used for registering client memory with the runtime
 */
struct ClientShmInfo {
  std::string shm_name;        // Shared memory name (chimaera_{pid}_{count})
  pid_t owner_pid;             // PID of the owning process
  u32 shm_index;               // Index within the owner's shm segments
  size_t size;                 // Size of the shared memory segment
  hipc::AllocatorId alloc_id;  // Allocator ID for this segment

  ClientShmInfo() : owner_pid(0), shm_index(0), size(0) {}

  ClientShmInfo(const std::string &name, pid_t pid, u32 idx, size_t sz,
                const hipc::AllocatorId &id)
      : shm_name(name),
        owner_pid(pid),
        shm_index(idx),
        size(sz),
        alloc_id(id) {}

  /**
   * Serialization support for cereal
   */
  template <class Archive>
  void serialize(Archive &ar) {
    ar(shm_name, owner_pid, shm_index, size, alloc_id.major_, alloc_id.minor_);
  }
};

/**
 * GPU data transfer object for IpcManager initialization.
 *
 * Memory topology:
 *   - backend: primary scratch / GPU→GPU alloc backend.
 *       For orchestrator: GpuShmMmap scratch (split per block by INIT macro).
 *       For client kernels: client's GpuMalloc (device memory for GPU→GPU
 *       FutureShm).
 *   - gpu2gpu_queue: TaskQueue in device memory (GpuMalloc, orchestrator polls)
 *   - cpu2gpu_queue: TaskQueue in pinned host (GpuShmMmap, orchestrator polls)
 *   - gpu2cpu_queue: TaskQueue in pinned host (GpuShmMmap, CPU worker polls)
 *   - gpu2cpu_backend: pinned-host backend for GPU→CPU FutureShm allocation
 *       (GPU client allocates here when routing ToLocalCpu)
 */
struct IpcManagerGpuInfo {
  /** Primary backend: orchestrator scratch or client GPU→GPU alloc memory */
  hipc::MemoryBackend backend;

  /** GPU→GPU queue in device memory (orchestrator polls, client pushes) */
  TaskQueue *gpu2gpu_queue = nullptr;
  /** Base of gpu2gpu queue backend for device-side ShmPtr resolution */
  char *gpu2gpu_queue_base = nullptr;

  /** Internal subtask queue in device memory (orchestrator polls, runtime
   * pushes) */
  TaskQueue *internal_queue = nullptr;
  /** Base of internal queue backend for device-side ShmPtr resolution */
  char *internal_queue_base = nullptr;

  /** CPU→GPU queue in pinned host (orchestrator polls, CPU pushes) */
  TaskQueue *cpu2gpu_queue = nullptr;
  /** Base of cpu2gpu copy-space backend for GPU-side ShmPtr→ptr conversion */
  char *cpu2gpu_queue_base = nullptr;

  /** GPU→CPU queue in pinned host (CPU worker polls, GPU pushes) */
  TaskQueue *gpu2cpu_queue = nullptr;

  /**
   * Pinned-host backend for GPU→CPU FutureShm + copy_space allocation.
   * GPU client allocates from this when routing ToLocalCpu.
   * Alloc table set up by ClientInitGpu alongside primary backend.
   */
  hipc::MemoryBackend gpu2cpu_backend;

  u32 gpu_queue_depth = 16;

  /** Number of lanes in the gpu2gpu TaskQueue (one per orchestrator warp) */
  u32 gpu2gpu_num_lanes = 1;

  /** Number of lanes in the internal subtask queue */
  u32 internal_num_lanes = 1;

  /** When true, skip GPU heap re-initialization (heap allocators persist across
   *  pause/resume cycles — re-init would destroy existing container
   * allocations). */
  bool skip_heap_init = false;

  /** When true, skip scratch allocator re-initialization (backend + gpu2cpu).
   *  Set to true for non-block-0 blocks (they wait for block 0 to init).
   *  Set to false on resume when warp count changes so scratch gets
   *  re-partitioned for the new warp count. */
  bool skip_scratch_init = false;

  /** GPU-accessible allocator table for resolving ShmPtrs on the GPU.
   *  Each entry maps an AllocatorId to a base pointer.
   *  Populated by RegisterGpuAllocator on the host side.
   *  Used by ToFullPtr on the GPU side when FindGpuAlloc doesn't match. */
  static constexpr u32 kMaxGpuAllocs = 8;
  struct GpuAllocEntry {
    hipc::AllocatorId alloc_id;
    char *base = nullptr;
    HSHM_CROSS_FUN GpuAllocEntry() = default;
  };
  GpuAllocEntry gpu_allocs[kMaxGpuAllocs];
  u32 num_gpu_allocs = 0;

  /** Number of warps in the GPU grid */
  u32 num_warps = 1;
  /** Number of warp groups (sqrt partitioning) */
  u32 num_warp_groups = 1;

  /** Per-warp-group load counters (device memory, [num_warp_groups]) */
  static constexpr u32 kMaxWarpGroups = 32;
  hipc::atomic<u32> *warp_group_load = nullptr;
  /** Per-warp load counters (device memory, [num_warps]) */
  hipc::atomic<u32> *warp_load = nullptr;

  /** Cross-warp task queue (device memory) */
  TaskQueue *warp_group_queue = nullptr;
  /** Base of warp group queue backend for ShmPtr resolution */
  char *warp_group_queue_base = nullptr;
  /** Number of lanes in the warp group queue */
  u32 warp_group_num_lanes = 1;

  /** Scratch allocator generation (from WorkOrchestratorControl) */
  u32 scratch_gen = 0;

  /** Per-warp private BuddyAllocator region size (allocated from shared) */
  size_t priv_region_size = 4096;

  HSHM_CROSS_FUN IpcManagerGpuInfo() = default;

  /** Convenience constructor: backend + gpu2gpu queue */
  HSHM_CROSS_FUN IpcManagerGpuInfo(const hipc::MemoryBackend &be,
                                   TaskQueue *g2g_queue)
      : backend(be), gpu2gpu_queue(g2g_queue) {}
};

/** Backward compatibility alias */
using IpcManagerGpu = IpcManagerGpuInfo;

/**
 * IPC Manager singleton for inter-process communication
 *
 * Manages ZeroMQ server using lightbeam from HSHM, three memory segments,
 * and priority queues for task processing.
 * Uses HSHM global cross pointer variable singleton pattern.
 */
class IpcManager {
 public:
  /** Per-warp cached state for GPU serialization and allocation.
   *  buffer_ is the shared serialize/deserialize buffer — allocated once,
   *  reused for every task. Archives bind to it via reference at construction. */
  struct WarpIpcManager {
    CHI_PRIV_ALLOC_T priv_alloc_;   /**< Per-warp private allocator (value) */
    hshm::priv::wrap_vector buffer_;  /**< Wraps copy_space task data region */
    WrapSaveArchive save_ar_;
    WrapLoadArchive load_ar_;
    static constexpr size_t kCopySpaceSize = 4096;
    static constexpr size_t kHeaderSize = sizeof(PreallocHeader);

    /**
     * Construct WarpIpcManager with a private BuddyAllocator.
     * @param priv_region Pre-allocated region from the shared PartitionedAllocator
     * @param priv_region_size Size of the private region in bytes
     */
    HSHM_CROSS_FUN WarpIpcManager(char *priv_region, size_t priv_region_size)
        : priv_alloc_(),
          buffer_(),
          save_ar_(LocalMsgType::kSerializeOut, &priv_alloc_, buffer_),
          load_ar_(&priv_alloc_, buffer_) {
#if HSHM_IS_GPU
      hipc::MemoryBackend backend;
      backend.data_ = priv_region;
      backend.data_capacity_ = priv_region_size;
      priv_alloc_.shm_init(backend, priv_region_size, /*shifted=*/true);
#else
      (void)priv_region; (void)priv_region_size;
#endif
    }

    /** Set up buffer_ to point at copy_space task data region */
    HSHM_CROSS_FUN void BindCopySpace(char *copy_space) {
      hipc::FullPtr<char> fp;
      fp.ptr_ = copy_space + kHeaderSize;
      fp.shm_.alloc_id_.SetNull();
      fp.shm_.off_ = reinterpret_cast<size_t>(fp.ptr_);
      buffer_.set(fp, kCopySpaceSize - kHeaderSize);
    }
  };
 public:
  /**
   * Initialize client components
   * @return true if initialization successful, false otherwise
   */
  bool ClientInit();

  /**
   * Initialize server/runtime components
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();

  /**
   * Client finalize - does nothing for now
   */
  void ClientFinalize();

  /**
   * Server finalize - cleanup all IPC resources
   */
  void ServerFinalize();

  /**
   * Initialize GPU client components with per-thread allocators.
   *
   * Sets up two alloc tables:
   *   gpu_alloc_table_   — primary table from gpu_info.backend
   *                        (device memory for GPU→GPU tasks, or orchestrator
   *                         scratch)
   *   gpu2cpu_alloc_table_ — secondary table from gpu_info.gpu2cpu_backend
   *                          (pinned host for GPU→CPU FutureShm allocation)
   *
   * Each table layout: [ptr_table (N ptrs)] [BuddyAlloc_0] ... [BuddyAlloc_N-1]
   * Thread 0 of each block initializes; all threads sync before using.
   *
   * @param gpu_info IpcManagerGpuInfo with backends and queue pointers
   * @param num_threads Total number of GPU threads in this block
   */
  HSHM_CROSS_FUN
  void ClientInitGpu(IpcManagerGpuInfo &gpu_info, int num_threads,
                     int num_blocks = 1) {
#if !HSHM_IS_HOST
    // Clear stale is_gpu_runtime_ from previous kernel's __shared__ memory.
    // CHIMAERA_GPU_ORCHESTRATOR_INIT overrides this to true after this call.
    is_gpu_runtime_ = false;
#endif
    // Store queue pointers
    gpu2gpu_queue_ = gpu_info.gpu2gpu_queue;
    gpu2gpu_queue_base_ = gpu_info.gpu2gpu_queue_base;
    gpu2gpu_num_lanes_ = gpu_info.gpu2gpu_num_lanes;
    internal_queue_ = gpu_info.internal_queue;
    internal_queue_base_ = gpu_info.internal_queue_base;
    internal_num_lanes_ = gpu_info.internal_num_lanes;
    cpu2gpu_queue_ = gpu_info.cpu2gpu_queue;
    cpu2gpu_queue_base_ = gpu_info.cpu2gpu_queue_base;
    gpu2cpu_queue_ = gpu_info.gpu2cpu_queue;

    // Set up primary allocator (GPU→GPU device memory or orchestrator scratch)
    // Partition by warps so each warp gets its own BuddyAllocator.
    // Use kMaxCachedWarps as minimum to support future resizes (pause/resume
    // reuses the allocator with skip_scratch_init=true).
    gpu_backend_ = gpu_info.backend;
    gpu_backend_initialized_ = true;
    u32 num_warps = num_threads / 32;
    if (num_warps < kMaxCachedWarps) num_warps = kMaxCachedWarps;
    if (gpu_backend_.data_ != nullptr) {
      if (gpu_info.skip_scratch_init) {
        gpu_alloc_ = reinterpret_cast<hipc::PartitionedAllocator *>(gpu_backend_.data_);
        gpu_alloc_->WaitReady();
      } else {
        InitHeapAllocator(gpu_backend_, num_warps, &gpu_alloc_);
      }
    }

    // Set up GPU→CPU allocator (pinned host, for ToLocalCpu FutureShm)
    if (gpu_info.gpu2cpu_backend.data_ != nullptr) {
      gpu2cpu_backend_ = gpu_info.gpu2cpu_backend;
      if (gpu_info.skip_scratch_init) {
        gpu2cpu_alloc_ =
            reinterpret_cast<hipc::PartitionedAllocator *>(gpu2cpu_backend_.data_);
        gpu2cpu_alloc_->WaitReady();
      } else {
        InitHeapAllocator(gpu2cpu_backend_, num_warps, &gpu2cpu_alloc_);
      }
    }

    // Copy GPU allocator registrations (for ShmPtr resolution via ToFullPtr)
    gpu_num_allocs_ = gpu_info.num_gpu_allocs;
    for (u32 i = 0; i < gpu_info.num_gpu_allocs; ++i) {
      gpu_allocs_[i] = gpu_info.gpu_allocs[i];
    }

    // Scratch generation — containers compare to detect stale metadata
    scratch_gen_ = gpu_info.scratch_gen;

    // Per-warp private region size (configurable from GPU info)
    priv_region_size_ = gpu_info.priv_region_size;

    // Zero the per-warp caches; populated lazily by GetWarpManager()
    for (u32 i = 0; i < kMaxCachedWarps; ++i) {
      warp_mgrs_[i] = nullptr;
    }
  }

  /**
   * Initialize a single PartitionedAllocator from a MemoryBackend.
   * The PartitionedAllocator manages per-warp BuddyAllocator partitions internally,
   * eliminating the need for a separate pointer table.
   *
   * @param backend GpuMalloc backend (device memory)
   * @param num_threads Number of GPU warps (used as max_threads for
   * partitioning)
   * @param alloc_out Output: pointer to the PartitionedAllocator
   */
  HSHM_CROSS_FUN
  void InitHeapAllocator(const hipc::MemoryBackend &backend, int num_threads,
                         hipc::PartitionedAllocator **alloc_out) {
    char *base = backend.data_;
    size_t data_capacity = backend.data_capacity_;

    auto *alloc = reinterpret_cast<hipc::PartitionedAllocator *>(base);
    new (alloc) hipc::PartitionedAllocator();

    hipc::MemoryBackend sub_backend;
    sub_backend.data_ = base;
    sub_backend.data_capacity_ = data_capacity;
    sub_backend.id_ = backend.id_;

    // Calculate per-thread partition size (leave room for allocator header)
    size_t overhead = sizeof(hipc::PartitionedAllocator);
    size_t thread_unit = (data_capacity - overhead) / num_threads;

    alloc->shm_init(sub_backend, 0, num_threads, thread_unit);

    // Signal that the allocator layout is ready (grid-level sync).
    // Each thread/block lazily initializes its own partition on first use.
    alloc->MarkReady();

    *alloc_out = alloc;
  }

  /**
   * Initialize a per-block PrivateBuddyAllocator directly over a memory
   * backend partition.  Each block gets (data_capacity / num_blocks) bytes.
   * Block 0 initializes its slice; other blocks reattach via
   * GetBlockPrivAllocator().
   *
   * @param backend GpuMalloc backend for private allocations
   * @param num_blocks Number of blocks sharing this backend
   */
  /**
   * Return the private allocator for CHI_PRIV_ALLOC.
   * On GPU: returns the per-warp PrivateBuddyAllocator from PartitionedAllocator.
   * On host: returns nullptr (host uses HSHM_MALLOC via CHI_PRIV_ALLOC).
   */
  /** Get the per-warp manager (lazy: allocate from shared PartitionedAllocator).
   *  Allocates WarpIpcManager + a private region, then constructs a private
   *  BuddyAllocator over the private region for warp-local allocations. */
  HSHM_CROSS_FUN WarpIpcManager *GetWarpManager() {
#if HSHM_IS_GPU
    u32 local_warp = threadIdx.x / 32;
    if (local_warp < kMaxCachedWarps && warp_mgrs_[local_warp]) {
      return warp_mgrs_[local_warp];
    }
    if (gpu_alloc_) {
      auto *shared_alloc = gpu_alloc_->GetWarpAllocator();
      if (shared_alloc) {
        // Allocate WarpIpcManager from shared allocator
        auto p = shared_alloc->template AllocateObjs<WarpIpcManager>(1);
        if (p.IsNull()) return nullptr;
        // Allocate private region from shared allocator
        auto priv = shared_alloc->template AllocateObjs<char>(priv_region_size_);
        if (priv.IsNull()) {
          shared_alloc->Free(p);
          return nullptr;
        }
        auto *mgr = new (p.ptr_) WarpIpcManager(priv.ptr_, priv_region_size_);
        if (local_warp < kMaxCachedWarps) warp_mgrs_[local_warp] = mgr;
        return mgr;
      }
    }
    return nullptr;
#else
    return nullptr;
#endif
  }

  HSHM_CROSS_FUN hipc::PrivateBuddyAllocator *GetPrivAlloc() {
#if HSHM_IS_GPU
    auto *mgr = GetWarpManager();
    return mgr ? &mgr->priv_alloc_ : nullptr;
#else
    return nullptr;
#endif
  }

  /** Get per-warp cached save archive */
  HSHM_CROSS_FUN WrapSaveArchive *GetSaveArchive() {
#if HSHM_IS_GPU
    auto *mgr = GetWarpManager();
    return mgr ? &mgr->save_ar_ : nullptr;
#else
    return nullptr;
#endif
  }

  /** Get per-warp cached load archive */
  HSHM_CROSS_FUN WrapLoadArchive *GetLoadArchive() {
#if HSHM_IS_GPU
    auto *mgr = GetWarpManager();
    return mgr ? &mgr->load_ar_ : nullptr;
#else
    return nullptr;
#endif
  }

  /** Get cached FutureShm + copy_space (lazy-allocated once per warp).
   *  Also binds the wrap_vector buffer_ to the copy_space task data region. */
  /**
   * Get the co-located FutureShm for a task allocated by NewTask (GPU).
   * The FutureShm is placed right after the task in the bulk allocation.
   */
  template <typename TaskT>
  HSHM_CROSS_FUN static FutureShm *GetTaskFutureShm(TaskT *task) {
    return reinterpret_cast<FutureShm *>(
        reinterpret_cast<char *>(task) + sizeof(TaskT));
  }

  /**
   * Allocate and construct an object with the given allocator scope.
   * kPrivate: use the warp-local BuddyAllocator (CHI_PRIV_ALLOC)
   * kShared:  use the PartitionedAllocator (CHI_PRIV_SHARED_ALLOC)
   *           which dispatches to the calling warp's partition via GetAutoTid()
   *
   * DelObj is not needed — just use CHI_PRIV_ALLOC->DelObj() which routes
   * frees to the correct partition by address arithmetic.
   */
  template <typename T, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<T> NewObj(AllocScope scope, Args&&... args) {
#if HSHM_IS_GPU
    if (scope == AllocScope::kShared) {
      return gpu_alloc_->template NewObj<T>(std::forward<Args>(args)...);
    }
    return GetPrivAlloc()->template NewObj<T>(std::forward<Args>(args)...);
#else
    (void)scope;
    return HSHM_MALLOC->template NewObj<T>(std::forward<Args>(args)...);
#endif
  }

  /**
   * Returns the per-thread task allocator (CHI_TASK_ALLOC_T*).
   * CPU: main_allocator_ (MultiProcessAllocator)
   * GPU: PartitionedAllocator (gpu_alloc_) shared across threads in partition
   * Used by CHI_PRIV_ALLOC on GPU for chi::priv string/vector operations.
   */
  HSHM_CROSS_FUN CHI_TASK_ALLOC_T *GetMainAllocator() {
#if HSHM_IS_HOST
    return main_allocator_;
#else
    return static_cast<CHI_TASK_ALLOC_T *>(static_cast<void *>(gpu_alloc_));
#endif
  }

#if HSHM_IS_HOST
  /**
   * Pack current GPU orchestrator info into an IpcManagerGpuInfo struct.
   * @return IpcManagerGpuInfo for passing to orchestrator kernel
   */
  IpcManagerGpuInfo GetIpcManagerGpuInfo() { return gpu_orchestrator_info_; }

  /** Backward-compatible alias */
  IpcManagerGpuInfo GetIpcManagerGpu() { return GetIpcManagerGpuInfo(); }
#endif

  /**
   * Get linear GPU thread ID for 1D/2D/3D blocks
   * @return Linear thread index within the block
   */
#if HSHM_IS_GPU_COMPILER
  static HSHM_GPU_FUN inline int GetGpuThreadId() {
    return threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
  }
  static HSHM_GPU_FUN inline int GetGpuNumThreads() {
    return blockDim.x * blockDim.y * blockDim.z;
  }

  /** Get the global warp ID within the grid */
  static HSHM_GPU_FUN inline u32 GetWarpId() {
    return (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  }

  /** Get the lane ID (0-31) within the warp */
  static HSHM_GPU_FUN inline u32 GetLaneId() { return threadIdx.x % 32; }

  /** Whether this thread is the warp scheduler (lane 0) */
  static HSHM_GPU_FUN inline bool IsWarpScheduler() { return GetLaneId() == 0; }

  /** Get total number of warps in the grid */
  static HSHM_GPU_FUN inline u32 GetNumWarps() {
    return (gridDim.x * blockDim.x) / 32;
  }
  /** Compute number of warp groups (sqrt partitioning) */
  static HSHM_GPU_FUN inline u32 GetNumWarpGroups(u32 num_warps) {
    u32 s = 1;
    while (s * s < num_warps) ++s;
    return s;
  }
  /** Get the warp group ID for this warp */
  static HSHM_GPU_FUN inline u32 GetWarpGroupId(u32 num_warp_groups) {
    return GetWarpId() / num_warp_groups;
  }
  /** Get the minor ID within a warp group */
  static HSHM_GPU_FUN inline u32 GetWarpMinorId(u32 num_warp_groups) {
    return GetWarpId() % num_warp_groups;
  }
  /** Check if this warp is the leader of its warp group */
  static HSHM_GPU_FUN inline bool IsWarpGroupLeader(u32 num_warp_groups) {
    return GetWarpMinorId(num_warp_groups) == 0;
  }
#else
  /** Host-side stubs for warp utilities (always lane 0 / warp 0) */
  static inline u32 GetWarpId() { return 0; }
  static inline u32 GetLaneId() { return 0; }
  static inline bool IsWarpScheduler() { return true; }
#endif

  /**
   * Create a new task in private memory
   * Host: uses standard new
   * GPU: uses AllocateBuffer from shared memory
   * @param args Constructor arguments for the task
   * @return FullPtr wrapping the task with null allocator
   */
  template <typename TaskT, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<TaskT> NewTask(Args &&...args) {
#if HSHM_IS_HOST
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    hipc::FullPtr<TaskT> result(ptr);
    return result;
#else
    if (!IsWarpScheduler()) return hipc::FullPtr<TaskT>();
    auto *priv = GetPrivAlloc();
    if (!priv) return hipc::FullPtr<TaskT>();
    // Bulk allocation: Task + FutureShm + copy_space in one call.
    // Each task gets its own FutureShm so multiple in-flight tasks
    // from the same warp don't corrupt each other's completion state.
    size_t total = sizeof(TaskT) + sizeof(FutureShm)
                   + WarpIpcManager::kCopySpaceSize;
    auto fp = priv->template AllocateObjs<char>(total);
    if (fp.IsNull()) return hipc::FullPtr<TaskT>();
    auto *task = new (fp.ptr_) TaskT(std::forward<Args>(args)...);
    // Construct the co-located FutureShm (right after the task)
    new (fp.ptr_ + sizeof(TaskT)) FutureShm();
    return fp.template Cast<TaskT>();
#endif
  }

  /**
   * Delete a task
   * Host: destructor + operator delete
   * GPU: destructor + private allocator free
   */
  template <typename TaskT>
  HSHM_CROSS_FUN void DelTask(hipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) return;
#if HSHM_IS_HOST
    task_ptr.ptr_->~TaskT();
    void *raw = static_cast<void *>(task_ptr.ptr_);
    ::operator delete(raw);
#else
    task_ptr.ptr_->~TaskT();
    auto *priv = GetPrivAlloc();
    if (priv) priv->Free(task_ptr.template Cast<char>());
#endif
  }

  /**
   * Delete a heap-allocated object from private memory.
   * Same as DelTask but for arbitrary (non-Task) types.
   * Host: destructor + operator delete.
   * GPU: destructor + FreeBuffer.
   * @param obj_ptr FullPtr to object to delete
   */
  template <typename T>
  HSHM_CROSS_FUN void DelObj(hipc::FullPtr<T> obj_ptr) {
    if (obj_ptr.IsNull()) return;
#if HSHM_IS_HOST
    obj_ptr.ptr_->~T();
    void *raw = static_cast<void *>(obj_ptr.ptr_);
    ::operator delete(raw);
#else
    obj_ptr.ptr_->~T();
    FreeBuffer(obj_ptr.template Cast<char>());
#endif
  }

  /**
   * Allocate buffer in appropriate memory segment
   * Client uses cdata segment, runtime uses rdata segment
   * Yields until buffer is allocated successfully
   * @param size Size in bytes to allocate
   * @return FullPtr<char> to allocated memory
   */
  HSHM_CROSS_FUN FullPtr<char> AllocateBuffer(size_t size);

  /**
   * Push a bump arena on CHI_PRIV_ALLOC for fast allocation.
   * All subsequent CHI_PRIV_ALLOC allocations (serialization scratch,
   * chi::priv vectors/strings) will bump-allocate from this arena
   * until it is popped (RAII).
   *
   * @param size Arena size in bytes
   * @return Arena RAII handle
   */
  HSHM_CROSS_FUN hipc::Arena<hipc::PrivateBuddyAllocator> PushPrivArena(size_t size);

  /**
   * Push a bump arena on the GPU primary allocator for fast allocation.
   * All subsequent AllocateBuffer/NewObj/NewTask calls will bump-allocate
   * from this arena until it is popped (RAII).
   *
   * @param size Arena size in bytes
   * @return Arena RAII handle
   */
  HSHM_CROSS_FUN hipc::Arena<hipc::PartitionedAllocator> PushArena(size_t size);

  /**
   * Allocate GPU device data from the client's GpuMalloc backend.
   * On GPU: uses per-thread BuddyAllocator (same as AllocateBuffer).
   * On host: uses AllocateGpuBuffer from the GPU queue backend.
   * The resulting ShmPtrs are resolvable server-side via gpu_alloc_map_.
   *
   * @param size Number of bytes to allocate
   * @param gpu_id GPU device ID (host path only)
   * @return FullPtr<char> to allocated GPU memory
   */
#if HSHM_IS_GPU_COMPILER
  HSHM_GPU_FUN hipc::FullPtr<char> AllocateDeviceData(size_t size) {
    // GPU PATH: allocate from scratch (shared data)
    if (gpu_backend_initialized_ && gpu_alloc_ != nullptr) {
      return gpu_alloc_->AllocateObjs<char>(size);
    }
    return hipc::FullPtr<char>::GetNull();
  }
#endif
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  hipc::FullPtr<char> AllocateDeviceData(size_t size, u32 gpu_id = 0) {
    return AllocateGpuBuffer(size, gpu_id);
  }
#endif

  /**
   * Free buffer from appropriate memory segment
   * Host: uses allocator's Free method
   * GPU: uses ArenaAllocator's Free method
   * @param buffer_ptr FullPtr to buffer to free
   */
  HSHM_CROSS_FUN void FreeBuffer(FullPtr<char> buffer_ptr);

  /**
   * Free buffer from appropriate memory segment (hipc::ShmPtr<> overload)
   * Converts hipc::ShmPtr<> to FullPtr<char> and calls the main FreeBuffer
   * @param buffer_ptr hipc::ShmPtr<> to buffer to free
   */
  HSHM_CROSS_FUN void FreeBuffer(hipc::ShmPtr<char> buffer_ptr) {
    if (buffer_ptr.IsNull()) {
      return;
    }
#if HSHM_IS_GPU
#endif
    // Convert hipc::ShmPtr<> to FullPtr<char> and call main FreeBuffer
    hipc::FullPtr<char> full_ptr(ToFullPtr<char>(buffer_ptr));
    FreeBuffer(full_ptr);
  }

  /**
   * Allocate and construct an object using placement new
   * Combines AllocateBuffer and placement new construction
   * @tparam T Type of object to construct
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   * @return FullPtr<T> to constructed object
   */
  template <typename T, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<T> NewObj(Args &&...args) {
#if HSHM_IS_HOST
    // Host path: allocate from bulk buffer
    hipc::FullPtr<char> buffer = AllocateBuffer(sizeof(T));
    if (buffer.IsNull()) {
      return hipc::FullPtr<T>();
    }
    T *obj = new (buffer.ptr_) T(std::forward<Args>(args)...);
    return buffer.Cast<T>();
#else
    // GPU path: allocate from data allocator (PartitionedAllocator)
    hipc::FullPtr<char> buffer = AllocateBuffer(sizeof(T));
    if (buffer.IsNull()) return hipc::FullPtr<T>();
    T *obj = new (buffer.ptr_) T(std::forward<Args>(args)...);
    return buffer.Cast<T>();
#endif
  }

  /**
   * Create Future by copying/serializing task (GPU-compatible)
   * Always serializes the task into FutureShm's copy_space
   * Used by clients and GPU kernels
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to serialize into Future
   * @return Future<TaskT> with serialized task data
   */
  template <typename TaskT>
  HSHM_CROSS_FUN Future<TaskT> MakeCopyFuture(hipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Allocate FutureShm with copy_space (lightbeam handles the data transfer)
    size_t copy_space_size = task_ptr->GetCopySpaceSize();
    if (copy_space_size == 0) copy_space_size = KILOBYTES(4);
    size_t alloc_size = sizeof(FutureShm) + copy_space_size;
    hipc::FullPtr<char> buffer = AllocateBuffer(alloc_size);
    if (buffer.IsNull()) {
      return Future<TaskT>();
    }

    // Construct FutureShm in-place
    FutureShm *future_shm_ptr = new (buffer.ptr_) FutureShm();
    future_shm_ptr->pool_id_ = task_ptr->pool_id_;
    future_shm_ptr->method_id_ = task_ptr->method_;
    future_shm_ptr->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    future_shm_ptr->client_task_vaddr_ =
        reinterpret_cast<uintptr_t>(task_ptr.ptr_);
    future_shm_ptr->input_.copy_space_size_ = copy_space_size;
    future_shm_ptr->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

    // Create and return Future
    hipc::ShmPtr<FutureShm> future_shm_shmptr =
        buffer.shm_.template Cast<FutureShm>();
    return Future<TaskT>(future_shm_shmptr, task_ptr);
  }

  /**
   * GPU-side send: serialize task and enqueue it for processing.
   *
   * Routes based on the task's RoutingMode:
   *   - Local / LocalGpuBcast / ToLocalGpu → GPU→GPU path
   *       FutureShm allocated from gpu_alloc_table_ (device memory)
   *       Pushed to gpu2gpu_queue_ (device memory)
   *   - ToLocalCpu (and all other modes) → GPU→CPU path
   *       FutureShm allocated from gpu2cpu_alloc_table_ (pinned host)
   *       Pushed to gpu2cpu_queue_ (pinned host)
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to send
   * @return Future<TaskT> for polling completion
   */
#if HSHM_IS_GPU_COMPILER
  template <typename TaskT>
  HSHM_GPU_FUN Future<TaskT> SendGpu(const hipc::FullPtr<TaskT> &task_ptr) {
    // Warp-cooperative: all lanes enter, lane 0 handles control plane,
    // all lanes participate in serialization via warp_converged write_binary.
    u32 lane = GetLaneId();

    // === Phase 1: Lane 0 setup ===
    unsigned long long mgr_ull = 0, task_ull = 0;
    FutureShm *fshm = nullptr;
    hipc::ShmPtr<FutureShm> fshmptr;
    bool to_cpu = false;
    TaskQueue *queue = nullptr;
    size_t copy_space_size = WarpIpcManager::kCopySpaceSize;

    if (lane == 0) {
      if (task_ptr.IsNull()) {
        mgr_ull = 0;
      } else {
        RoutingMode mode = task_ptr->pool_query_.GetRoutingMode();
        to_cpu = (mode == RoutingMode::ToLocalCpu);

        if (to_cpu) {
          if (!gpu2cpu_alloc_ || !gpu2cpu_queue_) { mgr_ull = 0; goto send_bcast; }
          queue = gpu2cpu_queue_;
        } else {
          if (!gpu_alloc_ || !gpu2gpu_queue_) { mgr_ull = 0; goto send_bcast; }
          queue = gpu2gpu_queue_;
        }

        // FutureShm is co-located right after the task (bulk-allocated by NewTask)
        fshm = reinterpret_cast<FutureShm *>(
            reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
        fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
        fshm->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

        fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
        fshmptr.off_ = reinterpret_cast<size_t>(fshm);

        // Prepare WarpIpcManager: bind buffer_ to this task's copy_space
        auto *mgr = GetWarpManager();
        mgr->BindCopySpace(fshm->copy_space);
        mgr->buffer_.clear();
        mgr->save_ar_.Reset(LocalMsgType::kSerializeIn);
        mgr->save_ar_.SetWarpConverged(true);

        mgr_ull = reinterpret_cast<unsigned long long>(mgr);
        task_ull = reinterpret_cast<unsigned long long>(task_ptr.ptr_);
      }
    }
send_bcast:

    // === Phase 2: Warp-parallel SerializeIn directly into copy_space ===
    // Task data is written via wrap_vector → copy_space + kHeaderSize
    mgr_ull = hipc::shfl_sync_u64(0xFFFFFFFF, mgr_ull, 0);
    task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);

    if (mgr_ull != 0 && task_ull != 0) {
      auto *mgr = reinterpret_cast<WarpIpcManager *>(mgr_ull);
      auto *task = reinterpret_cast<TaskT *>(task_ull);
      task->SerializeIn(mgr->save_ar_);
    }
    __syncwarp();

    // === Phase 3: Write header + queue push ===
    unsigned long long cs_ull = 0;
    unsigned long long si_ull = 0;
    int is_to_cpu = 0;

    if (lane == 0 && mgr_ull != 0) {
      auto *mgr = reinterpret_cast<WarpIpcManager *>(mgr_ull);
      mgr->save_ar_.SetWarpConverged(false);
      is_to_cpu = to_cpu ? 1 : 0;

      if (!to_cpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_DEVICE_SCOPE);
        fshm->input_.copy_space_size_.store(copy_space_size);
        fshm->output_.copy_space_size_.store(copy_space_size);
      } else {
        fshm->input_.copy_space_size_.store_system(copy_space_size);
        fshm->output_.copy_space_size_.store_system(copy_space_size);
      }

      cs_ull = reinterpret_cast<unsigned long long>(fshm->copy_space);
      si_ull = reinterpret_cast<unsigned long long>(&fshm->input_);
    }

    cs_ull = hipc::shfl_sync_u64(0xFFFFFFFF, cs_ull, 0);
    si_ull = hipc::shfl_sync_u64(0xFFFFFFFF, si_ull, 0);
    is_to_cpu = __shfl_sync(0xFFFFFFFF, is_to_cpu, 0);

    Future<TaskT> future;
    if (mgr_ull != 0 && cs_ull != 0 && !is_to_cpu) {
      // GPU→GPU preallocated path: task data already in copy_space,
      // just write the PreallocHeader and mark ready.
      auto *mgr = reinterpret_cast<WarpIpcManager *>(mgr_ull);
      PreallocHeader hdr;
      hdr.msg_type = LocalMsgType::kSerializeIn;
      hdr.data_size = static_cast<u32>(mgr->buffer_.size());

      hshm::lbm::LbmContext ctx;
      ctx.copy_space = reinterpret_cast<char*>(cs_ull);
      ctx.shm_info_ = reinterpret_cast<hshm::lbm::ShmTransferInfo*>(si_ull);
      hshm::lbm::ShmTransport::SendDevicePrealloc(
          reinterpret_cast<const char*>(&hdr), sizeof(hdr),
          hdr.data_size, ctx);
    }

    // Lane 0: queue push + Future creation
    if (lane == 0 && mgr_ull != 0) {
      u32 queue_lane_id = 0;
      if (!to_cpu && gpu2gpu_num_lanes_ > 1) {
        queue_lane_id = GetWarpId() % gpu2gpu_num_lanes_;
      }
      auto &qlane = queue->GetLane(queue_lane_id, 0);
      future = Future<TaskT>(fshmptr, task_ptr);
      Future<Task> task_future(future.GetFutureShmPtr());

      if (to_cpu) {
        // CPU path: use old SPSC Send path
        auto *mgr = reinterpret_cast<WarpIpcManager *>(mgr_ull);
        hshm::lbm::LbmContext cpu_ctx;
        cpu_ctx.copy_space = fshm->copy_space;
        cpu_ctx.shm_info_ = &fshm->input_;
        hshm::lbm::ShmTransport::Send(mgr->save_ar_, cpu_ctx);
        qlane.PushSystem(task_future);
      } else {
        hipc::threadfence();
        qlane.Push(task_future);
      }
    }
    return future;
  }

  /**
   * Send a task to another GPU worker WITHOUT serialization.
   *
   * Used for intra-GPU subtask dispatch where the task is already in
   * GPU-accessible memory. The worker uses DispatchTaskDirect to run
   * the task in-place; results are read directly from the task pointer.
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to send (must be in GPU memory)
   * @return Future<TaskT> for co_await / polling
   */
  template <typename TaskT>
  HSHM_CROSS_FUN Future<TaskT> SendGpuDirect(
      const hipc::FullPtr<TaskT> &task_ptr) {
    // Use internal queue if available (orchestrator context), else gpu2gpu
    TaskQueue *queue = internal_queue_ ? internal_queue_ : gpu2gpu_queue_;
    if (!queue) return Future<TaskT>();

    // FutureShm is co-located right after the task (bulk-allocated by NewTask)
    FutureShm *fshm = reinterpret_cast<FutureShm *>(
        reinterpret_cast<char *>(task_ptr.ptr_) + sizeof(TaskT));
    fshm->Reset(task_ptr->pool_id_, task_ptr->method_);
    fshm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    // Store task UVA so DispatchTaskDirect can reconstruct the pointer
    fshm->client_task_vaddr_ =
        reinterpret_cast<size_t>(static_cast<Task *>(task_ptr.ptr_));
    // NOT setting FUTURE_COPY_FROM_CLIENT — worker will use direct path
    fshm->flags_.SetBits(FutureShm::FUTURE_DEVICE_SCOPE);

    hipc::ShmPtr<FutureShm> fshmptr;
    fshmptr.alloc_id_ = hipc::AllocatorId::GetNull();
    fshmptr.off_ = reinterpret_cast<size_t>(fshm);
    Future<TaskT> future(fshmptr, task_ptr);

    // Distribute across queue lanes using warp ID
    u32 lane_id = 0;
#if HSHM_IS_GPU
    if (queue == internal_queue_) {
      if (internal_num_lanes_ > 1) {
        lane_id = GetWarpId() % internal_num_lanes_;
      }
    } else {
      if (gpu2gpu_num_lanes_ > 1) {
        lane_id = GetWarpId() % gpu2gpu_num_lanes_;
      }
    }
#endif
    auto &lane = queue->GetLane(lane_id, 0);
    Future<Task> task_future(future.GetFutureShmPtr());

    // No input serialization — task is already populated in GPU memory
    // No copy_space_size needed for ring buffer
    // Fence: ensure FutureShm fields are visible before queue entry
    hipc::threadfence();
    lane.Push(task_future);
    return future;
  }

  /**
   * Receive task results on the GPU.
   *
   * Waits for FUTURE_COMPLETE (set by the worker after serializing output),
   * then deserializes the output from the ring buffer if data was written.
   *
   * @tparam TaskT Task type
   * @param future Future to receive results from
   * @param task_ptr Pointer to the task to deserialize output into
   */
  template <typename TaskT>
  HSHM_GPU_FUN void RecvGpu(Future<TaskT> &future, TaskT *task_ptr) {
    // Warp-cooperative: all lanes enter, lane 0 handles control plane,
    // all lanes participate in deserialization via warp_converged read_binary.
    u32 lane = GetLaneId();

    // === Phase 1: Broadcast FutureShm pointer from lane 0 ===
    unsigned long long fshm_ull = 0;
    if (lane == 0) {
      hipc::FullPtr<FutureShm> fshm_full = future.GetFutureShm();
      if (!fshm_full.IsNull()) {
        fshm_ull = reinterpret_cast<unsigned long long>(fshm_full.ptr_);
      }
    }
    fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);
    if (fshm_ull == 0) return;
    FutureShm *fshm = reinterpret_cast<FutureShm *>(fshm_ull);

    // Broadcast device_scope flag
    int ds_int = 0;
    if (lane == 0) {
      ds_int = fshm->flags_.AnySystem(FutureShm::FUTURE_DEVICE_SCOPE) ? 1 : 0;
    }
    ds_int = __shfl_sync(0xFFFFFFFF, ds_int, 0);
    bool device_scope = (ds_int != 0);

    // === Phase 2: All lanes spin-wait on FUTURE_COMPLETE ===
#if !HSHM_IS_HOST
    if (device_scope) {
      int spin_count = 0;
      while (!fshm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
        ++spin_count;
        if (spin_count == 1000000 && lane == 0) {
          u32 raw_flags = fshm->flags_.bits_.load_device();
          size_t out_tw = fshm->output_.total_written_.load_device();
          printf("[RECV-STUCK] blk=%u flags=0x%x out_tw=%llu spin=%d\n",
                 blockIdx.x, raw_flags, (unsigned long long)out_tw, spin_count);
          spin_count = 0;
        }
      }
      hipc::threadfence();
    } else
#endif
    {
      (void)device_scope;
      while (!fshm->flags_.AnySystem(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
      }
      hipc::threadfence();
    }

    // === Phase 3: Read output via preallocated path ===
    unsigned long long task_ull = 0;
    int has_output = 0;
    unsigned long long recv_cs_ull = 0, recv_si_ull = 0;

    if (lane == 0) {
      size_t output_written = device_scope
          ? fshm->output_.total_written_.load_device()
          : fshm->output_.total_written_.load_system();

      if (output_written > 0) {
        task_ull = reinterpret_cast<unsigned long long>(task_ptr);
        recv_cs_ull = reinterpret_cast<unsigned long long>(fshm->copy_space);
        recv_si_ull = reinterpret_cast<unsigned long long>(&fshm->output_);
        has_output = 1;
      }
    }

    task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);
    has_output = __shfl_sync(0xFFFFFFFF, has_output, 0);
    recv_cs_ull = hipc::shfl_sync_u64(0xFFFFFFFF, recv_cs_ull, 0);
    recv_si_ull = hipc::shfl_sync_u64(0xFFFFFFFF, recv_si_ull, 0);

    // Lane 0: read PreallocHeader, reconstruct wrap_vector, deserialize output
    if (lane == 0 && has_output && device_scope) {
      // Read header from copy_space
      PreallocHeader hdr;
      hshm::lbm::LbmContext ctx;
      ctx.copy_space = reinterpret_cast<char*>(recv_cs_ull);
      ctx.shm_info_ = reinterpret_cast<hshm::lbm::ShmTransferInfo*>(recv_si_ull);
      hshm::lbm::ShmTransport::RecvDevicePrealloc(
          reinterpret_cast<char*>(&hdr), sizeof(hdr), ctx);

      // Construct wrap_vector pointing at task data in copy_space
      hipc::FullPtr<char> data_fp;
      data_fp.ptr_ = ctx.copy_space + WarpIpcManager::kHeaderSize;
      data_fp.shm_.alloc_id_.SetNull();
      data_fp.shm_.off_ = reinterpret_cast<size_t>(data_fp.ptr_);
      hshm::priv::wrap_vector recv_buf(data_fp, hdr.data_size);
      recv_buf.resize(hdr.data_size);

      WrapLoadArchive load_ar(recv_buf);
      load_ar.SetMsgType(hdr.msg_type);
      auto *task = reinterpret_cast<TaskT *>(task_ull);
      task->SerializeOut(load_ar);
    }

    // Lane 0: non-device-scope path (CPU→GPU response via SPSC)
    if (lane == 0 && has_output && !device_scope) {
      auto *mgr = GetWarpManager();
      hshm::lbm::LbmContext cpu_ctx;
      cpu_ctx.copy_space = fshm->copy_space;
      cpu_ctx.shm_info_ = &fshm->output_;
      hshm::lbm::ShmTransport::Recv(mgr->load_ar_, cpu_ctx);
      mgr->load_ar_.SetMsgType(LocalMsgType::kSerializeOut);
      auto *task = reinterpret_cast<TaskT *>(task_ull);
      task->SerializeOut(mgr->load_ar_);
    }
    __syncwarp();

    // === Phase 5: Lane 0 cleanup ===
    if (lane == 0) {
      future.Destroy(true);
    }
  }

  /**
   * Find the per-thread GPU allocator matching the given AllocatorId.
   * Checks primary table first, then GPU→CPU table.
   * Falls back to primary if no match.
   *
   * @param id AllocatorId to search for
   * @return Pointer to the matching BuddyAllocator for this thread
   */
  HSHM_GPU_FUN HSHM_DEFAULT_ALLOC_GPU_T *FindGpuAlloc(
      const hipc::AllocatorId &id) {
    if (gpu_alloc_) {
      if (gpu_alloc_->GetId() == id) {
        return static_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
            static_cast<void *>(gpu_alloc_));
      }
    }
    if (gpu2cpu_alloc_) {
      if (gpu2cpu_alloc_->GetId() == id) {
        return static_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
            static_cast<void *>(gpu2cpu_alloc_));
      }
    }
    // Fallback: use primary
    return gpu_alloc_ ? static_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
                            static_cast<void *>(gpu_alloc_))
                      : nullptr;
  }
#endif  // HSHM_IS_GPU_COMPILER

#if HSHM_IS_GPU_COMPILER
  /**
   * Per-block IpcManager singleton in __shared__ memory.
   * __noinline__ ensures a single __shared__ variable instance per block,
   * making this a per-block singleton accessible from any device function.
   * The object is NOT constructed — use ClientInitGpu to set up fields.
   * @return Pointer to the per-block IpcManager
   */
  static HSHM_GPU_FUN __noinline__ IpcManager *GetBlockIpcManager() {
    // Use raw bytes to avoid invoking host-only constructors of IpcManager
    // members. The memory is initialized via ClientInitGpu before any fields
    // are accessed.
    __shared__ char s_ipc_bytes[sizeof(IpcManager)];
    return reinterpret_cast<IpcManager *>(s_ipc_bytes);
  }

#endif  // HSHM_IS_GPU_COMPILER

  /**
   * Create Future by wrapping task pointer (runtime-only, no serialization)
   * Used by runtime workers to avoid unnecessary copying
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to wrap in Future
   * @return Future<TaskT> wrapping task pointer directly
   */
  template <typename TaskT>
  Future<TaskT> MakePointerFuture(hipc::FullPtr<TaskT> task_ptr) {
    // Check task_ptr validity
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Allocate and construct FutureShm (no copy_space for runtime path)
    hipc::FullPtr<FutureShm> future_shm = NewObj<FutureShm>();
    if (future_shm.IsNull()) {
      return Future<TaskT>();
    }

    // Initialize FutureShm fields
    future_shm.ptr_->pool_id_ = task_ptr->pool_id_;
    future_shm.ptr_->method_id_ = task_ptr->method_;
    future_shm.ptr_->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    future_shm.ptr_->client_task_vaddr_ = 0;
    // No copy_space in runtime path — ShmTransferInfo defaults are fine

    // Create Future with ShmPtr and task_ptr (no serialization)
    Future<TaskT> future(future_shm.shm_, task_ptr);
    return future;
  }

  /**
   * Create a Future for a task with optional serialization
   * Used internally by Send and as a public interface for future creation
   *
   * Two execution paths:
   * - Client thread (IsClientThread=true): Serialize the task into the Future
   * - Runtime thread (IsClientThread=false): Wrap task_ptr directly without
   * serialization
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to wrap in Future
   * @return Future<TaskT> wrapping the task
   */
  template <typename TaskT>
  Future<TaskT> MakeFuture(const hipc::FullPtr<TaskT> &task_ptr) {
#if HSHM_IS_GPU
    // GPU PATH: Always use MakeCopyFutureGpu to serialize the task
    return MakeCopyFutureGpu(task_ptr);
#else
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    Worker *worker = CHI_CUR_WORKER;

    // Runtime path requires BOTH IsRuntime AND worker to be non-null
    bool use_runtime_path = is_runtime && worker != nullptr;

    if (!use_runtime_path) {
      // CLIENT PATH: Use MakeCopyFuture to serialize the task
      return MakeCopyFuture(task_ptr);
    } else {
      // RUNTIME PATH: Use MakePointerFuture to wrap pointer without
      // serialization
      return MakePointerFuture(task_ptr);
    }
#endif
  }

  /**
   * Send task asynchronously (serializes into Future)
   * Creates a Future wrapper, serializes task inputs, and enqueues to worker
   *
   * Two execution paths:
   * - Client thread (IsClientThread=true): Serialize task and copy Future with
   * null task pointer
   * - Runtime thread (IsClientThread=false): Create Future with task pointer
   * directly (no copy)
   *
   * @param task_ptr Task to send
   * @param awake_event Whether to awaken worker after enqueueing
   * @return Future<TaskT> for polling completion and retrieving results
   */
#if HSHM_IS_GPU_COMPILER
  /** GPU-side RouteTask: all routing is handled in SendGpu() itself now. */
  HSHM_GPU_FUN RouteResult RouteTask(const hipc::FullPtr<Task> &task_ptr) {
    RoutingMode mode = task_ptr->pool_query_.GetRoutingMode();
    if (mode == RoutingMode::ToLocalCpu) {
      return RouteResult::Network;
    }
    return RouteResult::Local;
  }
#endif

  template <typename TaskT>
  HSHM_CROSS_FUN Future<TaskT> Send(const hipc::FullPtr<TaskT> &task_ptr,
                                    bool awake_event = true) {
#if HSHM_IS_GPU_COMPILER
    if (is_gpu_runtime_) {
      return SendGpuDirect(task_ptr);
    }
#endif
#if HSHM_IS_GPU
    {
      return SendGpu(task_ptr);
    }
#else  // HOST PATH
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    {
      RoutingMode mode = task_ptr->pool_query_.GetRoutingMode();
      if (mode == RoutingMode::LocalGpuBcast ||
          mode == RoutingMode::ToLocalGpu) {
        u32 gpu_id = (mode == RoutingMode::ToLocalGpu)
                         ? task_ptr->pool_query_.GetNodeId()
                         : 0;
        return SendToGpu(task_ptr, gpu_id);
      }
    }
#endif
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    Worker *worker = CHI_CUR_WORKER;

    // Client TCP/IPC path: serialize and send via ZMQ
    // Runtime always uses SHM path internally, even from the main thread
    if (!is_runtime && ipc_mode_ != IpcMode::kShm) {
      return SendZmq(task_ptr, ipc_mode_);
    }

    // Client SHM path: use SendShm (lightbeam transport)
    if (!is_runtime) {
      return SendShm(task_ptr);
    }

    // Runtime SHM path: worker threads use SendRuntimeClient (simple
    // ClientMapTask path), non-worker threads use SendRuntime.
    Future<Task> base_future;
    if (worker != nullptr) {
      base_future = SendRuntimeClient(task_ptr.template Cast<Task>());
    } else {
      base_future = SendRuntime(task_ptr.template Cast<Task>());
    }
    return base_future.Cast<TaskT>();
#endif
  }

  /** Send from a worker thread: creates pointer future, uses ClientMapTask */
  Future<Task> SendRuntimeClient(const hipc::FullPtr<Task> &task_ptr);

  /** Send from non-worker thread: full routing (pool query, local/global) */
  Future<Task> SendRuntime(const hipc::FullPtr<Task> &task_ptr);

  /**
   * Initialize RunContext for a task before routing.
   * @param future Future containing the task
   * @param container Container for the task (can be nullptr)
   * @param lane Lane for the task (can be nullptr)
   */
  void BeginTask(Future<Task> &future, Container *container, TaskLane *lane);

  /** Route a task: resolve pool query, determine local vs global.
   * If force_enqueue is true, always enqueue to the destination worker's lane
   * (used by SendRuntime which cannot execute tasks directly). */
  RouteResult RouteTask(Future<Task> &future, bool force_enqueue = false);

  /** Resolve a pool query into concrete physical addresses */
  std::vector<PoolQuery> ResolvePoolQuery(const PoolQuery &query,
                                          PoolId pool_id,
                                          const FullPtr<Task> &task_ptr);

  /** Check if task should be processed locally */
  bool IsTaskLocal(const FullPtr<Task> &task_ptr,
                   const std::vector<PoolQuery> &pool_queries);

  /** Route task locally.
   * If force_enqueue is true, always enqueue even if dest == current worker. */
  RouteResult RouteLocal(Future<Task> &future, bool force_enqueue = false);

  /** Route task globally via network */
  RouteResult RouteGlobal(Future<Task> &future,
                          const std::vector<PoolQuery> &pool_queries);

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /** Route a task from CPU to GPU orchestrator (non-template, type-erased).
   * Uses Container::LocalSaveTask for serialization. */
  void RouteToGpu(const hipc::FullPtr<Task> &task_ptr, Container *container,
                  u32 gpu_id = 0);
#endif

  /**
   * Send a task via SHM lightbeam transport
   * Allocates FutureShm with copy_space, enqueues to worker lane,
   * then streams task data through shared memory using lightbeam protocol
   * @param task_ptr Task to send
   * @return Future for polling completion
   */
  template <typename TaskT>
  Future<TaskT> SendShm(const hipc::FullPtr<TaskT> &task_ptr) {
    if (task_ptr.IsNull()) return Future<TaskT>();

    // Allocate FutureShm with copy_space
    size_t copy_space_size = task_ptr->GetCopySpaceSize();
    if (copy_space_size == 0) copy_space_size = KILOBYTES(4);
    size_t alloc_size = sizeof(FutureShm) + copy_space_size;
    auto buffer = AllocateBuffer(alloc_size);
    if (buffer.IsNull()) return Future<TaskT>();

    FutureShm *future_shm = new (buffer.ptr_) FutureShm();
    future_shm->pool_id_ = task_ptr->pool_id_;
    future_shm->method_id_ = task_ptr->method_;
    future_shm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    future_shm->client_task_vaddr_ = reinterpret_cast<uintptr_t>(task_ptr.ptr_);
    future_shm->input_.copy_space_size_ = copy_space_size;
    future_shm->output_.copy_space_size_ = copy_space_size;
    future_shm->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

    // Create Future
    auto future_shm_shmptr = buffer.shm_.template Cast<FutureShm>();
    Future<TaskT> future(future_shm_shmptr, task_ptr);

    // Build SHM context for transfer
    hshm::lbm::LbmContext ctx;
    ctx.copy_space = future_shm->copy_space;
    ctx.shm_info_ = &future_shm->input_;

    // Enqueue BEFORE sending (worker must start RecvMetadata concurrently)
    LaneId lane_id =
        scheduler_->ClientMapTask(this, future.template Cast<Task>());
    auto &lane = worker_queues_->GetLane(lane_id, 0);
    bool was_empty = lane.Empty();
    lane.Push(future.template Cast<Task>());
    if (was_empty) {
      AwakenWorker(&lane);
    }

    SaveTaskArchive archive(MsgType::kSerializeIn, shm_send_transport_.get());
    archive << (*task_ptr.ptr_);
    shm_send_transport_->Send(archive, ctx);

    return future;
  }

  /**
   * Send a task via lightbeam transport (TCP or IPC)
   * Serializes the task, creates a private-memory FutureShm, sends via
   * lightbeam PUSH/PULL
   * @param task_ptr Task to send
   * @param mode Transport mode (kTcp or kIpc)
   * @return Future for polling completion
   */
  template <typename TaskT>
  Future<TaskT> SendZmq(const hipc::FullPtr<TaskT> &task_ptr, IpcMode mode) {
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Set net_key for response routing (use task's address as unique key)
    size_t net_key = reinterpret_cast<size_t>(task_ptr.ptr_);
    task_ptr->task_id_.net_key_ = net_key;

    // Serialize the task inputs using network archive
    SaveTaskArchive archive(MsgType::kSerializeIn, zmq_transport_.get());
    archive << (*task_ptr.ptr_);

    // Allocate FutureShm via HSHM_MALLOC (no copy_space needed)
    size_t alloc_size = sizeof(FutureShm);
    hipc::FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(alloc_size);
    if (buffer.IsNull()) {
      HLOG(kError, "SendZmq: Failed to allocate FutureShm ({} bytes)",
           alloc_size);
      return Future<TaskT>();
    }
    FutureShm *future_shm = new (buffer.ptr_) FutureShm();

    // Initialize FutureShm fields
    future_shm->pool_id_ = task_ptr->pool_id_;
    future_shm->method_id_ = task_ptr->method_;
    future_shm->origin_ = (mode == IpcMode::kTcp)
                              ? FutureShm::FUTURE_CLIENT_TCP
                              : FutureShm::FUTURE_CLIENT_IPC;
    future_shm->client_task_vaddr_ = net_key;
    // No copy_space for ZMQ path — ShmTransferInfo defaults are fine

    // Register in pending futures map keyed by net_key
    {
      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      pending_zmq_futures_[net_key] = future_shm;
    }

    // Send via lightbeam PUSH client
    {
      std::lock_guard<std::mutex> lock(zmq_client_send_mutex_);
      zmq_transport_->Send(archive, hshm::lbm::LbmContext());
    }

    // Create Future wrapping the HSHM_MALLOC-allocated FutureShm
    hipc::ShmPtr<FutureShm> future_shm_shmptr =
        buffer.shm_.template Cast<FutureShm>();

    return Future<TaskT>(future_shm_shmptr, task_ptr);
  }

  /**
   * Receive task results (deserializes from completed Future)
   * Called after Future::Wait() has confirmed task completion
   *
   * Two execution paths:
   * - Path 1 (fits): Data fits in serialized_task_capacity, deserialize
   * directly
   * - Path 2 (streaming): Data larger than capacity, assemble from stream
   * - Runtime mode (IsRuntime): No-op (task already has correct outputs)
   *
   * @param future Future containing completed task
   */
  template <typename TaskT>
  bool Recv(Future<TaskT> &future, float max_sec = 0) {
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    if (is_runtime) return true;

    auto future_shm = future.GetFutureShm();
    TaskT *task_ptr = future.get();
    u32 origin = future_shm->origin_;

    // === SHM path: server must be alive to use ring buffer ===
    if (origin == FutureShm::FUTURE_CLIENT_SHM && server_alive_.load()) {
      // Normal SHM path: server is alive, use ring buffer recv
      hshm::lbm::LbmContext ctx;
      ctx.copy_space = future_shm->copy_space;
      ctx.shm_info_ = &future_shm->output_;

      LoadTaskArchive archive;
      auto info = shm_recv_transport_->Recv(archive, ctx);
      (void)info;

      // Wait for FUTURE_COMPLETE, but bail if the server dies or times out
      hshm::abitfield32_t &flags = future_shm->flags_;
      auto shm_start = std::chrono::steady_clock::now();
      while (!flags.Any(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
        if (!server_alive_.load()) {
          HLOG(kWarning, "Recv(SHM): Server died while waiting for response");
          break;
        }
        if (max_sec > 0) {
          float elapsed = std::chrono::duration<float>(
                              std::chrono::steady_clock::now() - shm_start)
                              .count();
          if (elapsed >= max_sec) {
            HLOG(kWarning, "Recv(SHM): Timeout after {:.1f}s", elapsed);
            return false;
          }
        }
      }

      if (flags.Any(FutureShm::FUTURE_COMPLETE)) {
        // Deserialize outputs
        archive.ResetBulkIndex();
        archive.msg_type_ = MsgType::kSerializeOut;
        archive >> (*task_ptr);
        return true;
      }
      // Server died — fall through to reconnection path below
    }

    // === ZMQ path: covers TCP, IPC, and SHM-with-dead-server ===
    // If origin was SHM but server is dead, reconnect and resend via ZMQ
    if (origin == FutureShm::FUTURE_CLIENT_SHM) {
      if (client_retry_timeout_ == 0 && client_try_new_servers_ <= 0) {
        HLOG(kError,
             "Recv(SHM): Server dead, no retry/failover configured, failing");
        return false;
      }
      HLOG(kWarning, "Recv(SHM): Server dead, attempting reconnect...");
      auto start = std::chrono::steady_clock::now();
      if (!WaitForServerAndReconnect(start)) return false;
      ResendTask(future);
      // Refresh future_shm after resend (origin now TCP/IPC)
      future_shm = future.GetFutureShm();
    }

    // ZMQ wait loop: spin until FUTURE_COMPLETE
    auto start = std::chrono::steady_clock::now();
    while (!future_shm->flags_.Any(FutureShm::FUTURE_COMPLETE)) {
      HSHM_THREAD_MODEL->Yield();
      float elapsed =
          std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
              .count();

      // User-specified max timeout
      if (max_sec > 0 && elapsed >= max_sec) return false;

      // Check heartbeat-cached server liveness
      // Skip if already inside a reconnect attempt (prevents recursion from
      // WaitForLocalServer → Recv → WaitForServerAndReconnect)
      if (!server_alive_.load() && !reconnecting_.load()) {
        if (client_retry_timeout_ == 0 && client_try_new_servers_ <= 0) {
          HLOG(kError,
               "Recv: Server dead, retry_timeout=0, no failover configured, "
               "failing");
          return false;
        }
        HLOG(kWarning, "Recv: Server unreachable, reconnecting...");
        if (!WaitForServerAndReconnect(start)) return false;
        ResendTask(future);
        future_shm = future.GetFutureShm();
        start = std::chrono::steady_clock::now();
        continue;
      }
    }

    // Memory fence
    std::atomic_thread_fence(std::memory_order_acquire);

    // Deserialize from pending_response_archives_
    size_t net_key = future_shm->client_task_vaddr_;
    {
      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      auto it = pending_response_archives_.find(net_key);
      if (it != pending_response_archives_.end()) {
        LoadTaskArchive *archive = it->second.get();
        archive->ResetBulkIndex();
        archive->msg_type_ = MsgType::kSerializeOut;
        *archive >> (*task_ptr);
      }
    }
    return true;
  }

  /**
   * Re-send a task via ZMQ after server restart
   * Works for all origin types (TCP, IPC, SHM)
   * After reconnect, always uses ZMQ transport
   * Updates FutureShm origin to match current ipc_mode_
   * @param future Future containing the task to re-send
   */
  template <typename TaskT>
  void ResendTask(Future<TaskT> &future) {
    auto future_shm = future.GetFutureShm();
    TaskT *task_ptr = future.get();
    size_t old_net_key = future_shm->client_task_vaddr_;

    // Remove old pending entries
    {
      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      pending_zmq_futures_.erase(old_net_key);
      auto it = pending_response_archives_.find(old_net_key);
      if (it != pending_response_archives_.end()) {
        zmq_transport_->ClearRecvHandles(*(it->second));
        pending_response_archives_.erase(it);
      }
    }

    // Use task pointer address as net_key
    size_t net_key = reinterpret_cast<size_t>(task_ptr);
    task_ptr->task_id_.net_key_ = net_key;

    // Re-serialize task inputs
    SaveTaskArchive archive(MsgType::kSerializeIn, zmq_transport_.get());
    archive << (*task_ptr);

    // Clear completion flag
    future_shm->flags_.UnsetBits(FutureShm::FUTURE_COMPLETE);

    // Update origin to current IPC mode (SHM falls back to TCP/IPC)
    future_shm->origin_ = (ipc_mode_ == IpcMode::kIpc)
                              ? FutureShm::FUTURE_CLIENT_IPC
                              : FutureShm::FUTURE_CLIENT_TCP;

    // Update client_task_vaddr_ for response routing
    future_shm->client_task_vaddr_ = net_key;

    // Re-register in pending futures
    {
      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      pending_zmq_futures_[net_key] = future_shm.ptr_;
    }

    // Re-send
    {
      std::lock_guard<std::mutex> lock(zmq_client_send_mutex_);
      zmq_transport_->Send(archive, hshm::lbm::LbmContext());
    }
  }

  /**
   * Set the IsClientThread flag for the current thread
   * @param is_client_thread true if thread is running client code, false
   * otherwise
   */
  void SetIsClientThread(bool is_client_thread);

  /**
   * Get the IsClientThread flag for the current thread
   * @return true if thread is running client code, false otherwise
   */
  bool GetIsClientThread() const;

  /**
   * Get TaskQueue for task processing
   * @return Pointer to the TaskQueue or nullptr if not available
   */
  TaskQueue *GetTaskQueue();

  /**
   * Check if IPC manager is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

  /**
   * Get the current IPC transport mode
   * @return IpcMode enum value (kTcp, kIpc, or kShm)
   */
  IpcMode GetIpcMode() const { return ipc_mode_; }

  /**
   * Get the server's generation counter
   * @return Server generation value, 0 if not available
   */
  u64 GetServerGeneration() const {
    return server_generation_.load(std::memory_order_acquire);
  }

  u64 GetWorkerQueuesOffset() const { return worker_queues_off_; }

  /**
   * Check if the runtime server process is alive
   * SHM mode: checks runtime PID via kill(pid, 0)
   * Other modes: returns true (assume alive until timeout)
   * @return true if server is believed alive
   */
  bool IsServerAlive() const;

  /** Check cached server liveness (set by heartbeat thread) */
  bool IsServerAliveCache() const {
    return server_alive_.load(std::memory_order_acquire);
  }

  /** Background heartbeat thread function */
  void HeartbeatThread();

  /**
   * Reconnect to a restarted server (all transports)
   * Re-attaches SHM, re-verifies server via ClientConnectTask
   * @return true if reconnection succeeded
   */
  bool ReconnectToOriginalHost();

  /**
   * Wait for server to come back and reconnect
   * Polls with 1-second intervals up to client_retry_timeout_
   * @param start Time point when the wait started (for overall timeout)
   * @return true if reconnection succeeded within timeout
   */
  bool WaitForServerAndReconnect(std::chrono::steady_clock::time_point start);

  /**
   * Reconnect the ZMQ transport to a different host.
   * Stops recv thread, destroys old transport, creates new TCP transport
   * to new_addr, restarts recv thread, verifies connectivity.
   * Forces ipc_mode_ to kTcp (SHM/IPC are same-machine only).
   * @param new_addr IP address of the new host
   * @return true if successfully connected to the new host
   */
  bool ReconnectToNewHost(const std::string &new_addr);

  /**
   * Get number of workers from shared memory header
   * @return Number of workers, 0 if not initialized
   */
  u32 GetWorkerCount();

  /**
   * Get number of scheduling queues from shared memory header
   * @return Number of scheduling queues, 0 if not initialized
   */
  u32 GetNumSchedQueues() const;

  /**
   * Set number of scheduling queues in shared memory header
   * Called by scheduler after DivideWorkers to inform IpcManager of actual
   * scheduler worker count
   * @param num_sched_queues Number of scheduler workers that process tasks
   */
  void SetNumSchedQueues(u32 num_sched_queues);

  /**
   * Awaken a worker by sending a signal to its thread
   * Sends SIGUSR1 to the worker's thread ID stored in the TaskLane
   * Only sends signal if the worker is inactive (blocked in epoll_wait)
   * @param lane Pointer to the TaskLane containing the worker's tid and active
   * status
   */
  void AwakenWorker(TaskLane *lane);

  /**
   * Set the node ID in the shared memory header
   * @param hostname Hostname string to hash and store
   */
  void SetNodeId(const std::string &hostname);

  /**
   * Get the node ID from the shared memory header
   * @return 64-bit node ID, 0 if not initialized
   */
  u64 GetNodeId() const;

  /**
   * Load hostfile and populate hostfile map
   * Uses hostfile path from ConfigManager
   * @return true if loaded successfully, false otherwise
   */
  bool LoadHostfile();

  /**
   * Get Host struct by node ID
   * @param node_id 64-bit node ID
   * @return Pointer to Host struct if found, nullptr otherwise
   */
  const Host *GetHost(u64 node_id) const;

  /**
   * Get Host struct by IP address
   * @param ip_address IP address string
   * @return Pointer to Host struct if found, nullptr otherwise
   */
  const Host *GetHostByIp(const std::string &ip_address) const;

  /**
   * Get all hosts from hostfile
   * @return Const reference to vector of all Host structs
   */
  const std::vector<Host> &GetAllHosts() const;

  /**
   * Get number of hosts in the cluster
   * @return Number of hosts
   */
  size_t GetNumHosts() const;

  /**
   * Check if a node is believed to be alive
   * @param node_id Node to check
   * @return true if alive, false if dead or unknown
   */
  bool IsAlive(u64 node_id) const;

  /**
   * Mark a node as dead and record it for retry tracking
   * Removes cached client connections for the dead node
   * @param node_id Node to mark as dead
   */
  void SetDead(u64 node_id);

  /**
   * Mark a node as alive and remove it from dead-node tracking
   * @param node_id Node to mark as alive
   */
  void SetAlive(u64 node_id);

  /**
   * Get the SWIM node state for a node
   * @param node_id Node to query
   * @return NodeState (kDead for unknown nodes)
   */
  NodeState GetNodeState(u64 node_id) const;

  /**
   * Set the SWIM node state and update state_changed_at timestamp
   * @param node_id Node to update
   * @param new_state New state to set
   */
  void SetNodeState(u64 node_id, NodeState new_state);

  /**
   * Set self-fenced status (partition detection)
   * @param fenced true if this node should fence itself
   */
  void SetSelfFenced(bool fenced);

  /**
   * Check if this node is self-fenced
   * @return true if self-fenced
   */
  bool IsSelfFenced() const { return self_fenced_; }

  /**
   * Get the leader node ID (lowest alive node_id)
   * All nodes compute the same leader deterministically from local state
   */
  u64 GetLeaderNodeId() const;

  /**
   * Check if this node is the current leader
   */
  bool IsLeader() const;

  struct DeadNodeEntry {
    u64 node_id;
    std::chrono::steady_clock::time_point detected_at;
  };

  /**
   * Get the list of dead nodes for retry queue scanning
   * @return Const reference to dead_nodes_ vector
   */
  const std::vector<DeadNodeEntry> &GetDeadNodes() const { return dead_nodes_; }

  /**
   * Add a new node to the internal hostfile
   * @param ip_address IP address of the new node
   * @param port Port of the new node's runtime
   * @return Assigned node ID for the new node
   */
  u64 AddNode(const std::string &ip_address, u32 port);

  /**
   * Identify current host from hostfile by attempting TCP server binding
   * Uses hostfile path from ConfigManager
   * @return true if host identified successfully, false otherwise
   */
  bool IdentifyThisHost();

  /**
   * Get current hostname identified during host identification
   * @return Current hostname string
   */
  const std::string &GetCurrentHostname() const;

  /**
   * Set lane mapping policy for task distribution
   * @param policy Lane mapping policy to use
   */
  /**
   * Get the main ZeroMQ server for network communication
   * @return Pointer to main server or nullptr if not initialized
   */
  hshm::lbm::Transport *GetMainTransport() const;

  /**
   * Get this host identified during host identification
   * @return Const reference to this Host struct
   */
  const Host &GetThisHost() const;

  /**
   * Get the lightbeam server for receiving client tasks
   * @param mode IPC mode (kTcp or kIpc)
   * @return Lightbeam Server pointer, or nullptr
   */
  hshm::lbm::Transport *GetClientTransport(IpcMode mode) const;

  /**
   * Client-side thread that receives completed task outputs via lightbeam
   */
  void RecvZmqClientThread();

  /**
   * Clean up a response archive and its zmq_msg_t handles
   * Called from Future::Destroy() to free zero-copy recv buffers
   * @param net_key Net key (client_task_vaddr_) used as map key
   */
  void CleanupResponseArchive(size_t net_key);

  /**
   * Start local ZeroMQ server
   * Uses ZMQ port + 1 for local server operations
   * Must be called after ServerInit completes to ensure runtime is ready
   * @return true if successful, false otherwise
   */
  bool StartLocalServer();

  /**
   * Convert ShmPtr to FullPtr by checking allocator IDs
   * Handles three cases:
   * 1. AllocatorId::GetNull() - offset is the actual memory address (raw
   * pointer)
   * 2. Main allocator - runtime shared memory for queues/futures
   * 3. Per-process shared memory allocators via alloc_map_
   * Acquires reader lock on allocator_map_lock_ for thread-safe access
   * @param shm_ptr The ShmPtr to convert
   * @return FullPtr with matching allocator and pointer, or null FullPtr if no
   * match
   */
  template <typename T>
  HSHM_CROSS_FUN hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
#if HSHM_IS_GPU
    // GPU PATH: find the right allocator then resolve
    if (shm_ptr.IsNull()) {
      return hipc::FullPtr<T>();
    }
    // Null alloc_id_ means off_ stores the raw UVA pointer (GPU→GPU path)
    if (shm_ptr.alloc_id_ == hipc::AllocatorId::GetNull()) {
      T *raw_ptr = reinterpret_cast<T *>(shm_ptr.off_.load());
      return hipc::FullPtr<T>(raw_ptr);
    }
    // Check registered GPU allocators (data backends, etc.)
    for (u32 i = 0; i < gpu_num_allocs_; ++i) {
      if (gpu_allocs_[i].alloc_id == shm_ptr.alloc_id_) {
        T *ptr =
            reinterpret_cast<T *>(gpu_allocs_[i].base + shm_ptr.off_.load());
        return hipc::FullPtr<T>(ptr);
      }
    }
    auto *alloc = FindGpuAlloc(shm_ptr.alloc_id_);
    if (!alloc) return hipc::FullPtr<T>();
    return hipc::FullPtr<T>(alloc, shm_ptr);
#else
    // HOST PATH: Full allocator lookup implementation
    // Case 1: AllocatorId is null - offset IS the raw memory address
    // This is used for private memory allocations (new/delete)
    if (shm_ptr.alloc_id_ == hipc::AllocatorId::GetNull()) {
      // The offset field contains the raw pointer address
      T *raw_ptr = reinterpret_cast<T *>(shm_ptr.off_.load());
      return hipc::FullPtr<T>(raw_ptr);
    }

    // Case 2: Check main allocator (runtime shared memory)
    if (main_allocator_ && shm_ptr.alloc_id_ == main_allocator_->GetId()) {
      return hipc::FullPtr<T>(main_allocator_, shm_ptr);
    }

    // Case 3: Check per-process shared memory allocators via alloc_map_
    // Acquire reader lock for thread-safe access to allocator_map_
    allocator_map_lock_.ReadLock();

    // Convert AllocatorId to lookup key (combine major and minor)
    u64 alloc_key = (static_cast<u64>(shm_ptr.alloc_id_.major_) << 32) |
                    static_cast<u64>(shm_ptr.alloc_id_.minor_);
    auto it = alloc_map_.find(alloc_key);
    hipc::FullPtr<T> result;
    if (it != alloc_map_.end()) {
      result = hipc::FullPtr<T>(it->second, shm_ptr);
    } else {
      // Case 4: Check GPU backend memory registrations
      auto git = gpu_alloc_map_.find(alloc_key);
      if (git != gpu_alloc_map_.end()) {
        size_t off = shm_ptr.off_.load();
        if (off < git->second.capacity) {
          result.ptr_ = reinterpret_cast<T *>(git->second.data + off);
          result.shm_ = shm_ptr;
        }
      }
    }

    // Release the lock before returning
    allocator_map_lock_.ReadUnlock();

    return result;
#endif
  }

  /**
   * Convert raw pointer to FullPtr by checking allocators
   * Uses ContainsPtr() on each allocator to find the matching one
   * Checks main allocator first, then per-process allocators
   * If no allocator contains the pointer, returns a FullPtr with null allocator
   * (private memory)
   * Acquires reader lock on allocator_map_lock_ for thread-safe access
   * @param ptr The raw pointer to convert
   * @return FullPtr with matching allocator and pointer, or FullPtr with null
   * allocator if no match (private memory)
   */
  template <typename T>
  HSHM_CROSS_FUN hipc::FullPtr<T> ToFullPtr(T *ptr) {
#if HSHM_IS_GPU
    // GPU PATH: Wrap raw pointer with primary PartitionedAllocator
    if (ptr == nullptr) {
      return hipc::FullPtr<T>();
    }
    if (!gpu_alloc_) return hipc::FullPtr<T>();
    return hipc::FullPtr<T>(reinterpret_cast<hipc::Allocator *>(gpu_alloc_),
                            ptr);
#else
    // HOST PATH: Full allocator lookup implementation
    if (ptr == nullptr) {
      return hipc::FullPtr<T>();
    }

    // Check main allocator
    if (main_allocator_ && main_allocator_->ContainsPtr(ptr)) {
      return hipc::FullPtr<T>(main_allocator_, ptr);
    }

    // Check per-process shared memory allocators
    // Acquire reader lock for thread-safe access
    allocator_map_lock_.ReadLock();

    hipc::FullPtr<T> result;
    for (auto *alloc : alloc_vector_) {
      if (alloc && alloc->ContainsPtr(ptr)) {
        result = hipc::FullPtr<T>(alloc, ptr);
        allocator_map_lock_.ReadUnlock();
        return result;
      }
    }

    // Release the lock before returning
    allocator_map_lock_.ReadUnlock();

    // No matching allocator found - treat as private memory
    // Return FullPtr with the raw pointer (null allocator ID)
    return hipc::FullPtr<T>(ptr);
#endif
  }

  /**
   * Get or create a persistent ZeroMQ client connection from the pool
   * Creates a new connection if one doesn't exist for the given address:port
   * Thread-safe using internal mutex protection
   * @param addr IP address to connect to
   * @param port Port number to connect to
   * @return Pointer to the ZeroMQ client (owned by the pool)
   */
  hshm::lbm::Transport *GetOrCreateClient(const std::string &addr, int port);

  /**
   * Clear all cached client connections
   * Should be called during shutdown
   */
  void ClearClientPool();

  /**
   * Set the net worker's lane pointer for signaling on EnqueueNetTask
   * Called by scheduler after DivideWorkers assigns net_worker_
   * @param lane Pointer to the net worker's TaskLane
   */
  void SetNetLane(TaskLane *lane) { net_lane_ = lane; }

  /**
   * Enqueue a Future<SendTask> to the network queue
   * @param future Future containing the SendTask to enqueue
   * @param priority Network queue priority (kSendIn or kSendOut)
   */
  void EnqueueNetTask(Future<Task> future, NetQueuePriority priority);

  /**
   * Try to pop a Future<SendTask> from the network queue
   * @param priority Network queue priority to pop from
   * @param future Output parameter for the popped Future
   * @return true if a Future was popped, false if queue is empty
   */
  bool TryPopNetTask(NetQueuePriority priority, Future<Task> &future);

  /**
   * Get the network queue for direct access
   * @return Pointer to the network queue or nullptr if not initialized
   */
  NetQueue *GetNetQueue() { return net_queue_.ptr_; }

  /**
   * Get number of GPU→CPU queues (one per GPU device).
   * The CPU GPU worker polls these.
   */
  size_t GetGpuQueueCount() const {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    return gpu2cpu_queues_.size();
#else
    return gpu_queues_.size();
#endif
  }

  /**
   * Get GPU→CPU queue by index (CPU worker polls this).
   * @param gpu_id GPU device ID (0-based)
   */
  TaskQueue *GetGpuQueue(size_t gpu_id) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    if (gpu_id < gpu2cpu_queues_.size()) {
      return gpu2cpu_queues_[gpu_id].ptr_;
    }
    return nullptr;
#else
    if (gpu_id < gpu_queues_.size()) {
      return gpu_queues_[gpu_id].ptr_;
    }
    return nullptr;
#endif
  }

  /**
   * Register a GPU queue (non-CUDA fallback path).
   * @param queue FullPtr to a TaskQueue in GPU-accessible shared memory
   */
  void RegisterGpuQueue(hipc::FullPtr<TaskQueue> queue) {
    gpu_queues_.push_back(queue);
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /**
   * Get number of CPU→GPU queues (one per GPU device).
   * Orchestrator polls these.
   */
  size_t GetToGpuQueueCount() const { return cpu2gpu_queues_.size(); }

  /**
   * Get CPU→GPU queue by index (orchestrator polls this).
   * Non-inline to avoid ODR layout mismatch between nvcc and g++ compilations.
   * @param gpu_id GPU device ID (0-based)
   */
  TaskQueue *GetToGpuQueue(size_t gpu_id);

  /**
   * Get number of GPU→GPU queues (one per GPU device, in device memory).
   */
  size_t GetGpuToGpuQueueCount() const { return gpu2gpu_queues_.size(); }

  /**
   * Get GPU→GPU queue by index (device memory, orchestrator polls).
   * Non-inline to avoid ODR layout mismatch between nvcc and g++ compilations.
   * @param gpu_id GPU device ID (0-based)
   */
  TaskQueue *GetGpuToGpuQueue(size_t gpu_id);
  /**
   * Allocate a buffer from the GPU queue backend (pinned host memory).
   *
   * Used by SendToGpu to allocate FutureShm in GPU-accessible memory
   * that both the CPU and the orchestrator can access.
   *
   * @param size Number of bytes to allocate
   * @param gpu_id GPU device ID (0-based)
   * @return FullPtr to allocated buffer, or null on failure
   */
  hipc::FullPtr<char> AllocateGpuBuffer(size_t size, u32 gpu_id = 0);

  /**
   * Submit a task from the CPU to the GPU orchestrator.
   *
   * Allocates FutureShm in the GPU queue backend (pinned host memory),
   * serializes the task input via ShmTransport::Send, and pushes
   * a Future<Task> onto to_gpu_queues_[gpu_id] for the orchestrator to pop.
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to submit
   * @param gpu_id Target GPU device ID (0-based)
   * @return Future<TaskT> for polling completion
   */
  template <typename TaskT>
  Future<TaskT> SendToGpu(const hipc::FullPtr<TaskT> &task_ptr,
                          u32 gpu_id = 0) {
#if HSHM_IS_HOST
    if (task_ptr.IsNull() || gpu_id >= cpu2gpu_queues_.size()) {
      return Future<TaskT>();
    }

    // Allocate FutureShm in GPU queue backend
    size_t copy_space_size = task_ptr->GetCopySpaceSize();
    if (copy_space_size == 0) copy_space_size = 4096;
    size_t alloc_size = sizeof(FutureShm) + copy_space_size;
    hipc::FullPtr<char> buffer = AllocateGpuBuffer(alloc_size, gpu_id);
    if (buffer.IsNull()) {
      return Future<TaskT>();
    }

    // Construct FutureShm
    FutureShm *future_shm = new (buffer.ptr_) FutureShm();
    future_shm->pool_id_ = task_ptr->pool_id_;
    future_shm->method_id_ = task_ptr->method_;
    future_shm->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    future_shm->client_task_vaddr_ = 0;
    future_shm->input_.copy_space_size_ = copy_space_size;
    future_shm->output_.copy_space_size_ = copy_space_size;
    future_shm->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

    // Create Future
    hipc::ShmPtr<FutureShm> fshmptr = buffer.shm_.template Cast<FutureShm>();
    Future<TaskT> future(fshmptr, task_ptr);

    // Serialize task input into copy_space ring buffer
    hshm::lbm::LbmContext ctx;
    ctx.copy_space = future_shm->copy_space;
    ctx.shm_info_ = &future_shm->input_;
    chi::priv::vector<char> save_buf;
    save_buf.reserve(256);
    DefaultSaveArchive save_ar(LocalMsgType::kSerializeIn, save_buf);
    task_ptr->SerializeIn(save_ar);
    hshm::lbm::ShmTransport::Send(save_ar, ctx);

    // Flush FutureShm header + copy_space data to DRAM so the GPU can read
    // all fields via system-scope loads.  CPU writes go to CPU L1/L2/LLC;
    // without clflush they may not reach DRAM before the GPU (on discrete
    // PCIe) reads them.  Must flush the entire allocation (header + ring
    // buffer data written by ShmTransport::Send above).
#if defined(__x86_64__) || defined(__i386__)
    {
      const char *base = reinterpret_cast<const char *>(future_shm);
      for (const char *cl = base; cl < base + alloc_size; cl += 64) {
        _mm_clflush(cl);
      }
      _mm_sfence();
    }
#endif

    // Push to CPU→GPU queue (orchestrator polls this)
    auto &lane = cpu2gpu_queues_[gpu_id].ptr_->GetLane(0, 0);
    Future<Task> task_future(future.GetFutureShmPtr());
    lane.Push(task_future);

    // Flush the queue lane's tail pointer to DRAM so the GPU sees the push.
    // The FutureShm data was flushed above; this covers the queue header
    // which the GPU polls via system-scope loads.
#if defined(__x86_64__) || defined(__i386__)
    {
      const char *q_base = reinterpret_cast<const char *>(&lane);
      for (const char *cl = q_base; cl < q_base + sizeof(lane); cl += 64) {
        _mm_clflush(cl);
      }
      _mm_sfence();
    }
#endif

    return future;
#else
    (void)task_ptr;
    (void)gpu_id;
    return Future<TaskT>();
#endif  // HSHM_IS_HOST
  }
#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

  /**
   * Assign all registered GPU queue lanes to the GPU worker.
   * Call after RegisterGpuQueue to make the worker poll GPU lanes.
   */
  void AssignGpuLanesToWorker();

  /**
   * Get the scheduler instance
   * IpcManager is the single owner of the scheduler.
   * WorkOrchestrator and Worker should use this method to get the scheduler.
   * @return Pointer to the scheduler or nullptr if not initialized
   */
  Scheduler *GetScheduler() { return scheduler_.get(); }

  /**
   * Register an existing shared memory segment into the IpcManager
   * Called by worker when encountering an unknown allocator in a FutureShm
   * Derives shm_name from alloc_id: chimaera_{pid}_{index}
   * @param alloc_id Allocator ID (major=pid, minor=index)
   * @return true if successful (or already registered), false on error
   */
  bool RegisterMemory(const hipc::AllocatorId &alloc_id);

  /**
   * Get the current process's shared memory info for registration
   * @param index Index of the shared memory segment (0 to shm_count_-1)
   * @return ClientShmInfo for the specified segment
   */
  ClientShmInfo GetClientShmInfo(u32 index) const;

  /**
   * Reap shared memory segments from dead processes
   *
   * Iterates over all registered shared memory segments and checks if the
   * owning process (identified by pid = AllocatorId.major) is still alive.
   * For segments belonging to dead processes, destroys the shared memory
   * backend and removes tracking entries.
   *
   * Does not reap:
   * - Segments owned by the current process
   * - The main allocator segment (AllocatorId 1.0)
   *
   * @return Number of shared memory segments reaped
   */
  size_t WreapDeadIpcs();

  /**
   * Reap all shared memory segments
   *
   * Destroys all shared memory backends (except main allocator) and clears
   * all tracking structures. This is typically called during shutdown to
   * clean up all IPC resources.
   *
   * @return Number of shared memory segments reaped
   */
  size_t WreapAllIpcs();

  /**
   * Clear all memfd symlinks from the per-user chimaera directory.
   *
   * Called during RuntimeInit to clean up leftover memfd symlinks
   * from previous runs or crashed processes. Since the directory is
   * per-user, all entries are cleaned up.
   *
   * @return Number of memfd symlinks successfully removed
   */
  size_t ClearUserIpcs();

  /**
   * Register GPU accelerator memory backend (GPU kernel use only)
   *
   * Called from GPU kernels to store GPU memory backend reference.
   * Per-thread BuddyAllocators are initialized in CHIMAERA_GPU_INIT macro.
   *
   * @param backend GPU memory backend to register
   * @return true on success, false on failure
   */
  HSHM_CROSS_FUN
  bool RegisterAcceleratorMemory(const hipc::MemoryBackend &backend);

 private:
  // Pool query resolution helpers
  std::vector<PoolQuery> ResolveLocalQuery(const PoolQuery &query,
                                           const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveDirectIdQuery(const PoolQuery &query,
                                              PoolId pool_id,
                                              const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveDirectHashQuery(const PoolQuery &query,
                                                PoolId pool_id,
                                                const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveRangeQuery(const PoolQuery &query,
                                           PoolId pool_id,
                                           const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveBroadcastQuery(const PoolQuery &query,
                                               PoolId pool_id,
                                               const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolvePhysicalQuery(const PoolQuery &query,
                                              PoolId pool_id,
                                              const FullPtr<Task> &task_ptr);

  /**
   * Initialize memory segments for server
   * @return true if successful, false otherwise
   */
  bool ServerInitShm();

  /**
   * Initialize memory segments for client
   * @return true if successful, false otherwise
   */
  bool ClientInitShm();

  /**
   * Initialize priority queues for server
   * @return true if successful, false otherwise
   */
  bool ServerInitQueues();

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /**
   * Initialize GPU queues for server (one ring buffer per GPU)
   * Uses pinned host memory with NUMA awareness
   * @return true if successful, false otherwise
   */
  bool ServerInitGpuQueues();
  bool InitGpuBackendsForDevice(int gpu_id, u32 queue_depth);
  void BuildOrchestratorInfo(u32 gpu_id, u32 queue_depth);

 public:
  /**
   * Launch the persistent GPU work orchestrator
   * Must be called after ServerInitGpuQueues and pool manager init
   * @return true if successful or no GPUs available
   */
  bool LaunchGpuOrchestrator();

  /**
   * Allocate a GPU container via the work orchestrator.
   * @param pool_id Pool identifier
   * @param container_id Container ID (typically node_id)
   * @param chimod_name Name of the ChiMod
   * @return Device pointer to allocated gpu::Container, or nullptr
   */
  void *AllocGpuContainer(const PoolId &pool_id, u32 container_id,
                          const std::string &chimod_name);

 private:
  /**
   * Stop the GPU work orchestrator and free resources
   */
  void FinalizeGpuOrchestrator();

#endif

  /**
   * Initialize priority queues for client
   * @return true if successful, false otherwise
   */
  bool ClientInitQueues();

  /**
   * Wait for local server to become available via lightbeam transport
   * Sends a ClientConnectTask and waits for response with timeout
   * Uses CHI_WAIT_SERVER environment variable for timeout (default 30s)
   * @return true if server responded, false on timeout
   */
  bool WaitForLocalServer();

  /**
   * Try to start main server on given hostname
   * Helper method for host identification
   * Uses ZMQ port from ConfigManager and sets main_transport_
   * @param hostname Hostname to bind to
   * @return true if server started successfully, false otherwise
   */
  bool TryStartMainServer(const std::string &hostname);

  bool is_initialized_ = false;

  // Shared memory backend for main segment (task data, FutureShm)
  hipc::PosixShmMmap main_backend_;

  // Allocator ID for main segment
  hipc::AllocatorId main_allocator_id_;

  // Main allocator pointer for runtime shared memory (task data, FutureShm)
  // CPU: MultiProcessAllocator — shared across runtime + client processes
  // GPU: unused (gpu_alloc_table_ provides per-thread BuddyAllocator)
  CHI_TASK_ALLOC_T *main_allocator_ = nullptr;

  // Shared memory backend for queue segment (TaskQueue ring buffers)
  hipc::PosixShmMmap queue_backend_;

  // Allocator ID for queue segment
  hipc::AllocatorId queue_allocator_id_;

  // Queue allocator pointer — ArenaAllocator for all TaskQueue structures
  CHI_QUEUE_ALLOC_T *queue_allocator_ = nullptr;

  // Number of workers for which queues are allocated
  u32 num_workers_ = 0;

  // Number of scheduling queues for task distribution
  u32 num_sched_queues_ = 0;

  // PID of the runtime process (for tgkill)
  pid_t runtime_pid_ = 0;

  // Monotonic counter, set from epoch nanos at init
  std::atomic<u64> server_generation_{0};

  // The worker task queues (multi-lane queue)
  hipc::FullPtr<TaskQueue> worker_queues_;
  // SHM offset of worker_queues_ within queue_allocator_ (server sets it;
  // client receives it via ClientConnectTask and stores here for
  // ClientInitQueues)
  u64 worker_queues_off_ = 0;

  // Network queue for send operations (one lane, two priorities)
  hipc::FullPtr<NetQueue> net_queue_;

  // Net worker's lane pointer for signaling on EnqueueNetTask
  TaskLane *net_lane_ = nullptr;

  // GPU task queues (one ring buffer per GPU device, empty when no GPU)
  std::vector<hipc::FullPtr<TaskQueue>> gpu_queues_;

  // Local ZeroMQ transport (server mode, using lightbeam)
  hshm::lbm::TransportPtr local_transport_;

  // Main ZeroMQ transport (server mode) for distributed communication
  hshm::lbm::TransportPtr main_transport_;

  // IPC transport mode (TCP default, configurable via CHI_IPC_MODE)
  IpcMode ipc_mode_ = IpcMode::kTcp;

  // SHM lightbeam transport (for SendShm / RecvShm)
  hshm::lbm::TransportPtr shm_send_transport_;
  hshm::lbm::TransportPtr shm_recv_transport_;

  // Client-side: DEALER transport for sending tasks and receiving responses
  hshm::lbm::TransportPtr zmq_transport_;
  std::mutex zmq_client_send_mutex_;

  // Server-side: ROUTER transport for receiving client tasks and sending
  // responses
  hshm::lbm::TransportPtr client_tcp_transport_;
  // Server-side: Socket transport for IPC client communication
  hshm::lbm::TransportPtr client_ipc_transport_;

  // Client recv thread (receives completed task outputs via lightbeam)
  std::thread zmq_recv_thread_;
  std::atomic<bool> zmq_recv_running_{false};

  // Background heartbeat thread for server liveness detection
  std::thread heartbeat_thread_;
  std::atomic<bool> heartbeat_running_{false};
  std::atomic<bool> server_alive_{true};

  // Pending futures (client-side, keyed by net_key)
  std::unordered_map<size_t, FutureShm *> pending_zmq_futures_;
  std::mutex pending_futures_mutex_;

  // Pending response archives (client-side, keyed by net_key)
  // Archives stay alive after Recv() deserialization so that zmq zero-copy
  // buffers (stored in recv[].desc) remain valid until Future::Destroy().
  std::unordered_map<size_t, std::unique_ptr<LoadTaskArchive>>
      pending_response_archives_;

  // Dead node tracking for failure detection
  std::vector<DeadNodeEntry> dead_nodes_;

  // Self-fencing flag for partition detection (SWIM protocol)
  bool self_fenced_ = false;

  // Hostfile management
  std::unordered_map<u64, Host> hostfile_map_;  // Map node_id -> Host
  mutable std::vector<Host>
      hosts_cache_;  // Cached vector of hosts for GetAllHosts
  mutable bool hosts_cache_valid_ = false;  // Flag to track cache validity
  Host this_host_;                          // Identified host for this node

  // Client-side server waiting configuration (from environment variables)
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  float wait_server_timeout_ =
      30.0f;  // CHI_WAIT_SERVER: timeout in seconds (default 30)
  u32 poll_server_interval_ =
      1;  // CHI_POLL_SERVER: poll interval in seconds (default 1)

  // Client-side retry configuration
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  u64 client_generation_ = 0;  // Cached server generation at connect time
  float client_retry_timeout_ =
      60.0f;                        // CHI_CLIENT_RETRY_TIMEOUT (default 60s)
  int client_try_new_servers_ = 0;  // CHI_CLIENT_TRY_NEW_SERVERS (default 0)
  std::atomic<bool> reconnecting_{false};  // Guards against recursive reconnect

  // Persistent ZeroMQ transport connection pool
  // Key format: "ip_address:port"
  std::unordered_map<std::string, hshm::lbm::TransportPtr> client_pool_;
  mutable std::mutex client_pool_mutex_;  // Mutex for thread-safe pool access

  // Scheduler for task routing
  std::unique_ptr<Scheduler> scheduler_;

  //============================================================================
  // Per-Process Shared Memory Management
  //============================================================================

  /** Counter for shared memory segments created by this process (starts at 0)
   */
  std::atomic<u32> shm_count_{0};

  /**
   * Map of AllocatorId -> Allocator for all registered shared memory segments
   * Key is the allocator ID (major.minor), value is the allocator pointer
   * Used by ToFullPtr to find the correct allocator for a ShmPtr
   * Protected by allocator_map_lock_ for thread-safe access
   */
  std::unordered_map<u64, hipc::MultiProcessAllocator *> alloc_map_;

  /**
   * Map of AllocatorId -> {data_ptr, capacity} for GPU backend memory
   * Used by ToFullPtr to resolve ShmPtrs allocated by GPU kernels.
   * GPU backends use pinned host memory, so data_ptr is CPU-accessible.
   */
  struct GpuAllocInfo {
    char *data;
    size_t capacity;
  };
  std::unordered_map<u64, GpuAllocInfo> gpu_alloc_map_;

 public:
  /**
   * Register a GPU backend's memory region for host-side ShmPtr resolution.
   * Call this from the host before launching GPU kernels that submit tasks.
   * @param id Backend ID (must match the ID used by GPU-side allocators)
   * @param data Pointer to the GPU backend's data region (pinned host memory)
   * @param capacity Size of the data region in bytes
   */
  /**
   * Non-inline to avoid ODR layout mismatch between nvcc and g++ compilations.
   */
  void RegisterGpuAllocator(const hipc::MemoryBackendId &id, char *data,
                            size_t capacity);
  /**
   * Wait for local server to stop by polling with ClientConnectTask.
   * Sends repeated ClientConnectTask probes with a short timeout.
   * Returns true once the runtime stops responding (connection fails/times
   * out).
   * @param timeout_sec Maximum time to wait for the runtime to stop
   * @return true if runtime stopped, false if still running after timeout
   */
  bool WaitForLocalRuntimeStop(u32 timeout_sec = 30);

  /**
   * RwLock for protecting allocator_map_ access
   * Reader lock: for normal ToFullPtr lookups and allocation attempts
   * Writer lock: for IpcManager cleanup and memory increase operations
   */
  chi::CoRwLock allocator_map_lock_;

  //============================================================================
  // GPU Memory Management (public for CHIMAERA_GPU_INIT macro access)
  //============================================================================

  // --- Primary backend: orchestrator scratch / client GPU→GPU alloc ---
  /** Primary GPU backend (orchestrator scratch or client device memory) */
  hipc::MemoryBackend gpu_backend_;
  /** PartitionedAllocator for GPU→GPU FutureShm or orch scratch (device memory) */
  hipc::PartitionedAllocator *gpu_alloc_ = nullptr;
  static constexpr u32 kMaxCachedWarps = 32;
  WarpIpcManager *warp_mgrs_[kMaxCachedWarps] = {};
  size_t priv_region_size_ = 4096;  /**< Per-warp private BuddyAllocator region */

  // --- GPU→CPU backend: pinned host for cross-direction FutureShm ---
  /** Pinned-host backend for GPU→CPU FutureShm allocation (ToLocalCpu path) */
  hipc::MemoryBackend gpu2cpu_backend_;
  /** PartitionedAllocator for GPU→CPU FutureShm (pinned host) */
  hipc::PartitionedAllocator *gpu2cpu_alloc_ = nullptr;

  // --- GPU allocator table for GPU-side ShmPtr resolution ---
  IpcManagerGpuInfo::GpuAllocEntry
      gpu_allocs_[IpcManagerGpuInfo::kMaxGpuAllocs];
  u32 gpu_num_allocs_ = 0;

  // --- Queue pointers (filled by ClientInitGpu from IpcManagerGpuInfo) ---
  /** GPU→GPU task queue (device memory, orchestrator polls) */
  TaskQueue *gpu2gpu_queue_ = nullptr;
  /** Base of gpu2gpu queue backend for device-side ShmPtr resolution */
  char *gpu2gpu_queue_base_ = nullptr;
  /** Number of lanes in gpu2gpu queue */
  u32 gpu2gpu_num_lanes_ = 1;
  /** Internal subtask queue (device memory, orchestrator polls) */
  TaskQueue *internal_queue_ = nullptr;
  /** Base of internal queue backend */
  char *internal_queue_base_ = nullptr;
  /** Number of lanes in internal queue */
  u32 internal_num_lanes_ = 1;
  /** CPU→GPU task queue (pinned host, orchestrator polls) */
  TaskQueue *cpu2gpu_queue_ = nullptr;
  /** Base of cpu2gpu copy-space backend for GPU-side ShmPtr→ptr conversion */
  char *cpu2gpu_queue_base_ = nullptr;
  /** GPU→CPU task queue (pinned host, CPU worker polls) */
  TaskQueue *gpu2cpu_queue_ = nullptr;

  /** Stored IpcManagerGpuInfo for GPU orchestrator launch */
  IpcManagerGpuInfo gpu_orchestrator_info_;

  /** Flag indicating if GPU backend is initialized */
  bool gpu_backend_initialized_ = false;

  /** True when this IpcManager belongs to the GPU runtime (orchestrator).
   *  When true, Send() uses SendGpuDirect (no serialization) instead of
   *  SendGpu (serialization). Client kernels leave this false (default). */
  bool is_gpu_runtime_ = false;

  /** Scratch allocator generation — incremented each time the scratch
   *  PartitionedAllocator is reinitialized (pause/resume).  GPU containers
   *  compare this against a saved value to detect stale scratch pointers. */
  volatile u32 scratch_gen_ = 0;

  /** Opaque pointer to gpu::WorkOrchestrator (defined in
   * work_orchestrator_gpu.cc) */
  void *gpu_orchestrator_ = nullptr;

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // =========================================================================
  // Server-side GPU backends (one per GPU device)
  // =========================================================================

  /** GPU→GPU queue backend: device memory (GpuMalloc) */
  std::vector<std::unique_ptr<hipc::GpuMalloc>> gpu2gpu_queue_backends_;

  /** Internal subtask queue backend: device memory (GpuMalloc) */
  std::vector<std::unique_ptr<hipc::GpuMalloc>> internal_queue_backends_;

  /** GPU→CPU queue backend: pinned host (GpuShmMmap) */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> gpu2cpu_queue_backends_;

  /** CPU→GPU queue backend: pinned host (GpuShmMmap) */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> cpu2gpu_queue_backends_;

  /** GPU→CPU copy-space backend: pinned host (GpuShmMmap) for GPU→CPU FutureShm
   */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> gpu2cpu_copy_backends_;

  /** CPU→GPU copy-space backend: pinned host (GpuShmMmap) for CPU→GPU FutureShm
   */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> cpu2gpu_copy_backends_;

  /** GPU orchestrator scratch backends (one per GPU, for per-block
   * ArenaAllocators) */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> gpu_orchestrator_backends_;

  /** GPU→GPU queues (device memory, one per GPU) */
  std::vector<hipc::FullPtr<TaskQueue>> gpu2gpu_queues_;

  /** Internal subtask queues (device memory, one per GPU) */
  std::vector<hipc::FullPtr<TaskQueue>> internal_queues_;

  /** GPU→CPU queues (pinned host, one per GPU) */
  std::vector<hipc::FullPtr<TaskQueue>> gpu2cpu_queues_;

  /** CPU→GPU queues (pinned host, one per GPU) */
  std::vector<hipc::FullPtr<TaskQueue>> cpu2gpu_queues_;

  // =========================================================================
  // Client-side GPU backends (attached during ClientInitGpuQueues)
  // =========================================================================

  /** Client-attached GPU→CPU backends (GpuShmMmap, one per GPU) */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> client_gpu2cpu_backends_;

  /** Client-attached CPU→GPU backends (GpuShmMmap, one per GPU) */
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> client_cpu2gpu_backends_;

  /** Server-side: GPU device memory registered by clients (IPC handles) */
  std::vector<std::unique_ptr<hipc::GpuMalloc>> client_gpu_data_backends_;

  // =========================================================================
  // Server-side GPU methods
  // =========================================================================

  /** Register a GPU container with the orchestrator's pool manager */
  void RegisterGpuOrchestratorContainer(const PoolId &pool_id,
                                        void *gpu_container_ptr);

  /**
   * Get queue offsets within backends for ClientConnect response.
   * These are byte offsets from the respective backend's data_ base.
   */
  u64 GetCpu2GpuQueueOffset(u32 gpu_id) const;
  u64 GetGpu2CpuQueueOffset(u32 gpu_id) const;
  u64 GetGpu2GpuQueueOffset(u32 gpu_id) const;
  u64 GetCpu2GpuBackendSize(u32 gpu_id) const;
  u64 GetGpu2CpuBackendSize(u32 gpu_id) const;
  void GetGpu2GpuIpcHandle(u32 gpu_id, char *out_bytes) const;

  /**
   * Register GPU device memory from a client process via IPC handle.
   * Opens the IPC handle, registers in gpu_alloc_map_ for ShmPtr resolution.
   */
  bool RegisterGpuMemoryFromClient(const hipc::MemoryBackendId &backend_id,
                                   const hshm::GpuIpcMemHandle &ipc_handle,
                                   size_t data_capacity);

  /**
   * Initialize client-side GPU backends from ClientConnect response.
   * Attaches to server's GpuShmMmap backends for GPU→CPU and CPU→GPU paths.
   */
  bool ClientInitGpuQueues(u32 num_gpus, const u64 *cpu2gpu_offsets,
                           const u64 *gpu2cpu_offsets,
                           const u64 *gpu2gpu_offsets, const u64 *cpu2gpu_sizes,
                           const u64 *gpu2cpu_sizes, u32 queue_depth,
                           const char gpu2gpu_ipc_handles[][64]);

  /**
   * Build IpcManagerGpuInfo for a client GPU kernel launch.
   * @param gpu_id Target GPU device
   * @return IpcManagerGpuInfo with queue pointers and backend info
   */
  IpcManagerGpuInfo GetClientGpuInfo(u32 gpu_id = 0);
#endif

  /** Print GPU orchestrator profiling breakdown from pinned host memory */
  void PrintGpuOrchestratorProfile();

  /** Pause the GPU orchestrator to free SMs for other GPU kernels.
   *  Returns true if actually paused, false if already paused. */
  bool PauseGpuOrchestrator();

  /** Resume the GPU orchestrator after other GPU kernels complete */
  void ResumeGpuOrchestrator();

  /**
   * Update the GPU orchestrator's block/thread configuration.
   * Non-inline to avoid ODR layout mismatch between nvcc and g++ compilations.
   */
  void SetGpuOrchestratorBlocks(u32 blocks, u32 threads_per_block);

  /** Recreate the gpu2gpu queue with a different lane count (must be paused) */
  void RebuildGpu2GpuQueue(u32 gpu_id, u32 new_lanes);
  void RebuildInternalQueue(u32 gpu_id, u32 new_lanes);

 private:
#if HSHM_IS_HOST
  /**
   * Create a new per-process shared memory segment and register it with the
   * runtime Client-only: sends Admin::RegisterMemory and waits for the server
   * to attach
   * @param size Size in bytes to allocate (32MB will be added for metadata)
   * @return true if successful, false otherwise
   */
  bool IncreaseClientShm(size_t size);

  /**
   * Vector of allocators owned by this process
   * Used for allocation attempts before calling IncreaseClientShm
   */
  std::vector<hipc::MultiProcessAllocator *> alloc_vector_;

  /**
   * Vector of backends owned by this process
   * Stored to ensure backends outlive allocators
   */
  std::vector<std::unique_ptr<hipc::PosixShmMmap>> client_backends_;

  /**
   * Most recently accessed allocator for fast allocation path
   * Checked first in AllocateBuffer before iterating alloc_vector_
   */
  hipc::MultiProcessAllocator *last_alloc_ = nullptr;

  /** Mutex for thread-safe access to shared memory structures */
  mutable std::mutex shm_mutex_;
#endif

  /** Metadata overhead to add to each shared memory segment: 32MB */
  static constexpr size_t kShmMetadataOverhead = 32ULL * 1024 * 1024;

  /** Multiplier for shared memory allocation to ensure space for metadata */
  static constexpr float kShmAllocationMultiplier = 2.5f;
};

}  // namespace chi

// Global pointer variable declaration for IPC manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::IpcManager, g_ipc_manager);

#if HSHM_IS_GPU_COMPILER
namespace chi {
HSHM_CROSS_FUN inline IpcManager *GetIpcManager() {
#if HSHM_IS_GPU
  return IpcManager::GetBlockIpcManager();
#else
  return HSHM_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager);
#endif
}
}  // namespace chi
#define CHI_IPC ::chi::GetIpcManager()
#else
#define CHI_IPC HSHM_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)
#endif

// Define GetPrivAllocGpu and GetSharedAllocGpu now that CHI_IPC is available
#if !HSHM_IS_HOST
namespace chi {
HSHM_GPU_FUN inline hipc::PrivateBuddyAllocator *GetPrivAllocGpu() {
  return CHI_IPC->GetPrivAlloc();
}
HSHM_GPU_FUN inline hipc::PartitionedAllocator *GetSharedAllocGpu() {
  return CHI_IPC->gpu_alloc_;
}
}  // namespace chi
#endif

// Include local_task_archives after CHI_IPC is defined, since on GPU
// CHI_PRIV_ALLOC expands to chi::GetPrivAllocGpu()
#include "chimaera/local_task_archives.h"

// GPU kernel initialization macro
// Creates a shared IPC manager instance in GPU __shared__ memory
// Each thread has its own ArenaAllocator for memory allocation
// Supports 1D, 2D, and 3D thread blocks (max 1024 threads per block)
//
// Usage in GPU kernel:
//   __global__ void my_kernel(const hipc::MemoryBackend* backend) {
//     CHIMAERA_GPU_INIT(gpu_info);
//     // Now CHI_IPC->AllocateBuffer() works for this thread
//   }
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#define CHIMAERA_GPU_INIT(gpu_info)                                           \
  chi::IpcManager *g_ipc_manager_ptr = chi::IpcManager::GetBlockIpcManager(); \
  int thread_id = chi::IpcManager::GetGpuThreadId();                          \
  int num_threads = chi::IpcManager::GetGpuNumThreads();                      \
  if (thread_id == 0) {                                                       \
    chi::IpcManagerGpuInfo g_gpu_info_ = gpu_info;                            \
    g_ipc_manager_ptr->ClientInitGpu(g_gpu_info_, num_threads);               \
  }                                                                           \
  __syncthreads();                                                            \
  chi::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

/**
 * Client process GPU kernel initialization macro.
 * Functionally identical to CHIMAERA_GPU_INIT but with a clearer name
 * for use in separate client processes (not the runtime itself).
 *
 * Usage in client GPU kernel:
 *   __global__ void my_kernel(IpcManagerGpuInfo info) {
 *     CHI_CLIENT_GPU_INIT(info);
 *     FullPtr<char> data = CHI_IPC->AllocateDeviceData(1024);
 *     // ... submit tasks using data ...
 *   }
 */
#define CHI_CLIENT_GPU_INIT(gpu_info) CHIMAERA_GPU_INIT(gpu_info)

/**
 * GPU orchestrator initialization macro — splits both backends per block.
 *
 * Each block gets (data_capacity / num_blocks) bytes from:
 *   - backend: primary scratch memory (device or pinned host)
 *   - gpu2cpu_backend: pinned host for GPU→CPU FutureShm allocation
 * ClientInitGpu further splits each region among threads.
 * Thread 0 of each block performs initialization; all threads sync.
 *
 * @param gpu_info IpcManagerGpuInfo with backends and queue pointers
 * @param num_blocks Total number of blocks in the orchestrator grid
 */
/**
 * Multi-block GPU client kernel initialization macro.
 *
 * Like CHIMAERA_GPU_ORCHESTRATOR_INIT but does NOT set is_gpu_runtime_.
 * This means Send() uses the regular SendGpu() path (serialized into
 * gpu2gpu_queue ring buffer) instead of SendGpuDirect() (raw pointers
 * into internal_queue).
 *
 * Use this for client kernels that submit tasks TO the orchestrator.
 * Use CHIMAERA_GPU_ORCHESTRATOR_INIT only for the orchestrator itself.
 */
#define CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks)                         \
  chi::IpcManager *g_ipc_manager_ptr = chi::IpcManager::GetBlockIpcManager();  \
  int thread_id = chi::IpcManager::GetGpuThreadId();                           \
  int num_threads = chi::IpcManager::GetGpuNumThreads();                       \
  if (thread_id == 0) {                                                        \
    chi::IpcManagerGpuInfo block_info = gpu_info;                              \
    if (blockIdx.x != 0) {                                                     \
      block_info.skip_heap_init = true;                                        \
      block_info.skip_scratch_init = true;                                     \
    }                                                                          \
    g_ipc_manager_ptr->ClientInitGpu(block_info, num_threads, num_blocks);     \
  }                                                                            \
  __syncthreads();                                                             \
  chi::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

#define CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks)                   \
  chi::IpcManager *g_ipc_manager_ptr = chi::IpcManager::GetBlockIpcManager();  \
  int thread_id = chi::IpcManager::GetGpuThreadId();                           \
  int num_threads = chi::IpcManager::GetGpuNumThreads();                       \
  if (thread_id == 0) {                                                        \
    chi::IpcManagerGpuInfo block_info = gpu_info;                              \
    /* Only block 0 initializes heap; others reattach via skip_heap_init */    \
    if (blockIdx.x != 0) {                                                     \
      block_info.skip_heap_init = true;                                        \
      block_info.skip_scratch_init = true;                                     \
    }                                                                          \
    g_ipc_manager_ptr->ClientInitGpu(block_info, num_threads, num_blocks);     \
    g_ipc_manager_ptr->is_gpu_runtime_ = true;                                 \
  }                                                                            \
  __syncthreads();                                                             \
  chi::IpcManager &g_ipc_manager = *g_ipc_manager_ptr
#endif

// Define Future methods after IpcManager and CHI_IPC are fully defined
// This avoids circular dependency issues between task.h and ipc_manager.h
namespace chi {

// Unified AllocateBuffer implementation for GPU (host version is in
// ipc_manager.cc)
// Allocates from gpu_alloc_ (scratch PartitionedAllocator) — shared data visible
// to other warps/blocks and the orchestrator.
#if !HSHM_IS_HOST
inline HSHM_CROSS_FUN hipc::FullPtr<char> IpcManager::AllocateBuffer(
    size_t size) {
  if (gpu_backend_initialized_ && gpu_alloc_ != nullptr) {
    return gpu_alloc_->AllocateObjs<char>(size);
  }
  return hipc::FullPtr<char>::GetNull();
}

inline HSHM_CROSS_FUN hipc::Arena<hipc::PrivateBuddyAllocator> IpcManager::PushPrivArena(
    size_t size) {
  auto *alloc = GetPrivAlloc();
  if (alloc != nullptr) {
    return alloc->PushArena(size);
  }
  return hipc::Arena<hipc::PrivateBuddyAllocator>();
}

inline HSHM_CROSS_FUN hipc::Arena<hipc::PartitionedAllocator> IpcManager::PushArena(
    size_t size) {
  // PushArena is for bulk allocator (PartitionedAllocator-based)
  if (gpu_backend_initialized_ && gpu_alloc_ != nullptr) {
    return gpu_alloc_->PushArena(size);
  }
  return hipc::Arena<hipc::PartitionedAllocator>();
}

// Unified FreeBuffer implementation for GPU (host version is in ipc_manager.cc)
// Routes to the allocator that owns the buffer based on alloc_id.
// Used for bulk allocations (AllocateBuffer path).
inline HSHM_CROSS_FUN void IpcManager::FreeBuffer(FullPtr<char> buffer_ptr) {
  if (buffer_ptr.IsNull()) return;
  if (buffer_ptr.shm_.alloc_id_ == hipc::AllocatorId::GetNull()) {
    // GPU→GPU path: off_ holds an absolute UVA pointer, not a relative offset.
    // ptr_ contains the absolute address.  Use FullPtr(alloc, ptr) to compute
    // the correct relative offset before calling Free.
#if HSHM_IS_GPU_COMPILER
    // Try scratch allocator first (AllocateBuffer source)
    if (gpu_alloc_ && buffer_ptr.ptr_) {
      hipc::FullPtr<char> proper(
          reinterpret_cast<hipc::Allocator *>(gpu_alloc_), buffer_ptr.ptr_);
      if (!proper.IsNull()) {
        gpu_alloc_->Free(proper);
        return;
      }
    }
    if (gpu2cpu_alloc_ && buffer_ptr.ptr_) {
      hipc::FullPtr<char> proper(
          reinterpret_cast<hipc::Allocator *>(gpu2cpu_alloc_), buffer_ptr.ptr_);
      if (!proper.IsNull()) {
        gpu2cpu_alloc_->Free(proper);
        return;
      }
    }
#endif
    // No owning allocator found; silently drop (e.g. private/new memory).
    return;
  }
  auto *alloc = FindGpuAlloc(buffer_ptr.shm_.alloc_id_);
  if (alloc) {
    alloc->Free(buffer_ptr);
  }
}
#endif  // !HSHM_IS_HOST

// ~Future() implementation - frees resources if consumed (via
// Wait/await_resume)
template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN Future<TaskT, AllocT>::~Future() {
  if (consumed_) {
#if HSHM_IS_HOST
    // Clean up zero-copy response archive (TCP/IPC only, never used on GPU)
    if (!future_shm_.IsNull()) {
      hipc::FullPtr<FutureShm> fs = CHI_IPC->ToFullPtr(future_shm_);
      if (!fs.IsNull() && (fs->origin_ == FutureShm::FUTURE_CLIENT_TCP ||
                           fs->origin_ == FutureShm::FUTURE_CLIENT_IPC)) {
        CHI_IPC->CleanupResponseArchive(fs->client_task_vaddr_);
      }
    }
#endif
    // Free FutureShm
    if (!future_shm_.IsNull()) {
      hipc::ShmPtr<char> buffer_shm = future_shm_.template Cast<char>();
      CHI_IPC->FreeBuffer(buffer_shm);
      future_shm_.SetNull();
    }
    // Auto-free the task (only when consumed to avoid double-free
    // from runtime-internal Future copies in event queues / RunContext)
    DelTask();
    // PartitionedAllocator uses individual Free() via address arithmetic, not bulk
    // Reset(). FutureShm freed above via FreeBuffer(); no arena reset needed.
  }
}

// GetFutureShm() implementation - converts internal ShmPtr to FullPtr
// GPU-compatible: uses CHI_IPC macro which works on both CPU and GPU
template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN hipc::FullPtr<typename Future<TaskT, AllocT>::FutureT>
Future<TaskT, AllocT>::GetFutureShm() const {
  if (future_shm_.IsNull()) {
    return hipc::FullPtr<FutureT>();
  }
  return CHI_IPC->ToFullPtr(future_shm_);
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN bool Future<TaskT, AllocT>::Wait(float max_sec,
                                                 bool reuse_task) {
#if HSHM_IS_GPU
  // GPU CLIENT PATH: All lanes enter RecvGpu for warp-cooperative
  // serialization via __shfl_sync. Non-leaders have null future_shm_ —
  // RecvGpu broadcasts the real FutureShm pointer from lane 0.
  CHI_IPC->RecvGpu(*this, task_ptr_.ptr_);
  // Only lane 0 owns the Future — only lane 0 marks consumed
  if (IpcManager::IsWarpScheduler()) {
    if (reuse_task) {
      // Caller will reuse the task — detach so ~Future() won't free it
      task_ptr_.SetNull();
    }
    // Destroy already called inside RecvGpu for lane 0
  }
  return true;
#else
  if (!task_ptr_.IsNull() && !future_shm_.IsNull()) {
    // Fire-and-forget: return immediately without waiting.
    // Detach so ~Future() won't free the in-flight task.
    if (task_ptr_->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
      task_ptr_.SetNull();
      future_shm_.SetNull();
      return true;
    }

    // Convert ShmPtr to FullPtr to access flags_
    hipc::FullPtr<FutureShm> future_full = CHI_IPC->ToFullPtr(future_shm_);
    if (future_full.IsNull()) {
      HLOG(kError, "Future::Wait: ToFullPtr returned null for future_shm_");
      return false;
    }

    // Determine path: client vs runtime
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();

    if (is_runtime) {
      // RUNTIME PATH: Wait for FUTURE_COMPLETE (task outputs are direct,
      // no deserialization needed). Covers both worker threads and main thread.
      hshm::abitfield32_t &flags = future_full->flags_;
      auto start = std::chrono::steady_clock::now();
      while (!flags.Any(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
        if (max_sec > 0) {
          float elapsed = std::chrono::duration<float>(
                              std::chrono::steady_clock::now() - start)
                              .count();
          if (elapsed >= max_sec) return false;
        }
      }

      // Deserialize output from GPU FutureShm ring buffer (SendToGpu path)
      if (flags.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
          future_full->output_.total_written_.load() > 0) {
        hshm::lbm::LbmContext ctx;
        ctx.copy_space = future_full->copy_space;
        ctx.shm_info_ = &future_full->output_;
        chi::priv::vector<char> load_buf;
        load_buf.reserve(256);
        DefaultLoadArchive load_ar(load_buf);
        load_ar.SetMsgType(LocalMsgType::kSerializeOut);
        hshm::lbm::ShmTransport::Recv(load_ar, ctx);
        task_ptr_->SerializeOut(load_ar);
      }
    } else {
      // CLIENT PATH
      // Detect GPU-style futures: FUTURE_COPY_FROM_CLIENT set AND
      // client_task_vaddr_==0 (SendGpu/SendToGpu set it to 0; SendShm sets it
      // to the task pointer which is always non-zero).
      // SHM client futures (SendShm) also set FUTURE_COPY_FROM_CLIENT but use
      // cereal serialization — they must go through CHI_IPC->Recv() instead.
      bool is_gpu_future =
          future_full->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
          (future_full->client_task_vaddr_ == 0);
      if (is_gpu_future) {
        // GPU FUTURE IN CLIENT MODE: poll FUTURE_COMPLETE (set by GPU with
        // system-scope atomics after writing output), then deserialize.
        hshm::abitfield32_t &flags = future_full->flags_;
        auto start = std::chrono::steady_clock::now();
        while (!flags.AnySystem(FutureShm::FUTURE_COMPLETE)) {
          HSHM_THREAD_MODEL->Yield();
          if (max_sec > 0) {
            float elapsed = std::chrono::duration<float>(
                                std::chrono::steady_clock::now() - start)
                                .count();
            if (elapsed >= max_sec) {
              task_ptr_->SetReturnCode(static_cast<u32>(-3));
              return false;
            }
          }
        }
        // Deserialize output from GPU FutureShm ring buffer if present
        if (future_full->output_.total_written_.load() > 0) {
          hshm::lbm::LbmContext ctx;
          ctx.copy_space = future_full->copy_space;
          ctx.shm_info_ = &future_full->output_;
          chi::priv::vector<char> load_buf;
          load_buf.reserve(256);
          DefaultLoadArchive load_ar(load_buf);
          load_ar.SetMsgType(LocalMsgType::kSerializeOut);
          hshm::lbm::ShmTransport::Recv(load_ar, ctx);
          task_ptr_->SerializeOut(load_ar);
        }
      } else {
        // Normal SHM lightbeam or ZMQ streaming
        // FUTURE_COMPLETE will be set by worker after all data is sent.
        // Don't wait for FUTURE_COMPLETE first - that causes deadlock for
        // streaming.
        if (!CHI_IPC->Recv(*this, max_sec)) {
          task_ptr_->SetReturnCode(static_cast<u32>(-1));
          return false;
        }
      }
    }

    // PostWait + free FutureShm; task freed by ~Future()
    if (reuse_task) {
      // Caller will reuse the task — detach so ~Future() won't free it
      task_ptr_.SetNull();
    }
    Destroy(true);
  }
  return true;
#endif
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::Destroy(bool post_wait) {
#if HSHM_IS_HOST
  if (post_wait && !task_ptr_.IsNull()) {
    task_ptr_->PostWait();
  }
#endif
  consumed_ = true;
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::DelTask() {
  if (!task_ptr_.IsNull()) {
    CHI_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
}

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_