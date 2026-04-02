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
#include "hermes_shm/memory/allocator/round_robin_allocator.h"
#include "chimaera/chimaera_manager.h"
#include "chimaera/corwlock.h"
#include "chimaera/scheduler/scheduler.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
#include "chimaera/ipc/ipc_cpu2self.h"
#include "chimaera/ipc/ipc_cpu2cpu.h"
#include "chimaera/ipc/ipc_cpu2cpu_zmq.h"
#include "chimaera/ipc/ipc_cpu2gpu.h"
#include "chimaera/ipc/ipc_gpu2cpu.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "hermes_shm/lightbeam/transport_factory_impl.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"
#include "chimaera/gpu/gpu_info.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include "chimaera/gpu/gpu_ipc_manager.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#endif

namespace chi {

// Forward declaration of gpu::IpcManager
namespace gpu { class IpcManager; }

// Forward declarations — full definitions in local_task_archives.h,
// included after CHI_IPC is defined at the bottom of this header.
template <typename BufferT> class LocalSaveTaskArchive;
template <typename BufferT> class LocalLoadTaskArchive;
using DefaultSaveArchive = LocalSaveTaskArchive<chi::priv::vector<char>>;
using DefaultLoadArchive = LocalLoadTaskArchive<chi::priv::vector<char>>;
enum class LocalMsgType : uint8_t;

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

// IpcManagerGpuInfo and IpcManagerGpu are defined in gpu_info.h

/**
 * IPC Manager singleton for inter-process communication
 *
 * Manages ZeroMQ server using lightbeam from HSHM, three memory segments,
 * and priority queues for task processing.
 * Uses HSHM global cross pointer variable singleton pattern.
 */
class IpcManager {
  friend struct IpcCpu2Self;
  friend struct IpcCpu2Cpu;
  friend struct IpcCpu2CpuZmq;
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  friend struct IpcGpu2Cpu;
  friend struct IpcCpu2Gpu;
#endif

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
   * Returns the per-thread task allocator (CHI_TASK_ALLOC_T*).
   * Host: main_allocator_ (MultiProcessAllocator)
   */
  CHI_TASK_ALLOC_T *GetMainAllocator() {
    return main_allocator_;
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
   * Host-side stubs for GPU warp utilities (always lane 0 / warp 0)
   * These are used by host code that still references these methods
   * (e.g., Future templates). GPU code uses gpu::IpcManager versions.
   */
  static inline u32 GetWarpId() { return 0; }
  static inline u32 GetLaneId() { return 0; }
  static inline bool IsWarpScheduler() { return true; }

  /**
   * Base task allocation: Task + append_size extra bytes.
   * The caller is responsible for constructing any co-located objects
   * (FutureShm, RunContext, etc.) in the appended region.
   */
  template <typename TaskT, typename... Args>
  hipc::FullPtr<TaskT> NewTaskBase(size_t append_size,
                                   Args &&...args) {
    (void)append_size;
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    return hipc::FullPtr<TaskT>(ptr);
  }

  /**
   * Create a task for client or orchestrator use.
   */
  template <typename TaskT, typename... Args>
  hipc::FullPtr<TaskT> NewTask(Args &&...args) {
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    ptr->pod_size_ = static_cast<u32>(sizeof(TaskT));
    hipc::FullPtr<TaskT> result(ptr);
    return result;
  }

  /**
   * Allocate a task for orchestrator-side execution (copy path).
   * Same as NewTask, but with explicit stack_size parameter.
   * Called by autogenerated AllocTaskImpl in LocalAllocLoadDeser.
   */
  template <typename TaskT, typename... Args>
  hipc::FullPtr<TaskT> NewTaskExec(size_t stack_size,
                                   Args &&...args) {
    (void)stack_size;
    return NewTask<TaskT>(std::forward<Args>(args)...);
  }

  /**
   * Delete a task
   * Destructor + operator delete
   */
  template <typename TaskT>
  void DelTask(hipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) return;
    task_ptr.ptr_->~TaskT();
    void *raw = static_cast<void *>(task_ptr.ptr_);
    ::operator delete(raw);
  }

  /**
   * Delete a heap-allocated object.
   * Destructor + operator delete.
   * @param obj_ptr FullPtr to object to delete
   */
  template <typename T>
  void DelObj(hipc::FullPtr<T> obj_ptr) {
    if (obj_ptr.IsNull()) return;
    obj_ptr.ptr_->~T();
    void *raw = static_cast<void *>(obj_ptr.ptr_);
    ::operator delete(raw);
  }

  /**
   * Allocate buffer in appropriate memory segment
   * Client uses cdata segment, runtime uses rdata segment
   * Yields until buffer is allocated successfully
   * @param size Size in bytes to allocate
   * @return FullPtr<char> to allocated memory
   */
  FullPtr<char> AllocateBuffer(size_t size);

  /**
   * Push a bump arena for fast allocation.
   * All subsequent AllocateBuffer/NewObj/NewTask calls will bump-allocate
   * from this arena until it is popped (RAII).
   *
   * @param size Arena size in bytes
   * @return Arena RAII handle
   */
  hipc::Arena<hipc::RoundRobinAllocator> PushArena(size_t size);

  /**
   * Allocate GPU device data from the client's GpuMalloc backend.
   * Uses AllocateGpuBuffer from the GPU queue backend.
   * The resulting ShmPtrs are resolvable server-side via gpu_alloc_map_.
   *
   * @param size Number of bytes to allocate
   * @param gpu_id GPU device ID
   * @return FullPtr<char> to allocated GPU memory
   */
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  hipc::FullPtr<char> AllocateDeviceData(size_t size, u32 gpu_id = 0) {
    return AllocateGpuBuffer(size, gpu_id);
  }
#endif

  /**
   * Free buffer from appropriate memory segment
   * Uses allocator's Free method
   * @param buffer_ptr FullPtr to buffer to free
   */
  void FreeBuffer(FullPtr<char> buffer_ptr);

  /**
   * Free buffer from appropriate memory segment (hipc::ShmPtr<> overload)
   * Converts hipc::ShmPtr<> to FullPtr<char> and calls the main FreeBuffer
   * @param buffer_ptr hipc::ShmPtr<> to buffer to free
   */
  void FreeBuffer(hipc::ShmPtr<char> buffer_ptr) {
    if (buffer_ptr.IsNull()) {
      return;
    }
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
  hipc::FullPtr<T> NewObj(Args &&...args) {
    // Allocate from bulk buffer
    hipc::FullPtr<char> buffer = AllocateBuffer(sizeof(T));
    if (buffer.IsNull()) {
      return hipc::FullPtr<T>();
    }
    T *obj = new (buffer.ptr_) T(std::forward<Args>(args)...);
    return buffer.Cast<T>();
  }

  /**
   * Create Future by copying/serializing task
   * Serializes the task into FutureShm's copy_space
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to serialize into Future
   * @return Future<TaskT> with serialized task data
   */
  template <typename TaskT>
  Future<TaskT> MakeCopyFuture(hipc::FullPtr<TaskT> task_ptr) {
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

  template <typename TaskT>
  Future<TaskT> Send(const hipc::FullPtr<TaskT> &task_ptr,
                     bool awake_event = true) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    {
      RoutingMode mode = task_ptr->pool_query_.GetRoutingMode();
      if (mode == RoutingMode::LocalGpuBcast ||
          mode == RoutingMode::ToLocalGpu) {
        u32 gpu_id = (mode == RoutingMode::ToLocalGpu)
                         ? task_ptr->pool_query_.GetNodeId()
                         : 0;
        if (gpu_ipc_) return IpcCpu2Gpu::ClientSend(gpu_ipc_.get(), task_ptr, gpu_id);
        return Future<TaskT>();
      }
    }
#endif
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    Worker *worker = CHI_CUR_WORKER;

    // Client TCP/IPC path
    if (!is_runtime && ipc_mode_ != IpcMode::kShm) {
      return IpcCpu2CpuZmq::ClientSend(this, task_ptr, ipc_mode_);
    }

    // Client SHM path
    if (!is_runtime) {
      return IpcCpu2Cpu::ClientSend(this, task_ptr);
    }

    // Runtime self-send: enqueue task by pointer (no serialization)
    Future<Task> base_future =
        IpcCpu2Self::ClientSend(this, task_ptr.template Cast<Task>());
    return base_future.Cast<TaskT>();
  }


  /**
   * Receive a task on the runtime side: deserialize from Future if needed.
   * Handles origin-based dispatch (SHM, GPU2CPU, CPU2GPU, etc.).
   *
   * @param future Future containing the task
   * @param container Container for deserialization
   * @param method_id Method ID for task allocation
   * @param recv_transport SHM transport for receiving serialized data
   * @return FullPtr to the deserialized/retrieved task
   */
  hipc::FullPtr<Task> RecvRuntime(
      Future<Task> &future, Container *container, u32 method_id,
      hshm::lbm::Transport *recv_transport);

  /**
   * Send the runtime response back to the client after task execution.
   * Serializes outputs, completes futures, and handles cleanup for each
   * origin transport mode (SHM, TCP, IPC, GPU2CPU, CPU2GPU).
   *
   * @param task_ptr Executed task
   * @param run_ctx RunContext with future and execution state
   * @param container Container for serialization
   * @param send_transport SHM transport for sending serialized data
   */
  void SendRuntime(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                   Container *container,
                   hshm::lbm::Transport *send_transport);

  /**
   * Initialize RunContext for a task before routing.
   * @param future Future containing the task
   * @param container Container for the task (can be nullptr)
   * @param lane Lane for the task (can be nullptr)
   */
  void BeginTask(Future<Task> &future, Container *container, TaskLane *lane);

  /** Route a task: resolve pool query, determine local vs global.
   * If force_enqueue is true, always enqueue to the destination worker's lane
   * (used by EnqueueRuntime which cannot execute tasks directly). */
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
   * Receive task results on the client side.
   * Dispatches to the appropriate transport class based on origin.
   */
  template <typename TaskT>
  bool Recv(Future<TaskT> &future, float max_sec = 0) {
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    if (is_runtime) return true;

    auto future_shm = future.GetFutureShm();
    u32 origin = future_shm->origin_;

    if (origin == FutureShm::FUTURE_CLIENT_SHM && server_alive_.load()) {
      return IpcCpu2Cpu::ClientRecv(this, future, max_sec);
    }
    return IpcCpu2CpuZmq::ClientRecv(this, future, max_sec);
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
  hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
    // Full allocator lookup implementation
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
#if HSHM_IS_HOST && (HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM)
      if (gpu_ipc_) {
        auto git = gpu_ipc_->gpu_alloc_map_.find(alloc_key);
        if (git != gpu_ipc_->gpu_alloc_map_.end()) {
          size_t off = shm_ptr.off_.load();
          if (off < git->second.capacity) {
            result.ptr_ = reinterpret_cast<T *>(git->second.base + off);
            result.shm_ = shm_ptr;
          }
        }
      }
#endif
    }

    // Release the lock before returning
    allocator_map_lock_.ReadUnlock();

    return result;
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
  hipc::FullPtr<T> ToFullPtr(T *ptr) {
    // Full allocator lookup implementation
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
   * Delegates to gpu_ipc_ when CUDA/ROCm is enabled.
   */
  size_t GetGpuQueueCount() const {
#if HSHM_IS_HOST && (HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM)
    if (gpu_ipc_) return gpu_ipc_->gpu_devices_.size();
#endif
    return gpu_queues_.size();
  }

  /**
   * Get GPU→CPU queue by index (CPU worker polls this).
   * @param gpu_id GPU device ID (0-based)
   */
  GpuTaskQueue *GetGpuQueue(size_t gpu_id) {
#if HSHM_IS_HOST && (HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM)
    if (gpu_ipc_ && gpu_id < gpu_ipc_->gpu_devices_.size()) {
      return gpu_ipc_->gpu_devices_[gpu_id].gpu2cpu_queue.ptr_;
    }
#endif
    if (gpu_id < gpu_queues_.size()) return gpu_queues_[gpu_id].ptr_;
    return nullptr;
  }

  /**
   * Register a GPU queue (non-CUDA fallback path).
   * @param queue FullPtr to a TaskQueue in GPU-accessible shared memory
   */
  void RegisterGpuQueue(hipc::FullPtr<GpuTaskQueue> queue) {
    gpu_queues_.push_back(queue);
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /** Get the GPU IPC manager for direct access to GPU operations. */
  gpu::IpcManager *GetGpuIpcManager() { return gpu_ipc_.get(); }
  const gpu::IpcManager *GetGpuIpcManager() const { return gpu_ipc_.get(); }
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
   * Register GPU accelerator memory backend
   *
   * Called from GPU kernels to store GPU memory backend reference.
   * Per-thread BuddyAllocators are initialized in CHIMAERA_GPU_INIT macro.
   *
   * @param backend GPU memory backend to register
   * @return true on success, false on failure
   */
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

  /**
   * Allocate GPU buffer from the cpu2gpu copy backend.
   * @param size Number of bytes to allocate
   * @param gpu_id GPU device ID
   * @return FullPtr<char> to allocated GPU-accessible memory
   */
  hipc::FullPtr<char> AllocateGpuBuffer(size_t size, u32 gpu_id = 0);

 private:
  /**
   * Stop the GPU work orchestrator and free resources
   */
  void FinalizeGpuOrchestrator();

#endif

 public:
  /**
   * Thin forwarding methods for GPU orchestrator lifecycle.
   * Safe to call even when CUDA/ROCm is disabled (no-ops).
   */
  bool PauseGpuOrchestrator() {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    if (auto *g = GetGpuIpcManager()) return g->PauseGpuOrchestrator();
#endif
    return false;
  }
  void ResumeGpuOrchestrator() {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    if (auto *g = GetGpuIpcManager()) g->ResumeGpuOrchestrator();
#endif
  }
  void SetGpuOrchestratorBlocks(u32 blocks, u32 threads_per_block = 0) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    if (auto *g = GetGpuIpcManager()) g->SetGpuOrchestratorBlocks(blocks, threads_per_block);
#endif
  }
  void PrintGpuOrchestratorProfile() {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    if (auto *g = GetGpuIpcManager()) g->PrintGpuOrchestratorProfile();
#endif
  }

 private:
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
  std::vector<hipc::FullPtr<GpuTaskQueue>> gpu_queues_;

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
 public:
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


  /** Stored IpcManagerGpuInfo for GPU orchestrator launch */
  IpcManagerGpuInfo gpu_orchestrator_info_;

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /** GPU IPC manager: owns all host-side GPU infrastructure.
   *  CPU-side IpcManager delegates GPU operations through this. */
  std::unique_ptr<gpu::IpcManager> gpu_ipc_;
#endif

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

#define CHI_IPC HSHM_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)
#define CHI_CPU_IPC CHI_IPC

// Include local_task_archives after CHI_IPC is defined, since on GPU
// CHI_PRIV_ALLOC expands to chi::GetPrivAllocGpu() (defined below)
#include "chimaera/local_task_archives.h"

// ================================================================
// GPU translation unit support: override CHI_IPC for device code
// ================================================================
#if HSHM_IS_GPU_COMPILER

namespace chi {
namespace gpu {
HSHM_CROSS_FUN inline IpcManager *GetGpuIpcManager() {
#if HSHM_IS_GPU
  return IpcManager::GetBlockIpcManager();
#else
  return nullptr;
#endif
}
}  // namespace gpu
}  // namespace chi

#undef CHI_IPC
#define CHI_IPC (::chi::gpu::GetGpuIpcManager())
#undef CHI_CPU_IPC
#define CHI_CPU_IPC HSHM_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)

namespace chi {
// GPU allocator getters: accessible after CHI_IPC override above
#if !HSHM_IS_HOST
HSHM_GPU_FUN inline hipc::PrivateBuddyAllocator *GetPrivAllocGpu() {
  // WarpIpcManager removed; private alloc not available on GPU.
  // TODO(gpu-alloc): Provide a proper per-warp allocator if needed.
  return nullptr;
}
HSHM_GPU_FUN inline hipc::RoundRobinAllocator *GetSharedAllocGpu() {
  return CHI_IPC->gpu_alloc_;
}
#endif
}  // namespace chi

#endif  // HSHM_IS_GPU_COMPILER

// ================================================================
// Future method implementations (unified for CPU and GPU TUs)
// ================================================================
namespace chi {

// ~Future() - frees resources if consumed (via Wait/await_resume)
template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN Future<TaskT, AllocT>::~Future() {
  if (consumed_) {
    // Clean up zero-copy response archive (TCP/IPC only, never used on GPU)
    if (!future_shm_.IsNull()) {
#if HSHM_IS_HOST
      hipc::FullPtr<FutureShm> fs = CHI_CPU_IPC->ToFullPtr(future_shm_);
      if (!fs.IsNull() && (fs->origin_ == FutureShm::FUTURE_CLIENT_TCP ||
                           fs->origin_ == FutureShm::FUTURE_CLIENT_IPC)) {
        CHI_CPU_IPC->CleanupResponseArchive(fs->client_task_vaddr_);
      }
      // Free FutureShm (host only)
      hipc::ShmPtr<char> buffer_shm = future_shm_.template Cast<char>();
      CHI_CPU_IPC->FreeBuffer(buffer_shm);
      future_shm_.SetNull();
#endif
    }
    // Auto-free the task (only when consumed to avoid double-free
    // from runtime-internal Future copies in event queues / RunContext)
    DelTask();
  }
}

// GetFutureShm() - converts internal ShmPtr to FullPtr
template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN hipc::FullPtr<typename Future<TaskT, AllocT>::FutureT>
Future<TaskT, AllocT>::GetFutureShm() const {
  if (future_shm_.IsNull()) {
    return hipc::FullPtr<FutureT>();
  }
#if HSHM_IS_GPU
  return CHI_IPC->ToFullPtr(future_shm_);
#else
  return CHI_CPU_IPC->ToFullPtr(future_shm_);
#endif
}

// ----------------------------------------------------------------
// IsComplete variants
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN bool Future<TaskT, AllocT>::IsComplete() const {
  if (future_shm_.IsNull()) {
    return false;
  }
#if HSHM_IS_GPU
  return IsCompleteGpu2Gpu();
#else
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  if (future_shm_.alloc_id_ == FutureShm::GetCpu2GpuAllocId()) {
    return IsCompleteCpu2Gpu();
  }
#endif
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  bool is_gpu_future =
      future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
      (future_shm->client_task_vaddr_ == 0);
  if (is_gpu_future) {
    return IsCompleteGpu2Cpu();
  }
  return IsCompleteCpu2Cpu();
#endif
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteCpu2Cpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.Any(FutureShm::FUTURE_COMPLETE);
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteGpu2Cpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.AnySystem(FutureShm::FUTURE_COMPLETE);
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteCpu2Gpu() const {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // ShmPtr offset points to pinned-host gpu::FutureShm
  void *host_fshm = reinterpret_cast<void *>(future_shm_.off_.load());
  u32 flags_val = 0;
  size_t flags_offset = offsetof(gpu::FutureShm, flags_);
  hshm::GpuApi::Memcpy(
      &flags_val,
      reinterpret_cast<u32 *>(
          static_cast<char *>(host_fshm) + flags_offset),
      sizeof(u32));
  return (flags_val & gpu::FutureShm::FUTURE_COMPLETE) != 0;
#else
  return false;
#endif
}

#if HSHM_IS_GPU_COMPILER
template <typename TaskT, typename AllocT>
HSHM_GPU_FUN bool Future<TaskT, AllocT>::IsCompleteGpu2Gpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE);
}
#endif

// ----------------------------------------------------------------
// Wait dispatcher
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN bool Future<TaskT, AllocT>::Wait(float max_sec,
                                                 bool reuse_task) {
#if HSHM_IS_GPU
  return WaitGpu2Gpu(max_sec, reuse_task);
#else
  if (task_ptr_.IsNull() || future_shm_.IsNull()) {
    return true;
  }
  // Fire-and-forget: return immediately without waiting.
  if (task_ptr_->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
    task_ptr_.SetNull();
    future_shm_.SetNull();
    return true;
  }

  bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // CPU→GPU POD path: sentinel allocator ID marks device pointers.
  if (is_runtime &&
      future_shm_.alloc_id_ == FutureShm::GetCpu2GpuAllocId()) {
    return WaitCpu2Gpu(max_sec, reuse_task);
  }
#endif

  // Resolve FutureShm for non-GPU paths
  hipc::FullPtr<FutureShm> future_full = CHI_CPU_IPC->ToFullPtr(future_shm_);
  if (future_full.IsNull()) {
    HLOG(kError, "Future::Wait: ToFullPtr returned null");
    return false;
  }

  if (is_runtime) {
    return WaitCpu2Cpu(max_sec, reuse_task);
  }

  // Client path: detect GPU-originated futures
  bool is_gpu_future =
      future_full->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
      (future_full->client_task_vaddr_ == 0);
  if (is_gpu_future) {
    return WaitGpu2Cpu(max_sec, reuse_task);
  }
  return WaitCpu2Cpu(max_sec, reuse_task);
#endif
}

// ----------------------------------------------------------------
// GPU Wait paths (HSHM_GPU_FUN)
// ----------------------------------------------------------------

#if HSHM_IS_GPU_COMPILER
template <typename TaskT, typename AllocT>
HSHM_GPU_FUN bool Future<TaskT, AllocT>::WaitGpu2Gpu(float max_sec,
                                                      bool reuse_task) {
  // chi::Future should not be used for GPU-to-GPU paths.
  // Use gpu::Future::WaitGpu2Gpu instead.
  (void)max_sec; (void)reuse_task;
  return true;
}
#endif

// ----------------------------------------------------------------
// Host Wait paths (HSHM_HOST_FUN)
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::WaitCpu2Gpu(float max_sec,
                                                       bool reuse_task) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  bool ok = IpcCpu2Gpu::ClientRecv(*this, max_sec);
  if (reuse_task) task_ptr_.SetNull();
  Destroy(true);
  return ok;
#else
  (void)max_sec; (void)reuse_task;
  return true;
#endif
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::WaitCpu2Cpu(float max_sec,
                                                       bool reuse_task) {
  bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
  if (is_runtime) {
    // Runtime self-send: poll FUTURE_COMPLETE in SHM
    hipc::FullPtr<FutureShm> future_full =
        CHI_CPU_IPC->ToFullPtr(future_shm_);
    if (future_full.IsNull()) return false;
    bool ok = IpcCpu2Self::ClientRecv(*this, max_sec, future_full);
    if (!ok) return false;
  } else {
    // Client: SHM or ZMQ recv path
    if (!CHI_CPU_IPC->Recv(*this, max_sec)) {
      task_ptr_->SetReturnCode(static_cast<u32>(-1));
      return false;
    }
  }
  if (reuse_task) task_ptr_.SetNull();
  Destroy(true);
  return true;
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::WaitGpu2Cpu(float max_sec,
                                                       bool reuse_task) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // Host-side polling path (test harness): polls chi::FutureShm with
  // system-scope atomics. The GPU kernel uses IpcGpu2Cpu::ClientRecv
  // (device-side) which polls gpu::FutureShm instead.
  hipc::FullPtr<FutureShm> future_full =
      CHI_CPU_IPC->ToFullPtr(future_shm_);
  if (future_full.IsNull()) {
    HLOG(kError, "Future::WaitGpu2Cpu: ToFullPtr returned null");
    return false;
  }
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

  // Deserialize output from ring buffer if present
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

  if (reuse_task) task_ptr_.SetNull();
  Destroy(true);
  return true;
#else
  (void)max_sec; (void)reuse_task;
  return true;
#endif
}

// ----------------------------------------------------------------
// Shared helpers (HSHM_CROSS_FUN)
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::Destroy(bool post_wait) {
  if (post_wait && !task_ptr_.IsNull()) {
    task_ptr_->PostWait();
  }
  consumed_ = true;
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::DelTask() {
#if HSHM_IS_GPU
  if (!task_ptr_.IsNull()) {
    CHI_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
#else
  if (!task_ptr_.IsNull()) {
    CHI_CPU_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
#endif
}

// ----------------------------------------------------------------
// WaitPoll / WaitRecv (GPU-only, stubs on host)
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::WaitPoll(float max_sec,
                                                     bool reuse_task) {
  (void)max_sec; (void)reuse_task;
#if HSHM_IS_GPU
  u32 lane = gpu::IpcManager::GetLaneId();
  // Broadcast FutureShm pointer from lane 0
  unsigned long long fshm_ull = 0;
  if (lane == 0) {
    auto fshm_full = GetFutureShm();
    if (!fshm_full.IsNull())
      fshm_ull = reinterpret_cast<unsigned long long>(fshm_full.ptr_);
  }
  fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);
  if (fshm_ull == 0) return;
  auto *fshm = reinterpret_cast<FutureShm *>(fshm_ull);

  // Spin-wait on FUTURE_COMPLETE (device-scope atomics)
  while (!fshm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
    HSHM_THREAD_MODEL->Yield();
  }
  hipc::threadfence();
#endif
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::WaitRecv(float max_sec,
                                                     bool reuse_task) {
  (void)max_sec; (void)reuse_task;
#if HSHM_IS_GPU
  u32 lane = gpu::IpcManager::GetLaneId();
  // Broadcast FutureShm pointer from lane 0
  unsigned long long fshm_ull = 0;
  if (lane == 0) {
    auto fshm_full = GetFutureShm();
    if (!fshm_full.IsNull())
      fshm_ull = reinterpret_cast<unsigned long long>(fshm_full.ptr_);
  }
  fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);
  if (fshm_ull == 0) return;
  auto *fshm = reinterpret_cast<FutureShm *>(fshm_ull);

  // Read output if any was written
  unsigned long long task_ull = 0;
  int has_output = 0;
  if (lane == 0) {
    size_t output_written = fshm->output_.total_written_.load_device();
    if (output_written > 0) {
      task_ull = reinterpret_cast<unsigned long long>(task_ptr_.ptr_);
      has_output = 1;
    }
  }
  task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);
  has_output = __shfl_sync(0xFFFFFFFF, has_output, 0);

  if (lane == 0 && has_output) {
    size_t data_size = fshm->output_.total_written_.load_device();
    hipc::threadfence();

    // Inline deserialization: create local buffer + archive
    hipc::FullPtr<char> fp;
    fp.ptr_ = fshm->copy_space;
    fp.shm_.alloc_id_.SetNull();
    fp.shm_.off_ = reinterpret_cast<size_t>(fp.ptr_);
    hshm::priv::wrap_vector buffer;
    buffer.set(fp, data_size);
    buffer.resize(data_size);
    GpuLoadTaskArchive load_ar(buffer);
    load_ar.SetMsgType(LocalMsgType::kSerializeOut);
    task_ptr_.ptr_->SerializeOut(load_ar);
  }

  // Cleanup
  if (gpu::IpcManager::IsWarpScheduler()) {
    Destroy(true);
    if (reuse_task) {
      task_ptr_.SetNull();
    }
  }
#endif
}

// ================================================================
// gpu::Future method implementations
// ================================================================

namespace gpu {

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN Future<TaskT, AllocT>::~Future() {
  if (consumed_) {
    DelTask();
  }
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN hipc::FullPtr<typename Future<TaskT, AllocT>::FutureT>
Future<TaskT, AllocT>::GetFutureShm() const {
  if (future_shm_.IsNull()) {
    return hipc::FullPtr<FutureT>();
  }
#if HSHM_IS_GPU
  return CHI_IPC->ToFullPtr(future_shm_);
#else
  // Host stub — gpu::Future is not used for host-side resolution
  return hipc::FullPtr<FutureT>();
#endif
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN bool Future<TaskT, AllocT>::IsComplete() const {
  if (future_shm_.IsNull()) {
    return false;
  }
#if HSHM_IS_GPU
  return IsCompleteGpu2Gpu();
#else
  return false;
#endif
}

#if HSHM_IS_GPU_COMPILER
template <typename TaskT, typename AllocT>
HSHM_GPU_FUN bool Future<TaskT, AllocT>::IsCompleteGpu2Gpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE);
}
#endif

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteCpu2Gpu() const {
  return false;  // Host uses chi::Future::IsCompleteCpu2Gpu
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteGpu2Cpu() const {
  return false;  // Host uses chi::Future::IsCompleteGpu2Cpu
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN bool Future<TaskT, AllocT>::Wait(float max_sec,
                                                  bool reuse_task) {
#if HSHM_IS_GPU
  return WaitGpu2Gpu(max_sec, reuse_task);
#else
  (void)max_sec; (void)reuse_task;
  return true;
#endif
}

#if HSHM_IS_GPU_COMPILER
template <typename TaskT, typename AllocT>
HSHM_GPU_FUN bool Future<TaskT, AllocT>::WaitGpu2Gpu(float max_sec,
                                                      bool reuse_task) {
  (void)max_sec;
  CHI_IPC->Recv(*this, task_ptr_.ptr_);
  if (gpu::IpcManager::IsWarpScheduler()) {
    if (reuse_task) {
      task_ptr_.SetNull();
    }
  }
  return true;
}
#endif

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::WaitCpu2Gpu(float max_sec,
                                                       bool reuse_task) {
  (void)max_sec; (void)reuse_task;
  return true;  // Host uses chi::Future::WaitCpu2Gpu
}

template <typename TaskT, typename AllocT>
HSHM_HOST_FUN bool Future<TaskT, AllocT>::WaitGpu2Cpu(float max_sec,
                                                       bool reuse_task) {
  (void)max_sec; (void)reuse_task;
  return true;  // Host uses chi::Future::WaitGpu2Cpu
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::WaitPoll(float max_sec,
                                                     bool reuse_task) {
  (void)max_sec; (void)reuse_task;
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::WaitRecv(float max_sec,
                                                     bool reuse_task) {
  (void)max_sec; (void)reuse_task;
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::Destroy(bool post_wait) {
  if (post_wait && !task_ptr_.IsNull()) {
    task_ptr_->PostWait();
  }
  consumed_ = true;
}

template <typename TaskT, typename AllocT>
HSHM_CROSS_FUN void Future<TaskT, AllocT>::DelTask() {
#if HSHM_IS_GPU
  if (!task_ptr_.IsNull()) {
    CHI_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
#else
  if (!task_ptr_.IsNull()) {
    task_ptr_.SetNull();
  }
#endif
}

}  // namespace gpu

}  // namespace chi

// Template implementations for transport classes (need full IpcManager definition)
#include "chimaera/ipc/ipc_cpu2cpu_impl.h"
#include "chimaera/ipc/ipc_cpu2cpu_zmq_impl.h"
#include "chimaera/ipc/ipc_cpu2gpu_impl.h"

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_