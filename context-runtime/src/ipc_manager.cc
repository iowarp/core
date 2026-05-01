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
 * IPC manager implementation
 */

#include "chimaera/ipc_manager.h"

#ifndef _WIN32
#include <arpa/inet.h>
#include <dirent.h>
#include <endian.h>
#include <netdb.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#include <zmq.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>

#include "chimaera/admin.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/chimaera_manager.h"
#include "chimaera/config_manager.h"
#include "chimaera/container.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/pool_manager.h"
#include "chimaera/scheduler/scheduler_factory.h"
#include "chimaera/task_archives.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include "chimaera/gpu/work_orchestrator.h"
#include <hermes_shm/util/gpu_api.h>
#endif

// Global pointer variable definition for IPC manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::IpcManager, g_ipc_manager);

namespace chi {

// Host struct methods

// IpcManager methods

// Constructor and destructor removed - handled by HSHM singleton pattern

bool IpcManager::ClientInit() {
  HLOG(kDebug, "IpcManager::ClientInit");
  if (is_initialized_) {
    return true;
  }

  // Parse CHI_IPC_MODE environment variable (default: TCP)
  const char *ipc_mode_env = std::getenv("CHI_IPC_MODE");
  if (ipc_mode_env != nullptr) {
    std::string mode_str(ipc_mode_env);
    if (mode_str == "SHM" || mode_str == "shm") {
      ipc_mode_ = IpcMode::kShm;
    } else if (mode_str == "IPC" || mode_str == "ipc") {
      ipc_mode_ = IpcMode::kIpc;
    } else {
      ipc_mode_ = IpcMode::kTcp;  // Default
    }
  }
  HLOG(kInfo, "IpcManager::ClientInit: IPC mode = {}",
       ipc_mode_ == IpcMode::kShm   ? "SHM"
       : ipc_mode_ == IpcMode::kIpc ? "IPC"
                                    : "TCP");

  // Parse retry timeout environment variable
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  const char *retry_env = std::getenv("CHI_CLIENT_RETRY_TIMEOUT");
  if (retry_env) {
    client_retry_timeout_ = static_cast<float>(std::atof(retry_env));
  }
  HLOG(kInfo, "IpcManager::ClientInit: retry_timeout = {}s",
       client_retry_timeout_);

  // Parse CHI_CLIENT_TRY_NEW_SERVERS environment variable
  const char *try_new_env = std::getenv("CHI_CLIENT_TRY_NEW_SERVERS");
  if (try_new_env) {
    client_try_new_servers_ = std::atoi(try_new_env);
  }
  HLOG(kInfo, "IpcManager::ClientInit: try_new_servers = {}",
       client_try_new_servers_);

  // Load hostfile so Phase 2 failover has hosts to try
  if (client_try_new_servers_ > 0) {
    if (LoadHostfile()) {
      HLOG(kInfo, "IpcManager::ClientInit: Loaded {} hosts from hostfile",
           hostfile_map_.size());
    } else {
      HLOG(kWarning, "IpcManager::ClientInit: Failed to load hostfile, "
           "Phase 2 failover will be disabled");
    }
  }

  // Create lightbeam transport for client-server communication
  {
    auto *config = CHI_CONFIG_MANAGER;
    u32 port = config->GetPort();

    if (ipc_mode_ == IpcMode::kIpc) {
      // IPC mode: Unix domain socket transport
      std::string ipc_path =
          hshm::SystemInfo::GetMemfdPath("chimaera_" + std::to_string(port) + ".ipc");
      try {
        zmq_transport_ = hshm::lbm::TransportFactory::Get(
            ipc_path, hshm::lbm::TransportType::kSocket,
            hshm::lbm::TransportMode::kClient, "ipc", 0);
        HLOG(kInfo, "IpcManager: IPC transport connected to {}", ipc_path);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to create IPC transport: {}",
             e.what());
        return false;
      }
    } else {
      // TCP mode: ZMQ DEALER transport
      try {
        zmq_transport_ = hshm::lbm::TransportFactory::Get(
            config->GetServerAddr(), hshm::lbm::TransportType::kZeroMq,
            hshm::lbm::TransportMode::kClient, "tcp", port + 3);
        HLOG(kInfo, "IpcManager: DEALER transport connected to port {}",
             port + 3);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to create DEALER transport: {}",
             e.what());
        return false;
      }
    }

    zmq_recv_running_.store(true);
    zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });
  }

  // Initialize HSHM TLS key for task counter before calling WaitForLocalServer,
  // which calls CreateTaskId(). Without the key registered first, GetTls() on
  // the zero-initialized key may return a stale/freed pointer → crash.
  HSHM_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);
  auto *tls_counter = new TaskCounter();
  HSHM_THREAD_MODEL->SetTls(chi_task_counter_key_, tls_counter);

  // Wait for local server using lightbeam transport
  if (!WaitForLocalServer()) {
    HLOG(kError, "CRITICAL ERROR: Cannot connect to local server.");
    HLOG(kError, "Client initialization failed. Exiting.");
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) zmq_recv_thread_.join();
    zmq_transport_.reset();
    return false;
  }

  // Start heartbeat thread for server liveness detection
  server_alive_.store(true);
  heartbeat_running_.store(true);
  heartbeat_thread_ = std::thread([this]() { HeartbeatThread(); });

  // Create TLS key for current worker if not already created.
  // Must happen before any CoRwLock/CoMutex operations (e.g. IncreaseClientShm).
  // Server mode creates it earlier in WorkOrchestrator::Init.
  if (!chi_cur_worker_key_created_) {
    HSHM_THREAD_MODEL->CreateTls<Worker>(chi_cur_worker_key_, nullptr);
    chi_cur_worker_key_created_ = true;
  }
  HSHM_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<Worker *>(nullptr));

  // SHM mode: Attach to main SHM segment and initialize queues
  if (ipc_mode_ == IpcMode::kShm) {
    if (!ClientInitShm()) {
      return false;
    }
    if (!ClientInitQueues()) {
      return false;
    }

    // Create per-process shared memory for client allocations
    auto *config = CHI_CONFIG_MANAGER;
    size_t initial_size =
        config && config->IsValid()
            ? config->GetMemorySegmentSize(kClientDataSegment)
            : hshm::Unit<size_t>::Megabytes(256);  // Default 256MB
    if (!IncreaseClientShm(initial_size)) {
      HLOG(
          kError,
          "IpcManager::ClientInit: Failed to create per-process shared memory");
      return false;
    }

    // Create SHM lightbeam transports for client-side transport
    shm_send_transport_ = hshm::lbm::TransportFactory::Get(
        "", hshm::lbm::TransportType::kShm, hshm::lbm::TransportMode::kClient);
    shm_recv_transport_ = hshm::lbm::TransportFactory::Get(
        "", hshm::lbm::TransportType::kShm, hshm::lbm::TransportMode::kServer);
  }

  // Default host until identified
  this_host_ = Host();

  // Task counter TLS key was already created before WaitForLocalServer (above).
  // Do NOT create it again here — doing so leaks the previous pthread key and
  // causes all TLS operations to collide on key 0.

  // Create scheduler using factory
  auto *config = CHI_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    std::string sched_name = config->GetLocalSched();
    scheduler_ = SchedulerFactory::Get(sched_name);
    HLOG(kDebug, "Scheduler initialized: {}", sched_name);
  }

  is_initialized_ = true;
  return true;
}

bool IpcManager::ServerInit() {
  if (is_initialized_) {
    return true;
  }

  // Create chi_cur_worker_key_ TLS key early in server path.
  // ServerInitGpuQueues() calls RegisterGpuAllocator() which acquires a
  // CoRwLock, which calls GetCurrentLockOwnerId() → pthread_getspecific().
  // Without a valid TLS key, pthread_getspecific(0) returns a garbage pointer
  // that crashes on dereference.  WorkOrchestrator::Init() normally creates
  // this key, but it runs after ServerInit(), so we create it here first.
  if (!chi_cur_worker_key_created_) {
    HSHM_THREAD_MODEL->CreateTls<Worker>(chi_cur_worker_key_, nullptr);
    chi_cur_worker_key_created_ = true;
  }
  HSHM_THREAD_MODEL->SetTls(chi_cur_worker_key_, static_cast<Worker *>(nullptr));

  // Clear leftover shared memory segments from previous runs
  ClearUserIpcs();

  // Initialize memory segments for server
  if (!ServerInitShm()) {
    return false;
  }

  // Initialize priority queues
  if (!ServerInitQueues()) {
    return false;
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // Initialize GPU queues (one ring buffer per GPU)
  // GPU orchestrator launch is deferred to after all initial pools are created
  // (see ChimaeraManager::ServerInit) to avoid cudaMalloc deadlocks
  // during GPU container allocation.
  if (!ServerInitGpuQueues()) {
    return false;
  }
#endif

  // Identify this host
  if (!IdentifyThisHost()) {
    HLOG(kError, "Warning: Could not identify host, using default node ID");
    this_host_ = Host();  // Default constructor gives node_id = 0
  } else {
    HLOG(kDebug, "Node ID identified: 0x{:x}", this_host_.node_id);
  }

  // Initialize HSHM TLS key for task counter (needed for CreateTaskId in
  // runtime)
  HSHM_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);

  // Create scheduler using factory
  auto *config = CHI_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    std::string sched_name = config->GetLocalSched();
    scheduler_ = SchedulerFactory::Get(sched_name);
    HLOG(kDebug, "Scheduler initialized: {}", sched_name);
  }

  // Create lightbeam transports for client task reception
  {
    u32 port = config->GetPort();

    try {
      // TCP ROUTER server on port+3
      client_tcp_transport_ = hshm::lbm::TransportFactory::Get(
          "0.0.0.0", hshm::lbm::TransportType::kZeroMq,
          hshm::lbm::TransportMode::kServer, "tcp", port + 3);
      HLOG(kInfo, "IpcManager: TCP ROUTER transport bound on port {}",
           port + 3);
    } catch (const std::exception &e) {
      HLOG(kError, "IpcManager::ServerInit: Failed to bind TCP server: {}",
           e.what());
    }

    try {
      // IPC server on Unix domain socket
      std::string ipc_path =
          hshm::SystemInfo::GetMemfdPath("chimaera_" + std::to_string(port) + ".ipc");
      client_ipc_transport_ = hshm::lbm::TransportFactory::Get(
          ipc_path, hshm::lbm::TransportType::kSocket,
          hshm::lbm::TransportMode::kServer, "ipc", 0);
      HLOG(kInfo, "IpcManager: IPC lightbeam server bound on {}", ipc_path);
    } catch (const std::exception &e) {
      HLOG(kError, "IpcManager::ServerInit: Failed to bind IPC server: {}",
           e.what());
    }
  }

  is_initialized_ = true;
  return true;
}

void IpcManager::ClientFinalize() {
  // Clean up thread-local task counter
  TaskCounter *counter =
      HSHM_THREAD_MODEL->GetTls<TaskCounter>(chi_task_counter_key_);
  if (counter) {
    delete counter;
    HSHM_THREAD_MODEL->SetTls(chi_task_counter_key_,
                              static_cast<TaskCounter *>(nullptr));
  }

  // Stop heartbeat thread
  if (heartbeat_running_.load()) {
    heartbeat_running_.store(false);
    if (heartbeat_thread_.joinable()) {
      heartbeat_thread_.join();
    }
  }

  // Stop recv thread
  if (zmq_recv_running_.load()) {
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) {
      zmq_recv_thread_.join();
    }
  }

  // Clean up lightbeam transport objects
  zmq_transport_.reset();

  // Clients should not destroy shared resources
}

void IpcManager::ServerFinalize() {
  if (!is_initialized_) {
    return;
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // Finalize GPU orchestrator before cleaning up GPU resources
  FinalizeGpuOrchestrator();
#endif

  // Close persistent outbound DEALER sockets before resetting transports
  ClearClientPool();

  // Cleanup servers
  local_transport_.reset();
  main_transport_.reset();

  // Clean up lightbeam client transport objects
  client_tcp_transport_.reset();
  client_ipc_transport_.reset();

  // Clear main allocator pointer
  main_allocator_ = nullptr;

  is_initialized_ = false;
}

// Template methods (NewTask, DelTask, AllocateBuffer, Enqueue) are implemented
// inline in the header

TaskQueue *IpcManager::GetTaskQueue() { return worker_queues_.ptr_; }

bool IpcManager::IsInitialized() const { return is_initialized_; }

u32 IpcManager::GetWorkerCount() {
  return num_workers_;
}

u32 IpcManager::GetNumSchedQueues() const {
  return num_sched_queues_;
}

void IpcManager::SetNumSchedQueues(u32 num_sched_queues) {
  num_sched_queues_ = num_sched_queues;
  HLOG(kInfo, "IpcManager: Updated num_sched_queues to {}", num_sched_queues);
}

void IpcManager::AwakenWorker(TaskLane *lane) {
  if (!lane) {
    HLOG(kWarning, "AwakenWorker: lane is null");
    return;
  }

  // Always send signal to ensure worker wakes up
  // The worker may transition from active->inactive between our check and
  // signal send Sending signal when already active is safe - it's a no-op if
  // worker is processing
  pid_t tid = lane->GetTid();
  if (tid > 0) {
    pid_t runtime_pid = runtime_pid_ ? runtime_pid_ : getpid();

    // Send SIGUSR1 to the worker thread in the runtime process
    int result = hshm::lbm::EventManager::Signal(runtime_pid, tid);
    if (result != 0) {
      HLOG(kError,
           "AwakenWorker: Failed to send SIGUSR1 to runtime_pid={}, tid={} "
           "(active={}) - errno={}",
           runtime_pid, tid, lane->IsActive(), errno);
    }
  } else {
    HLOG(kWarning, "AwakenWorker: tid={} (invalid), cannot send signal", tid);
  }
}

bool IpcManager::ServerInitShm() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Set allocator ID for main segment
    main_allocator_id_ = hipc::AllocatorId::Get(1, 0);

    // Get configurable segment name
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);

    // Use calculated or explicit main_segment_size
    size_t main_segment_size = config->CalculateMainSegmentSize();

    HLOG(kInfo, "Initializing main shared memory segment: {} bytes ({} MB)",
         main_segment_size, main_segment_size / (1024 * 1024));

    // Initialize main backend with custom header size
    if (!main_backend_.shm_init(main_allocator_id_,
                                hshm::Unit<size_t>::Bytes(main_segment_size),
                                main_segment_name)) {
      return false;
    }

    // Create main allocator (CHI_TASK_ALLOC_T = BuddyAllocator) for task data
    main_allocator_ = main_backend_.MakeAlloc<CHI_TASK_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    // Initialize queue segment (CHI_QUEUE_ALLOC_T = ArenaAllocator) for TaskQueues
    queue_allocator_id_ = hipc::AllocatorId::Get(2, 0);
    std::string queue_segment_name =
        config->GetSharedMemorySegmentName(kQueueSegment);
    size_t queue_segment_size = config->CalculateQueueSegmentSize();
    HLOG(kInfo, "Initializing queue shared memory segment: {} bytes ({} KB)",
         queue_segment_size, queue_segment_size / 1024);
    if (!queue_backend_.shm_init(queue_allocator_id_,
                                 hshm::Unit<size_t>::Bytes(queue_segment_size),
                                 queue_segment_name)) {
      return false;
    }
    queue_allocator_ = queue_backend_.MakeAlloc<CHI_QUEUE_ALLOC_T>();
    if (!queue_allocator_) {
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ClientInitShm() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Set allocator IDs (must match server)
    main_allocator_id_ = hipc::AllocatorId(1, 0);
    queue_allocator_id_ = hipc::AllocatorId(2, 0);

    // Get configurable segment names with environment variable expansion
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);
    std::string queue_segment_name =
        config->GetSharedMemorySegmentName(kQueueSegment);

    // Attach to existing main shared memory segment created by server
    if (!main_backend_.shm_attach(main_segment_name)) {
      return false;
    }

    // Attach to main allocator (CHI_TASK_ALLOC_T = BuddyAllocator)
    main_allocator_ = main_backend_.AttachAlloc<CHI_TASK_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    // Attach to queue segment (CHI_QUEUE_ALLOC_T = ArenaAllocator)
    if (!queue_backend_.shm_attach(queue_segment_name)) {
      return false;
    }
    queue_allocator_ = queue_backend_.AttachAlloc<CHI_QUEUE_ALLOC_T>();
    if (!queue_allocator_) {
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ServerInitQueues() {
  if (!queue_allocator_) {
    return false;
  }

  try {
    // Initialize runtime metadata
    runtime_pid_ = getpid();
    server_generation_.store(
        static_cast<u64>(
            std::chrono::steady_clock::now().time_since_epoch().count()),
        std::memory_order_release);

    // Get worker counts from ConfigManager
    ConfigManager *config = CHI_CONFIG_MANAGER;
    u32 thread_count = config->GetNumThreads();
    // Note: Last worker serves dual roles as both task worker and network
    // worker
    u32 total_workers = thread_count;

    // Store worker count and scheduling queue count
    num_workers_ = total_workers;
    num_sched_queues_ = thread_count;

    // Get configured queue depth (no longer hardcoded)
    u32 queue_depth = config->GetQueueDepth();

    HLOG(kInfo,
         "Initializing {} worker queues with depth {} (last worker serves dual "
         "role)",
         total_workers, queue_depth);

    // Allocate TaskQueue in queue segment (CHI_QUEUE_ALLOC_T = ArenaAllocator)
    worker_queues_ = queue_allocator_->NewObj<TaskQueue>(
        queue_allocator_,
        total_workers,  // num_lanes equals total worker count
        2,  // num_priorities (2 priorities: 0=normal, 1=resumed tasks)
        queue_depth);  // Use configured depth instead of hardcoded 1024
    worker_queues_off_ = worker_queues_.shm_.off_.load();

    // Initialize network queue for send operations
    // One lane with four priorities (SendIn, SendOut, ClientSendTcp,
    // ClientSendIpc)
    net_queue_ = queue_allocator_->NewObj<NetQueue>(
        queue_allocator_,
        1,             // num_lanes: single lane for network operations
        4,             // num_priorities: 0=SendIn, 1=SendOut, 2=ClientSendTcp,
                       // 3=ClientSendIpc
        queue_depth);  // Use configured depth instead of hardcoded 1024

    return !worker_queues_.IsNull() && !net_queue_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

void IpcManager::AssignGpuLanesToWorker() {
  size_t num_gpus = GetGpuQueueCount();
  if (num_gpus == 0 || !scheduler_) return;
  Worker *gpu_worker = scheduler_->GetGpuWorker();
  if (!gpu_worker) return;
  std::vector<GpuTaskLane *> gpu_lanes;
  gpu_lanes.reserve(num_gpus);
  for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    GpuTaskQueue *gpu_queue = GetGpuQueue(gpu_id);
    if (gpu_queue) {
      GpuTaskLane *gpu_lane = &gpu_queue->GetLane(0, 0);
      gpu_lanes.push_back(gpu_lane);
      gpu_lane->SetAssignedWorkerId(gpu_worker->GetId());
    }
  }
  gpu_worker->SetGpuLanes(gpu_lanes);

  // Wake the GPU worker in case it's sleeping in epoll_wait.
  // The worker may have entered sleep before gpu_lanes_ was set.
  TaskLane *lane = gpu_worker->GetLane();
  if (lane) {
    pid_t tid = lane->GetTid();
    if (tid > 0) {
      hshm::lbm::EventManager::Signal(getpid(), tid);
    }
  }
}

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
bool IpcManager::ServerInitGpuQueues() {
  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    HLOG(kDebug, "No GPUs detected, skipping GPU queue initialization");
    return true;
  }

  HLOG(kInfo, "Initializing {} GPU(s) with new direction-separated backends",
       num_gpus);

  try {
    ConfigManager *config = CHI_CONFIG_MANAGER;
    u32 queue_depth = config->GetQueueDepth();

    // Initialize the GPU IPC manager if not already done.
    // gpu_ipc_ is a std::unique_ptr default-constructed to nullptr;
    // it must be created here before any gpu_ipc_->... access.
    if (!gpu_ipc_) {
      gpu_ipc_ = std::make_unique<gpu::IpcManager>();
    }

    // Resize gpu_devices_ to hold all GPUs
    gpu_ipc_->gpu_devices_.resize(num_gpus);

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      if (!InitGpuBackendsForDevice(gpu_id, queue_depth)) {
        HLOG(kError, "Failed to initialize GPU backends for device {}", gpu_id);
        return false;
      }
    }

    // Populate gpu_orchestrator_info_ for GPU 0
    if (num_gpus > 0) {
      BuildOrchestratorInfo(0, queue_depth);
    }

    return true;
  } catch (const std::exception &e) {
    HLOG(kError, "Exception during GPU queue initialization: {}", e.what());
    return false;
  }
}

/**
 * Initialize all backends and queues for a single GPU device.
 *
 * Creates five backends per GPU:
 *   1. gpu2gpu_queue_backend_: GpuMalloc (device) — GPU→GPU TaskQueue
 *   2. gpu2cpu_queue_backend_: GpuShmMmap (pinned) — GPU→CPU TaskQueue
 *   3. cpu2gpu_queue_backend_: GpuShmMmap (pinned) — CPU→GPU TaskQueue
 *   4. gpu2cpu_copy_backend_:  GpuShmMmap (pinned) — GPU→CPU FutureShm alloc
 *   5. cpu2gpu_copy_backend_:  GpuShmMmap (pinned) — CPU→GPU FutureShm alloc
 *   6. gpu_orchestrator_backend_: GpuShmMmap (pinned) — orchestrator scratch
 *
 * @param gpu_id Target GPU device index
 * @param queue_depth Ring buffer depth for TaskQueue objects
 * @return true on success
 */
bool IpcManager::InitGpuBackendsForDevice(int gpu_id, u32 queue_depth) {
  const std::string sid = std::to_string(gpu_id);
  ConfigManager *config = CHI_CONFIG_MANAGER;
  u32 gpu_blocks = config->GetGpuBlocks();
  u32 gpu_threads = config->GetGpuThreadsPerBlock();
  u32 num_lanes = (gpu_blocks * gpu_threads) / 32;  // One lane per warp

  // --- 1. GPU→GPU queue backend (device memory, GpuMalloc) ---
  // Device memory: CPU cannot directly dereference it. Use a GPU kernel
  // (gpu::InitQueueOnDevice) to initialize the ArenaAllocator and TaskQueue
  // on device, then retrieve the queue pointer via cudaMemcpy.
  {
    hipc::MemoryBackendId bid(3000 + gpu_id, 0);
    std::string url = "/chi_gpu2gpu_q_" + sid;
    auto backend = std::make_unique<hipc::GpuMalloc>();
    // Scale backend size with number of lanes
    size_t backend_size = std::max(
        hshm::Unit<size_t>::Megabytes(4),
        static_cast<size_t>(num_lanes) * queue_depth * 256 +
            hshm::Unit<size_t>::Megabytes(1));
    if (!backend->shm_init(bid, backend_size, url, gpu_id)) {
      HLOG(kError, "Failed to init gpu2gpu queue backend for GPU {}", gpu_id);
      return false;
    }
    // Initialize allocator + queue entirely on the GPU
    hipc::FullPtr<GpuTaskQueue> q = gpu::InitQueueOnDevice(
        backend->data_, backend->data_capacity_, num_lanes, queue_depth);
    if (q.IsNull()) {
      HLOG(kError, "Failed to init gpu2gpu TaskQueue on device for GPU {}", gpu_id);
      return false;
    }
    gpu_ipc_->RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
    gpu_ipc_->gpu_devices_[gpu_id].gpu2gpu_queue = q;
    gpu_ipc_->gpu_devices_[gpu_id].gpu2gpu_queue_backend = std::move(backend);
    gpu_ipc_->gpu_orchestrator_info_.gpu2gpu_num_lanes = num_lanes;
  }

  // --- 2. GPU→CPU queue backend (pinned host, GpuShmMmap) ---
  {
    hipc::MemoryBackendId bid(4000 + gpu_id, 0);
    std::string url = "/chi_gpu2cpu_q_" + sid;
    auto backend = std::make_unique<hipc::GpuShmMmap>();
    if (!backend->shm_init(bid, hshm::Unit<size_t>::Megabytes(4), url,
                           gpu_id)) {
      HLOG(kError, "Failed to init gpu2cpu queue backend for GPU {}", gpu_id);
      return false;
    }
    auto *alloc = backend->template MakeAlloc<CHI_QUEUE_ALLOC_T>(
        backend->data_capacity_);
    if (!alloc) return false;
    hipc::FullPtr<GpuTaskQueue> q = alloc->template NewObj<GpuTaskQueue>(
        alloc, 1, 2, queue_depth);
    if (q.IsNull()) return false;
    gpu_ipc_->RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
    gpu_ipc_->gpu_devices_[gpu_id].gpu2cpu_queue = q;
    gpu_ipc_->gpu_devices_[gpu_id].gpu2cpu_queue_backend = std::move(backend);
  }

  // --- 3. CPU→GPU queue backend (pinned host, GpuShmMmap) ---
  {
    hipc::MemoryBackendId bid(5000 + gpu_id, 0);
    std::string url = "/chi_cpu2gpu_q_" + sid;
    auto backend = std::make_unique<hipc::GpuShmMmap>();
    if (!backend->shm_init(bid, hshm::Unit<size_t>::Megabytes(4), url,
                           gpu_id)) {
      HLOG(kError, "Failed to init cpu2gpu queue backend for GPU {}", gpu_id);
      return false;
    }
    auto *alloc = backend->template MakeAlloc<CHI_QUEUE_ALLOC_T>(
        backend->data_capacity_);
    if (!alloc) return false;
    hipc::FullPtr<GpuTaskQueue> q = alloc->template NewObj<GpuTaskQueue>(
        alloc, 1, 2, queue_depth);
    if (q.IsNull()) return false;
    gpu_ipc_->RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
    gpu_ipc_->gpu_devices_[gpu_id].cpu2gpu_queue = q;
    gpu_ipc_->gpu_devices_[gpu_id].cpu2gpu_queue_backend = std::move(backend);
  }

  // --- 4. GPU→CPU copy-space backend (pinned host, GpuShmMmap) ---
  {
    hipc::MemoryBackendId bid(6000 + gpu_id, 0);
    std::string url = "/chi_gpu2cpu_cp_" + sid;
    auto backend = std::make_unique<hipc::GpuShmMmap>();
    if (!backend->shm_init(bid, hshm::Unit<size_t>::Megabytes(32), url,
                           gpu_id)) {
      HLOG(kError, "Failed to init gpu2cpu copy backend for GPU {}", gpu_id);
      return false;
    }
    gpu_ipc_->RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
    gpu_ipc_->gpu_devices_[gpu_id].gpu2cpu_copy_backend = std::move(backend);
  }

  // --- 5. CPU→GPU copy-space backend (pinned host, GpuShmMmap) ---
  // AllocateGpuBuffer allocates FutureShm from this backend; the GPU
  // orchestrator resolves offsets relative to this base (cpu2gpu_queue_base).
  // MakeAlloc must be called here so AllocateGpuBuffer's AttachAlloc works.
  {
    hipc::MemoryBackendId bid(7000 + gpu_id, 0);
    std::string url = "/chi_cpu2gpu_cp_" + sid;
    auto backend = std::make_unique<hipc::GpuShmMmap>();
    if (!backend->shm_init(bid, hshm::Unit<size_t>::Megabytes(32), url,
                           gpu_id)) {
      HLOG(kError, "Failed to init cpu2gpu copy backend for GPU {}", gpu_id);
      return false;
    }
    // Initialize allocator so AllocateGpuBuffer can use AttachAlloc
    auto *alloc = backend->template MakeAlloc<CHI_QUEUE_ALLOC_T>(
        backend->data_capacity_);
    if (!alloc) {
      HLOG(kError, "Failed to init allocator for cpu2gpu copy backend GPU {}", gpu_id);
      return false;
    }
    gpu_ipc_->RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
    gpu_ipc_->gpu_devices_[gpu_id].cpu2gpu_copy_backend = std::move(backend);
  }

  // --- 6. Orchestrator scratch backend (pinned host, GpuShmMmap) ---
  // Size must scale with block count: each block's PartitionedAllocator partition
  // must have enough room for BuddyAllocator free-list over many iterations.
  // 4MB per block minimum to avoid fragmentation-induced deadlocks.
  {
    size_t scratch_per_block = hshm::Unit<size_t>::Megabytes(4);
    size_t scratch_total = std::max(
        hshm::Unit<size_t>::Megabytes(64),
        static_cast<size_t>(gpu_blocks) * scratch_per_block);
    hipc::MemoryBackendId bid(8000 + gpu_id, 0);
    std::string url = "/chi_gpu_orch_" + sid;
    auto backend = std::make_unique<hipc::GpuShmMmap>();
    if (!backend->shm_init(bid, scratch_total, url,
                           gpu_id)) {
      HLOG(kError, "Failed to init orchestrator backend for GPU {}", gpu_id);
      return false;
    }
    gpu_ipc_->gpu_devices_[gpu_id].gpu_orchestrator_backend = std::move(backend);
  }

  // --- 7. Internal subtask queue backend (device memory, GpuMalloc) ---
  // Separate queue for orchestrator subtasks to prevent deadlock with
  // client tasks on the gpu2gpu queue.
  {
    hipc::MemoryBackendId bid(10000 + gpu_id, 0);
    std::string url = "/chi_internal_q_" + sid;
    auto backend = std::make_unique<hipc::GpuMalloc>();
    size_t backend_size = std::max(
        hshm::Unit<size_t>::Megabytes(4),
        static_cast<size_t>(num_lanes) * queue_depth * 256 +
            hshm::Unit<size_t>::Megabytes(1));
    if (!backend->shm_init(bid, backend_size, url, gpu_id)) {
      HLOG(kError, "Failed to init internal queue backend for GPU {}", gpu_id);
      return false;
    }
    hipc::FullPtr<GpuTaskQueue> q = gpu::InitQueueOnDevice(
        backend->data_, backend->data_capacity_, num_lanes, queue_depth);
    if (q.IsNull()) {
      HLOG(kError, "Failed to init internal TaskQueue on device for GPU {}", gpu_id);
      return false;
    }
    gpu_ipc_->RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
    gpu_ipc_->gpu_devices_[gpu_id].internal_queue = q;
    gpu_ipc_->gpu_devices_[gpu_id].internal_queue_backend = std::move(backend);
  }

  HLOG(kInfo, "GPU {} backends initialized (queue_depth={})", gpu_id,
       queue_depth);

  // Initialize CPU→GPU send pools for this device
  gpu_ipc_->InitCpu2GpuSendPools(gpu_id);

  return true;
}

void gpu::IpcManager::InitCpu2GpuSendPools(u32 gpu_id,
                                             size_t task_pool_size,
                                             size_t fshm_pool_size) {
  auto &dev = gpu_devices_[gpu_id];
  (void)task_pool_size;  // Not used — pinned host pool serves both

  // Allocate a single pinned-host pool for [Task | FutureShm] pairs.
  // Pinned host is GPU-accessible via UVM — no cudaMemcpy needed.
  // The GPU orchestrator reads tasks directly from pinned host memory.
  dev.cpu2gpu_fshm_pool =
      hshm::GpuApi::MallocHost<char>(fshm_pool_size);
  if (!dev.cpu2gpu_fshm_pool) {
    HLOG(kError, "InitCpu2GpuSendPools: Failed to allocate {} bytes "
         "pinned host memory for GPU {}", fshm_pool_size, gpu_id);
    return;
  }
  memset(dev.cpu2gpu_fshm_pool, 0, fshm_pool_size);
  dev.cpu2gpu_fshm_pool_size = fshm_pool_size;
  dev.cpu2gpu_fshm_next = 0;

  HLOG(kInfo, "GPU {} CPU→GPU send pool initialized ({}MB pinned host)",
       gpu_id, fshm_pool_size / (1024*1024));
}

/**
 * Build gpu_orchestrator_info_ from the backends for a given GPU.
 *
 * @param gpu_id Target GPU device index
 * @param queue_depth Queue depth (stored in info struct)
 */
void IpcManager::BuildOrchestratorInfo(u32 gpu_id, u32 queue_depth) {
  gpu_ipc_->gpu_orchestrator_info_.backend =
      static_cast<hipc::MemoryBackend &>(*gpu_ipc_->gpu_devices_[gpu_id].gpu_orchestrator_backend);
  gpu_ipc_->gpu_orchestrator_info_.gpu2gpu_queue = gpu_ipc_->gpu_devices_[gpu_id].gpu2gpu_queue.ptr_;
  gpu_ipc_->gpu_orchestrator_info_.internal_queue = gpu_ipc_->gpu_devices_[gpu_id].internal_queue.ptr_;
  gpu_ipc_->gpu_orchestrator_info_.cpu2gpu_queue = gpu_ipc_->gpu_devices_[gpu_id].cpu2gpu_queue.ptr_;
  gpu_ipc_->gpu_orchestrator_info_.gpu2cpu_queue = gpu_ipc_->gpu_devices_[gpu_id].gpu2cpu_queue.ptr_;
  gpu_ipc_->gpu_orchestrator_info_.gpu2cpu_backend =
      static_cast<hipc::MemoryBackend &>(*gpu_ipc_->gpu_devices_[gpu_id].gpu2cpu_copy_backend);
  gpu_ipc_->gpu_orchestrator_info_.gpu_queue_depth = queue_depth;
}

bool IpcManager::LaunchGpuOrchestrator() {
  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    return true;  // No GPUs, nothing to do
  }

  ConfigManager *config = CHI_CONFIG_MANAGER;
  u32 blocks = config->GetGpuBlocks();
  u32 threads_per_block = config->GetGpuThreadsPerBlock();

  auto *orchestrator = new gpu::WorkOrchestrator();
  char *cpu2gpu_base = gpu_ipc_->gpu_devices_[0].cpu2gpu_copy_backend
                           ? gpu_ipc_->gpu_devices_[0].cpu2gpu_copy_backend->data_
                           : nullptr;
  if (!orchestrator->Launch(gpu_ipc_->gpu_orchestrator_info_, blocks,
                            threads_per_block, cpu2gpu_base)) {
    HLOG(kError, "Failed to launch GPU work orchestrator");
    delete orchestrator;
    return false;
  }

  gpu_ipc_->gpu_orchestrator_= orchestrator;
  return true;
}

void IpcManager::FinalizeGpuOrchestrator() {
  if (gpu_ipc_->gpu_orchestrator_) {
    auto *orchestrator = static_cast<gpu::WorkOrchestrator *>(gpu_ipc_->gpu_orchestrator_);
    orchestrator->Finalize();
    delete orchestrator;
    gpu_ipc_->gpu_orchestrator_= nullptr;
  }
}

hipc::FullPtr<char> IpcManager::AllocateGpuBuffer(size_t size, u32 gpu_id) {
  // Server path: use cpu2gpu COPY backend (7000+gpu_id, GpuShmMmap, 32MB).
  // The GPU orchestrator resolves FutureShm offsets relative to this backend's
  // base (cpu2gpu_queue_base_ = cpu2gpu_copy_backend->data_), so
  // FutureShm must be allocated here — not in the queue backend (5000).
  if (gpu_id < gpu_ipc_->gpu_devices_.size()) {
    auto *alloc = gpu_ipc_->gpu_devices_[gpu_id].cpu2gpu_copy_backend->template AttachAlloc<CHI_QUEUE_ALLOC_T>();
    if (alloc) {
      return alloc->AllocateObjs<char>(size);
    }
  }
  // Client path: use client_cpu2gpu backends (attached to server's GpuShmMmap)
  if (gpu_id < gpu_ipc_->gpu_devices_.size()) {
    auto *alloc = gpu_ipc_->gpu_devices_[gpu_id].client_cpu2gpu_backend->template AttachAlloc<CHI_QUEUE_ALLOC_T>();
    if (alloc) {
      return alloc->AllocateObjs<char>(size);
    }
  }
  return hipc::FullPtr<char>::GetNull();
}

// CUDA helpers removed — use hshm::GpuApi directly.

void gpu::IpcManager::RegisterGpuOrchestratorContainer(const PoolId &pool_id,
                                                    void *gpu_container_ptr) {
  if (!gpu_orchestrator_) {
    return;
  }
  auto *orchestrator = static_cast<gpu::WorkOrchestrator *>(gpu_orchestrator_);
  orchestrator->RegisterGpuContainer(pool_id, gpu_container_ptr);
}

void *IpcManager::AllocGpuContainer(const PoolId &pool_id, u32 container_id,
                                      const std::string &chimod_name) {
  if (!gpu_ipc_->gpu_orchestrator_) {
    return nullptr;
  }
  auto *orchestrator = static_cast<gpu::WorkOrchestrator *>(gpu_ipc_->gpu_orchestrator_);
  return orchestrator->AllocGpuContainer(pool_id, container_id, chimod_name);
}

#endif

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
void gpu::IpcManager::PrintGpuOrchestratorProfile() {
  if (!gpu_orchestrator_) return;
  auto *orch = static_cast<gpu::WorkOrchestrator *>(gpu_orchestrator_);
  if (!orch->control_) return;
  // Ensure GPU kernel has flushed profile data to pinned memory
  cudaDeviceSynchronize();
  auto *ctrl = orch->control_;
  for (int w = 0; w < gpu::WorkOrchestratorControl::kMaxDebugWorkers; ++w) {
    long long n = ctrl->prof_task_count[w];
    if (n == 0) continue;
    printf("\n--- Orchestrator Worker %d Profile (%lld tasks) ---\n", w, n);
    printf("  1. QueuePop+Resolve:     %lld  (%lld/task)\n", (long long)ctrl->prof_queue_pop[w], (long long)ctrl->prof_queue_pop[w]/n);
    printf("  2. RecvDevice (input):   %lld  (%lld/task)\n", (long long)ctrl->prof_recv_device[w], (long long)ctrl->prof_recv_device[w]/n);
    printf("  3. AllocTask:            %lld  (%lld/task)\n", (long long)ctrl->prof_alloc_task[w], (long long)ctrl->prof_alloc_task[w]/n);
    printf("  4. LoadTask (SerIn):     %lld  (%lld/task)\n", (long long)ctrl->prof_load_task[w], (long long)ctrl->prof_load_task[w]/n);
    printf("  5. AllocContext:         %lld  (%lld/task)\n", (long long)ctrl->prof_alloc_ctx[w], (long long)ctrl->prof_alloc_ctx[w]/n);
    printf("  6. CoroCreate:           %lld  (%lld/task)\n", (long long)ctrl->prof_coro_create[w], (long long)ctrl->prof_coro_create[w]/n);
    printf("  7. CoroResume (Run):     %lld  (%lld/task)\n", (long long)ctrl->prof_coro_resume[w], (long long)ctrl->prof_coro_resume[w]/n);
    printf("  8. CoroDestroy:          %lld  (%lld/task)\n", (long long)ctrl->prof_coro_destroy[w], (long long)ctrl->prof_coro_destroy[w]/n);
    printf("  9. SaveTask (SerOut):    %lld  (%lld/task)\n", (long long)ctrl->prof_save_task[w], (long long)ctrl->prof_save_task[w]/n);
    printf("  10. SendDevice (output): %lld  (%lld/task)\n", (long long)ctrl->prof_send_device[w], (long long)ctrl->prof_send_device[w]/n);
    printf("  11. Complete+Free:       %lld  (%lld/task)\n", (long long)ctrl->prof_complete[w], (long long)ctrl->prof_complete[w]/n);
    long long total = ctrl->prof_queue_pop[w] + ctrl->prof_recv_device[w] +
                      ctrl->prof_alloc_task[w] + ctrl->prof_load_task[w] +
                      ctrl->prof_alloc_ctx[w] + ctrl->prof_coro_create[w] +
                      ctrl->prof_coro_resume[w] + ctrl->prof_coro_destroy[w] +
                      ctrl->prof_save_task[w] + ctrl->prof_send_device[w] +
                      ctrl->prof_complete[w];
    printf("  TOTAL:                   %lld  (%lld/task)\n", total, total/n);
    // AllocContext sub-breakdown
    long long ac_alloc = (long long)ctrl->prof_ctx_alloc[w];
    long long ac_copy = (long long)ctrl->prof_ctx_copy[w];
    long long ac_zero = (long long)ctrl->prof_ctx_zero[w];
    printf("  AllocContext sub-breakdown:\n");
    printf("    alloc/cache:           %lld  (%lld/task)\n", ac_alloc, ac_alloc/n);
    printf("    struct copy:           %lld  (%lld/task)\n", ac_copy, ac_copy/n);
    printf("    zero coro handles:     %lld  (%lld/task)\n", ac_zero, ac_zero/n);
    // AllocTask sub-breakdown
    long long at_buddy = (long long)ctrl->prof_alloc_task_buddy[w];
    long long at_ctor = (long long)ctrl->prof_alloc_task_ctor[w];
    long long at_deser = (long long)ctrl->prof_alloc_task_deser[w];
    printf("  AllocTask sub-breakdown:\n");
    printf("    buddy alloc:           %lld  (%lld/task)\n", at_buddy, at_buddy/n);
    printf("    placement new:         %lld  (%lld/task)\n", at_ctor, at_ctor/n);
    printf("    deserialize:           %lld  (%lld/task)\n", at_deser, at_deser/n);
  }
}

bool gpu::IpcManager::PauseGpuOrchestrator() {
  if (!gpu_orchestrator_) {
    return false;
  }
  auto *orchestrator = static_cast<gpu::WorkOrchestrator *>(gpu_orchestrator_);
  if (!orchestrator->is_launched_) {
    return false;  // Already paused by someone else
  }
  orchestrator->Pause();

  // After the kernel is stopped, check if the lane count changed.
  // Recreate the gpu2gpu queue now so that GetClientGpuInfo() returns
  // correct pointers before the next kernel launch.
  u32 new_lanes = (orchestrator->blocks_ * orchestrator->threads_per_block_) / 32;
  if (new_lanes < 1) new_lanes = 1;
  if (new_lanes != gpu_orchestrator_info_.gpu2gpu_num_lanes && !gpu_devices_.empty()) {
    RebuildGpu2GpuQueue(0, new_lanes);
    if (!gpu_devices_.empty()) {
      RebuildInternalQueue(0, new_lanes);
    }
  }

  // Pre-allocate cross-warp resources (warp_group_queue, load arrays)
  // while no persistent GPU kernels are running.  This must happen now
  // because Resume() may be called while client kernels are active, and
  // cudaMalloc/cudaFree/InitQueueOnDevice synchronize with the default
  // stream, which would deadlock with a running persistent kernel.
  orchestrator->PrepareResume();

  return true;
}

void gpu::IpcManager::ResumeGpuOrchestrator() {
  if (!gpu_orchestrator_) {
    return;
  }
  auto *orchestrator = static_cast<gpu::WorkOrchestrator *>(gpu_orchestrator_);
  orchestrator->Resume(gpu_orchestrator_info_);
}

void gpu::IpcManager::SetGpuOrchestratorBlocks(u32 blocks, u32 threads_per_block) {
  if (!gpu_orchestrator_) {
    return;
  }
  auto *orchestrator = static_cast<gpu::WorkOrchestrator *>(gpu_orchestrator_);
  orchestrator->blocks_ = blocks;
  orchestrator->threads_per_block_ = threads_per_block;
}
void gpu::IpcManager::RebuildGpu2GpuQueue(u32 gpu_id, u32 new_lanes) {
  u32 queue_depth = gpu_orchestrator_info_.gpu_queue_depth;

  // Destroy old gpu2gpu queue backend (frees device memory)
  gpu_devices_[gpu_id].gpu2gpu_queue_backend.reset();
  gpu_devices_[gpu_id].gpu2gpu_queue = hipc::FullPtr<GpuTaskQueue>::GetNull();

  // Create new backend with enough space for the new lane count
  hipc::MemoryBackendId bid(3000 + gpu_id, 0);
  auto backend = std::make_unique<hipc::GpuMalloc>();
  size_t backend_size = std::max(
      hshm::Unit<size_t>::Megabytes(4),
      static_cast<size_t>(new_lanes) * queue_depth * 256 +
          hshm::Unit<size_t>::Megabytes(1));
  if (!backend->shm_init(bid, backend_size, "", gpu_id)) {
    HLOG(kError, "RebuildGpu2GpuQueue: Failed to create backend for {} lanes",
         new_lanes);
    return;
  }

  hipc::FullPtr<GpuTaskQueue> q = gpu::InitQueueOnDevice(
      backend->data_, backend->data_capacity_, new_lanes, queue_depth);
  if (q.IsNull()) {
    HLOG(kError, "RebuildGpu2GpuQueue: Failed to init queue with {} lanes",
         new_lanes);
    return;
  }

  RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
  gpu_devices_[gpu_id].gpu2gpu_queue = q;
  gpu_devices_[gpu_id].gpu2gpu_queue_backend = std::move(backend);
  gpu_orchestrator_info_.gpu2gpu_num_lanes = new_lanes;

  // Update orchestrator info with new queue pointers
  gpu_orchestrator_info_.gpu2gpu_queue = gpu_devices_[gpu_id].gpu2gpu_queue.ptr_;

  HLOG(kInfo, "RebuildGpu2GpuQueue: Recreated gpu2gpu queue with {} lanes "
       "(depth {})", new_lanes, queue_depth);
}

void gpu::IpcManager::RebuildInternalQueue(u32 gpu_id, u32 new_lanes) {
  u32 queue_depth = gpu_orchestrator_info_.gpu_queue_depth;

  // Destroy old internal queue backend
  gpu_devices_[gpu_id].internal_queue_backend.reset();
  gpu_devices_[gpu_id].internal_queue = hipc::FullPtr<GpuTaskQueue>::GetNull();

  // Create new backend
  hipc::MemoryBackendId bid(10000 + gpu_id, 0);
  auto backend = std::make_unique<hipc::GpuMalloc>();
  size_t backend_size = std::max(
      hshm::Unit<size_t>::Megabytes(4),
      static_cast<size_t>(new_lanes) * queue_depth * 256 +
          hshm::Unit<size_t>::Megabytes(1));
  if (!backend->shm_init(bid, backend_size, "", gpu_id)) {
    HLOG(kError, "RebuildInternalQueue: Failed to create backend for {} lanes",
         new_lanes);
    return;
  }

  hipc::FullPtr<GpuTaskQueue> q = gpu::InitQueueOnDevice(
      backend->data_, backend->data_capacity_, new_lanes, queue_depth);
  if (q.IsNull()) {
    HLOG(kError, "RebuildInternalQueue: Failed to init queue with {} lanes",
         new_lanes);
    return;
  }

  RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
  gpu_devices_[gpu_id].internal_queue = q;
  gpu_devices_[gpu_id].internal_queue_backend = std::move(backend);

  // Update orchestrator info with new queue pointers
  gpu_orchestrator_info_.internal_queue = gpu_devices_[gpu_id].internal_queue.ptr_;

  HLOG(kInfo, "RebuildInternalQueue: Recreated internal queue with {} lanes "
       "(depth {})", new_lanes, queue_depth);
}

void gpu::IpcManager::RegisterGpuAllocator(const hipc::MemoryBackendId &id,
                                       char *data, size_t capacity) {
  // Host-side map for ToFullPtr resolution on CPU
  u64 key = (static_cast<u64>(id.major_) << 32) |
            static_cast<u64>(id.minor_);
  gpu_alloc_map_[key] = GpuAllocInfo{
      hipc::AllocatorId{id.major_, id.minor_}, data, capacity};
}

IpcManagerGpuInfo gpu::IpcManager::CreateGpuAllocator(size_t gpu_memory_size,
                                                       u32 gpu_id) {
  (void)gpu_memory_size;
  // Share the orchestrator's scratch backend with client kernels.
  // Both the client and orchestrator use the same RoundRobinAllocator.
  return GetGpuInfo(gpu_id);
}
#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

bool IpcManager::ClientInitQueues() {
  if (!queue_allocator_) {
    return false;
  }

  try {
    // Reconstruct the worker_queues_ FullPtr from the SHM offset received
    // during WaitForLocalServer() (via ClientConnectTask::worker_queues_off_).
    // The offset is relative to queue_allocator_->GetBackendData().
    if (worker_queues_off_ == 0) {
      HLOG(kError, "ClientInitQueues: worker_queues_off_ not set "
           "(server did not send queue offset)");
      return false;
    }

    worker_queues_.shm_.off_ = worker_queues_off_;
    worker_queues_.shm_.alloc_id_ = queue_allocator_->GetId();
    worker_queues_.ptr_ = reinterpret_cast<TaskQueue *>(
        queue_allocator_->GetBackendData() + worker_queues_off_);

    return !worker_queues_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::StartLocalServer() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Start local ZeroMQ server using HSHM Lightbeam
    std::string addr = "127.0.0.1";
    std::string protocol = "tcp";
    u32 port = config->GetPort() + 1;  // Use ZMQ port + 1 for local server

    local_transport_ = hshm::lbm::TransportFactory::Get(
        addr, hshm::lbm::TransportType::kZeroMq,
        hshm::lbm::TransportMode::kServer, protocol, port);

    if (local_transport_ != nullptr) {
      HLOG(kSuccess, "Successfully started local server at {}:{}", addr, port);
      return true;
    }

    HLOG(kError, "Failed to start local server at {}:{}", addr, port);
    return false;
  } catch (const std::exception &e) {
    HLOG(kError, "Exception starting local server: {}", e.what());
    return false;
  }
}

bool IpcManager::WaitForLocalServer() {
  // Read environment variables for wait configuration
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  const char *wait_env = std::getenv("CHI_WAIT_SERVER");
  if (wait_env != nullptr) {
    wait_server_timeout_ = static_cast<float>(std::atof(wait_env));
  }

  HLOG(kInfo, "Waiting for runtime via lightbeam (timeout={}s)",
       wait_server_timeout_);

  // 0 = don't wait at all
  if (wait_server_timeout_ == 0) {
    HLOG(kError, "CHI_WAIT_SERVER=0: not waiting for runtime");
    return false;
  }

  // Send a ClientConnectTask via the lightbeam transport
  auto task = NewTask<chimaera::admin::ClientConnectTask>(
      CreateTaskId(), kAdminPoolId, PoolQuery::Local());
  auto future = IpcCpu2CpuZmq::ClientSend(this,task, ipc_mode_);

  // Wait for response with timeout (-1 → pass 0 to Wait which means no limit)
  float effective_timeout = wait_server_timeout_ > 0 ? wait_server_timeout_ : 0;
  if (!future.Wait(effective_timeout)) {
    HLOG(kError, "Timeout waiting for runtime after {} seconds",
         wait_server_timeout_);
    HLOG(kError, "This usually means:");
    HLOG(kError, "1. Chimaera runtime is not running");
    HLOG(kError, "2. Runtime failed to start");
    HLOG(kError, "3. Network connectivity issues");
    DelTask(task);
    return false;
  }

  if (task->response_ == 0) {
    client_generation_ = task->server_generation_;
    worker_queues_off_ = task->worker_queues_off_;
    if (task->server_pid_ > 0) {
      runtime_pid_ = static_cast<pid_t>(task->server_pid_);
    }
    HLOG(kInfo, "Successfully connected to runtime (generation={}, server_pid={})",
         client_generation_, runtime_pid_);

    // Initialize client GPU queues from server response
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    if (task->num_gpus_ > 0) {
      if (!gpu_ipc_) {
        gpu_ipc_ = std::make_unique<gpu::IpcManager>();
      }
      if (!gpu_ipc_->ClientInitGpuQueues(
              task->num_gpus_,
              task->cpu2gpu_queue_off_,
              task->gpu2cpu_queue_off_,
              task->gpu2gpu_queue_off_,
              task->cpu2gpu_backend_size_,
              task->gpu2cpu_backend_size_,
              task->gpu_queue_depth_,
              task->gpu2gpu_ipc_handle_bytes_)) {
        HLOG(kWarning, "Failed to initialize client GPU queues "
             "(GPU submission from this client will not work)");
      } else {
        HLOG(kInfo, "Client GPU queues initialized ({} GPUs)", task->num_gpus_);
      }
    }
#endif

    // Task cleanup is handled by ~Future() since Wait() marked it consumed.
    return true;
  }

  HLOG(kError, "Runtime responded with error code: {}", task->response_);
  // Task cleanup is handled by ~Future() since Wait() marked it consumed.
  return false;
}

bool IpcManager::WaitForLocalRuntimeStop(u32 timeout_sec) {
  HLOG(kInfo, "Waiting for runtime to stop (timeout={}s)", timeout_sec);

  // Temporarily disable reconnection so that Recv() returns false
  // immediately when the heartbeat detects the server is dead, instead
  // of blocking in WaitForServerAndReconnect for up to 60 seconds.
  float saved_retry = client_retry_timeout_;
  int saved_try_new = client_try_new_servers_;
  client_retry_timeout_ = 0;
  client_try_new_servers_ = 0;

  for (u32 elapsed = 0; elapsed < timeout_sec; ++elapsed) {
    // Send a ClientConnectTask with a 1-second timeout
    auto task = NewTask<chimaera::admin::ClientConnectTask>(
        CreateTaskId(), kAdminPoolId, PoolQuery::Local());
    auto future = IpcCpu2CpuZmq::ClientSend(this,task, ipc_mode_);

    if (!future.Wait(1.0f)) {
      // Timeout or server dead: runtime is no longer responding
      client_retry_timeout_ = saved_retry;
      client_try_new_servers_ = saved_try_new;
      HLOG(kInfo, "Runtime stopped (no response after {}s)", elapsed + 1);
      return true;
    }

    // Runtime still responded — it's still alive, keep waiting
    HLOG(kDebug, "Runtime still alive after {}s, retrying...", elapsed + 1);
  }

  client_retry_timeout_ = saved_retry;
  client_try_new_servers_ = saved_try_new;
  HLOG(kError, "Runtime still running after {}s timeout", timeout_sec);
  return false;
}

void IpcManager::SetNodeId(const std::string &hostname) {
  (void)hostname;  // Unused parameter
}

u64 IpcManager::GetNodeId() const {
  // Return the node ID from the identified host
  return this_host_.node_id;
}

bool IpcManager::LoadHostfile() {
  ConfigManager *config = CHI_CONFIG_MANAGER;
  std::string hostfile_path = config->GetHostfilePath();

  // Clear existing hostfile map
  hostfile_map_.clear();
  hosts_cache_valid_ = false;

  if (hostfile_path.empty()) {
    // No hostfile configured - assume localhost as node 0
    HLOG(kDebug, "No hostfile configured, using localhost as node 0");
    Host host(config->GetServerAddr(), 0);
    hostfile_map_[0] = host;
    return true;
  }

  try {
    // Use HSHM to parse hostfile
    std::vector<std::string> host_ips =
        hshm::ConfigParse::ParseHostfile(hostfile_path);

    // Create Host structs and populate map using linear offset-based node IDs
    HLOG(kDebug, "=== Container to Node ID Mapping (Linear Offset) ===");
    for (size_t offset = 0; offset < host_ips.size(); ++offset) {
      u64 node_id = static_cast<u64>(offset);
      Host host(host_ips[offset], node_id);
      hostfile_map_[node_id] = host;
      HLOG(kDebug, "  Hostfile[{}]: {} -> Node ID: {}", offset,
           host_ips[offset], node_id);
    }
    HLOG(kDebug, "=== Total hosts loaded: {} ===", hostfile_map_.size());
    if (hostfile_map_.empty()) {
      HLOG(kFatal, "There were no hosts in the hostfile {}", hostfile_path);
    }
    return true;

  } catch (const std::exception &e) {
    HLOG(kError, "Error loading hostfile {}: {}", hostfile_path, e.what());
    return false;
  }
}

const Host *IpcManager::GetHost(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) {
    // Log all available node IDs when lookup fails
    HLOG(kError,
         "GetHost: Looking for node_id {} but not found. Available nodes:",
         node_id);
    for (const auto &pair : hostfile_map_) {
      HLOG(kError, "  Node ID: {} -> IP: {}", pair.first,
           pair.second.ip_address);
    }
    return nullptr;
  }
  return &it->second;
}

const Host *IpcManager::GetHostByIp(const std::string &ip_address) const {
  // Search through hostfile_map_ for matching IP address
  for (const auto &pair : hostfile_map_) {
    if (pair.second.ip_address == ip_address) {
      return &pair.second;
    }
  }
  return nullptr;
}

const std::vector<Host> &IpcManager::GetAllHosts() const {
  // Rebuild cache if invalid
  if (!hosts_cache_valid_) {
    hosts_cache_.clear();
    hosts_cache_.reserve(hostfile_map_.size());

    for (const auto &pair : hostfile_map_) {
      hosts_cache_.push_back(pair.second);
    }

    hosts_cache_valid_ = true;
  }

  return hosts_cache_;
}

size_t IpcManager::GetNumHosts() const { return hostfile_map_.size(); }

bool IpcManager::IsAlive(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return false;
  return it->second.state == NodeState::kAlive;
}

void IpcManager::SetDead(u64 node_id) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  if (it->second.state == NodeState::kDead) return;  // Already dead

  SetNodeState(node_id, NodeState::kDead);

  // Record dead-node entry for retry tracking
  DeadNodeEntry entry;
  entry.node_id = node_id;
  entry.detected_at = std::chrono::steady_clock::now();
  dead_nodes_.push_back(entry);

  // Remove cached client connections to the dead node
  {
    std::lock_guard<std::mutex> lock(client_pool_mutex_);
    auto *config_manager = CHI_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    std::string key = it->second.ip_address + ":" + std::to_string(port);
    client_pool_.erase(key);
  }

  HLOG(kWarning, "IpcManager: Node {} ({}) marked as DEAD", node_id,
       it->second.ip_address);
}

void IpcManager::SetAlive(u64 node_id) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  if (it->second.state == NodeState::kAlive) return;  // Already alive

  SetNodeState(node_id, NodeState::kAlive);

  // Remove from dead_nodes_ list
  dead_nodes_.erase(std::remove_if(dead_nodes_.begin(), dead_nodes_.end(),
                                   [node_id](const DeadNodeEntry &e) {
                                     return e.node_id == node_id;
                                   }),
                    dead_nodes_.end());

  HLOG(kInfo, "IpcManager: Node {} ({}) marked as ALIVE", node_id,
       it->second.ip_address);
}

NodeState IpcManager::GetNodeState(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return NodeState::kDead;
  return it->second.state;
}

void IpcManager::SetNodeState(u64 node_id, NodeState new_state) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  it->second.state = new_state;
  it->second.state_changed_at = std::chrono::steady_clock::now();
  hosts_cache_valid_ = false;
}

void IpcManager::SetSelfFenced(bool fenced) { self_fenced_ = fenced; }

u64 IpcManager::GetLeaderNodeId() const {
  u64 leader = std::numeric_limits<u64>::max();
  for (const auto &[id, host] : hostfile_map_) {
    if (host.state == NodeState::kAlive && host.node_id < leader) {
      leader = host.node_id;
    }
  }
  return (leader == std::numeric_limits<u64>::max()) ? 0 : leader;
}

bool IpcManager::IsLeader() const { return GetNodeId() == GetLeaderNodeId(); }

u64 IpcManager::AddNode(const std::string &ip_address, u32 port) {
  (void)port;  // Port stored elsewhere (ConfigManager) for now

  // Check if node already exists
  for (const auto &pair : hostfile_map_) {
    if (pair.second.ip_address == ip_address) {
      HLOG(kInfo, "AddNode: Node {} already registered as node_id={}",
           ip_address, pair.first);
      SetAlive(pair.first);
      return pair.first;
    }
  }

  // Assign next node ID (linear offset)
  u64 new_node_id = static_cast<u64>(hostfile_map_.size());
  Host host(ip_address, new_node_id);
  hostfile_map_[new_node_id] = host;
  hosts_cache_valid_ = false;

  HLOG(kInfo, "AddNode: Registered {} as node_id={}", ip_address, new_node_id);
  return new_node_id;
}

bool IpcManager::IdentifyThisHost() {
  HLOG(kDebug, "Identifying current host");

  // Load hostfile if not already loaded
  if (hostfile_map_.empty()) {
    if (!LoadHostfile()) {
      HLOG(kError, "Error: Failed to load hostfile");
      return false;
    }
  }

  if (hostfile_map_.empty()) {
    HLOG(kError, "ERROR: No hosts available for identification");
    return false;
  }

  HLOG(kDebug, "Attempting to identify host among {} candidates",
       hostfile_map_.size());

  // Get port number for error reporting
  ConfigManager *config = CHI_CONFIG_MANAGER;
  u32 port = config->GetPort();

  // Collect list of attempted hosts for error reporting
  std::vector<std::string> attempted_hosts;

  // Resolve our local hostname so we can identify which hostfile entry
  // corresponds to this node *without* using bind-failure as the test.
  // Bind-failure-based identity used to work, but breaks on multi-rail
  // fabrics like Aurora's Slingshot HSN: binding to a specific FQDN
  // succeeds on whichever rail the FQDN resolves to, then peers
  // routing via the *other* rail get silently dropped (the listener
  // is on the wrong interface). Solution: identify by hostname match,
  // then bind the actual server on "0.0.0.0" so it listens on every
  // local interface (mirrors how `client_tcp_transport_` is bound).
  char local_host_buf[256] = {0};
  if (gethostname(local_host_buf, sizeof(local_host_buf) - 1) != 0) {
    HLOG(kError, "Error: gethostname() failed: {}", std::strerror(errno));
    return false;
  }
  std::string local_host(local_host_buf);
  std::string local_short =
      local_host.substr(0, local_host.find('.'));

  // Try to identify (by hostname match) and start the server.
  for (const auto &pair : hostfile_map_) {
    const Host &host = pair.second;
    attempted_hosts.push_back(host.ip_address);
    std::string entry_short =
        host.ip_address.substr(0, host.ip_address.find('.'));

    bool is_me = (host.ip_address == local_host) ||
                 (entry_short == local_short);
    if (!is_me) continue;

    HLOG(kDebug, "Hostfile entry {} matches local host {}; binding 0.0.0.0",
         host.ip_address, local_host);

    try {
      if (TryStartMainServer("0.0.0.0")) {
        HLOG(kInfo,
             "SUCCESS: Main server started on 0.0.0.0:{} "
             "(advertised as {}, node={})",
             port, host.ip_address, host.node_id);
        this_host_ = host;
        return true;
      }
    } catch (const std::exception &e) {
      HLOG(kDebug, "Failed to bind 0.0.0.0:{} for {}: {}",
           port, host.ip_address, e.what());
    } catch (...) {
      HLOG(kDebug, "Failed to bind 0.0.0.0:{} for {}: unknown error",
           port, host.ip_address);
    }
  }

  // Build detailed error message with hosts and port
  HLOG(kError, "ERROR: Could not start TCP server on any host from hostfile");
  HLOG(kError, "Port attempted: {}", port);
  HLOG(kError, "Hosts checked ({} total):", attempted_hosts.size());
  for (const auto &host_ip : attempted_hosts) {
    HLOG(kError, "  - {}", host_ip);
  }
  HLOG(kError, "");
  HLOG(
      kError,
      "This usually means another process is already running on the same port");
  HLOG(kError, "");
  HLOG(kError, "To check which process is using port {}, run:", port);
  HLOG(kError, "  Linux:   sudo lsof -i :{} -P -n", port);
  HLOG(kError, "           sudo netstat -tulpn | grep :{}", port);
  HLOG(kError, "  macOS:   sudo lsof -i :{} -P -n", port);
  HLOG(kError, "           sudo lsof -nP -iTCP:{} | grep LISTEN", port);
  HLOG(kError, "");
  HLOG(kError, "To stop the Chimaera runtime, run:");
  HLOG(kError, "  chimaera runtime stop");
  HLOG(kError, "");
  HLOG(kError, "Or kill the process directly:");
  HLOG(kError, "  pkill -9 chimaera");
  HLOG(kFatal, "  kill -9 <PID>");
  return false;
}

const std::string &IpcManager::GetCurrentHostname() const {
  return this_host_.ip_address;
}

bool IpcManager::TryStartMainServer(const std::string &hostname) {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Create main server using Lightbeam TransportFactory
    std::string protocol = "tcp";
    u32 port = config->GetPort();

    HLOG(kDebug, "Attempting to start main server on {}:{}", hostname, port);

    main_transport_ = hshm::lbm::TransportFactory::Get(
        hostname, hshm::lbm::TransportType::kZeroMq,
        hshm::lbm::TransportMode::kServer, protocol, port);

    if (!main_transport_) {
      HLOG(kDebug,
           "Failed to create main server on {}:{} - server creation returned "
           "null",
           hostname, port);
      return false;
    }

    HLOG(kDebug, "Main server successfully bound to {}:{}", hostname, port);

    return true;

  } catch (const std::exception &e) {
    HLOG(kDebug, "Failed to start main server on {}:{} - exception: {}",
         hostname, config->GetPort(), e.what());
    return false;
  } catch (...) {
    HLOG(kDebug, "Failed to start main server on {}:{} - unknown exception",
         hostname, config->GetPort());
    return false;
  }
}

hshm::lbm::Transport *IpcManager::GetMainTransport() const {
  return main_transport_.get();
}

hshm::lbm::Transport *IpcManager::GetClientTransport(IpcMode mode) const {
  if (mode == IpcMode::kTcp) return client_tcp_transport_.get();
  if (mode == IpcMode::kIpc) return client_ipc_transport_.get();
  return nullptr;
}

const Host &IpcManager::GetThisHost() const { return this_host_; }

FullPtr<char> IpcManager::AllocateBuffer(size_t size) {
#if HSHM_IS_HOST
  // HOST-ONLY PATH: The device implementation is in ipc_manager.h

  // RUNTIME PATH: Use private memory (HSHM_MALLOC) — runtime never uses
  // per-process shared memory segments
  if (CHI_CHIMAERA_MANAGER && CHI_CHIMAERA_MANAGER->IsRuntime()) {
    // Use HSHM_MALLOC allocator for private memory allocation
    FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      HLOG(kError, "AllocateBuffer: HSHM_MALLOC failed for {} bytes", size);
    }
    return buffer;
  }

  // CLIENT TCP/IPC PATH: Use private memory (no shared memory needed)
  if (ipc_mode_ != IpcMode::kShm) {
    FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      HLOG(kError,
           "AllocateBuffer: HSHM_MALLOC failed for {} bytes (client ZMQ mode)",
           size);
    }
    return buffer;
  }

  // CLIENT SHM PATH: Use per-process shared memory allocation strategy
  // 1. Check last accessed allocator first (fast path)
  if (last_alloc_ != nullptr) {
    FullPtr<char> buffer = last_alloc_->AllocateObjs<char>(size);
    if (!buffer.IsNull()) {
      return buffer;
    }
  }

  // 2. Check all allocators in alloc_vector_
  {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    for (auto *alloc : alloc_vector_) {
      if (alloc != nullptr && alloc != last_alloc_) {
        FullPtr<char> buffer = alloc->AllocateObjs<char>(size);
        if (!buffer.IsNull()) {
          last_alloc_ = alloc;  // Update last accessed
          return buffer;
        }
      }
    }
  }

  // 3. All existing allocators are full - create new shared memory segment
  // Calculate segment size: (requested_size + 32MB metadata) * 1.2 multiplier
  size_t new_size = static_cast<size_t>((size + kShmMetadataOverhead) *
                                        kShmAllocationMultiplier);
  if (!IncreaseClientShm(new_size)) {
    HLOG(kError, "AllocateBuffer: Failed to increase memory for {} bytes",
         size);
    return FullPtr<char>::GetNull();
  }

  // 4. Retry allocation from the newly created allocator (last_alloc_)
  if (last_alloc_ != nullptr) {
    FullPtr<char> buffer = last_alloc_->AllocateObjs<char>(size);
    if (!buffer.IsNull()) {
      return buffer;
    }
  }

  HLOG(kError,
       "AllocateBuffer: Failed to allocate {} bytes even after increasing "
       "memory",
       size);
  return FullPtr<char>::GetNull();
#else
  // GPU PATH: Implementation is in ipc_manager.h as inline function
  return FullPtr<char>::GetNull();
#endif  // HSHM_IS_HOST
}

void IpcManager::FreeBuffer(FullPtr<char> buffer_ptr) {
#if HSHM_IS_HOST
  // HOST PATH: Check various allocators
  if (buffer_ptr.IsNull()) {
    return;
  }

  // Check if allocator ID is null (private memory allocated with HSHM_MALLOC)
  if (buffer_ptr.shm_.alloc_id_ == hipc::AllocatorId::GetNull()) {
    // Private memory - use HSHM_MALLOC->Free() for RUNTIME-allocated buffers
    // In RUNTIME mode, AllocateBuffer uses HSHM_MALLOC which adds MallocPage
    // header
    HSHM_MALLOC->Free(buffer_ptr);
    return;
  }

  // Check main allocator
  if (main_allocator_ && buffer_ptr.shm_.alloc_id_ == main_allocator_id_) {
    main_allocator_->Free(buffer_ptr);
    return;
  }

  // Check per-process shared memory allocators via alloc_map_
  u64 alloc_key = (static_cast<u64>(buffer_ptr.shm_.alloc_id_.major_) << 32) |
                  static_cast<u64>(buffer_ptr.shm_.alloc_id_.minor_);
  auto it = alloc_map_.find(alloc_key);
  if (it != alloc_map_.end()) {
    it->second->Free(buffer_ptr);
    return;
  }

  // Check GPU backend registrations (e.g., data backends from benchmarks).
  // These are raw memory regions without a proper allocator — the GPU-side
  // allocator manages them. On the host we silently skip the free since the
  // memory is owned by the GPU backend and freed when it's destroyed.
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  if (gpu_ipc_) {
    auto git = gpu_ipc_->gpu_alloc_map_.find(alloc_key);
    if (git != gpu_ipc_->gpu_alloc_map_.end()) {
      return;
    }
  }
#endif

  HLOG(kWarning, "FreeBuffer: Could not find allocator for alloc_id ({}.{})",
       buffer_ptr.shm_.alloc_id_.major_, buffer_ptr.shm_.alloc_id_.minor_);
#else
  // GPU PATH: Implementation is in ipc_manager.h as inline function
#endif  // HSHM_IS_HOST
}

hshm::lbm::Transport *IpcManager::GetOrCreateClient(const std::string &addr,
                                                    int port) {
  // Create key for the pool map
  std::string key = addr + ":" + std::to_string(port);

  // Lock the pool for thread-safe access
  std::lock_guard<std::mutex> lock(client_pool_mutex_);

  // Check if client already exists
  auto it = client_pool_.find(key);
  if (it != client_pool_.end()) {
    HLOG(kDebug, "[ClientPool] Reusing existing connection to {}", key);
    return it->second.get();
  }

  // Create new persistent client connection
  HLOG(kInfo, "[ClientPool] Creating new persistent connection to {}", key);
  auto transport = hshm::lbm::TransportFactory::Get(
      addr, hshm::lbm::TransportType::kZeroMq,
      hshm::lbm::TransportMode::kClient, "tcp", port);

  if (!transport) {
    HLOG(kError, "[ClientPool] Failed to create client for {}", key);
    return nullptr;
  }

  // Store in pool and return raw pointer
  hshm::lbm::Transport *raw_ptr = transport.get();
  client_pool_[key] = std::move(transport);

  HLOG(kInfo, "[ClientPool] Connection established to {}", key);
  return raw_ptr;
}

void IpcManager::ClearClientPool() {
  std::lock_guard<std::mutex> lock(client_pool_mutex_);
  HLOG(kInfo, "[ClientPool] Clearing {} persistent connections",
       client_pool_.size());
  client_pool_.clear();
}

void IpcManager::EnqueueNetTask(Future<Task> future,
                                NetQueuePriority priority) {
  if (net_queue_.IsNull()) {
    HLOG(kError, "EnqueueNetTask: net_queue_ is null");
    return;
  }

  // Get lane 0 (single lane) with the specified priority
  u32 priority_idx = static_cast<u32>(priority);
  auto &lane = net_queue_->GetLane(0, priority_idx);
  bool was_empty = lane.Empty();
  lane.Push(future);

  // Signal the net worker if the lane was empty (same pattern as
  // admin_runtime.cc:1086-1089)
  if (was_empty && net_lane_) {
    AwakenWorker(net_lane_);
  }

  HLOG(kDebug, "EnqueueNetTask: priority={}, was_empty={}, net_lane={}",
       priority_idx, was_empty, net_lane_ != nullptr);
}

bool IpcManager::TryPopNetTask(NetQueuePriority priority,
                               Future<Task> &future) {
  if (net_queue_.IsNull()) {
    return false;
  }

  // Get lane 0 (single lane) with the specified priority
  u32 priority_idx = static_cast<u32>(priority);
  auto &lane = net_queue_->GetLane(0, priority_idx);

  if (lane.Pop(future)) {
    return true;
  }

  return false;
}

//==============================================================================
// Per-Process Shared Memory Management
//==============================================================================

bool IpcManager::IncreaseClientShm(size_t size) {
  HLOG(kDebug, "IncreaseClientShm CALLED: size={}", size);
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during memory increase
  // This ensures exclusive access to the allocator_map_ structures
  allocator_map_lock_.WriteLock();

  pid_t pid = getpid();
  u32 index = shm_count_.fetch_add(1, std::memory_order_relaxed);

  // Create shared memory name: chimaera_{pid}_{index}
  std::string shm_name =
      "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  // Add 32MB metadata overhead
  size_t total_size = size + kShmMetadataOverhead;

  HLOG(kInfo,
       "IpcManager::IncreaseClientShm: Creating {} with size {} ({} + {} "
       "overhead)",
       shm_name, total_size, size, kShmMetadataOverhead);

  try {
    // Create the shared memory backend
    auto backend = std::make_unique<hipc::PosixShmMmap>();

    // Create allocator ID: major = pid, minor = index
    hipc::AllocatorId alloc_id(static_cast<u32>(pid), index);

    // Initialize shared memory using backend's shm_init method
    if (!backend->shm_init(alloc_id, hshm::Unit<size_t>::Bytes(total_size),
                           shm_name)) {
      HLOG(kError, "IpcManager::IncreaseClientShm: Failed to create shm for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Create allocator using backend's MakeAlloc method
    hipc::MultiProcessAllocator *allocator =
        backend->MakeAlloc<hipc::MultiProcessAllocator>();

    if (allocator == nullptr) {
      HLOG(kError,
           "IpcManager::IncreaseClientShm: Failed to create allocator for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Add to our tracking structures
    u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                    static_cast<u64>(alloc_id.minor_);
    alloc_map_[alloc_key] = allocator;
    alloc_vector_.push_back(allocator);
    client_backends_.push_back(std::move(backend));
    last_alloc_ = allocator;

    HLOG(kInfo,
         "IpcManager::IncreaseClientShm: Created allocator {} with ID ({}.{})",
         shm_name, alloc_id.major_, alloc_id.minor_);

    // Release the lock before returning
    allocator_map_lock_.WriteUnlock();

    // Tell the runtime server to attach to this new shared memory segment.
    // Use kAdminPoolId directly (not admin_client->pool_id_) because
    // the admin client may not be initialized yet during ClientInit.
    auto reg_task = NewTask<chimaera::admin::RegisterMemoryTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
        alloc_id);
    IpcCpu2CpuZmq::ClientSend(this,reg_task, IpcMode::kTcp).Wait();

    return true;

  } catch (const std::exception &e) {
    allocator_map_lock_.WriteUnlock();
    HLOG(kError, "IpcManager::IncreaseClientShm: Exception creating {}: {}",
         shm_name, e.what());
    shm_count_.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
}

bool IpcManager::RegisterMemory(const hipc::AllocatorId &alloc_id) {
  HLOG(kDebug, "RegisterMemory CALLED: alloc_id=({}.{})", alloc_id.major_,
       alloc_id.minor_);
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during memory registration
  allocator_map_lock_.WriteLock();

  // Derive shm_name from alloc_id: chimaera_{pid}_{index}
  pid_t owner_pid = static_cast<pid_t>(alloc_id.major_);
  u32 shm_index = alloc_id.minor_;
  std::string shm_name =
      "chimaera_" + std::to_string(owner_pid) + "_" + std::to_string(shm_index);

  HLOG(kInfo, "IpcManager::RegisterMemory: Registering {} from pid {}",
       shm_name, owner_pid);

  // Check if already registered
  u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                  static_cast<u64>(alloc_id.minor_);
  if (alloc_map_.find(alloc_key) != alloc_map_.end()) {
    HLOG(kInfo, "IpcManager::RegisterMemory: {} already registered, skipping",
         shm_name);
    allocator_map_lock_.WriteUnlock();
    return true;  // Already registered
  }

  try {
    // Attach to the shared memory backend (already created by client)
    auto backend = std::make_unique<hipc::PosixShmMmap>();
    if (!backend->shm_attach(shm_name)) {
      HLOG(kError, "IpcManager::RegisterMemory: Failed to attach to shm {}",
           shm_name);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Attach to the existing allocator in the backend
    hipc::MultiProcessAllocator *allocator =
        backend->AttachAlloc<hipc::MultiProcessAllocator>();

    if (allocator == nullptr) {
      HLOG(kError,
           "IpcManager::RegisterMemory: Failed to attach allocator for {}",
           shm_name);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Add to our tracking structures
    alloc_map_[alloc_key] = allocator;
    // Note: Don't add to alloc_vector_ since this is not our memory
    // (we don't allocate from it, just need to resolve ShmPtrs)
    client_backends_.push_back(std::move(backend));

    HLOG(kInfo, "IpcManager::RegisterMemory: Successfully registered {}",
         shm_name);

    // Release the lock before returning
    allocator_map_lock_.WriteUnlock();

    return true;

  } catch (const std::exception &e) {
    allocator_map_lock_.WriteUnlock();
    HLOG(kError, "IpcManager::RegisterMemory: Exception registering {}: {}",
         shm_name, e.what());
    return false;
  }
}

ClientShmInfo IpcManager::GetClientShmInfo(u32 index) const {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  if (index >= alloc_vector_.size()) {
    return ClientShmInfo();  // Return empty info
  }

  pid_t pid = getpid();
  std::string shm_name =
      "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  hipc::MultiProcessAllocator *allocator = alloc_vector_[index];
  hipc::AllocatorId alloc_id = allocator->GetId();

  // Get size from backend if available, otherwise use 0
  size_t size = 0;
  if (index < client_backends_.size() && client_backends_[index]) {
    size = client_backends_[index]->backend_size_;
  }

  return ClientShmInfo(shm_name, pid, index, size, alloc_id);
}

size_t IpcManager::WreapDeadIpcs() {
  HLOG(kDebug, "WreapDeadIpcs CALLED");
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during reaping
  allocator_map_lock_.WriteLock();

  pid_t current_pid = getpid();
  size_t reaped_count = 0;

  // Build list of allocator keys to remove (can't modify map while iterating)
  std::vector<u64> keys_to_remove;

  for (const auto &pair : alloc_map_) {
    u64 alloc_key = pair.first;

    // Extract pid from allocator key (major is in upper 32 bits)
    u32 major = static_cast<u32>(alloc_key >> 32);
    u32 minor = static_cast<u32>(alloc_key & 0xFFFFFFFF);

    // Skip main allocator (1.0)
    if (major == 1 && minor == 0) {
      continue;
    }

    // Skip our own process's segments
    pid_t owner_pid = static_cast<pid_t>(major);
    if (owner_pid == current_pid) {
      continue;
    }

    // Check if the owning process is still alive
    // kill(pid, 0) returns 0 if process exists, -1 with ESRCH if not
    if (kill(owner_pid, 0) == -1 && errno == ESRCH) {
      // Process is dead - mark for removal
      HLOG(kInfo,
           "WreapDeadIpcs: Process {} is dead, marking allocator ({}.{}) for "
           "removal",
           owner_pid, major, minor);
      keys_to_remove.push_back(alloc_key);
    }
  }

  // Remove marked allocators and their backends
  for (u64 key : keys_to_remove) {
    // Find the allocator in the map
    auto map_it = alloc_map_.find(key);
    if (map_it == alloc_map_.end()) {
      continue;
    }

    hipc::MultiProcessAllocator *allocator = map_it->second;

    // Get the allocator ID to construct shm_name
    hipc::AllocatorId alloc_id = allocator->GetId();
    std::string shm_name = "chimaera_" + std::to_string(alloc_id.major_) + "_" +
                           std::to_string(alloc_id.minor_);

    // Find and destroy the corresponding backend
    for (auto backend_it = client_backends_.begin();
         backend_it != client_backends_.end(); ++backend_it) {
      if (*backend_it && (*backend_it)->header_ &&
          (*backend_it)->header_->id_.major_ == alloc_id.major_ &&
          (*backend_it)->header_->id_.minor_ == alloc_id.minor_) {
        // Destroy the shared memory
        HLOG(kInfo,
             "WreapDeadIpcs: Destroying shared memory {} for allocator ({}.{})",
             shm_name, alloc_id.major_, alloc_id.minor_);
        (*backend_it)->shm_destroy();
        client_backends_.erase(backend_it);
        break;
      }
    }

    // Remove from alloc_vector_ if present
    auto vec_it =
        std::find(alloc_vector_.begin(), alloc_vector_.end(), allocator);
    if (vec_it != alloc_vector_.end()) {
      alloc_vector_.erase(vec_it);
    }

    // Clear last_alloc_ if it points to this allocator
    if (last_alloc_ == allocator) {
      last_alloc_ = alloc_vector_.empty() ? nullptr : alloc_vector_.back();
    }

    // Remove from alloc_map_
    alloc_map_.erase(map_it);
    reaped_count++;
  }

  if (reaped_count > 0) {
    HLOG(kInfo,
         "WreapDeadIpcs: Reaped {} shared memory segments from dead processes",
         reaped_count);
  }

  // Release the lock before returning
  allocator_map_lock_.WriteUnlock();

  return reaped_count;
}

size_t IpcManager::WreapAllIpcs() {
  HLOG(kDebug, "WreapAllIpcs CALLED");
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during cleanup
  allocator_map_lock_.WriteLock();

  size_t reaped_count = 0;

  // Build list of all allocator keys except main allocator (1.0)
  std::vector<u64> keys_to_remove;

  for (const auto &pair : alloc_map_) {
    u64 alloc_key = pair.first;

    // Extract pid from allocator key (major is in upper 32 bits)
    u32 major = static_cast<u32>(alloc_key >> 32);
    u32 minor = static_cast<u32>(alloc_key & 0xFFFFFFFF);

    // Skip main allocator (1.0) - it's managed separately
    if (major == 1 && minor == 0) {
      continue;
    }

    keys_to_remove.push_back(alloc_key);
  }

  // Destroy all backends and remove from tracking structures
  for (u64 key : keys_to_remove) {
    auto map_it = alloc_map_.find(key);
    if (map_it == alloc_map_.end()) {
      continue;
    }

    hipc::MultiProcessAllocator *allocator = map_it->second;

    // Get the allocator ID to construct shm_name
    hipc::AllocatorId alloc_id = allocator->GetId();
    std::string shm_name = "chimaera_" + std::to_string(alloc_id.major_) + "_" +
                           std::to_string(alloc_id.minor_);

    // Find and destroy the corresponding backend
    for (auto backend_it = client_backends_.begin();
         backend_it != client_backends_.end(); ++backend_it) {
      if (*backend_it && (*backend_it)->header_ &&
          (*backend_it)->header_->id_.major_ == alloc_id.major_ &&
          (*backend_it)->header_->id_.minor_ == alloc_id.minor_) {
        // Destroy the shared memory
        HLOG(kInfo,
             "WreapAllIpcs: Destroying shared memory {} for allocator ({}.{})",
             shm_name, alloc_id.major_, alloc_id.minor_);
        (*backend_it)->shm_destroy();
        client_backends_.erase(backend_it);
        break;
      }
    }

    // Remove from alloc_map_
    alloc_map_.erase(map_it);
    reaped_count++;
  }

  // Clear remaining structures
  alloc_vector_.clear();
  last_alloc_ = nullptr;

  // Note: client_backends_ may still have some entries if backends were
  // not found in the loop above (shouldn't happen in normal operation)
  if (!client_backends_.empty()) {
    HLOG(kWarning, "WreapAllIpcs: {} backends remaining after cleanup",
         client_backends_.size());
    // Destroy any remaining backends
    for (auto &backend : client_backends_) {
      if (backend) {
        backend->shm_destroy();
        reaped_count++;
      }
    }
    client_backends_.clear();
  }

  HLOG(kInfo, "WreapAllIpcs: Reaped {} shared memory segments", reaped_count);

  // Release the lock before returning
  allocator_map_lock_.WriteUnlock();

  return reaped_count;
}

size_t IpcManager::ClearUserIpcs() {
  size_t removed_count = 0;
  std::string memfd_dir = hshm::SystemInfo::GetMemfdDir();

  // Open per-user memfd symlink directory
  DIR *dir = opendir(memfd_dir.c_str());
  if (dir == nullptr) {
    // Directory may not exist yet, that's fine
    return 0;
  }

  // Iterate through directory entries and remove all symlinks
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    // Skip "." and ".."
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    // Construct full path and remove the symlink
    std::string full_path = memfd_dir + "/" + entry->d_name;
    if (unlink(full_path.c_str()) == 0) {
      HLOG(kDebug, "ClearUserIpcs: Removed memfd symlink: {}", entry->d_name);
      removed_count++;
    } else {
      if (errno != EACCES && errno != EPERM && errno != ENOENT) {
        HLOG(kDebug, "ClearUserIpcs: Could not remove {} ({}): {}",
             entry->d_name, errno, strerror(errno));
      }
    }
  }

  closedir(dir);

  if (removed_count > 0) {
    HLOG(kInfo, "ClearUserIpcs: Removed {} memfd symlinks from previous runs",
         removed_count);
  }

  return removed_count;
}

void IpcManager::SetIsClientThread(bool is_client_thread) {
  // Create TLS key if not already created
  HSHM_THREAD_MODEL->CreateTls<bool>(chi_is_client_thread_key_, nullptr);

  // Set the flag for the current thread
  bool *flag = new bool(is_client_thread);
  HSHM_THREAD_MODEL->SetTls(chi_is_client_thread_key_, flag);

  HLOG(kDebug, "SetIsClientThread: Set to {} for current thread",
       is_client_thread);
}

bool IpcManager::GetIsClientThread() const {
  // Get the TLS value, defaulting to false if not set
  bool *flag = HSHM_THREAD_MODEL->GetTls<bool>(chi_is_client_thread_key_);
  if (!flag) {
    return false;
  }
  return *flag;
}

//==============================================================================
// GPU Memory Management
//==============================================================================

//==============================================================================
// Client Retry / Reconnect Methods
//==============================================================================

bool IpcManager::IsServerAlive() const {
  if (!zmq_transport_) return false;
  hshm::lbm::LbmContext ctx;
  if (ipc_mode_ == IpcMode::kShm) {
    ctx.server_pid_ = static_cast<int>(runtime_pid_);
  }
  return zmq_transport_->IsServerAlive(ctx);
}

bool IpcManager::ReconnectToOriginalHost() {
  HLOG(kInfo, "ReconnectToOriginalHost: Attempting to reconnect to restarted server");

  if (ipc_mode_ == IpcMode::kShm) {
    // Detach old shared memory (don't destroy — server owns it)
    main_allocator_ = nullptr;
    worker_queues_ = hipc::FullPtr<TaskQueue>();
    main_backend_ = hipc::PosixShmMmap();

    // Re-attach to new shared memory
    if (!ClientInitShm()) return false;
    if (!ClientInitQueues()) return false;

    // Re-create SHM lightbeam transports
    shm_send_transport_ = hshm::lbm::TransportFactory::Get(
        "", hshm::lbm::TransportType::kShm, hshm::lbm::TransportMode::kClient);
    shm_recv_transport_ = hshm::lbm::TransportFactory::Get(
        "", hshm::lbm::TransportType::kShm, hshm::lbm::TransportMode::kServer);

    // Re-register per-process shared memory segments with new server
    for (auto *alloc : alloc_vector_) {
      auto alloc_id = alloc->GetId();
      auto reg_task = NewTask<chimaera::admin::RegisterMemoryTask>(
          chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
          alloc_id);
      IpcCpu2CpuZmq::ClientSend(this,reg_task, IpcMode::kTcp).Wait();
    }
  }

  // Re-verify server via ClientConnectTask (updates client_generation_)
  if (!WaitForLocalServer()) return false;

  server_alive_.store(true, std::memory_order_release);
  HLOG(kInfo, "ReconnectToOriginalHost: Reconnected, new generation={}",
       client_generation_);
  return true;
}

bool IpcManager::ReconnectToNewHost(const std::string &new_addr) {
  HLOG(kInfo, "ReconnectToNewHost: Switching to {}", new_addr);
  auto *config = CHI_CONFIG_MANAGER;
  u32 port = config->GetPort();

  // Stop recv thread
  if (zmq_recv_running_.load()) {
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) {
      zmq_recv_thread_.join();
    }
  }

  // Destroy old transport
  zmq_transport_.reset();

  // Clear orphaned pending state
  {
    std::lock_guard<std::mutex> lock(pending_futures_mutex_);
    pending_zmq_futures_.clear();
    pending_response_archives_.clear();
  }

  // Disable SHM/IPC — remote hosts require TCP
  ipc_mode_ = IpcMode::kTcp;
  shm_send_transport_.reset();
  shm_recv_transport_.reset();
  main_allocator_ = nullptr;
  runtime_pid_ = 0;

  // Create new ZMQ DEALER transport
  try {
    zmq_transport_ = hshm::lbm::TransportFactory::Get(
        new_addr, hshm::lbm::TransportType::kZeroMq,
        hshm::lbm::TransportMode::kClient, "tcp", port + 3);
  } catch (const std::exception &e) {
    HLOG(kError, "ReconnectToNewHost: Transport to {} failed: {}",
         new_addr, e.what());
    return false;
  }

  // Restart recv thread
  zmq_recv_running_.store(true);
  zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });

  // Verify connectivity — the server should respond almost instantly
  // if it's alive.  No long timer; just a quick round-trip check.
  float saved_timeout = wait_server_timeout_;
  wait_server_timeout_ = 0.5f;
  bool ok = WaitForLocalServer();
  wait_server_timeout_ = saved_timeout;
  if (!ok) {
    HLOG(kWarning, "ReconnectToNewHost: {} not responding", new_addr);
    return false;
  }

  server_alive_.store(true, std::memory_order_release);
  HLOG(kInfo, "ReconnectToNewHost: Connected to {} (generation={})",
       new_addr, client_generation_);
  return true;
}

bool IpcManager::WaitForServerAndReconnect(
    std::chrono::steady_clock::time_point start) {
  // Guard against recursive re-entry (WaitForLocalServer → Recv → here)
  reconnecting_.store(true, std::memory_order_release);

  // Phase 1: Try reconnecting to the original server
  // Skip entirely when client_retry_timeout_==0 (go straight to Phase 2).
  // Use a short WaitForLocalServer timeout so each attempt doesn't
  // block for the full 30s default.
  float saved_timeout = wait_server_timeout_;
  if (client_retry_timeout_ != 0) {
    float per_attempt_timeout = std::min(wait_server_timeout_, 3.0f);
    wait_server_timeout_ = per_attempt_timeout;
    while (true) {
      float elapsed =
          std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
              .count();
      if (client_retry_timeout_ >= 0 && elapsed >= client_retry_timeout_) {
        HLOG(kWarning, "WaitForServerAndReconnect: Original server timed out "
             "after {:.1f}s", elapsed);
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (ReconnectToOriginalHost()) {
        wait_server_timeout_ = saved_timeout;
        reconnecting_.store(false, std::memory_order_release);
        return true;
      }
    }
    wait_server_timeout_ = saved_timeout;
  } else {
    HLOG(kInfo, "WaitForServerAndReconnect: retry_timeout=0, "
         "skipping Phase 1, going straight to Phase 2");
  }

  // Phase 2: Try random hosts from the hostfile
  if (client_try_new_servers_ <= 0 || hostfile_map_.empty()) {
    reconnecting_.store(false, std::memory_order_release);
    return false;
  }

  const auto &hosts = GetAllHosts();
  if (hosts.empty()) {
    HLOG(kWarning, "WaitForServerAndReconnect: No hosts in hostfile");
    reconnecting_.store(false, std::memory_order_release);
    return false;
  }

  // Pick random hosts and try each (may retry same host — that's fine)
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, hosts.size() - 1);

  HLOG(kInfo, "WaitForServerAndReconnect: Trying {} random hosts",
       client_try_new_servers_);
  for (int i = 0; i < client_try_new_servers_; ++i) {
    size_t idx = dist(rng);
    const std::string &addr = hosts[idx].ip_address;
    HLOG(kInfo, "WaitForServerAndReconnect: Trying {}/{}: {}",
         i + 1, client_try_new_servers_, addr);
    if (ReconnectToNewHost(addr)) {
      reconnecting_.store(false, std::memory_order_release);
      return true;
    }
  }

  HLOG(kError, "WaitForServerAndReconnect: All {} random hosts failed",
       client_try_new_servers_);
  reconnecting_.store(false, std::memory_order_release);
  return false;
}

//==============================================================================
// ZMQ Transport Methods
//==============================================================================

void IpcManager::RecvZmqClientThread() {
  // Client-side thread: polls for completed task responses from the server
  // DEALER transport receives responses on the same socket used for sending
  if (!zmq_transport_) {
    HLOG(kError, "RecvZmqClientThread: No DEALER transport");
    return;
  }

  // Set up EventManager for ZMQ transport polling
  hshm::lbm::EventManager em;
  zmq_transport_->RegisterEventManager(em);

  while (zmq_recv_running_.load()) {
    // Drain all available messages first
    bool drained_any = false;
    bool got_message = true;
    while (got_message) {
      got_message = false;
      auto archive = std::make_unique<LoadTaskArchive>();
      auto info = zmq_transport_->Recv(*archive);
      int rc = info.rc;
      if (rc == EAGAIN) break;
      if (rc != 0) {
        zmq_transport_->ClearRecvHandles(*archive);
        if (!zmq_recv_running_.load()) break;
        // ETERM means the ZMQ context is being shut down (zmq_ctx_shutdown was
        // called).  Exit immediately so the context destructor is not blocked.
        if (rc == ETERM) return;
        HLOG(kDebug, "RecvZmqClientThread: Recv returned: {}", rc);
        continue;
      }
      got_message = true;
      drained_any = true;

      // Look up pending future by net_key from task_infos
      if (archive->task_infos_.empty()) {
        HLOG(kError, "RecvZmqClientThread: No task_infos in response");
        continue;
      }
      size_t net_key = archive->task_infos_[0].task_id_.net_key_;

      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      auto it = pending_zmq_futures_.find(net_key);
      if (it == pending_zmq_futures_.end()) {
        HLOG(kError, "RecvZmqClientThread: No pending future for net_key {}",
             net_key);
        zmq_transport_->ClearRecvHandles(*archive);
        continue;
      }

      FutureShm *future_shm = it->second;

      // Store the archive for Recv() to pick up
      pending_response_archives_[net_key] = std::move(archive);

      // Memory fence before setting complete
      std::atomic_thread_fence(std::memory_order_release);

      // Signal completion
      future_shm->flags_.SetBits(FutureShm::FUTURE_NEW_DATA |
                                 FutureShm::FUTURE_COMPLETE);

      // Remove from pending futures map
      pending_zmq_futures_.erase(it);
    }

    // Only block on epoll when the drain loop found nothing;
    // if we just processed messages, loop back immediately.
    if (!drained_any) {
      em.Wait(100);  // 100μs (precise with epoll_pwait2)
    }
  }
}

void IpcManager::HeartbeatThread() {
  while (heartbeat_running_.load()) {
    bool alive = IsServerAlive();
    server_alive_.store(alive, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void IpcManager::CleanupResponseArchive(size_t net_key) {
  std::lock_guard<std::mutex> lock(pending_futures_mutex_);
  auto it = pending_response_archives_.find(net_key);
  if (it != pending_response_archives_.end()) {
    zmq_transport_->ClearRecvHandles(*(it->second));
    pending_response_archives_.erase(it);
  }
}

bool IpcManager::RegisterAcceleratorMemory(const hipc::MemoryBackend &backend) {
#if !HSHM_ENABLE_CUDA && !HSHM_ENABLE_ROCM
  HLOG(kError,
       "RegisterAcceleratorMemory: GPU support not enabled at compile time");
  return false;
#else
  // Store the GPU backend for later use
  // This is called from GPU kernels where we have limited capability
  // The actual allocation happens in CHIMAERA_GPU_INIT macro where
  // each thread gets its own ArenaAllocator instance
  gpu_ipc_->gpu_orchestrator_info_.backend = backend;

  // Note: In GPU kernels, each thread maintains its own ArenaAllocator
  // The macro CHIMAERA_GPU_INIT handles per-thread allocator setup
  // No need to initialize allocators here as they're created per-thread in
  // __shared__ memory

  return true;
#endif
}

void IpcManager::BeginTask(Future<Task> &future, Container *container,
                           TaskLane *lane) {
  FullPtr<Task> task_ptr = future.GetTaskPtr();
  if (task_ptr.IsNull()) {
    HLOG(kError, "BeginTask: task_ptr is null!");
    return;
  }
#if HSHM_IS_HOST
  Worker *worker = CHI_CUR_WORKER;

  // Initialize or reset the task's owned RunContext
  task_ptr->SetRunCtx(new RunContext());
  RunContext *run_ctx = task_ptr->GetRunCtx();

  // Clear and initialize RunContext for new task execution
  run_ctx->worker_id_ = worker ? worker->GetId() : 0;
  run_ctx->task_ = task_ptr;        // Store task in RunContext
  run_ctx->is_yielded_ = false;     // Initially not blocked
  run_ctx->container_ = container;  // Store container for CHI_CUR_CONTAINER
  run_ctx->lane_ = lane;            // Store lane for CHI_CUR_LANE
  run_ctx->event_queue_ =
      worker ? worker->GetEventQueue() : nullptr;  // Set event queue
  run_ctx->future_ = future;        // Store future in RunContext
  run_ctx->coro_handle_ = nullptr;  // Coroutine not started yet

  // Initialize adaptive polling fields for periodic tasks
  if (task_ptr->IsPeriodic()) {
    run_ctx->true_period_ns_ = task_ptr->period_ns_;
    run_ctx->yield_time_us_ =
        task_ptr->period_ns_ / 1000.0;  // Initialize with true period
    run_ctx->did_work_ = false;         // Initially no work done
  } else {
    run_ctx->true_period_ns_ = 0.0;
    run_ctx->yield_time_us_ = 0.0;
    run_ctx->did_work_ = false;
  }

  // Mark that RunContext now exists for this task
  task_ptr->SetFlags(TASK_RUN_CTX_EXISTS);

  // NOTE: Do NOT call SetCurrentRunContext here. BeginTask may be called
  // from SendRuntimeClient inside a running coroutine. Overwriting the
  // current RunContext would cause await_suspend_impl to store the parent
  // coroutine handle on the subtask's RunContext instead of the parent's,
  // leading to premature task completion. StartCoroutine and ResumeCoroutine
  // already set the current RunContext before executing the task.
#endif
}

RouteResult IpcManager::RouteTask(Future<Task> &future, bool force_enqueue) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  if (task_ptr.IsNull()) {
    Worker *worker = CHI_CUR_WORKER;
    HLOG(kWarning, "Worker {}: RouteTask - task_ptr is null",
         worker ? worker->GetId() : 0);
    return RouteResult::Dne;
  }

  // Check if task has already been routed - if so, return ExecHere
  if (task_ptr->IsRouted()) {
    return RouteResult::ExecHere;
  }

  // Only call ScheduleTask for Dynamic pool queries.
  // ScheduleTask resolves Dynamic routing into concrete modes (e.g.,
  // Broadcast, DirectHash, Local). Concrete routing modes (Range, Physical,
  // Local, Broadcast, etc.) were set by a previous routing step and must
  // not be overridden — doing so would cause infinite re-broadcast loops
  // when tasks arrive at remote nodes (e.g., GetOrCreatePool returns
  // Broadcast on every node since the pool doesn't exist yet).
  auto *pool_manager = CHI_POOL_MANAGER;
  Container *static_container =
      pool_manager->GetStaticContainer(task_ptr->pool_id_);
  PoolQuery resolved_query = task_ptr->pool_query_;
  if (static_container && resolved_query.IsDynamicMode()) {
    resolved_query = static_container->ScheduleTask(task_ptr);
    task_ptr->pool_query_ = resolved_query;
  }

  // Resolve pool query into concrete physical addresses
  std::vector<PoolQuery> pool_queries =
      ResolvePoolQuery(resolved_query, task_ptr->pool_id_, task_ptr);

  // Check if pool_queries is empty - this indicates an error in resolution
  if (pool_queries.empty()) {
    Worker *worker = CHI_CUR_WORKER;
    HLOG(kError,
         "Worker {}: Task routing failed - no pool queries resolved. "
         "Pool ID: {}, Method: {}",
         worker ? worker->GetId() : 0, task_ptr->pool_id_, task_ptr->method_);
    return RouteResult::Dne;
  }

  // Check if task should be processed locally
  bool is_local = IsTaskLocal(task_ptr, pool_queries);
  if (is_local) {
    RouteResult result = RouteLocal(future, force_enqueue);
    // If container is plugged or gone, add to retry queue
    if (result == RouteResult::Retry || result == RouteResult::Dne) {
      Worker *worker = CHI_CUR_WORKER;
      HLOG(kError, "RouteTask: RouteLocal returned {} for pool={} method={}, worker={}",
           (int)result, task_ptr->pool_id_, task_ptr->method_,
           worker ? (int)worker->GetId() : -1);
      if (worker && task_ptr->GetRunCtx()) {
        worker->AddToRetryQueue(task_ptr->GetRunCtx());
      }
    }
    return result;
  } else {
    return RouteGlobal(future, pool_queries);
  }
}

bool IpcManager::IsTaskLocal(const FullPtr<Task> &task_ptr,
                             const std::vector<PoolQuery> &pool_queries) {
  // If task has TASK_FORCE_NET flag, force it through network code
  if (task_ptr->task_flags_.Any(TASK_FORCE_NET)) {
    return false;
  }

  // If there's only one node, all tasks are local
  if (GetNumHosts() == 1) {
    return true;
  }

  // Task is local only if there is exactly one pool query
  if (pool_queries.size() != 1) {
    return false;
  }

  const PoolQuery &query = pool_queries[0];

  // Check routing mode first, then specific conditions
  RoutingMode routing_mode = query.GetRoutingMode();

  switch (routing_mode) {
    case RoutingMode::Local:
      return true;  // Always local

    case RoutingMode::Dynamic:
      // Dynamic queries should have been resolved by ScheduleTask before
      // reaching here. Treat as local as a safe fallback.
      return true;

    case RoutingMode::Physical: {
      // Physical mode is local only if targeting local node
      u64 local_node_id = GetNodeId();
      return query.GetNodeId() == local_node_id;
    }

    case RoutingMode::DirectId:
    case RoutingMode::DirectHash:
    case RoutingMode::Range:
    case RoutingMode::Broadcast:
      // These modes should have been resolved to Physical queries by now
      // If we still see them here, they are not local
      return false;

    case RoutingMode::LocalGpuBcast:
    case RoutingMode::ToLocalGpu:
    case RoutingMode::ToLocalCpu:
      return true;  // GPU routing modes are always local

    case RoutingMode::Null:
      return true;  // Null mode is a no-op, treat as local
  }

  return false;
}

RouteResult IpcManager::RouteLocal(Future<Task> &future, bool force_enqueue) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  // Mark as routed so the task is not re-routed on subsequent passes.
  task_ptr->SetFlags(TASK_ROUTED);

  // Resolve the actual execution container
  auto *pool_manager = CHI_POOL_MANAGER;
  bool is_plugged = false;
  ContainerId container_id = task_ptr->pool_query_.GetContainerId();
  Container *exec_container =
      pool_manager->GetContainer(task_ptr->pool_id_, container_id, is_plugged);

  if (!exec_container) {
    HLOG(kError, "RouteLocal: Container not found for pool={} container_id={} method={}",
         task_ptr->pool_id_, container_id, task_ptr->method_);
    return RouteResult::Dne;
  }
  if (is_plugged) {
    HLOG(kWarning, "RouteLocal: Container plugged for pool={}", task_ptr->pool_id_);
    return RouteResult::Retry;
  }

  // GPU routing modes: dispatch to GPU instead of CPU worker
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  RoutingMode mode = task_ptr->pool_query_.GetRoutingMode();
  if (mode == RoutingMode::LocalGpuBcast || mode == RoutingMode::ToLocalGpu) {
    u32 gpu_id = 0;  // TODO: extract from pool_query when multi-GPU
    if (mode == RoutingMode::ToLocalGpu) {
      gpu_id = task_ptr->pool_query_.GetNodeId();
    }
    RouteToGpu(task_ptr, exec_container, gpu_id);
    return RouteResult::Local;  // Enqueued to GPU
  }
#endif

  // Set the completer_ field to track which container will execute this task
  task_ptr->SetCompleter(exec_container->container_id_);

  // Update RunContext to use the resolved execution container
  if (task_ptr->GetRunCtx()) {
    task_ptr->GetRunCtx()->container_ = exec_container;
  }

  // Use scheduler to pick the destination worker
  Worker *worker = CHI_CUR_WORKER;
  u32 dest_worker_id =
      scheduler_->RuntimeMapTask(worker, future, exec_container);

  // If destination matches this worker and not forced to enqueue, execute directly
  if (!force_enqueue && worker && dest_worker_id == worker->GetId()) {
    return RouteResult::ExecHere;
  }

  // Enqueue to the destination worker's lane
  auto &dest_lane = worker_queues_->GetLane(dest_worker_id, 0);
  bool was_empty = dest_lane.Empty();
  dest_lane.Push(future);
  if (was_empty) {
    AwakenWorker(&dest_lane);
  }
  return RouteResult::Local;
}

RouteResult IpcManager::RouteGlobal(Future<Task> &future,
                             const std::vector<PoolQuery> &pool_queries) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  // Log the global routing for debugging
  if (!pool_queries.empty()) {
    Worker *worker = CHI_CUR_WORKER;
    const auto &query = pool_queries[0];
    HLOG(kDebug,
         "Worker {}: RouteGlobal - routing task method={}, pool_id={} to node "
         "{} (routing_mode={})",
         worker ? worker->GetId() : 0, task_ptr->method_, task_ptr->pool_id_,
         query.GetNodeId(), static_cast<int>(query.GetRoutingMode()));
  }

  // Store pool_queries in task's RunContext for SendIn to access
  if (task_ptr->GetRunCtx()) {
    RunContext *run_ctx = task_ptr->GetRunCtx();
    run_ctx->pool_queries_ = pool_queries;
  }

  // Enqueue the original task directly to net_queue_ priority 0 (SendIn)
  EnqueueNetTask(future, NetQueuePriority::kSendIn);

  // Set TASK_ROUTED flag on original task
  task_ptr->SetFlags(TASK_ROUTED);

  Worker *worker = CHI_CUR_WORKER;
  HLOG(kDebug, "Worker {}: RouteGlobal - task enqueued to net_queue",
       worker ? worker->GetId() : 0);

  return RouteResult::Network;
}

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
void IpcManager::RouteToGpu(const FullPtr<Task> &task_ptr,
                             Container *container, u32 gpu_id) {
  (void)container;
  if (task_ptr.IsNull() || !gpu_ipc_ ||
      gpu_id >= gpu_ipc_->gpu_devices_.size()) {
    return;
  }

  u32 task_size = task_ptr->pod_size_;
  if (task_size == 0) {
    HLOG(kError, "RouteToGpu: pod_size_=0 for pool={} method={}",
         task_ptr->pool_id_, task_ptr->method_);
    return;
  }

  // 1. Alloc device buffer [Task | gpu::FutureShm] and copy task H2D
  size_t device_buf_size = task_size + sizeof(gpu::FutureShm);
  void *device_buf = hshm::GpuApi::MallocAndCopy(
      static_cast<const char *>(
          static_cast<const void *>(task_ptr.ptr_)),
      task_size, device_buf_size);
  if (!device_buf) return;

  // 2. Allocate pinned host gpu::FutureShm
  FullPtr<char> fshm_buf = AllocateGpuBuffer(sizeof(gpu::FutureShm), gpu_id);
  if (fshm_buf.IsNull()) {
    hshm::GpuApi::Free(static_cast<char *>(device_buf));
    return;
  }

  // 3. Construct gpu::FutureShm in pinned host
  gpu::FutureShm *fshm = new (fshm_buf.ptr_) gpu::FutureShm();
  fshm->pool_id_ = task_ptr->pool_id_;
  fshm->method_id_ = task_ptr->method_;
  fshm->origin_ = gpu::FutureShm::FUTURE_CLIENT_SHM;
  fshm->client_task_vaddr_ = reinterpret_cast<uintptr_t>(device_buf);
  fshm->task_device_ptr_ = reinterpret_cast<uintptr_t>(device_buf);
  fshm->task_size_ = task_size;
  fshm->flags_.SetBits(gpu::FutureShm::FUTURE_POD_COPY);

  // 4. Flush and push to CPU→GPU queue
#if defined(__x86_64__) || defined(__i386__)
  {
    const char *base = reinterpret_cast<const char *>(fshm);
    for (const char *cl = base; cl < base + sizeof(gpu::FutureShm); cl += 64) {
      _mm_clflush(cl);
    }
    _mm_sfence();
  }
#endif

  auto &lane = gpu_ipc_->gpu_devices_[gpu_id].cpu2gpu_queue.ptr_->GetLane(0, 0);
  hipc::ShmPtr<gpu::FutureShm> fshmptr = fshm_buf.shm_.template Cast<gpu::FutureShm>();
  gpu::Future<Task> future(fshmptr);
  lane.Push(future);

#if defined(__x86_64__) || defined(__i386__)
  {
    const char *q_base = reinterpret_cast<const char *>(&lane);
    for (const char *cl = q_base; cl < q_base + sizeof(lane); cl += 64) {
      _mm_clflush(cl);
    }
    _mm_sfence();
  }
#endif
}
#endif

std::vector<PoolQuery> IpcManager::ResolvePoolQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Basic validation
  if (pool_id.IsNull()) {
    return {};  // Invalid pool ID
  }

  RoutingMode routing_mode = query.GetRoutingMode();
  std::vector<PoolQuery> result;

  switch (routing_mode) {
    case RoutingMode::Local:
      result = ResolveLocalQuery(query, task_ptr);
      break;
    case RoutingMode::Dynamic:
      // Dynamic queries should have been resolved by Container::ScheduleTask
      // before reaching ResolvePoolQuery. Fall through to Local as safe default.
      result = ResolveLocalQuery(query, task_ptr);
      break;
    case RoutingMode::DirectId:
      result = ResolveDirectIdQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::DirectHash:
      result = ResolveDirectHashQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Range:
      result = ResolveRangeQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Broadcast:
      result = ResolveBroadcastQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Physical:
      result = ResolvePhysicalQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::LocalGpuBcast:
    case RoutingMode::ToLocalGpu:
    case RoutingMode::ToLocalCpu:
    case RoutingMode::Null:
      // GPU routing modes are handled by the GPU orchestrator, not CPU routing
      result = {query};
      break;
  }

  // Set ret_node_ on all resolved queries to this node's ID
  u32 this_node_id = GetNodeId();
  for (auto &pq : result) {
    pq.SetReturnNode(this_node_id);
  }

  return result;
}

std::vector<PoolQuery> IpcManager::ResolveLocalQuery(
    const PoolQuery &query, const FullPtr<Task> &task_ptr) {
  // Local routing - process on current node
  return {query};
}

std::vector<PoolQuery> IpcManager::ResolveDirectIdQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get the container ID from the query
  ContainerId container_id = query.GetContainerId();

  // Boundary case optimization: Check if container exists on this node
  if (pool_manager->HasContainer(pool_id, container_id)) {
    // Container is local, resolve to Local query
    return {PoolQuery::Local()};
  }

  // Get the physical node ID for this container
  u32 node_id = pool_manager->GetContainerNodeId(pool_id, container_id);

  // Create a Physical PoolQuery to that node
  return {PoolQuery::Physical(node_id)};
}

std::vector<PoolQuery> IpcManager::ResolveDirectHashQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get pool info to find the number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    return {query};  // Fallback to original query
  }

  // Hash to get container ID
  u32 hash_value = query.GetHash();
  ContainerId container_id = hash_value % pool_info->num_containers_;

  // Boundary case optimization: Check if container exists on this node
  if (pool_manager->HasContainer(pool_id, container_id)) {
    // Container is local, resolve to Local query
    return {PoolQuery::Local()};
  }

  // Check if the address_map_ points this container to the local node.
  // After migration, the container may be mapped here but not yet in
  // containers_ (e.g., forwarded tasks arriving at the destination).
  // Returning Local() prevents an infinite forwarding loop.
  u32 mapped_node = pool_manager->GetContainerNodeId(pool_id, container_id);
  if (mapped_node == GetNodeId()) {
    return {PoolQuery::Local()};
  }

  // Resolve to DirectId so SendIn can dynamically look up the current
  // node via GetContainerNodeId.  This preserves the container_id through
  // the routing chain, which is required for retry-after-recovery: if the
  // original node dies and the container is recovered elsewhere, the retry
  // queue re-resolves DirectId to the new node.
  return {PoolQuery::DirectId(container_id)};
}

std::vector<PoolQuery> IpcManager::ResolveRangeQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  auto *config_manager = CHI_CONFIG_MANAGER;
  if (config_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  u32 range_offset = query.GetRangeOffset();
  u32 range_count = query.GetRangeCount();

  // Validate range
  if (range_count == 0) {
    return {};  // Empty range
  }

  // Boundary case optimization: Check if single-container range is local
  if (range_count == 1) {
    ContainerId container_id = range_offset;
    if (pool_manager->HasContainer(pool_id, container_id)) {
      // Container is local, resolve to Local query
      return {PoolQuery::Local()};
    }
    // Check if address_map_ maps this container to the local node
    u32 mapped_node = pool_manager->GetContainerNodeId(pool_id, container_id);
    if (mapped_node == GetNodeId()) {
      return {PoolQuery::Local()};
    }
    // Resolve to DirectId to preserve container info for retry-after-recovery
    return {PoolQuery::DirectId(container_id)};
  }

  std::vector<PoolQuery> result_queries;

  // Get neighborhood size from configuration (maximum number of queries)
  u32 neighborhood_size = config_manager->GetNeighborhoodSize();

  // Calculate queries needed, capped at neighborhood_size
  u32 ideal_queries = (range_count + neighborhood_size - 1) / neighborhood_size;
  u32 queries_to_create = std::min(ideal_queries, neighborhood_size);

  // Create one query per container
  if (queries_to_create <= 1) {
    queries_to_create = range_count;
  }

  u32 containers_per_query = range_count / queries_to_create;
  u32 remaining_containers = range_count % queries_to_create;

  u32 current_offset = range_offset;
  for (u32 i = 0; i < queries_to_create; ++i) {
    u32 current_count = containers_per_query;
    if (i < remaining_containers) {
      current_count++;  // Distribute remainder across first queries
    }

    if (current_count > 0) {
      result_queries.push_back(PoolQuery::Range(current_offset, current_count));
      current_offset += current_count;
    }
  }

  return result_queries;
}

std::vector<PoolQuery> IpcManager::ResolveBroadcastQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get pool info to find the total number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    return {query};  // Fallback to original query
  }

  // Create a Range query that covers all containers, then resolve it
  PoolQuery range_query = PoolQuery::Range(0, pool_info->num_containers_);
  return ResolveRangeQuery(range_query, pool_id, task_ptr);
}

std::vector<PoolQuery> IpcManager::ResolvePhysicalQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Physical routing - query is already resolved to a specific node
  return {query};
}

hipc::FullPtr<Task> IpcManager::RecvRuntime(
    Future<Task> &future, Container *container, u32 method_id,
    hshm::lbm::Transport *recv_transport) {
  auto future_shm = future.GetFutureShm();

  // Self-send path: no deserialization needed
  if (!future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) ||
      future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED)) {
    return IpcCpu2Self::RuntimeRecv(future);
  }

  u32 origin = future_shm->origin_;
  switch (origin) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    case FutureShm::FUTURE_CLIENT_GPU2CPU:
      return IpcGpu2Cpu::RuntimeRecv(this, future, container,
                                      method_id, recv_transport);
#endif
    case FutureShm::FUTURE_CLIENT_SHM:
    default:
      return IpcCpu2Cpu::RuntimeRecv(this, future, container,
                                      method_id, recv_transport);
  }
}

void IpcManager::SendRuntime(
    const FullPtr<Task> &task_ptr, RunContext *run_ctx,
    Container *container, hshm::lbm::Transport *send_transport) {
  auto future_shm = run_ctx->future_.GetFutureShm();
  u32 origin = future_shm->origin_;

  switch (origin) {
    case FutureShm::FUTURE_CLIENT_SHM:
    default:
      IpcCpu2Cpu::RuntimeSend(this, task_ptr, run_ctx, container,
                               send_transport);
      break;
    case FutureShm::FUTURE_CLIENT_TCP:
    case FutureShm::FUTURE_CLIENT_IPC:
      IpcCpu2CpuZmq::EnqueueRuntimeSend(this, run_ctx, origin);
      break;
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    case FutureShm::FUTURE_CLIENT_GPU2CPU:
      IpcGpu2Cpu::RuntimeSend(this, task_ptr, run_ctx, container);
      break;
    case FutureShm::FUTURE_CLIENT_CPU2GPU:
      IpcCpu2Gpu::RuntimeSend(this, task_ptr, run_ctx, container);
      break;
#endif
  }
}

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

//==============================================================================
// GPU Queue Offset Helpers (for ClientConnect response)
//==============================================================================

u64 gpu::IpcManager::GetCpu2GpuQueueOffset(u32 gpu_id) const {
  if (gpu_id >= gpu_devices_.size()) return 0;
  return gpu_devices_[gpu_id].cpu2gpu_queue.shm_.off_.load();
}

u64 gpu::IpcManager::GetGpu2CpuQueueOffset(u32 gpu_id) const {
  if (gpu_id >= gpu_devices_.size()) return 0;
  return gpu_devices_[gpu_id].gpu2cpu_queue.shm_.off_.load();
}

u64 gpu::IpcManager::GetGpu2GpuQueueOffset(u32 gpu_id) const {
  // GPU→GPU queue is in device memory; offset within device backend.
  if (gpu_id >= gpu_devices_.size()) return 0;
  return gpu_devices_[gpu_id].gpu2gpu_queue.shm_.off_.load();
}

u64 gpu::IpcManager::GetCpu2GpuBackendSize(u32 gpu_id) const {
  if (gpu_id >= gpu_devices_.size()) return 0;
  return gpu_devices_[gpu_id].cpu2gpu_queue_backend->data_capacity_;
}

u64 gpu::IpcManager::GetGpu2CpuBackendSize(u32 gpu_id) const {
  if (gpu_id >= gpu_devices_.size()) return 0;
  return gpu_devices_[gpu_id].gpu2cpu_queue_backend->data_capacity_;
}

void gpu::IpcManager::GetGpu2GpuIpcHandle(u32 gpu_id, char *out_bytes) const {
  memset(out_bytes, 0, 64);
  if (gpu_id >= gpu_devices_.size()) return;
  auto *backend = gpu_devices_[gpu_id].gpu2gpu_queue_backend.get();
  if (!backend || !backend->region_) return;
  // Get IPC handle on-demand from the GPU allocation
  hshm::GpuIpcMemHandle ipc_handle;
  hshm::GpuApi::GetIpcMemHandle(ipc_handle, backend->region_);
  static_assert(sizeof(ipc_handle) <= 64, "GpuIpcMemHandle exceeds 64 bytes");
  memcpy(out_bytes, &ipc_handle, sizeof(ipc_handle));
}

//==============================================================================
// RegisterGpuMemoryFromClient
//==============================================================================

bool gpu::IpcManager::RegisterGpuMemoryFromClient(
    const hipc::MemoryBackendId &backend_id,
    const hshm::GpuIpcMemHandle &ipc_handle,
    size_t data_capacity) {
  HLOG(kInfo, "RegisterGpuMemoryFromClient: backend_id=({}.{}), capacity={}",
       backend_id.major_, backend_id.minor_, data_capacity);

  auto gpu_backend = std::make_unique<hipc::GpuMalloc>();
  if (!gpu_backend->shm_attach_ipc(ipc_handle)) {
    HLOG(kError, "RegisterGpuMemoryFromClient: Failed to open IPC handle");
    return false;
  }

  // Register for ShmPtr resolution
  RegisterGpuAllocator(backend_id, gpu_backend->data_,
                       gpu_backend->data_capacity_);

  client_gpu_data_backends_.push_back(std::move(gpu_backend));
  HLOG(kInfo, "RegisterGpuMemoryFromClient: Successfully registered ({}.{})",
       backend_id.major_, backend_id.minor_);
  return true;
}

//==============================================================================
// ClientInitGpuQueues
//==============================================================================

bool gpu::IpcManager::ClientInitGpuQueues(
    u32 num_gpus,
    const u64 *cpu2gpu_offsets,
    const u64 *gpu2cpu_offsets,
    const u64 *gpu2gpu_offsets,
    const u64 *cpu2gpu_sizes,
    const u64 *gpu2cu_sizes,
    u32 queue_depth,
    const char gpu2gpu_ipc_handles[][64]) {
  if (num_gpus == 0) {
    HLOG(kDebug, "ClientInitGpuQueues: No GPUs reported by server");
    return true;
  }

  HLOG(kInfo, "ClientInitGpuQueues: Attaching to {} GPU queue(s)", num_gpus);

  gpu_devices_.resize(num_gpus);

  for (u32 gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    const std::string sid = std::to_string(gpu_id);

    // --- Attach to CPU→GPU queue backend (GpuShmMmap, pinned host) ---
    {
      std::string url = "/chi_cpu2gpu_q_" + sid;
      hipc::MemoryBackendId bid(5000 + gpu_id, 0);
      auto backend = std::make_unique<hipc::GpuShmMmap>();
      if (!backend->shm_attach(url)) {
        HLOG(kError, "ClientInitGpuQueues: Failed to attach to {}", url);
        return false;
      }
      RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);

      hipc::FullPtr<GpuTaskQueue> q;
      q.shm_.off_ = cpu2gpu_offsets[gpu_id];
      q.shm_.alloc_id_ = hipc::AllocatorId(bid.major_, bid.minor_);
      q.ptr_ = reinterpret_cast<GpuTaskQueue *>(backend->data_ + cpu2gpu_offsets[gpu_id]);
      gpu_devices_[gpu_id].cpu2gpu_queue = q;
      gpu_devices_[gpu_id].client_cpu2gpu_backend = std::move(backend);
    }

    // --- Attach to GPU→CPU queue backend (GpuShmMmap, pinned host) ---
    {
      std::string url = "/chi_gpu2cpu_q_" + sid;
      hipc::MemoryBackendId bid(4000 + gpu_id, 0);
      auto backend = std::make_unique<hipc::GpuShmMmap>();
      if (!backend->shm_attach(url)) {
        HLOG(kError, "ClientInitGpuQueues: Failed to attach to {}", url);
        return false;
      }
      RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);

      hipc::FullPtr<GpuTaskQueue> q;
      q.shm_.off_ = gpu2cpu_offsets[gpu_id];
      q.shm_.alloc_id_ = hipc::AllocatorId(bid.major_, bid.minor_);
      q.ptr_ = reinterpret_cast<GpuTaskQueue *>(backend->data_ + gpu2cpu_offsets[gpu_id]);
      gpu_devices_[gpu_id].gpu2cpu_queue = q;
      gpu_devices_[gpu_id].client_gpu2cpu_backend = std::move(backend);
    }

    // --- Attach to GPU→GPU queue backend via IPC handle (GpuMalloc, device) ---
    {
      hipc::MemoryBackendId bid(3000 + gpu_id, 0);
      hshm::GpuIpcMemHandle handle;
      memcpy(&handle, gpu2gpu_ipc_handles[gpu_id], sizeof(handle));
      auto backend = std::make_unique<hipc::GpuMalloc>();
      if (!backend->shm_attach_ipc(handle)) {
        HLOG(kWarning, "ClientInitGpuQueues: Failed to attach GPU→GPU device "
             "backend for GPU {} (GPU→GPU dispatch unavailable)", gpu_id);
        // Set null queue so gpu_id indexing stays consistent
        gpu_devices_[gpu_id].gpu2gpu_queue = hipc::FullPtr<GpuTaskQueue>();
        gpu_devices_[gpu_id].gpu2gpu_queue_backend = nullptr;
      } else {
        RegisterGpuAllocator(bid, backend->data_, backend->data_capacity_);
        hipc::FullPtr<GpuTaskQueue> q;
        q.shm_.off_ = gpu2gpu_offsets[gpu_id];
        q.shm_.alloc_id_ = hipc::AllocatorId(bid.major_, bid.minor_);
        q.ptr_ = reinterpret_cast<GpuTaskQueue *>(backend->data_ + gpu2gpu_offsets[gpu_id]);
        gpu_devices_[gpu_id].gpu2gpu_queue = q;
        gpu_devices_[gpu_id].gpu2gpu_queue_backend = std::move(backend);
      }
    }

    HLOG(kInfo, "ClientInitGpuQueues: GPU {} queues attached", gpu_id);
  }

  return true;
}

//==============================================================================
// GetGpuInfo
//==============================================================================

IpcManagerGpuInfo gpu::IpcManager::GetGpuInfo(u32 gpu_id) const {
  IpcManagerGpuInfo info;

  if (gpu_id >= gpu_devices_.size()) {
    return info;
  }

  auto &dev = gpu_devices_[gpu_id];

  // Primary backend: orchestrator scratch (pinned host, accessible from GPU)
  if (dev.gpu_orchestrator_backend) {
    info.backend =
        static_cast<hipc::MemoryBackend &>(*dev.gpu_orchestrator_backend);
  }

  // GPU->GPU queue (device memory)
  info.gpu2gpu_queue = dev.gpu2gpu_queue.ptr_;
  info.gpu2gpu_num_lanes = gpu_orchestrator_info_.gpu2gpu_num_lanes;

  // Internal subtask queue (device memory)
  info.internal_queue = dev.internal_queue.ptr_;
  info.internal_num_lanes = gpu_orchestrator_info_.internal_num_lanes;

  // CPU->GPU queue (pinned host, orchestrator polls)
  info.cpu2gpu_queue = dev.cpu2gpu_queue.ptr_;

  // GPU->CPU queue (pinned host, CPU worker polls)
  info.gpu2cpu_queue = dev.gpu2cpu_queue.ptr_;

  // GPU->CPU copy-space backend (pinned host)
  if (dev.gpu2cpu_copy_backend) {
    info.gpu2cpu_backend =
        static_cast<hipc::MemoryBackend &>(*dev.gpu2cpu_copy_backend);
  }

  info.gpu_queue_depth = gpu_orchestrator_info_.gpu_queue_depth;

  return info;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

}  // namespace chi