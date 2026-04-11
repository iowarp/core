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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_

#ifndef CHI_GPU_WORKER_DEBUG
#define CHI_GPU_WORKER_DEBUG 0
#endif

#ifndef CHI_GPU_WORKER_PROFILE
#define CHI_GPU_WORKER_PROFILE 1
#endif

#if CHI_GPU_WORKER_DEBUG
#define GPU_WORKER_DPRINTF(...) printf(__VA_ARGS__)
#else
#define GPU_WORKER_DPRINTF(...) ((void)0)
#endif

#if CHI_GPU_WORKER_PROFILE
#define GPU_WORKER_TIMER_DEF(var) long long var = 0
#define GPU_WORKER_TIMER_START(var) var = clock64()
#define GPU_WORKER_TIMER_END(counter, var) counter += clock64() - var
#else
#define GPU_WORKER_TIMER_DEF(var) ((void)0)
#define GPU_WORKER_TIMER_START(var) ((void)0)
#define GPU_WORKER_TIMER_END(counter, var) ((void)0)
#endif

#include "chimaera/gpu/container.h"
#include "chimaera/gpu/pool_manager.h"
#include "chimaera/gpu/work_orchestrator.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/task.h"
#include "hermes_shm/lightbeam/shm_transport.h"

namespace chi {
namespace gpu {

// Forward declaration
class Worker;

// ============================================================================
// CDP Child Kernel: RunTask
// ============================================================================

/**
 * __global__ kernel launched by Worker::TryPopFromQueue() for each task.
 *
 * Executes the task method with the given parallelism (gridDim.x * blockDim.x).
 * Thread 0 marks completion after container->Run() returns.
 * Fire-and-forget: relies on CDP implicit synchronization at parent exit.
 */
__global__ void RunTask(Container *container, u32 method,
                        Task *task_raw, size_t task_shm_off,
                        FutureShm *fshm, bool is_gpu2gpu,
                        chi::IpcManagerGpuInfo *gpu_info_ptr) {
  // Initialize IpcManager for this CDP child kernel block.
  // Reattaches to the orchestrator's existing RoundRobinAllocator and
  // claims a partition for this block.
  chi::IpcManagerGpuInfo gpu_info = *gpu_info_ptr;
  CHIMAERA_GPU_SUBTASK_INIT(gpu_info, gridDim.x);

  // Reconstruct FullPtr from raw pointer + offset (avoids passing
  // user-defined-copy-ctor type to kernel)
  hipc::FullPtr<Task> task_ptr;
  task_ptr.ptr_ = task_raw;
  task_ptr.shm_.off_ = task_shm_off;
  task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

  // Construct RunContext on the stack of the child kernel
  RunContext rctx;
  rctx.container_ = container;
  rctx.method_id_ = method;
  rctx.task_ptr_ = task_ptr;
  rctx.task_fshm_ = fshm;
  rctx.parallelism_ = gridDim.x * blockDim.x;

  // Fix up SSO/SVO pointers for CPU→GPU POD-copied tasks
  if (!is_gpu2gpu) {
    container->FixupTask(method, task_ptr);
  }

  // Execute the task method
  container->Run(method, task_ptr, rctx);

  // Thread 0 marks task as complete via device-scope atomic.
  // The fshm is always in device memory (even for CPU→GPU tasks, where it
  // sits right after the task copy in device space). The orchestrator parent
  // kernel polls this flag and relays completion to the pinned-host FutureShm
  // mirror via system-scope write (see RelayPendingCompletions).
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    atomicOr(reinterpret_cast<unsigned int*>(&fshm->flags_.bits_.x),
             static_cast<unsigned int>(FutureShm::FUTURE_COMPLETE));
    __threadfence();
  }
}

// ============================================================================
// Worker Class: CDP-based Persistent Kernel Dispatcher
// ============================================================================

/**
 * GPU-side worker (single thread-0 based in persistent kernel).
 *
 * Thread 0 polls gpu2gpu_queue_, internal_queue_, and cpu2gpu_queue_,
 * launching a CDP child kernel for each popped task.
 * Child kernels execute fire-and-forget without explicit synchronization.
 */
class Worker {
 public:
  u32 worker_id_;
  volatile bool is_running_;
  GpuTaskQueue *cpu2gpu_queue_;
  GpuTaskQueue *gpu2gpu_queue_;
  GpuTaskQueue *internal_queue_;
  PoolManager *pool_mgr_;
  char *queue_backend_base_;
  WorkOrchestratorControl *dbg_ctrl_;
  IpcManagerGpuInfo *gpu_info_ptr_; /**< Device ptr to gpu_info for CDP child init */

  /** Pending CPU→GPU tasks: device FutureShm pointers awaiting completion.
   *  Parent kernel relays FUTURE_COMPLETE to pinned-host copy because
   *  CDP child writes to device memory are only visible to the parent. */
  static constexpr u32 kMaxPendingCpu2Gpu = 64;
  FutureShm *pending_device_fshm_[kMaxPendingCpu2Gpu];
  FutureShm *pending_host_fshm_[kMaxPendingCpu2Gpu];  /**< Pinned host mirrors */
  u32 num_pending_;


  // Profiling counters
  long long prof_queue_pop_, prof_task_count_;

  HSHM_GPU_FUN void Init(u32 worker_id, GpuTaskQueue *cpu2gpu_queue,
                         GpuTaskQueue *gpu2gpu_queue, GpuTaskQueue *internal_queue,
                         PoolManager *pool_mgr, char *queue_backend_base,
                         WorkOrchestratorControl *dbg_ctrl) {
    worker_id_ = worker_id;
    cpu2gpu_queue_ = cpu2gpu_queue;
    gpu2gpu_queue_ = gpu2gpu_queue;
    internal_queue_ = internal_queue;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    dbg_ctrl_ = dbg_ctrl;
    gpu_info_ptr_ = nullptr;  // Set by orchestrator after init
    is_running_ = true;
    num_pending_ = 0;
    for (u32 i = 0; i < kMaxPendingCpu2Gpu; ++i) {
      pending_device_fshm_[i] = nullptr;
      pending_host_fshm_[i] = nullptr;
    }
    prof_queue_pop_ = 0;
    prof_task_count_ = 0;
  }

  HSHM_GPU_FUN void FlushProfile() {
    if (!dbg_ctrl_ || worker_id_ >= WorkOrchestratorControl::kMaxDebugWorkers)
      return;
    dbg_ctrl_->prof_queue_pop[worker_id_] = prof_queue_pop_;
    dbg_ctrl_->prof_task_count[worker_id_] = prof_task_count_;
  }

  HSHM_GPU_FUN void Stop() { is_running_ = false; }
  HSHM_GPU_FUN void Finalize() { is_running_ = false; }

  // ================================================================
  // Main poll loop (thread 0 only)
  // ================================================================

  /** Poll GPU→GPU queues (high priority) */
  HSHM_GPU_FUN int PollGpu2Gpu() {
    int count = 0;
    for (int i = 0; i < 16; ++i) {
      count += TryPopFromQueue(gpu2gpu_queue_, worker_id_, true);
      count += TryPopFromQueue(internal_queue_, worker_id_, true);
    }
    return count;
  }

  /** Poll CPU→GPU queue AND check pending task completions */
  HSHM_GPU_FUN int PollCpu2Gpu() {
    // Check pending CPU→GPU tasks for completion and relay to host
    RelayPendingCompletions();
    return TryPopFromQueue(cpu2gpu_queue_, 0, false);
  }

  /**
   * Check pending CPU→GPU tasks. If a CDP child completed (device FutureShm
   * has FUTURE_COMPLETE), relay the flag to the pinned-host FutureShm mirror
   * via system-scope write so the CPU can see it.
   */
  HSHM_GPU_FUN void RelayPendingCompletions() {
    for (u32 i = 0; i < num_pending_; ) {
      FutureShm *dev_fshm = pending_device_fshm_[i];
      FutureShm *host_fshm = pending_host_fshm_[i];
      // Parent kernel CAN see CDP child writes to device memory
      if (dev_fshm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
        // Relay to pinned host via system-scope write (parent kernel writes
        // ARE visible to CPU, unlike CDP child kernel writes)
        __threadfence_system();
        volatile u32 *host_flags = reinterpret_cast<volatile u32 *>(
            &host_fshm->flags_.bits_.x);
        u32 old_val = *host_flags;
        *host_flags = old_val | FutureShm::FUTURE_COMPLETE;
        __threadfence_system();
        // Remove from pending (swap with last)
        --num_pending_;
        pending_device_fshm_[i] = pending_device_fshm_[num_pending_];
        pending_host_fshm_[i] = pending_host_fshm_[num_pending_];
        // Don't increment i — check the swapped-in entry
      } else {
        ++i;
      }
    }
  }

  /** Poll all queues (backward-compatible) */
  HSHM_GPU_FUN int PollOnce() {
    DbgPoll();
    int count = PollGpu2Gpu();
    if (worker_id_ == 0) {
      count += PollCpu2Gpu();
    }
    return count;
  }

 private:
  // ================================================================
  // Queue polling and CDP launch
  // ================================================================

  HSHM_GPU_FUN int TryPopFromQueue(GpuTaskQueue *queue, u32 qlane, bool is_gpu2gpu) {
    if (!queue) return 0;

    auto &lane = queue->GetLane(qlane, 0);

    // Debug: track queue state
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      u64 h = lane.GetHeadDevice();
      u64 t = lane.GetTailDevice();
      if (queue == internal_queue_) {
        dbg_ctrl_->dbg_iq_head[worker_id_] = h;
        dbg_ctrl_->dbg_iq_tail[worker_id_] = t;
      } else if (is_gpu2gpu) {
        dbg_ctrl_->dbg_input_tw[worker_id_] = t;
        dbg_ctrl_->dbg_input_cs[worker_id_] = h;
        if (dbg_ctrl_->dbg_ser_total_written[worker_id_] == 0) {
          dbg_ctrl_->dbg_ser_total_written[worker_id_] =
              reinterpret_cast<unsigned long long>(queue);
        }
      }
    }

    // Attempt to pop a task from the queue
    GPU_WORKER_TIMER_DEF(_qpop_tc);
    GPU_WORKER_TIMER_START(_qpop_tc);
    Future<Task> future;
    bool popped = lane.Pop(future);
    if (!popped) {
      return 0;
    }

    GPU_WORKER_TIMER_END(prof_queue_pop_, _qpop_tc);
    DbgQueuePop();

    // Track internal queue pops
    if (queue == internal_queue_ && dbg_ctrl_ &&
        worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_iq_pops[worker_id_] =
          dbg_ctrl_->dbg_iq_pops[worker_id_] + 1;
    }

    // Extract FutureShm from the popped future
    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) {
      return 0;
    }

    size_t off = sptr.off_.load();
    FutureShm *fshm;
    // Sentinel alloc_id = SendCpuToGpu device pointer
    // Null alloc_id (UINT32_MAX, UINT32_MAX) = GPU→GPU raw device pointer
    FutureShm *host_fshm_for_relay = nullptr;  // Set for cpu2gpu tasks
    if (sptr.alloc_id_ == FutureShm::GetCpu2GpuAllocId()) {
      // SendCpuToGpu: off = pinned-host FutureShm address (UVA accessible).
      // Read device task address and compute device FutureShm.
      FutureShm *host_fshm = reinterpret_cast<FutureShm *>(off);
      __threadfence_system();  // Ensure host writes are visible
      uintptr_t device_task = host_fshm->client_task_vaddr_;
      u32 task_size = host_fshm->task_size_;
      fshm = reinterpret_cast<FutureShm *>(device_task + task_size);
      host_fshm_for_relay = host_fshm;  // Track for completion relay
    } else if (sptr.alloc_id_ == hipc::AllocatorId::GetNull()) {
      // GPU→GPU direct path: raw device pointer
      fshm = reinterpret_cast<FutureShm *>(off);
    } else if (!is_gpu2gpu) {
      fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
    } else {
      fshm = reinterpret_cast<FutureShm *>(off);
    }

    if (!fshm) {
      return 0;
    }

    // Ensure fshm is visible
    hipc::threadfence();

    // Look up the container from the pool
    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
    Container *container = pool_mgr_->GetContainer(pool_id);
    if (!container) {
      DbgNoContainer(pool_id.major_, pool_id.minor_);
      MarkComplete(fshm, is_gpu2gpu);
      return 0;
    }

    // Build task pointer
    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    // Get parallelism from task
    u32 parallelism = task_ptr.ptr_->pool_query_.GetParallelism();
    u32 grid_dim = (parallelism + 31) / 32;  // Number of blocks for 32 threads
    if (grid_dim == 0) grid_dim = 1;

    DbgTaskPopped();
    ++prof_task_count_;

    printf("[POP] pool=(%u,%u) method=%u grid=%u is_gpu2gpu=%d task=%p fshm=%p\n",
           pool_id.major_, pool_id.minor_, method_id, grid_dim,
           (int)is_gpu2gpu, (void*)task_ptr.ptr_, (void*)fshm);

    // Launch CDP child on an explicit non-blocking stream so multiple
    // children can run concurrently (default stream serializes them).
    cudaStream_t child_stream;
    cudaStreamCreateWithFlags(&child_stream, cudaStreamNonBlocking);
    RunTask<<<grid_dim, 32, 0, child_stream>>>(
        container, method_id, task_ptr.ptr_,
        task_ptr.shm_.off_.load(), fshm, is_gpu2gpu,
        gpu_info_ptr_);
    cudaError_t cdp_err = cudaGetLastError();
    if (cdp_err != cudaSuccess) {
      printf("[ORCH-CDP] RunTask launch FAILED: %d (%s)\n",
             (int)cdp_err, cudaGetErrorString(cdp_err));
    } else {
      printf("[POP] RunTask launched OK\n");
    }

    // Track CPU→GPU tasks for completion relay
    if (host_fshm_for_relay && num_pending_ < kMaxPendingCpu2Gpu) {
      pending_device_fshm_[num_pending_] = fshm;
      pending_host_fshm_[num_pending_] = host_fshm_for_relay;
      ++num_pending_;
    }

    return 1;
  }

  // ================================================================
  // Completion helpers
  // ================================================================

  HSHM_GPU_FUN void MarkComplete(FutureShm *fshm, bool is_gpu2gpu) {
    if (is_gpu2gpu) {
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }
  }

  // ================================================================
  // Debug helpers
  // ================================================================

  HSHM_GPU_FUN void DbgPoll() {
#ifndef NDEBUG
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_poll_count[worker_id_] =
          dbg_ctrl_->dbg_poll_count[worker_id_] + 1;
    }
#endif
  }

  HSHM_GPU_FUN void DbgTaskPopped() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_tasks_popped[worker_id_] =
          dbg_ctrl_->dbg_tasks_popped[worker_id_] + 1;
  }

  HSHM_GPU_FUN void DbgQueuePop() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_queue_pops[worker_id_] =
          dbg_ctrl_->dbg_queue_pops[worker_id_] + 1;
  }

  HSHM_GPU_FUN void DbgNoContainer(unsigned int pool_major,
                                   unsigned int pool_minor) {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_no_container[worker_id_] =
          dbg_ctrl_->dbg_no_container[worker_id_] + 1;
      dbg_ctrl_->dbg_last_method[worker_id_] =
          (pool_major << 16) | (pool_minor & 0xFFFF);
    }
  }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
