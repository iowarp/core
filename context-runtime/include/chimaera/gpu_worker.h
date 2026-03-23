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

// Set to 1 to enable verbose GPU worker debug printf (very slow)
#ifndef CHI_GPU_WORKER_DEBUG
#define CHI_GPU_WORKER_DEBUG 0
#endif

#if CHI_GPU_WORKER_DEBUG
#define GPU_WORKER_DPRINTF(...) printf(__VA_ARGS__)
#else
#define GPU_WORKER_DPRINTF(...) ((void)0)
#endif

#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/gpu_work_orchestrator.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"
#include "hermes_shm/lightbeam/shm_transport.h"

namespace chi {
namespace gpu {

/**
 * Coroutine frame allocator using CHI_PRIV_ALLOC (PrivateBuddyAllocator).
 * alloc_ctx is hipc::PrivateBuddyAllocator* (cached per-warp pointer).
 * Stores FullPtr offset in FrameHeader::opaque_ for deallocation.
 */
__device__ inline void *HeapCoroAlloc(size_t size, void *alloc_ctx) {
  auto *alloc = static_cast<hipc::PrivateBuddyAllocator *>(alloc_ctx);
  auto fp = alloc->template AllocateObjs<char>(size);
  if (fp.IsNull()) return nullptr;
  return fp.ptr_;
}

__device__ inline void HeapCoroFree(void *ptr, void *alloc_ctx) {
  auto *alloc = static_cast<hipc::PrivateBuddyAllocator *>(alloc_ctx);
  hipc::FullPtr<char> fp(static_cast<char *>(ptr));
  fp.shm_.off_ = static_cast<size_t>(
      static_cast<char *>(ptr) - alloc->GetBackendData());
  fp.shm_.alloc_id_ = alloc->GetId();
  alloc->Free(fp);
}

static constexpr u32 kWarpSize = 32;

/**
 * Per-lane active task context, heap-allocated as array of 32.
 * The __shared__ pointer points to this array so all lanes can access it.
 */
struct GpuRunContext {
  // -- Shared across all 32 lanes (set by lane 0) --
  Container *container;
  u32 method_id;
  hipc::FullPtr<Task> task_ptr;
  FutureShm *fshm;
  bool is_gpu2gpu;
  bool is_copy_path;

  // -- Per-lane state --
  RunContext rctx;             /**< Per-lane RunContext (lane_id_ differs) */
  TaskResume coro;             /**< Per-lane coroutine handle */
};

/**
 * GPU-side warp-level worker.
 *
 * Each warp in the GPU work orchestrator gets one Worker. Lane 0 of the
 * warp acts as the scheduler (queue polling, deserialization, serialization,
 * suspended task management). ALL 32 threads call container->Run() so that
 * tasks can distribute work across the warp via rctx.lane_id_.
 *
 * No STL, no TLS — all HSHM_GPU_FUN.  Container::Run() returns
 * chi::gpu::TaskResume (a C++20 coroutine) enabling cooperative yielding.
 */
class Worker {
 public:
  u32 worker_id_;                    /**< Worker identity (= warp ID) */
  u32 lane_id_;                      /**< Queue lane this warp polls */
  volatile bool is_running_;         /**< Running flag for the poll loop */
  TaskQueue *cpu2gpu_queue_;         /**< CPU → GPU queue (GPU work orchestrator polls) */
  TaskQueue *gpu2gpu_queue_;         /**< GPU → GPU queue (GPU work orchestrator polls) */
  TaskQueue *internal_queue_;        /**< Internal subtask queue (GPU orchestrator polls) */
  PoolManager *pool_mgr_;            /**< GPU-side container lookup table */
  char *queue_backend_base_;         /**< Base of queue backend for ShmPtr resolution */
  RunContext rctx_;                   /**< Template RunContext (copied per task) */
  WorkOrchestratorControl *dbg_ctrl_; /**< Debug control struct (pinned, CPU-readable) */

  /**
   * Pointer to __shared__ memory holding the active task context array.
   * Lane 0 sets *active_tasks_ptr_ to a heap-allocated GpuRunContext[32].
   * All lanes read it after __syncwarp().
   */
  GpuRunContext **active_tasks_ptr_;

  static constexpr u32 kMaxSuspended = 128;

  /** State for a suspended warp-task */
  struct SuspendedTask {
    GpuRunContext *contexts;       /**< Heap array of GpuRunContexts */
    u32 num_contexts;              /**< 1 or 32 */
    bool occupied;
  };
  SuspendedTask suspended_[kMaxSuspended];
  u32 num_suspended_;
  u32 active_num_contexts_;        /**< Number of contexts in active task (1 or 32) */

  /**
   * Initialize the worker with queue and pool manager pointers.
   */
  HSHM_GPU_FUN void Init(u32 worker_id,
                          u32 lane_id,
                          TaskQueue *cpu2gpu_queue,
                          TaskQueue *gpu2gpu_queue,
                          TaskQueue *internal_queue,
                          PoolManager *pool_mgr,
                          char *queue_backend_base,
                          WorkOrchestratorControl *dbg_ctrl,
                          GpuRunContext **active_tasks_ptr) {
    worker_id_ = worker_id;
    lane_id_ = lane_id;
    cpu2gpu_queue_ = cpu2gpu_queue;
    gpu2gpu_queue_ = gpu2gpu_queue;
    internal_queue_ = internal_queue;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    dbg_ctrl_ = dbg_ctrl;
    active_tasks_ptr_ = active_tasks_ptr;
    is_running_ = true;
    num_suspended_ = 0;
    active_num_contexts_ = 0;
    for (u32 i = 0; i < kMaxSuspended; ++i) {
      suspended_[i].occupied = false;
      suspended_[i].contexts = nullptr;
      suspended_[i].num_contexts = 0;
    }
#if HSHM_IS_GPU_COMPILER
    u32 warp_id = IpcManager::GetWarpId();
    u32 warp_lane = IpcManager::GetLaneId();
    rctx_ = RunContext(blockIdx.x, threadIdx.x, warp_id, warp_lane);
    // Wire coroutine frame allocation to CHI_PRIV_ALLOC (cached BuddyAllocator)
    auto *priv_alloc = CHI_PRIV_ALLOC;
    if (priv_alloc) {
      rctx_.alloc_fn_ = HeapCoroAlloc;
      rctx_.free_fn_ = HeapCoroFree;
      rctx_.alloc_ctx_ = priv_alloc;
    }
#else
    rctx_ = RunContext(0, 0, 0, 0);
#endif
  }

  /** Write debug state to pinned memory (CPU-readable) */
  HSHM_GPU_FUN void DbgState([[maybe_unused]] u32 state) {
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_last_state[lane_id_] = state;
      dbg_ctrl_->dbg_num_suspended[lane_id_] = num_suspended_;
    }
#endif
  }

  HSHM_GPU_FUN void DbgPoll() {
#ifndef NDEBUG
    if (dbg_ctrl_ && lane_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_poll_count[lane_id_]++;
    }
#endif
  }

  HSHM_GPU_FUN void DbgTaskPopped() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_popped[worker_id_]++;
    }
  }
  HSHM_GPU_FUN void DbgTaskCompleted() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_completed[worker_id_]++;
#ifdef HSHM_BUDDY_ALLOC_DEBUG
      u32 completed = dbg_ctrl_->dbg_tasks_completed[worker_id_];
      if (completed % 50 == 0) {
        auto *priv = &CHI_IPC->gpu_priv_alloc_;
        if (CHI_IPC->gpu_priv_alloc_init_) {
          printf("[W%u] task#%u priv: allocs=%llu frees=%llu net=%llu "
                 "bigheap=%llu/%llu\n",
                 worker_id_, completed,
                 (unsigned long long)priv->DbgAllocCount(),
                 (unsigned long long)priv->DbgFreeCount(),
                 (unsigned long long)priv->DbgNetBytes(),
                 (unsigned long long)priv->DbgBigHeapOffset(),
                 (unsigned long long)priv->DbgBigHeapMax());
        }
      }
#endif
    }
  }
  HSHM_GPU_FUN void DbgTaskResumed() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_resumed[worker_id_]++;
    }
  }
  HSHM_GPU_FUN void DbgAllocFailure() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_alloc_failures[worker_id_]++;
    }
  }
  HSHM_GPU_FUN void DbgQueuePop() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_queue_pops[worker_id_]++;
    }
  }
  HSHM_GPU_FUN void DbgNoContainer(unsigned int pool_major, unsigned int pool_minor) {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_no_container[worker_id_]++;
      // Stash the last pool_id that failed
      dbg_ctrl_->dbg_last_method[worker_id_] =
          (pool_major << 16) | (pool_minor & 0xFFFF);
    }
  }

  /**
   * Process one iteration of the warp-level poll loop.
   *
   * Phase 1 (lane 0): Check suspended tasks, pop new tasks from queues,
   *   deserialize, allocate GpuRunContext[32], set __shared__ pointer.
   * Phase 2 (all lanes): Execute active task — all 32 threads call
   *   container->Run() with their per-lane RunContext.
   * Phase 3 (lane 0): Check completion, serialize output or suspend.
   *
   * @param lane_id Warp lane of the calling thread (0-31)
   */
  HSHM_GPU_FUN bool PollOnce(u32 lane_id) {
    bool did_work = false;

    // ================================================================
    // Phase 1: Lane 0 schedules work
    // ================================================================
    if (lane_id == 0) {
      DbgPoll();
      *active_tasks_ptr_ = nullptr;

      // Check suspended tasks for completion
      if (num_suspended_ > 0) {
        did_work |= CheckAndResumeSuspended();
      }

      // Pop new task if no resumed task and we have capacity
      if (*active_tasks_ptr_ == nullptr && num_suspended_ < kMaxSuspended) {
        did_work |= TryPopNewTask();
      }
    }

#if HSHM_IS_GPU_COMPILER
    __syncwarp();
#endif

    // ================================================================
    // Phase 2: Participating lanes execute the active task
    // ================================================================
    GpuRunContext *contexts = *active_tasks_ptr_;
    u32 num_ctx = active_num_contexts_;
    if (contexts != nullptr && lane_id == 0) {
      GPU_WORKER_DPRINTF("[W%u] Phase2: method %u, num_ctx %u\n",
                         worker_id_, contexts[0].method_id, num_ctx);
    }
    if (contexts != nullptr && lane_id < num_ctx) {
      GpuRunContext &my_ctx = contexts[lane_id];

      if (!my_ctx.coro) {
        // First execution: create coroutine
        my_ctx.coro = my_ctx.container->Run(
            my_ctx.method_id, my_ctx.task_ptr, my_ctx.rctx);
        if (!my_ctx.coro.get_handle()) {
          printf("[W%u] CORO ALLOC FAILED: method=%u pool=%u.%u\n",
                 worker_id_, my_ctx.method_id,
                 my_ctx.task_ptr.ptr_->pool_id_.major_,
                 my_ctx.task_ptr.ptr_->pool_id_.minor_);
        } else {
          my_ctx.coro.get_handle().promise().set_run_context(&my_ctx.rctx);
          my_ctx.coro.resume();
        }
      } else {
        // Resuming from suspension
        if (!my_ctx.coro.done()) {
          my_ctx.rctx.is_yielded_ = false;
          // Chain-resume: resume the innermost yielded coroutine
          if (my_ctx.rctx.coro_handle_ &&
              !my_ctx.rctx.coro_handle_.done()) {
            my_ctx.rctx.coro_handle_.resume();
          } else {
            my_ctx.coro.resume();
          }
        }
      }
      did_work = true;
    }

#if HSHM_IS_GPU_COMPILER
    __syncwarp();
#endif

    // ================================================================
    // Phase 3: Lane 0 handles completion / suspension
    // ================================================================
    if (lane_id == 0 && contexts != nullptr) {
      // Check lane 0's coroutine as representative (SIMT: all lanes
      // follow the same control flow for well-structured tasks)
      bool task_done = contexts[0].coro.done();

      if (task_done) {
        // Task complete: serialize output and free contexts
        DbgState(5);
        DbgTaskCompleted();
        GPU_WORKER_DPRINTF("[W%u] Task DONE: pool %u.%u method %u\n",
                           worker_id_, contexts[0].task_ptr.ptr_->pool_id_.major_,
                           contexts[0].task_ptr.ptr_->pool_id_.minor_,
                           contexts[0].method_id);
        if (contexts[0].is_copy_path) {
          SerializeAndComplete(contexts[0].fshm, contexts[0].container,
                               contexts[0].method_id, contexts[0].task_ptr,
                               contexts[0].is_gpu2gpu);
        } else {
          CompleteAndResumeParent(contexts[0].fshm, contexts[0].is_gpu2gpu);
        }
        FreeContexts(contexts, num_ctx);
        *active_tasks_ptr_ = nullptr;
      } else {
        // Task suspended (co_await): save to suspended list
        DbgState(3);
        u32 slot = FindFreeSlot();
        GPU_WORKER_DPRINTF("[W%u] Task SUSPEND: pool %u.%u method %u -> slot %u (nsus=%u)\n",
                           worker_id_, contexts[0].task_ptr.ptr_->pool_id_.major_,
                           contexts[0].task_ptr.ptr_->pool_id_.minor_,
                           contexts[0].method_id, slot, num_suspended_ + 1);
        suspended_[slot].contexts = contexts;
        suspended_[slot].num_contexts = num_ctx;
        suspended_[slot].occupied = true;
        ++num_suspended_;
        *active_tasks_ptr_ = nullptr;
      }
    }

#if HSHM_IS_GPU_COMPILER
    __syncwarp();
#endif

    return did_work;
  }

  /**
   * Signal the worker to stop polling.
   */
  HSHM_GPU_FUN void Stop() {
    is_running_ = false;
  }

  /**
   * Finalize and cleanup the worker.
   */
  HSHM_GPU_FUN void Finalize() {
    is_running_ = false;
  }

 private:
  // ================================================================
  // Context allocation
  // ================================================================

  /**
   * Allocate and initialize GpuRunContext array from the heap.
   * @param num_contexts 1 for lane-0-only, 32 for full warp parallelism
   */
  HSHM_GPU_FUN GpuRunContext *AllocContexts(
      Container *container, u32 method_id,
      hipc::FullPtr<Task> task_ptr, FutureShm *fshm,
      bool is_gpu2gpu, bool is_copy_path, u32 num_contexts) {
    auto *ipc = CHI_IPC;
    auto alloc_result = ipc->gpu_alloc_->AllocateObjs<GpuRunContext>(num_contexts);
    GpuRunContext *ctxs = alloc_result.ptr_;
    if (!ctxs) return nullptr;

    memset(ctxs, 0, num_contexts * sizeof(GpuRunContext));
    for (u32 i = 0; i < num_contexts; ++i) {
      ctxs[i].container = container;
      ctxs[i].method_id = method_id;
      ctxs[i].task_ptr = task_ptr;
      ctxs[i].fshm = fshm;
      ctxs[i].is_gpu2gpu = is_gpu2gpu;
      ctxs[i].is_copy_path = is_copy_path;
      ctxs[i].rctx = rctx_;
      ctxs[i].rctx.lane_id_ = i;
    }
    active_num_contexts_ = num_contexts;
    return ctxs;
  }

  /**
   * Destroy all coroutines in the context array and free it.
   */
  HSHM_GPU_FUN void FreeContexts(GpuRunContext *ctxs, u32 num_contexts) {
    for (u32 i = 0; i < num_contexts; ++i) {
      if (ctxs[i].coro) {
        ctxs[i].coro.destroy();
      }
    }
    auto *ipc = CHI_IPC;
    hipc::FullPtr<GpuRunContext> ptr(
        reinterpret_cast<hipc::Allocator *>(ipc->gpu_alloc_), ctxs);
    ipc->gpu_alloc_->Free(ptr);
  }

  // ================================================================
  // Suspended task management
  // ================================================================

  HSHM_GPU_FUN u32 FindFreeSlot() {
    for (u32 i = 0; i < kMaxSuspended; ++i) {
      if (!suspended_[i].occupied) return i;
    }
    return 0;
  }

  /**
   * Check suspended tasks. If any are ready to resume, set active_tasks_ptr_.
   * Called only by lane 0.
   */
  HSHM_GPU_FUN bool CheckAndResumeSuspended() {
    bool did_work = false;
    for (u32 i = 0; i < kMaxSuspended; ++i) {
      if (!suspended_[i].occupied) continue;

      GpuRunContext *ctxs = suspended_[i].contexts;
      u32 nc = suspended_[i].num_contexts;
      // Use lane 0's context as representative
      GpuRunContext &ctx0 = ctxs[0];

      // Check if the awaited sub-task is complete
      if (ctx0.rctx.awaited_fshm_) {
        auto *awaited = reinterpret_cast<FutureShm *>(ctx0.rctx.awaited_fshm_);
        if (!awaited->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
          continue;  // Sub-task not done yet
        }
        GPU_WORKER_DPRINTF("[W%u] Suspended[%u]: sub-task complete, resuming\n",
                           worker_id_, i);
        // Deserialize sub-task output before resuming
#if !HSHM_IS_HOST
        DeserializeAwaitedOutput(ctx0.rctx, awaited);
#endif
        ctx0.rctx.awaited_fshm_ = nullptr;
        ctx0.rctx.awaited_task_ = nullptr;
      }

      // Check if the top-level coroutine completed (from previous resume)
      if (ctx0.coro.done()) {
        // Completed while suspended — finalize
        DbgTaskCompleted();
        suspended_[i].occupied = false;
        --num_suspended_;
        if (ctx0.is_copy_path) {
          SerializeAndComplete(ctx0.fshm, ctx0.container,
                               ctx0.method_id, ctx0.task_ptr,
                               ctx0.is_gpu2gpu);
        } else {
          CompleteAndResumeParent(ctx0.fshm, ctx0.is_gpu2gpu);
        }
        FreeContexts(ctxs, nc);
        did_work = true;
        continue;
      }

      // Ready to resume: set as active and remove from suspended list
      DbgTaskResumed();
      *active_tasks_ptr_ = ctxs;
      active_num_contexts_ = nc;
      suspended_[i].occupied = false;
      --num_suspended_;
      did_work = true;
      break;  // Resume one at a time
    }
    return did_work;
  }

  // ================================================================
  // New task processing
  // ================================================================

  /**
   * Try to pop a new task from gpu2gpu or cpu2gpu queue.
   * On success, prepares GpuRunContext[32] and sets active_tasks_ptr_.
   * Called only by lane 0.
   */
  HSHM_GPU_FUN bool TryPopNewTask() {
    // Poll internal queue first (highest priority) to prevent deadlock:
    // orchestrator subtasks must drain before client tasks can unblock.
    if (TryPopFromQueue(internal_queue_, lane_id_, true)) return true;
    if (TryPopFromQueue(gpu2gpu_queue_, lane_id_, true)) return true;
    if (lane_id_ == 0) {
      if (TryPopFromQueue(cpu2gpu_queue_, 0, false)) return true;
    }
    return false;
  }

  HSHM_GPU_FUN bool TryPopFromQueue(TaskQueue *queue, u32 qlane,
                                      bool is_gpu2gpu) {
    if (!queue) return false;

    auto &lane = queue->GetLane(qlane, 0);

    // Debug: snapshot ring buffer head/tail via device-scope reads (bypass L1)
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      u64 h = lane.GetHeadDevice();
      u64 t = lane.GetTailDevice();
      if (queue == internal_queue_) {
        dbg_ctrl_->dbg_iq_head[worker_id_] = h;
        dbg_ctrl_->dbg_iq_tail[worker_id_] = t;
      } else if (is_gpu2gpu) {
        dbg_ctrl_->dbg_input_tw[worker_id_] = t;
        dbg_ctrl_->dbg_input_cs[worker_id_] = h;
        // Store queue pointer for verification (first time only)
        if (dbg_ctrl_->dbg_ser_total_written[worker_id_] == 0) {
          dbg_ctrl_->dbg_ser_total_written[worker_id_] =
              reinterpret_cast<unsigned long long>(queue);
        }
      }
    }

    Future<Task> future;
    if (is_gpu2gpu) {
      if (!lane.PopDevice(future)) return false;
    } else {
      if (!lane.Pop(future)) return false;
    }

    // Count every successful pop (before sptr check)
    DbgQueuePop();
    if (queue == internal_queue_ && dbg_ctrl_ &&
        worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_iq_pops[worker_id_]++;
    }

    // Resolve FutureShm
    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) return true;

    size_t off = sptr.off_.load();
    FutureShm *fshm;
    if (!is_gpu2gpu) {
      fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
    } else {
      fshm = reinterpret_cast<FutureShm *>(off);
    }
    if (!fshm) return true;

    // Fence: the client's SendGpu fenced before Push, and our PopDevice
    // used device-scope atomics. This fence ensures we see all of the
    // client's writes to the FutureShm fields (pool_id, method_id, flags,
    // copy_space data) that were flushed to L2 by the client's fence.
    hipc::threadfence();

    // Look up target container
    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
    Container *container = pool_mgr_->GetContainer(pool_id);
    if (!container) {
      DbgNoContainer(pool_id.major_, pool_id.minor_);
      CompleteAndResumeParent(fshm, is_gpu2gpu);
      return true;
    }

    DbgTaskPopped();
    bool is_copy = fshm->flags_.AnyDevice(FutureShm::FUTURE_COPY_FROM_CLIENT);
    if (is_copy) {
      return PrepareTaskCopy(fshm, container, method_id, is_gpu2gpu);
    } else {
      return PrepareTaskDirect(fshm, container, method_id, is_gpu2gpu);
    }
  }

  /**
   * Prepare a copy-path task: deserialize input, allocate contexts.
   * Sets *active_tasks_ptr_ on success.
   */
  HSHM_GPU_FUN void DbgStep(unsigned int step) {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_dispatch_step[worker_id_] = step;
    }
  }

  HSHM_GPU_FUN bool PrepareTaskCopy(FutureShm *fshm, Container *container,
                                      u32 method_id, bool is_gpu2gpu) {
#if !HSHM_IS_HOST
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;

    GPU_WORKER_DPRINTF("[W%u] PrepCopy: start method %u\n", worker_id_, method_id);

    // Set allocator on container
    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(alloc));

    // Deserialize input
    hshm::lbm::LbmContext in_ctx;
    in_ctx.copy_space = fshm->copy_space;
    in_ctx.shm_info_ = &fshm->input_;

    GPU_WORKER_DPRINTF("[W%u] PrepCopy: deserializing\n", worker_id_);
    auto *priv_alloc = CHI_PRIV_ALLOC;
    if (!priv_alloc) {
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }
    // Validate copy_space data: check total_written_ is sane
    size_t tw = fshm->input_.total_written_.load_device();
    size_t cs = fshm->input_.copy_space_size_.load_device();
    if (tw == 0 || tw > cs) {
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }
    LocalLoadTaskArchive load_ar(CHI_PRIV_ALLOC);
    if (is_gpu2gpu) {
      hshm::lbm::ShmTransport::RecvDevice(load_ar, in_ctx);
    } else {
      hshm::lbm::ShmTransport::Recv(load_ar, in_ctx);
    }
    load_ar.SetMsgType(LocalMsgType::kSerializeIn);

    hipc::FullPtr<Task> task_ptr =
        container->LocalAllocLoadTask(method_id, load_ar);
    if (task_ptr.IsNull()) {
      GPU_WORKER_DPRINTF("[W%u] PrepCopy: task alloc FAILED\n", worker_id_);
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }

    // Allocate GpuRunContexts based on method parallelism
    GPU_WORKER_DPRINTF("[W%u] PrepCopy: allocating contexts\n", worker_id_);
    u32 num_ctx = container->GetGpuParallelism(method_id);
    GpuRunContext *ctxs = AllocContexts(
        container, method_id, task_ptr, fshm, is_gpu2gpu, true, num_ctx);
    if (!ctxs) {
      DbgAllocFailure();
      GPU_WORKER_DPRINTF("[W%u] PrepCopy: context alloc FAILED\n", worker_id_);
      // Must signal completion so the client doesn't spin forever
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }

    GPU_WORKER_DPRINTF("[W%u] PrepCopy: done\n", worker_id_);
    *active_tasks_ptr_ = ctxs;
    return true;
#else
    return false;
#endif
  }

  /**
   * Prepare a direct-path task: no deserialization needed.
   * Sets *active_tasks_ptr_ on success.
   */
  HSHM_GPU_FUN bool PrepareTaskDirect(FutureShm *fshm, Container *container,
                                        u32 method_id, bool is_gpu2gpu) {
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_;

    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(alloc));

    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    u32 num_ctx = container->GetGpuParallelism(method_id);
    GpuRunContext *ctxs = AllocContexts(
        container, method_id, task_ptr, fshm, is_gpu2gpu, false, num_ctx);
    if (!ctxs) {
      DbgAllocFailure();
      if (is_gpu2gpu) {
        fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      } else {
        fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      }
      return true;
    }

    *active_tasks_ptr_ = ctxs;
    return true;
  }

  // ================================================================
  // Output serialization and completion
  // ================================================================

  HSHM_GPU_FUN void SerializeAndComplete(FutureShm *fshm, Container *container,
                                           u32 method_id,
                                           hipc::FullPtr<Task> &task_ptr,
                                           bool is_gpu2gpu) {
#if !HSHM_IS_HOST
    hshm::lbm::LbmContext out_ctx;
    out_ctx.copy_space = fshm->copy_space;
    out_ctx.shm_info_ = &fshm->output_;

    LocalSaveTaskArchive save_ar(LocalMsgType::kSerializeOut, CHI_PRIV_ALLOC);
    container->LocalSaveTask(method_id, save_ar, task_ptr);
    if (is_gpu2gpu) {
      hshm::lbm::ShmTransport::SendDevice(save_ar, out_ctx);
    } else {
      hshm::lbm::ShmTransport::Send(save_ar, out_ctx);
    }

    // Destroy the deserialized task via its derived destructor.
    // Task has a non-virtual destructor, so calling ~Task() directly would
    // skip derived members (priv::vector, priv::string), leaking gpu_alloc_.
    container->LocalDestroyTask(method_id, task_ptr);
    // Free task memory back to scratch allocator.
    // LocalAllocLoadTask allocated from gpu_alloc_ (container's allocator);
    // without this Free the scratch partition leaks on every completed task,
    // eventually exhausting the partition and causing SendGpu() to fail.
    CHI_IPC->gpu_alloc_->Free(task_ptr.template Cast<char>());
    hipc::threadfence();

    if (is_gpu2gpu) {
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }

    ResumeParentIfPresent(fshm);
#endif
  }

  HSHM_GPU_FUN void CompleteAndResumeParent(FutureShm *fshm,
                                              bool is_gpu2gpu) {
    // Fence before setting FUTURE_COMPLETE so that all prior writes
    // (task output, FutureShm fields) are visible to the waiter.
    if (is_gpu2gpu) {
      hipc::threadfence();
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
    } else {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
    }
    ResumeParentIfPresent(fshm);
  }

  /**
   * Signal that a sub-task is complete.
   *
   * Do NOT resume the parent coroutine here — the sub-task may have been
   * processed by a different warp than the one that owns the parent.
   * Cross-warp coroutine resumption is unsafe.  Instead, just mark the
   * FutureShm complete and let the parent warp's CheckAndResumeSuspended
   * pick it up on its next poll iteration.
   */
  HSHM_GPU_FUN void ResumeParentIfPresent(FutureShm *fshm) {
    (void)fshm;
    // The FutureShm FUTURE_COMPLETE flag was already set by the caller
    // (SerializeAndComplete / CompleteAndResumeParent).  The parent warp
    // will detect this in CheckAndResumeSuspended → DeserializeAwaitedOutput.
  }

#if !HSHM_IS_HOST
  HSHM_GPU_FUN void DeserializeAwaitedOutput(RunContext &rctx,
                                              FutureShm *awaited) {
    if (!rctx.awaited_task_ || awaited->output_.total_written_.load_device() == 0) {
      return;
    }
    hipc::threadfence();
    auto *sub_task = reinterpret_cast<Task *>(rctx.awaited_task_);
    Container *sub_container = pool_mgr_->GetContainer(sub_task->pool_id_);
    if (!sub_container) return;

    hshm::lbm::LbmContext ctx;
    ctx.copy_space = awaited->copy_space;
    ctx.shm_info_ = &awaited->output_;
    LocalLoadTaskArchive load_ar(CHI_PRIV_ALLOC);
    hshm::lbm::ShmTransport::RecvDevice(load_ar, ctx);
    hipc::FullPtr<Task> sub_task_ptr;
    sub_task_ptr.ptr_ = sub_task;
    sub_container->LocalLoadTaskOutput(
        sub_task->method_, load_ar, sub_task_ptr);
  }
#endif
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
