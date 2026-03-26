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

#if CHI_GPU_WORKER_DEBUG
#define GPU_WORKER_DPRINTF(...) printf(__VA_ARGS__)
#else
#define GPU_WORKER_DPRINTF(...) ((void)0)
#endif

#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/gpu_work_orchestrator.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/task.h"
#include "hermes_shm/lightbeam/shm_transport.h"

namespace chi {
namespace gpu {

/**
 * GPU-side warp-level worker (one per warp, stack-local in orchestrator).
 *
 * Lane 0 acts as the scheduler: polling queues, managing suspended tasks,
 * and signaling completion. All 32 lanes participate in coroutine execution.
 */
class Worker {
 public:
  u32 worker_id_;
  u32 lane_id_;
  volatile bool is_running_;
  TaskQueue *cpu2gpu_queue_;
  TaskQueue *gpu2gpu_queue_;
  TaskQueue *internal_queue_;
  TaskQueue *warp_group_queue_;
  char *warp_group_queue_base_;
  PoolManager *pool_mgr_;
  char *queue_backend_base_;
  RunContext rctx_;
  WorkOrchestratorControl *dbg_ctrl_;

  using SuspendedQueue =
      hshm::ipc::ext_ring_buffer<RunContext *, CHI_PRIV_ALLOC_T>;
  SuspendedQueue *suspended_;
  u32 num_suspended_;

  HSHM_GPU_FUN void Init(u32 worker_id, u32 lane_id, TaskQueue *cpu2gpu_queue,
                         TaskQueue *gpu2gpu_queue, TaskQueue *internal_queue,
                         PoolManager *pool_mgr, char *queue_backend_base,
                         WorkOrchestratorControl *dbg_ctrl,
                         TaskQueue *warp_group_queue = nullptr,
                         char *warp_group_queue_base = nullptr) {
    worker_id_ = worker_id;
    lane_id_ = lane_id;
    cpu2gpu_queue_ = cpu2gpu_queue;
    gpu2gpu_queue_ = gpu2gpu_queue;
    internal_queue_ = internal_queue;
    warp_group_queue_ = warp_group_queue;
    warp_group_queue_base_ = warp_group_queue_base;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    dbg_ctrl_ = dbg_ctrl;
    is_running_ = true;
    num_suspended_ = 0;
    suspended_ = nullptr;
    u32 warp_id = IpcManager::GetWarpId();
    u32 warp_lane = IpcManager::GetLaneId();
    if (warp_lane == 0) {
      auto fp = CHI_IPC->NewObj<SuspendedQueue>(CHI_PRIV_ALLOC, 128);
      suspended_ = fp.ptr_;
    }
    rctx_ = RunContext(blockIdx.x, threadIdx.x, warp_id, warp_lane);
  }

  HSHM_GPU_FUN void Stop() { is_running_ = false; }
  HSHM_GPU_FUN void Finalize() { is_running_ = false; }

  // ================================================================
  // Main poll loop
  // ================================================================

  HSHM_GPU_FUN bool PollOnce(u32 lane_id) {
    if (lane_id == 0) {
      DbgPoll();
    }
    if (CheckAndResumeSuspended(lane_id)) return true;
    if (TryPopFromQueue(lane_id, internal_queue_, lane_id_, true)) return true;
    if (TryPopCrossWarpTask(lane_id)) return true;
    if (TryPopFromQueue(lane_id, gpu2gpu_queue_, lane_id_, true)) return true;
    if (lane_id_ == 0) {
      if (TryPopFromQueue(lane_id, cpu2gpu_queue_, 0, false)) return true;
    }
    return false;
  }

  // ================================================================
  // ExecTask: coroutine creation, execution, completion (all lanes)
  // ================================================================

  HSHM_GPU_FUN void ExecTask(u32 lane_id, RunContext *ctx) {
    // Broadcast ctx from lane 0 to all lanes
    {
      u64 ctx_bits = reinterpret_cast<u64>(ctx);
      u32 lo = static_cast<u32>(ctx_bits);
      u32 hi = static_cast<u32>(ctx_bits >> 32);
      lo = __shfl_sync(0xFFFFFFFF, lo, 0);
      hi = __shfl_sync(0xFFFFFFFF, hi, 0);
      ctx = reinterpret_cast<RunContext *>((static_cast<u64>(hi) << 32) | lo);
    }

    if (ctx == nullptr) return;

    // --- Coroutine creation and resume ---
    bool participate = (ctx->parallelism_ > 1) || (lane_id == 0);
    if (participate) {
      if (ctx->task_coros_[0] == nullptr) {
        auto *container = ctx->container_;
        TaskResume tmp = container->Run(ctx->method_id_, ctx->task_ptr_, *ctx);
        if (!tmp.get_handle()) {
          if (lane_id == 0) {
            printf("[W%u] CORO ALLOC FAILED: method=%u\n", worker_id_,
                   ctx->method_id_);
          }
        } else {
          tmp.get_handle().promise().set_run_context(ctx);
          tmp.get_handle().promise().set_lane_id(lane_id);
          ctx->task_coros_[lane_id] = tmp.release();
        }
        if (ctx->parallelism_ > 1) __syncwarp();
      }

      if (ctx->task_coros_[lane_id] && !ctx->task_coros_[lane_id].done()) {
        if (lane_id == 0) ctx->is_yielded_ = false;
        if (ctx->parallelism_ > 1) __syncwarp();

        auto &coro_h = ctx->coro_handles_[lane_id];
        if (coro_h && !coro_h.done()) {
          coro_h.resume();
        } else {
          ctx->task_coros_[lane_id].resume();
        }
      }

      // Destroy completed coroutine frames (all participating lanes together)
      if (ctx->task_coros_[lane_id] && ctx->task_coros_[lane_id].done()) {
        ctx->task_coros_[lane_id].destroy();
        ctx->task_coros_[lane_id] = nullptr;
        ctx->coro_handles_[lane_id] = nullptr;
      }
    }

    __syncwarp();

    // --- Lane 0: completion or suspension ---
    if (lane_id == 0) {
      EndTask(ctx);
    }

    __syncwarp();
  }

 private:
  // ================================================================
  // EndTask: handle completion or suspension (lane 0 only)
  // ================================================================

  HSHM_GPU_FUN void EndTask(RunContext *ctx) {
    bool task_done = true;
    u32 par = ctx->parallelism_ > kWarpSize ? kWarpSize : ctx->parallelism_;
    for (u32 i = 0; i < par; ++i) {
      if (ctx->task_coros_[i] != nullptr) {
        task_done = false;
        if (ctx->method_id_ >= 12) {
          printf("[EndTask] lane %u NOT done (method=%u par=%u)\n",
                 i, ctx->method_id_, par);
        }
        break;
      }
    }

    if (task_done) {
      DbgState(5);
      DbgTaskCompleted();
      GPU_WORKER_DPRINTF("[W%u] Task DONE: method %u\n", worker_id_,
                         ctx->method_id_);
      auto *fshm = ctx->task_fshm_;
      if (fshm->total_warps_ > 1) {
#if !HSHM_IS_HOST
        u32 prev = fshm->completion_counter_.fetch_add(1);
        __threadfence();
#else
        u32 prev = 0;
#endif
        if (prev + 1 < fshm->total_warps_) {
          FreeContext(ctx);
          return;
        }
      }
      if (ctx->is_copy_path_) {
        auto *container = ctx->container_;
        SerializeAndComplete(fshm, container, ctx->method_id_, ctx->task_ptr_,
                             ctx->is_gpu2gpu_);
      } else {
        CompleteAndResumeParent(fshm, ctx->is_gpu2gpu_);
      }
      FreeContext(ctx);
    } else {
      DbgState(3);
      GPU_WORKER_DPRINTF("[W%u] Task SUSPEND: method %u (nsus=%u)\n",
                         worker_id_, ctx->method_id_, num_suspended_ + 1);
      suspended_->Push(ctx);
      ++num_suspended_;
    }
  }

  // ================================================================
  // Context allocation / free
  // ================================================================

  HSHM_GPU_FUN RunContext *AllocContext(Container *container, u32 method_id,
                                        hipc::FullPtr<Task> task_ptr,
                                        FutureShm *fshm, bool is_gpu2gpu,
                                        bool is_copy_path, u32 parallelism,
                                        u32 range_off = 0,
                                        u32 range_width = 0) {
    auto *ipc = CHI_IPC;
    auto alloc_result = ipc->gpu_alloc_->AllocateObjs<RunContext>(1);
    RunContext *ctx = alloc_result.ptr_;
    if (!ctx) return nullptr;

    *ctx = rctx_;
    ctx->container_ = container;
    ctx->method_id_ = method_id;
    ctx->parallelism_ = parallelism;
    ctx->task_ptr_ = task_ptr;
    ctx->task_fshm_ = fshm;
    ctx->is_gpu2gpu_ = is_gpu2gpu;
    ctx->is_copy_path_ = is_copy_path;
    ctx->range_off_ = range_off;
    ctx->range_width_ = (range_width > 0) ? range_width : parallelism;
    ctx->is_yielded_ = false;
    ctx->awaited_fshm_ = nullptr;
    ctx->awaited_task_ = nullptr;
    for (u32 i = 0; i < kWarpSize; ++i) {
      ctx->task_coros_[i] = nullptr;
      ctx->coro_handles_[i] = nullptr;
    }
    return ctx;
  }

  HSHM_GPU_FUN void FreeContext(RunContext *ctx) {
    ctx->FreeFramesDirect();
    auto *ipc = CHI_IPC;
    hipc::FullPtr<RunContext> ptr(
        reinterpret_cast<hipc::Allocator *>(ipc->gpu_alloc_), ctx);
    ipc->gpu_alloc_->Free(ptr);
  }

  // ================================================================
  // Suspended task management (lane 0 only)
  // ================================================================

  HSHM_GPU_FUN bool CheckAndResumeSuspended(u32 lane_id) {
    RunContext *ctx = nullptr;
    if (lane_id == 0 && num_suspended_ > 0) {
      u32 count = num_suspended_;
      for (u32 i = 0; i < count; ++i) {
        RunContext *entry;
        if (!suspended_->Pop(entry)) break;

        if (entry->awaited_fshm_) {
          if (!entry->awaited_fshm_->flags_.AnyDevice(
                  FutureShm::FUTURE_COMPLETE)) {
            suspended_->Push(entry);
            continue;
          }
          entry->awaited_fshm_ = nullptr;
        }

        DbgTaskResumed();
        --num_suspended_;
        ctx = entry;
        break;
      }
    }
    ExecTask(lane_id, ctx);
    return ctx != nullptr;
  }

  // ================================================================
  // New task popping (lane 0 only)
  // ================================================================

  HSHM_GPU_FUN bool TryPopFromQueue(u32 lane_id, TaskQueue *queue, u32 qlane,
                                    bool is_gpu2gpu) {
    RunContext *ctx = nullptr;
    if (lane_id == 0 && queue) {
      auto &lane = queue->GetLane(qlane, 0);

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

      Future<Task> future;
      bool popped = is_gpu2gpu ? lane.PopDevice(future) : lane.Pop(future);
      if (popped) {
        DbgQueuePop();
        if (queue == internal_queue_ && dbg_ctrl_ &&
            worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
          dbg_ctrl_->dbg_iq_pops[worker_id_] =
              dbg_ctrl_->dbg_iq_pops[worker_id_] + 1;
        }

        hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
        if (!sptr.IsNull()) {
          size_t off = sptr.off_.load();
          FutureShm *fshm;
          if (!is_gpu2gpu) {
            fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
          } else {
            fshm = reinterpret_cast<FutureShm *>(off);
          }
          if (fshm) {
            hipc::threadfence();
            PoolId pool_id = fshm->pool_id_;
            u32 method_id = fshm->method_id_;
            Container *container = pool_mgr_->GetContainer(pool_id);
            if (!container) {
              DbgNoContainer(pool_id.major_, pool_id.minor_);
              CompleteAndResumeParent(fshm, is_gpu2gpu);
            } else {
              DbgTaskPopped();
              bool is_copy =
                  fshm->flags_.AnyDevice(FutureShm::FUTURE_COPY_FROM_CLIENT);
              if (is_copy) {
                ctx = PrepareTaskCopy(fshm, container, method_id, is_gpu2gpu);
              } else {
                ctx = PrepareTaskDirect(fshm, container, method_id, is_gpu2gpu);
              }
            }
          }
        }
      }
    }
    ExecTask(lane_id, ctx);
    return ctx != nullptr;
  }

  // ================================================================
  // Task preparation: allocate RunContext from queue entry
  // ================================================================

  HSHM_GPU_FUN RunContext *PrepareTaskDirect(FutureShm *fshm,
                                             Container *container,
                                             u32 method_id, bool is_gpu2gpu) {
    auto *ipc = CHI_IPC;
    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(ipc->gpu_alloc_));

    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    u32 parallelism = task_ptr.ptr_->pool_query_.GetParallelism();
    return AllocForParallelism(fshm, container, method_id, task_ptr, is_gpu2gpu,
                               false, parallelism);
  }

  HSHM_GPU_FUN RunContext *PrepareTaskCopy(FutureShm *fshm,
                                           Container *container, u32 method_id,
                                           bool is_gpu2gpu) {
    auto *ipc = CHI_IPC;
    container->gpu_alloc_ = reinterpret_cast<HSHM_DEFAULT_ALLOC_GPU_T *>(
        static_cast<void *>(ipc->gpu_alloc_));

    auto *priv_alloc = CHI_PRIV_ALLOC;
    if (!priv_alloc) {
      MarkComplete(fshm, is_gpu2gpu);
      return nullptr;
    }
    size_t tw = fshm->input_.total_written_.load_device();
    size_t cs = fshm->input_.copy_space_size_.load_device();
    if (tw == 0 || tw > cs) {
      MarkComplete(fshm, is_gpu2gpu);
      return nullptr;
    }

    hshm::lbm::LbmContext in_ctx;
    in_ctx.copy_space = fshm->copy_space;
    in_ctx.shm_info_ = &fshm->input_;
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
      MarkComplete(fshm, is_gpu2gpu);
      return nullptr;
    }

    u32 parallelism = task_ptr.ptr_->pool_query_.GetParallelism();
    return AllocForParallelism(fshm, container, method_id, task_ptr, is_gpu2gpu,
                               true, parallelism);
  }

  HSHM_GPU_FUN RunContext *AllocForParallelism(
      FutureShm *fshm, Container *container, u32 method_id,
      hipc::FullPtr<Task> task_ptr, bool is_gpu2gpu, bool is_copy_path,
      u32 parallelism) {
    if (parallelism <= 32) {
      RunContext *rctx =
          AllocContext(container, method_id, task_ptr, fshm, is_gpu2gpu,
                       is_copy_path, parallelism, 0, parallelism);
      if (!rctx) {
        DbgAllocFailure();
        MarkComplete(fshm, is_gpu2gpu);
        return nullptr;
      }
      return rctx;
    }
    // Cross-warp path
    u32 num_warps_needed = (parallelism + 31) / 32;
    fshm->total_warps_ = num_warps_needed;
    fshm->completion_counter_.store(0);

    RunContext *rctx = AllocContext(container, method_id, task_ptr, fshm,
                                    is_gpu2gpu, is_copy_path, 32, 0, 32);
    if (!rctx) {
      DbgAllocFailure();
      MarkComplete(fshm, is_gpu2gpu);
      return nullptr;
    }
    for (u32 w = 1; w < num_warps_needed; ++w) {
      u32 off = w * 32;
      u32 width = 32;
      if (off + width > parallelism) width = parallelism - off;
      PushCrossWarpSubTask(fshm, container, method_id, task_ptr, is_gpu2gpu,
                           is_copy_path, off, width);
    }
    return rctx;
  }

  // ================================================================
  // Cross-warp sub-task dispatch
  // ================================================================

  struct CrossWarpDescriptor {
    FutureShm *fshm;
    Container *container;
    hipc::FullPtr<Task> task_ptr;
    u32 method_id;
    u32 range_off;
    u32 range_width;
    u32 parallelism;
    bool is_gpu2gpu;
    bool is_copy_path;
  };

  HSHM_GPU_FUN void PushCrossWarpSubTask(FutureShm *fshm, Container *container,
                                         u32 method_id,
                                         hipc::FullPtr<Task> task_ptr,
                                         bool is_gpu2gpu, bool is_copy_path,
                                         u32 range_off, u32 range_width) {
    if (!warp_group_queue_) return;
    auto *ipc = CHI_IPC;
    auto desc_fp =
        ipc->gpu_alloc_->template AllocateObjs<CrossWarpDescriptor>(1);
    if (desc_fp.IsNull()) return;
    auto *desc = desc_fp.ptr_;
    desc->fshm = fshm;
    desc->container = container;
    desc->task_ptr = task_ptr;
    desc->method_id = method_id;
    desc->range_off = range_off;
    desc->range_width = range_width;
    desc->parallelism = range_width;
    desc->is_gpu2gpu = is_gpu2gpu;
    desc->is_copy_path = is_copy_path;

    Future<Task> future;
    hipc::ShmPtr<FutureShm> sptr;
    sptr.off_ = reinterpret_cast<size_t>(desc);
    sptr.alloc_id_ = hipc::AllocatorId::GetNull();
    future = Future<Task>(sptr);

    u32 num_lanes = warp_group_queue_->GetNumLanes();
    u32 lane = (range_off / 32) % num_lanes;
    auto &qlane = warp_group_queue_->GetLane(lane, 0);
    if (!qlane.Push(future)) {
      ipc->gpu_alloc_->Free(desc_fp);
    }
  }

  HSHM_GPU_FUN bool TryPopCrossWarpTask(u32 lane_id) {
    RunContext *ctx = nullptr;
    if (lane_id == 0 && warp_group_queue_) {
      u32 qlane = worker_id_ % warp_group_queue_->GetNumLanes();
      auto &qlane_ref = warp_group_queue_->GetLane(qlane, 0);

      Future<Task> future;
      if (qlane_ref.PopDevice(future)) {
        hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
        if (!sptr.IsNull()) {
          auto *desc =
              reinterpret_cast<CrossWarpDescriptor *>(sptr.off_.load());
          if (desc) {
            ctx = AllocContext(desc->container, desc->method_id, desc->task_ptr,
                               desc->fshm, desc->is_gpu2gpu, desc->is_copy_path,
                               desc->parallelism, desc->range_off,
                               desc->range_width);

            auto *ipc = CHI_IPC;
            hipc::FullPtr<CrossWarpDescriptor> fp(
                reinterpret_cast<hipc::Allocator *>(ipc->gpu_alloc_), desc);
            ipc->gpu_alloc_->Free(fp);
          }
        }
      }
    }
    ExecTask(lane_id, ctx);
    return ctx != nullptr;
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

  HSHM_GPU_FUN void SerializeAndComplete(FutureShm *fshm, Container *container,
                                         u32 method_id,
                                         hipc::FullPtr<Task> &task_ptr,
                                         bool is_gpu2gpu) {
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

    container->LocalDestroyTask(method_id, task_ptr);
    CHI_IPC->gpu_alloc_->Free(task_ptr.template Cast<char>());
    hipc::threadfence();
    MarkComplete(fshm, is_gpu2gpu);
    ResumeParentIfPresent(fshm);
  }

  HSHM_GPU_FUN void CompleteAndResumeParent(FutureShm *fshm, bool is_gpu2gpu) {
    if (is_gpu2gpu) {
      hipc::threadfence();
    }
    MarkComplete(fshm, is_gpu2gpu);
    ResumeParentIfPresent(fshm);
  }

  HSHM_GPU_FUN void ResumeParentIfPresent(FutureShm *fshm) { (void)fshm; }

  // ================================================================
  // Debug helpers
  // ================================================================

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
      dbg_ctrl_->dbg_poll_count[lane_id_] =
          dbg_ctrl_->dbg_poll_count[lane_id_] + 1;
    }
#endif
  }
  HSHM_GPU_FUN void DbgTaskPopped() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_tasks_popped[worker_id_] =
          dbg_ctrl_->dbg_tasks_popped[worker_id_] + 1;
  }
  HSHM_GPU_FUN void DbgTaskCompleted() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_tasks_completed[worker_id_] =
          dbg_ctrl_->dbg_tasks_completed[worker_id_] + 1;
#ifdef HSHM_BUDDY_ALLOC_DEBUG
      u32 completed = dbg_ctrl_->dbg_tasks_completed[worker_id_];
      if (completed % 50 == 0) {
        auto *priv = CHI_PRIV_ALLOC;
        if (priv) {
          printf(
              "[W%u] task#%u priv: allocs=%llu frees=%llu net=%llu "
              "bigheap=%llu/%llu\n",
              worker_id_, completed, (unsigned long long)priv->DbgAllocCount(),
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
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_tasks_resumed[worker_id_] =
          dbg_ctrl_->dbg_tasks_resumed[worker_id_] + 1;
  }
  HSHM_GPU_FUN void DbgAllocFailure() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_alloc_failures[worker_id_] =
          dbg_ctrl_->dbg_alloc_failures[worker_id_] + 1;
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
  HSHM_GPU_FUN void DbgStep(unsigned int step) {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_dispatch_step[worker_id_] = step;
  }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
