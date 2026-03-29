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
#define GPU_WORKER_TIMER_DEF(var) long long var = 0
#define GPU_WORKER_TIMER_START(var) var = clock64()
#define GPU_WORKER_TIMER_END(counter, var) counter += clock64() - var
#else
#define GPU_WORKER_DPRINTF(...) ((void)0)
#define GPU_WORKER_TIMER_DEF(var) ((void)0)
#define GPU_WORKER_TIMER_START(var) ((void)0)
#define GPU_WORKER_TIMER_END(counter, var) ((void)0)
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
  RunContext *cached_rctx_;  /**< Cached RunContext to avoid per-task alloc */

  // Profiling counters (lane 0 only, flushed to pinned host memory on exit)
  long long prof_queue_pop_, prof_recv_device_, prof_alloc_task_;
  long long prof_load_task_, prof_alloc_ctx_, prof_coro_create_;
  long long prof_coro_resume_, prof_coro_destroy_, prof_save_task_;
  long long prof_send_device_, prof_complete_, prof_task_count_;
  // AllocContext sub-breakdown
  long long prof_ctx_alloc_, prof_ctx_copy_, prof_ctx_zero_;


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
    rctx_ = RunContext();
    cached_rctx_ = nullptr;
    prof_queue_pop_ = 0; prof_recv_device_ = 0; prof_alloc_task_ = 0;
    prof_load_task_ = 0; prof_alloc_ctx_ = 0; prof_coro_create_ = 0;
    prof_coro_resume_ = 0; prof_coro_destroy_ = 0; prof_save_task_ = 0;
    prof_send_device_ = 0; prof_complete_ = 0; prof_task_count_ = 0;
    prof_ctx_alloc_ = 0; prof_ctx_copy_ = 0; prof_ctx_zero_ = 0;
  }

  HSHM_GPU_FUN void FlushProfile() {
    if (!dbg_ctrl_ || worker_id_ >= WorkOrchestratorControl::kMaxDebugWorkers) return;
    dbg_ctrl_->prof_queue_pop[worker_id_] = prof_queue_pop_;
    dbg_ctrl_->prof_recv_device[worker_id_] = prof_recv_device_;
    dbg_ctrl_->prof_alloc_task[worker_id_] = prof_alloc_task_;
    dbg_ctrl_->prof_load_task[worker_id_] = prof_load_task_;
    dbg_ctrl_->prof_alloc_ctx[worker_id_] = prof_alloc_ctx_;
    dbg_ctrl_->prof_coro_create[worker_id_] = prof_coro_create_;
    dbg_ctrl_->prof_coro_resume[worker_id_] = prof_coro_resume_;
    dbg_ctrl_->prof_coro_destroy[worker_id_] = prof_coro_destroy_;
    dbg_ctrl_->prof_save_task[worker_id_] = prof_save_task_;
    dbg_ctrl_->prof_send_device[worker_id_] = prof_send_device_;
    dbg_ctrl_->prof_complete[worker_id_] = prof_complete_;
    dbg_ctrl_->prof_task_count[worker_id_] = prof_task_count_;
    dbg_ctrl_->prof_ctx_alloc[worker_id_] = prof_ctx_alloc_;
    dbg_ctrl_->prof_ctx_copy[worker_id_] = prof_ctx_copy_;
    dbg_ctrl_->prof_ctx_zero[worker_id_] = prof_ctx_zero_;
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
    // cpu2gpu_queue only polled by warp 0 — pass null for other warps
    // so all lanes still enter TryPopFromQueue for __shfl_sync convergence
    if (TryPopFromQueue(lane_id, lane_id_ == 0 ? cpu2gpu_queue_ : nullptr,
                        0, false)) return true;
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
    GPU_WORKER_TIMER_DEF(_etc);
    bool participate = (ctx->parallelism_ > 1) || (lane_id == 0);
    if (participate) {
      if (ctx->task_coros_[0] == nullptr) {
        if (lane_id == 0) { GPU_WORKER_TIMER_START(_etc); }
        auto *container = ctx->container_;
        TaskResume tmp = container->Run(ctx->method_id_, ctx->task_ptr_, *ctx);
        if (!tmp.get_handle()) {
          if (lane_id == 0) {
            GPU_WORKER_DPRINTF("[W%u] CORO ALLOC FAILED: method=%u\n",
                               worker_id_, ctx->method_id_);
          }
        } else {
          tmp.get_handle().promise().set_run_context(ctx);
          tmp.get_handle().promise().set_lane_id(lane_id);
          ctx->task_coros_[lane_id] = tmp.release();
        }
        if (ctx->parallelism_ > 1) __syncwarp();
        if (lane_id == 0) { GPU_WORKER_TIMER_END(prof_coro_create_, _etc); }
      }

      if (ctx->task_coros_[lane_id] && !ctx->task_coros_[lane_id].done()) {
        if (lane_id == 0) { ctx->is_yielded_ = false; GPU_WORKER_TIMER_START(_etc); }
        if (ctx->parallelism_ > 1) __syncwarp();

        auto &coro_h = ctx->coro_handles_[lane_id];
        if (coro_h && !coro_h.done()) {
          coro_h.resume();
        } else {
          ctx->task_coros_[lane_id].resume();
        }
        if (lane_id == 0) { GPU_WORKER_TIMER_END(prof_coro_resume_, _etc); }
      }

      // Destroy completed coroutine frames (all participating lanes together)
      if (lane_id == 0) { GPU_WORKER_TIMER_START(_etc); }
      if (ctx->task_coros_[lane_id] && ctx->task_coros_[lane_id].done()) {
        ctx->task_coros_[lane_id].destroy();
        ctx->task_coros_[lane_id] = nullptr;
        ctx->coro_handles_[lane_id] = nullptr;
      }
      if (lane_id == 0) { GPU_WORKER_TIMER_END(prof_coro_destroy_, _etc); }
    }

    __syncwarp();

    // --- Lane 0: detect completion or suspension ---
    int needs_serde = 0;
    if (lane_id == 0) {
      needs_serde = EndTask(ctx);
    }

    // Warp-parallel serialization if needed
    needs_serde = __shfl_sync(0xFFFFFFFF, needs_serde, 0);
    if (needs_serde) {
      SerializeAndComplete(lane_id, ctx->task_fshm_, ctx->container_,
                           ctx->method_id_, ctx->task_ptr_, ctx->is_gpu2gpu_);
      if (lane_id == 0) {
        FreeContext(ctx);
      }
    }

    __syncwarp();
  }

 private:
  // ================================================================
  // EndTask: handle completion or suspension (lane 0 only)
  // ================================================================

  HSHM_GPU_FUN int EndTask(RunContext *ctx) {
    bool task_done = true;
    u32 par = ctx->parallelism_ > kWarpSize ? kWarpSize : ctx->parallelism_;
    for (u32 i = 0; i < par; ++i) {
      if (ctx->task_coros_[i] != nullptr) {
        task_done = false;
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
          return 0;
        }
      }
      if (ctx->is_copy_path_) {
        // Caller does warp-parallel SerializeAndComplete + FreeContext
        return 1;
      } else {
        CompleteAndResumeParent(fshm, ctx->is_gpu2gpu_);
        FreeContext(ctx);
        return 0;
      }
    } else {
      DbgState(3);
      GPU_WORKER_DPRINTF("[W%u] Task SUSPEND: method %u (nsus=%u)\n",
                         worker_id_, ctx->method_id_, num_suspended_ + 1);
      suspended_->Push(ctx);
      ++num_suspended_;
      return 0;
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
    static constexpr size_t kStackSize = 4096;
    GPU_WORKER_TIMER_DEF(_actc);
    GPU_WORKER_TIMER_START(_actc);
    auto *ipc = CHI_IPC;
    auto *priv = ipc->GetPrivAlloc();
    // Single allocation: RunContext + stack region contiguous
    size_t total = sizeof(RunContext) + kStackSize;
    auto alloc_result = priv->template AllocateObjs<char>(total);
    if (alloc_result.IsNull()) return nullptr;
    RunContext *ctx = reinterpret_cast<RunContext *>(alloc_result.ptr_);
    new (ctx) RunContext();
    char *stack = alloc_result.ptr_ + sizeof(RunContext);
    ctx->InitStack(stack, kStackSize);
    GPU_WORKER_TIMER_END(prof_ctx_alloc_, _actc);
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
    return ctx;
  }

  HSHM_GPU_FUN void FreeContext(RunContext *ctx) {
    // RunContext is co-allocated with the Task — freed by DelTask.
    // Just reset coroutine frame state (stack frames freed in bulk).
    ctx->FreeFramesDirect();
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
    // Broadcast result to all lanes to avoid warp divergence
    int found = (ctx != nullptr) ? 1 : 0;
    found = __shfl_sync(0xFFFFFFFF, found, 0);
    return found != 0;
  }

  // ================================================================
  // New task popping (lane 0 only)
  // ================================================================

  HSHM_GPU_FUN bool TryPopFromQueue(u32 lane_id, TaskQueue *queue, u32 qlane,
                                    bool is_gpu2gpu) {
    RunContext *ctx = nullptr;

    // Lane 0: pop queue, validate, look up container
    unsigned long long fshm_ull = 0, container_ull = 0;
    u32 method_id = 0;
    int is_copy = 0, need_prepare = 0;

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

      GPU_WORKER_TIMER_DEF(_qpop_tc);
      GPU_WORKER_TIMER_START(_qpop_tc);
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
            method_id = fshm->method_id_;
            Container *container = pool_mgr_->GetContainer(pool_id);
            if (!container) {
              DbgNoContainer(pool_id.major_, pool_id.minor_);
              CompleteAndResumeParent(fshm, is_gpu2gpu);
            } else {
              DbgTaskPopped();
              GPU_WORKER_TIMER_END(prof_queue_pop_, _qpop_tc);
              fshm_ull = reinterpret_cast<unsigned long long>(fshm);
              container_ull = reinterpret_cast<unsigned long long>(container);
              is_copy = fshm->flags_.AnyDevice(
                  FutureShm::FUTURE_COPY_FROM_CLIENT) ? 1 : 0;
              need_prepare = 1;
            }
          }
        }
      }
    }

    // Broadcast for warp-parallel PrepareTaskCopy
    fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);
    container_ull = hipc::shfl_sync_u64(0xFFFFFFFF, container_ull, 0);
    method_id = __shfl_sync(0xFFFFFFFF, method_id, 0);
    is_copy = __shfl_sync(0xFFFFFFFF, is_copy, 0);
    need_prepare = __shfl_sync(0xFFFFFFFF, need_prepare, 0);

    if (need_prepare) {
      auto *fshm = reinterpret_cast<FutureShm *>(fshm_ull);
      auto *container = reinterpret_cast<Container *>(container_ull);
      if (is_copy) {
        ctx = PrepareTaskCopy(lane_id, fshm, container, method_id, is_gpu2gpu);
      } else if (lane_id == 0) {
        ctx = PrepareTaskDirect(fshm, container, method_id, is_gpu2gpu);
      }
    }

    ExecTask(lane_id, ctx);
    // Use need_prepare (broadcast to all lanes) instead of ctx != nullptr.
    // ctx is only non-null on lane 0, so using it would cause warp divergence
    // in PollOnce's early-return path, breaking __shfl_sync convergence.
    return need_prepare != 0;
  }

  // ================================================================
  // Task preparation: allocate RunContext from queue entry
  // ================================================================

  HSHM_GPU_FUN RunContext *PrepareTaskDirect(FutureShm *fshm,
                                             Container *container,
                                             u32 method_id, bool is_gpu2gpu) {
    auto *ipc = CHI_IPC;
    // gpu_alloc_ accessed via CHI_IPC->gpu_alloc_ directly

    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    u32 parallelism = task_ptr.ptr_->pool_query_.GetParallelism();
    return AllocForParallelism(fshm, container, method_id, task_ptr, is_gpu2gpu,
                               false, parallelism);
  }

  HSHM_GPU_FUN RunContext *PrepareTaskCopy(u32 lane_id, FutureShm *fshm,
                                           Container *container, u32 method_id,
                                           bool is_gpu2gpu) {
    GPU_WORKER_TIMER_DEF(_tc); GPU_WORKER_TIMER_DEF(_tc2); GPU_WORKER_TIMER_DEF(_tc3);
    auto *ipc = CHI_IPC;

    // Phase 1 (lane 0): validate + read PreallocHeader
    int valid = 0;
    PreallocHeader hdr;

    if (lane_id == 0) {
      auto *priv_alloc = CHI_PRIV_ALLOC;
      if (!priv_alloc) {
        MarkComplete(fshm, is_gpu2gpu);
      } else {
        size_t tw = fshm->input_.total_written_.load_device();
        size_t cs = fshm->input_.copy_space_size_.load_device();
        if (tw == 0 || tw > cs) {
          MarkComplete(fshm, is_gpu2gpu);
        } else {
          // Read PreallocHeader from copy_space
          hshm::lbm::LbmContext in_ctx;
          in_ctx.copy_space = fshm->copy_space;
          in_ctx.shm_info_ = &fshm->input_;
          hshm::lbm::ShmTransport::RecvDevicePrealloc(
              reinterpret_cast<char*>(&hdr), sizeof(hdr), in_ctx);
          valid = 1;
        }
      }
    }

    valid = __shfl_sync(0xFFFFFFFF, valid, 0);
    if (!valid) return nullptr;

    if (lane_id == 0) { GPU_WORKER_TIMER_START(_tc); }

    // Phase 2 (lane 0): single alloc for Task + RunContext + stack
    static constexpr size_t kStackSize = 4096;

    TaskContextBlock block = {hipc::FullPtr<Task>::GetNull(), nullptr};

    // Phase 2 (lane 0): single dispatch — alloc + deserialize
    if (lane_id == 0) {
      GPU_WORKER_TIMER_END(prof_recv_device_, _tc);

      GPU_WORKER_TIMER_START(_tc2);
      hipc::FullPtr<char> data_fp;
      data_fp.ptr_ = fshm->copy_space + IpcManager::WarpIpcManager::kHeaderSize;
      data_fp.shm_.alloc_id_.SetNull();
      data_fp.shm_.off_ = reinterpret_cast<size_t>(data_fp.ptr_);
      hshm::priv::wrap_vector recv_buf(data_fp, hdr.data_size);
      recv_buf.resize(hdr.data_size);
      WrapLoadArchive load_ar(recv_buf);

      block = container->LocalAllocLoadDeser(method_id, kStackSize, load_ar);
      GPU_WORKER_TIMER_END(prof_alloc_task_, _tc2);

      if (block.task_ptr.IsNull()) {
        MarkComplete(fshm, is_gpu2gpu);
      }
    }

    if (block.task_ptr.IsNull() && lane_id == 0) return nullptr;

    // Phase 3 (lane 0): fill in RunContext fields
    RunContext *result = nullptr;
    if (lane_id == 0) {
      GPU_WORKER_TIMER_START(_tc3);
      RunContext *ctx = block.rctx;
      u32 parallelism = block.task_ptr.ptr_->pool_query_.GetParallelism();
      ctx->container_ = container;
      ctx->method_id_ = method_id;
      ctx->parallelism_ = parallelism;
      ctx->task_ptr_ = block.task_ptr;
      ctx->task_fshm_ = fshm;
      ctx->is_gpu2gpu_ = is_gpu2gpu;
      ctx->is_copy_path_ = true;
      ctx->range_off_ = 0;
      ctx->range_width_ = parallelism;
      ctx->is_yielded_ = false;
      ctx->awaited_fshm_ = nullptr;
      ctx->awaited_task_ = nullptr;
      result = ctx;
      GPU_WORKER_TIMER_END(prof_alloc_ctx_, _tc3);
    }
    return result;
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
    u32 push_count = 0;
    for (u32 w = 1; w < num_warps_needed; ++w) {
      u32 off = w * 32;
      u32 width = 32;
      if (off + width > parallelism) width = parallelism - off;
      PushCrossWarpSubTask(fshm, container, method_id, task_ptr, is_gpu2gpu,
                           is_copy_path, off, width);
      push_count++;
    }
    GPU_WORKER_DPRINTF("[CrossWarp] pushed %u sub-tasks for %u warps\n",
                       push_count, num_warps_needed);
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
      GPU_WORKER_DPRINTF("[CrossWarp] PUSH FAILED lane=%u\n", lane);
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
    // Broadcast result to all lanes to avoid warp divergence
    int found = (ctx != nullptr) ? 1 : 0;
    found = __shfl_sync(0xFFFFFFFF, found, 0);
    return found != 0;
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

  HSHM_GPU_FUN void SerializeAndComplete(u32 lane_id, FutureShm *fshm,
                                         Container *container, u32 method_id,
                                         hipc::FullPtr<Task> &task_ptr,
                                         bool is_gpu2gpu) {
    GPU_WORKER_TIMER_DEF(_tc); GPU_WORKER_TIMER_DEF(_tc2); GPU_WORKER_TIMER_DEF(_tc3);
    auto *ipc = CHI_IPC;

    // Phase 1 (lane 0): serialize task output directly into copy_space
    unsigned long long ctx_cs = 0, ctx_si = 0;
    unsigned int data_size = 0;

    if (lane_id == 0) {
      GPU_WORKER_TIMER_START(_tc);
      // Rebind buffer to CLIENT's copy_space and serialize output there
      auto *mgr = ipc->GetWarpManager();
      mgr->BindCopySpace(fshm->copy_space);
      auto *save_ar_ptr = &mgr->save_ar_;
      save_ar_ptr->Reset(LocalMsgType::kSerializeOut);
      container->LocalSaveTask(method_id, *save_ar_ptr, task_ptr);
      GPU_WORKER_TIMER_END(prof_save_task_, _tc);

      ctx_cs = reinterpret_cast<unsigned long long>(fshm->copy_space);
      ctx_si = reinterpret_cast<unsigned long long>(&fshm->output_);
      data_size = static_cast<unsigned int>(mgr->buffer_.size());
    }

    // Broadcast for SendDevicePrealloc
    ctx_cs = hipc::shfl_sync_u64(0xFFFFFFFF, ctx_cs, 0);
    ctx_si = hipc::shfl_sync_u64(0xFFFFFFFF, ctx_si, 0);
    data_size = __shfl_sync(0xFFFFFFFF, data_size, 0);

    // Phase 2: Write PreallocHeader + mark ready
    if (lane_id == 0) { GPU_WORKER_TIMER_START(_tc2); }
    if (is_gpu2gpu) {
      PreallocHeader hdr;
      hdr.msg_type = LocalMsgType::kSerializeOut;
      hdr.data_size = data_size;
      hshm::lbm::LbmContext out_ctx;
      out_ctx.copy_space = reinterpret_cast<char *>(ctx_cs);
      out_ctx.shm_info_ = reinterpret_cast<hshm::lbm::ShmTransferInfo *>(ctx_si);
      hshm::lbm::ShmTransport::SendDevicePrealloc(
          reinterpret_cast<const char*>(&hdr), sizeof(hdr),
          hdr.data_size, out_ctx);
    } else if (lane_id == 0) {
      auto *mgr = ipc->GetWarpManager();
      hshm::lbm::LbmContext out_ctx;
      out_ctx.copy_space = reinterpret_cast<char *>(ctx_cs);
      out_ctx.shm_info_ = reinterpret_cast<hshm::lbm::ShmTransferInfo *>(ctx_si);
      hshm::lbm::ShmTransport::Send(mgr->save_ar_, out_ctx);
    }
    __syncwarp();
    if (lane_id == 0) {
      GPU_WORKER_TIMER_END(prof_send_device_, _tc2);
    }

    // Phase 3 (lane 0): cleanup + mark complete
    if (lane_id == 0) {
      GPU_WORKER_TIMER_START(_tc3);
      container->LocalDestroyTask(method_id, task_ptr);
      ipc->DelTask(task_ptr);
      hipc::threadfence();
      MarkComplete(fshm, is_gpu2gpu);
      ResumeParentIfPresent(fshm);
      GPU_WORKER_TIMER_END(prof_complete_, _tc3);
      ++prof_task_count_;
    }
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
