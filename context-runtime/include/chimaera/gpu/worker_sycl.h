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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_SYCL_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_SYCL_H_

#include "chimaera/gpu/container.h"
#include "chimaera/gpu/pool_manager.h"
#include "chimaera/gpu/work_orchestrator.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/task.h"
#include "hermes_shm/util/gpu_intrinsics.h"

#if HSHM_ENABLE_SYCL

namespace chi {
namespace gpu {

/**
 * SYCL twin of chi::gpu::Worker (worker.h).
 *
 * Differences from the CUDA worker:
 *   1. No CUDA Dynamic Parallelism. SYCL has no portable kernel-from-kernel
 *      mechanism, so popped tasks execute inline within the persistent
 *      orchestrator kernel — the task method is invoked through the
 *      Container function-pointer table directly. Per-task warp-level
 *      parallelism is a Phase 3b follow-up (will require nd_range launch
 *      + sub_group dispatch).
 *   2. No completion relay. CUDA CDP child writes are only visible to the
 *      parent kernel until the child exits, so the parent must relay
 *      completion to a pinned-host mirror. Under SYCL inline execution,
 *      the persistent kernel's writes are directly visible to the host
 *      via HSHM_DEVICE_FENCE_SYSTEM, so no relay is needed. The pending
 *      arrays are kept for API parity but unused on this path.
 *   3. All CUDA intrinsics (__threadfence, atomicOr, clock64, printf)
 *      are routed through HSHM_DEVICE_* wrappers from gpu_intrinsics.h.
 *
 * Same field layout, same method names, same call sites — so
 * work_orchestrator_sycl.cc mirrors work_orchestrator_gpu.cc structurally.
 */
class WorkerSycl {
 public:
  u32 worker_id_;
  bool is_running_;
  GpuTaskQueue *cpu2gpu_queue_;
  GpuTaskQueue *gpu2gpu_queue_;
  GpuTaskQueue *internal_queue_;
  PoolManager *pool_mgr_;
  char *queue_backend_base_;
  WorkOrchestratorControl *dbg_ctrl_;
  IpcManagerGpuInfo *gpu_info_ptr_;
  /** Kernel-scope IpcManager pointer captured by the orchestrator. Used to
   *  populate RunContext::ipc_mgr_ before each chimod dispatch so the
   *  chimod method body can resolve CHI_IPC. Set by the kernel after
   *  Init() and read inside TryPopFromQueue. */
  IpcManager *ipc_mgr_ = nullptr;

  // CPU→GPU completion relay (kept for symmetry with CUDA Worker).
  // Unused on this path: completion is signalled inline via system-scope
  // fence + atomic_ref<system> after Container::Run returns.
  static constexpr u32 kMaxPendingCpu2Gpu = 64;
  FutureShm *pending_device_fshm_[kMaxPendingCpu2Gpu];
  FutureShm *pending_host_fshm_[kMaxPendingCpu2Gpu];
  u32 num_pending_;

  long long prof_queue_pop_, prof_task_count_;

  void Init(u32 worker_id, GpuTaskQueue *cpu2gpu_queue,
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
    gpu_info_ptr_ = nullptr;
    is_running_ = true;
    num_pending_ = 0;
    for (u32 i = 0; i < kMaxPendingCpu2Gpu; ++i) {
      pending_device_fshm_[i] = nullptr;
      pending_host_fshm_[i] = nullptr;
    }
    prof_queue_pop_ = 0;
    prof_task_count_ = 0;
  }

  void FlushProfile() {
    if (!dbg_ctrl_ || worker_id_ >= WorkOrchestratorControl::kMaxDebugWorkers)
      return;
    dbg_ctrl_->prof_queue_pop[worker_id_] = prof_queue_pop_;
    dbg_ctrl_->prof_task_count[worker_id_] = prof_task_count_;
  }

  void Stop() { is_running_ = false; }
  void Finalize() { is_running_ = false; }

  // ================================================================
  // Main poll loop (single work-item)
  // ================================================================

  int PollGpu2Gpu() {
    int count = 0;
    for (int i = 0; i < 16; ++i) {
      count += TryPopFromQueue(gpu2gpu_queue_, worker_id_, true);
      count += TryPopFromQueue(internal_queue_, worker_id_, true);
    }
    return count;
  }

  int PollCpu2Gpu() {
    // No relay needed under SYCL inline execution; kept as no-op for
    // call-site parity with the CUDA worker.
    RelayPendingCompletions();
    return TryPopFromQueue(cpu2gpu_queue_, 0, false);
  }

  /**
   * No-op under SYCL: completion is published inline via system-scope
   * fence + atomic_ref<system> in MarkComplete(). The CUDA worker needs
   * this because CDP child kernel writes aren't host-visible.
   */
  void RelayPendingCompletions() {
    // Drain the pending arrays defensively: any entries here would
    // indicate a path that pre-populated them expecting CUDA semantics.
    num_pending_ = 0;
  }

  int PollOnce() {
    DbgPoll();
    int count = PollGpu2Gpu();
    if (worker_id_ == 0) {
      count += PollCpu2Gpu();
    }
    return count;
  }

 private:
  // ================================================================
  // Queue polling and inline task dispatch
  // ================================================================

  int TryPopFromQueue(GpuTaskQueue *queue, u32 qlane, bool is_gpu2gpu) {
    if (!queue) return 0;

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
    bool popped = lane.Pop(future);
    if (!popped) {
      return 0;
    }

    DbgQueuePop();

    if (queue == internal_queue_ && dbg_ctrl_ &&
        worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_iq_pops[worker_id_] =
          dbg_ctrl_->dbg_iq_pops[worker_id_] + 1;
    }

    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) {
      return 0;
    }

    size_t off = sptr.off_.load();
    FutureShm *fshm;
    FutureShm *host_fshm_for_relay = nullptr;
    if (sptr.alloc_id_ == FutureShm::GetCpu2GpuAllocId()) {
      // SendCpuToGpu: off = pinned-host FutureShm address.
      FutureShm *host_fshm = reinterpret_cast<FutureShm *>(off);
      HSHM_DEVICE_FENCE_SYSTEM();
      uintptr_t device_task = host_fshm->client_task_vaddr_;
      u32 task_size = host_fshm->task_size_;
      fshm = reinterpret_cast<FutureShm *>(device_task + task_size);
      host_fshm_for_relay = host_fshm;
    } else if (sptr.alloc_id_ == hipc::AllocatorId::GetNull()) {
      fshm = reinterpret_cast<FutureShm *>(off);
    } else if (!is_gpu2gpu) {
      fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
    } else {
      fshm = reinterpret_cast<FutureShm *>(off);
    }

    if (!fshm) {
      return 0;
    }

    HSHM_DEVICE_FENCE_DEVICE();

    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
    Container *container = pool_mgr_->GetContainer(pool_id);
    if (!container) {
      DbgNoContainer(pool_id.major_, pool_id.minor_);
      MarkComplete(fshm, is_gpu2gpu, host_fshm_for_relay);
      return 0;
    }

    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    DbgTaskPopped();
    ++prof_task_count_;

    // Inline execution (no CDP). Build RunContext on the stack and dispatch
    // through the Container function-pointer table. ipc_mgr_ propagates the
    // kernel-scope IpcManager into chimod methods so CHI_IPC under SYCL can
    // resolve to it (the chimod method body opens with
    // `auto *g_ipc_manager_ptr = rctx.ipc_mgr_;` to bind the local).
    RunContext rctx;
    rctx.container_ = container;
    rctx.method_id_ = method_id;
    rctx.task_ptr_ = task_ptr;
    rctx.task_fshm_ = fshm;
    rctx.parallelism_ = 1;
    rctx.ipc_mgr_ = ipc_mgr_;
    // Also stash the IpcManager pointer on the container so autogen-emitted
    // *Impl static methods (which don't receive rctx) can resolve CHI_IPC
    // via self_->ipc_mgr_.
    container->ipc_mgr_ = ipc_mgr_;

    if (!is_gpu2gpu) {
      container->FixupTask(method_id, task_ptr);
    }
    container->Run(method_id, task_ptr, rctx);

    // Completion: device-scope flag for GPU→GPU, system-scope for CPU→GPU
    // (so the host CPU that called Wait sees the bit).
    MarkComplete(fshm, is_gpu2gpu, host_fshm_for_relay);

    return 1;
  }

  // ================================================================
  // Completion helpers
  // ================================================================

  /**
   * Publish FUTURE_COMPLETE on the appropriate FutureShm.
   *
   * - GPU→GPU: device-scope atomic on the device FutureShm.
   * - CPU→GPU: system-scope atomic on the pinned-host FutureShm so the
   *   CPU caller of Wait() sees it. The device-side FutureShm
   *   (also at fshm) is updated for any GPU peer that races to read it.
   */
  void MarkComplete(FutureShm *fshm, bool is_gpu2gpu,
                    FutureShm *host_fshm_for_relay) {
    HSHM_DEVICE_FENCE_DEVICE();
    HSHM_DEVICE_ATOMIC_OR_U32_DEVICE(&fshm->flags_.bits_.x,
                                     FutureShm::FUTURE_COMPLETE);
    if (!is_gpu2gpu && host_fshm_for_relay) {
      HSHM_DEVICE_FENCE_SYSTEM();
      HSHM_DEVICE_ATOMIC_OR_U32_SYSTEM(&host_fshm_for_relay->flags_.bits_.x,
                                       FutureShm::FUTURE_COMPLETE);
      HSHM_DEVICE_FENCE_SYSTEM();
    } else {
      HSHM_DEVICE_FENCE_DEVICE();
    }
  }

  // ================================================================
  // Debug helpers — same as CUDA worker
  // ================================================================

  void DbgPoll() {
#ifndef NDEBUG
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers) {
      dbg_ctrl_->dbg_poll_count[worker_id_] =
          dbg_ctrl_->dbg_poll_count[worker_id_] + 1;
    }
#endif
  }

  void DbgTaskPopped() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_tasks_popped[worker_id_] =
          dbg_ctrl_->dbg_tasks_popped[worker_id_] + 1;
  }

  void DbgQueuePop() {
    if (dbg_ctrl_ && worker_id_ < WorkOrchestratorControl::kMaxDebugWorkers)
      dbg_ctrl_->dbg_queue_pops[worker_id_] =
          dbg_ctrl_->dbg_queue_pops[worker_id_] + 1;
  }

  void DbgNoContainer(unsigned int pool_major, unsigned int pool_minor) {
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

#endif  // HSHM_ENABLE_SYCL
#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_SYCL_H_
