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

#include "chimaera/gpu_container.h"
#include "chimaera/gpu_pool_manager.h"
#include "chimaera/task.h"
#include "chimaera/local_task_archives.h"
#include "hermes_shm/lightbeam/shm_transport.h"

namespace chi {
namespace gpu {

/**
 * GPU-side worker that mirrors the CPU Worker API.
 *
 * Runs on block 0, thread 0 of the GPU work orchestrator. Polls both the
 * CPU→GPU queue (cpu2gpu_queue) and the GPU→GPU queue (gpu2gpu_queue)
 * for incoming tasks, deserializes inputs, dispatches to GPU containers,
 * serializes outputs, and signals completion.
 *
 * No STL, no TLS — all HSHM_GPU_FUN.  Container::Run() returns
 * chi::gpu::TaskResume (a C++20 coroutine) enabling cooperative yielding.
 */
class Worker {
 public:
  u32 worker_id_;                    /**< Worker identity */
  volatile bool is_running_;         /**< Running flag for the poll loop */
  TaskQueue *cpu2gpu_queue_;         /**< CPU → GPU queue (GPU work orchestrator polls) */
  TaskQueue *gpu2gpu_queue_;         /**< GPU → GPU queue (GPU work orchestrator polls) */
  PoolManager *pool_mgr_;            /**< GPU-side container lookup table */
  char *queue_backend_base_;         /**< Base of queue backend for ShmPtr resolution */
  RunContext rctx_;               /**< Reused run context per task */

  /**
   * Initialize the worker with queue and pool manager pointers.
   *
   * @param worker_id Logical worker ID
   * @param cpu2gpu_queue CPU→GPU queue pointer (pinned host memory)
   * @param gpu2gpu_queue GPU→GPU queue pointer (device memory)
   * @param pool_mgr GPU-side pool manager for container lookup
   * @param queue_backend_base Base pointer of queue backend for ShmPtr offsets
   */
  HSHM_GPU_FUN void Init(u32 worker_id,
                          TaskQueue *cpu2gpu_queue,
                          TaskQueue *gpu2gpu_queue,
                          PoolManager *pool_mgr,
                          char *queue_backend_base) {
    worker_id_ = worker_id;
    cpu2gpu_queue_ = cpu2gpu_queue;
    gpu2gpu_queue_ = gpu2gpu_queue;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    is_running_ = true;
#if HSHM_IS_GPU_COMPILER
    rctx_ = RunContext(blockIdx.x, threadIdx.x);
#else
    rctx_ = RunContext(0, 0);
#endif
  }

  /**
   * Process one iteration of the poll loop.
   *
   * Checks both queues for pending work. Returns true if any work was done,
   * false if both queues were empty (caller can decide to spin or yield).
   *
   * @return true if at least one task was processed
   */
  HSHM_GPU_FUN bool PollOnce() {
    bool did_work = false;
    did_work |= ProcessNewTask(cpu2gpu_queue_);
    did_work |= ProcessNewTask(gpu2gpu_queue_);
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
   * Currently a no-op; placeholder for future resource release.
   */
  HSHM_GPU_FUN void Finalize() {
    is_running_ = false;
  }

 private:
  /**
   * Try to pop and process one task from the given queue.
   *
   * Steps:
   * 1. Pop a Future<Task> from lane (0,0)
   * 2. Resolve FutureShm via queue_backend_base_
   * 3. Look up the target container in pool_mgr_
   * 4. Deserialize input, dispatch Run(), serialize output
   * 5. Mark FUTURE_COMPLETE
   *
   * @param queue Queue to pop from
   * @return true if a task was processed, false if queue was empty
   */
  HSHM_GPU_FUN bool ProcessNewTask(TaskQueue *queue) {
    if (!queue) return false;

    // Try to pop from lane (0, 0)
    auto &lane = queue->GetLane(0, 0);
    Future<Task> future;
    if (!lane.Pop(future)) {
      return false;
    }

    // Resolve FutureShm: queue-dependent resolution
    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) {
      return true;
    }
    size_t off = sptr.off_.load();
    FutureShm *fshm;
    if (queue == cpu2gpu_queue_) {
      // CPU→GPU (SendToGpu): relative offset from queue backend (pinned host)
      fshm = reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
    } else {
      // GPU→GPU: absolute UVA pointer (device memory or pinned host via UVA)
      fshm = reinterpret_cast<FutureShm *>(off);
    }
    if (!fshm) {
      return true;  // Consumed slot but bad pointer
    }

    // Look up the target container
    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
    Container *container = pool_mgr_->GetContainer(pool_id);
    if (!container) {
      // No container registered — mark complete with no output
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      return true;
    }

    if (fshm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT)) {
      DispatchTask(fshm, container, method_id);
    } else {
      DispatchTaskDirect(fshm, container, method_id);
    }
    return true;
  }

  /**
   * Run a task directly from its pointer without (de)serialization.
   *
   * Used by the SendGpuForward path where the task object is already in
   * GPU-accessible memory. Results are written in-place to the task object;
   * the caller reads them after FUTURE_COMPLETE is observed.
   *
   * @param fshm FutureShm with client_task_vaddr_ set to absolute Task*
   * @param container Target GPU container
   * @param method_id Method to dispatch
   */
  HSHM_GPU_FUN void DispatchTaskDirect(FutureShm *fshm, Container *container,
                                        u32 method_id) {
    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_table_[ipc->GetGpuThreadId()];

    // Set allocator on container for cross-library calls
    container->gpu_alloc_ = alloc;

    // Reconstruct FullPtr from absolute UVA pointer stored in client_task_vaddr_
    hipc::FullPtr<Task> task_ptr;
    task_ptr.ptr_ = reinterpret_cast<Task *>(fshm->client_task_vaddr_);
    task_ptr.shm_.off_ = fshm->client_task_vaddr_;
    task_ptr.shm_.alloc_id_ = hipc::AllocatorId::GetNull();

    // Execute the task — results are written into task_ptr in-place.
    // Run() returns a TaskResume coroutine; resume it to execute the body,
    // then let the destructor clean up the coroutine frame.
    {
      TaskResume coro = container->Run(method_id, task_ptr, rctx_);
      coro.resume();
      // TODO: handle yielding coroutines (coro.done() == false)
    }

    // System-scope OR: fence + atomicOr_system ensures all results written
    // above are globally visible to CPU before FUTURE_COMPLETE is observed.
    fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);

    // Reset the orchestrator's arena. For the forward (in-place) path, the
    // task lives in the client's memory (task_ptr is a UVA pointer, not in
    // this arena), so there is no arena-allocated task to worry about.
    // NOTE: safe only when one task is processed per thread at a time.
    alloc->Reset();
  }

  /**
   * Deserialize input, run container method, serialize output, and complete.
   *
   * @param fshm FutureShm containing I/O ring buffers
   * @param container Target GPU container
   * @param method_id Method to dispatch
   */
  HSHM_GPU_FUN void DispatchTask(FutureShm *fshm, Container *container,
                                  u32 method_id) {
#if !HSHM_IS_HOST
    // Step 1: Deserialize input from FutureShm ring buffer
    hshm::lbm::LbmContext in_ctx;
    in_ctx.copy_space = fshm->copy_space;
    in_ctx.shm_info_ = &fshm->input_;

    auto *ipc = CHI_IPC;
    int thread_id = ipc->GetGpuThreadId();
    auto *alloc = ipc->gpu_alloc_table_[thread_id];

    // Set allocator on container for cross-library calls
    container->gpu_alloc_ = alloc;

    LocalLoadTaskArchive load_ar(CHI_GPU_HEAP);
    hshm::lbm::ShmTransport::Recv(load_ar, in_ctx);
    load_ar.SetMsgType(LocalMsgType::kSerializeIn);

    // Step 2: Allocate and load the task via container
    hipc::FullPtr<Task> task_ptr =
        container->LocalAllocLoadTask(method_id, load_ar);

    if (task_ptr.IsNull()) {
      fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
      alloc->Reset();
      return;
    }

    // Step 3: Execute the task (coroutine: resume + destroy)
    {
      TaskResume coro = container->Run(method_id, task_ptr, rctx_);
      coro.resume();
    }

    // Step 4: Serialize output into FutureShm ring buffer
    hshm::lbm::LbmContext out_ctx;
    out_ctx.copy_space = fshm->copy_space;
    out_ctx.shm_info_ = &fshm->output_;

    auto *heap = CHI_GPU_HEAP;
    LocalSaveTaskArchive save_ar(LocalMsgType::kSerializeOut, heap);
    container->LocalSaveTask(method_id, save_ar, task_ptr);
    hshm::lbm::ShmTransport::Send(save_ar, out_ctx);

    // Step 5: Destroy task
    task_ptr.ptr_->~Task();

    // Step 6: Signal completion
    fshm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);

    // Step 7: Reset arena
    alloc->Reset();
#endif  // !HSHM_IS_HOST
  }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
