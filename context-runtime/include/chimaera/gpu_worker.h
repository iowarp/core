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
 * Runs on block 0, thread 0 of the megakernel. Polls both the
 * CPU→GPU queue (to_gpu_queue) and the GPU→GPU queue (gpu_to_gpu_queue)
 * for incoming tasks, deserializes inputs, dispatches to GPU containers,
 * serializes outputs, and signals completion.
 *
 * No STL, no coroutines, no TLS — all HSHM_GPU_FUN.
 */
class Worker {
 public:
  u32 worker_id_;                    /**< Worker identity */
  volatile bool is_running_;         /**< Running flag for the poll loop */
  TaskQueue *to_gpu_queue_;          /**< CPU → GPU queue (megakernel polls) */
  TaskQueue *gpu_to_gpu_queue_;      /**< GPU → GPU queue (megakernel polls) */
  PoolManager *pool_mgr_;            /**< GPU-side container lookup table */
  char *queue_backend_base_;         /**< Base of queue backend for ShmPtr resolution */
  GpuRunContext rctx_;               /**< Reused run context per task */

  /**
   * Initialize the worker with queue and pool manager pointers.
   *
   * @param worker_id Logical worker ID
   * @param to_gpu_queue CPU→GPU queue pointer (pinned host memory)
   * @param gpu_to_gpu_queue GPU→GPU queue pointer (pinned host memory)
   * @param pool_mgr GPU-side pool manager for container lookup
   * @param queue_backend_base Base pointer of queue backend for ShmPtr offsets
   */
  HSHM_GPU_FUN void Init(u32 worker_id,
                          TaskQueue *to_gpu_queue,
                          TaskQueue *gpu_to_gpu_queue,
                          PoolManager *pool_mgr,
                          char *queue_backend_base) {
    worker_id_ = worker_id;
    to_gpu_queue_ = to_gpu_queue;
    gpu_to_gpu_queue_ = gpu_to_gpu_queue;
    pool_mgr_ = pool_mgr;
    queue_backend_base_ = queue_backend_base;
    is_running_ = true;
#if HSHM_IS_GPU_COMPILER
    rctx_ = GpuRunContext(blockIdx.x, threadIdx.x);
#else
    rctx_ = GpuRunContext(0, 0);
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
    did_work |= ProcessNewTask(to_gpu_queue_);
    did_work |= ProcessNewTask(gpu_to_gpu_queue_);
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

    // Resolve FutureShm from the queue backend
    FutureShm *fshm = ResolveFutureShm(future);
    if (!fshm) {
      return true;  // Consumed slot but bad pointer
    }

    // Look up the target container
    PoolId pool_id = fshm->pool_id_;
    u32 method_id = fshm->method_id_;
    Container *container = pool_mgr_->GetContainer(pool_id);
    if (!container) {
      // No container registered — mark complete with no output
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      return true;
    }

    // Deserialize input, dispatch, serialize output
    DispatchTask(fshm, container, method_id);
    return true;
  }

  /**
   * Resolve a FutureShm pointer from a Future's ShmPtr.
   *
   * The FutureShm is allocated in the queue backend (pinned host memory).
   * We resolve it by adding the ShmPtr offset to queue_backend_base_.
   *
   * @param future The future whose FutureShm to resolve
   * @return Pointer to the FutureShm, or nullptr on failure
   */
  HSHM_GPU_FUN FutureShm *ResolveFutureShm(Future<Task> &future) {
    hipc::ShmPtr<FutureShm> sptr = future.GetFutureShmPtr();
    if (sptr.IsNull()) return nullptr;
    size_t off = sptr.off_.load();
    return reinterpret_cast<FutureShm *>(queue_backend_base_ + off);
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
    // Step 1: Deserialize input from FutureShm ring buffer
    hshm::lbm::LbmContext in_ctx;
    in_ctx.copy_space = fshm->copy_space;
    in_ctx.shm_info_ = &fshm->input_;

    auto *ipc = CHI_IPC;
    auto *alloc = ipc->gpu_alloc_table_[ipc->GetGpuThreadId()];

    // Set allocator on container for cross-library calls
    container->gpu_alloc_ = alloc;

    LocalLoadTaskArchive load_ar(alloc);
    hshm::lbm::ShmTransport::Recv(load_ar, in_ctx);
    load_ar.SetMsgType(LocalMsgType::kSerializeIn);

    // Step 2: Allocate and load the task via container
    hipc::FullPtr<Task> task_ptr =
        container->LocalAllocLoadTask(method_id, load_ar);
    if (task_ptr.IsNull()) {
      fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
      return;
    }

    // Step 3: Execute the task
    container->Run(method_id, task_ptr, rctx_);

    // Step 4: Serialize output into FutureShm ring buffer
    hshm::lbm::LbmContext out_ctx;
    out_ctx.copy_space = fshm->copy_space;
    out_ctx.shm_info_ = &fshm->output_;

    hshm::priv::vector<char, HSHM_DEFAULT_ALLOC_GPU_T> buf(alloc);
    LocalSaveTaskArchive save_ar(LocalMsgType::kSerializeOut, buf, alloc);
    container->LocalSaveTask(method_id, save_ar, task_ptr);
    hshm::lbm::ShmTransport::Send(save_ar, out_ctx);

    // Step 5: System-scope memory fence + mark complete
    hipc::threadfence_system();
    fshm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);

    // Step 6: Reset this thread's allocator (bulk free scratch)
    alloc->Reset();
  }
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_WORKER_H_
