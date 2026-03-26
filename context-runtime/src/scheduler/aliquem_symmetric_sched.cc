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

// Copyright 2024 IOWarp contributors
#include "chimaera/scheduler/aliquem_symmetric_sched.h"

#include "chimaera/config_manager.h"
#include "chimaera/container.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/task.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void AliquemSymmetricSched::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    return;
  }

  u32 total_workers = work_orch->GetTotalWorkerCount();
  scheduler_worker_ = nullptr;
  io_workers_.clear();
  net_worker_ = nullptr;
  gpu_worker_ = nullptr;

  scheduler_worker_ = work_orch->GetWorker(0);
  net_worker_ = work_orch->GetWorker(total_workers - 1);

  if (total_workers > 2) {
    gpu_worker_ = work_orch->GetWorker(total_workers - 2);
  }

  if (total_workers > 2) {
    for (u32 i = 1; i < total_workers - 1; ++i) {
      Worker *worker = work_orch->GetWorker(i);
      if (worker) {
        io_workers_.push_back(worker);
      }
    }
  }

  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(1);
    if (net_worker_) {
      ipc->SetNetLane(net_worker_->GetLane());
    }
  }
}

u32 AliquemSymmetricSched::ClientMapTask(IpcManager *ipc_manager,
                                         const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  return num_lanes > 0 ? 0 : 0;
}

u32 AliquemSymmetricSched::RuntimeMapTask(Worker *worker,
                                          const Future<Task> &task,
                                          Container *container) {
  // === SYMMETRIC ARCHITECTURE: Local-First Routing ===
  // Skip null tasks
  if (task.IsNull()) {
    return scheduler_worker_ ? scheduler_worker_->GetId() : 0;
  }

  Task *task_ptr = task.get();
  if (task_ptr == nullptr) {
    return scheduler_worker_ ? scheduler_worker_->GetId() : 0;
  }

  // --- Phase 1: Detect Submitting Thread ---
  // Get the current worker from thread-local storage (who is submitting?)
  // This preserves L1 cache by routing task to the submitting thread's worker
  Worker *cur_worker = CHI_CUR_WORKER;

  // --- Phase 2: Local-First Routing ---
  // If submitting thread is an I/O worker, route task to that worker
  // (locality principle: same core, same L1/L2 cache, no cross-core overhead)
  if (cur_worker != nullptr) {
    u32 cur_worker_id = cur_worker->GetId();

    // Check if current worker is an I/O worker
    for (size_t i = 0; i < io_workers_.size(); ++i) {
      if (io_workers_[i] != nullptr && io_workers_[i]->GetId() == cur_worker_id) {
        // Current thread is an I/O worker - route to itself (local-first)
        return cur_worker_id;
      }
    }
  }

  // --- Phase 3: Fallback Hash Routing ---
  // Submitting thread is not an I/O worker (could be scheduler, network, GPU)
  // Use hash-based routing to distribute evenly across I/O workers
  if (io_workers_.empty()) {
    // No I/O workers available, fall back to scheduler
    return scheduler_worker_ ? scheduler_worker_->GetId() : 0;
  }

  // Hash task group ID to select a worker
  // If task has no group, use task_ptr as hash input for determinism
  uint64_t hash_input = task_ptr->task_group_.IsNull()
                            ? reinterpret_cast<uint64_t>(task_ptr)
                            : static_cast<uint64_t>(task_ptr->task_group_.id_);

  // FNV-1a 64-bit hash (simple, fast, good distribution)
  uint64_t hash = 14695981039346656037ULL;  // FNV basis
  hash ^= hash_input;
  hash *= 1099511628211ULL;  // FNV prime

  // Select I/O worker using modulo
  size_t selected_idx = hash % io_workers_.size();
  if (io_workers_[selected_idx] != nullptr) {
    return io_workers_[selected_idx]->GetId();
  }

  // Absolute fallback
  return io_workers_[0] != nullptr ? io_workers_[0]->GetId()
                                   : (scheduler_worker_ ? scheduler_worker_->GetId()
                                                        : 0);
}

void AliquemSymmetricSched::RebalanceWorker(Worker *worker) {
  // === SYMMETRIC WORK-STEALING: BLIND STEALING ===
  // This function is called when 'worker' is idle with no pending tasks.
  // Use round-robin victim selection to steal exactly one task per invocation.
  //
  // KEY ARCHITECTURAL DECISION: "Blind Stealing"
  // - Never probe/peek (Pop + Push destroys FIFO ordering in ring buffer)
  // - Round-robin victim selection (all workers equally fair)
  // - Steal exactly one task per call (reactive, low overhead)
  // - FIFO integrity preserved (Pop = head, Push = tail separation)

  if (worker == nullptr || io_workers_.empty()) {
    return;  // No worker or no I/O workers to steal from
  }

  TaskLane *idle_lane = worker->GetLane();
  if (idle_lane == nullptr) {
    return;  // Worker has no assigned lane
  }

  // --- Phase 1: Round-Robin Victim Selection ---
  // Use a static atomic counter to track next victim across all invocations
  // This ensures fair load distribution over time
  static std::atomic<u32> next_victim_idx{0};

  u32 num_io_workers = io_workers_.size();
  u32 start_idx = next_victim_idx.load(std::memory_order_relaxed);

  // --- Phase 2: Iterate and Steal ---
  // Try up to num_io_workers candidates (full scan, then give up)
  for (u32 attempt = 0; attempt < num_io_workers; ++attempt) {
    u32 candidate_idx =
        (start_idx + attempt) % num_io_workers;  // Circular iteration
    Worker *candidate = io_workers_[candidate_idx];

    if (candidate == nullptr) {
      continue;  // Skip null workers
    }

    if (candidate->GetId() == worker->GetId()) {
      continue;  // Never steal from ourselves
    }

    TaskLane *victim_lane = candidate->GetLane();
    if (victim_lane == nullptr) {
      continue;  // Skip workers with no lane
    }

    // --- Phase 3: Attempt Single Pop ---
    // Try to steal exactly one task from this victim
    // CRITICAL: Do NOT push it back. Pop = head extraction, Push = tail append.
    //           Popping and pushing would destroy FIFO order.
    Future<Task> stolen_task;
    if (victim_lane->Pop(stolen_task)) {
      // SUCCESS: Stolen one task
      // Push to idle worker's lane (not back to victim!)
      idle_lane->Push(stolen_task);

      // Update round-robin pointer for next invocation
      next_victim_idx.store((candidate_idx + 1) % num_io_workers,
                            std::memory_order_relaxed);

      // Awaken the idle worker with the new task
      IpcManager *ipc = CHI_IPC;
      if (ipc) {
        ipc->AwakenWorker(idle_lane);
      }

      return;  // Done - one task stolen and scheduled
    }
    // If Pop() failed, this victim is empty. Try next.
  }

  // All workers scanned, none had tasks. Idle worker remains idle.
  // next_victim_idx will continue from where we left off on next call.
}

void AliquemSymmetricSched::AdjustPolling(RunContext *run_ctx) {
  // Stub implementation - no polling adjustment needed for symmetric model
  // Work-stealing is reactive (triggered on worker idle), not proactive
}

}  // namespace chi
