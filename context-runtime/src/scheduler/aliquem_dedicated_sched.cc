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
#include "chimaera/scheduler/aliquem_dedicated_sched.h"

#include "chimaera/config_manager.h"
#include "chimaera/container.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/task.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void AliquemDedicatedSched::DivideWorkers(WorkOrchestrator *work_orch) {
  // Initialize RCFS deficit trackers to zero
  for (int i = 0; i < 8; i++) {
    worker_deficits_[i].store(0, std::memory_order_relaxed);
  }

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

u32 AliquemDedicatedSched::ClientMapTask(IpcManager *ipc_manager,
                                         const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  return num_lanes > 0 ? 0 : 0;
}

u32 AliquemDedicatedSched::RuntimeMapTask(Worker *worker,
                                          const Future<Task> &task,
                                          Container *container) {
  // Skip null tasks
  if (task.IsNull()) {
    return scheduler_worker_ ? scheduler_worker_->GetId() : 0;
  }

  Task *task_ptr = task.get();
  if (task_ptr == nullptr) {
    return scheduler_worker_ ? scheduler_worker_->GetId() : 0;
  }

  // --- Phase 1: RCFS Edge Heuristic ---
  // Convert I/O bytes and compute microseconds into unified "Deficit Cost"
  // Retrieve TaskStat from container for this method
  uint64_t io_cost = 0;
  uint64_t compute_cost = 0;

  if (container != nullptr) {
    TaskStat stat = container->GetTaskStats(task_ptr->method_);
    // Calibration: On Pi 4, transferring 100 bytes ≈ 1 microsecond of compute
    io_cost = stat.io_size_ / 100;
    compute_cost = stat.compute_;
  }

  uint64_t total_task_cost = io_cost + compute_cost;

  // Prevent zero-cost starvation - assign minimum cost of 1
  if (total_task_cost == 0) {
    total_task_cost = 1;
  }

  // --- Phase 2: Aliquem O(1) Routing ---
  // Find the worker with the minimum deficit (lowest historical burden)
  Worker *selected = nullptr;
  uint64_t lowest_deficit = UINT64_MAX;
  u32 selected_idx = 0;

  // Scan strictly the I/O compute workers
  for (size_t i = 0; i < io_workers_.size(); ++i) {
    Worker *w = io_workers_[i];
    if (w == nullptr) continue;

    u32 worker_id = w->GetId();
    if (worker_id >= 8) continue;  // Bounds check on deficit array

    uint64_t current_deficit =
        worker_deficits_[worker_id].load(std::memory_order_relaxed);

    if (current_deficit < lowest_deficit) {
      lowest_deficit = current_deficit;
      selected = w;
      selected_idx = worker_id;
    }
  }

  // Absolute fallback: use scheduler worker if no I/O workers
  if (selected == nullptr) {
    if (!io_workers_.empty() && io_workers_[0] != nullptr) {
      selected_idx = io_workers_[0]->GetId();
    } else if (scheduler_worker_ != nullptr) {
      return scheduler_worker_->GetId();
    } else {
      return 0;
    }
  }

  // --- Phase 3: Update Worker's Watermark ---
  // Atomically accrue the cost to this worker's deficit
  if (selected_idx < 8) {
    worker_deficits_[selected_idx].fetch_add(total_task_cost,
                                              std::memory_order_relaxed);
  }

  return selected_idx;
}

void AliquemDedicatedSched::RebalanceWorker(Worker *worker) {
  // Stub implementation
}

void AliquemDedicatedSched::AdjustPolling(RunContext *run_ctx) {
  // Stub implementation
}

}  // namespace chi
