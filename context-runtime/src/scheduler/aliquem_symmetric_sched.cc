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
  return scheduler_worker_ ? scheduler_worker_->GetId() : 0;
}

void AliquemSymmetricSched::RebalanceWorker(Worker *worker) {
  // Stub implementation
}

void AliquemSymmetricSched::AdjustPolling(RunContext *run_ctx) {
  // Stub implementation
}

}  // namespace chi
