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
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_

#include <atomic>
#include <vector>

#include "chimaera/scheduler/scheduler.h"

namespace chi {

/**
 * Default scheduler implementation.
 * Uses PID+TID hash-based lane mapping and provides no rebalancing.
 * All workers process tasks; scheduler tracks worker groups for routing decisions.
 */
class DefaultScheduler : public Scheduler {
 public:
  /**
   * Constructor
   */
  DefaultScheduler() : net_worker_(nullptr), gpu_worker_(nullptr) {}

  /**
   * Destructor
   */
  ~DefaultScheduler() override = default;

  /**
   * Initialize scheduler with all available workers.
   * Tracks scheduler workers and network worker for routing decisions.
   * @param work_orch Pointer to the work orchestrator
   */
  void DivideWorkers(WorkOrchestrator *work_orch) override;

  /**
   * Map task to lane using PID+TID hash.
   */
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;

  /**
   * Return current worker (no migration).
   * @param worker The worker that called this method
   * @param task The task to be scheduled
   * @return Worker ID to assign the task to
   */
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) override;

  /**
   * No rebalancing in default scheduler.
   */
  void RebalanceWorker(Worker *worker) override;

  /**
   * Adjust polling interval for periodic tasks based on work done.
   * Implements exponential backoff when tasks aren't doing work.
   */
  void AdjustPolling(RunContext *run_ctx) override;

  /**
   * Get the designated GPU worker.
   */
  Worker *GetGpuWorker() const override { return gpu_worker_; }

 private:
  /**
   * Map task to lane by PID+TID hash
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  u32 MapByPidTid(u32 num_lanes);

  // Internal worker tracking for routing decisions
  std::vector<Worker *> scheduler_workers_;  ///< Task processing workers
  Worker *net_worker_;                        ///< Network worker (for routing periodic Send/Recv)
  Worker *gpu_worker_;                        ///< GPU queue polling worker
  std::atomic<u32> next_sched_idx_{0};        ///< Round-robin index for GPU task forwarding
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
