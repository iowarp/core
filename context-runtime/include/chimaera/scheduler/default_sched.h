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
  DefaultScheduler() : net_worker_(nullptr) {}

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
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
