// Copyright 2024 IOWarp contributors
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_

#include "chimaera/scheduler/scheduler.h"

namespace chi {

/**
 * Default scheduler implementation.
 * Uses PID+TID hash-based lane mapping and provides no rebalancing.
 */
class DefaultScheduler : public Scheduler {
 public:
  /**
   * Constructor
   */
  DefaultScheduler() = default;

  /**
   * Destructor
   */
  ~DefaultScheduler() override = default;

  /**
   * No special worker division in default scheduler.
   */
  void DivideWorkers(WorkOrchestrator *work_orch) override;

  /**
   * Map task to lane using PID+TID hash.
   */
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;

  /**
   * Return current worker (no migration).
   */
  u32 RuntimeMapTask(const Future<Task> &task) override;

  /**
   * No rebalancing in default scheduler.
   */
  void RebalanceWorker(Worker *worker) override;

 private:
  /**
   * Map task to lane by PID+TID hash
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  u32 MapByPidTid(u32 num_lanes);
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
