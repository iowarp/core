// Copyright 2024 IOWarp contributors
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_SCHEDULER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_SCHEDULER_H_

#include "chimaera/types.h"
#include "chimaera/task.h"

namespace chi {

// Forward declarations
class IpcManager;
class WorkOrchestrator;
class Worker;

/**
 * Base class for task scheduling strategies.
 * Implementations decide how to map tasks to workers and balance load.
 */
class Scheduler {
 public:
  /**
   * Virtual destructor
   */
  virtual ~Scheduler() = default;

  /**
   * Decides how to pin workers to cores and create worker groups.
   * Called after all workers have been spawned in the WorkOrchestrator.
   *
   * @param work_orch Pointer to the work orchestrator
   */
  virtual void DivideWorkers(WorkOrchestrator *work_orch) = 0;

  /**
   * Determines which worker to initially map a task to from clients.
   * First few workers are always the scheduling workers.
   * Analogous to the old MapTaskToLane function.
   *
   * @param ipc_manager Pointer to the IPC manager
   * @param task The task to be scheduled
   * @return Worker lane ID to assign the task to
   */
  virtual u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) = 0;

  /**
   * Determines which worker to initially map a task to from runtime.
   * Called in RouteTask.
   *
   * @param task The task to be scheduled
   * @return Worker ID to assign the task to
   */
  virtual u32 RuntimeMapTask(const Future<Task> &task) = 0;

  /**
   * Either steal or delegate tasks on a worker to balance load.
   * Should be called after every ProcessNewTasks loop before SuspendMe.
   *
   * @param worker Pointer to the worker to rebalance
   */
  virtual void RebalanceWorker(Worker *worker) = 0;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_SCHEDULER_H_
