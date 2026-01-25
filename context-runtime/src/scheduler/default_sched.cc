// Copyright 2024 IOWarp contributors
#include "chimaera/scheduler/default_sched.h"

#include <functional>

#include "chimaera/ipc_manager.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void DefaultScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  // No special worker division in default scheduler
  // Workers are already created and mapped in WorkOrchestrator
  (void)work_orch;
}

u32 DefaultScheduler::ClientMapTask(IpcManager *ipc_manager,
                                     const Future<Task> &task) {
  // Get number of scheduling queues
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  HLOG(kDebug, "ClientMapTask: num_sched_queues={}", num_lanes);
  if (num_lanes == 0) {
    return 0;
  }

  // Always use PID+TID hash-based mapping
  u32 lane = MapByPidTid(num_lanes);
  HLOG(kDebug, "ClientMapTask: PID+TID hash mapped to lane {}", lane);
  return lane;
}

u32 DefaultScheduler::RuntimeMapTask(const Future<Task> &task) {
  // Return current worker - no migration in default scheduler
  // The task will execute on whichever worker picked it up
  auto *worker = CHI_CUR_WORKER;
  if (worker) {
    return worker->GetId();
  }
  return 0;
}

void DefaultScheduler::RebalanceWorker(Worker *worker) {
  // No rebalancing in default scheduler
  (void)worker;
}

void DefaultScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }

  // Maximum polling interval in microseconds (100ms)
  const double kMaxPollingIntervalUs = 100000.0;

  if (run_ctx->did_work_) {
    // Task did work - use the true (responsive) period
    run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
  } else {
    // Task didn't do work - increase polling interval (exponential backoff)
    double current_interval = run_ctx->yield_time_us_;

    // If uninitialized, start backoff from the true period
    if (current_interval <= 0.0) {
      current_interval = run_ctx->true_period_ns_ / 1000.0;
    }

    // Exponential backoff: double the interval
    double new_interval = current_interval * 2.0;

    // Cap at maximum polling interval
    if (new_interval > kMaxPollingIntervalUs) {
      new_interval = kMaxPollingIntervalUs;
    }

    run_ctx->yield_time_us_ = new_interval;
  }
}

u32 DefaultScheduler::MapByPidTid(u32 num_lanes) {
  // Use HSHM_SYSTEM_INFO to get both PID and TID for lane hashing
  auto *sys_info = HSHM_SYSTEM_INFO;
  pid_t pid = sys_info->pid_;
  auto tid = HSHM_THREAD_MODEL->GetTid();

  // Combine PID and TID for hashing to ensure different processes/threads use
  // different lanes
  size_t combined_hash =
      std::hash<pid_t>{}(pid) ^ (std::hash<void *>{}(&tid) << 1);
  return static_cast<u32>(combined_hash % num_lanes);
}

}  // namespace chi
