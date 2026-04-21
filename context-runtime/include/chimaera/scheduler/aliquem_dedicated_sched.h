#pragma once
#include "scheduler.h"
#include <atomic>
#include <cstdint>

namespace chi {

class AliquemDedicatedSched : public Scheduler {
 public:
  AliquemDedicatedSched() = default;
  ~AliquemDedicatedSched() override = default;

  void DivideWorkers(WorkOrchestrator *work_orch) override;
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task,
                     Container *container) override;
  void RebalanceWorker(Worker *worker) override;
  void AdjustPolling(RunContext *run_ctx) override;

 private:
  Worker *scheduler_worker_ = nullptr;
  std::vector<Worker *> io_workers_;
  Worker *net_worker_ = nullptr;
  Worker *gpu_worker_ = nullptr;

  // RCFS Deficit Cost Tracker: lock-free scoreboards for up to 8 workers
  // Maintains O(1) routing by tracking cumulative workload per worker
  std::atomic<uint64_t> worker_deficits_[8];
};

}  // namespace chi