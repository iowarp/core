#pragma once
#include "scheduler.h"
#include <atomic>
#include <vector>

namespace chi {

class AliquemSymmetricSched : public Scheduler {
 public:
  AliquemSymmetricSched() = default;
  ~AliquemSymmetricSched() override = default;

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
};

}  // namespace chi