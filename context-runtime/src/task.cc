/**
 * Task implementation
 */

#include "chimaera/task.h"

#include <algorithm>

#include "chimaera/container.h"
#include "chimaera/future.h"
#include "chimaera/singletons.h"
#include "chimaera/worker.h"

// Namespace alias for boost::context::detail
namespace bctx = boost::context::detail;

namespace chi {

void Task::Wait(std::atomic<u32> &is_complete, double yield_time_us) {
  HLOG(kInfo, "[TRACE] Task::Wait START - task_id={}, pool_id={}, method={}, is_complete={}",
       task_id_, pool_id_, method_, is_complete.load());
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    // Runtime implementation: Yield until is_complete is set

    // Get current run context from worker
    Worker *worker = CHI_CUR_WORKER;
    RunContext *run_ctx = worker ? worker->GetCurrentRunContext() : nullptr;

    if (!worker || !run_ctx) {
      // No worker or run context available, fall back to client implementation
      // Busy-wait on is_complete flag
      while (is_complete.load() == 0) {
        YieldBase();
      }
      return;
    }

    // Check if task is already yielded - this should never happen
    if (run_ctx->is_yielded_) {
      HLOG(kFatal,
            "Worker {}: Task is already yielded when calling Wait()! "
            "Task ptr: {:#x}, Pool: {}, Method: {}, TaskId: {}.{}.{}.{}.{}",
            worker->GetId(), reinterpret_cast<uintptr_t>(this), pool_id_,
            method_, task_id_.pid_, task_id_.tid_, task_id_.major_,
            task_id_.replica_id_, task_id_.unique_);
      std::abort();
    }

    // Store yield duration in RunContext (use provided value directly)
    // yield_time_us is passed by the caller - no estimation
    run_ctx->yield_time_us_ = yield_time_us;

    // Yield execution back to worker in do-while loop until is_complete is set
    // Use wait_for_task=true to indicate task is waiting for subtask completion
    // Task will be woken by event queue when subtask completes
    do {
      worker->AddToBlockedQueue(run_ctx, true);  // wait_for_task = true
      YieldBase();
      // After yielding, assume blocked work (will be corrected by worker if
      // task completes)
      worker->SetTaskDidWork(false);
    } while (is_complete.load() == 0);
  } else {
    // Client implementation: Busy-wait on is_complete flag
    HLOG(kInfo, "[TRACE] Task::Wait - client mode, busy-waiting on is_complete");
    int wait_count = 0;
    while (is_complete.load() == 0) {
      wait_count++;
      if (wait_count % 1000000 == 0) {
        HLOG(kInfo, "[TRACE] Task::Wait - still waiting, count={}, task_id={}", wait_count, task_id_);
      }
      YieldBase();
    }
    HLOG(kInfo, "[TRACE] Task::Wait - client mode, wait complete after {} iterations", wait_count);
  }
  HLOG(kInfo, "[TRACE] Task::Wait END - task_id={}, is_complete={}", task_id_, is_complete.load());
}

void Task::YieldBase() {
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    // Get current run context from worker
    Worker *worker = CHI_CUR_WORKER;
    RunContext *run_ctx = worker ? worker->GetCurrentRunContext() : nullptr;

    if (!run_ctx) {
      // No run context available, fall back to client implementation
      HSHM_THREAD_MODEL->Yield();
      return;
    }

    // Mark this task as yielded
    run_ctx->is_yielded_ = true;

    // Jump back to worker using boost::fiber

    // Jump back to worker - the task has been added to blocked queue
    // Store the result (task's yield point) in resume_context for later
    // resumption Use temporary variables to store the yield context before
    // jumping
    bctx::fcontext_t yield_fctx = run_ctx->yield_context.fctx;
    void *yield_data = run_ctx->yield_context.data;

    // Jump back to worker and capture the result
    bctx::transfer_t yield_result = bctx::jump_fcontext(yield_fctx, yield_data);

    // CRITICAL: Update yield_context with the new worker context from the
    // resume operation This ensures that subsequent yields or completion
    // returns to the correct worker location
    run_ctx->yield_context = yield_result;

    // Store where we can resume from for the next yield cycle
    run_ctx->resume_context = yield_result;
  } else {
    // Outside runtime mode, just yield
    HSHM_THREAD_MODEL->Yield();
  }
}

void Task::Yield(double yield_time_us) {
  // Yield execution without waiting for any specific completion condition
  // This is used for cooperative yielding during blocking operations (I/O,
  // locks, etc.)
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    // Get current run context from worker
    Worker *worker = CHI_CUR_WORKER;
    RunContext *run_ctx = worker ? worker->GetCurrentRunContext() : nullptr;

    if (worker && run_ctx) {
      // Store yield duration in RunContext
      run_ctx->yield_time_us_ = yield_time_us;

      // Add to blocked queue with wait_for_task=false (not waiting for specific
      // task)
      worker->AddToBlockedQueue(run_ctx, false);
      YieldBase();

      // After yielding, assume blocked work
      worker->SetTaskDidWork(false);
    } else {
      // No worker context, just yield
      YieldBase();
    }
  } else {
    // Client mode: just yield
    YieldBase();
  }
}

// Task::Aggregate is now a template method in task.h

size_t Task::EstCpuTime() const {
  // Calculate: io_size / 4GBps + compute + 5
  // 4 GBps = 4 * 1024 * 1024 * 1024 bytes/second = 4294967296 bytes/second
  // Convert to microseconds: (io_size / 4294967296) * 1000000
  size_t io_time_us = (stat_.io_size_ * 1000000) / 4294967296ULL;
  return io_time_us + stat_.compute_ + 5;
}

}  // namespace chi