/**
 * Task implementation
 *
 * With C++20 stackless coroutines, task suspension and resumption is handled
 * by the coroutine machinery via co_await. Use co_await chi::yield() for
 * yielding in coroutine context, or HSHM_THREAD_MODEL->Yield() for non-coroutine
 * contexts.
 */

#include "chimaera/task.h"

#include "chimaera/future.h"
#include "chimaera/singletons.h"
#include "chimaera/worker.h"

namespace chi {

void Task::Wait(std::atomic<u32> &is_complete, double yield_time_us) {
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager != nullptr && chimaera_manager->IsRuntime()) {
    // Runtime implementation: Yield until is_complete is set
    // In coroutine context, this is handled by co_await future
    // This path is for non-coroutine runtime contexts

    // Get current run context from worker
    Worker *worker = CHI_CUR_WORKER;
    RunContext *run_ctx = worker != nullptr ? worker->GetCurrentRunContext() : nullptr;

    if (worker == nullptr || run_ctx == nullptr) {
      // No worker or run context available, fall back to client implementation
      // Busy-wait on is_complete flag
      while (is_complete.load() == 0) {
        HSHM_THREAD_MODEL->Yield();
      }
      return;
    }

    // For coroutine-based tasks, this path should not be hit
    // Coroutines use co_await future which suspends via await_suspend
    // This is a fallback for any non-coroutine runtime paths

    // Store yield duration in RunContext
    run_ctx->yield_time_us_ = yield_time_us;

    // Busy-wait with yield - task will be resumed when subtask completes
    while (is_complete.load() == 0) {
      run_ctx->is_yielded_ = true;
      worker->AddToBlockedQueue(run_ctx, true);  // wait_for_task = true
      // In coroutine model, the worker will resume the coroutine
      // For non-coroutine contexts, we just yield the thread
      HSHM_THREAD_MODEL->Yield();
      worker->SetTaskDidWork(false);
    }
  } else {
    // Client implementation: Busy-wait on is_complete flag
    while (is_complete.load() == 0) {
      HSHM_THREAD_MODEL->Yield();
    }
  }
}

size_t Task::EstCpuTime() const {
  // Calculate: io_size / 4GBps + compute + 5
  // 4 GBps = 4 * 1024 * 1024 * 1024 bytes/second = 4294967296 bytes/second
  // Convert to microseconds: (io_size / 4294967296) * 1000000
  size_t io_time_us = (stat_.io_size_ * 1000000) / 4294967296ULL;
  return io_time_us + stat_.compute_ + 5;
}

}  // namespace chi
