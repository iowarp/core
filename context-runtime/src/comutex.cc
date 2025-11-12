#include "chimaera/comutex.h"
#include "chimaera/task.h"
#include "chimaera/worker.h"
#include "chimaera/singletons.h"

namespace chi {

void CoMutex::Lock() {
  // Get current task from the current worker
  auto* worker = CHI_CUR_WORKER;
  if (!worker) {
    return; // No worker context
  }

  FullPtr<Task> task = worker->GetCurrentTask();
  if (task.IsNull()) {
    return;
  }

  // Try to acquire the lock
  bool expected = false;
  while (!is_locked_.compare_exchange_weak(expected, true)) {
    // Lock is held - yield and try again later
    task->Yield();
    expected = false;
  }

  // Lock acquired successfully
}

void CoMutex::Unlock() {
  // Release the lock
  is_locked_.store(false);
}

bool CoMutex::TryLock() {
  // Try to acquire without blocking
  bool expected = false;
  return is_locked_.compare_exchange_strong(expected, true);
}

}  // namespace chi