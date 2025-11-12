#include "chimaera/corwlock.h"
#include "chimaera/task.h"
#include "chimaera/worker.h"
#include "chimaera/singletons.h"

namespace chi {

void CoRwLock::ReadLock() {
  // Get current task from the current worker
  auto* worker = CHI_CUR_WORKER;
  if (!worker) {
    return; // No worker context
  }

  FullPtr<Task> task = worker->GetCurrentTask();
  if (task.IsNull()) {
    return;
  }

  // Wait until no writer is active
  while (writer_active_.load()) {
    task->Yield();
  }

  // Increment reader count
  reader_count_.fetch_add(1);
}

void CoRwLock::ReadUnlock() {
  // Decrement reader count
  reader_count_.fetch_sub(1);
}

void CoRwLock::WriteLock() {
  // Get current task from the current worker
  auto* worker = CHI_CUR_WORKER;
  if (!worker) {
    return; // No worker context
  }

  FullPtr<Task> task = worker->GetCurrentTask();
  if (task.IsNull()) {
    return;
  }

  // Try to acquire write lock
  bool expected = false;
  while (!writer_active_.compare_exchange_weak(expected, true)) {
    // Writer is active - yield and try again
    task->Yield();
    expected = false;
  }

  // Wait until all readers have finished
  while (reader_count_.load() > 0) {
    task->Yield();
  }

  // Write lock acquired successfully
}

void CoRwLock::WriteUnlock() {
  // Release the write lock
  writer_active_.store(false);
}

bool CoRwLock::TryReadLock() {
  // Check if writer is active
  if (writer_active_.load()) {
    return false;
  }

  // Increment reader count
  reader_count_.fetch_add(1);

  // Double-check writer didn't start between check and increment
  if (writer_active_.load()) {
    reader_count_.fetch_sub(1);
    return false;
  }

  return true;
}

bool CoRwLock::TryWriteLock() {
  // Try to acquire write lock
  bool expected = false;
  if (!writer_active_.compare_exchange_strong(expected, true)) {
    return false;
  }

  // Check if any readers are active
  if (reader_count_.load() > 0) {
    writer_active_.store(false);
    return false;
  }

  return true;
}

}  // namespace chi
