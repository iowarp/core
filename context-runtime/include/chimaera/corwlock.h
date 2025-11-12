#ifndef CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_

#include <atomic>
#include <hermes_shm/hermes_shm.h>

#include "chimaera/types.h"

namespace chi {

/**
 * Simplified coroutine reader-writer lock that uses Yield for blocking.
 * Tasks that cannot acquire the lock will call Yield and be placed
 * in the blocked queue, where they will be retried later.
 *
 * Supports multiple concurrent readers or a single writer.
 */
class CoRwLock {
public:
  CoRwLock() : reader_count_(0), writer_active_(false) {}

  /**
   * Acquire a read lock for the current task (retrieved from CHI_CUR_WORKER)
   * If no writer is active, acquires read lock
   * If writer is active, calls Yield to place task in blocked queue
   */
  void ReadLock();

  /**
   * Release a read lock (uses current task from CHI_CUR_WORKER)
   */
  void ReadUnlock();

  /**
   * Acquire a write lock for the current task (retrieved from CHI_CUR_WORKER)
   * If no readers or writers are active, acquires write lock
   * If readers or writers are active, calls Yield to place task in blocked queue
   */
  void WriteLock();

  /**
   * Release a write lock
   * (uses current task from CHI_CUR_WORKER)
   */
  void WriteUnlock();

  /**
   * Try to acquire a read lock without blocking
   * (uses current task from CHI_CUR_WORKER)
   * @return true if acquired successfully, false otherwise
   */
  bool TryReadLock();

  /**
   * Try to acquire a write lock without blocking
   * (uses current task from CHI_CUR_WORKER)
   * @return true if acquired successfully, false otherwise
   */
  bool TryWriteLock();

private:
  std::atomic<int> reader_count_;      // Number of active readers
  std::atomic<bool> writer_active_;    // Whether a writer is active
};

/**
 * RAII-style scoped read lock for CoRwLock
 */
class ScopedCoRwReadLock {
public:
  /**
   * Constructor that acquires the read lock
   * @param rwlock CoRwLock to acquire read lock on
   */
  explicit ScopedCoRwReadLock(CoRwLock& rwlock)
      : rwlock_(rwlock) {
    rwlock_.ReadLock();
  }

  /**
   * Destructor that releases the read lock
   */
  ~ScopedCoRwReadLock() {
    rwlock_.ReadUnlock();
  }

  // Non-copyable
  ScopedCoRwReadLock(const ScopedCoRwReadLock&) = delete;
  ScopedCoRwReadLock& operator=(const ScopedCoRwReadLock&) = delete;

  // Non-movable
  ScopedCoRwReadLock(ScopedCoRwReadLock&&) = delete;
  ScopedCoRwReadLock& operator=(ScopedCoRwReadLock&&) = delete;

private:
  CoRwLock& rwlock_;
};

/**
 * RAII-style scoped write lock for CoRwLock
 */
class ScopedCoRwWriteLock {
public:
  /**
   * Constructor that acquires the write lock
   * @param rwlock CoRwLock to acquire write lock on
   */
  explicit ScopedCoRwWriteLock(CoRwLock& rwlock)
      : rwlock_(rwlock) {
    rwlock_.WriteLock();
  }

  /**
   * Destructor that releases the write lock
   */
  ~ScopedCoRwWriteLock() {
    rwlock_.WriteUnlock();
  }

  // Non-copyable
  ScopedCoRwWriteLock(const ScopedCoRwWriteLock&) = delete;
  ScopedCoRwWriteLock& operator=(const ScopedCoRwWriteLock&) = delete;

  // Non-movable
  ScopedCoRwWriteLock(ScopedCoRwWriteLock&&) = delete;
  ScopedCoRwWriteLock& operator=(ScopedCoRwWriteLock&&) = delete;

private:
  CoRwLock& rwlock_;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_