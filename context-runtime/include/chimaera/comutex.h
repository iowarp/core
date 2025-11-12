#ifndef CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_
#define CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_

#include <atomic>
#include <hermes_shm/hermes_shm.h>

#include "chimaera/types.h"

namespace chi {

// Forward declarations
class Task;
template<typename T> using FullPtr = hipc::FullPtr<T>;

/**
 * Simplified coroutine mutex that uses Yield for blocking.
 * Tasks that cannot acquire the lock will call Yield and be placed
 * in the blocked queue, where they will be retried later.
 */
class CoMutex {
public:
  CoMutex() : is_locked_(false) {}

  // Non-copyable (atomics can't be copied)
  CoMutex(const CoMutex&) = delete;
  CoMutex& operator=(const CoMutex&) = delete;

  // Movable (for use in vectors)
  CoMutex(CoMutex&& other) noexcept : is_locked_(other.is_locked_.load()) {}
  CoMutex& operator=(CoMutex&& other) noexcept {
    if (this != &other) {
      is_locked_.store(other.is_locked_.load());
    }
    return *this;
  }

  /**
   * Acquire the mutex for the current task (retrieved from CHI_CUR_WORKER)
   * If the mutex is free, acquires it
   * If the mutex is locked, calls Yield to place task in blocked queue
   */
  void Lock();

  /**
   * Release the mutex
   * (uses current task from CHI_CUR_WORKER)
   */
  void Unlock();

  /**
   * Try to acquire the mutex without blocking
   * (uses current task from CHI_CUR_WORKER)
   * @return true if acquired successfully, false otherwise
   */
  bool TryLock();

private:
  std::atomic<bool> is_locked_;  // Whether the mutex is currently locked
};

/**
 * RAII-style scoped mutex lock for CoMutex
 */
class ScopedCoMutex {
public:
  /**
   * Constructor that acquires the mutex
   * @param mutex CoMutex to acquire
   */
  explicit ScopedCoMutex(CoMutex& mutex) 
      : mutex_(mutex) {
    mutex_.Lock();
  }

  /**
   * Destructor that releases the mutex
   */
  ~ScopedCoMutex() {
    mutex_.Unlock();
  }

  // Non-copyable
  ScopedCoMutex(const ScopedCoMutex&) = delete;
  ScopedCoMutex& operator=(const ScopedCoMutex&) = delete;

  // Non-movable
  ScopedCoMutex(ScopedCoMutex&&) = delete;
  ScopedCoMutex& operator=(ScopedCoMutex&&) = delete;

private:
  CoMutex& mutex_;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_