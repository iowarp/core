/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HSHM_THREAD_STD_THREAD_H_
#define HSHM_THREAD_STD_THREAD_H_

// StdThread is always available as it uses standard C++ features
#include <thread>

#include "hermes_shm/introspect/system_info.h"
#include "thread_model.h"

// Windows <windows.h> defines Yield() as a macro, which breaks our method.
#ifdef _WIN32
#ifdef Yield
#undef Yield
#endif
#endif

namespace hshm::thread {

/** Represents the generic operations of a thread */
class StdThread : public ThreadModel {
 public:
  /** Initializer */
  HSHM_INLINE_CROSS_FUN
  StdThread() : ThreadModel(ThreadType::kStdThread) {}

  /** Initialize std thread */
  HSHM_CROSS_FUN
  void Init() {}

  /** Sleep thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() { std::this_thread::yield(); }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#if HSHM_IS_HOST
    return SystemInfo::CreateTls(key, (void *)data);
#else
    return false;
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
#if HSHM_IS_HOST
    return SystemInfo::SetTls(key, (void *)data);
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#if HSHM_IS_HOST
    return static_cast<TLS *>(SystemInfo::GetTls(key));
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  ThreadId GetTid() { return ThreadId(SystemInfo::GetTid()); }

  /** Get the thread model type */
  HSHM_INLINE_CROSS_FUN
  ThreadType GetType() { return type_; }

  /** Create a thread group */
  HSHM_CROSS_FUN
  ThreadGroup CreateThreadGroup(const ThreadGroupContext &ctx) {
    return ThreadGroup{};
  }

  /** Spawn a thread */
  template <typename FUNC, typename... Args>
  HSHM_CROSS_FUN Thread Spawn(ThreadGroup &group, FUNC &&func, Args &&...args) {
    Thread thread;
    thread.group_ = group;
    thread.std_thread_ =
        std::thread(std::forward<FUNC>(func), std::forward<Args>(args)...);
    return thread;
  }

  /** Join a thread */
  HSHM_CROSS_FUN
  void Join(Thread &thread) { thread.std_thread_.join(); }

  /** Set CPU affinity for thread */
  HSHM_CROSS_FUN
  void SetAffinity(Thread &thread, int cpu_id) {}
};

}  // namespace hshm::thread

#endif  // HSHM_THREAD_STD_THREAD_H_
