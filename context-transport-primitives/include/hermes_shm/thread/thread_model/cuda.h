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

#ifndef HSHM_THREAD_CUDA_H_
#define HSHM_THREAD_CUDA_H_

#if HSHM_ENABLE_CUDA

#include <errno.h>

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/util/errors.h"
#include "thread_model.h"

namespace hshm::thread {

class Cuda : public ThreadModel {
 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Cuda() : ThreadModel(ThreadType::kCuda) {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~Cuda() = default;

  /** Yield the current thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) {}

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() {
    hshm::ipc::threadfence();
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
    return false;
  }

  /** Get thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
    return nullptr;
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
    return false;
  }

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  ThreadId GetTid() { return ThreadId::GetNull(); }
};

}  // namespace hshm::thread

#endif  // HSHM_ENABLE_CUDA

#endif  // HSHM_THREAD_CUDA_H_
