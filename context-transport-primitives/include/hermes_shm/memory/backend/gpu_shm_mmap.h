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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

/**
 * GPU+CPU coherent shared memory backend using CUDA Unified Virtual Memory.
 *
 * Uses cudaMallocManaged (UVM) so both CPU and GPU can access the same memory
 * with hardware-managed cache coherence.  This avoids the need for clflush,
 * write-combining tricks, or system-scope atomics that require
 * cudaDevAttrHostNativeAtomicSupported (not available on all discrete GPUs).
 *
 * On GPUs with ConcurrentManagedAccess (SM 7.0+), the hardware migration
 * engine handles coherence transparently so std::atomic on CPU and device-
 * scope atomics on GPU both see each other's writes immediately.
 *
 * Not shareable across processes — single-process use only.
 *
 * Memory layout:
 *   region_ -> [4KB MemoryBackendHeader | Data]
 */
class GpuShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  std::string url_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~GpuShmMmap() {
#if HSHM_IS_HOST
    _Destroy();
#endif
  }

  /**
   * Initialize UVM backend accessible from both CPU and GPU.
   *
   * @param backend_id Unique identifier for this backend
   * @param backend_size Total size in bytes (headers + data)
   * @param url Identifier string (informational only)
   * @param gpu_id GPU device ID (informational only)
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size,
                const std::string &url, int gpu_id = 0) {
    constexpr size_t kMinBackendSize = 1024 * 1024;
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }
    url_ = url;

    // Allocate Unified Virtual Memory (UVM).
    // cudaMallocManaged provides hardware cache coherence between CPU and GPU
    // on systems with ConcurrentManagedAccess (SM 7.0+), eliminating the need
    // for clflush, write-combining memory, or native host atomics.
    void *ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, backend_size,
                                        cudaMemAttachGlobal);
    if (err != cudaSuccess || !ptr) {
      HLOG(kError, "GpuShmMmap: cudaMallocManaged failed for {}: {}", url,
           cudaGetErrorString(err));
      return false;
    }

    region_ = reinterpret_cast<char *>(ptr);
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    data_ = region_ + kBackendHeaderSize;

    id_ = backend_id;
    backend_size_ = backend_size;
    data_capacity_ = backend_size - kBackendHeaderSize;
    data_id_ = gpu_id;
    flags_.Clear();

    new (header_) MemoryBackendHeader();
    (*header_) = (const MemoryBackendHeader &)*this;

    SetOwner();
    return true;
  }

  /**
   * Attach — not supported (single-process UVM, no cross-process sharing).
   * Returns false. Kept for API compatibility.
   */
  bool shm_attach(const std::string &url) {
    HLOG(kError, "GpuShmMmap::shm_attach not supported (UVM is process-local)");
    return false;
  }

  /** Detach / free */
  void shm_detach() { _Destroy(); }
  void shm_destroy() { _Destroy(); }

 protected:
  void _Detach() { _Destroy(); }

  void _Destroy() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }
    if (region_) {
      cudaFree(region_);
      region_ = nullptr;
      data_ = nullptr;
      header_ = nullptr;
    }
    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
