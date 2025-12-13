/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

/**
 * Header extension for GpuMalloc backend
 * Stores IPC handle for GPU memory sharing
 */
struct GpuMallocPrivateHeader {
  GpuIpcMemHandle ipc_handle_;  // IPC handle for data_ buffer
};

/**
 * GPU-only memory backend using cudaMalloc/hipMalloc
 *
 * Memory layout (all in GPU memory):
 *   region_ -> [MemoryBackendHeader | GpuMallocPrivateHeader | Data]
 *
 * All memory is allocated with cudaMalloc/hipMalloc on GPU.
 * IPC handle stored in private header enables sharing across processes.
 */
class GpuMalloc : public MemoryBackend, public UrlMemoryBackend {
 protected:
  std::string url_;  // Identifier for this backend (not used for shm)

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuMalloc() = default;

  /** Destructor */
  ~GpuMalloc() {
    if (IsOwner()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /**
   * Initialize backend with GPU memory
   *
   * @param backend_id Unique identifier for this backend
   * @param data_size Size of GPU data buffer
   * @param url Identifier for this backend (not used for shared memory)
   * @param gpu_id GPU device ID
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t data_size,
                const std::string &url, int gpu_id = 0) {
    // Enforce minimum data size of 1MB
    constexpr size_t kMinDataSize = 1024 * 1024;  // 1MB
    if (data_size < kMinDataSize) {
      data_size = kMinDataSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();
    url_ = url;

    // Calculate total size: backend header + private header + data
    size_t header_size = 2 * kBackendHeaderSize;
    size_t total_size = header_size + data_size;

    // Allocate entire region with GPU memory
    region_ = GpuApi::Malloc<char>(total_size);
    if (!region_) {
      HILOG(kError, "Failed to allocate GPU memory");
      return false;
    }

    // Layout in region_: [MemoryBackendHeader | GpuMallocPrivateHeader | Data]
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    GpuMallocPrivateHeader *priv_header =
        reinterpret_cast<GpuMallocPrivateHeader *>(region_ + kBackendHeaderSize);
    data_ = region_ + header_size;

    // Initialize headers on GPU
    MemoryBackendHeader header_init;
    header_init.id_ = backend_id;
    header_init.backend_size_ = total_size;
    header_init.data_capacity_ = data_size;
    header_init.data_id_ = gpu_id;
    header_init.priv_header_off_ = kBackendHeaderSize;
    header_init.flags_.Clear();

    // Copy header to GPU
    GpuApi::Memcpy(header_, &header_init, sizeof(MemoryBackendHeader));

    // Initialize private header with IPC handle
    GpuMallocPrivateHeader priv_header_init;
    GpuApi::GetIpcMemHandle(priv_header_init.ipc_handle_, (void *)region_);
    GpuApi::Memcpy(priv_header, &priv_header_init, sizeof(GpuMallocPrivateHeader));

    // Copy to local object
    id_ = backend_id;
    backend_size_ = total_size;
    data_capacity_ = data_size;
    data_id_ = gpu_id;
    priv_header_off_ = kBackendHeaderSize;

    // Set GPU-only flag
    SetGpuOnly();

    // Mark this process as the owner of the backend
    SetOwner();

    return true;
  }

  /**
   * Attach to existing GPU memory backend
   *
   * @param url Identifier for the backend (must match the IPC handle lookup mechanism)
   * @return true on success, false on failure
   *
   * NOTE: This requires an external IPC handle registry mechanism to share handles between processes.
   * For now, this is a placeholder that will need implementation.
   */
  bool shm_attach(const std::string &url) {
    flags_.Clear();
    url_ = url;

    // TODO: Implement IPC handle registry lookup by URL
    // For now, we can't attach without a shared mechanism to exchange IPC handles
    HILOG(kError, "GpuMalloc::shm_attach requires IPC handle registry (not yet implemented)");
    return false;

    // Future implementation would:
    // 1. Lookup IPC handle from registry using url
    // 2. GpuApi::OpenIpcMemHandle(ipc_handle, &region_)
    // 3. Copy header from GPU to host to read metadata
    // 4. Set up local pointers and flags
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Detach from memory */
  void _Detach() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Clear GPU memory pointers (don't free, we're not the owner)
    region_ = nullptr;
    header_ = nullptr;
    data_ = nullptr;

    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }

  /** Destroy memory */
  void _Destroy() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Free entire GPU region (includes headers and data)
    if (region_) {
      GpuApi::Free(region_);
      region_ = nullptr;
      header_ = nullptr;
      data_ = nullptr;
    }

    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // GPU_MALLOC_H
