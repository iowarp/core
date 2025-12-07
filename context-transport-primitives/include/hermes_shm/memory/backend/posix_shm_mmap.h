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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~PosixShmMmap() = default;

  /**
   * Initialize backend with mixed private/shared mapping
   *
   * @param backend_id Unique identifier for this backend
   * @param backend_size Total size of the region (including headers and data)
   * @param url POSIX shared memory object name (e.g., "/my_shm")
   * @return true on success, false on failure
   *
   * Memory layout in file:
   *   [4KB backend header] [4KB shared header] [data]
   *
   * Memory layout in virtual memory (region_):
   *   [4KB private header] [4KB shared header] [data]
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size,
                const std::string &url) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    // File layout: [4KB backend header] [4KB shared header] [data]
    size_t shared_size = backend_size - kBackendHeaderSize;

    // Create shared memory object
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, shared_size)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map the first 4KB (backend header) using MapShared
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, kBackendHeaderSize, 0));
    if (!header_) {
      HILOG(kError, "Failed to map header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Map the remaining using MapMixed
    // Size: 2 * kBackendHeaderSize + custom_header_size + data_size
    size_t remaining_size = backend_size - kBackendHeaderSize;
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(fd_, kBackendHeaderSize, remaining_size, kBackendHeaderSize));
    if (!region_) {
      HILOG(kError, "Failed to create mixed mapping");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // region_ points to start of private header
    // region_ + kBackendHeaderSize points to start of shared header  (maps to file offset 4KB)
    // region_ + 2*kBackendHeaderSize points to start of data

    // Calculate pointers
    char *shared_header_ptr = region_ + kBackendHeaderSize;
    data_ = shared_header_ptr + kBackendHeaderSize;

    md_ = shared_header_ptr;
    md_size_ = kBackendHeaderSize;
    data_capacity_ = backend_size - 2 * kBackendHeaderSize;
    data_id_ = -1;

    // Initialize the header
    new (header_) MemoryBackendHeader();
    header_->id_ = backend_id;
    header_->md_size_ = kBackendHeaderSize;
    header_->backend_size_ = backend_size;
    header_->data_size_ = data_capacity_;
    header_->data_id_ = -1;
    header_->priv_header_off_ = static_cast<size_t>(data_ - region_);
    header_->flags_.Clear();

    return true;
  }

  /**
   * Attach to existing backend with mixed private/shared mapping
   *
   * @param url POSIX shared memory object name
   * @return true on success, false on failure
   */
  bool shm_attach(const std::string &url) {
    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // First, map the first 4KB (backend header) using MapShared
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, kBackendHeaderSize, 0));
    if (!header_) {
      HILOG(kError, "Failed to map header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Read backend size from header
    size_t backend_size = header_->backend_size_;

    // Map the remaining using MapMixed
    size_t remaining_size = backend_size - kBackendHeaderSize;
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(fd_, kBackendHeaderSize, remaining_size, kBackendHeaderSize));
    if (!region_) {
      HILOG(kError, "Failed to create mixed mapping during attach");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Set up pointers (same layout as shm_init)
    char *shared_header_ptr = region_ + kBackendHeaderSize;
    data_ = shared_header_ptr + kBackendHeaderSize;

    md_ = shared_header_ptr;
    md_size_ = kBackendHeaderSize;
    data_capacity_ = header_->data_size_;
    data_id_ = header_->data_id_;

    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  char *_ShmMap(size_t size, i64 off) {
    char *ptr =
        reinterpret_cast<char *>(SystemInfo::MapSharedMemory(fd_, size, off));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (header_ == nullptr) {
      return;
    }
    // Unmap the entire contiguous region
    if (region_ != nullptr) {
      SystemInfo::UnmapMemory(region_, header_->backend_size_ - kBackendHeaderSize);
    }
    if (header_ != nullptr) {
      SystemInfo::UnmapMemory(reinterpret_cast<char *>(header_), kBackendHeaderSize);
    }
    SystemInfo::CloseSharedMemory(fd_);
    header_ = nullptr;
    region_ = nullptr;
  }

  /** Destroy shared memory */
  void _Destroy() {
    _Detach();
    SystemInfo::DestroySharedMemory(url_);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
