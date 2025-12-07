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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixMmap : public MemoryBackend {
 private:
  size_t total_size_;
  void *map_ptr_;  // Actual mapping start (includes private region)

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixMmap() = default;

  /** Destructor */
  ~PosixMmap() = default;

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    // Total layout: [2*kBackendHeaderSize headers] [data]
    total_size_ = backend_size;

    // Map memory
    char *ptr = _Map(total_size_);
    if (!ptr) {
      return false;
    }
    map_ptr_ = ptr;  // Save mapping start for cleanup

    region_ = ptr;
    char *shared_header_ptr = ptr + kBackendHeaderSize;

    // Initialize header at shared header location
    header_ = reinterpret_cast<MemoryBackendHeader *>(shared_header_ptr);
    new (header_) MemoryBackendHeader();
    header_->id_ = backend_id;
    header_->md_size_ = kBackendHeaderSize;
    header_->backend_size_ = backend_size;
    header_->data_size_ = backend_size - 2 * kBackendHeaderSize;
    header_->data_id_ = -1;
    header_->priv_header_off_ = static_cast<size_t>(shared_header_ptr + kBackendHeaderSize - ptr);
    header_->flags_.Clear();

    // md_ points to the shared header
    md_ = shared_header_ptr;
    md_size_ = kBackendHeaderSize;

    // data_ starts after shared header
    data_ = shared_header_ptr + kBackendHeaderSize;
    data_capacity_ = header_->data_size_;
    data_id_ = -1;

    return true;
  }

  /** Deserialize the backend */
  bool shm_attach(const std::string &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  template <typename T = char>
  T *_Map(size_t size) {
    T *ptr = reinterpret_cast<T *>(
        SystemInfo::MapPrivateMemory(MemoryAlignment::AlignToPageSize(size)));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (map_ptr_) {
      SystemInfo::UnmapMemory(map_ptr_, total_size_);  // Unmap from mapping start (includes private region)
      map_ptr_ = nullptr;
    }
  }

  /** Destroy shared memory */
  void _Destroy() {
    _Detach();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
