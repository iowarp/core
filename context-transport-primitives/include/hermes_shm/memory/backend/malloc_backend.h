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

#ifndef HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
#define HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class MallocBackend : public MemoryBackend {
 private:
  size_t total_size_;
  void *alloc_ptr_;  // Actual allocation start (includes private region)

 public:
  HSHM_CROSS_FUN
  MallocBackend() = default;

  ~MallocBackend() {}

  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    // Total layout: [2*kBackendHeaderSize headers] [data]
    total_size_ = backend_size;

    // Allocate total memory
    char *ptr = (char *)malloc(total_size_);
    if (!ptr) {
      return false;
    }
    alloc_ptr_ = ptr;  // Save allocation start for cleanup

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

  bool shm_attach(const std::string &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  void shm_detach() { _Detach(); }

  void shm_destroy() { _Destroy(); }

 protected:
  void _Detach() {
    if (alloc_ptr_) {
      free(alloc_ptr_);  // Free from allocation start (includes private region)
      alloc_ptr_ = nullptr;
    }
  }

  void _Destroy() {
    if (alloc_ptr_) {
      free(alloc_ptr_);  // Free from allocation start (includes private region)
      alloc_ptr_ = nullptr;
    }
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
