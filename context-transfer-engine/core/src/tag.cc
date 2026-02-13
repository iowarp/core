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

#include <wrp_cte/core/core_client.h>
#include <cstring>
#include <stdexcept>

namespace wrp_cte::core {

Tag::Tag(const std::string &tag_name) : tag_name_(tag_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetOrCreateTag(tag_name);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("GetOrCreateTag operation failed");
  }

  tag_id_ = task->tag_id_;
}

Tag::Tag(const TagId &tag_id) : tag_id_(tag_id), tag_name_("") {}

void Tag::PutBlob(const std::string &blob_name, const char *data, size_t data_size,
                  size_t off, float score, const Context &context) {
  // Allocate shared memory for the data
  auto *ipc_manager = CHI_IPC;
  hipc::FullPtr<char> shm_fullptr = ipc_manager->AllocateBuffer(data_size);

  if (shm_fullptr.IsNull()) {
    throw std::runtime_error("Failed to allocate shared memory for PutBlob");
  }

  // Copy data to shared memory
  memcpy(shm_fullptr.ptr_, data, data_size);

  // Convert to hipc::ShmPtr<> for API call
  hipc::ShmPtr<> shm_ptr(shm_fullptr.shm_);

  // Call SHM version with provided score and context
  PutBlob(blob_name, shm_ptr, data_size, off, score, context);

  // Explicitly free shared memory buffer
  ipc_manager->FreeBuffer(shm_fullptr);
}

void Tag::PutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data, size_t data_size,
                  size_t off, float score, const Context &context) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncPutBlob(tag_id_, blob_name,
                                       off, data_size, data, score, context, 0,
                                       chi::PoolQuery::Dynamic());
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("PutBlob operation failed");
  }
}

// NOTE: AsyncPutBlob(const char*) overload removed due to memory management issues.
// For async operations, the caller must manage shared memory lifecycle by:
// 1. Allocating: hipc::FullPtr<char> shm_ptr = CHI_IPC->AllocateBuffer(data_size);
// 2. Copying data: memcpy(shm_ptr.ptr_, data, data_size);
// 3. Calling: AsyncPutBlob(blob_name, shm_ptr.shm_, data_size, off, score);
// 4. Keeping shm_ptr alive until task completes

chi::Future<PutBlobTask> Tag::AsyncPutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data,
                                             size_t data_size, size_t off, float score,
                                             const Context &context) {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->AsyncPutBlob(tag_id_, blob_name,
                                  off, data_size, data, score, context);
}

void Tag::GetBlob(const std::string &blob_name, char *data, size_t data_size, size_t off) {
  // Validate input parameters
  if (data_size == 0) {
    throw std::invalid_argument("data_size must be specified for GetBlob");
  }

  if (data == nullptr) {
    throw std::invalid_argument("data buffer must be pre-allocated by caller");
  }

  // Allocate shared memory for the data
  auto *ipc_manager = CHI_IPC;
  hipc::FullPtr<char> shm_fullptr = ipc_manager->AllocateBuffer(data_size);

  if (shm_fullptr.IsNull()) {
    throw std::runtime_error("Failed to allocate shared memory for GetBlob");
  }

  // Convert to hipc::ShmPtr<> for API call
  hipc::ShmPtr<> shm_ptr(shm_fullptr.shm_);

  // Call SHM version
  GetBlob(blob_name, shm_ptr, data_size, off);

  // Copy data from shared memory to output buffer
  memcpy(data, shm_fullptr.ptr_, data_size);

  // Explicitly free shared memory buffer
  ipc_manager->FreeBuffer(shm_fullptr);
}

void Tag::GetBlob(const std::string &blob_name, hipc::ShmPtr<> data, size_t data_size, size_t off) {
  // Validate input parameters
  if (data_size == 0) {
    throw std::invalid_argument("data_size must be specified for GetBlob");
  }

  if (data.IsNull()) {
    throw std::invalid_argument("data pointer must be pre-allocated by caller. "
                               "Use CHI_IPC->AllocateBuffer(data_size) to allocate shared memory.");
  }

  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlob(tag_id_, blob_name,
                                       off, data_size, 0, data);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("GetBlob operation failed");
  }

}

float Tag::GetBlobScore(const std::string &blob_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlobScore(tag_id_, blob_name);
  task.Wait();

  float score = task->score_;
  return score;
}

chi::u64 Tag::GetBlobSize(const std::string &blob_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetBlobSize(tag_id_, blob_name);
  task.Wait();

  chi::u64 size = task->size_;
  return size;
}

std::vector<std::string> Tag::GetContainedBlobs() {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetContainedBlobs(tag_id_);
  task.Wait();

  std::vector<std::string> blobs = task->blob_names_;
  return blobs;
}

void Tag::ReorganizeBlob(const std::string &blob_name, float new_score) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncReorganizeBlob(tag_id_, blob_name, new_score);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    throw std::runtime_error("ReorganizeBlob operation failed");
  }
}

} // namespace wrp_cte::core
