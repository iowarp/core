#include <wrp_cte/core/core_client.h>
#include <cstring>
#include <stdexcept>

namespace wrp_cte::core {

Tag::Tag(const std::string &tag_name) : tag_name_(tag_name) {
  // Call the WRP_CTE client GetOrCreateTag function
  auto *cte_client = WRP_CTE_CLIENT;
  tag_id_ = cte_client->GetOrCreateTag(tag_name);
}

Tag::Tag(const TagId &tag_id) : tag_id_(tag_id), tag_name_("") {}

void Tag::PutBlob(const std::string &blob_name, const char *data, size_t data_size, size_t off) {
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

  // Call SHM version with default score of 1.0
  PutBlob(blob_name, shm_ptr, data_size, off, 1.0f);

  // Explicitly free shared memory buffer
  ipc_manager->FreeBuffer(shm_fullptr);
}

void Tag::PutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data, size_t data_size,
                  size_t off, float score) {
  auto *cte_client = WRP_CTE_CLIENT;
  bool result = cte_client->PutBlob(tag_id_, blob_name,
                                    off, data_size, data, score, 0);
  if (!result) {
    throw std::runtime_error("PutBlob operation failed");
  }
}

// NOTE: AsyncPutBlob(const char*) overload removed due to memory management issues.
// For async operations, the caller must manage shared memory lifecycle by:
// 1. Allocating: hipc::FullPtr<char> shm_ptr = CHI_IPC->AllocateBuffer(data_size);
// 2. Copying data: memcpy(shm_ptr.ptr_, data, data_size);
// 3. Calling: AsyncPutBlob(blob_name, shm_ptr.shm_, data_size, off, score);
// 4. Keeping shm_ptr alive until task completes

hipc::FullPtr<PutBlobTask> Tag::AsyncPutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data,
                                             size_t data_size, size_t off, float score) {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->AsyncPutBlob(tag_id_, blob_name,
                                  off, data_size, data, score, 0);
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
  bool result = cte_client->GetBlob(tag_id_, blob_name,
                                    off, data_size, 0, data);
  if (!result) {
    throw std::runtime_error("GetBlob operation failed");
  }
}

float Tag::GetBlobScore(const std::string &blob_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->GetBlobScore(tag_id_, blob_name);
}

chi::u64 Tag::GetBlobSize(const std::string &blob_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->GetBlobSize(tag_id_, blob_name);
}

std::vector<std::string> Tag::GetContainedBlobs() {
  auto *cte_client = WRP_CTE_CLIENT;
  return cte_client->GetContainedBlobs(tag_id_);
}

} // namespace wrp_cte::core