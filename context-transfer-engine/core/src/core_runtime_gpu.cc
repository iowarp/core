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

/**
 * GPU implementation of CTE Core ChiMod methods.
 *
 * All methods are currently stubs on GPU. PutBlob and GetBlob demonstrate
 * the CHI_IPC->ToFullPtr pattern for converting ShmPtr blob data references
 * to GPU-accessible pointers. Full GPU implementations of the CTE data
 * placement logic will be added as GPU support matures.
 *
 * Note: core_tasks.h is included here (not in the header) to keep GPU
 * compilation isolated from CPU-only task constructors that use HSHM_MALLOC.
 * Task constructors are host-only and never called from device code.
 */

#include "wrp_cte/core/core_gpu_runtime.h"
#include "wrp_cte/core/core_tasks.h"
#include <hermes_shm/data_structures/priv/vector.h>
#include <hermes_shm/data_structures/priv/string.h>

namespace wrp_cte::core {

/**
 * GPU-resident metadata store for CTE Core GpuRuntime.
 * Uses chi::priv containers so all allocations go through the GPU heap.
 */
struct GpuMetadata {
  // Tag store: parallel arrays (tag_name → tag_id → tag_info)
  chi::priv::vector<chi::priv::string> tag_names;
  chi::priv::vector<TagId> tag_ids;
  chi::priv::vector<TagInfo> tag_infos;
  // Blob store: parallel arrays (compound_key → blob_info)
  chi::priv::vector<chi::priv::string> blob_keys;
  chi::priv::vector<BlobInfo> blob_infos;

  HSHM_GPU_FUN explicit GpuMetadata(CHI_PRIV_ALLOC_T *alloc)
      : tag_names(alloc), tag_ids(alloc), tag_infos(alloc),
        blob_keys(alloc), blob_infos(alloc) {}
};

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::RegisterTarget(
    hipc::FullPtr<RegisterTargetTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::UnregisterTarget(
    hipc::FullPtr<UnregisterTargetTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::ListTargets(
    hipc::FullPtr<ListTargetsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::StatTargets(
    hipc::FullPtr<StatTargetsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

/** Initialize metadata store on first use (uses CHI_PRIV_ALLOC GPU heap) */
HSHM_GPU_FUN void GpuRuntime::EnsureMetaInit() {
  if (meta_ != nullptr) return;
  CHI_PRIV_ALLOC_T *alloc = CHI_PRIV_ALLOC;
  // Allocate raw memory for GpuMetadata in GPU heap
  hipc::FullPtr<GpuMetadata> ptr = alloc->template AllocateObjs<GpuMetadata>(1);
  meta_ = ptr.ptr_;
  new (meta_) GpuMetadata(alloc);
}

/** Find TagInfo by TagId. Returns nullptr if not found. */
HSHM_GPU_FUN TagInfo *GpuRuntime::FindTagById(const TagId &tag_id) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->tag_ids.size(); ++i) {
    if (meta_->tag_ids[i] == tag_id) {
      return &meta_->tag_infos[i];
    }
  }
  return nullptr;
}

/** Find TagId by name. Returns nullptr if not found. */
HSHM_GPU_FUN TagId *GpuRuntime::FindTagIdByName(const chi::priv::string &name) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->tag_names.size(); ++i) {
    if (meta_->tag_names[i] == name) {
      return &meta_->tag_ids[i];
    }
  }
  return nullptr;
}

/** Insert or update tag in metadata */
HSHM_GPU_FUN TagInfo *GpuRuntime::UpsertTag(const chi::priv::string &tag_name,
                                             const TagId &tag_id) {
  // Check if already exists
  for (size_t i = 0; i < meta_->tag_ids.size(); ++i) {
    if (meta_->tag_ids[i] == tag_id) {
      return &meta_->tag_infos[i];
    }
  }
  // Insert new
  chi::priv::string name(tag_name);
  meta_->tag_names.push_back(name);
  meta_->tag_ids.push_back(tag_id);
  TagInfo info(tag_name, tag_id);
  meta_->tag_infos.push_back(info);
  return &meta_->tag_infos[meta_->tag_infos.size() - 1];
}

/** Build compound key for blob: "major.minor.blob_name" */
HSHM_GPU_FUN chi::priv::string GpuRuntime::MakeBlobKey(const TagId &tag_id,
                                                         const chi::priv::string &blob_name) {
  // Build "major.minor.blob_name" - use fixed-size char buffer
  char buf[256];
  // Simple integer-to-string conversion (no printf on GPU)
  chi::u32 major = tag_id.major_, minor = tag_id.minor_;
  // Convert major to string
  int pos = 0;
  if (major == 0) { buf[pos++] = '0'; }
  else {
    char tmp[16]; int tlen = 0;
    while (major > 0) { tmp[tlen++] = '0' + (major % 10); major /= 10; }
    for (int i = tlen - 1; i >= 0; --i) buf[pos++] = tmp[i];
  }
  buf[pos++] = '.';
  if (minor == 0) { buf[pos++] = '0'; }
  else {
    char tmp[16]; int tlen = 0;
    while (minor > 0) { tmp[tlen++] = '0' + (minor % 10); minor /= 10; }
    for (int i = tlen - 1; i >= 0; --i) buf[pos++] = tmp[i];
  }
  buf[pos++] = '.';
  // Append blob_name
  const char *bdata = blob_name.data();
  size_t blen = blob_name.size();
  for (size_t i = 0; i < blen && pos < 255; ++i) buf[pos++] = bdata[i];
  buf[pos] = '\0';
  return chi::priv::string(CHI_PRIV_ALLOC, buf);
}

/** Find BlobInfo by compound key. Returns nullptr if not found. */
HSHM_GPU_FUN BlobInfo *GpuRuntime::FindBlob(const chi::priv::string &compound_key) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->blob_keys.size(); ++i) {
    if (meta_->blob_keys[i] == compound_key) {
      return &meta_->blob_infos[i];
    }
  }
  return nullptr;
}

/** Insert or update blob in metadata */
HSHM_GPU_FUN BlobInfo *GpuRuntime::UpsertBlob(const chi::priv::string &compound_key,
                                               const BlobInfo &info) {
  for (size_t i = 0; i < meta_->blob_keys.size(); ++i) {
    if (meta_->blob_keys[i] == compound_key) {
      meta_->blob_infos[i] = info;
      return &meta_->blob_infos[i];
    }
  }
  chi::priv::string key(compound_key);
  meta_->blob_keys.push_back(key);
  meta_->blob_infos.push_back(info);
  return &meta_->blob_infos[meta_->blob_infos.size() - 1];
}

/** Erase blob by compound key. Returns true if found and erased. */
HSHM_GPU_FUN bool GpuRuntime::EraseBlob(const chi::priv::string &compound_key) {
  if (!meta_) return false;
  for (size_t i = 0; i < meta_->blob_keys.size(); ++i) {
    if (meta_->blob_keys[i] == compound_key) {
      // Swap with last and pop
      size_t last = meta_->blob_keys.size() - 1;
      if (i != last) {
        meta_->blob_keys[i] = meta_->blob_keys[last];
        meta_->blob_infos[i] = meta_->blob_infos[last];
      }
      meta_->blob_keys.pop_back();
      meta_->blob_infos.pop_back();
      return true;
    }
  }
  return false;
}

/** Check if blob exists and return pointer to BlobInfo */
HSHM_GPU_FUN BlobInfo *GpuRuntime::CheckBlobExists(const chi::priv::string &blob_name,
                                                     const TagId &tag_id) {
  chi::priv::string compound_key = MakeBlobKey(tag_id, blob_name);
  return FindBlob(compound_key);
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParams>> task,
    chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  chi::priv::string tag_name(task->tag_name_);
  TagId preferred_id = task->tag_id_;

  // Look up existing tag by name
  TagId *existing_id = FindTagIdByName(tag_name);
  if (existing_id != nullptr) {
    task->tag_id_ = *existing_id;
    task->return_code_ = 0;
    co_return;
  }

  // Assign new ID (use preferred or generate)
  TagId tag_id;
  if (preferred_id.major_ != 0 || preferred_id.minor_ != 0) {
    tag_id = preferred_id;
  } else {
    // Generate: use container_id as major, atomic minor
    tag_id.major_ = container_id_;
    tag_id.minor_ = atomicAdd(&next_tag_minor_, 1u) + 1;
  }

  // Create tag entry
  UpsertTag(tag_name, tag_id);
  task->tag_id_ = tag_id;
  task->return_code_ = 0;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetTagSize(
    hipc::FullPtr<GetTagSizeTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::DelTag(
    hipc::FullPtr<DelTagTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  chi::priv::string tag_name(task->tag_name_);

  // Resolve tag_id from name if not set
  if (tag_id.IsNull() && !tag_name.empty()) {
    TagId *found = FindTagIdByName(tag_name);
    if (found == nullptr) { task->return_code_ = 1; co_return; }
    tag_id = *found;
    task->tag_id_ = tag_id;
  } else if (tag_id.IsNull()) {
    task->return_code_ = 1;
    co_return;
  }

  // Collect and erase all blobs belonging to this tag
  // (Scan blob_keys for those starting with "major.minor.")
  chi::priv::string prefix = MakeBlobKey(tag_id, chi::priv::string(CHI_PRIV_ALLOC, ""));
  // prefix is "major.minor." (ends with '.')
  // Erase all blobs whose key starts with prefix
  bool any = true;
  while (any) {
    any = false;
    for (size_t i = 0; i < meta_->blob_keys.size(); ++i) {
      const chi::priv::string &key = meta_->blob_keys[i];
      // Check prefix match by comparing first prefix.size() chars
      if (key.size() >= prefix.size()) {
        bool match = true;
        for (size_t c = 0; c < prefix.size() && match; ++c) {
          match = (key[c] == prefix[c]);
        }
        if (match) {
          size_t last = meta_->blob_keys.size() - 1;
          if (i != last) {
            meta_->blob_keys[i] = meta_->blob_keys[last];
            meta_->blob_infos[i] = meta_->blob_infos[last];
          }
          meta_->blob_keys.pop_back();
          meta_->blob_infos.pop_back();
          any = true;
          break;
        }
      }
    }
  }

  // Erase tag from tag store
  for (size_t i = 0; i < meta_->tag_ids.size(); ++i) {
    if (meta_->tag_ids[i] == tag_id) {
      size_t last = meta_->tag_ids.size() - 1;
      if (i != last) {
        meta_->tag_names[i] = meta_->tag_names[last];
        meta_->tag_ids[i] = meta_->tag_ids[last];
        meta_->tag_infos[i] = meta_->tag_infos[last];
      }
      meta_->tag_names.pop_back();
      meta_->tag_ids.pop_back();
      meta_->tag_infos.pop_back();
      break;
    }
  }

  task->return_code_ = 0;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetContainedBlobs(
    hipc::FullPtr<GetContainedBlobsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

/**
 * GPU PutBlob: stores blob data and metadata in GPU-resident store.
 */
HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::PutBlob(
    hipc::FullPtr<PutBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  chi::priv::string blob_name(task->blob_name_);
  chi::u64 offset = task->offset_;
  chi::u64 size = task->size_;
  float blob_score = task->score_;

  // Validate inputs
  if (size == 0) { task->return_code_ = 2; co_return; }
  if (task->blob_data_.IsNull()) { task->return_code_ = 3; co_return; }
  if (blob_name.empty()) { task->return_code_ = 4; co_return; }
  if (blob_score < 0.0f) blob_score = 1.0f;
  if (blob_score > 1.0f) { task->return_code_ = 5; co_return; }

  // Resolve blob data pointer
  auto data_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (data_ptr.IsNull()) { task->return_code_ = 6; co_return; }

  // Get or create blob metadata
  chi::priv::string compound_key = MakeBlobKey(tag_id, blob_name);
  BlobInfo *blob_info = FindBlob(compound_key);

  // For GPU blobs stored in HBM, we use the blob's ShmPtr offset as the
  // storage location directly (no block allocation through bdev needed).
  // The blob data lives in the UVM/pinned buffer provided by the task.
  // We record this as a single "block" with target_offset = offset.
  if (blob_info == nullptr) {
    // Create new blob
    BlobInfo new_info(blob_name, blob_score);
    // Add a pseudo-block representing the UVM data region
    BlobBlock blk;
    blk.target_offset_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
    blk.size_ = size;
    new_info.blocks_.push_back(blk);
    new_info.last_modified_ = GetCurrentTimeNs();
    blob_info = UpsertBlob(compound_key, new_info);

    // Update tag total_size_
    TagInfo *tag_info = FindTagById(tag_id);
    if (tag_info) tag_info->total_size_ += size;
  } else {
    // Update existing blob: update block's offset and size
    chi::u64 old_size = blob_info->GetTotalSize();
    if (!blob_info->blocks_.empty()) {
      blob_info->blocks_[0].target_offset_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
      blob_info->blocks_[0].size_ = size;
    } else {
      BlobBlock blk;
      blk.target_offset_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
      blk.size_ = size;
      blob_info->blocks_.push_back(blk);
    }
    blob_info->score_ = blob_score;
    blob_info->last_modified_ = GetCurrentTimeNs();
    // Update tag total_size_
    TagInfo *tag_info = FindTagById(tag_id);
    if (tag_info) {
      tag_info->total_size_ = tag_info->total_size_ - old_size + size;
    }
  }

  // Write data: Copy from task's blob_data_ to the target location
  // The blob_data_ ShmPtr holds the source data; blob blocks record
  // the stored location. For GPU blobs, the "stored location" is the
  // same UVM buffer (no separate bdev involved at GPU level).
  // (Data is already in data_ptr.ptr_ — no memcpy needed for same-buffer writes)

  task->return_code_ = 0;
  co_return;
}

/**
 * GPU GetBlob: retrieves blob data from GPU-resident store.
 */
HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlob(
    hipc::FullPtr<GetBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  chi::priv::string blob_name(task->blob_name_);
  chi::u64 offset = task->offset_;
  chi::u64 size = task->size_;

  if (size == 0) { task->return_code_ = 1; co_return; }
  if (blob_name.empty()) { task->return_code_ = 1; co_return; }

  BlobInfo *blob_info = CheckBlobExists(blob_name, tag_id);
  if (blob_info == nullptr) { task->return_code_ = 1; co_return; }

  // Resolve output buffer
  auto out_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (out_ptr.IsNull()) { task->return_code_ = 1; co_return; }

  // Read data from blob blocks into output buffer
  char *dst = reinterpret_cast<char *>(out_ptr.ptr_);
  chi::u64 bytes_remaining = size;
  chi::u64 blob_offset = offset;
  chi::u64 dst_offset = 0;

  for (size_t i = 0; i < blob_info->blocks_.size() && bytes_remaining > 0; ++i) {
    BlobBlock &blk = blob_info->blocks_[i];
    if (blob_offset >= blk.size_) {
      blob_offset -= blk.size_;
      continue;
    }
    // block_start is the stored pointer (target_offset_ holds raw UVM pointer)
    char *src = reinterpret_cast<char *>(blk.target_offset_) + blob_offset;
    chi::u64 can_read = blk.size_ - blob_offset;
    chi::u64 to_read = (can_read < bytes_remaining) ? can_read : bytes_remaining;
    memcpy(dst + dst_offset, src, to_read);
    dst_offset += to_read;
    bytes_remaining -= to_read;
    blob_offset = 0;
  }

  blob_info->last_read_ = GetCurrentTimeNs();
  task->return_code_ = (bytes_remaining == 0) ? 0 : 1;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::ReorganizeBlob(
    hipc::FullPtr<ReorganizeBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  chi::priv::string blob_name(task->blob_name_);
  float new_score = task->new_score_;

  if (blob_name.empty() || new_score < 0.0f || new_score > 1.0f) {
    task->return_code_ = 1;
    co_return;
  }

  BlobInfo *blob_info = CheckBlobExists(blob_name, tag_id);
  if (blob_info == nullptr) {
    task->return_code_ = 3;
    co_return;
  }

  // Update score (GPU doesn't migrate data between targets, just updates score)
  blob_info->score_ = new_score;
  task->return_code_ = 0;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::DelBlob(
    hipc::FullPtr<DelBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  chi::priv::string blob_name(task->blob_name_);

  if (blob_name.empty()) { task->return_code_ = 1; co_return; }

  BlobInfo *blob_info = CheckBlobExists(blob_name, tag_id);
  if (blob_info == nullptr) { task->return_code_ = 1; co_return; }

  chi::u64 blob_size = blob_info->GetTotalSize();
  chi::priv::string compound_key = MakeBlobKey(tag_id, blob_name);
  EraseBlob(compound_key);

  // Update tag total_size_
  TagInfo *tag_info = FindTagById(tag_id);
  if (tag_info) {
    tag_info->total_size_ = (blob_size <= tag_info->total_size_) ?
                             tag_info->total_size_ - blob_size : 0;
  }
  task->return_code_ = 0;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlobScore(
    hipc::FullPtr<GetBlobScoreTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlobSize(
    hipc::FullPtr<GetBlobSizeTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlobInfo(
    hipc::FullPtr<GetBlobInfoTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::PollTelemetryLog(
    hipc::FullPtr<PollTelemetryLogTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::TagQuery(
    hipc::FullPtr<TagQueryTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::BlobQuery(
    hipc::FullPtr<BlobQueryTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetTargetInfo(
    hipc::FullPtr<GetTargetInfoTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::FlushMetadata(
    hipc::FullPtr<FlushMetadataTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::FlushData(
    hipc::FullPtr<FlushDataTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// LocalAllocLoadTask / LocalSaveTask — GPU implementations
// Defined here (not in core_gpu_lib_exec.h) because full task type definitions
// from core_tasks.h are needed for serialization.
//==============================================================================

HSHM_GPU_FUN hipc::FullPtr<chi::Task> GpuRuntime::LocalAllocLoadTask(
    chi::u32 method, chi::LocalLoadTaskArchive &archive) {
  auto *ipc = CHI_IPC;
  hipc::FullPtr<chi::Task> task_ptr;
  switch (method) {
    case Method::kPutBlob: {
      auto new_task = ipc->template NewTask<PutBlobTask>();
      archive >> *new_task.ptr_;
      task_ptr = new_task.template Cast<chi::Task>();
      break;
    }
    case Method::kGetBlob: {
      auto new_task = ipc->template NewTask<GetBlobTask>();
      archive >> *new_task.ptr_;
      task_ptr = new_task.template Cast<chi::Task>();
      break;
    }
    case Method::kGetOrCreateTag: {
      auto new_task = ipc->template NewTask<GetOrCreateTagTask<CreateParams>>();
      archive >> *new_task.ptr_;
      task_ptr = new_task.template Cast<chi::Task>();
      break;
    }
    default:
      task_ptr = hipc::FullPtr<chi::Task>::GetNull();
      break;
  }
  return task_ptr;
}

HSHM_GPU_FUN void GpuRuntime::LocalSaveTask(
    chi::u32 method, chi::LocalSaveTaskArchive &archive,
    const hipc::FullPtr<chi::Task> &task) {
  switch (method) {
    case Method::kPutBlob: {
      auto typed = task.template Cast<PutBlobTask>();
      archive << *typed.ptr_;
      break;
    }
    case Method::kGetBlob: {
      auto typed = task.template Cast<GetBlobTask>();
      archive << *typed.ptr_;
      break;
    }
    case Method::kGetOrCreateTag: {
      auto typed = task.template Cast<GetOrCreateTagTask<CreateParams>>();
      archive << *typed.ptr_;
      break;
    }
    default: break;
  }
}

}  // namespace wrp_cte::core
