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
 * Uses chi::priv data structures (string, vector) backed by the
 * PartitionedAllocator which provides per-block BuddyAllocator partitions,
 * eliminating cross-block allocator contention (CUDA Error 700).
 *
 * Note: core_tasks.h is included here (not in the header) to keep GPU
 * compilation isolated from CPU-only task constructors that use HSHM_MALLOC.
 */

#include "wrp_cte/core/core_gpu_runtime.h"
#include "wrp_cte/core/core_tasks.h"
#include <hermes_shm/data_structures/priv/vector.h>
#include <hermes_shm/data_structures/priv/string.h>
#include <hermes_shm/data_structures/priv/unordered_map_ll.h>
#include <hermes_shm/thread/lock/mutex.h>

// Backend-conditional GPU intrinsic wrappers (HSHM_DEVICE_FENCE_DEVICE,
// HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE) so this TU compiles under both
// nvcc/hipcc (CUDA path) and DPC++/AdaptiveCpp (SYCL path).
#include "hermes_shm/util/gpu_intrinsics.h"

namespace wrp_cte::core {

/** Default number of blob map slots (open addressing) */
static constexpr chi::u32 kDefaultBlobMapCapacity = 64;

/**
 * GPU-side blob entry using chi::priv data structures.
 * blocks_ uses the shared (PartitionedAllocator) scope because blob entries
 * are stored in the cross-warp blob_map_ and may be modified by any warp.
 */
struct GpuBlobEntry {
  chi::priv::shared_vector<BlobBlock> blocks_;
  chi::u64 size_;               // blob size in bytes
  float score_;
  Timestamp last_modified_;
  Timestamp last_read_;

  HSHM_CROSS_FUN GpuBlobEntry()
      : blocks_(CHI_PRIV_SHARED_ALLOC), size_(0), score_(0.0f),
        last_modified_(0), last_read_(0) {}

  HSHM_CROSS_FUN GpuBlobEntry(const GpuBlobEntry &other)
      : blocks_(other.blocks_), size_(other.size_),
        score_(other.score_),
        last_modified_(other.last_modified_),
        last_read_(other.last_read_) {}

  HSHM_CROSS_FUN GpuBlobEntry &operator=(const GpuBlobEntry &other) {
    if (this != &other) {
      blocks_ = other.blocks_;
      size_ = other.size_;
      score_ = other.score_;
      last_modified_ = other.last_modified_;
      last_read_ = other.last_read_;
    }
    return *this;
  }
};

/** Blob map: compound key → GpuBlobEntry (shared across warps) */
using GpuBlobMap = chi::priv::shared_unordered_map<
    chi::priv::string, GpuBlobEntry>;

/** Tag map: TagId (as u64) → TagInfo (shared across warps) */
using GpuTagIdMap = chi::priv::shared_unordered_map<
    chi::u64, TagInfo>;

/** Reverse tag index: tag name → TagId (as u64) (shared across warps) */
using GpuTagNameMap = chi::priv::shared_unordered_map<
    chi::priv::string, chi::u64>;

/** Default tag map capacity */
static constexpr chi::u32 kDefaultTagMapCapacity = 64;

/**
 * GPU-resident metadata store for CTE Core GpuRuntime.
 * All maps and vectors use the shared allocator (PartitionedAllocator)
 * so multiple warps can safely allocate/free map infrastructure concurrently.
 */
struct GpuMetadata {
  GpuBlobMap blob_map_;
  GpuTagIdMap tag_id_map_;
  GpuTagNameMap tag_name_to_id_;
  chi::priv::shared_vector<TargetInfo> targets_;

  HSHM_GPU_FUN GpuMetadata()
      : blob_map_(CHI_PRIV_SHARED_ALLOC, kDefaultBlobMapCapacity),
        tag_id_map_(CHI_PRIV_SHARED_ALLOC, kDefaultTagMapCapacity),
        tag_name_to_id_(CHI_PRIV_SHARED_ALLOC, kDefaultTagMapCapacity),
        targets_(CHI_PRIV_SHARED_ALLOC) {
  }
};

//==============================================================================
// Helper: build compound key "major.minor.blob_name"
//==============================================================================

/**
 * Append the decimal digits of a u32 to a priv::string.
 */
HSHM_GPU_FUN static void AppendU32(chi::priv::string &s, chi::u32 val) {
  if (val == 0) {
    s.push_back('0');
    return;
  }
  // Find digit count
  int ndigits = 0;
  chi::u32 tmp = val;
  while (tmp > 0) { ++ndigits; tmp /= 10; }
  // Append digits in forward order
  size_t start = s.size();
  s.resize(start + ndigits);
  for (int i = ndigits - 1; i >= 0; --i) {
    s[start + i] = '0' + static_cast<char>(val % 10);
    val /= 10;
  }
}

HSHM_GPU_FUN static chi::priv::string MakeCompoundKey(
    const TagId &tag_id, const char *blob_name, int blob_name_len) {
  // Build directly into heap-backed priv::string (no stack buffers)
  chi::priv::string ck(CHI_PRIV_ALLOC);
  // Reserve: up to 10+1+10+1+blob_name_len
  ck.reserve(22 + blob_name_len);
  AppendU32(ck, tag_id.major_);
  ck.push_back('.');
  AppendU32(ck, tag_id.minor_);
  ck.push_back('.');
  for (int i = 0; i < blob_name_len; ++i) {
    ck.push_back(blob_name[i]);
  }
  return ck;
}

//==============================================================================
// Stub methods (no-ops on GPU)
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::RegisterTarget(
    hipc::FullPtr<RegisterTargetTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;

  EnsureMetaInit(g_ipc_manager_ptr);

  // Build TargetInfo directly — avoid string allocations that require
  // the GPU allocator (may not be available in CPU→GPU CDP context).
  TargetInfo info;

  // Use task's SSO string data directly (FixupAfterCopy already reseated data_)
  info.target_name_.InitFromSso(task->target_name_);

  info.bdev_client_.Init(task->bdev_id_);
  info.target_query_ = task->target_query_;
  info.remaining_space_ = task->total_size_;

  // Push to targets vector
  meta_->targets_.push_back(info);

  task->return_code_ = 0;
}

HSHM_GPU_FUN void GpuRuntime::UnregisterTarget(
    hipc::FullPtr<UnregisterTargetTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::ListTargets(
    hipc::FullPtr<ListTargetsTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::StatTargets(
    hipc::FullPtr<StatTargetsTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

//==============================================================================
// EnsureMetaInit — double-checked locking with threadfence
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::EnsureMetaInit(
    chi::gpu::IpcManager *ipc_mgr_in) {
  // Bind g_ipc_manager_ptr so CHI_IPC (= g_ipc_manager_ptr under SYCL)
  // resolves correctly inside this method body. ipc_mgr_in is passed by
  // every caller from its rctx.ipc_mgr_; on CUDA/ROCm it's nullptr and
  // CHI_IPC reaches the per-block __shared__ singleton.
  [[maybe_unused]] auto *g_ipc_manager_ptr = ipc_mgr_in;
  // Scratch persists across pause/resume — metadata is never invalidated.
  GpuMetadata *m = *reinterpret_cast<GpuMetadata *volatile *>(&meta_);
  if (m != nullptr) return;
  hshm::ScopedMutex guard(init_lock_, 0);
  m = *reinterpret_cast<GpuMetadata *volatile *>(&meta_);
  if (m != nullptr) return;
  hipc::FullPtr<GpuMetadata> ptr =
      CHI_IPC->NewObj<GpuMetadata>();
  ptr->targets_.reserve(16);
  HSHM_DEVICE_FENCE_DEVICE();
  meta_ = ptr.ptr_;
  HSHM_DEVICE_FENCE_DEVICE();
}

//==============================================================================
// Tag operations
//==============================================================================

HSHM_GPU_FUN TagInfo *GpuRuntime::FindTagById(const TagId &tag_id) {
  if (!meta_) return nullptr;
  return meta_->tag_id_map_.find(tag_id.ToU64());
}

HSHM_GPU_FUN chi::u64 *GpuRuntime::FindTagIdByName(
    const chi::priv::string &name) {
  if (!meta_) return nullptr;
  return meta_->tag_name_to_id_.find(name);
}

HSHM_GPU_FUN TagInfo *GpuRuntime::UpsertTag(const chi::priv::string &tag_name,
                                             const TagId &tag_id) {
  if (!meta_) return nullptr;
  chi::u64 id_key = tag_id.ToU64();
  // Check if tag exists
  TagInfo *existing = meta_->tag_id_map_.find(id_key);
  if (existing) return existing;
  // Insert into both maps
  TagInfo info(tag_name, tag_id);
  meta_->tag_id_map_.insert(id_key, info);
  meta_->tag_name_to_id_.insert(tag_name, id_key);
  return meta_->tag_id_map_.find(id_key);
}

//==============================================================================
// Blob map operations
//==============================================================================

HSHM_GPU_FUN chi::priv::string GpuRuntime::MakeBlobKey(const TagId &tag_id,
                                                         const chi::priv::string &blob_name) {
  return MakeCompoundKey(tag_id, blob_name.data(),
                         static_cast<int>(blob_name.size()));
}

//==============================================================================
// GetOrCreateTag
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParams>> task,
    chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;
  EnsureMetaInit(g_ipc_manager_ptr);

  chi::priv::string name(CHI_PRIV_ALLOC, task->tag_name_.data());
  TagId preferred_id = task->tag_id_;

  // Look up existing tag by name (fine-grained lock inside map)
  chi::u64 *existing = FindTagIdByName(name);
  if (existing != nullptr) {
    task->tag_id_ = TagId::FromU64(*existing);
    task->return_code_ = 0;
    return;
  }

  // Assign new ID
  TagId tag_id;
  if (preferred_id.major_ != 0 || preferred_id.minor_ != 0) {
    tag_id = preferred_id;
  } else {
    tag_id.major_ = container_id_;
    tag_id.minor_ = HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE(&next_tag_minor_, 1u) + 1;
  }

  // Insert into both maps (fine-grained locks inside)
  UpsertTag(name, tag_id);

  task->tag_id_ = tag_id;
  task->return_code_ = 0;
}

//==============================================================================
// GetTagSize
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::GetTagSize(
    hipc::FullPtr<GetTagSizeTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

//==============================================================================
// DelTag
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::DelTag(
    hipc::FullPtr<DelTagTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;
  EnsureMetaInit(g_ipc_manager_ptr);

  TagId tag_id = task->tag_id_;

  // Resolve tag_id from name if needed
  if (tag_id.IsNull()) {
    chi::priv::string name(CHI_PRIV_ALLOC, task->tag_name_.data());
    if (name.size() == 0) {
      task->return_code_ = 1;
      return;
    }
    chi::u64 *found = FindTagIdByName(name);
    if (found == nullptr) {
      task->return_code_ = 1;
      return;
    }
    tag_id = TagId::FromU64(*found);
    task->tag_id_ = tag_id;
  }

  // Build prefix "major.minor." for matching and erase all matching blobs
  chi::priv::string prefix = MakeCompoundKey(tag_id, "", 0);
  // Collect keys to erase (can't erase during for_each)
  chi::priv::vector<chi::priv::string> keys_to_erase(CHI_PRIV_ALLOC);
  meta_->blob_map_.for_each(
      [&](const chi::priv::string &key, GpuBlobEntry &) {
        if (key.size() >= prefix.size()) {
          bool match = true;
          const char *kd = key.data();
          const char *pd = prefix.data();
          for (size_t c = 0; c < prefix.size() && match; ++c) {
            match = (kd[c] == pd[c]);
          }
          if (match) keys_to_erase.push_back(key);
        }
      });
  for (size_t i = 0; i < keys_to_erase.size(); ++i) {
    meta_->blob_map_.erase(keys_to_erase[i]);
  }

  // Erase tag from both tag maps
  TagInfo *tag_info = FindTagById(tag_id);
  if (tag_info != nullptr) {
    chi::priv::string tag_name = tag_info->tag_name_;
    meta_->tag_id_map_.erase(tag_id.ToU64());
    meta_->tag_name_to_id_.erase(tag_name);
  }

  task->return_code_ = 0;
}

//==============================================================================
// GetContainedBlobs
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::GetContainedBlobs(
    hipc::FullPtr<GetContainedBlobsTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

//==============================================================================
// PutBlob
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::PutBlob(
    hipc::FullPtr<PutBlobTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;
  EnsureMetaInit(g_ipc_manager_ptr);
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  chi::u64 size = task->size_;
  float blob_score = task->score_;

  // Validate inputs
  if (size == 0) { task->return_code_ = 2; return; }
  if (task->blob_data_.IsNull()) { task->return_code_ = 3; return; }
  if (blob_name_len == 0) { task->return_code_ = 4; return; }
  if (blob_score < 0.0f) blob_score = 1.0f;
  if (blob_score > 1.0f) { task->return_code_ = 5; return; }

  // Resolve blob data pointer
  auto data_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (data_ptr.IsNull()) { task->return_code_ = 6; return; }

  // Build compound key and lock the blob map bucket
  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  GpuBlobEntry *entry = nullptr;
  bool is_new_blob = false;

  // Scope 1: Lock blob key, find or create entry, clear old blocks if needed
  {
    meta_->blob_map_.lock_key(ck);
    entry = meta_->blob_map_.find_locked(ck);
    if (entry == nullptr) {
      auto result = meta_->blob_map_.insert_locked(ck, GpuBlobEntry());
      if (result.value == nullptr) {
        meta_->blob_map_.unlock_key(ck);
        task->return_code_ = 10;
        return;
      }
      entry = result.value;
      is_new_blob = true;
    } else {
      // Clear old blocks
      entry->blocks_.clear();
    }
    meta_->blob_map_.unlock_key(ck);
  }

  // Find a target with sufficient space (write-once targets, no lock needed)
  TargetInfo target_info;
  chi::u64 old_blob_size = entry->size_;
  {
    bool found = false;
    for (size_t i = 0; i < meta_->targets_.size(); ++i) {
      if (meta_->targets_[i].remaining_space_ >= size) {
        target_info = meta_->targets_[i];
        found = true;
        break;
      }
    }
    if (!found) {
      task->return_code_ = 7;  // No space available
      return;
    }
  }

  // Allocate blocks via bdev (outside locks)
  // Direct NewTask/Send — client Async* methods are host-only
  auto alloc_task_ptr = CHI_IPC->NewTask<chimaera::bdev::AllocateBlocksTask>(
      chi::CreateTaskId(), target_info.bdev_client_.pool_id_,
      target_info.target_query_, size);
  auto alloc_future = CHI_IPC->Send(alloc_task_ptr);
  // WaitGpu lives on chi::gpu::Future (device-pass return type of
  // CHI_IPC->Send). Gate against HSHM_IS_DEVICE_PASS so the SYCL host
  // pass — which sees chi::Future (no WaitGpu) — skips this line.
#if HSHM_IS_DEVICE_PASS
  alloc_future.WaitGpu();
#endif

  if (alloc_task_ptr->blocks_.empty()) {
    task->return_code_ = 8;  // Allocation failed
    return;
  }

  // Copy data to allocated blocks via bdev
  chi::PoolQuery warp_query = target_info.target_query_;
  warp_query.SetParallelism(32);
  auto write_task_ptr = CHI_IPC->NewTask<chimaera::bdev::WriteTask>(
      chi::CreateTaskId(), target_info.bdev_client_.pool_id_,
      warp_query, alloc_task_ptr->blocks_, task->blob_data_, size);
  auto write_future = CHI_IPC->Send(write_task_ptr);
#if HSHM_IS_DEVICE_PASS
  write_future.WaitGpu();
#endif

  if (write_task_ptr->return_code_ != 0) {
    task->return_code_ = 9;  // Write failed
    return;
  }

  // Re-lock blob key and update entry with blocks
  {
    meta_->blob_map_.lock_key(ck);

    // Re-find entry after completion (pointer may be stale after rehash)
    entry = meta_->blob_map_.find_locked(ck);
    if (entry == nullptr) {
      meta_->blob_map_.unlock_key(ck);
      task->return_code_ = 11;
      return;
    }

    // Create BlobBlock structs from allocated blocks
    entry->blocks_.clear();
    for (size_t i = 0; i < alloc_task_ptr->blocks_.size(); ++i) {
      BlobBlock blk(target_info.bdev_client_, target_info.target_query_,
                    alloc_task_ptr->blocks_[i].offset_, alloc_task_ptr->blocks_[i].size_);
      entry->blocks_.push_back(blk);
    }

    entry->size_ = size;
    entry->score_ = blob_score;
    entry->last_modified_ = GetCurrentTimeNs();
    if (is_new_blob) {
      entry->last_read_ = 0;
    }
    meta_->blob_map_.unlock_key(ck);
  }

  // Update tag total_size_ (fine-grained lock inside map)
  {
    chi::u64 tag_key = tag_id.ToU64();
    meta_->tag_id_map_.lock_key(tag_key);
    TagInfo *tag = meta_->tag_id_map_.find_locked(tag_key);
    if (tag != nullptr) {
      if (is_new_blob) {
        tag->total_size_ += size;
      } else {
        tag->total_size_ = tag->total_size_ - old_blob_size + size;
      }
    }
    meta_->tag_id_map_.unlock_key(tag_key);
  }

  // Update target remaining_space_ (write-once targets, no lock needed)
  {
    for (size_t i = 0; i < meta_->targets_.size(); ++i) {
      if (meta_->targets_[i].target_name_ == target_info.target_name_) {
        chi::u64 space_used = is_new_blob ? size : size - old_blob_size;
        meta_->targets_[i].remaining_space_ =
            (space_used <= meta_->targets_[i].remaining_space_)
                ? meta_->targets_[i].remaining_space_ - space_used
                : 0;
        break;
      }
    }
  }

  task->return_code_ = 0;
}

//==============================================================================
// GetBlob
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::GetBlob(
    hipc::FullPtr<GetBlobTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;
  EnsureMetaInit(g_ipc_manager_ptr);
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  chi::u64 offset = task->offset_;
  chi::u64 size = task->size_;

  if (size == 0 || blob_name_len == 0) { task->return_code_ = 1; return; }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);

  // Resolve output buffer
  auto out_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (out_ptr.IsNull()) { task->return_code_ = 1; return; }

  // Get blob entry and blocks (copy blocks locally)
  chi::priv::shared_vector<BlobBlock> blocks(CHI_PRIV_SHARED_ALLOC);
  chi::u64 blob_size = 0;
  {
    meta_->blob_map_.lock_key(ck);
    GpuBlobEntry *entry = meta_->blob_map_.find_locked(ck);
    if (entry == nullptr) {
      meta_->blob_map_.unlock_key(ck);
      task->return_code_ = 1;
      return;
    }

    blocks = entry->blocks_;
    blob_size = entry->size_;
    entry->last_read_ = GetCurrentTimeNs();
    meta_->blob_map_.unlock_key(ck);
  }

  // For GPU simplicity, assume single-block blobs (typical case)
  if (blocks.empty()) {
    task->return_code_ = 2;  // No blocks allocated
    return;
  }

  if (blocks.size() != 1) {
    // Multi-block read not yet implemented on GPU
    task->return_code_ = 3;
    return;
  }

  // Build a vector of blocks for the read operation
  chi::priv::vector<chimaera::bdev::Block> read_blocks(CHI_PRIV_ALLOC);
  chimaera::bdev::Block blk(blocks[0].target_offset_, blocks[0].size_, 0);
  read_blocks.push_back(blk);

  // Read the single block via bdev (full-warp parallelism)
  chi::PoolQuery warp_query = blocks[0].target_query_;
  warp_query.SetParallelism(32);
  // Direct NewTask/Send — client Async* methods are host-only
  auto read_task_ptr = CHI_IPC->NewTask<chimaera::bdev::ReadTask>(
      chi::CreateTaskId(), blocks[0].bdev_client_.pool_id_,
      warp_query, read_blocks, task->blob_data_, size);
  auto read_future = CHI_IPC->Send(read_task_ptr);
#if HSHM_IS_DEVICE_PASS
  read_future.WaitGpu();
#endif

  if (read_task_ptr->return_code_ != 0) {
    task->return_code_ = 4;  // Read failed
    return;
  }

  // Validate we read the expected amount
  chi::u64 can_read = (offset < blob_size) ? (blob_size - offset) : 0;
  chi::u64 expected_read = (can_read < size) ? can_read : size;

  task->return_code_ = (read_task_ptr->bytes_read_ == expected_read) ? 0 : 1;
}

//==============================================================================
// ReorganizeBlob
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::ReorganizeBlob(
    hipc::FullPtr<ReorganizeBlobTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;
  EnsureMetaInit(g_ipc_manager_ptr);
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  float new_score = task->new_score_;

  if (blob_name_len == 0 || new_score < 0.0f || new_score > 1.0f) {
    task->return_code_ = 1;
    return;
  }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  meta_->blob_map_.lock_key(ck);

  GpuBlobEntry *entry = meta_->blob_map_.find_locked(ck);
  if (entry == nullptr) {
    meta_->blob_map_.unlock_key(ck);
    task->return_code_ = 3;
    return;
  }

  entry->score_ = new_score;
  meta_->blob_map_.unlock_key(ck);
  task->return_code_ = 0;
}

//==============================================================================
// DelBlob
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::DelBlob(
    hipc::FullPtr<DelBlobTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)rctx;
  if (!chi::gpu::IpcManager::IsWarpScheduler()) return;
  EnsureMetaInit(g_ipc_manager_ptr);
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());

  if (blob_name_len == 0) { task->return_code_ = 1; return; }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);

  chi::u64 blob_size = 0;
  {
    meta_->blob_map_.lock_key(ck);
    GpuBlobEntry *entry = meta_->blob_map_.find_locked(ck);
    if (entry == nullptr) {
      meta_->blob_map_.unlock_key(ck);
      task->return_code_ = 1;
      return;
    }

    blob_size = entry->size_;
    meta_->blob_map_.erase_locked(ck);
    meta_->blob_map_.unlock_key(ck);
  }

  // Update tag total_size_ (fine-grained lock inside map)
  {
    chi::u64 tag_key = tag_id.ToU64();
    meta_->tag_id_map_.lock_key(tag_key);
    TagInfo *tag = meta_->tag_id_map_.find_locked(tag_key);
    if (tag != nullptr) {
      tag->total_size_ =
          (blob_size <= tag->total_size_) ? tag->total_size_ - blob_size : 0;
    }
    meta_->tag_id_map_.unlock_key(tag_key);
  }

  task->return_code_ = 0;
}

//==============================================================================
// Remaining stubs
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::GetBlobScore(
    hipc::FullPtr<GetBlobScoreTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetBlobSize(
    hipc::FullPtr<GetBlobSizeTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetBlobInfo(
    hipc::FullPtr<GetBlobInfoTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::PollTelemetryLog(
    hipc::FullPtr<PollTelemetryLogTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::TagQuery(
    hipc::FullPtr<TagQueryTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::BlobQuery(
    hipc::FullPtr<BlobQueryTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetTargetInfo(
    hipc::FullPtr<GetTargetInfoTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::FlushMetadata(
    hipc::FullPtr<FlushMetadataTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::FlushData(
    hipc::FullPtr<FlushDataTask> task, chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext so CHI_IPC under
  // SYCL resolves to the kernel-scope IpcManager pointer.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  (void)task; (void)rctx;
}

// LocalAllocLoadTask / LocalSaveTask / etc. are now generated by the autogen
// (core_gpu_lib_exec.h) as static functions with function pointer dispatch.
// No hand-written out-of-class definitions needed.

}  // namespace wrp_cte::core
