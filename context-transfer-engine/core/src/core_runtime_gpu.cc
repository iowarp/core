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
 * ThreadAllocator which provides per-block BuddyAllocator partitions,
 * eliminating cross-block allocator contention (CUDA Error 700).
 *
 * Note: core_tasks.h is included here (not in the header) to keep GPU
 * compilation isolated from CPU-only task constructors that use HSHM_MALLOC.
 */

#include "wrp_cte/core/core_gpu_runtime.h"
#include "wrp_cte/core/core_tasks.h"
#include <hermes_shm/data_structures/priv/vector.h>
#include <hermes_shm/data_structures/priv/string.h>
#include <hermes_shm/thread/lock/mutex.h>

namespace wrp_cte::core {

/** Number of blob lock buckets — must be power of 2 */
static constexpr int kNumBlobLocks = 8;
static constexpr int kBlobLockMask = kNumBlobLocks - 1;

/** FNV-1a hash for chi::priv::string */
HSHM_GPU_FUN static chi::u32 HashString(const chi::priv::string &s) {
  chi::u32 hash = 2166136261u;
  const char *data = s.data();
  size_t len = s.size();
  for (size_t i = 0; i < len; ++i) {
    hash ^= static_cast<chi::u32>(data[i]);
    hash *= 16777619u;
  }
  return hash;
}

/**
 * GPU-side blob entry using chi::priv data structures.
 */
struct GpuBlobEntry {
  chi::priv::string key_;       // compound key "major.minor.blob_name"
  chi::u64 data_ptr_;           // GPU pointer to blob data
  chi::u64 size_;               // blob size in bytes
  float score_;
  Timestamp last_modified_;
  Timestamp last_read_;

  HSHM_CROSS_FUN GpuBlobEntry()
      : key_(CHI_PRIV_ALLOC),
        data_ptr_(0), size_(0), score_(0.0f),
        last_modified_(0), last_read_(0) {}

  HSHM_CROSS_FUN GpuBlobEntry(const GpuBlobEntry &other)
      : key_(other.key_),
        data_ptr_(other.data_ptr_), size_(other.size_),
        score_(other.score_),
        last_modified_(other.last_modified_),
        last_read_(other.last_read_) {}

  HSHM_CROSS_FUN GpuBlobEntry &operator=(const GpuBlobEntry &other) {
    if (this != &other) {
      key_ = other.key_;
      data_ptr_ = other.data_ptr_;
      size_ = other.size_;
      score_ = other.score_;
      last_modified_ = other.last_modified_;
      last_read_ = other.last_read_;
    }
    return *this;
  }
};

/**
 * Per-bucket blob data using chi::priv::vector.
 */
struct BlobBucketData {
  chi::priv::vector<GpuBlobEntry> entries_;

  HSHM_CROSS_FUN BlobBucketData() : entries_(CHI_PRIV_ALLOC) {}
};

/**
 * GPU-resident metadata store for CTE Core GpuRuntime.
 * Uses chi::priv data structures backed by ThreadAllocator.
 *
 * Locking strategy:
 *   tag_lock_  — single lock for the tag store
 *   blob_locks_[kNumBlobLocks] — hash-partitioned locks for the blob store
 */
struct GpuMetadata {
  hshm::Mutex tag_lock_;
  hshm::Mutex blob_locks_[kNumBlobLocks];
  BlobBucketData blob_data_[kNumBlobLocks];
  chi::priv::vector<TagInfo> tags_;

  HSHM_GPU_FUN GpuMetadata() : tags_(CHI_PRIV_ALLOC) {
    tag_lock_.Init();
    for (int i = 0; i < kNumBlobLocks; ++i) {
      blob_locks_[i].Init();
    }
  }

  /** Get the bucket index for a given compound key */
  HSHM_GPU_FUN static int BucketIdx(const chi::priv::string &key) {
    return HashString(key) & kBlobLockMask;
  }

  /** Get the bucket lock for a given compound key */
  HSHM_GPU_FUN hshm::Mutex &BlobLock(const chi::priv::string &key) {
    return blob_locks_[BucketIdx(key)];
  }

  /** Get the bucket data for a given compound key */
  HSHM_GPU_FUN BlobBucketData &BlobData(const chi::priv::string &key) {
    return blob_data_[BucketIdx(key)];
  }

  /** Lock ALL blob buckets (for operations that scan the entire blob store) */
  HSHM_GPU_FUN void LockAllBuckets() {
    for (int i = 0; i < kNumBlobLocks; ++i) {
      blob_locks_[i].Lock(0);
    }
  }

  HSHM_GPU_FUN void UnlockAllBuckets() {
    for (int i = kNumBlobLocks - 1; i >= 0; --i) {
      blob_locks_[i].Unlock();
    }
  }
};

//==============================================================================
// Helper: build compound key "major.minor.blob_name"
//==============================================================================

HSHM_GPU_FUN static chi::priv::string MakeCompoundKey(
    const TagId &tag_id, const char *blob_name, int blob_name_len) {
  // Build into a stack buffer then construct string
  char buf[128];
  int pos = 0;

  // Convert major to string
  chi::u32 major = tag_id.major_;
  if (major == 0) {
    buf[pos++] = '0';
  } else {
    char tmp[16]; int tlen = 0;
    while (major > 0) { tmp[tlen++] = '0' + (major % 10); major /= 10; }
    for (int i = tlen - 1; i >= 0; --i) buf[pos++] = tmp[i];
  }
  buf[pos++] = '.';

  // Convert minor to string
  chi::u32 minor = tag_id.minor_;
  if (minor == 0) {
    buf[pos++] = '0';
  } else {
    char tmp[16]; int tlen = 0;
    while (minor > 0) { tmp[tlen++] = '0' + (minor % 10); minor /= 10; }
    for (int i = tlen - 1; i >= 0; --i) buf[pos++] = tmp[i];
  }
  buf[pos++] = '.';

  // Append blob_name
  for (int i = 0; i < blob_name_len && pos < 127; ++i) {
    buf[pos++] = blob_name[i];
  }
  buf[pos] = '\0';

  return chi::priv::string(CHI_PRIV_ALLOC, buf);
}

//==============================================================================
// Stub methods (no-ops on GPU)
//==============================================================================

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

//==============================================================================
// EnsureMetaInit — double-checked locking with threadfence
//==============================================================================

HSHM_GPU_FUN void GpuRuntime::EnsureMetaInit() {
  GpuMetadata *m = *reinterpret_cast<GpuMetadata *volatile *>(&meta_);
  if (m != nullptr) return;
  hshm::ScopedMutex guard(init_lock_, 0);
  m = *reinterpret_cast<GpuMetadata *volatile *>(&meta_);
  if (m != nullptr) return;
  CHI_PRIV_ALLOC_T *alloc = CHI_PRIV_ALLOC;
  hipc::FullPtr<GpuMetadata> ptr = alloc->template AllocateObjs<GpuMetadata>(1);
  new (ptr.ptr_) GpuMetadata();
  __threadfence();
  meta_ = ptr.ptr_;
  __threadfence();
}

//==============================================================================
// Tag operations
//==============================================================================

HSHM_GPU_FUN TagInfo *GpuRuntime::FindTagById(const TagId &tag_id) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->tags_.size(); ++i) {
    if (meta_->tags_[i].tag_id_ == tag_id) {
      return &meta_->tags_[i];
    }
  }
  return nullptr;
}

HSHM_GPU_FUN TagId *GpuRuntime::FindTagIdByName(const chi::priv::string &name) {
  if (!meta_) return nullptr;
  for (size_t i = 0; i < meta_->tags_.size(); ++i) {
    if (meta_->tags_[i].tag_name_ == name) {
      return &meta_->tags_[i].tag_id_;
    }
  }
  return nullptr;
}

HSHM_GPU_FUN TagInfo *GpuRuntime::UpsertTag(const chi::priv::string &tag_name,
                                             const TagId &tag_id) {
  if (!meta_) return nullptr;
  // Check if tag exists by name
  for (size_t i = 0; i < meta_->tags_.size(); ++i) {
    if (meta_->tags_[i].tag_name_ == tag_name) {
      return &meta_->tags_[i];
    }
  }
  // Insert new
  TagInfo info(tag_name, tag_id);
  meta_->tags_.push_back(info);
  return &meta_->tags_.back();
}

//==============================================================================
// Blob bucket operations
//==============================================================================

/** Find blob entry by compound key in bucket. Returns index or -1. */
HSHM_GPU_FUN static int FindGpuBlob(BlobBucketData &bdata,
                                      const chi::priv::string &key) {
  for (size_t i = 0; i < bdata.entries_.size(); ++i) {
    if (bdata.entries_[i].key_ == key) return static_cast<int>(i);
  }
  return -1;
}

HSHM_GPU_FUN chi::priv::string GpuRuntime::MakeBlobKey(const TagId &tag_id,
                                                         const chi::priv::string &blob_name) {
  return MakeCompoundKey(tag_id, blob_name.data(),
                         static_cast<int>(blob_name.size()));
}

//==============================================================================
// GetOrCreateTag
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParams>> task,
    chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  hshm::ScopedMutex guard(meta_->tag_lock_, 0);

  chi::priv::string name(CHI_PRIV_ALLOC, task->tag_name_.data());
  TagId preferred_id = task->tag_id_;

  // Look up existing tag by name
  TagId *existing = FindTagIdByName(name);
  if (existing != nullptr) {
    task->tag_id_ = *existing;
    task->return_code_ = 0;
    co_return;
  }

  // Assign new ID
  TagId tag_id;
  if (preferred_id.major_ != 0 || preferred_id.minor_ != 0) {
    tag_id = preferred_id;
  } else {
    tag_id.major_ = container_id_;
    tag_id.minor_ = atomicAdd(&next_tag_minor_, 1u) + 1;
  }

  // Insert
  UpsertTag(name, tag_id);

  task->tag_id_ = tag_id;
  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// GetTagSize
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetTagSize(
    hipc::FullPtr<GetTagSizeTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// DelTag
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::DelTag(
    hipc::FullPtr<DelTagTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
  meta_->LockAllBuckets();

  TagId tag_id = task->tag_id_;

  // Resolve tag_id from name if needed
  if (tag_id.IsNull()) {
    chi::priv::string name(CHI_PRIV_ALLOC, task->tag_name_.data());
    if (name.size() == 0) {
      meta_->UnlockAllBuckets();
      task->return_code_ = 1;
      co_return;
    }
    TagId *found = FindTagIdByName(name);
    if (found == nullptr) {
      meta_->UnlockAllBuckets();
      task->return_code_ = 1;
      co_return;
    }
    tag_id = *found;
    task->tag_id_ = tag_id;
  }

  // Build prefix "major.minor." for matching
  chi::priv::string prefix = MakeCompoundKey(tag_id, "", 0);

  // Scan all buckets and erase matching blobs
  for (int b = 0; b < kNumBlobLocks; ++b) {
    BlobBucketData &bdata = meta_->blob_data_[b];
    size_t i = 0;
    while (i < bdata.entries_.size()) {
      const chi::priv::string &key = bdata.entries_[i].key_;
      // Check if key starts with prefix
      bool match = (key.size() >= prefix.size());
      if (match) {
        const char *kd = key.data();
        const char *pd = prefix.data();
        for (size_t c = 0; c < prefix.size() && match; ++c) {
          match = (kd[c] == pd[c]);
        }
      }
      if (match) {
        // Swap with last and decrement
        size_t last = bdata.entries_.size() - 1;
        if (i != last) {
          bdata.entries_[i] = bdata.entries_[last];
        }
        bdata.entries_.pop_back();
      } else {
        ++i;
      }
    }
  }

  // Erase tag from tag store
  TagInfo *tag_info = FindTagById(tag_id);
  if (tag_info != nullptr) {
    // Find index of tag_info in tags_ vector
    size_t tag_idx = tag_info - meta_->tags_.data();
    size_t last = meta_->tags_.size() - 1;
    if (tag_idx != last) {
      meta_->tags_[tag_idx] = meta_->tags_[last];
    }
    meta_->tags_.pop_back();
  }

  meta_->UnlockAllBuckets();
  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// GetContainedBlobs
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetContainedBlobs(
    hipc::FullPtr<GetContainedBlobsTask> task, chi::gpu::RunContext &rctx) {
  (void)task; (void)rctx;
  co_return;
}

//==============================================================================
// PutBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::PutBlob(
    hipc::FullPtr<PutBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  chi::u64 size = task->size_;
  float blob_score = task->score_;

  // Validate inputs
  if (size == 0) { task->return_code_ = 2; co_return; }
  if (task->blob_data_.IsNull()) { task->return_code_ = 3; co_return; }
  if (blob_name_len == 0) { task->return_code_ = 4; co_return; }
  if (blob_score < 0.0f) blob_score = 1.0f;
  if (blob_score > 1.0f) { task->return_code_ = 5; co_return; }

  // Resolve blob data pointer
  auto data_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (data_ptr.IsNull()) { task->return_code_ = 6; co_return; }

  // Build compound key and lock the bucket
  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->BlobLock(ck), 0);
  BlobBucketData &bdata = meta_->BlobData(ck);

  int idx = FindGpuBlob(bdata, ck);

  if (idx < 0) {
    // Create new blob entry
    GpuBlobEntry entry;
    entry.key_ = ck;
    entry.data_ptr_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
    entry.size_ = size;
    entry.score_ = blob_score;
    entry.last_modified_ = GetCurrentTimeNs();
    entry.last_read_ = 0;
    bdata.entries_.push_back(entry);

    // Update tag total_size_
    hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
    TagInfo *tag = FindTagById(tag_id);
    if (tag != nullptr) tag->total_size_ += size;
  } else {
    // Update existing blob
    chi::u64 old_size = bdata.entries_[idx].size_;
    bdata.entries_[idx].data_ptr_ = reinterpret_cast<chi::u64>(data_ptr.ptr_);
    bdata.entries_[idx].size_ = size;
    bdata.entries_[idx].score_ = blob_score;
    bdata.entries_[idx].last_modified_ = GetCurrentTimeNs();

    // Update tag total_size_
    hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
    TagInfo *tag = FindTagById(tag_id);
    if (tag != nullptr) {
      tag->total_size_ = tag->total_size_ - old_size + size;
    }
  }

  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// GetBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::GetBlob(
    hipc::FullPtr<GetBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  chi::u64 offset = task->offset_;
  chi::u64 size = task->size_;

  if (size == 0 || blob_name_len == 0) { task->return_code_ = 1; co_return; }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->BlobLock(ck), 0);
  BlobBucketData &bdata = meta_->BlobData(ck);

  int idx = FindGpuBlob(bdata, ck);
  if (idx < 0) { task->return_code_ = 1; co_return; }

  // Resolve output buffer
  auto out_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  if (out_ptr.IsNull()) { task->return_code_ = 1; co_return; }

  // Copy data from blob to output buffer
  GpuBlobEntry &entry = bdata.entries_[idx];
  char *src = reinterpret_cast<char *>(entry.data_ptr_) + offset;
  char *dst = reinterpret_cast<char *>(out_ptr.ptr_);
  chi::u64 can_read = (offset < entry.size_) ? (entry.size_ - offset) : 0;
  chi::u64 to_read = (can_read < size) ? can_read : size;
  memcpy(dst, src, to_read);

  entry.last_read_ = GetCurrentTimeNs();
  task->return_code_ = (to_read == size) ? 0 : 1;
  co_return;
}

//==============================================================================
// ReorganizeBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::ReorganizeBlob(
    hipc::FullPtr<ReorganizeBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());
  float new_score = task->new_score_;

  if (blob_name_len == 0 || new_score < 0.0f || new_score > 1.0f) {
    task->return_code_ = 1;
    co_return;
  }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->BlobLock(ck), 0);
  BlobBucketData &bdata = meta_->BlobData(ck);

  int idx = FindGpuBlob(bdata, ck);
  if (idx < 0) { task->return_code_ = 3; co_return; }

  bdata.entries_[idx].score_ = new_score;
  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// DelBlob
//==============================================================================

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::DelBlob(
    hipc::FullPtr<DelBlobTask> task, chi::gpu::RunContext &rctx) {
  (void)rctx;
  EnsureMetaInit();
  TagId tag_id = task->tag_id_;
  const char *blob_name = task->blob_name_.data();
  int blob_name_len = static_cast<int>(task->blob_name_.size());

  if (blob_name_len == 0) { task->return_code_ = 1; co_return; }

  chi::priv::string ck = MakeCompoundKey(tag_id, blob_name, blob_name_len);
  hshm::ScopedMutex blob_guard(meta_->BlobLock(ck), 0);
  BlobBucketData &bdata = meta_->BlobData(ck);

  int idx = FindGpuBlob(bdata, ck);
  if (idx < 0) { task->return_code_ = 1; co_return; }

  chi::u64 blob_size = bdata.entries_[idx].size_;

  // Swap with last and pop
  size_t last = bdata.entries_.size() - 1;
  if (static_cast<size_t>(idx) != last) {
    bdata.entries_[idx] = bdata.entries_[last];
  }
  bdata.entries_.pop_back();

  // Update tag total_size_
  hshm::ScopedMutex tag_guard(meta_->tag_lock_, 0);
  TagInfo *tag = FindTagById(tag_id);
  if (tag != nullptr) {
    tag->total_size_ =
        (blob_size <= tag->total_size_) ? tag->total_size_ - blob_size : 0;
  }

  task->return_code_ = 0;
  co_return;
}

//==============================================================================
// Remaining stubs
//==============================================================================

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
