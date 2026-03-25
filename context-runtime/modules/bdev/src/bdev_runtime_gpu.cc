/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * GPU-side bdev container implementation.
 *
 * Compiled as CUDA device code (picked up by chimaera_cxx_gpu via the
 * modules/<star>_gpu.cc glob in src/CMakeLists.txt).
 *
 * Implements Update, AllocateBlocks, FreeBlocks, Write, Read using
 * device-resident atomics and memcpy.
 */

#include "chimaera/bdev/bdev_gpu_runtime.h"
#include "chimaera/singletons.h"

namespace chimaera::bdev {

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Update(hipc::FullPtr<UpdateTask> task,
                                      chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)rctx; co_return; }
  hbm_ptr_    = task->hbm_ptr_;
  pinned_ptr_ = task->pinned_ptr_;
  hbm_size_   = task->hbm_size_;
  pinned_size_ = task->pinned_size_;
  total_size_  = task->total_size_;
  bdev_type_   = task->bdev_type_;
  alignment_   = (task->alignment_ > 0) ? task->alignment_ : 4096;
  // Reset the bump allocator and allocate per-warp free lists
  gpu_heap_ = 0;
  num_warps_ = chi::IpcManager::GetNumWarps();
  if (num_warps_ == 0) num_warps_ = 1;
  warp_caches_.clear();
  warp_caches_.resize(num_warps_);
  for (chi::u32 w = 0; w < num_warps_; ++w) {
    for (chi::u32 c = 0; c < GpuWarpBlockCache::kNumCategories; ++c) {
      warp_caches_[w].lists_[c].count_ = 0;
    }
  }
  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// AllocateBlocks
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::AllocateBlocks(
    hipc::FullPtr<AllocateBlocksTask> task,
    chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)rctx; co_return; }
  chi::u64 req = task->size_;
  if (req == 0 || total_size_ == 0) {
    task->return_code_ = 0;
    (void)rctx;
    co_return;
  }

  // Find the smallest size category that fits the request
  int cat = FindSizeCategory(req);
  chi::u64 alloc_size;
  chi::u32 block_type;
  if (cat >= 0) {
    alloc_size = kGpuBlockSizes[cat];
    block_type = static_cast<chi::u32>(cat);
  } else {
    // Larger than any cached size — align to alignment_ and use heap directly
    chi::u32 align = (alignment_ > 0) ? alignment_ : 4096;
    alloc_size = ((req + (chi::u64)align - 1) / (chi::u64)align) * (chi::u64)align;
    block_type = static_cast<chi::u32>(GpuBlockSizeCategory::kNumCategories);
  }

  // Try the calling warp's free list first
  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  if (warp_id >= num_warps_) warp_id = 0;

  Block blk;
  bool found = false;
  if (cat >= 0 && num_warps_ > 0) {
    found = warp_caches_[warp_id].lists_[cat].Pop(blk);
  }

  if (!found) {
    // Fall back to bump allocator
    chi::u64 old_pos = (chi::u64)atomicAdd(
        (unsigned long long *)&gpu_heap_,
        (unsigned long long)alloc_size);

    if (old_pos + alloc_size > total_size_) {
      // Rollback
      atomicAdd((unsigned long long *)&gpu_heap_,
                (unsigned long long)(-(long long)alloc_size));
      task->return_code_ = 1;  // out of space
      (void)rctx;
      co_return;
    }

    blk.offset_ = old_pos;
    blk.size_ = alloc_size;
    blk.block_type_ = block_type;
  }

  task->blocks_.push_back(blk);
  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// FreeBlocks — return blocks to the calling warp's free list
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::FreeBlocks(hipc::FullPtr<FreeBlocksTask> task,
                                           chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)task; (void)rctx; co_return; }

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  if (warp_id >= num_warps_) warp_id = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    Block &blk = task->blocks_[i];
    int cat = static_cast<int>(blk.block_type_);
    if (cat >= 0 && cat < static_cast<int>(GpuBlockSizeCategory::kNumCategories)) {
      // Push to this warp's free list; if full, block is leaked (reclaimed on destroy)
      warp_caches_[warp_id].lists_[cat].Push(blk);
    }
    // Oversized blocks (block_type >= kNumCategories) are not cached
  }

  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Write(hipc::FullPtr<WriteTask> task,
                                     chi::gpu::RunContext &rctx) {
  chi::u32 lane = chi::IpcManager::GetLaneId();
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);
  static constexpr chi::u32 kNoop   = static_cast<chi::u32>(BdevType::kNoop);

  // Noop: immediate success, no data movement
  if (bdev_type_ == kNoop) {
    if (lane == 0) {
      task->bytes_written_ = task->length_;
      task->return_code_ = 0;
    }
    co_return;
  }

  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    if (lane == 0) task->return_code_ = 1;
    co_return;
  }

  char *dst_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *src = data_ptr.ptr_;

  // Copy across all blocks with yield for asynchronicity.
  // In GPU coroutines, only lane 0 (warp scheduler) is active.
  // __activemask() may return stale values in coroutine context.
  // Always use lane-0 sequential copy for correctness.
  bool warp_wide = false;
  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  long long t_start = clock64();
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    char *dst = dst_base + block.offset_;
    const char *block_src = src + data_off;

    bool aligned16 = ((reinterpret_cast<uintptr_t>(dst) |
                        reinterpret_cast<uintptr_t>(block_src)) & 15) == 0;
    if (warp_wide) {
      // Warp-wide coalesced copy (all 32 lanes participate)
      if (aligned16) {
        chi::u64 vec_elems = copy_size / sizeof(uint4);
        const uint4 *src4 = reinterpret_cast<const uint4 *>(block_src);
        uint4 *dst4 = reinterpret_cast<uint4 *>(dst);
        for (chi::u64 idx = lane; idx < vec_elems; idx += 32) {
          dst4[idx] = src4[idx];
        }
        chi::u64 tail_start = vec_elems * sizeof(uint4);
        for (chi::u64 b = tail_start + lane; b < copy_size; b += 32) {
          dst[b] = block_src[b];
        }
      } else {
        chi::u64 vec_elems = copy_size / sizeof(chi::u32);
        const chi::u32 *src4 = reinterpret_cast<const chi::u32 *>(block_src);
        chi::u32 *dst4 = reinterpret_cast<chi::u32 *>(dst);
        for (chi::u64 idx = lane; idx < vec_elems; idx += 32) {
          dst4[idx] = src4[idx];
        }
        chi::u64 tail_start = vec_elems * sizeof(chi::u32);
        for (chi::u64 b = tail_start + lane; b < copy_size; b += 32) {
          dst[b] = block_src[b];
        }
      }
    } else {
      // Lane-0 sequential copy (subtask path, only 1 lane active)
      if (lane == 0) {
        if (aligned16) {
          chi::u64 vec_elems = copy_size / sizeof(uint4);
          const uint4 *src4 = reinterpret_cast<const uint4 *>(block_src);
          uint4 *dst4 = reinterpret_cast<uint4 *>(dst);
          for (chi::u64 idx = 0; idx < vec_elems; ++idx) {
            dst4[idx] = src4[idx];
          }
          chi::u64 tail_start = vec_elems * sizeof(uint4);
          for (chi::u64 b = tail_start; b < copy_size; ++b) {
            dst[b] = block_src[b];
          }
        } else {
          for (chi::u64 b = 0; b < copy_size; ++b) {
            dst[b] = block_src[b];
          }
        }
      }
    }
    data_off += copy_size;

    // Yield between blocks to let other coroutines run
    if (i + 1 < num_blocks) {
      co_await chi::gpu::yield(2);
    }
  }
  long long t_end = clock64();

  if (lane == 0) {
    task->bytes_written_ = data_off;
    task->return_code_ = 0;
  }
  co_return;
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Read(hipc::FullPtr<ReadTask> task,
                                    chi::gpu::RunContext &rctx) {
  chi::u32 lane = chi::IpcManager::GetLaneId();
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);
  static constexpr chi::u32 kNoop   = static_cast<chi::u32>(BdevType::kNoop);

  // Noop: immediate success, no data movement
  if (bdev_type_ == kNoop) {
    if (lane == 0) {
      task->bytes_read_ = task->length_;
      task->return_code_ = 0;
    }
    co_return;
  }

  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    if (lane == 0) task->return_code_ = 1;
    co_return;
  }

  char *src_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *dst = data_ptr.ptr_;

  // Copy across all blocks with yield for asynchronicity.
  // Use warp-wide coalesced copy when all 32 lanes are active (parallelism > 1),
  // otherwise fall back to lane-0 sequential copy.
  // In GPU coroutines, only lane 0 (warp scheduler) is active.
  // __activemask() may return stale values in coroutine context.
  // Always use lane-0 sequential copy for correctness.
  bool warp_wide = false;
  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  long long t_start = clock64();
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    const char *block_src = src_base + block.offset_;
    char *block_dst = dst + data_off;

    bool aligned16 = ((reinterpret_cast<uintptr_t>(block_dst) |
                        reinterpret_cast<uintptr_t>(block_src)) & 15) == 0;
    if (warp_wide) {
      // Warp-wide coalesced copy (all 32 lanes participate)
      if (aligned16) {
        chi::u64 vec_elems = copy_size / sizeof(uint4);
        const uint4 *src4 = reinterpret_cast<const uint4 *>(block_src);
        uint4 *dst4 = reinterpret_cast<uint4 *>(block_dst);
        for (chi::u64 idx = lane; idx < vec_elems; idx += 32) {
          dst4[idx] = src4[idx];
        }
        chi::u64 tail_start = vec_elems * sizeof(uint4);
        for (chi::u64 b = tail_start + lane; b < copy_size; b += 32) {
          block_dst[b] = block_src[b];
        }
      } else {
        chi::u64 vec_elems = copy_size / sizeof(chi::u32);
        const chi::u32 *src4 = reinterpret_cast<const chi::u32 *>(block_src);
        chi::u32 *dst4 = reinterpret_cast<chi::u32 *>(block_dst);
        for (chi::u64 idx = lane; idx < vec_elems; idx += 32) {
          dst4[idx] = src4[idx];
        }
        chi::u64 tail_start = vec_elems * sizeof(chi::u32);
        for (chi::u64 b = tail_start + lane; b < copy_size; b += 32) {
          block_dst[b] = block_src[b];
        }
      }
    } else {
      // Lane-0 sequential copy (subtask path, only 1 lane active)
      if (lane == 0) {
        if (aligned16) {
          chi::u64 vec_elems = copy_size / sizeof(uint4);
          const uint4 *src4 = reinterpret_cast<const uint4 *>(block_src);
          uint4 *dst4 = reinterpret_cast<uint4 *>(block_dst);
          for (chi::u64 idx = 0; idx < vec_elems; ++idx) {
            dst4[idx] = src4[idx];
          }
          chi::u64 tail_start = vec_elems * sizeof(uint4);
          for (chi::u64 b = tail_start; b < copy_size; ++b) {
            block_dst[b] = block_src[b];
          }
        } else {
          for (chi::u64 b = 0; b < copy_size; ++b) {
            block_dst[b] = block_src[b];
          }
        }
      }
    }
    data_off += copy_size;

    // Yield between blocks to let other coroutines run
    if (i + 1 < num_blocks) {
      co_await chi::gpu::yield(2);
    }
  }
  long long t_end = clock64();

  // ALL lanes must fence their writes before signaling completion.
  // Without this, only lane 0's writes are visible to the consumer.
  __syncwarp();
  __threadfence_system();

  if (lane == 0) {
    task->bytes_read_ = data_off;
    task->return_code_ = 0;
  }
  co_return;
}

}  // namespace chimaera::bdev
