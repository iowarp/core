/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
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

  int cat = FindSizeCategory(req);
  chi::u64 alloc_size;
  chi::u32 block_type;
  if (cat >= 0) {
    alloc_size = kGpuBlockSizes[cat];
    block_type = static_cast<chi::u32>(cat);
  } else {
    chi::u32 align = (alignment_ > 0) ? alignment_ : 4096;
    alloc_size = ((req + (chi::u64)align - 1) / (chi::u64)align) * (chi::u64)align;
    block_type = static_cast<chi::u32>(GpuBlockSizeCategory::kNumCategories);
  }

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  if (warp_id >= num_warps_) warp_id = 0;

  Block blk;
  bool found = false;
  if (cat >= 0 && num_warps_ > 0) {
    found = warp_caches_[warp_id].lists_[cat].Pop(blk);
  }

  if (!found) {
    chi::u64 old_pos = (chi::u64)atomicAdd(
        (unsigned long long *)&gpu_heap_,
        (unsigned long long)alloc_size);

    if (old_pos + alloc_size > total_size_) {
      atomicAdd((unsigned long long *)&gpu_heap_,
                (unsigned long long)(-(long long)alloc_size));
      task->return_code_ = 1;
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
// FreeBlocks
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
      warp_caches_[warp_id].lists_[cat].Push(blk);
    }
  }

  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// Write — lane-0 only for now
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Write(hipc::FullPtr<WriteTask> task,
                                     chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)rctx; co_return; }
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);
  static constexpr chi::u32 kNoop   = static_cast<chi::u32>(BdevType::kNoop);

  if (bdev_type_ == kNoop) {
    task->bytes_written_ = task->length_;
    task->return_code_ = 0;
    co_return;
  }
  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    task->return_code_ = 1;
    co_return;
  }

  char *dst_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *src = data_ptr.ptr_;

  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    char *block_dst = dst_base + block.offset_;
    const char *block_src = src + data_off;

    bool aligned16 = ((reinterpret_cast<uintptr_t>(block_dst) |
                        reinterpret_cast<uintptr_t>(block_src)) & 15) == 0;
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
    __threadfence_system();
    data_off += copy_size;

    if (i + 1 < num_blocks) {
      co_await chi::gpu::yield(2);
    }
  }

  task->bytes_written_ = data_off;
  task->return_code_ = 0;
  co_return;
}

// ---------------------------------------------------------------------------
// Read — memset(lane) stripe test
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Read(hipc::FullPtr<ReadTask> task,
                                    chi::gpu::RunContext &rctx) {
  chi::u32 lane = threadIdx.x % 32;
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);
  static constexpr chi::u32 kNoop   = static_cast<chi::u32>(BdevType::kNoop);

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

  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *dst = data_ptr.ptr_;

  chi::u32 num_lanes = rctx.parallelism_;
  if (num_lanes == 0) num_lanes = 1;
  if (lane >= num_lanes) co_return;

  // Instead of copying from bdev, just memset each stripe with lane id
  // to verify that per-lane writes reach distinct memory regions.
  chi::u64 total = task->length_;
  chi::u64 stripe = total / num_lanes;
  chi::u64 my_start = lane * stripe;
  chi::u64 my_end = (lane == num_lanes - 1) ? total : (lane + 1) * stripe;

  for (chi::u64 b = my_start; b < my_end; ++b) {
    dst[b] = static_cast<char>(lane);
  }
  __threadfence_system();

  if (lane == 0) {
    task->bytes_read_ = total;
    task->return_code_ = 0;
  }
  co_return;
}

}  // namespace chimaera::bdev
