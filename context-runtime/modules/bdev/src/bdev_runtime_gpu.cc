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
  // Reset the bump allocator for the new memory region
  gpu_heap_ = 0;
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

  // Align requested size
  chi::u32 align = (alignment_ > 0) ? alignment_ : 4096;
  chi::u64 aligned = ((req + (chi::u64)align - 1) / (chi::u64)align) * (chi::u64)align;

  // Atomically bump the heap cursor
  chi::u64 old_pos = (chi::u64)atomicAdd(
      (unsigned long long *)&gpu_heap_,
      (unsigned long long)aligned);

  if (old_pos + aligned > total_size_) {
    // Rollback
    atomicAdd((unsigned long long *)&gpu_heap_,
              (unsigned long long)(-(long long)aligned));
    task->return_code_ = 1;  // out of space
    (void)rctx;
    co_return;
  }

  Block blk;
  blk.offset_     = old_pos;
  blk.size_       = aligned;
  blk.block_type_ = 0;  // GPU bump allocator has no size categories
  task->blocks_.push_back(blk);

  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// FreeBlocks — no-op for bump allocator
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::FreeBlocks(hipc::FullPtr<FreeBlocksTask> task,
                                           chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)task; (void)rctx; co_return; }
  // GPU bump allocator does not support per-block free.
  // Memory is reclaimed when the bdev pool is destroyed.
  task->return_code_ = 0;
  (void)task; (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Write(hipc::FullPtr<WriteTask> task,
                                     chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)rctx; co_return; }
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);

  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    task->return_code_ = 1;  // unsupported on GPU for other types
    (void)rctx;
    co_return;
  }

  char *dst_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);

  // Resolve data ShmPtr via IpcManager (handles both raw UVA and allocator-based)
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *src = data_ptr.ptr_;

  // Sequential copy (lane 0 only; multi-lane coroutines not yet supported)
  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    char *dst = dst_base + block.offset_;
    const char *block_src = src + data_off;
    memcpy(dst, block_src, copy_size);
    data_off += copy_size;
  }

  task->bytes_written_ = data_off;
  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

HSHM_GPU_FUN chi::gpu::TaskResume GpuRuntime::Read(hipc::FullPtr<ReadTask> task,
                                    chi::gpu::RunContext &rctx) {
  if (!chi::IpcManager::IsWarpScheduler()) { (void)rctx; co_return; }
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);

  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    task->return_code_ = 1;  // unsupported on GPU for other types
    (void)rctx;
    co_return;
  }

  char *src_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);

  // Resolve data ShmPtr via IpcManager (handles both raw UVA and allocator-based)
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *dst = data_ptr.ptr_;

  // Sequential copy (lane 0 only; multi-lane coroutines not yet supported)
  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    const char *block_src = src_base + block.offset_;
    char *block_dst = dst + data_off;
    memcpy(block_dst, block_src, copy_size);
    data_off += copy_size;
  }

  // Ensure GPU writes to pinned host memory are visible to CPU before
  // the task is marked complete.
  __threadfence_system();

  task->bytes_read_ = data_off;
  task->return_code_ = 0;
  (void)rctx;
  co_return;
}

}  // namespace chimaera::bdev
