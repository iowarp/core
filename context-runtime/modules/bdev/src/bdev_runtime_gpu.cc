/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

#include "chimaera/bdev/bdev_gpu_runtime.h"
#include "chimaera/singletons.h"

// Backend-conditional GPU intrinsic wrappers (HSHM_DEVICE_FENCE_SYSTEM,
// HSHM_DEVICE_ATOMIC_ADD_U64_DEVICE, HSHM_DEVICE_PRINTF).
#include "hermes_shm/util/gpu_intrinsics.h"

namespace chimaera::bdev {

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

HSHM_GPU_FUN void GpuRuntime::Update(hipc::FullPtr<UpdateTask> task,
                                      chi::gpu::RunContext &rctx) {
  if (!chi::gpu::IpcManager::IsWarpScheduler()) { (void)rctx; return; }
  hbm_ptr_    = task->hbm_ptr_;
  pinned_ptr_ = task->pinned_ptr_;
  hbm_size_   = task->hbm_size_;
  pinned_size_ = task->pinned_size_;
  total_size_  = task->total_size_;
  bdev_type_   = task->bdev_type_;
  alignment_   = (task->alignment_ > 0) ? task->alignment_ : 4096;
  gpu_heap_ = 0;
  num_warps_ = chi::gpu::IpcManager::GetNumWarps();
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
}

// ---------------------------------------------------------------------------
// AllocateBlocks
// ---------------------------------------------------------------------------

HSHM_GPU_FUN void GpuRuntime::AllocateBlocks(
    hipc::FullPtr<AllocateBlocksTask> task,
    chi::gpu::RunContext &rctx) {
  if (!chi::gpu::IpcManager::IsWarpScheduler()) { (void)rctx; return; }
  chi::u64 req = task->size_;
  if (req == 0 || total_size_ == 0) {
    task->return_code_ = 0;
    (void)rctx;
    return;
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

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  if (warp_id >= num_warps_) warp_id = 0;

  Block blk;
  bool found = false;
  if (cat >= 0 && num_warps_ > 0) {
    found = warp_caches_[warp_id].lists_[cat].Pop(blk);
  }

  if (!found) {
    // Bump-pointer device-scope reservation. Wrap the rollback as
    // unsigned-add of a negative-cast so the same expression works under
    // CUDA atomicAdd and SYCL atomic_ref::fetch_add.
    chi::u64 old_pos = static_cast<chi::u64>(
        HSHM_DEVICE_ATOMIC_ADD_U64_DEVICE(&gpu_heap_, alloc_size));

    if (old_pos + alloc_size > total_size_) {
      HSHM_DEVICE_ATOMIC_ADD_U64_DEVICE(
          &gpu_heap_,
          static_cast<unsigned long long>(-static_cast<long long>(alloc_size)));
      task->return_code_ = 1;
      (void)rctx;
      return;
    }

    blk.offset_ = old_pos;
    blk.size_ = alloc_size;
    blk.block_type_ = block_type;
  }

  task->blocks_.push_back(blk);
  task->return_code_ = 0;
  (void)rctx;
}

// ---------------------------------------------------------------------------
// FreeBlocks
// ---------------------------------------------------------------------------

HSHM_GPU_FUN void GpuRuntime::FreeBlocks(hipc::FullPtr<FreeBlocksTask> task,
                                           chi::gpu::RunContext &rctx) {
  if (!chi::gpu::IpcManager::IsWarpScheduler()) { (void)task; (void)rctx; return; }

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
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
}

// ---------------------------------------------------------------------------
// Write — per-lane stripe copy
// ---------------------------------------------------------------------------

HSHM_GPU_FUN void GpuRuntime::Write(hipc::FullPtr<WriteTask> task,
                                     chi::gpu::RunContext &rctx) {
  // Bind g_ipc_manager_ptr from RunContext::ipc_mgr_ so CHI_IPC (which
  // expands to g_ipc_manager_ptr under SYCL) resolves correctly inside
  // this method body. On CUDA/ROCm rctx.ipc_mgr_ is nullptr and CHI_IPC
  // continues to reach the per-block __shared__ singleton via
  // GetBlockIpcManager(); the unused local is elided by the compiler.
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  // GetLaneId() returns threadIdx.x % 32 on CUDA/ROCm and 0 under SYCL
  // single_task (where the kernel runs as 1 work-item, so the warp-stripe
  // copy collapses to lane-0-does-everything). num_lanes follows
  // GetGpuNumThreads() so the SYCL single-thread path copies the full
  // range instead of just stripe 0.
  chi::u32 lane = chi::gpu::IpcManager::GetLaneId();
  if (lane == 0) {
    auto *ipc_tmp = CHI_IPC;
    auto dp = ipc_tmp->ToFullPtr(task->data_).template Cast<char>();
    char *db = reinterpret_cast<char*>((bdev_type_ == 2) ? hbm_ptr_ : pinned_ptr_);
    HSHM_DEVICE_PRINTF(
        "[BDEV-WRITE] len=%llu src=%p dst_base=%p blk0_off=%llu first_src=%02x\n",
        (unsigned long long)task->length_, (void*)dp.ptr_, (void*)db,
        (unsigned long long)(task->blocks_.size() > 0 ? task->blocks_[0].offset_ : 0),
        (unsigned)(dp.ptr_ ? (unsigned char)dp.ptr_[0] : 0));
  }
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);
  static constexpr chi::u32 kNoop   = static_cast<chi::u32>(BdevType::kNoop);

  if (bdev_type_ == kNoop) {
    if (lane == 0) {
      task->bytes_written_ = task->length_;
      task->return_code_ = 0;
    }
    return;
  }
  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    if (lane == 0) task->return_code_ = 1;
    return;
  }

  char *dst_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *src = data_ptr.ptr_;

  chi::u32 num_lanes = static_cast<chi::u32>(chi::gpu::IpcManager::GetGpuNumThreads());

  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    char *block_dst = dst_base + block.offset_;
    const char *block_src = src + data_off;

    if (lane >= num_lanes) { data_off += copy_size; continue; }
    chi::u64 stripe = copy_size / num_lanes;
    chi::u64 my_start = lane * stripe;
    chi::u64 my_end = (lane == num_lanes - 1) ? copy_size : (lane + 1) * stripe;
    chi::u64 my_len = my_end - my_start;

    const char *my_src = block_src + my_start;
    char *my_dst = block_dst + my_start;

#if HSHM_IS_GPU_COMPILER
    // CUDA/ROCm fast path: 16-byte vector loads/stores when both ptrs are
    // aligned. uint4 is a CUDA built-in type from <vector_types.h>.
    bool aligned16 = ((reinterpret_cast<uintptr_t>(my_dst) |
                        reinterpret_cast<uintptr_t>(my_src)) & 15) == 0;
    if (aligned16 && my_len >= sizeof(uint4)) {
      chi::u64 vec_elems = my_len / sizeof(uint4);
      const uint4 *src4 = reinterpret_cast<const uint4 *>(my_src);
      uint4 *dst4 = reinterpret_cast<uint4 *>(my_dst);
      for (chi::u64 idx = 0; idx < vec_elems; ++idx) {
        dst4[idx] = src4[idx];
      }
      chi::u64 tail_start = vec_elems * sizeof(uint4);
      for (chi::u64 b = tail_start; b < my_len; ++b) {
        my_dst[b] = my_src[b];
      }
    } else {
      for (chi::u64 b = 0; b < my_len; ++b) {
        my_dst[b] = my_src[b];
      }
    }
#else
    // SYCL: __builtin_memcpy. The compiler vectorizes through SPIR-V
    // intrinsics; uint4 isn't part of the standard SYCL types and would
    // require sycl::vec which complicates the dual-backend source.
    __builtin_memcpy(my_dst, my_src, my_len);
#endif
    HSHM_DEVICE_FENCE_SYSTEM();
    data_off += copy_size;
  }

  if (lane == 0) {
    task->bytes_written_ = data_off;
    task->return_code_ = 0;
    HSHM_DEVICE_PRINTF("[BDEV-WRITE] done bytes=%llu rc=%d\n",
                       (unsigned long long)data_off,
                       (int)task->return_code_);
  }
}

// ---------------------------------------------------------------------------
// Read — per-lane stripe copy
// ---------------------------------------------------------------------------

HSHM_GPU_FUN void GpuRuntime::Read(hipc::FullPtr<ReadTask> task,
                                    chi::gpu::RunContext &rctx) {
  [[maybe_unused]] auto *g_ipc_manager_ptr = rctx.ipc_mgr_;
  chi::u32 lane = chi::gpu::IpcManager::GetLaneId();
  static constexpr chi::u32 kHbm    = static_cast<chi::u32>(BdevType::kHbm);
  static constexpr chi::u32 kPinned = static_cast<chi::u32>(BdevType::kPinned);
  static constexpr chi::u32 kNoop   = static_cast<chi::u32>(BdevType::kNoop);

  if (bdev_type_ == kNoop) {
    if (lane == 0) {
      task->bytes_read_ = task->length_;
      task->return_code_ = 0;
    }
    return;
  }
  if (bdev_type_ != kHbm && bdev_type_ != kPinned) {
    if (lane == 0) task->return_code_ = 1;
    return;
  }

  char *src_base = reinterpret_cast<char *>(
      (bdev_type_ == kHbm) ? hbm_ptr_ : pinned_ptr_);
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).template Cast<char>();
  char *dst = data_ptr.ptr_;
  if (lane == 0) {
    HSHM_DEVICE_PRINTF(
        "[BDEV-READ] len=%llu src_base=%p dst=%p blk0_off=%llu\n",
        (unsigned long long)task->length_, (void*)src_base, (void*)dst,
        (unsigned long long)(task->blocks_.size() > 0 ? task->blocks_[0].offset_ : 0));
  }

  chi::u32 num_lanes = static_cast<chi::u32>(chi::gpu::IpcManager::GetGpuNumThreads());

  size_t num_blocks = task->blocks_.size();
  chi::u64 data_off = 0;
  for (size_t i = 0; i < num_blocks; ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - data_off;
    if (remaining == 0) break;
    chi::u64 copy_size = (block.size_ < remaining) ? block.size_ : remaining;

    const char *block_src = src_base + block.offset_;
    char *block_dst = dst + data_off;

    if (lane >= num_lanes) { data_off += copy_size; continue; }
    chi::u64 stripe = copy_size / num_lanes;
    chi::u64 my_start = lane * stripe;
    chi::u64 my_end = (lane == num_lanes - 1) ? copy_size : (lane + 1) * stripe;
    chi::u64 my_len = my_end - my_start;

    const char *my_src = block_src + my_start;
    char *my_dst = block_dst + my_start;

#if HSHM_IS_GPU_COMPILER
    bool aligned16 = ((reinterpret_cast<uintptr_t>(my_dst) |
                        reinterpret_cast<uintptr_t>(my_src)) & 15) == 0;
    if (aligned16 && my_len >= sizeof(uint4)) {
      chi::u64 vec_elems = my_len / sizeof(uint4);
      const uint4 *src4 = reinterpret_cast<const uint4 *>(my_src);
      uint4 *dst4 = reinterpret_cast<uint4 *>(my_dst);
      for (chi::u64 idx = 0; idx < vec_elems; ++idx) {
        dst4[idx] = src4[idx];
      }
      chi::u64 tail_start = vec_elems * sizeof(uint4);
      for (chi::u64 b = tail_start; b < my_len; ++b) {
        my_dst[b] = my_src[b];
      }
    } else {
      for (chi::u64 b = 0; b < my_len; ++b) {
        my_dst[b] = my_src[b];
      }
    }
#else
    __builtin_memcpy(my_dst, my_src, my_len);
#endif
    HSHM_DEVICE_FENCE_SYSTEM();
    data_off += copy_size;
  }

  HSHM_DEVICE_FENCE_SYSTEM();

  if (lane == 0) {
    task->bytes_read_ = data_off;
    task->return_code_ = 0;
  }
}

}  // namespace chimaera::bdev
