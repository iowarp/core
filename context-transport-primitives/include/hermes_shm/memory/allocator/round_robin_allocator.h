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

#ifndef HSHM_MEMORY_ALLOCATOR_ROUND_ROBIN_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_ROUND_ROBIN_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"

namespace hshm::ipc {

class _RoundRobinAllocator;
typedef BaseAllocator<_RoundRobinAllocator> RoundRobinAllocator;

/**
 * Per-partition block with a locked BuddyAllocator.
 *
 * Uses BuddyAllocator<kShared> (offset-based, position-independent) with
 * a spinlock for concurrent access from CDP parent + child kernels.
 * The BuddyAllocator MUST BE LAST — it is followed by its managed region.
 */
struct RrPartitionBlock {
  hipc::atomic<int> initialized_;  /**< 0=uninitialized, 1=ready */
  hipc::atomic<int> lock_;         /**< Spinlock: 0=unlocked, 1=locked */
  BuddyAllocator alloc_;          /**< Shared buddy allocator (MUST BE LAST) */

  HSHM_CROSS_FUN
  RrPartitionBlock() : initialized_(0), lock_(0) {}

  /**
   * Initialize the partition's buddy allocator.
   * @param backend Memory backend
   * @param region_size Total partition size including this header
   * @return true on success
   */
  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackend &backend, size_t region_size) {
    size_t alloc_region_size = region_size - sizeof(RrPartitionBlock);
    alloc_.shm_init(backend, alloc_region_size);
    lock_.store(0);
    initialized_.store(1);
#if !HSHM_IS_HOST
    __threadfence_system();
#endif
    return true;
  }

  /** Acquire the spinlock (busy-wait with nanosleep on GPU) */
  HSHM_INLINE_CROSS_FUN void Lock() {
    int expected = 0;
#if HSHM_IS_GPU
    while (!lock_.compare_exchange_strong(expected, 1)) {
      expected = 0;
      __nanosleep(50);
    }
#else
    while (!lock_.compare_exchange_strong(expected, 1)) {
      expected = 0;
    }
#endif
  }

  /** Release the spinlock */
  HSHM_INLINE_CROSS_FUN void Unlock() {
    lock_.store(0);
#if !HSHM_IS_HOST
    __threadfence();
#endif
  }

  /** Allocate with lock held */
  HSHM_INLINE_CROSS_FUN OffsetPtr<> LockedAllocate(size_t size) {
    Lock();
    OffsetPtr<> p = alloc_.AllocateOffset(size);
    Unlock();
    return p;
  }

  /** Free with lock held */
  HSHM_INLINE_CROSS_FUN void LockedFree(OffsetPtr<> p) {
    if (p.IsNull()) return;
    Lock();
    alloc_.FreeOffsetNoNullCheck(p);
    Unlock();
  }
};

/**
 * Round-robin allocator with dynamically-assigned locked partitions.
 *
 * Designed for GPU CDP where the parent orchestrator kernel and child
 * task kernels run concurrently on different SMs. Each kernel block
 * claims a partition via atomic round-robin counter. Multiple blocks
 * may share a partition (the spinlock protects concurrent access).
 *
 * Memory layout:
 *   [_RoundRobinAllocator header]
 *   [RrPartitionBlock partition 0 (partition_size_ bytes)]
 *   [RrPartitionBlock partition 1 (partition_size_ bytes)]
 *   ...
 *   [RrPartitionBlock partition N-1 (partition_size_ bytes)]
 */
class _RoundRobinAllocator : public Allocator {
 public:
  hipc::atomic<int> heap_ready_;      /**< Grid-level sync: 0=not ready, 1=ready */
  hipc::atomic<int> next_partition_;  /**< Round-robin counter */
  volatile int num_partitions_;       /**< Number of partitions */
  volatile size_t partition_size_;    /**< Bytes per partition */
  char * volatile base_;              /**< Cached base pointer */

 public:
  HSHM_CROSS_FUN
  _RoundRobinAllocator()
      : heap_ready_(0), next_partition_(0),
        num_partitions_(0), partition_size_(0), base_(nullptr) {}

  /**
   * Initialize the round-robin allocator.
   *
   * Divides the region after the header into num_partitions fixed-size
   * partitions. Each partition contains an RrPartitionBlock header
   * followed by the BuddyAllocator's managed region. Partitions are
   * lazily initialized on first claim.
   *
   * @param backend Memory backend
   * @param region_size Size of region (0 = entire backend)
   * @param num_partitions Number of partitions to create
   * @param partition_size Bytes per partition (0 = auto-compute)
   * @return true on success
   */
  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackend &backend,
                size_t region_size = 0,
                int num_partitions = 64,
                size_t partition_size = 0) {
    if (region_size == 0) {
      region_size = backend.data_capacity_;
    }

    SetBackend(backend);
    this_ = reinterpret_cast<char *>(this) -
            reinterpret_cast<char *>(backend.data_);
    base_ = reinterpret_cast<char *>(backend.data_);
    alloc_header_size_ = sizeof(_RoundRobinAllocator);
    data_start_ = sizeof(_RoundRobinAllocator);
    region_size_ = region_size;
    num_partitions_ = num_partitions;
    next_partition_.store(0);

    if (partition_size == 0) {
      size_t overhead = sizeof(_RoundRobinAllocator);
      partition_size = (region_size - overhead) / num_partitions;
    }
    // Align down to 16 bytes
    partition_size_ = partition_size & ~(size_t)15;

    // Zero-init all partition headers
    char *base = base_;
    for (int i = 0; i < num_partitions_; ++i) {
      char *part = base + sizeof(_RoundRobinAllocator) +
                   static_cast<size_t>(i) * partition_size_;
      auto *block = reinterpret_cast<RrPartitionBlock *>(part);
      block->initialized_.store(0);
      block->lock_.store(0);
    }

    return true;
  }

  /**
   * Claim a partition via round-robin.
   * Thread-safe: uses atomic increment on the counter.
   * @return Partition index [0, num_partitions_)
   */
  HSHM_INLINE_CROSS_FUN
  int ClaimPartition() {
    int idx = next_partition_.fetch_add(1);
    return idx % num_partitions_;
  }

  /**
   * Get an RrPartitionBlock by partition index.
   * O(1) — computed directly from fixed layout.
   */
  HSHM_INLINE_CROSS_FUN
  RrPartitionBlock* GetPartitionBlock(int partition_id) {
    char *b = base_;
    size_t ps = partition_size_;
    char *part = b + sizeof(_RoundRobinAllocator) +
                 static_cast<size_t>(partition_id) * ps;
    return reinterpret_cast<RrPartitionBlock *>(part);
  }

  /**
   * Lazily initialize a partition.
   * Only called once per partition (checked via initialized_ flag).
   */
  HSHM_CROSS_FUN
  bool LazyInitPartition(int partition_id) {
    if (partition_id < 0 || partition_id >= num_partitions_) {
      return false;
    }
    RrPartitionBlock *block = GetPartitionBlock(partition_id);
    if (block->initialized_.load_device() == 1) {
      return true;
    }
    // Initialize in-place (only one thread should reach here per partition)
    new (block) RrPartitionBlock();
    MemoryBackend backend = GetBackend();
    block->shm_init(backend, partition_size_);
    return true;
  }

  /**
   * Get the locked BuddyAllocator for a partition.
   * Lazily initializes the partition if needed.
   * @param partition_id Partition index from ClaimPartition()
   * @return Pointer to the partition's BuddyAllocator, or nullptr
   */
  HSHM_INLINE_CROSS_FUN
  BuddyAllocator* GetAllocator(int partition_id) {
    if (!LazyInitPartition(partition_id)) return nullptr;
    return &GetPartitionBlock(partition_id)->alloc_;
  }

  /**
   * Allocate memory from a specific partition (locked).
   * @param size Bytes to allocate
   * @param partition_id Partition to allocate from
   * @return Offset pointer to allocated memory
   */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size, int partition_id) {
    if (!LazyInitPartition(partition_id)) {
      return OffsetPtr<>::GetNull();
    }
    return GetPartitionBlock(partition_id)->LockedAllocate(size);
  }

  /**
   * Allocate memory using auto-detected partition (NOT RECOMMENDED for CDP).
   * Falls back to warp-ID-based partition like PartitionedAllocator.
   * Prefer using the partition_id overload with ClaimPartition().
   */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size) {
#if HSHM_IS_GPU
    int tid = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    int partition_id = tid % num_partitions_;
#else
    int partition_id = 0;
#endif
    return AllocateOffset(size, partition_id);
  }

  /** Reallocate (not supported) */
  HSHM_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p; (void)new_size;
    return OffsetPtr<>::GetNull();
  }

  /**
   * Free memory. Determines owning partition by address arithmetic.
   */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    char *ptr_addr = base_ + p.load();
    char *partitions_base = base_ + sizeof(_RoundRobinAllocator);
    size_t offset_in_partitions =
        static_cast<size_t>(ptr_addr - partitions_base);
    int owner = static_cast<int>(offset_in_partitions / partition_size_);
    if (owner >= 0 && owner < num_partitions_) {
      GetPartitionBlock(owner)->LockedFree(p);
    }
  }

  /** Free memory (null-safe wrapper) */
  HSHM_CROSS_FUN
  void FreeOffset(OffsetPtr<> p) {
    if (p.IsNull()) return;
    FreeOffsetNoNullCheck(p);
  }

  /** Push arena on a specific partition's BuddyAllocator */
  HSHM_CROSS_FUN bool PushArenaState(ArenaState &prior, OffsetPtr<> &block,
                                      size_t size, int partition_id) {
    if (!LazyInitPartition(partition_id)) return false;
    auto *pblock = GetPartitionBlock(partition_id);
    pblock->Lock();
    bool ok = pblock->alloc_.PushArenaState(prior, block, size);
    pblock->Unlock();
    return ok;
  }

  /** Push arena with auto-detected partition */
  HSHM_CROSS_FUN bool PushArenaState(ArenaState &prior, OffsetPtr<> &block,
                                      size_t size) {
#if HSHM_IS_GPU
    int tid = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    return PushArenaState(prior, block, size, tid % num_partitions_);
#else
    return PushArenaState(prior, block, size, 0);
#endif
  }

  /** Pop arena on the owning partition */
  HSHM_CROSS_FUN void PopArenaState(const ArenaState &prior, OffsetPtr<> block) {
    if (block.IsNull()) return;
    char *ptr_addr = base_ + block.load();
    char *partitions_base = base_ + sizeof(_RoundRobinAllocator);
    size_t offset_in_partitions =
        static_cast<size_t>(ptr_addr - partitions_base);
    int owner = static_cast<int>(offset_in_partitions / partition_size_);
    if (owner >= 0 && owner < num_partitions_) {
      auto *pblock = GetPartitionBlock(owner);
      pblock->Lock();
      pblock->alloc_.PopArenaState(prior, block);
      pblock->Unlock();
    }
  }

  /** No-op TLS management */
  HSHM_CROSS_FUN void CreateTls() {}
  HSHM_CROSS_FUN void FreeTls() {}

  /** Mark the allocator as ready (grid-level sync) */
  HSHM_CROSS_FUN void MarkReady() {
    heap_ready_.store(1);
#if !HSHM_IS_HOST
    __threadfence_system();
#endif
  }

  /** Spin-wait until the allocator is marked ready */
  HSHM_CROSS_FUN void WaitReady() {
#if !HSHM_IS_HOST
    while (heap_ready_.load_device() != 1) {
      __nanosleep(100);
    }
    __threadfence_system();
#endif
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_ROUND_ROBIN_ALLOCATOR_H_
