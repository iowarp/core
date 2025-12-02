/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/thread/lock/mutex.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include <sys/types.h>
#include <unistd.h>

namespace hshm::ipc {

/**
 * Forward declarations
 */
class ProcessBlock;
class _MultiProcessAllocator;

/**
 * Typedef for the complete MultiProcessAllocator with BaseAllocator wrapper
 */
typedef BaseAllocator<_MultiProcessAllocator> MultiProcessAllocator;

/**
 * Per-thread allocator block providing lock-free fast-path allocation.
 *
 * Each thread has its own ThreadBlock with a private BuddyAllocator,
 * enabling concurrent allocations without contention. When a ThreadBlock
 * runs out of memory, it requests expansion from its parent ProcessBlock.
 */
class ThreadBlock : public pre::slist_node {
 public:
  int tid_;                    /**< Thread ID */
  BuddyAllocator alloc_;       /**< Private buddy allocator for this thread */

  /**
   * Default constructor
   */
  ThreadBlock() : tid_(-1) {}

  /**
   * Initialize the thread block with a memory region
   *
   * @param backend Memory backend containing the region for this thread
   * @param size Size of the memory region in bytes
   * @param tid Thread ID for this block
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t size, int tid) {
    tid_ = tid;

    // Create a shifted backend that starts after the ThreadBlock header
    size_t header_size = sizeof(ThreadBlock);
    MemoryBackend thread_backend = backend.Shift(header_size);

    // Initialize the buddy allocator with the shifted backend
    alloc_.shm_init(AllocatorId(backend.GetId(), tid_), thread_backend);
    return true;
  }

  /**
   * Allocate memory from this thread block
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr<> Allocate(size_t size) {
    return alloc_.AllocateOffset(size);
  }

  /**
   * Free memory back to this thread block
   *
   * @param ptr Offset pointer to memory to free
   */
  void Free(OffsetPtr<> ptr) {
    alloc_.FreeOffset(ptr);
  }

  /**
   * Reallocate memory within this thread block
   *
   * @param ptr Original offset pointer
   * @param new_size New size in bytes
   * @return New offset pointer, or null on failure
   */
  OffsetPtr<> Reallocate(OffsetPtr<> ptr, size_t new_size) {
    // Simple implementation: allocate new, copy, free old
    // TODO: Optimize with in-place reallocation when possible
    OffsetPtr<> new_ptr = Allocate(new_size);
    if (new_ptr.IsNull()) {
      return OffsetPtr<>::GetNull();
    }

    // For now, just return the new pointer
    // In a complete implementation, we would copy data and free old pointer
    Free(ptr);
    return new_ptr;
  }
};

/**
 * Per-process allocator block managing multiple ThreadBlocks.
 *
 * Each process has one ProcessBlock that manages a pool of ThreadBlocks.
 * The ProcessBlock uses a mutex to protect its thread list and can
 * expand by requesting more memory from the global allocator.
 */
class ProcessBlock : public pre::slist_node {
 public:
  int pid_;                    /**< Process ID */
  int tid_count_;              /**< Number of thread blocks allocated */
  hshm::Mutex lock_;           /**< Mutex protecting thread list */
  BuddyAllocator alloc_;       /**< Allocator for managing ThreadBlock regions */
  pre::slist<false> threads_;  /**< List of ThreadBlocks */

  // Thread-local storage keys (process-local, not shared)
  ThreadLocalKey tblock_key_;  /**< TLS key for ThreadBlock* */
  ThreadLocalKey pblock_key_;  /**< TLS key for ProcessBlock* */

  /**
   * Default constructor
   */
  ProcessBlock() : pid_(-1), tid_count_(0) {}

  /**
   * Set up thread-local storage keys
   * @return true on success, false on failure
   */
  bool SetupTls() {
    void *null_ptr = nullptr;
    if (!HSHM_THREAD_MODEL->CreateTls<void>(tblock_key_, null_ptr)) {
      return false;
    }
    if (!HSHM_THREAD_MODEL->CreateTls<void>(pblock_key_, null_ptr)) {
      return false;
    }
    return true;
  }

  /**
   * Initialize the process block with a memory region
   *
   * @param backend Memory backend for this process
   * @param region Start of the memory region
   * @param size Size of the memory region in bytes
   * @param pid Process ID
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, char *region, size_t size, int pid) {
    pid_ = pid;
    tid_count_ = 0;
    lock_.Init();
    threads_.Init();

    // Calculate space for ProcessBlock header and thread list
    size_t header_size = sizeof(ProcessBlock);

    // Create a shifted backend for the buddy allocator
    // Calculate offset from root to this region, then shift past header
    size_t region_offset = region - backend.data_;
    MemoryBackend process_backend = backend.Shift(region_offset + header_size);
    process_backend.data_size_ = size - header_size;

    // Initialize buddy allocator for managing thread blocks
    alloc_.shm_init(AllocatorId(backend.GetId(), pid_), process_backend);
    return true;
  }

  /**
   * Allocate a new ThreadBlock from this ProcessBlock
   *
   * @param backend Root memory backend (for FullPtr construction)
   * @param region_size Size of region to allocate for the ThreadBlock
   * @return FullPtr to the newly allocated ThreadBlock, or null on failure
   */
  FullPtr<ThreadBlock> AllocateThreadBlock(const MemoryBackend &backend, size_t region_size) {
    // Lock to protect thread list
    lock_.Lock(0);

    // Allocate memory for ThreadBlock from our buddy allocator
    OffsetPtr<> thread_offset = alloc_.AllocateOffset(region_size);
    if (thread_offset.IsNull()) {
      lock_.Unlock();
      return FullPtr<ThreadBlock>::GetNull();
    }

    // Convert offset to pointer
    char *thread_ptr = alloc_.GetBackend().data_ + thread_offset.load();
    ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(thread_ptr);

    // Initialize ThreadBlock in-place
    new (tblock) ThreadBlock();

    // Create backend for this ThreadBlock (keep root data_, set offset)
    MemoryBackend thread_backend = backend;
    thread_backend.data_offset_ = thread_offset.load();
    thread_backend.data_size_ = region_size;

    if (!tblock->shm_init(thread_backend, region_size, tid_count_++)) {
      alloc_.FreeOffset(thread_offset);
      lock_.Unlock();
      return FullPtr<ThreadBlock>::GetNull();
    }

    // Add to thread list
    ShmPtr<> thread_shm;
    thread_shm.off_ = thread_offset;
    FullPtr<pre::slist_node> node_ptr(reinterpret_cast<pre::slist_node*>(tblock), thread_shm);
    threads_.emplace(&alloc_, node_ptr);

    lock_.Unlock();

    // Return FullPtr to the ThreadBlock
    FullPtr<ThreadBlock> result;
    result.ptr_ = tblock;
    result.shm_.off_ = thread_offset;
    result.shm_.alloc_id_ = AllocatorId(backend.GetId(), 0);
    return result;
  }

  /**
   * Expand this ProcessBlock by freeing a region back to it
   *
   * This is called when memory is returned to the ProcessBlock allocator,
   * typically when a ThreadBlock is freed or when coalescing occurs.
   *
   * @param ptr Offset pointer to memory region to return
   */
  void Expand(OffsetPtr<> ptr) {
    lock_.Lock(0);
    alloc_.FreeOffset(ptr);
    lock_.Unlock();
  }
};

/**
 * Private header stored in backend's private region (process-local, not shared)
 */
struct MultiProcessAllocatorPrivateHeader {
  ProcessBlock *process_block_;  /**< Pointer to this process's ProcessBlock */

  MultiProcessAllocatorPrivateHeader() : process_block_(nullptr) {}
};

/**
 * Multi-process allocator with thread-local storage for lock-free fast path.
 *
 * Architecture:
 * - Global BuddyAllocator manages the entire shared memory region
 * - ProcessBlocks are allocated per process, each managing ThreadBlocks
 * - ThreadBlocks provide lock-free allocation for individual threads
 *
 * Allocation Strategy (3-tier fallback):
 * 1. Fast path: Allocate from thread-local ThreadBlock (no locks)
 * 2. Medium path: Expand ThreadBlock from ProcessBlock (one lock)
 * 3. Slow path: Expand ProcessBlock from global allocator (global lock)
 *
 * This design minimizes contention and provides malloc-like performance
 * for the common case of thread-local allocations.
 *
 * Memory Layout:
 * The allocator itself is placed at the beginning of shared memory.
 * Header fields (pid_count_, lists, lock) are part of the allocator structure.
 * The BuddyAllocator follows immediately after the allocator header.
 */
class _MultiProcessAllocator : public Allocator {
 public:
  // Header fields (shared memory compatible)
  int pid_count_;                /**< Number of processes */
  pre::slist<false> alloc_procs_;/**< Active ProcessBlocks */
  pre::slist<false> free_procs_; /**< Free ProcessBlocks */
  hshm::Mutex lock_;             /**< Mutex protecting process lists */

  // Allocator state (shared memory compatible)
  BuddyAllocator alloc_;         /**< Global buddy allocator */
  size_t process_unit_;          /**< Default ProcessBlock size (1GB) */
  size_t thread_unit_;           /**< Default ThreadBlock size (16MB) */

 private:
  // Process-local storage (NOT in shared memory)
  static thread_local ProcessBlock *process_block_;  /**< Thread-local process block pointer */

 public:
  /**
   * Default constructor
   */
  _MultiProcessAllocator() : pid_count_(0), process_unit_(0), thread_unit_(0) {}

  /**
   * Set the process block for this process
   */
  void SetProcessBlock(ProcessBlock *pblock) {
    process_block_ = pblock;
  }

  /**
   * Get the process block for this process
   */
  ProcessBlock* GetProcessBlock() const {
    return process_block_;
  }


  /**
   * Initialize the allocator with a new memory region
   *
   * @param id Allocator ID
   * @param backend Memory backend to use (allocator will be placed at backend.data_)
   * @param size Total size of the memory region
   * @return true on success, false on failure
   */
  bool shm_init(AllocatorId id, const MemoryBackend &backend, size_t size) {
    id_ = id;
    SetBackend(backend);
    alloc_header_size_ = sizeof(_MultiProcessAllocator);
    custom_header_size_ = 0;  // No custom header for MultiProcessAllocator

    // Initialize header fields directly (we are the header!)
    pid_count_ = 0;
    lock_.Init();
    alloc_procs_.Init();
    free_procs_.Init();

    // Create a shifted backend for the global allocator
    // It starts after this allocator object
    MemoryBackend alloc_backend = backend.Shift(sizeof(_MultiProcessAllocator));

    // Initialize global buddy allocator with shifted backend
    alloc_.shm_init(AllocatorId(backend.GetId(), 0), alloc_backend);

    // Set default sizes based on available memory
    // For testing with smaller memory regions, use smaller defaults
    if (size < 1ULL * 1024 * 1024 * 1024) {
      // For regions < 1GB, use smaller units
      process_unit_ = size / 4;                  // Use 1/4 of available space
      thread_unit_ = 4ULL * 1024 * 1024;         // 4MB
    } else {
      process_unit_ = 1ULL * 1024 * 1024 * 1024;  // 1GB
      thread_unit_ = 16ULL * 1024 * 1024;         // 16MB
    }

    // Allocate first ProcessBlock
    OffsetPtr<> pblock_offset = alloc_.AllocateOffset(process_unit_);
    if (pblock_offset.IsNull()) {
      return false;
    }

    // Initialize ProcessBlock
    MemoryBackend root_backend = GetBackend();
    char *pblock_ptr = root_backend.data_ + pblock_offset.load();
    ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(pblock_ptr);
    new (pblock) ProcessBlock();

    // The pblock_offset is already relative to root_backend.data_, so pblock_ptr is correct
    if (!pblock->shm_init(root_backend, pblock_ptr, process_unit_, getpid())) {
      alloc_.FreeOffset(pblock_offset);
      return false;
    }

    // Set up TLS for this process block
    if (!pblock->SetupTls()) {
      return false;
    }

    // Store process block pointer in private header
    SetProcessBlock(pblock);

    // Add to process list
    ShmPtr<> pblock_shm;
    pblock_shm.off_ = pblock_offset;
    FullPtr<pre::slist_node> pblock_node(reinterpret_cast<pre::slist_node*>(pblock), pblock_shm);
    alloc_procs_.emplace(&alloc_, pblock_node);
    pid_count_++;

    return true;
  }

  /**
   * Attach to an existing allocator (for multi-process scenarios)
   *
   * This method allows a process to attach to shared memory that was
   * initialized by another process. It reconstructs the allocator state
   * from the shared memory and sets up process-local state (TLS, ProcessBlock).
   *
   * @param backend Memory backend that was initialized by the creator process
   * @param process_unit Default size for ProcessBlocks (default: determined by memory size)
   * @param thread_unit Default size for ThreadBlocks (default: determined by memory size)
   * @return true on success, false on failure
   */
  bool shm_attach(const MemoryBackend &backend,
                  size_t process_unit = 0,
                  size_t thread_unit = 0) {
    // Verify allocator was initialized
    if (pid_count_ == 0) {
      return false;
    }

    // Create a shifted backend for the global allocator
    MemoryBackend alloc_backend = backend.Shift(sizeof(_MultiProcessAllocator));

    // Attach to the buddy allocator (don't reinitialize!)
    // Reconstruct pointers to the free lists that are already in shared memory
    alloc_.GetBackend() = alloc_backend;
    alloc_.GetBackend().SetInitialized();

    // Calculate aligned metadata size (must match what shm_init did)
    // Use the same constants as BuddyAllocator::shm_init
    constexpr size_t kNumRoundUpLists = 10;  // 5 to 14 = 10 lists (from BuddyAllocator)
    constexpr size_t kNumRoundDownLists = 6;  // 15 to 20 = 6 lists (from BuddyAllocator)
    constexpr size_t kNumFreeLists = kNumRoundUpLists + kNumRoundDownLists;
    size_t metadata_size = kNumFreeLists * sizeof(pre::slist<false>);
    size_t aligned_metadata = ((metadata_size + 63) / 64) * 64;

    // Reconstruct pointers to existing free lists
    char *region_start = alloc_backend.data_ + alloc_backend.data_offset_;
    alloc_.round_up_lists_ = reinterpret_cast<pre::slist<false>*>(region_start);
    alloc_.round_down_lists_ = alloc_.round_up_lists_ + kNumRoundUpLists;

    // Reconstruct heap boundaries (these should match what was initialized)
    alloc_.heap_begin_ = alloc_backend.data_offset_ + aligned_metadata;
    alloc_.heap_current_ = alloc_.heap_begin_;  // Note: This is approximate, actual value may be higher
    alloc_.heap_end_ = alloc_backend.data_offset_ + alloc_backend.data_size_;

    // Set default sizes from backend size if not specified
    size_t size = backend.data_size_;
    if (process_unit == 0) {
      if (size < 1ULL * 1024 * 1024 * 1024) {
        process_unit_ = size / 4;
      } else {
        process_unit_ = 1ULL * 1024 * 1024 * 1024;
      }
    } else {
      process_unit_ = process_unit;
    }

    if (thread_unit == 0) {
      if (size < 1ULL * 1024 * 1024 * 1024) {
        thread_unit_ = 4ULL * 1024 * 1024;
      } else {
        thread_unit_ = 16ULL * 1024 * 1024;
      }
    } else {
      thread_unit_ = thread_unit;
    }

    // Allocate a ProcessBlock for this process
    OffsetPtr<> pblock_offset = alloc_.AllocateOffset(process_unit_);
    if (pblock_offset.IsNull()) {
      return false;
    }

    // Initialize ProcessBlock
    MemoryBackend root_backend = GetBackend();
    char *pblock_ptr = root_backend.data_ + pblock_offset.load();
    ProcessBlock *pblock = reinterpret_cast<ProcessBlock*>(pblock_ptr);
    new (pblock) ProcessBlock();

    if (!pblock->shm_init(root_backend, pblock_ptr, process_unit_, getpid())) {
      alloc_.FreeOffset(pblock_offset);
      return false;
    }

    // Set up TLS for this process block
    if (!pblock->SetupTls()) {
      return false;
    }

    // Store process block pointer in private header
    SetProcessBlock(pblock);

    // Add to process list (with locking since other processes may be modifying)
    ShmPtr<> pblock_shm;
    pblock_shm.off_ = pblock_offset;
    FullPtr<pre::slist_node> pblock_node(reinterpret_cast<pre::slist_node*>(pblock), pblock_shm);

    // Lock while modifying shared header
    lock_.Lock(getpid());
    alloc_procs_.emplace(&alloc_, pblock_node);
    pid_count_++;
    lock_.Unlock();

    return true;
  }

  /**
   * Detach from the allocator (cleanup TLS)
   */
  void shm_detach() {
    // TLS cleanup is handled automatically by pthread_key_create destructor
  }

 public:

  /**
   * Ensure thread-local storage is set up for the current thread
   *
   * This is the critical function for lock-free fast path allocation.
   * It checks TLS for a ThreadBlock, and if not found, allocates one
   * from the ProcessBlock (creating a ProcessBlock if needed).
   *
   * @return Pointer to the thread's ThreadBlock, or nullptr on failure
   */
  ThreadBlock* EnsureTls() {
    // Get or create ProcessBlock from private header
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock == nullptr) {
      // First time - allocate ProcessBlock
      pblock = AllocateProcessBlock();
      if (pblock == nullptr) {
        return nullptr;
      }
      SetProcessBlock(pblock);
    }

    // Check if we already have a ThreadBlock in TLS
    void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock->tblock_key_);
    if (tblock_data != nullptr) {
      return reinterpret_cast<ThreadBlock*>(tblock_data);
    }

    // Allocate ThreadBlock from ProcessBlock
    FullPtr<ThreadBlock> tblock_ptr = pblock->AllocateThreadBlock(GetBackend(), thread_unit_);
    if (tblock_ptr.IsNull()) {
      // Try expanding ProcessBlock
      OffsetPtr<> expand_offset = alloc_.AllocateOffset(thread_unit_);
      if (!expand_offset.IsNull()) {
        pblock->Expand(expand_offset);
        tblock_ptr = pblock->AllocateThreadBlock(GetBackend(), thread_unit_);
      }
    }

    if (tblock_ptr.IsNull()) {
      return nullptr;
    }

    // Store in TLS
    ThreadBlock *tblock = tblock_ptr.ptr_;
    HSHM_THREAD_MODEL->SetTls<void>(pblock->tblock_key_, reinterpret_cast<void*>(tblock));
    return tblock;
  }

  /**
   * Allocate a ProcessBlock for the current process
   *
   * @return Pointer to ProcessBlock, or nullptr on failure
   */
  ProcessBlock* AllocateProcessBlock() {
    // Lock the global process list
    lock_.Lock(0);

    ProcessBlock *pblock = nullptr;
    OffsetPtr<> pblock_offset;
    int pid;
    bool reused = false;

    // Try to pop from free_procs_ first
    if (!free_procs_.empty()) {
      auto node = free_procs_.pop(&alloc_);
      if (!node.IsNull()) {
        pblock = reinterpret_cast<ProcessBlock*>(node.ptr_);
        pblock_offset = OffsetPtr<>(node.shm_.off_.load());
        pid = pblock->pid_;  // Reuse existing PID
        reused = true;
      }
    }

    // If not reused, allocate new ProcessBlock
    if (!reused) {
      pblock_offset = alloc_.AllocateOffset(process_unit_);
      if (pblock_offset.IsNull()) {
        lock_.Unlock();
        return nullptr;
      }

      // Initialize ProcessBlock
      char *pblock_ptr = alloc_.GetBackend().data_ + pblock_offset.load();
      pblock = reinterpret_cast<ProcessBlock*>(pblock_ptr);
      new (pblock) ProcessBlock();

      // Allocate new PID using pid_count_
      pid = pid_count_++;

      if (!pblock->shm_init(GetBackend(), pblock_ptr, process_unit_, pid)) {
        alloc_.FreeOffset(pblock_offset);
        lock_.Unlock();
        return nullptr;
      }
    }

    // Add to allocated process list
    ShmPtr<> pblock_shm;
    pblock_shm.off_ = pblock_offset;
    FullPtr<pre::slist_node> pblock_node(reinterpret_cast<pre::slist_node*>(pblock), pblock_shm);
    alloc_procs_.emplace(&alloc_, pblock_node);

    lock_.Unlock();
    return pblock;
  }

  /**
   * Allocate memory using the 3-tier allocation strategy
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr<> AllocateOffset(size_t size) {
    // Tier 1: Fast path - try thread-local allocator (lock-free)
    ThreadBlock *tblock = EnsureTls();
    if (tblock != nullptr) {
      OffsetPtr<> ptr = tblock->Allocate(size);
      if (!ptr.IsNull()) {
        return ptr;
      }
    }

    // Tier 2: Medium path - try expanding ThreadBlock from ProcessBlock
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock != nullptr) {
      OffsetPtr<> expand_offset = alloc_.AllocateOffset(size);
      if (!expand_offset.IsNull()) {
        pblock->Expand(expand_offset);
        if (tblock != nullptr) {
          OffsetPtr<> ptr = tblock->Allocate(size);
          if (!ptr.IsNull()) {
            return ptr;
          }
        }
      }
    }

    // Tier 3: Slow path - allocate directly from global allocator
    lock_.Lock(0);
    OffsetPtr<> ptr = alloc_.AllocateOffset(size);
    lock_.Unlock();
    return ptr;
  }

  /**
   * Reallocate memory to a new size
   *
   * @param offset Original offset pointer
   * @param new_size New size in bytes
   * @return New offset pointer, or null on failure
   */
  OffsetPtr<> ReallocateOffset(OffsetPtr<> offset, size_t new_size) {
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    // Allocate new memory
    OffsetPtr<> new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;
    }

    // Copy old data to new location
    // We need to determine the size of the old allocation
    // For now, copy up to new_size (assuming old is at least as large)
    // TODO: Store allocation sizes to copy exact old size
    char *old_data = GetBackend().data_ + offset.load();
    char *new_data = GetBackend().data_ + new_offset.load();
    memcpy(new_data, old_data, new_size);

    // Free old allocation
    FreeOffset(offset);

    return new_offset;
  }

  /**
   * Free allocated memory
   *
   * @param offset Offset pointer to memory to free
   */
  void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }

    // Determine which allocator owns this memory
    // For simplicity, try thread-local first, then global
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock != nullptr) {
      void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock->tblock_key_);
      ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(tblock_data);
      if (tblock != nullptr) {
        // Check if offset is within thread block range
        // For now, just try to free it
        tblock->Free(offset);
        return;
      }
    }

    // Fall back to global allocator
    lock_.Lock(0);
    alloc_.FreeOffset(offset);
    lock_.Unlock();
  }

  /**
   * Get the custom header of the shared-memory allocator
   *
   * MultiProcessAllocator does not support custom headers, so this returns nullptr.
   *
   * @return nullptr
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return nullptr;
  }
};

// Define static thread_local member
thread_local ProcessBlock* _MultiProcessAllocator::process_block_ = nullptr;

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
