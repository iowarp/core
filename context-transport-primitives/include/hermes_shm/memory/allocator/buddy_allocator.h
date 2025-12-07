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

#ifndef HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/heap.h"
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/data_structures/ipc/rb_tree_pre.h"
#include <cmath>

namespace hshm::ipc {

/**
 * Metadata stored after each allocation
 */
struct BuddyPage {
  size_t size_;  /**< Size of the allocated page (excluding BuddyPage header) */

  BuddyPage() : size_(0) {}
  explicit BuddyPage(size_t size) : size_(size) {}
};

/**
 * Free page node for small allocations (<16KB)
 */
struct FreeSmallBuddyPage : public pre::slist_node {
  size_t size_;  /**< Size of the free page (including BuddyPage header) */

  FreeSmallBuddyPage() : pre::slist_node(), size_(0) {}
  explicit FreeSmallBuddyPage(size_t size) : pre::slist_node(), size_(size) {}
};

/**
 * Free page node for large allocations (>16KB)
 */
struct FreeLargeBuddyPage : public pre::slist_node {
  size_t size_;  /**< Size of the free page (including BuddyPage header) */

  FreeLargeBuddyPage() : pre::slist_node(), size_(0) {}
  explicit FreeLargeBuddyPage(size_t size) : pre::slist_node(), size_(size) {}
};

/**
 * Coalesce page node for RB tree-based coalescing
 */
struct CoalesceBuddyPage : public pre::rb_node {
  OffsetPtr<> key_;  /**< Offset pointer used as key for RB tree */
  size_t size_;      /**< Size of the page */

  CoalesceBuddyPage() : pre::rb_node(), key_(OffsetPtr<>::GetNull()), size_(0) {}
  explicit CoalesceBuddyPage(const OffsetPtr<> &k, size_t size)
      : pre::rb_node(), key_(k), size_(size) {}

  // Comparison operators required by rb_tree
  bool operator<(const CoalesceBuddyPage &other) const {
    return key_.load() < other.key_.load();
  }
  bool operator>(const CoalesceBuddyPage &other) const {
    return key_.load() > other.key_.load();
  }
  bool operator==(const CoalesceBuddyPage &other) const {
    return key_.load() == other.key_.load();
  }
};

/**
 * Buddy allocator using power-of-two free lists
 *
 * This allocator manages memory using segregated free lists for different
 * size classes. Small allocations (<16KB) use round-up sizing, while large
 * allocations (>16KB) use round-down sizing with best-fit search.
 */
class _BuddyAllocator : public Allocator {
 private:
  static constexpr size_t kMinSize = 32;           /**< Minimum allocation size (2^5) */
  static constexpr size_t kSmallThreshold = 16384; /**< 16KB threshold (2^14) */
  static constexpr size_t kMaxSize = 1048576;      /**< Maximum size class (2^20 = 1MB) */

  static constexpr size_t kMinLog2 = 5;    /**< log2(32) */
  static constexpr size_t kSmallLog2 = 14; /**< log2(16384) */
  static constexpr size_t kMaxLog2 = 20;   /**< log2(1048576) */

  static constexpr size_t kMaxSmallPages = kSmallLog2 - kMinLog2 + 1; /**< 5 to 14 = 10 lists */
  static constexpr size_t kMaxLargePages = kMaxLog2 - kSmallLog2;     /**< 15 to 20 = 6 lists */

  static constexpr size_t kSmallArenaSize = 65536; /**< 64KB arena size */
  static constexpr size_t kSmallArenaPages = 128;  /**< Max pages in small arena */

  Heap<false> big_heap_;   /**< Heap for large allocations */
  Heap<false> small_arena_; /**< Arena for small allocations */

  pre::slist<false> small_pages_[kMaxSmallPages];   /**< Free lists for sizes 32B - 16KB */
  pre::slist<false> large_pages_[kMaxLargePages]; /**< Free lists for sizes 16KB - 1MB */

  // _MultiProcessAllocator needs access to reconstruct pointers when attaching
  friend class _MultiProcessAllocator;

 public:
  /**
   * Initialize the buddy allocator
   *
   * @param backend Memory backend
   * @param region_size Size of the region in bytes. If 0, defaults to backend.data_capacity_
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    // Store backend
    SetBackend(backend);
    alloc_header_size_ = sizeof(_BuddyAllocator);

    // Default region_size to data_capacity_ if not specified
    if (region_size == 0) {
      region_size = backend.data_capacity_;
    }

    // Store region_size for use in GetAllocatorDataSize()
    region_size_ = region_size;

    // Calculate data_start_ - where the allocator's managed region begins
    // For BuddyAllocator, data starts immediately after the allocator object
    data_start_ = sizeof(_BuddyAllocator);

    // Calculate and store the offset of this allocator object within the backend data
    // This must be calculated BEFORE any GetBackendData() calls
    this_ = reinterpret_cast<char*>(this) - reinterpret_cast<char*>(backend.data_);

    if (region_size < kMinSize) {
      return false;  // Not enough space
    }

    // Initialize all free lists
    for (size_t i = 0; i < kMaxSmallPages; ++i) {
      small_pages_[i].Init();
    }
    for (size_t i = 0; i < kMaxLargePages; ++i) {
      large_pages_[i].Init();
    }

    // Initialize heaps
    size_t heap_begin = GetAllocatorDataOff();
    size_t heap_max_offset = GetAllocatorDataOff() + GetAllocatorDataSize();

    // Big heap gets all available space initially
    big_heap_.Init(heap_begin, heap_max_offset);

    // Small arena is initially empty - will be populated on first small allocation
    small_arena_.Init(0, 0);

    return true;
  }

  /**
   * Allocate memory of specified size
   *
   * @param size Size in bytes to allocate (excluding BuddyPage header)
   * @param alignment Alignment requirement (ignored - buddy allocator uses power-of-2 sizes)
   * @return Offset pointer to allocated memory (after BuddyPage header)
   */
  OffsetPtr<> AllocateOffset(size_t size, size_t alignment = 1) {
    (void)alignment;  // Buddy allocator uses power-of-2 sizes, alignment is implicit

    if (size < kMinSize) {
      size = kMinSize;
    }

    OffsetPtr<> ptr;
    if (size <= kSmallThreshold) {
      ptr = AllocateSmall(size);
    } else {
      ptr = AllocateLarge(size);
    }

    return ptr;
  }

  /**
   * Reallocate previously allocated memory to a new size
   *
   * @param offset Offset pointer to existing memory (after BuddyPage header)
   * @param new_size New size in bytes (excluding BuddyPage header)
   * @return Offset pointer to reallocated memory (may be same or different location)
   */
  OffsetPtr<> ReallocateOffset(OffsetPtr<> offset, size_t new_size) {
    // Handle null pointer case
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    // Get the BuddyPage header (offset points after header)
    size_t page_offset = offset.load() - sizeof(BuddyPage);
    hipc::FullPtr<BuddyPage> page(this, OffsetPtr<BuddyPage>(page_offset));
    size_t old_size = page.ptr_->size_;  // Size without header

    // If new size fits in existing allocation, reuse it
    if (new_size <= old_size) {
      return offset;
    }

    // Allocate new memory
    OffsetPtr<> new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;  // Allocation failed
    }

    // Copy old data to new location
    hipc::FullPtr<char> old_data(this, OffsetPtr<char>(offset.load()));
    hipc::FullPtr<char> new_data(this, OffsetPtr<char>(new_offset.load()));
    memcpy(new_data.ptr_, old_data.ptr_, old_size);

    // Free old allocation
    FreeOffset(offset);

    return new_offset;
  }

  /**
   * Free previously allocated memory
   *
   * @param offset Offset pointer to memory (after BuddyPage header)
   */
  void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }
    FreeOffsetNoNullCheck(offset);
  }

  /**
   * Free previously allocated memory (without null check)
   *
   * @param offset Offset pointer to memory (after BuddyPage header) - must not be null
   */
  void FreeOffsetNoNullCheck(OffsetPtr<> offset) {
    // Get the BuddyPage header (offset points after header)
    size_t page_offset = offset.load() - sizeof(BuddyPage);
    hipc::FullPtr<BuddyPage> page(this, OffsetPtr<BuddyPage>(page_offset));
    size_t data_size = page.ptr_->size_;  // Size without header

    // Sanity check
    // if (page_offset > GetBackendDataCapacity() - data_size) {
    //   throw std::runtime_error("Allocation failed: Out of memory");
    // }

    // Determine which free list to add to based on data size
    if (data_size <= kSmallThreshold) {
      // Small page - add to small_pages_ list using exact size match
      size_t list_idx = GetSmallPageListIndexForFree(data_size);

      // Convert to FreeSmallBuddyPage and initialize (overlays BuddyPage structure)
      hipc::FullPtr<FreeSmallBuddyPage> free_page(this, OffsetPtr<FreeSmallBuddyPage>(page_offset));
      free_page.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node base
      free_page.ptr_->size_ = data_size + sizeof(BuddyPage);

      // Add to free list
      ShmPtr<pre::slist_node> shm_ptr(GetId(), page_offset);
      FullPtr<pre::slist_node> node_ptr(this, shm_ptr);
      small_pages_[list_idx].emplace(this, node_ptr);
    } else {
      // Large page - add to large_pages_ list
      size_t list_idx = GetLargePageListIndexForFree(data_size);

      // Convert to FreeLargeBuddyPage and initialize (overlays BuddyPage structure)
      hipc::FullPtr<FreeLargeBuddyPage> free_page(this, OffsetPtr<FreeLargeBuddyPage>(page_offset));
      free_page.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node base
      free_page.ptr_->size_ = data_size + sizeof(BuddyPage);

      // Add to free list
      ShmPtr<pre::slist_node> shm_ptr(GetId(), page_offset);
      FullPtr<pre::slist_node> node_ptr(this, shm_ptr);
      large_pages_[list_idx].emplace(this, node_ptr);
    }
  }

 private:
  /**
   * Allocate small memory (<16KB) using round-up sizing
   */
  OffsetPtr<> AllocateSmall(size_t size) {
    // Step 1: Get the free list for this size (round up to next largest)
    // This also modifies size to be the rounded-up value
    size_t list_idx = GetSmallPageListIndexForAlloc(size);

    // Step 2: Check if page exists in this free list
    if (!small_pages_[list_idx].empty()) {
      auto node = small_pages_[list_idx].pop(this);
      return FinalizeAllocation(node.shm_.off_.load(), size);
    }

    // Step 3: Try allocating from small_arena_
    size_t total_size = size + sizeof(BuddyPage);
    size_t arena_offset = small_arena_.Allocate(total_size);
    if (arena_offset != 0) {
      return FinalizeAllocation(arena_offset, size);
    }

    // Step 4: Repopulate the small arena
    if (RepopulateSmallArena()) {
      // Retry allocation from arena
      arena_offset = small_arena_.Allocate(total_size);
      if (arena_offset != 0) {
        return FinalizeAllocation(arena_offset, size);
      }
    }

    // Step 5: Out of memory
    return OffsetPtr<>::GetNull();
  }

  /**
   * Allocate large memory (>16KB) using round-down sizing with best-fit
   */
  OffsetPtr<> AllocateLarge(size_t size) {
    size_t total_size = size + sizeof(BuddyPage);

    // Step 1: Identify the free list using round down
    size_t list_idx = GetLargePageListIndexForAlloc(size);

    // Step 2: Check each entry in this list for a fit
    size_t found_offset = FindFirstFit(list_idx, total_size);
    if (found_offset != 0) {
      hipc::FullPtr<FreeLargeBuddyPage> free_page(this, OffsetPtr<FreeLargeBuddyPage>(found_offset));
      size_t page_size = free_page.ptr_->size_;

      // If there's remainder, add it back to appropriate list using exact size
      if (page_size > total_size) {
        AddRemainderToFreeList(found_offset + total_size, page_size - total_size);
      }

      return FinalizeAllocation(found_offset, size);
    }

    // Step 3: Check larger free lists for first match
    for (size_t i = list_idx + 1; i < kMaxLargePages; ++i) {
      if (!large_pages_[i].empty()) {
        auto node = large_pages_[i].pop(this);
        size_t page_offset = node.shm_.off_.load();
        hipc::FullPtr<FreeLargeBuddyPage> free_page(this, OffsetPtr<FreeLargeBuddyPage>(page_offset));
        size_t page_size = free_page.ptr_->size_;

        // Subset and allocate
        if (page_size > total_size) {
          AddRemainderToFreeList(page_offset + total_size, page_size - total_size);
        }

        return FinalizeAllocation(page_offset, size);
      }
    }

    // Step 4: Try allocating from heap
    size_t heap_offset = big_heap_.Allocate(total_size);
    if (heap_offset != 0) {
      return FinalizeAllocation(heap_offset, size);
    }

    // Step 5: Out of memory
    return OffsetPtr<>::GetNull();
  }

  /**
   * Find the first fit in a large page free list
   *
   * Iterates through all entries in the specified free list to find the first
   * entry with size >= required_size. Removes the entry from the list using PopAt.
   *
   * @param list_idx Index of the free list to search
   * @param required_size Minimum size required (including BuddyPage header)
   * @return Offset to the found page, or 0 if no fit found
   */
  size_t FindFirstFit(size_t list_idx, size_t required_size) {
    if (large_pages_[list_idx].empty()) {
      return 0;
    }

    // Iterate through all entries in the list using the iterator
    for (auto it = large_pages_[list_idx].begin(this); it != large_pages_[list_idx].end(); ++it) {
      hipc::FullPtr<FreeLargeBuddyPage> free_page(
          this, OffsetPtr<FreeLargeBuddyPage>(it.GetCurrent().load()));

      // Check if this page size is large enough
      if (free_page.ptr_->size_ >= required_size) {
        // Found a fit - capture offset before removing from list
        size_t offset = it.GetCurrent().load();
        // Remove from list using PopAt (iterator is invalidated after this)
        (void)large_pages_[list_idx].PopAt(this, it);
        return offset;
      }
    }

    return 0;  // No fit found
  }

  /**
   * Repopulate small arena with more space
   * @return true if arena was successfully repopulated
   */
  bool RepopulateSmallArena() {
    size_t arena_size = kSmallArenaSize + kSmallArenaPages * sizeof(BuddyPage);

    // Step 4.2.1: Search all large_pages_ lists for space
    {
      for (size_t list_idx = 0; list_idx < kMaxLargePages; ++list_idx) {
        size_t offset = FindFirstFit(list_idx, arena_size);
        if (offset != 0) {
          small_arena_.Init(offset, offset + arena_size);
          return true;
        }
      }
    }

    // Step 4.2.3: Try allocating from big_heap_
    size_t heap_offset = big_heap_.Allocate(arena_size);
    if (heap_offset != 0) {
      // Step 4.1: Divide into pages using greedy algorithm
      // Set arena bounds for DivideArenaIntoPages to use
      DivideArenaIntoPages();
      small_arena_.Init(heap_offset, heap_offset + arena_size);

      // Step 4.3: Arena is now fully divided into free list pages
      // No need to reset - arena remains empty as all space is in free lists

      return true;
    }

    return false;  // Could not repopulate
  }

  /**
   * Divide the current small_arena_ into pages using greedy algorithm
   *
   * This function operates on the current small_arena_ state and divides
   * the arena space into free pages, adding them to the appropriate free lists.
   * After this function, the arena is marked as fully consumed.
   */
  void DivideArenaIntoPages() {
    // Get the arena bounds from the heap
    // After Init(), heap_ points to the beginning and hasn't moved yet
    size_t arena_begin = small_arena_.GetOffset();
    size_t arena_end = small_arena_.GetMaxOffset();
    size_t remaining_offset = arena_begin;
    size_t remaining_size = arena_end - arena_begin;

    // Greedy algorithm: divide by largest page sizes first
    // Start from largest small page (16KB) down to smallest (32B)
    for (int i = static_cast<int>(kMaxSmallPages) - 1; i >= 0; --i) {
      size_t page_data_size = static_cast<size_t>(1) << (i + kMinLog2);  // 2^(i+5)
      size_t page_total_size = page_data_size + sizeof(BuddyPage);

      while (remaining_size >= page_total_size) {

        // Create a free page
        hipc::FullPtr<FreeSmallBuddyPage> free_page(this, OffsetPtr<FreeSmallBuddyPage>(remaining_offset));
        free_page.ptr_->size_ = page_total_size;

        // Add to free list
        ShmPtr<pre::slist_node> shm_ptr(GetId(), remaining_offset);
        FullPtr<pre::slist_node> node_ptr(this, shm_ptr);
        small_pages_[i].emplace(this, node_ptr);

        remaining_offset += page_total_size;
        remaining_size -= page_total_size;
      }
    }

    // Any remaining space smaller than kMinSize + sizeof(BuddyPage) is wasted

    // Mark arena as fully consumed by setting it to point to the end
    small_arena_.Init(arena_end, arena_end);
  }

  /**
   * Add a remainder page back to the appropriate free list
   *
   * @param remainder_offset Offset to the remainder page
   * @param remainder_total_size Total size of remainder (including BuddyPage header)
   */
  void AddRemainderToFreeList(size_t remainder_offset, size_t remainder_total_size) {
    size_t remainder_data_size = remainder_total_size - sizeof(BuddyPage);

    if (remainder_data_size <= kSmallThreshold) {
      // Small remainder - use exact size match (round down)
      size_t rem_list_idx = GetSmallPageListIndexForFree(remainder_data_size);

      hipc::FullPtr<FreeSmallBuddyPage> remainder(this, OffsetPtr<FreeSmallBuddyPage>(remainder_offset));
      remainder.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node base
      remainder.ptr_->size_ = remainder_total_size;

      ShmPtr<pre::slist_node> rem_shm(GetId(), remainder_offset);
      FullPtr<pre::slist_node> rem_node(this, rem_shm);
      small_pages_[rem_list_idx].emplace(this, rem_node);
    } else {
      // Large remainder - use exact size match
      size_t rem_list_idx = GetLargePageListIndexForFree(remainder_data_size);

      hipc::FullPtr<FreeLargeBuddyPage> remainder(this, OffsetPtr<FreeLargeBuddyPage>(remainder_offset));
      remainder.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node base
      remainder.ptr_->size_ = remainder_total_size;

      ShmPtr<pre::slist_node> rem_shm(GetId(), remainder_offset);
      FullPtr<pre::slist_node> rem_node(this, rem_shm);
      large_pages_[rem_list_idx].emplace(this, rem_node);
    }
  }

  /**
   * Finalize allocation by setting page header and returning user offset
   *
   * @param page_offset Offset to the page (including BuddyPage header)
   * @param data_size Size of the data portion (excluding BuddyPage header)
   * @return Offset pointer to usable memory (after BuddyPage header)
   */
  OffsetPtr<> FinalizeAllocation(size_t page_offset, size_t data_size) {
    hipc::FullPtr<BuddyPage> page(this, OffsetPtr<BuddyPage>(page_offset));
    page.ptr_->size_ = data_size;  // Store size without header

    // Sanity check
    // if (page_offset > GetBackendDataCapacity() - data_size) {
    //   throw std::runtime_error("Allocation failed: Out of memory");
    // }

    return OffsetPtr<>(page_offset + sizeof(BuddyPage));
  }

  /**
   * Get free list index for small allocations when allocating (round up to next largest)
   *
   * @param size Reference to size - will be modified to the rounded-up power-of-2 size
   * @return Index into the small_pages_ array
   */
  size_t GetSmallPageListIndexForAlloc(size_t &size) {
    if (size <= kMinSize) {
      size = kMinSize;
      return 0;
    }

    // Round up to next power of 2
    size_t log2 = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(size))));

    if (log2 < kMinLog2) {
      size = kMinSize;
      return 0;
    }
    if (log2 > kSmallLog2) {
      size = kSmallThreshold;
      return kMaxSmallPages - 1;
    }

    // Calculate the rounded-up size
    size = static_cast<size_t>(1) << log2;  // 2^log2
    return log2 - kMinLog2;
  }

  /**
   * Get free list index for small pages when freeing (round down to exact or next smallest)
   */
  size_t GetSmallPageListIndexForFree(size_t size) {
    if (size <= kMinSize) return 0;

    // Round down to exact power of 2
    size_t log2 = static_cast<size_t>(std::floor(std::log2(static_cast<double>(size))));

    if (log2 < kMinLog2) return 0;
    if (log2 > kSmallLog2) return kMaxSmallPages - 1;

    return log2 - kMinLog2;
  }

  /**
   * Get free list index for large allocations when allocating (round down)
   */
  size_t GetLargePageListIndexForAlloc(size_t size) {
    if (size <= kSmallThreshold) return 0;

    // Round down to previous power of 2
    size_t log2 = static_cast<size_t>(std::floor(std::log2(static_cast<double>(size))));

    if (log2 <= kSmallLog2) return 0;
    if (log2 > kMaxLog2) return kMaxLargePages - 1;

    return log2 - kSmallLog2 - 1;
  }

  /**
   * Get free list index for large pages when freeing (round down to exact)
   */
  size_t GetLargePageListIndexForFree(size_t size) {
    return GetLargePageListIndexForAlloc(size);  // Same logic for large pages
  }
};

/** Typedef for the complete BuddyAllocator with BaseAllocator wrapper */
using BuddyAllocator = BaseAllocator<_BuddyAllocator>;

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
