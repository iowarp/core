@CLAUDE.md

# BuddyAllocator

Build this allocator and an associated unit test.
This allocator is not thread-safe.

## Base classes

```
// This is the metadata stored after each AllocateOffset.
struct BuddyPage {
    size_t size;
}
```

struct FreeSmallBuddyPage : slist_node {
    size_t size;
}

struct FreeLargeBuddyPage : rb_node {
    size_t size;
}

// This is the metadata stored for coalescing.
struct CoalesceBuddyPage : rb_node<OffsetPointer> {
    size_t size;
}

class _BuddyAllocator : public Allocator {
    public:
      Heap big_heap_;
      Heap small_arena_;
      slist<FreeSmallBuddyPage> round_up_[kMaxSmallPages];
      rb_tree<FreeLargeBuddyPage> round_down_[kMaxLargePages];
}
```

## shm_init

### Parameters
1. Heap size

### Implementation

Store the Heap and heap beginning inside the shm header.
Create a fixed table for storing free lists by allocating from the heap.
round_up_list: Free list for every power of two between 32 bytes and 16KB should have a free list. 
round_down_list: Free list for every power of two between 16KB and 1MB.

## AllocateOffset
Takes as input size. 

Case 1: Size < 16KB
1. Get the free list for this size. Do not include BuddyPage in the calculation. Identify the free list using a logarithm base 2 of request size. Round up.
2. Check if there is a page existing in the free lists. If so, return it.
3. Try allocating from small_arena_ (include BuddyPage in this calculation). If successful, return it.
4. Repopulate the small arena with more space:
  1. Divide the remainder of small_arena_ into pages using a greedy algorithm.
    1. Let's say we have 36KB of space left in the arena
    2. First divide by ``16KB + sizeof(BuddyPage)`` (the largest size). The result is 2. So divide into 2 ``16KB + sizeof(BuddyPage)`` pages and place in free list. We have approximately 3.9KB left.
    3. Then divide by 8KB (the next largest size). The result is 0. Continue.
    4. Then divide by 4KB (the next largest size). The result is 0. Continue.
    5. Then divide by 2KB (the next largest size). The result is 1. Divide into 1 ``2KB + sizeof(BuddyPage)`` page and place in free list. Continue.
    6. So on and so forth until the entire set of round_up_ page sizes have been cached.
  2. Try to allocate 64KB + 128*sizeof(BuddyPage) from either big heap or a round_down_ page
    1. Search every round_down_ page larger than ``64KB + 128*sizeof(BuddyPage)``.
    2. If there is one, then split the page into two. Store the remainder in the free list most matching its size. It can be in round_up_ or round_down_. Return the ``64KB + 128*sizeof(BuddyPage)``.
    3. Otherwise, allocate from the big_heap_. Return that.
  3. If non-null, update the small arena with the ``64KB + 128*sizeof(BuddyPage)`` chunk and reattempt (3).
  4. If offset is non-null, then use FullPtr<BuddyPage>(this, offset) to convert to full pointer. Set the buddy page size to the data size, excluding the BuddyPage header.
  5. Return offset

Case 2: Size > 16KB
1. Identify the free list using a logarithm base 2 of request size (no buddy page). Round down. Cap at 20 (2^20 = 1MB).
2. Check each entry if there is a fit (i.e., the page size > requested size). Make a new helper method called FindFirstFit to find the first element matching. It should return null if there is none.
3. If not, check if a larger page exists in any of the larger free lists. If yes, remove the first match and then subset the requested size. Move the remainder to the most appropriate free list. return.
4. Try allocating from heap. Ensure the size is request size + sizeof(BuddyPage).  If successful, return
5. Return OffsetPointer::GetNull()

When returning a valid page, ensure you return (page + sizeof(BuddyPage)). 
Also ensure you set the page size before returning.

## FreeOffset

Add page to the free list matching its size.
The input is the offset + sizeof(BuddyPage), so you will have to subtract sizeof(BuddyPage) first to get the page size.
Depending on the size of the page, it will need to be added to either round_up_ list or round_down_ list.
It should be dependent on the size of the page excluding the BuddyPage header.

## ReallocateOffset

Takes as input the original OffsetPtr and new size.
Get the BuddyPage for the OffsetPtr. The input is the Page + sizeof(BuddyPage), so you will have to subtract sizeof(BuddyPage) first to get the page size.
Check to see if the new size is less than or equal to the new size. If it is, then do not reallocate and just return.
Otherwise, we will need to AllocateOffset, get the FullPtr from the offset, and then copy from the old offset into the new one. Call FreeOffset afterwards.
Ensure that the size stored in the BuddyPage is the size of the page without the BuddyPage metadata header. Verify that in AllocateOffset.

