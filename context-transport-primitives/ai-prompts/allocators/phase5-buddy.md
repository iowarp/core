@CLAUDE.md

# struct BuddyPage

This is the metadata stored after each AllocateOffset.
```
struct BuddyPage {
    size_t size;
}
```

# struct FreeBuddyPage
strut FreeBuddyPage : slist_node {
    size_t size;
}

# struct CoalesceBuddyPage

This is the metadata stored for coalescing.
```
struct CoalesceBuddyPage : rb_node<OffsetPointer> {
    size_t size;
}
```

# BuddyAllocator

Build this allocator and an associated unit test.
This allocator is not thread-safe. 
Ensure that you build unit tests covering the both large and small allocations.
Build a test that will trigger the coalescing.
You can use MallocBackend for the backend.

## shm_init

### Parameters
1. Heap size

### Implementation

Store the Heap and heap beginning inside the shm header.
Create a fixed table for storing free lists by allocating from the heap.
round_up_list: Free list for every power of two between 32 bytes and 16KB should have a free list. 
round_down_list: Free list for every power of two between 16KB and 1MB.

## AllocateOffset
Takes as input size. HSHM_MCTX is ignored.

Case 1: Size < 16KB
1. Get the free list for this size. Identify the free list using a logarithm base 2 of request size. Round up.
2. Check if there is a page existing in the free lists. If so, return it.
3. Check if a large page exists in the free lists, divide into smaller pages of this size. Return it.
4. Run Coalesce for all smaller page sizes. Repeat 2 & 3.
5. Try allocating from heap. Ensure the size is request size + sizeof(BuddyPage). If successful, return.
6. Return OffsetPointer::GetNull().

Case 2: Size > 16KB
1. Identify the free list using a logarithm base 2 of request size. Round down. Cap at 20 (2^20 = 1MB).
2. Check each entry if there is a fit (i.e., the page size > requested size).
3. If not, check if a larger page exists in any of the larger free lists. If yes, remove the first match and then subset the requested size. Move the remainder to the most appropriate free list. return.
4. Run Coalesce for all smaller than or equal to page sizes. Repeat 2 & 3.
5. Try allocating from heap. Ensure the size is request size + sizeof(BuddyPage).  If successful, return
6. Return OffsetPointer::GetNull()

When returning a valid page, ensure you return (page + sizeof(BuddyPage)). 
Also ensure you set the page size before returning.

## FreeOffset

Add page to the free list matching its size.
The input is the Page + sizeof(BuddyPage), so you will have to subtract sizeof(BuddyPage) first to get the page size.

## Coalesce

### Parameters

1. list_id_min: The offset of the free list with the first content to check
2. list_id_max: The offset of the free list with the last content to check

### Implementation

Build an RB-tree by popping every entry in list_id_min to list_id_max.
1. Pop entry from a free list. The entry should initially be of type FreeBuddyPage. Get the size as a variable.
2. Before adding the entry to the RB tree, cast it to CoalesceBuddyPage. Set the key to the OffsetPointer of the page. Set the size to be the size of the page.
3. When traversed, the RB tree will be the ordered list of all offsets. 
4. This way we can detect contiguity.
5. Each page should have offset, size. If offset+size is equal to the key of the left or right page, then merge the two. Merge in a loop until unsuccessful.
Then descend and continue merging. Eventually all nodes should be touched and the minimum set of contiguous blocks should be in the tree

Rebuild the free list by iterating over the RB tree
1. Pop the head of the tree and cache the page size. Cast to FreeBuddyPage and set size again. Add to the free list most appropriate for that size.
2. Continue until rb tree is completely free.
