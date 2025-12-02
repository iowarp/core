@CLAUDE.md

# Aligned Buddy Allocator

Similar to the Buddy Allocator, but with one major difference:
we store the set of all allocated pages in a table.



# DMA Allocator

This allocator focuses on optimizing 4KB aligned allocations
for DMA operations. Every allocation is aligned to 4KB.

This considers both the data_ pointer itself 

This is much like the MultiProcess allocator, except the
backend allocator is not the BuddyAllocator.

Instead, we will need to create 

Can we store the set of free pages in like a hashmap or something in the buddy allocator?

