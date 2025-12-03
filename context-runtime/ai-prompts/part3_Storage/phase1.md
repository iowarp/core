@CLAUDE.md Create a chimod called bdev, which stands for block device. Use the same namespace as MOD_NAME. Make sure to read @docs/MODULE_DEVELOPMENT_GUIDE.md and to use chi_refresh_repo.py when building the module. 

## CreateTask

The parameters for the CreateTask will contain a chi::string inidicating the path to a file to open. 

In the Create function, it will conduct a small benchmark to assess the performance of the device. These performance counters will be stored internally.

## AllocateTask

The task takes as input the amount of data to allocate, which is a u64.

In the runtime, this will implement a simple data allocator, similar to a memory allocator. For now, assume there are 4 different block sizes: 4KB, 64KB, 256KB, 1MB. 

AllocateBlocks:
1. Calculate the minimum set of blocks to allocate to meet the size requirement. If the size is less than 1MB, then allocate a single block. The block size should be the next largest. So if I have 256 bytes, it will round up to 4KB. If I have 8192 bytes, then it will round up to 64KB. If the size is larger than 1MB, we will allocate only 1MB blocks until the size requirement is met. For example, if we have 3MB request, we will allocate 3 1MB blocks. if we have 3.5MB, then we will allocate 4 1MB blocks.
2. To allocate blocks, we need to store a free list for each size type. First check the free list if there are any available blocks. If no free blocks are available, allocate off of the heap. The heap is an atomic, monotonically increasing counter with maximum size file_size. If both the heap and free lists are out of space, then error. 
3. Decrement remaining capacity based on total allocated block size.

FreeBlocks:
1. Simply add the set of blocks being freed to their respective free lists. Increment the remaining capacity.

When the AllocateTask comes in, map the size to the next largest size of data. Check the free list for the size type. If there is a free block, then use that. Otherwise, we will increment a heap offset and then allocate a new block off the heap. If there is no space left in the heap, then we should return an error. Do not use strings for the errors, use only numbers.

This task should also maintain the remaining size of data. This should be a simple atomic counter. Allocation decreases the counter.

## FreeTask

Takes as input a block to free. No need for complex free detection or corruption algorithms. 

In the runtime, this will add the block to the most appropriate free list and then increase the available remaining space.

## WriteTask and ReadTask

These tasks are similar. They take as input a Block and then read or write to the file asynchronously.

Bdev uses libaio to read and write data. Use direct I/O if libaio supports it. The data should always be aligned to 4KB offsets in the file, which I believe is the requirement for direct I/O. 

## StatTask

This task takes no inputs. As output it will return the performance and remaining size. 
