@CLAUDE.md

# ThreadLocalAllocator

I want to create an allocator that levearages thread-local storage for most operations.
Each process will have its own large block of shared memory allocated to it. 
Each process then will need 

## shm_init

### Additional Parameters

The maximum number of processes that can connect to the shared memory. Default is 256.

### Implementation
Store an atomic heap in the shared memory header.

Use heap to allocate an array of Process blocks.

Store a counter 

Allocate one block to this process.

## shm_attach
### Additional Parameters
1. The max amount of shared memory for this process.
2. Number of thread blocks to create for this process. Default 64.

### Implementation



## shm_detach

Delete all memory associated with this process and free back 

## AllocateOffset

If the MemContext has a ``tid < 0``, then we must attempt to derive it automatically.



## AlignedAllocateOffset

## FreeOffsetNoNullCheck

## Coalescing
