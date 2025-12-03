@CLAUDE.md

I want to completely redo the AllocateBlocks and FreeBlocks algorithms in bdev chimod chimods/bdev/src/bdev_runtime.cc. They are terrible and don't work.

# WorkerBlockMap

```cpp
class WorkerBlockMap {
    std::vector<std::list<Block>> blocks_;

    bool AllocateBlock(int block_type, Block &block);

    void FreeBlock(Block block);
}
```

We cache the following block sizes: 256B, 1KB, 4KB, 64KB, 128KB.

## AllocateBlock

Pop from the list the head of list block_type and return that block.

## FreeBlock

Append to the block list.

# GlobalBlockMap

```cpp
class GlobalBlockMap {
    std::vector<WorkerBlockMap> worker_maps_;
    std::vector<chi::Mutex> worker_lock_;

    bool AllocateBlock(int worker, size_t io_size, Block &block);

    bool FreeBlock(int worker, Block &block);
}
```

## AllocateBlock

Find the next block size that is larger than this in the cache.
Get the id of that in the WorkerBlockMap.

Acquire this worker's mutex using ScopedMutex.
First attempt to allocate the block from this worker's map.
If it succeeds return. Else continue, but go out of this scope.

If we fail, then try up to 4 other workers. Just iterate linearly
over the next 4 workers.

## FreeBlock

Just free on this worker's map.

# Heap

```cpp
class Heap {
  std::atomic<size_t> heap_;

  bool Allocate(size_t block_size, Block &block);
}
```

# bdev::AllocateBlocks

Divide the I/O request in to blocks. 
If I/O size >= 128KB, then divide into units of 128KB.
Else, just use this I/O size.
Store a vector of the expected I/O size divisions.

For each expected I/O size:
First attempt to allocate from the GlobalBlockMap.
If that fails allocate from heap.
If that fails, then print an error and set the return code to 1.

## bdev::FreeBlocks

Call GlobalBlockMap FreeBlock.
