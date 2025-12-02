@CLAUDE.md

# Shm Backend update
I want context-transport-primitives/include/hermes_shm/memory/backend/posix_shm_mmap.h to support a mix of private and shared mapping.

I need a contiguous region where the first say 16KB of the region is private memory and the following size bytes are shared memory.
I don't mind if this requires multiple mmap calls, but it needs to be guaranteed correct.
Is this possible?

@CLAUDE.md

# General Backend Update

Each backend should have the first 16KB dedicated to some private memory for allocators
to leverage thread-local storage semantics better.

MemoryBackend should look like this:
data_: the shared part of the region (for posix shm mmap)

Every backend should support:
(data_ - kBachendPrivate) to get a region of valid private memory.

The kBackendPrivate should be in addition to any size parameter given for the data segment.

Create a global constant called kBackendPrivate = 4KB. Update the PosixShmMmap allocator to use this constant for the Mixed allocation.

@CLAUDE.md

# Improving allocator ease-of-use

We need to avoid passing the allocator so much. 

Let's make the Allocator classes themselves shared-memory compatible. 

## General Observation

Containers should be able to get the pointer to the allocator class as follows:
1. Upon construction, the container is initially passed the Allocator pointer
2. The container should store OffsetPtr<> this_ = (this - alloc)
3. Allocator *alloc = (this - this_);

This assumes that the Allocator is allocated on the Memory backend.
Instead of passing the MemoryBackend to the Allocator, 
we should be casting the MemoryBackend data_ pointer to an Allocator*.

## MemoryBackend
We should add the following new apis to the MemoryBackend:
1. AllocT* cast<AllocT>: this will simply return reinterpret_cast<AllocT>(data_);


## Allocator
Remove the following from the Allocator:
```
MemoryBackend backend_;
int accel_id_;
char *custom_header_;
```

Add the following:
```
size_t size_;  // The size of the memory backend.
```

Update ContainsPtr to use the size_ variable only.
```
ContainsPtr(OffsetPtr &off) {  return off < size_;  }
ContainsPtr(char *ptr) { (ptr - this) < size_; }
```


## BuddyAllocator

Remove the fields:
```
  size_t heap_begin_;           /**< Offset to heap beginning */
  size_t heap_current_;         /**< Current heap offset */
  size_t heap_end_;             /**< End of heap */
```

Do not let the following be pointers:
```
  pre::slist<false> *round_up_lists_;    /**< Free lists for sizes 32B - 16KB (round up) */
  pre::slist<false> *round_down_lists_;  /**< Free lists for sizes 16KB - 1MB (round down) */
```

Change them to this:
```
  pre::slist<false> round_up_lists_;    /**< Free lists for sizes 32B - 16KB (round up) */
  pre::slist<false> round_down_lists_;  /**< Free lists for sizes 16KB - 1MB (round down) */
```

## MultiProcessAllocator

Add a method called GetPrivate() that returns (this - kBackendPrivate).
this should be backend.data_.

Store the TLS keys inside (backend.data_ - kBackendPrivate).
```
struct MpPrivateHeader {
    ThreadLocalKey tls_key_;
};

MpPrivateHeader* GetPrivate() {
    return ((char*)this - kBackendPrivate)
}
```

## CtxAllocator

Let's remove CtxAllocator concept completely.
We will pass allocator pointers around.

