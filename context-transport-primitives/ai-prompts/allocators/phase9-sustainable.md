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


@CLAUDE.md 
The current issue is that alloc_ must be the last entry of the shared memory in order to avoid corrupting class parameters. For pblock and tblock, this is an easy change.

However, the main block is different due to the custom header. Simply placing alloc_ at the end there is problematic.

How do we fix this:
1. Make custom header a part of the backend, not the allocator. I actually like this a lot. 

Each backend has a private header and a shared header, both 4KB long. 
Add a new method called GetSharedHeader to MemoryBackend. GetPrivateRegion should be renamed to GetPrivateHeader(). kBackendPrivate should be renamed to kBackendHeaderSize

GetPrivateRegion() should be GetSharedHeader() - kBackendHeaderSize. GetSharedHeader() should be data_ - kBackendHeaderSize().

Remove all logic in the allocators for considering custom_header_size_. Remove custom_header_size_ completely from allocators. We should rename GetCustomHeader in Allocator to GetSharedHeader().

We should add a new class variable to allocator called data_start_ (this is not custom_header_size_). This represents the start of data relative to this_. Technically, this is just the size of the allocator class: data_start_ = sizeof(AllocT).

GetAllocatorDataStart() should not depend on GetCustomHeader / GetSharedHeader anymore. Instead we should return (this) + data_start_

@CLAUDE.md

In the MemoryBackend, I want to add another variable called priv_header_off_.
This will store the difference between data_ and the beginning of the shared segment in the MemoryBackend.
In each MemoryBackend, we need to set this priv_header_off_. 

For example, for PosixShmMmap, we do a mixed allocation.
1. The very first kBackendHeaderSize bytes of the buffer returned is the private header
2. The next kBackendHeaderSize bytes are the shared header.
3. The next bytes can be the metadata.
4. And then data_ is set.
5. And then priv_header_off_ is data_ - (1).


For PosixMmap,
1. We mmap the buffer
2. The very first kBackendHeaderSize bytes of the buffer returned is the private header
3. The next kBackendHeaderSize bytes are the shared header. 
4. After md, the next kBackendHeader bytes are the private header and the next are the shared header.
5. After this is what gets stored in data_. 

In MemoryBackend:
```
GetPrivateHeader(): GetPrivateHeader(data_)
GetSharedHeader(): GetSharedHeader(data_)
GetPrivateHeader(char *data): (data - priv_header_off_)
GetSharedHeader(char *data): GetPrivateHeader<char>(data) + kBackendHeaderSize
```

In Allocator:
```
GetPrivateHeader(): backend_.GetPrivateHeader(GetBackendData());
GetSharedHeader(): backend_.GetSharedHeader(GetBackendData());
```

@CLAUDE.md
Allocators should take as input MemoryBackend and size_t region_size.
This is the size of the region the allocator is allowed to occupy, including the allocator header.

Let's remove data_offset_  and data_size_ from the MemoryBackend structure. Remove ShiftTo* functions.
For the allocator code that uses it, simply remove that code. Pass in the region_size to the allocator.
By default, region_size should be set to 0, in which case we set region_size equal to MemoryBackend.data_capacity_.
We should use region_size instead of backend.data_size_ in the shm_init code for all allocators.

Store region_size_ in the class Allocator. Set it in shm_init. Also use that in GetAllocatorDataSize().
Instead of GetBackendCapacity(), use region_size_

@CLAUDE.md

For PosixShmMmap, we do need two mmaps in both shm_init and shm_attach.

shm_init:
Use MapShared to map the first 4KB of the fd_
This will be header_.
Use MapMixed for the remaining.
This will be for the private header, shared header, and data.

It should look like this:
[backend header]
[private header] [shared header] [metadata] [data]

shm_attach:
First use MapShared to map the first 4KB of the fd_.
This will be header_.
Get the size of data from the data from the header and add 2*kBackendHeaderSize.
Use MapMixed for the remaining.

Add priv_header_off_ to 
data_ - ptr is wrong. 

@CLAUDE.md

The layout should be like this
header_: [backend header]
region: [private header] [shared header] [metadata] [data]
region is the return value of the mixed map.
priv_header_off_ should be (data - region).

private header is kBackendHeaderSize.
shared header is kBackendHeaderSize.

Add priv_header_off_ to the backend header.
Do not recalculate in shm_attach.

@CLAUDE.md

# Memory backend layout

MemoryBackendHeader needs to store the following:
```
  size_t md_size_;         // Aligned metadata size (4KB aligned)
  MemoryBackendId id_;
  bitfield64_t flags_;
  size_t custom_header_size_;  // The size of the custom header
  size_t backend_size_;    // Total size of region_
  size_t data_size_;       // Remaining size of data_
  int data_id_;            // Device ID for the data buffer (GPU ID, etc.)
  size_t priv_header_off_; // Offset from data_ back to start of private header
```

MemoryBackend needs to store those, in addition to various pointers:
```
char *md_;
char *region_;
char *data_;
```

In fact, MemoryBackend should just inherit MemoryBackendHeader to make this easier.

Every MemoryBackend has the following layout:
md_: [backend header]
region_: [private header (4KB)] [shared header (4KB)] [data]

GetPrivateHeader:

GetPrivateHeader(data): (data - priv_header_off_)
GetSharedHeader(data): GetPrivateHeader(data) + kBackendHeaderSize
GetCustomHeader(data): GetSharedHeader(data) + kBackendHeaderSize

GetPrivateHeader(): GetPrivateHeader(data_)
GetSharedHeader(): GetSharedHeader(data_)
GetCustomHeader(): GetCustomHeader(data_)

# PosixShmMmap

shm_init(url, backend_size, custom_header_size):
1. header_: Use MapShared to map the first 4KB of the fd_.
2. region_: Use MapMixed for backend_size.
3. Partition the region_ as described.
4. Calaculate priv_header_off: (data_ - region_)
5. Calculate data_size_: (backend_size_ - priv_header_off)

shm_attach(url):
1. header_: First use MapShared to map the first 4KB of the fd_.
2. Get backend_size_ from the header
3. region_: Use MapMixed for backend_size_.
4. Partition the region_ as described. Each 
