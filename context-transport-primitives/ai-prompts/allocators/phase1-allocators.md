@CLAUDE.md 

# Eliminate factory pattern entirely for memory objects
Remove AllocatorType and MemoryBackendType enums from the code.

# Update FullPtr
Update FullPtr to remove the following constructors:
```cpp
  /** SHM constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const PointerT &shm);

  /** Private half constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const T *ptr);

  /** Private half + alloc constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit FullPtr(hipc::Allocator *alloc, const T *ptr);

  /** Shared half + alloc constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit FullPtr(hipc::Allocator *alloc,
                                         const OffsetPointer &shm);
```

Merge memory.h into allocator.h. Remove all references to memory.h.

Remove Convert from allocator.h. After , let's implement the following
FullPtr constructors:

```
  /** Private half + alloc constructor (in memory_manager.h) */
  template<typename AllocT>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc, const T *ptr) {
    if (ctx_alloc->ContainsPtr(ptr)) {
      shm_.off_ = (size_t)(ptr - (*ctx_alloc).buffer_);
      shm_.alloc_id_ = ctx_alloc->alloc_id_;
      ptr_ = ptr;
    } else {
        HSHM_THROW_ERROR(PTR_NOT_IN_ALLOCATOR);
    }
  }

  /** Shared half + alloc constructor (in memory_manager.h) */
  template<typename AllocT, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc,
                                         const OffsetPointer<ATOMIC> &shm) {
    if (ctx_alloc->ContainsPtr(shm)) {
      shm_.off_ = shm;
      shm_.alloc_id_ = ctx_alloc->alloc_id_;
      ptr_ = ctx_alloc->buffer_ + shm;
    } else {
        HSHM_THROW_ERROR(PTR_NOT_IN_ALLOCATOR);
    }
 }

 /** Shared half + alloc constructor (in memory_manager.h) */
  template<typename AllocT, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc,
                                         const Pointer<ATOMIC> &shm) {
    if (ctx_alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.off_;
      shm_.alloc_id_ = shm.alloc_id_;
      ptr_ = ctx_alloc->buffer_ + shm.off_;
    } else {
        HSHM_THROW_ERROR(PTR_NOT_IN_ALLOCATOR);
    }
 }
```

You will need to implement overrides for ContainsPtr for the OffsetPointer and Pointer cases.
they should simply check to see if the offset is less than the size of the buffer.