@CLAUDE.md

# ShmContainer
Implement a base class called ShmContainer

```
template<typename AllocT>
class ShmContainer {
    OffsetPtr<void> this_;

    ShmContainer(AllocT *alloc) {
        this_ = OffsetPtr<void>(size_t((char*)this - (char*)alloc))
    }

    AllocT* GetAllocator() {
        return (AllocT*)((char*)this - this_);
    }
}

// Some compile-time macro to detect if T inherits from ShmContainer.
// We may need ShmContainer to have some additional type or something to detect this
#define IS_SHM_CONTAINER(T) 
```

# Vector

Implement a shared-memory vector and iterators for it.
It should implement most methods the same as you would expect.
Handle piece-of-data (POD) types differently from classes. 

```
namespace hshm::ipc {

template<typename T, typename AllocT>
class vector : public ShmContainer<AllocT> {
    size_t size_;
    size_t capacity_;
    OffsetPtr<T> data_;

    emplace_back(const T &value);
    emplace(T& value, int idx);
    replace(T& value, int off, int count);
    get(size_t idx);
    set(size_t idx, T& value)
    erase(int off, int count);
    clear();
}

}
```
