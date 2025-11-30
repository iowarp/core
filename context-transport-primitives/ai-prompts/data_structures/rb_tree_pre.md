

# Red Black Tree Preallocated

Instead of RBTree taking as input KeyT, I want it to take as input a NodeT. Assume that NodeT inherits from rb_node. Also assume that NodeT has comparison operators and NodeT::key variable.

Create data structure in context-transport-primitives/include/hermes_shm/data_structures/ipc/rb_tree_pre.h

This data structure does not perform allocations. It assumes the entries are pre-allocated.
this is a shared-memory compatible data structure. 

Build a unit test under context-transport-primitives/test/unit/data_structures for this class.
The unit test can use the ArenaAllocator over a MallocBackend.

Template parameters: 
1. KeyT: The type of the key used for all emplace operations.

## class rb_tree

template<typename KeyT>
class rb_tree {
    size_t size;
    rb_node head_;
}

## class rb_node

All entries must inherit from this.
```
template<typename KeyT>
class rb_node {
    Key key_;
    OffsetPointer left_;
    OffsetPointer right_;
}
```

## emplace

### Parameters
1. Allocator *alloc (the allocator used for convert OffsetPointer to FullPtr)
2. FullPtr<rb_node<KeyT>> node (the node being emplaced)

### Implementation

The Key for the red-black algorithm is node->key_;
For traversing, use FullPtr(alloc, node->left_) or FullPtr(alloc, node->right_).
Follow the traditional RBTree implementation otherwise.

## pop

### Parameters
1. Allocator *alloc (the allocator used for convert OffsetPointer to FullPtr)
2. FullPtr<rb_node<KeyT>> node (the node being emplaced)

### Implementation

The Key for the red-black algorithm is node->key_;
For traversing, use FullPtr(alloc, node->left_) or FullPtr(alloc, node->right_).
Follow the traditional RBTree implementation otherwise.

