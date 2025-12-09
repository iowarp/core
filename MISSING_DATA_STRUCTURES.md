# Missing Data Structures - Compilation Requirements

This document outlines the data structures that are referenced in the codebase but are currently undefined or have been removed during the namespace refactoring. These need to be either restored or replaced with appropriate alternatives.

## 1. TaskLane

**Status**: Referenced but undefined (removed from task.h RunContext)

**Location**:
- `context-runtime/include/chimaera/worker.h:22` - WorkQueue typedef
- `context-runtime/include/chimaera/worker.h:140, 185, 191, 202, 274, 309, 389` - Method signatures and member variables
- `context-runtime/include/chimaera/ipc_manager.h:22` - WorkQueue typedef

**Purpose**: Represents a task execution lane in the worker queue system. TaskLane was used to route tasks to specific workers and manage per-lane state.

**Required Functionality**:
- Should be a pointer type that can be stored in `mpsc_queue`
- Must integrate with the task routing system
- Used in `WorkQueue = chi::ipc::mpsc_queue<TaskLane*>` typedef

**Current Workaround**:
- Temporarily use `void*` or `hipc::FullPtr<void>` as placeholder
- Full implementation requires understanding the task routing and scheduling system

---

## 2. ext_ring_buffer<T>

**Status**: Referenced but undefined

**Location**:
- `context-runtime/include/chimaera/worker.h:244, 252` - Method parameters
- `context-runtime/include/chimaera/worker.h:393, 403, 413` - Member variables (stack_cache_, blocked_queues_, periodic_queues_)

**Purpose**: Extended ring buffer for managing blocked/periodic task contexts. Used for queue management in worker threads.

**Required Functionality**:
- Template container supporting efficient queue operations
- Must support storing `RunContext*` and `StackAndContext` objects
- Should be thread-safe or use appropriate synchronization
- Size: Must handle NUM_BLOCKED_QUEUES and NUM_PERIODIC_QUEUES arrays

**Current Workaround**:
- Replace with `std::vector<T>` or `std::deque<T>` temporarily
- Full implementation requires custom ring buffer with extended functionality (e.g., RDMA support)

---

## 3. WorkQueue Type Definition

**Status**: Undefined - depends on TaskLane

**Location**:
- `context-runtime/include/chimaera/worker.h:22`
- `context-runtime/include/chimaera/ipc_manager.h:22`

**Current Definition Attempt**:
```cpp
using WorkQueue = chi::ipc::mpsc_queue<hipc::TypedPointer<TaskLane>>;
```

**Issues**:
- `hipc::TypedPointer` doesn't exist (removed in refactoring)
- `TaskLane` is undefined
- `chi::ipc::mpsc_queue` requires proper template parameters

**Required Replacement**:
```cpp
// Option 1: Generic placeholder
using WorkQueue = chi::ipc::mpsc_queue<hipc::FullPtr<void>>;

// Option 2: When TaskLane is properly defined
using WorkQueue = chi::ipc::mpsc_queue<TaskLane*>;
```

---

## 4. Allocator Types (removed hipc::MallocAllocator)

**Status**: Removed from refactoring

**Location**:
- `context-runtime/include/chimaera/types.h:272-274` - Macro definitions
  - `CHI_MAIN_ALLOC_T`
  - `CHI_CDATA_ALLOC_T`
  - `CHI_RDATA_ALLOC_T`
- `context-runtime/include/chimaera/ipc_manager.h:197, 402-404` - Usage in method and member declarations

**Current Broken Definitions**:
```cpp
#define CHI_MAIN_ALLOC_T hipc::MallocAllocator
#define CHI_CDATA_ALLOC_T hipc::MallocAllocator
#define CHI_RDATA_ALLOC_T hipc::MallocAllocator
```

**Required Replacement Options**:
```cpp
// Option 1: Using new HSHM allocators
#define CHI_MAIN_ALLOC_T hshm::heap::HeapAllocator
#define CHI_CDATA_ALLOC_T hshm::heap::HeapAllocator
#define CHI_RDATA_ALLOC_T hshm::heap::HeapAllocator

// Option 2: Generic allocator base class
#define CHI_MAIN_ALLOC_T hipc::BaseAllocator
#define CHI_CDATA_ALLOC_T hipc::BaseAllocator
#define CHI_RDATA_ALLOC_T hipc::BaseAllocator
```

**Purpose**:
- Main allocator: For pool and container management
- Client data allocator: For client-side data structures
- Runtime data allocator: For runtime-specific allocations

---

## 5. hipc::Pointer (now hipc::ShmPointer)

**Status**: Renamed in refactoring

**Location**:
- `context-runtime/include/chimaera/task_archives.h:139, 237, 354, 436, 583, 755` - bulk() method parameters
- `context-runtime/include/chimaera/task_archives.h:765, 776` - GetNull() calls
- `context-runtime/include/chimaera/ipc_manager.h:147` - FreeBuffer() method parameter

**Required Changes**:
- Replace `hipc::Pointer ptr` with `hipc::ShmPointer ptr`
- Replace `hipc::Pointer::GetNull()` with `hipc::ShmPointer()` (default constructor)

---

## 6. hipc::CtxAllocator<T>

**Status**: Undefined (likely removed or renamed)

**Location**:
- `context-runtime/include/chimaera/container.h:165` - GetAllocator() return type

**Current Broken Code**:
```cpp
hipc::CtxAllocator<CHI_MAIN_ALLOC_T> GetAllocator() const {
```

**Required Replacement**:
```cpp
// Option 1: Use the allocator type directly
CHI_MAIN_ALLOC_T* GetAllocator() const {

// Option 2: Generic allocator wrapper
hipc::BaseAllocator* GetAllocator() const {
```

---

## 7. hipc::delay_ar<T> Template

**Status**: Undefined (likely removed or renamed)

**Location**:
- `context-runtime/include/chimaera/ipc_manager.h:29` - TaskQueue storage
- `context-runtime/include/chimaera/ipc_manager.h:31` - WorkQueue vector storage

**Current Broken Code**:
```cpp
hipc::delay_ar<TaskQueue>
hipc::delay_ar<chi::ipc::vector<WorkQueue>>
```

**Purpose**: Lazy allocation/initialization template wrapper

**Required Replacement Options**:
```cpp
// Option 1: Direct storage
TaskQueue task_queue_;
chi::ipc::vector<WorkQueue> worker_queues_;

// Option 2: Pointer storage with lazy init
TaskQueue* task_queue_;
chi::ipc::vector<WorkQueue>* worker_queues_;

// Option 3: std:: fallback
std::unique_ptr<TaskQueue> task_queue_;
std::unique_ptr<chi::ipc::vector<WorkQueue>> worker_queues_;
```

---

## Summary of Required Actions

### Immediate (to get compilation passing):

1. **Replace hipc::Pointer with hipc::ShmPointer** in task_archives.h
   - 8 occurrences identified
   - Use `hipc::ShmPointer()` for GetNull() calls

2. **Define WorkQueue placeholder** in worker.h and ipc_manager.h
   - Use generic type: `using WorkQueue = chi::ipc::mpsc_queue<hipc::FullPtr<void>>;`

3. **Fix allocator macros** in types.h
   - Replace `hipc::MallocAllocator` with `hshm::heap::HeapAllocator` or `hipc::BaseAllocator`

4. **Comment out TaskLane references** in worker.h
   - Keep the method signatures but comment out implementations
   - TaskLane needs full design before implementation

5. **Replace ext_ring_buffer with std::vector** in worker.h
   - Temporary solution to get compilation working

6. **Fix allocator references** in container.h and ipc_manager.h
   - Use direct pointer types instead of wrapper templates

### Medium-term (proper implementation):

1. **Design and implement TaskLane**
   - Define the structure for task execution lanes
   - Integrate with worker scheduling system
   - Support task routing based on lane assignment

2. **Implement ext_ring_buffer template**
   - Custom ring buffer with proper synchronization
   - Support for extensible buffer operations
   - Optional RDMA memory registration support

3. **Restore hipc::delay_ar or design replacement**
   - Lazy initialization wrapper template
   - Or restructure to use explicit initialization patterns

4. **Review and finalize allocator architecture**
   - Ensure proper allocator selection for different use cases
   - Document allocator responsibilities

---

## File Summary

| File | Status | Required Fixes |
|------|--------|-----------------|
| task.h | ✅ Completed | String template parameters fixed |
| task_archives.h | ⚠️ In Progress | Replace 8 hipc::Pointer → hipc::ShmPointer |
| worker.h | ⚠️ Blocked | Need TaskLane, ext_ring_buffer, WorkQueue definitions |
| ipc_manager.h | ⚠️ Blocked | Need WorkQueue, delay_ar, allocator macros |
| container.h | ⚠️ Blocked | Need allocator type fixes |
| types.h | ⚠️ Blocked | Need allocator macro definitions |
| zmq_transport.h | ✅ Completed | hipc::Pointer → hipc::ShmPointer fixed |
| lightbeam.h | ✅ Completed | hipc::Pointer → hipc::ShmPointer fixed |

