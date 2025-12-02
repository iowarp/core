# Mixed Private/Shared Memory Mapping

## Overview

All `MemoryBackend` implementations now support mixed private/shared memory mapping, creating a contiguous virtual memory region where:
- **First kBackendPrivate (16KB)**: Private (process-local, not shared between processes)
- **Remaining**: Shared (synchronized across all processes)

This is implemented via the `SystemInfo::MapMixedMemory()` method and is available through the base `MemoryBackend` class.

## Memory Layout

```
┌──────────────────────┬────────────────────────────────────┐
│  kBackendPrivate     │        SHARED REGION               │
│  (16KB per-process)  │     (inter-process)                │
└──────────────────────┴────────────────────────────────────┘
^                      ^                                     ^
ptr                    ptr + kBackendPrivate (data_)         ptr + kBackendPrivate + size
```

**Note**: `data_` always points to the start of the SHARED region, not the beginning of the mapped memory.

## API

### SystemInfo::MapMixedMemory

```cpp
void *SystemInfo::MapMixedMemory(const File &fd,
                                  size_t private_size,
                                  size_t shared_size,
                                  i64 shared_offset = 0);
```

**Parameters:**
- `fd`: File descriptor for shared memory (from `shm_open`)
- `private_size`: Size of private region at the beginning
- `shared_size`: Size of shared region following the private region
- `shared_offset`: Offset into fd for the shared mapping (usually 0)

**Returns:** Pointer to the beginning of the contiguous region, or nullptr on failure

**Note:** The entire region must be unmapped with a single `UnmapMemory` call using `total_size = private_size + shared_size`

### MemoryBackend Methods (Available to All Backends)

```cpp
/**
 * Get pointer to the private region (kBackendPrivate bytes before data_)
 *
 * This region is process-local and not shared between processes.
 * Each process that attaches gets its own independent copy.
 * Useful for thread-local storage and process-specific metadata.
 *
 * @return Pointer to the kBackendPrivate-byte private region, or nullptr if data_ is null
 */
char *GetPrivateRegion();

/**
 * Get size of the private region
 * @return Size of private region (always kBackendPrivate = 16KB)
 */
static constexpr size_t GetPrivateRegionSize();
```

**Important**: The private region is always located at `(data_ - kBackendPrivate)`.

### PosixShmMmap Methods

```cpp
// Initialize with kBackendPrivate private + size shared
bool shm_init(const MemoryBackendId &backend_id, size_t size, const std::string &url);

// Attach with same layout
bool shm_attach(const std::string &url);
```

## Use Cases

### 1. Process-Local TLS Pointers

```cpp
// Works with any MemoryBackend implementation
PosixShmMmap backend;
backend.shm_init(id, 1_GB, "/my_shm");

// Store process-local data in private region (available in ALL backends)
char *private_region = backend.GetPrivateRegion();
struct ProcessLocalData {
  pthread_key_t tls_key;
  void *cache_ptr;
  // ... other process-local data
};
auto *local_data = reinterpret_cast<ProcessLocalData*>(private_region);
```

### 2. Per-Process Allocator Metadata

```cpp
// Each process can maintain its own metadata without locks
struct PrivateMetadata {
  size_t local_allocations;
  void *thread_cache[16];
  pid_t process_id;
};

auto *metadata = reinterpret_cast<PrivateMetadata*>(backend.GetPrivateRegion());
metadata->process_id = getpid();
```

### 3. Process-Specific State

```cpp
// Store process-specific state that doesn't need synchronization
struct ProcessState {
  std::atomic<uint64_t> local_counter;
  void *process_specific_ptr;
  char scratch_space[8192];
};
```

## Implementation Details

The mixed mapping is created using two `mmap` calls:

1. **Reserve Address Space**: `mmap` the entire region as `MAP_PRIVATE | MAP_ANONYMOUS`
2. **Remap Shared Portion**: Use `MAP_FIXED` to replace the shared portion with `MAP_SHARED` from the fd

This guarantees:
- ✅ Contiguous virtual memory addresses
- ✅ Private region is truly process-local (each process gets independent copy)
- ✅ Shared region is synchronized across all processes
- ✅ Works correctly with `fork()` (child gets copy of private region)

## Platform Support

- **Linux/Unix**: Fully supported using `mmap64` with `MAP_FIXED`
- **Windows**: Falls back to shared-only mapping (no private region)

## Example: Multi-Process Test

```cpp
// Parent process: Initialize
PosixShmMmap backend;
backend.shm_init(MemoryBackendId(0, 0), 1_MB, "/test_shm");

char *private_region = backend.GetPrivateRegion();
char *shared_region = backend.data_;

strcpy(private_region, "PARENT_PRIVATE");
strcpy(shared_region, "PARENT_SHARED");

// Child process: Attach
if (fork() == 0) {
  PosixShmMmap child_backend;
  child_backend.shm_attach("/test_shm");

  char *child_private = child_backend.GetPrivateRegion();
  char *child_shared = child_backend.data_;

  // child_private is DIFFERENT from parent's (process-local)
  // child_shared is SAME as parent's (inter-process)

  strcpy(child_private, "CHILD_PRIVATE");
  strcpy(child_shared, "CHILD_SHARED");
}
```

## Global Constant

```cpp
namespace hshm::ipc {
  // Global constant for private memory region size
  static constexpr size_t kBackendPrivate = 16 * 1024;  // 16KB
}
```

This constant is defined in `memory_backend.h` and is used by all backend implementations.

## Caveats

1. **Fixed Size**: The private region is always `kBackendPrivate` (16KB, page-aligned)
2. **data_ Pointer**: Always points to the SHARED region, not the start of the mapping
3. **Access Pattern**: Use `backend.GetPrivateRegion()` to access private memory
4. **Cleanup**: Use `shm_destroy()` only from one process (typically rank 0)
5. **Fork Behavior**: Child processes get a copy of the private region (COW)
6. **Windows**: Private region feature not supported on Windows
