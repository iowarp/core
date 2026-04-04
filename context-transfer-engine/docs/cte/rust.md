# IOWarp CTE Rust Bindings API Documentation

Comprehensive API reference for the Rust bindings to IOWarp Context Transfer Engine (CTE), enabling Rust applications to interface with CTE for blob storage, retrieval, score adjustment, and telemetry collection.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Detailed Examples](#detailed-examples)
5. [Advanced Topics](#advanced-topics)
6. [Integration](#integration)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Introduction to Rust Bindings

The CTE Rust bindings provide a modern, idiomatic Rust API over the IOWarp Context Transfer Engine's C++ library. Built using the [`cxx`](https://github.com/dtolnay/cxx) crate, these bindings enable safe, zero-copy interoperability between Rust and C++ while maintaining thread safety guarantees.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Rust Application (async default)                            │
│                                                             │
│ let client = Client::new().await?;                          │
│ let tag = Tag::new("dataset").await?;                       │
│ tag.put_blob(...).await;                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Rust Bindings (wrp_cte)                                     │
│                                                             │
│ ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐   │
│ │ async.rs     │ │ sync.rs      │ │ types.rs           │   │
│ │ (default)    │ │ (optional)   │ │                    │   │
│ └──────┬───────┘ └──────┬───────┘ └────────────────────┘   │
          │                │                                   │
┌─────────▼────────────────▼──────────────────────────────────┐
│ CXX Bridge (ffi.rs)                                         │
│ Safe Rust/C++ FFI boundary                                  │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│ C++ Shim Layer (shim/shim.h, shim/shim.cc)                │
│ Wraps C++ CTE API for FFI                                   │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│ C++ CTE Library (libwrp_cte_core_client.so)               │
│ Provides: Client, Tag, blob operations, telemetry          │
└─────────────────────────────────────────────────────────────┘
```

### Feature Flags

TheBindings supports two feature flags:

| Feature | Default | Description |
|---------|---------|-------------|
| `async` | Yes | Async/await API using Tokio's `spawn_blocking` |
| `sync` | No | Synchronous (blocking) API |

**API Differences:**

- **Async API** (`feat ure = "async"`, default): Returns `Future`s, uses `tokio::task::spawn_blocking` for C++ calls
- **Sync API** (`feature = "sync"`): Blocking calls, simpler for debugging and single-threaded use

**Selector:**
```toml
# Async API (default)
[dependencies]
wrp-cte-rs = { path = "..." }

# Sync API only
[dependencies]
wrp-cte-rs = { path = "...", default-features = false, features = ["sync"] }

# Both APIs
[dependencies]
wrp-cte-rs = { path = "..." }
tokio = { version = "1.50", features = ["rt-multi-thread", "macros"] }
```

### Integration with CMake and Cargo

**CMake Integration:**
```bash
# Configure IOWarp Core with Rust support
cmake .. -DWRP_CORE_ENABLE_RUST=ON
make -j$(nproc)

# The Rust crate is automatically built with:
# - Proper RPATH configuration
# - All CTE dependencies linked
# - CXX bridge compiled
```

**Cargo.toml Setup:**
```toml
[dependencies]
wrp-cte-rs = { path = "/path/to/clio-core/context-transfer-engine/wrapper/rust" }
cxx = "1.0"
tokio = { version = "1.50", default-features = false, features = ["rt", "macros"] }

[build-dependencies]
cxx-build = "1.0"
```

---

## Quick Start

### Installation (CMake + Rust)

**Step 1: Configure CMake with Rust Support**

```bash
cd /path/to/clio-core
mkdir -p build && cd build

cmake .. \
  -DWRP_CORE_ENABLE_RUNTIME=ON \
  -DWRP_CORE_ENABLE_CTE=ON \
  -DWRP_CORE_ENABLE_RUST=ON
```

**Step 2: Build All Components**

```bash
make -j$(nproc)

# Install to system (optional, requires sudo)
sudo make install
```

**Step 3: Build Rust Crate**

The Rust bindings are built as part of the CMake build. The crate directory is:
```
context-transfer-engine/wrapper/rust/
```

### Basic Usage Example

**Async API (default):**
```rust
use wrp_cte::{Client, Tag, CteTagId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize and create client
    let client = Client::new().await?;
    
    // Create or open a tag
    let tag = Tag::new("my_dataset").await?;
    
    // Store data with placement score
    tag.put_blob("data.bin".to_string(), b"Hello, CTE!".to_vec(), 0, 1.0)
        .await?;
    
    // Retrieve data
    let size = tag.get_blob_size("data.bin").await?;
    let data = tag.get_blob("data.bin".to_string(), size, 0).await?;
    println!("Retrieved: {}", String::from_utf8_lossy(&data));
    
    // Get telemetry entries
    let telemetry = client.poll_telemetry(0).await?;
    println!("Got {} telemetry entries", telemetry.len());
    
    // Adjust blob placement score
    tag.reorganize_blob("data.bin".to_string(), 0.5).await?;
    
    Ok(())
}
```

**Sync API:**
```rust
use wrp_cte::sync::{init, Client, Tag};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CTE (embedded runtime)
    init("")?;
    
    // Create client and tag
    let client = Client::new()?;
    let tag = Tag::new("my_dataset");
    
    // Store data
    tag.put_blob("data.bin", b"Hello, CTE!");
    
    // Retrieve data
    let size = tag.get_blob_size("data.bin")?;
    let data = tag.get_blob("data.bin", size, 0)?;
    println!("Retrieved: {}", String::from_utf8_lossy(&data));
    
    Ok(())
}
```

---

## API Reference

### Core Types

#### `CteTagId`

Unique identifier for tags, blobs, and pools (8-byte layout: `major: u32 + minor: u32`).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CteTagId {
    pub major: u32,
    pub minor: u32,
}
```

**Methods:**
- `CteTagId::new(major: u32, minor: u32)` - Create new ID
- `CteTagId::null()` - Create null/invalid ID
- `id.is_null()` - Check if ID is null
- `id.to_u64()` - Convert to u64 for storage
- `CteTagId::from_u64(value: u64)` - Create from u64

**Example:**
```rust
use wrp_cte::CteTagId;

// Create from components
let tag_id = CteTagId::new(1, 2);

// Convert to u64
let as_u64 = tag_id.to_u64(); // 4294967298 (0x0000000100000002)

// Convert from u64
let from_u64 = CteTagId::from_u64(as_u64);
assert_eq!(from_u64, tag_id);

// Null ID
let null = CteTagId::null();
assert!(null.is_null());
```

#### `CteTelemetry`

Telemetry entry for monitoring CTE operations.

**Layout (52 bytes per entry):**
- `op`: u32 - Operation type
- `off`: u64 - Offset in blob
- `size`: u64 - Operation size
- `tag_id`: CteTagId - Associated tag (8 bytes)
- `mod_time`: SteadyTime - Modification time
- `read_time`: SteadyTime - Read time
- `logical_time`: u64 - Logical time counter

```rust
#[derive(Debug, Clone)]
pub struct CteTelemetry {
    pub op: CteOp,
    pub off: u64,
    pub size: u64,
    pub tag_id: CteTagId,
    pub mod_time: SteadyTime,
    pub read_time: SteadyTime,
    pub logical_time: u64,
}
```

#### `CteOp`

Operation types for CTE telemetry.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CteOp {
    PutBlob = 0,
    GetBlob = 1,
    DelBlob = 2,
    GetOrCreateTag = 3,
    DelTag = 4,
    GetTagSize = 5,
}
```

#### `SteadyTime`

Monotonic clock time point (nanosecond precision).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SteadyTime {
    pub nanos: i64,
}

impl SteadyTime {
    pub fn from_nanos(nanos: i64) -> Self
    pub fn duration_since(&self, earlier: &SteadyTime) -> std::time::Duration
    pub fn elapsed_from(&self, earlier: &SteadyTime) -> std::time::Duration
}
```

#### `PoolQuery`

Pool routing strategies.

```rust
#[derive(Debug, Clone, Copy)]
pub enum PoolQuery {
    Broadcast { net_timeout: f32 },
    Dynamic { net_timeout: f32 },
    Local,
}

impl PoolQuery {
    pub fn broadcast(timeout: f32) -> Self
    pub fn dynamic(timeout: f32) -> Self
    pub fn local() -> Self
    pub fn net_timeout(&self) -> f32
}
```

**Values:**
- `PoolQuery::Local()` - Execute on current node only (no network)
- `PoolQuery::dynamic(timeout)` - Automatic optimization based on load
- `PoolQuery::broadcast(timeout)` - Send to all nodes

### Error Handling

#### `CteError`

Detailed error enum with specific failure modes.

```rust
#[derive(Debug)]
pub enum CteError {
    /// Initialization failed
    InitFailed { reason: String },
    
    /// Pool operations failed
    PoolCreationFailed { message: String },
    PoolNotFound { pool_id: String },
    
    /// Tag operations failed
    TagNotFound { name: String },
    TagAlreadyExists { name: String },
    
    /// Blob operations failed
    BlobNotFound { tag: String, blob: String },
    BlobIOError { message: String },
    
    /// Storage target operations failed
    TargetRegistrationFailed { path: String },
    TargetNotFound { path: String },
    
    /// Telemetry unavailable
    TelemetryUnavailable,
    
    /// Invalid parameter provided
    InvalidParameter { message: String },
    
    /// C++ runtime returned error code
    RuntimeError { code: u32, message: String },
    
    /// Operation timed out
    Timeout,
    
    /// FFI bridge error
    FfiError { message: String },
    
    /// I/O error wrapper
    IoError { message: String },
    
    /// Feature not yet implemented
    NotImplemented { feature: String, reason: String },
}
```

**Type Alias:**
```rust
pub type CteResult<T> = Result<T, CteError>;
```

**Example:**
```rust
use wrp_cte::{Client, CteError};

match Client::new().await {
    Ok(client) => {
        // Use client
    }
    Err(CteError::InitFailed { reason }) => {
        eprintln!("CTE init failed: {}", reason);
        std::process::exit(1);
    }
    Err(CteError::RuntimeError { code, message }) => {
        eprintln!("CTE runtime error {}: {}", code, message);
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

### Client API

#### `Client` (Async API)

Provides async methods for client-level operations.

```rust
pub struct Client {
    _marker: std::marker::PhantomData<()>,
}

impl Client {
    /// Create a new CTE client
    pub async fn new() -> CteResult<Self>
    
    /// Poll telemetry log from CTE
    pub async fn poll_telemetry(&self, min_time: u64) -> CteResult<Vec<CteTelemetry>>
    
    /// Reorganize a blob (change placement score)
    pub async fn reorganize_blob(
        &self,
        tag_id: CteTagId,
        name: String,
        score: f32,
    ) -> CteResult<()>
    
    /// Delete a blob
    pub async fn del_blob(&self, tag_id: CteTagId, name: String) -> CteResult<()>
}
```

**Example:**
```rust
use wrp_cte::Client;

let client = Client::new().await?;

// Get all telemetry entries
let telemetry = client.poll_telemetry(0).await?;
for entry in telemetry {
    println!("Op: {:?}, Size: {} bytes", entry.op, entry.size);
}

// Reorganize blob (change placement score)
client.reorganize_blob(
    CteTagId::new(1, 2),
    "data.bin".to_string(),
    0.5,
).await?;

// Delete blob
client.del_blob(
    CteTagId::new(1, 2),
    "old_data.bin".to_string(),
).await?;
```

#### `Client` (Sync API)

Blocking wrapper around client operations.

```rust
impl Client {
    pub fn new() -> CteResult<Self>
    pub fn poll_telemetry(&self, min_time: u64) -> CteResult<Vec<CteTelemetry>>
    pub fn reorganize_blob(&self, tag_id: CteTagId, name: &str, score: f32) -> CteResult<()>
    pub fn del_blob(&self, tag_id: CteTagId, name: &str) -> CteResult<()>
}
```

### Tag API

#### `Tag` (Async API)

Provides async methods for tag/blob operations.

```rust
pub struct Tag {
    inner: Arc<std::sync::Mutex<SendableTag>>,
}

impl Tag {
    /// Create or get a tag by name
    pub async fn new(name: &str) -> CteResult<Self>
    
    /// Open an existing tag by ID
    pub async fn from_id(id: CteTagId) -> CteResult<Self>
    
    /// Get the tag ID
    pub async fn get_id(&self) -> CteResult<CteTagId>
    
    /// Get the placement score of a blob
    pub async fn get_blob_score(&self, name: &str) -> CteResult<f32>
    
    /// Reorganize a blob (change placement score)
    pub async fn reorganize_blob(&self, name: String, score: f32) -> CteResult<()>
    
    /// Write data into a blob
    pub async fn put_blob(&self, name: String, data: Vec<u8>, offset: u64, score: f32) -> CteResult<()>
    
    /// Read data from a blob
    pub async fn get_blob(&self, name: String, size: u64, offset: u64) -> CteResult<Vec<u8>>
    
    /// Get the size of a blob
    pub async fn get_blob_size(&self, name: &str) -> CteResult<u64>
    
    /// List all blobs in this tag
    pub async fn get_contained_blobs(&self) -> CteResult<Vec<String>>
}
```

#### `Tag` (Sync API)

Blocking wrapper around tag operations.

```rust
impl Tag {
    pub fn new(name: &str) -> Self
    pub fn from_id(id: CteTagId) -> Self
    pub fn get_blob_score(&self, name: &str) -> CteResult<f32>
    pub fn reorganize_blob(&self, name: &str, score: f32) -> CteResult<()>
    pub fn put_blob_with_options(&self, name: &str, data: &[u8], offset: u64, score: f32) -> CteResult<()>
    pub fn put_blob(&self, name: &str, data: &[u8])
    pub fn get_blob(&self, name: &str, size: u64, offset: u64) -> CteResult<Vec<u8>>
    pub fn get_blob_size(&self, name: &str) -> CteResult<u64>
    pub fn get_contained_blobs(&self) -> Vec<String>
    pub fn id(&self) -> CteTagId
}
```

**Note:** The `put_blob()` convenience method logs a warning and panics on validation errors. For production code, prefer `put_blob_with_options()` with explicit error handling.

### Initialization

#### Async API

Automatic initialization when creating the first `Client`:
```rust
let client = Client::new().await?; // Automatically initializes CTE
```

#### Sync API

Explicit initialization with embedded runtime:
```rust
use wrp_cte::sync::init;

// Initialize with embedded runtime (CHI_WITH_RUNTIME=1)
init("")?; // "" for default config, or path to config file
```

**Environment Variables:**
- `CHI_WITH_RUNTIME=1` - Start embedded CTE runtime
- `CHI_CONFIG_PATH` - Path to configuration file
- `CHI_IPC_MODE` - IPC transport mode: `SHM`, `TCP`, `IPC` (default: `TCP`)

---

## Detailed Examples

### Creating Tags and Blobs

**Async API:**
```rust
use wrp_cte::{Client, Tag, CteTagId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new().await?;
    
    // Create tag
    let tag = Tag::new("my_dataset").await?;
    
    // Store multiple blobs
    let blobs = vec![
        ("data1.bin", b"First blob data"),
        ("data2.bin", b"Second blob data"),
        ("data3.bin", b"Third blob data"),
    ];
    
    for (name, data) in blobs {
        tag.put_blob(
            name.to_string(),
            data.to_vec(),
            0,      // offset
            1.0,    // score (placement priority)
        ).await?;
    }
    
    // Retrieve all blob names
    let blob_names = tag.get_contained_blobs().await?;
    println!("Blobs in tag: {:?}", blob_names);
    
    Ok(())
}
```

**Sync API:**
```rust
use wrp_cte::sync::{init, Tag};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init("")?;
    
    let tag = Tag::new("my_dataset");
    
    // Store blob
    tag.put_blob_with_options("data.bin", b"Blob data", 0, 1.0)
        .expect("put_blob failed");
    
    // Store with defaults (offset=0, score=1.0)
    tag.put_blob("simple.bin", b"Simple data");
    
    Ok(())
}
```

### Streaming Telemetry

**Async API:**
```rust
use wrp_cte::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new().await?;
    
    // Poll telemetry for entries after min_time
    let telemetry = client.poll_telemetry(0).await?; // 0 for all entries
    
    println!("Telemetry entries:");
    for entry in telemetry {
        println!("  {:?}: {} bytes at offset {}", entry.op, entry.size, entry.off);
        println!("    Tag: {}.{}", entry.tag_id.major, entry.tag_id.minor);
        println!("    Logical time: {}", entry.logical_time);
    }
    
    // Filter by specific operation type
    let put_ops: Vec<_> = telemetry.iter()
        .filter(|e| e.op == CteOp::PutBlob)
        .collect();
    
    println!("Put operations: {}", put_ops.len());
    
    Ok(())
}
```

**Sync API:**
```rust
use wrp_cte::sync::{init, Client};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init("")?;
    
    let client = Client::new()?;
    let telemetry = client.poll_telemetry(0)?;
    
    for entry in telemetry {
        println!("Op: {:?}, Size: {}", entry.op, entry.size);
    }
    
    Ok(())
}
```

### Score Management

**Async API:**
```rust
use wrp_cte::{Client, Tag, CteTagId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new().await?;
    let tag = Tag::new("data").await?;
    
    // Store blob with high score (fast tier)
    tag.put_blob("hot_data.bin".to_string(), b"Hot data".to_vec(), 0, 1.0)
        .await?;
    
    // Store blob with low score (cold tier)
    tag.put_blob("cold_data.bin".to_string(), b"Cold data".to_vec(), 0, 0.1)
        .await?;
    
    // Get blob scores
    let hot_score = tag.get_blob_score("hot_data.bin").await?;
    let cold_score = tag.get_blob_score("cold_data.bin").await?;
    
    println!("Hot data score: {}", hot_score);
    println!("Cold data score: {}", cold_score);
    
    // Change hot data to cold (low score triggers migration to slower tier)
    tag.reorganize_blob("hot_data.bin".to_string(), 0.1).await?;
    
    // Change cold data to hot (high score triggers migration to faster tier)
    tag.reorganize_blob("cold_data.bin".to_string(), 1.0).await?;
    
    // Use client API
    client.reorganize_blob(
        tag.get_id().await?,
        "another_blob.bin".to_string(),
        0.5,  // Neutral score
    ).await?;
    
    Ok(())
}
```

**Sync API:**
```rust
use wrp_cte::sync::{init, Tag};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init("")?;
    
    let tag = Tag::new("data");
    
    // Store with low score (cold tier)
    tag.put_blob_with_options("cold_data.bin", b"Cold data", 0, 0.1)
        .expect("put_blob failed");
    
    // Get score
    let score = tag.get_blob_score("cold_data.bin")
        .expect("get_blob_score failed");
    println!("Score: {}", score);
    
    // Change score to hot tier
    tag.reorganize_blob("cold_data.bin", 1.0)
        .expect("reorganize failed");
    
    Ok(())
}
```

### Error Handling Patterns

**Pattern 1: Graceful Error Handling**
```rust
use wrp_cte::{Client, CteError};

let result = async {
    let client = Client::new().await?;
    // ... operations
    Ok::<(), CteError>(())
};

match result {
    Ok(()) => println!("Success"),
    Err(CteError::InitFailed { reason }) => {
        eprintln!("Initialization failed: {}", reason);
        std::process::exit(1);
    }
    Err(CteError::RuntimeError { code, message }) => {
        eprintln!("Runtime error {} (code {}): {}", message, code, reason);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

**Pattern 2: Retry on Transient Errors**
```rust
use wrp_cte::{Client, CteError};

async fn retry_with_backoff<F, T>(f: F, max_retries: usize) -> Result<T, CteError>
where
    F: Fn() -> futures::future::BoxFuture<'static, Result<T, CteError>>,
{
    let mut retries = 0;
    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(CteError::RuntimeError { code, .. }) if code == 2 => {
                // Tag not found - retry
                retries += 1;
                if retries >= max_retries {
                    return Err(CteError::RuntimeError {
                        code,
                        message: format!("Max retries ({}) reached", max_retries),
                    });
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

**Pattern 3: Validation Before FFI Calls**
```rust
use wrp_cte::sync::Tag;

let tag = Tag::new("data");

// Validate before making FFI calls
fn safe_put_blob(tag: &Tag, name: &str, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    if name.is_empty() {
        return Err("Blob name cannot be empty".into());
    }
    if data.is_empty() {
        return Err("Data cannot be empty".into());
    }
    
    tag.put_blob(name, data);
    Ok(())
}
```

---

## Advanced Topics

### Async vs Sync APIs

**When to Use Async API:**
- Building async applications with Tokio
- High-concurrency scenarios
- When integrating with other async libraries
- Production services requiring scalability

**When to Use Sync API:**
- Simple command-line tools
- Debugging and development
- Single-threaded applications
- When C++ calls need to block the current thread

**Performance Considerations:**

**Async API:**
- Uses `tokio::task::spawn_blocking` for C++ calls
- Overhead of thread pool scheduling
- Can handle many concurrent operations
- Thread-safe access via mutex-protected Tag

**Sync API:**
- Direct blocking FFI calls
- No thread pool overhead
- Blocks current thread during C++ calls
- Simpler for debugging

### Thread Safety Guarantees

**Global Initialization:**
The sync API uses `OnceLock` for thread-safe initialization:
```rust
static INIT_RESULT: OnceLock<CteResult<()>> = OnceLock::new();
```
- Only one thread performs initialization
- Other threads wait for initialization to complete
- Result is cached for all threads

**Async Tag Operations:**
Uses `Arc<Mutex<SendableTag>>` for thread-safe access:
```rust
pub struct Tag {
    inner: Arc<std::sync::Mutex<SendableTag>>,
}
```

**Safety Guarantees:**
1. **Mutex Synchronization**: Only one thread accesses Tag at a time
2. **spawn_blocking Isolation**: C++ calls run on dedicated blocking threads
3. **C++ Thread-Safety**: Underlying Tag class is designed for single-threaded operations
4. **No Interior Mutability**: No shared state that could cause data races

**Send Safety:**
- `SendableTag` and `SendableClient` implement `Send` for crossing thread boundaries
- FFI access is synchronized via `Arc<Mutex<_>>`
- C++ objects are properly destroyed in the same thread that created them

### FFI Safety Documentation

**CXX Bridge Design:**

The FFI boundary uses these patterns:

1. **Opaque Types**: C++ types (`Client`, `Tag`) are opaque from Rust's perspective
2. **Primitive Parameters**: All scalar types use C-compatible primitives
3. **Output Parameters**: Complex data passed through Rust-owned `Vec`s

**FFI Function Safety:**

```rust
// SAFETY: All parameters are primitives or borrowed strings
// Return values are primitives that can be freely copied
fn tag_get_blob_score(tag: &Tag, name: &str) -> f32;

// SAFETY: Buffer parameters use Vec<T> which cxx maps correctly to std::vector<T>
// C++ appends to output vectors, Rust owns the final data
fn tag_get_blob(tag: &Tag, name: &str, size: u64, offset: u64, out: &mut Vec<u8>);
```

**Memory Layout Guarantees:**
- cxx ensures identical memory layout for all types
- Primitive types have identical bit representations in Rust and C++
- `Vec<T>` maps to `std::vector<T>` correctly

### Memory Management

**C++ Object Lifecycle:**
```rust
// Factory returns UniquePtr which owns the C++ object
fn client_new() -> UniquePtr<Client>;
fn tag_new(name: &str) -> UniquePtr<Tag>;

//当 UniquePtr 被 drop 时，C++析构函数被调用
// drop(client) -> Client::~Client()
// drop(tag) -> Tag::~Tag()
```

**Buffer Management:**
```rust
// Output buffers are Rust-owned
fn tag_get_blob(..., out: &mut Vec<u8>);

// C++ appends to the buffer
// Rust owns the final Vec<u8>
```

**No Manual Memory Management:**
- Use RAII via `UniquePtr` and `Drop`
- Let Rust's ownership system handle cleanup
- No `free()` or manual deletion needed

---

## Integration

### CMake Configuration

**Enable Rust Bindings:**
```bash
cmake .. -DWRP_CORE_ENABLE_RUST=ON
```

**Rust Integration:**
```cmake
# Find the Rust crate (if built via CMake)
find_package(wrp_cte_core_rust REQUIRED)

# Link to Rust library
target_link_libraries(your_rust_binary wrp_cte_core_rust)
```

### Cargo.toml Setup

**Basic Configuration:**
```toml
[package]
name = "my-cte-app"
version = "0.1.0"
edition = "2021"

[dependencies]
wrp-cte-rs = { path = "/path/to/clio-core/context-transfer-engine/wrapper/rust" }
cxx = "1.0"
tokio = { version = "1.50", default-features = false, features = ["rt", "macros"] }

[build-dependencies]
cxx-build = "1.0"
```

**Build Script (build.rs):**
```rust
fn main() {
    // Ensure CTE libraries are buildable
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
}
```

### Linking Requirements

**Shared Libraries Required:**
```bash
# Set library path (CMake build)
export LD_LIBRARY_PATH=/path/to/clio-core/build/lib:$LD_LIBRARY_PATH

# Runtime libraries
libwrp_cte_core_client.so
libchimaera_cxx.so
libhermes_shm_host.so
libzmq.so (or libzmq.so.5)
libboost_*.so
```

**CXX Build Configuration:**
```rust
// In build.rs or project build script
cxx_build::bridge("src/ffi.rs")
    .file("shim/shim.cc")
    .std("c++20")
    .flag("-fcoroutines")
    .include("/path/to/include")
    .compile("cte_shim");
```

### Environment Variables

**Runtime Configuration:**
| Variable | Purpose | Example |
|----------|---------|---------|
| `CHI_WITH_RUNTIME=1` | Start embedded CTE runtime | `CHI_WITH_RUNTIME=1` |
| `CHI_CONFIG_PATH` | Path to configuration file | `/etc/cte/config.yaml` |
| `CHI_IPC_MODE` | IPC transport mode | `SHM`, `TCP`, `IPC` |
| `CHI_PORT` | RPC port for TCP mode | `9413` |
| `CHI_SERVER_ADDR` | Server address for TCP mode | `127.0.0.1` |
| `IOWARP_INCLUDE_DIR` | Header directory (build time) | `/usr/local/include` |
| `IOWARP_LIB_DIR` | Library directory (build time) | `/usr/local/lib` |
| `IOWARP_ZMQ_LIBS` | ZeroMQ libraries (build time) | `zmq;stdc++;gcc_s` |

---

## Examples

### Basic Blob Operations

**File: `examples/blob_basic.rs`**

```rust
use wrp_cte::{Client, Tag};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
    let client = Client::new().await?;
    
    // Create tag
    let tag = Tag::new("example_tag").await?;
    
    // Store data
    let data = b"Hello, CTE! This is a test blob.";
    tag.put_blob("test_blob.bin".to_string(), data.to_vec(), 0, 1.0)
        .await?;
    
    // Get blob size
    let size = tag.get_blob_size("test_blob.bin").await?;
    println!("Blob size: {} bytes", size);
    
    // Read blob
    let retrieved = tag.get_blob("test_blob.bin".to_string(), size, 0).await?;
    assert_eq!(retrieved, data);
    
    // Get score
    let score = tag.get_blob_score("test_blob.bin").await?;
    println!("Blob score: {}", score);
    
    // List blobs
    let blobs = tag.get_contained_blobs().await?;
    println!("Blobs in tag: {:?}", blobs);
    
    // Reorganize blob (change score)
    tag.reorganize_blob("test_blob.bin".to_string(), 0.7).await?;
    
    // Verify score change
    let new_score = tag.get_blob_score("test_blob.bin").await?;
    println!("New score: {}", new_score);
    
    // Get telemetry (may be empty if no operations)
    let telemetry = client.poll_telemetry(0).await?;
    for entry in telemetry {
        println!("Op: {:?}, Size: {}", entry.op, entry.size);
    }
    
    Ok(())
}
```

**Run:**
```bash
# Build with CMake (recommended)
cmake .. -DWRP_CORE_ENABLE_RUST=ON
make -j$(nproc)

# Or build standalone
cd context-transfer-engine/wrapper/rust
cargo build --release --features async

# Run with embedded runtime
CHI_WITH_RUNTIME=1 cargo run --example blob_basic
```

### Blob Monitor (Reference Implementation)

**File: `examples/blob_monitor.rs`**

Monitors CTE blob access patterns and auto-adjusts scores based on frecency.

**Architecture:**
1. **Telemetry Stats Map** - tracks access patterns by offset
2. **Blob Registry Map** - tracks blobs by name for score updates
3. **Main Loop** - calculates frecency and applies score updates

**Key Features:**
- Per-tag frecency calculation
- Score hysteresis (only update on bucket boundaries)
- Graceful shutdown with broadcast channel

**Example Output:**
```
Blob Monitor - Starting...
Refresh interval: 2000 ms
Press Ctrl+C to shut down gracefully.

==================================================================================================================
Blob Name                        | Accesses | Bytes Read | Bytes Writ | Frecency | Score | State
==================================================================================================================
large_data.bin                   |      100 |    500.00 KB |   100.00 KB |    12.34 |  0.90 |   HOT
small_config.json                |        5 |    100.00 B  |     1.00 B  |     0.50 |  0.20 |  COLD
==================================================================================================================
Named blobs in registry: 2 | Telemetry offsets: 45
```

**Run:**
```bash
# Build and run with embedded runtime
CHI_WITH_RUNTIME=1 cargo run --release --example blob_monitor 2000
```

### Telemetry Streaming

**File: `examples/telemetry_stream.rs`**

```rust
use wrp_cte::{Client, CteOp};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new().await?;
    
    println!("Streaming CTE telemetry (Ctrl+C to stop)...");
    
    // Infinite loop, streaming telemetry
    loop {
        tokio::select! {
            // Poll telemetry every 100ms
            _ = async {
                let entries = client.poll_telemetry(0).await?;
                for entry in entries {
                    println!(
                        "{:?}: {} bytes at offset {} on tag {}.{}", 
                        entry.op, 
                        entry.size, 
                        entry.off,
                        entry.tag_id.major,
                        entry.tag_id.minor
                    );
                }
                Ok::<(), Box<dyn std::error::Error>>(())
            } => {},
            
            // Listen for Ctrl+C
            _ = signal::ctrl_c() => {
                println!("\nReceived shutdown signal...");
                break;
            }
        }
    }
    
    Ok(())
}
```

---

## Troubleshooting

### Common Errors and Solutions

**Error 1: Library Loading Failure**

```
error: library not found: libwrp_cte_core_client.so
```

**Solution:**
```bash
# Set library path
export LD_LIBRARY_PATH=/path/to/clio-core/build/lib:$LD_LIBRARY_PATH

# Verify library exists
ls -la /path/to/clio-core/build/lib/libwrp_cte_core_client.so

# Check dependencies
ldd /path/to/clio-core/build/lib/libwrp_cte_core_client.so
```

**Error 2: Initialization Failed**

```
Error: CTE initialization failed: CTE initialization failed with code -1
```

**Causes and Solutions:**
1. **Missing runtime flag:**
   ```bash
   export CHI_WITH_RUNTIME=1
   cargo run
   ```

2. **Shared memory issues:**
   ```bash
   # Clean up old shared memory segments
   rm -rf /tmp/chimaera_$USER/*
   ```

3. **Port conflicts:**
   ```bash
   export CHI_PORT=9414
   ```

**Error 3: CXX Build Failure**

```
error: cannot find library: cxx
```

**Solution:**
```bash
# Install CXX crate
cargo add cxx

# Or add to Cargo.toml
[dependencies]
cxx = "1.0"
```

### Environment Variables

**Required for Build:**
| Variable | Description | Default |
|----------|-------------|---------|
| `IOWARP_INCLUDE_DIR` | Header directory | `/usr/local/include` |
| `IOWARP_LIB_DIR` | Library directory | `/usr/local/lib` |
| `IOWARP_EXTRA_INCLUDES` | Additional include paths | (empty) |
| `IOWARP_ZMQ_LIBS` | ZeroMQ libraries | (auto-detected) |

**Required for Runtime:**
| Variable | Description | Example |
|----------|-------------|---------|
| `CHI_WITH_RUNTIME=1` | Start embedded runtime | `CHI_WITH_RUNTIME=1` |
| `CHI_IPC_MODE` | IPC transport mode | `TCP`, `SHM`, `IPC` |
| `LD_LIBRARY_PATH` | Library search path | `/path/to/build/lib` |

### Debug Logging

**Enable CTE Debug Logging:**
```bash
# Set HSHM log level (0=debug, 1=info, 2=warn, 3=error)
export HSHM_LOG_LEVEL=0

# Run with debug output
CHI_WITH_RUNTIME=1 cargo run --example blob_basic 2>&1 | grep -i init
```

**Enable CXX FFI Tracing:**
```rust
// In main.rs
use cxx::bridge;

#[bridge]
mod cxxbridge {
    #[cfg(debug_assertions)]
    #[export = "cxxbridge1$cte_ffi$cte_init"]
    unsafe extern "C" fn cte_init(...) { ... }
}
```

### Performance Issues

**Problem: Operations are Slow or Hanging**

**Checklist:**

1. **IPC Mode:**
   ```bash
   # Use SHM for same-machine (lower latency)
   export CHI_IPC_MODE=SHM
   
   # Use TCP for distributed setup
   export CHI_IPC_MODE=TCP
   ```

2. **Shared Memory Size:**
   ```bash
   # Check shared memory limits
   cat /proc/sys/kernel/shmmax
   cat /proc/sys/kernel/shmall
   
   # Increase if needed (requires root)
   sudo sysctl -w kernel.shmmax=68719476736
   ```

3. **Worker Thread Count:**
   ```yaml
   # In config.yaml
   sched:
     workers: 8  # Increase for higher concurrency
   ```

### Async Tag Operations Not Working

**Problem: Async Tag methods panic with "use sync API"**

**Explanation:** Some async Tag methods are not yet implemented and fall back to sync API.

**Solution:** Use sync API for Tag operations:
```rust
use wrp_cte::sync::Tag;

let tag = Tag::new("dataset");
// Use sync operations directly
tag.put_blob("data.bin", b"data");
```

### Missing Telemetry

**Problem: `poll_telemetry()` returns empty vector**

**Explanation:** Telemetry is only collected for operations that occurred after `min_time`.

**Solution:** Use `min_time = 0` to get all telemetry:
```rust
let telemetry = client.poll_telemetry(0).await?; // Get all entries
```

---

## References

- **IOWarp Core Documentation:** `../../../../docs/`
- **CTE C++ API Documentation:** `core/include/wrp_cte/core/`
- **CXX Crate Documentation:** https://docs.rs/cxx/
- **Tokio Documentation:** https://docs.rs/tokio/
- **HermesShm Documentation:** `context-transport-primitives/docs/`

---

**License:** BSD 3-Clause License

This crate is part of IOWarp Core. See `../../../../LICENSE` for details.
