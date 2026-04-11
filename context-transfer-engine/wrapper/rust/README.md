# IOWarp CTE Rust Bindings

Rust bindings for the IOWarp Context Transfer Engine (CTE), enabling Rust applications to interface with CTE for blob storage, retrieval, score adjustment, and telemetry collection.

## Overview

The CTE Rust bindings provide a idiomatic Rust API over the underlying C++ CTE library. The bindings are built using the [cxx](https://github.com/dtolnay/cxx) crate for safe interoperability between Rust and C++.

### Key Features

- **Async API (Default)**: Non-blocking operations using Tokio's `spawn_blocking`
- **Sync API**: Blocking operations for debugging or single-threaded use
- **Thread-Safe Initialization**: Uses `OnceLock` pattern for safe concurrent initialization
- **Comprehensive Error Handling**: Detailed `CteError` enum with specific failure modes
- **Telemetry Support**: Collect and parse CTE operation telemetry

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Rust Application                                           │
│ (async default, tokio runtime)                             │
│                                                             │
│ let client = Client::new().await?;                         │
│ let tag = Tag::new("dataset").await?;                      │
│ tag.put_blob(...).await;                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Rust Bindings (wrp_cte)                                    │
│                                                             │
│ ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐   │
│ │ async.rs     │ │ sync.rs      │ │ types.rs           │   │
│ │ (default)    │ │ (optional)   │ │                    │   │
│ └──────┬───────┘ └──────┬───────┘ └────────────────────┘   │
└────────┼────────────────┼───────────────────────────────────┘
         │                │
┌────────▼────────────────▼────────────────────────────────┐
│ CXX Bridge (ffi.rs)                                       │
│ Safe Rust/C++ FFI boundary                                │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│ C++ Shim Layer (shim/shim.h, shim/shim.cc)               │
│ Wraps C++ CTE API for FFI                                 │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│ C++ CTE Library (libwrp_cte_core_client.so)              │
└─────────────────────────────────────────────────────────────┘
```

## Build Requirements

### Prerequisites

- **Rust**: 1.70 or later (Edition 2021)
- **C++ Compiler**: C++20 compatible (gcc, clang, or MSVC)
- **CMake**: 3.16 or later
- **CTE Library**: Built from IOWarp Core (see [Building](#building))

### Rust Dependencies

The bindings are self-contained with minimal dependencies:

```toml
[dependencies]
cxx = "1.0"
tokio = { version = "1.50", optional = true, default-features = false, features = ["rt"] }
```

### System Dependencies

- IOWarp Core libraries (libchimaera_cxx.so, libwrp_cte_core_client.so)
- Standard C++20 runtime

## Installation

### Option 1: Build with IOWarp Core (Recommended)

```bash
# Clone IOWarp Core
git clone https://github.com/iowarp/clio-core
cd clio-core

# Configure with Rust bindings
mkdir build && cd build
cmake .. -DWRP_CORE_ENABLE_RUST=ON

# Build
make -j$(nproc)

# The Rust bindings will be built as part of the project
```

### Option 2: Standalone Rust Build

```bash
# Navigate to Rust wrapper directory
cd context-transfer-engine/wrapper/rust

# Build with async API (default)
cargo build --release

# Build with sync API only
cargo build --release --no-default-features --features sync
```

### Option 3: Use as Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
wrp-cte-rs = { path = "/path/to/clio-core/context-transfer-engine/wrapper/rust" }
tokio = { version = "1.50", features = ["rt-multi-thread", "macros"] }
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `async` | Yes | Async/await API using Tokio |
| `sync` | No | Synchronous (blocking) API |

### Async API (Default)

```toml
[dependencies]
wrp-cte-rs = { path = "..." }  # async enabled by default
tokio = { version = "1.50", features = ["rt-multi-thread", "macros"] }
```

### Sync API Only

```toml
[dependencies]
wrp-cte-rs = { path = "...", default-features = false, features = ["sync"] }
```

## Quick Start

### Async API Example

```rust
use wrp_cte::{Client, Tag, CteResult};

#[tokio::main]
async fn main() -> CteResult<()> {
    // Initialize CTE (automatic via Client::new)
    let client = Client::new().await?;

    // Create or open a tag
    let tag = Tag::new("my_dataset").await?;

    // Store data with placement score
    tag.put_blob_with_options(
        "data.bin",
        b"Hello, CTE!",
        0,      // offset
        1.0,    // score (0.0-1.0)
    ).await;

    // Retrieve data
    let data = tag.get_blob("data.bin", 1024, 0).await;
    println!("Retrieved: {}", String::from_utf8_lossy(&data));

    // Get blob score
    let score = tag.get_blob_score("data.bin").await;
    println!("Blob score: {}", score);

    // Adjust blob placement score
    tag.reorganize_blob("data.bin", 0.5).await?;

    // List contained blobs
    let blobs = tag.get_contained_blobs().await;
    println!("Blobs in tag: {:?}", blobs);

    // Poll telemetry
    let telemetry = client.poll_telemetry(0).await?;
    for entry in telemetry {
        println!("Op: {:?}, Size: {}", entry.op, entry.size);
    }

    Ok(())
}
```

### Sync API Example

```rust
use wrp_cte::sync::{init, Client, Tag};

fn main() {
    // Initialize CTE
    init("").expect("CTE initialization failed");

    // Create client and tag
    let client = Client::new().unwrap();
    let tag = Tag::new("my_dataset");

    // Store data
    tag.put_blob_with_options("data.bin", b"Hello, CTE!", 0, 1.0);

    // Retrieve data
    let data = tag.get_blob("data.bin", 1024, 0);
    println!("Retrieved: {}", String::from_utf8_lossy(&data));

    // Get telemetry
    let telemetry = client.poll_telemetry(0).unwrap();
    for entry in telemetry {
        println!("Op: {:?}, Size: {}", entry.op, entry.size);
    }
}
```

## API Reference

### Initialization

#### Async API

```rust
use wrp_cte::Client;

// Initialize and create client
let client = Client::new().await?;
```

The async API initializes CTE automatically when creating the first `Client`.

#### Sync API

```rust
use wrp_cte::sync::init;

// Initialize CTE
init("")?;  // "" for default config, or path to config file

// Now create clients and tags
let client = Client::new()?;
let tag = Tag::new("my_dataset");
```

### Client Operations

#### `Client::new()`

Create a new CTE client.

**Async Signature:**
```rust
impl Client {
    pub async fn new() -> CteResult<Self>
}
```

**Sync Signature:**
```rust
impl Client {
    pub fn new() -> CteResult<Self>
}
```

**Returns:** `Ok(Client)` on success, `Err(CteError::InitFailed)` if initialization fails.

#### `client.poll_telemetry(min_time)`

Retrieve telemetry entries for operations that occurred after `min_time`.

```rust
let telemetry = client.poll_telemetry(0)?;  // 0 for all entries
for entry in telemetry {
    println!("Operation: {:?}", entry.op);
    println!("Size: {} bytes", entry.size);
    println!("Timestamp: {} ns", entry.mod_time.nanos);
}
```

#### `client.reorganize_blob(tag_id, name, score)`

Change the placement score of a blob, potentially triggering data migration.

```rust
// Reorganize blob to lower tier
client.reorganize_blob(tag_id, "data.bin", 0.3)?;
```

**Parameters:**
- `tag_id`: `CteTagId` containing the blob
- `name`: Blob name
- `score`: New placement score (0.0 = lowest priority, 1.0 = highest)

#### `client.del_blob(tag_id, name)`

Delete a blob from storage.

```rust
client.del_blob(tag_id, "old_data.bin")?;
```

### Tag Operations

#### `Tag::new(name)`

Create or open a tag by name.

**Async Signature:**
```rust
impl Tag {
    pub async fn new(name: &str) -> CteResult<Self>
}
```

**Sync Signature:**
```rust
impl Tag {
    pub fn new(name: &str) -> Self
}
```

#### `Tag::from_id(id)`

Open an existing tag by ID.

```rust
use wrp_cte::CteTagId;

let id = CteTagId::new(1, 2);
let tag = Tag::from_id(id);
```

#### `tag.put_blob_with_options(name, data, offset, score)`

Write data into a blob.

```rust
let data = b"Large blob content...";
tag.put_blob_with_options("large_data.bin", data, 0, 1.0);
```

**Parameters:**
- `name`: Blob name
- `data`: Byte slice to write
- `offset`: Offset in blob (0 for new blobs)
- `score`: Placement score (0.0-1.0)

#### `tag.put_blob(name, data)`

Convenience method with default offset (0) and score (1.0).

```rust
tag.put_blob("simple.bin", b"Data");
```

#### `tag.get_blob(name, size, offset)`

Read data from a blob.

```rust
let data = tag.get_blob("data.bin", 1024, 0);
```

**Parameters:**
- `name`: Blob name
- `size`: Number of bytes to read
- `offset`: Offset in blob

**Returns:** `Vec<u8>` containing the data.

#### `tag.get_blob_size(name)`

Get the size of a blob.

```rust
let size = tag.get_blob_size("data.bin");
println!("Blob size: {} bytes", size);
```

#### `tag.get_blob_score(name)`

Get the placement score of a blob.

```rust
let score = tag.get_blob_score("data.bin");
println!("Placement score: {}", score);
```

#### `tag.reorganize_blob(name, score)`

Change blob placement score.

```rust
tag.reorganize_blob("data.bin", 0.7)?;
```

#### `tag.get_contained_blobs()`

List all blobs in the tag.

```rust
let blobs = tag.get_contained_blobs();
println!("Blobs: {:?}", blobs);
```

## Types Reference

### `CteTagId`

Unique identifier for tags, blobs, and pools (8-byte layout: major.u32 + minor.u32).

```rust
use wrp_cte::CteTagId;

// Create from components
let id = CteTagId::new(1, 2);

// Create null ID
let null = CteTagId::null();

// Convert to/from u64
let as_u64 = id.to_u64();
let from_u64 = CteTagId::from_u64(as_u64);
```

### `CteTelemetry`

Telemetry entry for monitoring CTE operations.

```rust
pub struct CteTelemetry {
    pub op: CteOp,          // Operation type
    pub off: u64,           // Offset in blob
    pub size: u64,          // Operation size
    pub tag_id: CteTagId,   // Associated tag
    pub mod_time: SteadyTime,   // Modification time
    pub read_time: SteadyTime,  // Read time
    pub logical_time: u64,  // Logical time counter
}
```

### `CteOp`

Operation types for CTE.

```rust
pub enum CteOp {
    PutBlob = 0,
    GetBlob = 1,
    DelBlob = 2,
    GetOrCreateTag = 3,
    DelTag = 4,
    GetTagSize = 5,
}
```

### `SteadyTime`

Monotonic clock time point (nanosecond precision).

```rust
use wrp_cte::SteadyTime;

let t1 = SteadyTime::from_nanos(1000);
let t2 = SteadyTime::from_nanos(2000);

let duration = t2.duration_since(&t1);
println!("Duration: {} ns", duration.as_nanos());
```

### `PoolQuery`

Pool routing strategies.

```rust
use wrp_cte::PoolQuery;

// Local node only
let local = PoolQuery::local();

// Dynamic routing with timeout
let dynamic = PoolQuery::dynamic(30.0);

// Broadcast to all nodes
let broadcast = PoolQuery::broadcast(60.0);
```

## Error Handling

The bindings use a comprehensive `CteError` enum for error handling:

```rust
pub enum CteError {
    InitFailed { reason: String },
    PoolCreationFailed { message: String },
    PoolNotFound { pool_id: String },
    TagNotFound { name: String },
    TagAlreadyExists { name: String },
    BlobNotFound { tag: String, blob: String },
    BlobIOError { message: String },
    TargetRegistrationFailed { path: String },
    TargetNotFound { path: String },
    TelemetryUnavailable,
    InvalidParameter { message: String },
    RuntimeError { code: u32, message: String },
    Timeout,
    FfiError { message: String },
    IoError { message: String },
    NotImplemented { feature: String, reason: String },
}
```

### Example Error Handling

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

## Thread Safety

### Initialization

The sync API uses `OnceLock` for thread-safe initialization:

```rust
// First call initializes CTE
init("")?;

// Subsequent calls return cached result (no re-init)
init("")?;  // Returns same result as first call
```

This ensures:
- Only one thread performs initialization
- Other threads wait for initialization to complete
- Initialization result is cached for all threads

### Async Operations

Async operations use `tokio::task::spawn_blocking` to execute blocking C++ calls in a thread pool. The underlying C++ objects (`Client`, `Tag`) are wrapped in `Sendable*` types with proper `unsafe impl Send` bounds.

**Note**: The async `Tag` operations use `Mutex<Option<SendableTag>>` to ensure thread-safe access to the underlying C++ object.

## Async Limitations

The async API has a known limitation for `Tag` operations:

```rust
// These async Tag methods are not yet implemented
tag.put_blob(...).await;       // Panics: use sync API
tag.get_blob(...).await;       // Panics: use sync API
tag.get_blob_size(...).await;  // Panics: use sync API
tag.get_contained_blobs().await; // Panics: use sync API
```

**Workaround**: Use the sync API for Tag operations:

```rust
use wrp_cte::sync::Tag;

// Create sync Tag (can be used from async context)
let sync_tag = Tag::new("dataset");

// Use blocking operations (spawn_blocking handles this)
tokio::task::spawn_blocking(move || {
    sync_tag.put_blob("data.bin", b"data");
}).await;
```

## Deployment Guide

### Building the Runtime

#### Prerequisites

1. **IOWarp Core Dependencies** (via apt or source build):
   ```bash
   # Install dependencies via apt (on Ubuntu/Debian)
   sudo apt-get update
   sudo apt-get install -y \
       build-essential cmake git \
       libboost-all-dev libhdf5-dev \
       libzmq3-dev libyaml-cpp-dev \
       libpython3-dev python3-dev
   ```

2. **Rust Toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   rustc --version  # Verify Rust 1.70+
   ```

#### Building with CMake (Recommended)

Build IOWarp Core with Rust bindings enabled:

```bash
# Clone repository
git clone https://github.com/iowarp/clio-core
cd clio-core

# Configure CMake with Rust support
mkdir build && cd build
cmake .. -DWRP_CORE_ENABLE_RUST=ON

# Build all components
make -j$(nproc)
sudo make install  # Install to /usr/local
```

The CMake build will:
- Build the core C++ libraries (libchimaera_cxx.so, libwrp_cte_core_client.so)
- Build the Rust bindings (libwrp_cte.rlib)
- Set up proper RPATHs for library discovery

#### Building Standalone Rust Crate

For development or testing without installing:

```bash
cd context-transfer-engine/wrapper/rust

# Set library paths (CMake build directory)
export IOWARP_INCLUDE_DIR=/path/to/clio-core/build/include
export IOWARP_LIB_DIR=/path/to/clio-core/build/lib
export IOWARP_EXTRA_INCLUDES=/path/to/clio-core/context-transport-primitives/include

# Build with async API (default)
cargo build --release

# Build with sync API only
cargo build --release --no-default-features --features sync
```

#### Required Shared Libraries

At runtime, the following libraries must be in `LD_LIBRARY_PATH` or RPATH:

| Library | Purpose |
|---------|---------|
| `libwrp_cte_core_client.so` | CTE client implementation |
| `libchimaera_cxx.so` | Chimaera runtime core |
| `libhermes_shm_host.so` | Shared memory primitives |
| `libzmq.so` | ZeroMQ messaging |
| `libhdf5.so` | HDF5 storage backend |
| `libboost_*.so` | Boost libraries |

### Running Tests

#### Test Categories

1. **Unit Tests** (no runtime needed):
   - Test FFI bridge functionality
   - Validate data structures and error handling
   - Run with `cargo test` (no `#[ignore]` tests)

2. **Integration Tests** (require runtime):
   - Test actual CTE operations (PutBlob, GetBlob, etc.)
   - Marked with `#[ignore = "Requires running CTE runtime"]`
   - Require initialized CTE runtime

#### Running Unit Tests

```bash
cd context-transfer-engine/wrapper/rust

# Run unit tests only (no runtime needed)
cargo test --lib

# Run specific test
cargo test --lib test_cte_tag_id

# Run with verbose output
cargo test --lib -- --nocapture
```

#### Running Integration Tests

Integration tests require the CTE runtime to be initialized. There are two approaches:

**Method 1: Embedded Runtime (Recommended)**

Use the embedded runtime via `CHI_WITH_RUNTIME=1`:

```bash
cd context-transfer-engine/wrapper/rust

# Set library paths
export LD_LIBRARY_PATH=/path/to/clio-core/build/lib:$LD_LIBRARY_PATH

# Run all integration tests with embedded runtime
CHI_WITH_RUNTIME=1 cargo test --ignored --features async

# Run specific test
CHI_WITH_RUNTIME=1 cargo test --ignored test_blob_put_get --features sync

# Run all tests (both ignored and regular)
CHI_WITH_RUNTIME=1 cargo test -- --include-ignored
```

**Method 2: Separate Runtime Process**

Start the runtime first, then run tests:

```bash
# Terminal 1: Start CTE runtime
export LD_LIBRARY_PATH=/path/to/clio-core/build/lib:$LD_LIBRARY_PATH
wrp_cte --config /path/to/config.yaml

# Terminal 2: Run tests (no CHI_WITH_RUNTIME needed)
export LD_LIBRARY_PATH=/path/to/clio-core/build/lib:$LD_LIBRARY_PATH
cargo test --ignored --features async
```

#### Test Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CHI_WITH_RUNTIME` | Start embedded runtime | `CHI_WITH_RUNTIME=1` |
| `LD_LIBRARY_PATH` | Library search path | `/path/to/build/lib` |
| `IOWARP_INCLUDE_DIR` | Header directory (build time) | `/usr/local/include` |
| `IOWARP_LIB_DIR` | Library directory (build time) | `/usr/local/lib` |

### Using the Library

#### Initialization Process

The Rust bindings automatically handle CTE initialization when you call `init("")` (sync) or `Client::new().await` (async).

**Initialization Sequence:**

1. **Chimaera Runtime Init**: `chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)`
   - Initializes shared memory and IPC
   - Starts worker threads if `CHI_WITH_RUNTIME=1`
   
2. **CTE Client Init**: `wrp_cte::core::WRP_CTE_CLIENT_INIT(config_path)`
   - Creates CTE client instance
   - Sets up connection to runtime

**Thread Safety:**

The initialization uses `std::once_flag` and `OnceLock` to ensure:
- Only one thread performs initialization
- Other threads wait for initialization to complete
- Result is cached for subsequent calls

#### Sync API Example

```rust
use wrp_cte::sync::{init, Client, Tag};

fn main() {
    // Initialize CTE with embedded runtime (requires CHI_WITH_RUNTIME=1)
    init("").expect("CTE initialization failed");
    
    // Create client
    let client = Client::new().expect("Client creation failed");
    
    // Create tag and store data
    let tag = Tag::new("my_dataset");
    tag.put_blob_with_options("data.bin", b"Hello, CTE!", 0, 1.0)
        .expect("Blob put failed");
    
    // Retrieve data
    let data = tag.get_blob("data.bin", 12, 0).expect("Blob get failed");
    println!("Retrieved: {}", String::from_utf8_lossy(&data));
}
```

#### Async API Example

```rust
use wrp_cte::r#async::{Client, Tag};

#[tokio::main]
async fn main() {
    // Initialize CTE (happens automatically with Client::new())
    let client = Client::new().await.expect("Client creation failed");
    
    // Create tag
    let tag = Tag::new("my_dataset").await.expect("Tag creation failed");
    
    // Store data
    tag.put_blob("data.bin".to_string(), b"Hello, CTE!".to_vec(), 0, 1.0)
        .await
        .expect("Blob put failed");
    
    // Retrieve data
    let data = tag.get_blob("data.bin".to_string(), 12, 0)
        .await
        .expect("Blob get failed");
    println!("Retrieved: {}", String::from_utf8_lossy(&data));
}
```

#### Runtime Environment Variables

The following environment variables control CTE behavior:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHI_WITH_RUNTIME` | Set to `1` to start embedded runtime | Unset (no runtime) |
| `CHI_CONFIG_PATH` | Path to configuration file | Uses defaults |
| `LD_LIBRARY_PATH` | Shared library search path | System default |
| `CHI_IPC_MODE` | IPC transport mode: `SHM`, `TCP`, `IPC` | `TCP` |
| `CHI_PORT` | RPC port for TCP mode | `9413` |
| `CHI_SERVER_ADDR` | Server address for TCP mode | `127.0.0.1` |

#### Linking Requirements

When using the library as a dependency in another project:

**Cargo.toml:**
```toml
[dependencies]
wrp-cte-rs = { path = "/path/to/clio-core/context-transfer-engine/wrapper/rust" }
tokio = { version = "1.50", features = ["rt-multi-thread", "macros"] }
```

**Build Script (build.rs):**
```rust
fn main() {
    // Link to CTE libraries
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=dylib=wrp_cte_core_client");
    println!("cargo:rustc-link-lib=dylib=chimaera_cxx");
    println!("cargo:rustc-link-lib=dylib=hermes_shm_host");
}
```

### Troubleshooting

#### Library Loading Errors

**Problem**: `cannot open shared object file: No such file or directory`

**Solution**: Set `LD_LIBRARY_PATH` or use RPATH:

```bash
# Method 1: LD_LIBRARY_PATH (temporary)
export LD_LIBRARY_PATH=/path/to/clio-core/build/lib:$LD_LIBRARY_PATH
cargo run

# Method 2: RPATH (permanent, set during build)
export IOWARP_LIB_DIR=/path/to/clio-core/build/lib
cargo build --release
```

**Finding libraries:**
```bash
# Check if libraries are findable
ldd /path/to/clio-core/build/lib/libwrp_cte_core_client.so

# Check RPATH
readelf -d /path/to/clio-core/build/lib/libwrp_cte_core_client.so | grep RPATH
```

#### Initialization Failures

**Problem**: `CteError::InitFailed { reason: "CTE initialization failed with code -1" }`

**Causes and Solutions:**

1. **Missing runtime flag**:
   ```bash
   # Solution: Set CHI_WITH_RUNTIME=1
   CHI_WITH_RUNTIME=1 cargo test
   ```

2. **Shared memory issues**:
   ```bash
   # Clean up old shared memory segments
   rm -rf /tmp/chimaera_$USER/*
   
   # Check permissions
   ls -la /tmp/chimaera_$USER/
   ```

3. **Port conflicts**:
   ```bash
   # Check if default port is in use
   lsof -i :9413
   
   # Use different port
   export CHI_PORT=9414
   ```

4. **Insufficient resources**:
   ```bash
   # Check shared memory limits
   cat /proc/sys/kernel/shmmax
   cat /proc/sys/kernel/shmall
   
   # Increase limits if needed (requires root)
   sudo sysctl -w kernel.shmmax=68719476736
   sudo sysctl -w kernel.shmall=4294967296
   ```

#### Runtime Errors

**Problem**: `RuntimeError { code: 1, message: "Pool creation failed" }`

**Solutions:**

1. Ensure bdev (block device) is configured:
   ```bash
   # The runtime needs storage backends
   wrp_cte --config /path/to/config.yaml
   ```

2. Check configuration file syntax (YAML):
   ```bash
   # Validate YAML syntax
   python3 -c "import yaml; yaml.safe_load(open('/path/to/config.yaml'))"
   ```

#### Build Failures

**Problem**: CXX cannot find C++ headers

**Solutions:**

1. **Set include directories**:
   ```bash
   export IOWARP_INCLUDE_DIR=/path/to/clio-core/build/include
   export IOWARP_EXTRA_INCLUDES=/path/to/clio-core/context-transport-primitives/include
   cargo build
   ```

2. **Check CMake build**:
   ```bash
   # Ensure IOWarp Core built successfully
   cd /path/to/clio-core/build
   ls -la lib/libwrp_cte_core_client.so
   ls -la lib/libchimaera_cxx.so
   ```

3. **Enable verbose output**:
   ```bash
   cargo build --verbose 2>&1 | grep error
   ```

#### Debugging Initialization

Enable debug logging to trace initialization:

```bash
# Set HSHM log level (0=debug, 1=info, 2=warn, 3=error)
export HSHM_LOG_LEVEL=0

# Run with debug output
CHI_WITH_RUNTIME=1 cargo test -- --nocapture 2>&1 | grep -i init
```

#### Performance Issues

**Problem**: Operations are slow or hanging

**Checklist:**

1. **IPC Mode**: Use `SHM` for same-machine communication:
   ```bash
   export CHI_IPC_MODE=SHM  # Lower latency for same machine
   # OR
   export CHI_IPC_MODE=TCP  # Supports distributed setup
   ```

2. **Thread Pool Size**: Configure worker threads:
   ```bash
   # In config.yaml
   sched:
     workers: 8  # Number of worker threads
   ```

3. **Shared Memory Size**: Ensure adequate shared memory:
   ```bash
   # Check shared memory configuration
   df -h /dev/shm
   
   # Set in config.yaml
   main_segment_size: 2G
   client_data_segment_size: 512M
   runtime_data_segment_size: 512M
   ```

#### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `-1` | Initialization failed | Check `CHI_WITH_RUNTIME=1` and library paths |
| `1` | Pool creation failed | Check storage backend configuration |
| `2` | Tag not found | Create tag with `Tag::new()` first |
| `3` | Blob not found | Verify blob name and tag ID |
| `4` | Permission denied | Check shared memory permissions (`/tmp/chimaera_$USER/`) |

#### Getting Help

1. **Check logs**:
   ```bash
   # Enable debug logging
   export HSHM_LOG_LEVEL=0
   export CHI_WITH_RUNTIME=1
   cargo test -- --nocapture 2>&1 | tee cte_debug.log
   ```

2. **Verify environment**:
   ```bash
   # Library paths
   echo $LD_LIBRARY_PATH
   
   # Environment variables
   env | grep CHI_
   env | grep IOWARP_
   ```

3. **Run diagnostics**:
   ```bash
   # Check shared memory
   ls -la /tmp/chimaera_$USER/
   
   # Check processes
   ps aux | grep wrp_cte
   ps aux | grep chimaera
   ```

4. **Report issues**: Include:
   - Complete error message with stack trace
   - Environment variables (`env | grep -E 'CHI_|IOWARP|LD_LIBRARY'`)
   - Library versions (`ldd --version`, `rustc --version`)
   - Operating system version
   - Configuration file (if used)

#### Async Tag Operations Not Working

**Problem**: Async Tag methods panic with "use sync API"

**Explanation**: The async Tag operations have not been fully implemented due to thread-safety requirements for the underlying C++ objects.

**Solution**: Use the sync API for Tag operations:

```rust
use wrp_cte::sync::Tag;

let tag = Tag::new("dataset");
// Use sync operations directly
tag.put_blob("data.bin", b"data");
```

#### Missing Telemetry

**Problem**: `poll_telemetry()` returns empty vector

**Explanation**: Telemetry is only collected for operations that occurred after `min_time`.

**Solution**: Use `min_time = 0` to get all telemetry:

```rust
let telemetry = client.poll_telemetry(0)?;  // Get all entries
```

## Contributing

### Adding New FFI Functions

1. Add function declaration to `src/ffi.rs` in the `#[cxx::bridge]` block
2. Implement the function in `shim/shim.cc`
3. Create wrapper methods in `sync.rs` and/or `async.rs`
4. Update documentation

### Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Use clippy linting (`cargo clippy`)
- Document all public functions with doc comments
- Include examples in doc comments where helpful

## License

This crate is part of IOWarp Core and is licensed under the BSD 3-Clause License. See the [IOWarp Core LICENSE](../../../../LICENSE) for details.

## References

- [IOWarp Core Documentation](../../../../docs/)
- [CTE C++ API Documentation](../docs/cte/cte.md)
- [CXX Crate Documentation](https://docs.rs/cxx/)
- [Tokio Documentation](https://docs.rs/tokio/)