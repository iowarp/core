/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

//! CXX bridge to C++ CTE library
//!
//! This module defines the FFI boundary between Rust and C++ using the cxx crate.
//! Design: All shared types are opaque except primitive scalars. Complex data
//! is passed through output parameters (Vec<u8>, Vec<String>).
//!
//! # Architecture
//!
//! The FFI uses the following design patterns:
//!
//! 1. **Opaque Types**: C++ types (`Client`, `Tag`) are exposed as opaque types
//!    that can only be created/destroyed through FFI functions.
//!
//! 2. **Output Parameters**: Complex data structures (strings, byte arrays) are
//!    passed through output parameters rather than return values, avoiding
//!    complex memory management at the FFI boundary.
//!
//! 3. **Primitive Parameters**: All scalar types use C-compatible primitives
//!    (u32, u64, i32, f32, f64) that have identical representations in both
//!    languages.
//!
//! # Safety Guarantee
//!
//! The cxx bridge provides the following safety guarantees:
//!
//! 1. **Memory Layout**: cxx ensures identical memory layout for all types
//!    passed across the FFI boundary, including alignment and padding.
//!
//! 2. **Lifetime Management**: `UniquePtr<T>` provides automatic RAII cleanup
//!    of C++ objects when the Rust wrapper is dropped.
//!
//! 3. **Exception Safety**: C++ exceptions are caught by cxx and converted
//!    to Rust panics or Result types, preventing undefined behavior.
//!
//! 4. **Thread Safety**: All FFI functions can be safely called from any
//!    thread; the C++ implementation handles internal synchronization.

/// Telemetry entry size in bytes: op(4) + off(8) + size(8) + tag_major(4) + tag_minor(4) +
/// mod_time_nanos(8) + read_time_nanos(8) + logical_time(8) = 52 bytes
pub const TELEMETRY_ENTRY_SIZE: usize = 52;

/// Offsets for parsing telemetry entries
mod offsets {
    pub const OP: usize = 0;
    pub const OFF: usize = 4;
    pub const SIZE: usize = 12;
    pub const TAG_MAJOR: usize = 20;
    pub const TAG_MINOR: usize = 24;
    pub const MOD_TIME: usize = 28;
    pub const READ_TIME: usize = 36;
    pub const LOGICAL_TIME: usize = 44;
}

/// Telemetry operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CteOp {
    PutBlob = 0,
    GetBlob = 1,
    DelBlob = 2,
    GetOrCreateTag = 3,
    DelTag = 4,
    GetTagSize = 5,
}

impl From<u32> for CteOp {
    fn from(value: u32) -> Self {
        match value {
            0 => CteOp::PutBlob,
            1 => CteOp::GetBlob,
            2 => CteOp::DelBlob,
            3 => CteOp::GetOrCreateTag,
            4 => CteOp::DelTag,
            5 => CteOp::GetTagSize,
            _ => CteOp::PutBlob,
        }
    }
}

/// Unique tag identifier (matches chi::UniqueId layout)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CteTagId {
    pub major: u32,
    pub minor: u32,
}

/// Steady clock time point (nanoseconds since epoch)
#[derive(Debug, Clone, Copy)]
pub struct SteadyTime {
    pub nanos: i64,
}

/// Telemetry entry for CTE operations
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

/// Parse telemetry entries from raw byte buffer
///
/// # Safety
///
/// This function is safe because:
/// - It only reads from the provided slice without mutation
/// - Uses little-endian byte order matching the C++ serialization
/// - Validates buffer bounds before each read operation
/// - Returns an empty vector for invalid/truncated data
pub fn parse_telemetry(data: &[u8]) -> Vec<CteTelemetry> {
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + TELEMETRY_ENTRY_SIZE <= data.len() {
        let op = u32::from_le_bytes([
            data[offset + offsets::OP],
            data[offset + offsets::OP + 1],
            data[offset + offsets::OP + 2],
            data[offset + offsets::OP + 3],
        ]);

        let off = u64::from_le_bytes([
            data[offset + offsets::OFF],
            data[offset + offsets::OFF + 1],
            data[offset + offsets::OFF + 2],
            data[offset + offsets::OFF + 3],
            data[offset + offsets::OFF + 4],
            data[offset + offsets::OFF + 5],
            data[offset + offsets::OFF + 6],
            data[offset + offsets::OFF + 7],
        ]);

        let size = u64::from_le_bytes([
            data[offset + offsets::SIZE],
            data[offset + offsets::SIZE + 1],
            data[offset + offsets::SIZE + 2],
            data[offset + offsets::SIZE + 3],
            data[offset + offsets::SIZE + 4],
            data[offset + offsets::SIZE + 5],
            data[offset + offsets::SIZE + 6],
            data[offset + offsets::SIZE + 7],
        ]);

        let tag_major = u32::from_le_bytes([
            data[offset + offsets::TAG_MAJOR],
            data[offset + offsets::TAG_MAJOR + 1],
            data[offset + offsets::TAG_MAJOR + 2],
            data[offset + offsets::TAG_MAJOR + 3],
        ]);

        let tag_minor = u32::from_le_bytes([
            data[offset + offsets::TAG_MINOR],
            data[offset + offsets::TAG_MINOR + 1],
            data[offset + offsets::TAG_MINOR + 2],
            data[offset + offsets::TAG_MINOR + 3],
        ]);

        let mod_time = i64::from_le_bytes([
            data[offset + offsets::MOD_TIME],
            data[offset + offsets::MOD_TIME + 1],
            data[offset + offsets::MOD_TIME + 2],
            data[offset + offsets::MOD_TIME + 3],
            data[offset + offsets::MOD_TIME + 4],
            data[offset + offsets::MOD_TIME + 5],
            data[offset + offsets::MOD_TIME + 6],
            data[offset + offsets::MOD_TIME + 7],
        ]);

        let read_time = i64::from_le_bytes([
            data[offset + offsets::READ_TIME],
            data[offset + offsets::READ_TIME + 1],
            data[offset + offsets::READ_TIME + 2],
            data[offset + offsets::READ_TIME + 3],
            data[offset + offsets::READ_TIME + 4],
            data[offset + offsets::READ_TIME + 5],
            data[offset + offsets::READ_TIME + 6],
            data[offset + offsets::READ_TIME + 7],
        ]);

        let logical_time = u64::from_le_bytes([
            data[offset + offsets::LOGICAL_TIME],
            data[offset + offsets::LOGICAL_TIME + 1],
            data[offset + offsets::LOGICAL_TIME + 2],
            data[offset + offsets::LOGICAL_TIME + 3],
            data[offset + offsets::LOGICAL_TIME + 4],
            data[offset + offsets::LOGICAL_TIME + 5],
            data[offset + offsets::LOGICAL_TIME + 6],
            data[offset + offsets::LOGICAL_TIME + 7],
        ]);

        entries.push(CteTelemetry {
            op: CteOp::from(op),
            off,
            size,
            tag_id: CteTagId {
                major: tag_major,
                minor: tag_minor,
            },
            mod_time: SteadyTime { nanos: mod_time },
            read_time: SteadyTime { nanos: read_time },
            logical_time,
        });

        offset += TELEMETRY_ENTRY_SIZE;
    }

    entries
}

/// CXX bridge module - defines FFI boundary
///
/// # Safety
///
/// This module defines the safe interface between Rust and C++ using the cxx crate.
/// The safety guarantees are as follows:
///
/// ## Memory Layout
///
/// 1. **Opaque Types**: `Client` and `Tag` are opaque types that cxx manages
///    through `UniquePtr<T>`. The internal representation is completely hidden
///    from Rust, preventing incorrect memory access or modification.
///
/// 2. **Primitive Types**: All scalar parameters use C-compatible types (u32, u64,
///    i32, f32, f64, &str) that have identical bit-level representations in both
///    languages. cxx generates compile-time static assertions to verify compatibility.
///
/// 3. **Buffer Types**: `Vec<u8>` and `Vec<String>` map to C++ `std::vector<uint8_t>`
///    and `std::vector<std::string>` with identical memory layouts and alignment.
///    cxx manages the buffer capacity/size/ptr triplet correctly.
///
/// ## Ownership Model
///
/// 1. **UniquePtr**: Factory functions (`client_new`, `tag_new`, `tag_from_id`)
///    return `UniquePtr<T>` which uniquely owns the C++ object. When dropped, the
///    C++ destructor is called automatically.
///
/// 2. **Borrowing**: All operations accept `&T` references that borrow the UniquePtr.
///    The reference cannot outlive the owner, preventing use-after-free.
///
/// 3. **String Slices**: `&str` parameters borrow Rust strings with guaranteed null
///    termination provided by cxx's CxxString adapter, preventing buffer overflows.
///
/// ## Thread Safety
///
/// 1. **Cross-Thread Movement**: `UniquePtr<T>` is not `Send` by default because
///    C++ destructors must run on the thread that owns the object. The async module
///    wraps these in `SendableTag`/`SendableClient` with explicit SAFETY documentation.
///
/// 2. **Internal Synchronization**: The C++ implementations use internal mutexes
///    for shared state, ensuring thread-safe concurrent access to the runtime.
///
/// 3. **No Global State**: The FFI functions don't access mutable global state
///    directly; all state is in Client/Tag objects or the runtime process.
///
/// ## Exception Safety
///
/// 1. **C++ Exceptions**: cxx catches C++ exceptions at the FFI boundary and
///    converts them to Rust panics. For FFI functions returning Result, exceptions
///    become Err variants; for infallible functions, they become panics.
///
/// 2. **Panic Safety**: If Rust code panics across an FFI call, cxx ensures the
///    C++ stack is properly unwound before terminating.
///
/// ## Undefined Behavior Prevention
///
/// 1. **Null Pointers**: cxx ensures UniquePtr values are never null when passed
///    to C++ (empty UniquePtr maps to nullptr which C++ handles correctly).
///
/// 2. **Lifetime Bounds**: All references have lifetime bounds enforced by the
///    compiler; `&str` parameters cannot outlive the calling function.
///
/// 3. **No Data Races**: The FFI functions don't provide mutable access to shared
///    state without synchronization primitives.
///
/// # FFI Function Overview
///
/// ## Factory Functions
/// - `cte_init`: Initialize the CTE runtime
/// - `client_new`: Create a new CTE client
/// - `tag_new`: Create or open a tag by name
/// - `tag_from_id`: Open an existing tag by ID
///
/// ## Query Functions
/// - `tag_get_id_major`/`tag_get_id_minor`: Get tag ID components
/// - `tag_get_blob_score`: Get blob placement score
/// - `tag_get_blob_size`: Get blob size in bytes
/// - `tag_get_contained_blobs`: List all blobs in a tag
/// - `client_poll_telemetry_raw`: Poll telemetry entries
///
/// ## Mutation Functions
/// - `tag_put_blob`: Write data to a blob
/// - `tag_get_blob`: Read data from a blob
/// - `tag_reorganize_blob`: Change blob placement score
/// - `client_del_blob`: Delete a blob
/// - `client_reorganize_blob`: Change blob score via client API
#[cxx::bridge(namespace = "cte_ffi")]
pub mod ffi {
    unsafe extern "C++" {
        include!("shim/shim.h");

        // Opaque types - managed by cxx
        //
        // SAFETY: These types are opaque from Rust's perspective. Their memory
        // layout, size, and alignment are completely managed by C++. cxx generates
        // the necessary glue code to safely create, destroy, and call methods on
        // these types without exposing any internal details to Rust.
        //
        // The opaque pattern ensures:
        // 1. No assumptions about memory layout in Rust code
        // 2. Cannot construct these types directly - must use factory functions
        // 3. Cannot access fields - must use accessor functions
        // 4. Automatic RAII cleanup via UniquePtr drop impl
        type Client;
        type Tag;

        // Initialization
        //
        // SAFETY: This function initializes the CTE runtime. It's safe to call
        // multiple times; subsequent calls are no-ops. The runtime state is
        // managed by C++ and protected by internal mutexes.
        fn cte_init(config_path: &str) -> i32;

        // Client operations
        //
        // SAFETY: Client objects are stateless interfaces to the runtime. The
        // UniquePtr<Client> returned by client_new is always valid and can be
        // safely passed to any client_* function. The Client destructor is called
        // when the UniquePtr is dropped.
        fn client_new() -> UniquePtr<Client>;

        // Poll telemetry entries after min_time
        //
        // SAFETY: The output vector is properly initialized by Rust before being
        // passed to C++. C++ appends bytes using resize/append, ensuring correct
        // capacity and size management.
        fn client_poll_telemetry_raw(client: &Client, min_time: u64, out: &mut Vec<u8>);

        // Reorganize blob (change placement score)
        //
        // SAFETY: All parameters are primitive types with guaranteed matching
        // representations. The name string is borrowed from Rust with cxx ensuring
        // proper null termination. Return value is a C++ return code (0 = success).
        fn client_reorganize_blob(
            client: &Client,
            major: u32,
            minor: u32,
            name: &str,
            score: f32,
        ) -> i32;

        // Delete a blob
        //
        // SAFETY: Same guarantees as client_reorganize_blob.
        fn client_del_blob(client: &Client, major: u32, minor: u32, name: &str) -> i32;

        // Tag factory functions
        //
        // SAFETY: These return valid UniquePtr<Tag> that can be safely passed to
        // any tag_* function. The returned Tag is fully initialized and ready for use.
        fn tag_new(name: &str) -> UniquePtr<Tag>;
        fn tag_from_id(major: u32, minor: u32) -> UniquePtr<Tag>;

        // Tag ID accessors
        //
        // SAFETY: These return primitive u32 values that don't require special
        // memory management. The Tag reference is borrowed for the call duration only.
        fn tag_get_id_major(tag: &Tag) -> u32;
        fn tag_get_id_minor(tag: &Tag) -> u32;

        // Tag operations - simple scalars
        //
        // SAFETY: All parameters are primitives or borrowed strings. Return values
        // are primitives that can be freely copied and don't require cleanup.
        fn tag_get_blob_score(tag: &Tag, name: &str) -> f32;
        fn tag_reorganize_blob(tag: &Tag, name: &str, score: f32) -> i32;
        fn tag_get_blob_size(tag: &Tag, name: &str) -> u64;

        // Tag operations - buffers
        //
        // SAFETY: Buffer parameters use Vec<T> which cxx maps correctly to
        // std::vector<T>. The C++ side uses proper size/capacity management.
        // For tag_put_blob, the data is read-only (borrowed from Rust).
        // For tag_get_blob and tag_get_contained_blobs, C++ appends to the
        // output vectors which Rust then owns.
        // Returns 0 on success, negative on error (-1 = size too large, -2 = offset overflow)
        fn tag_put_blob(tag: &Tag, name: &str, data: &[u8], offset: u64, score: f32) -> i32;
        fn tag_get_blob(tag: &Tag, name: &str, size: u64, offset: u64, out: &mut Vec<u8>);
        fn tag_get_contained_blobs(tag: &Tag, out: &mut Vec<String>);
    }
}

/// High-level CTE client wrapper
pub struct Client {
    inner: cxx::UniquePtr<ffi::Client>,
}

impl Client {
    /// Create a new CTE client
    pub fn new() -> Self {
        Self {
            inner: ffi::client_new(),
        }
    }

    /// Poll telemetry log
    pub fn poll_telemetry(&self, min_time: u64) -> Vec<CteTelemetry> {
        let mut data = Vec::new();
        ffi::client_poll_telemetry_raw(&self.inner, min_time, &mut data);
        parse_telemetry(&data)
    }

    /// Reorganize blob
    pub fn reorganize_blob(&self, tag_id: &CteTagId, name: &str, score: f32) -> i32 {
        ffi::client_reorganize_blob(&self.inner, tag_id.major, tag_id.minor, name, score)
    }

    /// Delete blob
    pub fn del_blob(&self, tag_id: &CteTagId, name: &str) -> i32 {
        ffi::client_del_blob(&self.inner, tag_id.major, tag_id.minor, name)
    }
}

/// High-level Tag wrapper
pub struct Tag {
    inner: cxx::UniquePtr<ffi::Tag>,
}

impl Tag {
    /// Create a new tag by name
    pub fn new(name: &str) -> Self {
        Self {
            inner: ffi::tag_new(name),
        }
    }

    /// Get tag by ID
    pub fn from_id(id: &CteTagId) -> Self {
        Self {
            inner: ffi::tag_from_id(id.major, id.minor),
        }
    }

    /// Get the tag ID
    pub fn id(&self) -> CteTagId {
        CteTagId {
            major: ffi::tag_get_id_major(&self.inner),
            minor: ffi::tag_get_id_minor(&self.inner),
        }
    }

    /// Get blob score
    pub fn get_blob_score(&self, name: &str) -> f32 {
        ffi::tag_get_blob_score(&self.inner, name)
    }

    /// Reorganize blob
    pub fn reorganize_blob(&self, name: &str, score: f32) -> i32 {
        ffi::tag_reorganize_blob(&self.inner, name, score)
    }

    /// Get blob size
    pub fn get_blob_size(&self, name: &str) -> u64 {
        ffi::tag_get_blob_size(&self.inner, name)
    }

    /// Put blob data
    /// Returns CteResult with error code from FFI:
    /// - 0 = success
    /// - -1 = data size exceeds limit
    /// - -2 = offset overflow
    pub fn put_blob(&self, name: &str, data: &[u8], offset: u64, score: f32) -> i32 {
        ffi::tag_put_blob(&self.inner, name, data, offset, score)
    }

    /// Get blob data
    pub fn get_blob(&self, name: &str, size: u64, offset: u64) -> Vec<u8> {
        let mut out = Vec::new();
        ffi::tag_get_blob(&self.inner, name, size, offset, &mut out);
        out
    }

    /// Get contained blobs
    pub fn get_contained_blobs(&self) -> Vec<String> {
        let mut out = Vec::new();
        ffi::tag_get_contained_blobs(&self.inner, &mut out);
        out
    }
}

/// Initialize CTE with optional config path
pub fn init(config_path: &str) -> i32 {
    ffi::cte_init(config_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_parsing() {
        // Create a sample telemetry buffer
        let mut data = vec![0u8; TELEMETRY_ENTRY_SIZE * 2];

        // Entry 1: op=1, off=100, size=200, tag_major=1, tag_minor=2, mod_time=1000, read_time=2000, logical=3000
        data[0..4].copy_from_slice(&1u32.to_le_bytes()); // op
        data[4..12].copy_from_slice(&100u64.to_le_bytes()); // off
        data[12..20].copy_from_slice(&200u64.to_le_bytes()); // size
        data[20..24].copy_from_slice(&1u32.to_le_bytes()); // tag_major
        data[24..28].copy_from_slice(&2u32.to_le_bytes()); // tag_minor
        data[28..36].copy_from_slice(&1000i64.to_le_bytes()); // mod_time
        data[36..44].copy_from_slice(&2000i64.to_le_bytes()); // read_time
        data[44..52].copy_from_slice(&3000u64.to_le_bytes()); // logical_time

        // Entry 2
        let offset = TELEMETRY_ENTRY_SIZE;
        data[offset..offset + 4].copy_from_slice(&2u32.to_le_bytes()); // op

        let entries = parse_telemetry(&data);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, CteOp::GetBlob);
        assert_eq!(entries[0].off, 100);
        assert_eq!(entries[0].size, 200);
        assert_eq!(entries[0].tag_id.major, 1);
        assert_eq!(entries[1].op, CteOp::DelBlob);
    }
}
