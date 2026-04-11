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

//! Shared types for eBPF I/O interceptor communication.
//!
//! This crate provides common data structures used by both the eBPF kernel
//! program and user-space controller for I/O syscall interception.

#![no_std]
#![allow(dead_code)]

/// Maximum path length for file operations.
pub const MAX_PATH_LEN: usize = 256;

/// Maximum buffer size to capture for read/write operations.
pub const MAX_BUFFER_CAPTURE: usize = 64;

/// I/O operation types.
///
/// This enum represents different I/O operations that can be intercepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IoOp {
    /// File open operation (openat syscall enter)
    Open = 0,
    /// File open return (openat syscall exit)
    OpenReturn = 1,
    /// File read operation (read syscall enter)
    Read = 2,
    /// File read return (read syscall exit)
    ReadReturn = 3,
    /// File write operation (write syscall enter)
    Write = 4,
    /// File write return (write syscall exit)
    WriteReturn = 5,
    /// File close operation (close syscall enter)
    Close = 6,
    /// File close return (close syscall exit)
    CloseReturn = 7,
}

impl IoOp {
    /// Returns true if this is an enter (entry) event.
    pub fn is_enter(&self) -> bool {
        matches!(self, IoOp::Open | IoOp::Read | IoOp::Write | IoOp::Close)
    }

    /// Returns true if this is a return (exit) event.
    pub fn is_return(&self) -> bool {
        matches!(
            self,
            IoOp::OpenReturn | IoOp::ReadReturn | IoOp::WriteReturn | IoOp::CloseReturn
        )
    }

    /// Returns the corresponding return operation for an enter operation.
    pub fn return_op(&self) -> Option<IoOp> {
        match self {
            IoOp::Open => Some(IoOp::OpenReturn),
            IoOp::Read => Some(IoOp::ReadReturn),
            IoOp::Write => Some(IoOp::WriteReturn),
            IoOp::Close => Some(IoOp::CloseReturn),
            _ => None,
        }
    }
}

/// I/O event structure for eBPF communication.
///
/// This structure is sent from the eBPF kernel program to user-space
/// via a ring buffer for each intercepted syscall.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct IoEvent {
    /// Type of I/O operation
    pub op: IoOp,
    /// Process ID
    pub pid: u32,
    /// Thread ID
    pub tid: u32,
    /// File descriptor (-1 if not applicable or unknown)
    pub fd: i32,
    /// Size parameter (count for read/write, return value for syscall returns)
    pub size: u64,
    /// File path for open operations (null-terminated string)
    pub path: [u8; MAX_PATH_LEN],
    /// Captured buffer data for read/write operations
    pub buffer: [u8; MAX_BUFFER_CAPTURE],
}

impl IoEvent {
    /// Creates a new IoEvent with default values.
    pub fn new() -> Self {
        IoEvent {
            op: IoOp::Open,
            pid: 0,
            tid: 0,
            fd: -1,
            size: 0,
            path: [0u8; MAX_PATH_LEN],
            buffer: [0u8; MAX_BUFFER_CAPTURE],
        }
    }

    /// Returns the file path as a string slice if valid UTF-8.
    pub fn path_str(&self) -> Option<&str> {
        let end = self.path.iter().position(|&b| b == 0)?;
        core::str::from_utf8(&self.path[..end]).ok()
    }

    /// Returns the buffer data as a byte slice.
    pub fn buffer_bytes(&self) -> &[u8] {
        // Find first null byte or use full length
        let end = self
            .buffer
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_BUFFER_CAPTURE);
        &self.buffer[..end]
    }
}

impl Default for IoEvent {
    fn default() -> Self {
        Self::new()
    }
}

// Conditionally include aya support for user-space
#[cfg(feature = "userspace")]
pub mod userspace {
    //! User-space support for eBPF communication.

    use super::*;

    /// Event iterator for reading from ring buffer.
    pub struct EventIterator {
        /// Current position in the buffer
        pos: usize,
        /// Total readable length
        len: usize,
        /// Raw bytes buffer
        bytes: [u8; 1024 * 1024],
    }

    impl EventIterator {
        /// Creates a new event iterator with the given buffer.
        pub fn new() -> Self {
            EventIterator {
                pos: 0,
                len: 0,
                bytes: [0u8; 1024 * 1024],
            }
        }

        /// Reads an event from the current position.
        pub fn read_event(&mut self) -> Option<IoEvent> {
            if self.pos + core::mem::size_of::<IoEvent>() > self.len {
                return None;
            }

            let event_bytes = &self.bytes[self.pos..self.pos + core::mem::size_of::<IoEvent>()];
            // SAFETY: We know the size and alignment match
            let event =
                unsafe { core::ptr::read_unaligned(event_bytes.as_ptr() as *const IoEvent) };
            self.pos += core::mem::size_of::<IoEvent>();
            Some(event)
        }
    }

    impl Default for EventIterator {
        fn default() -> Self {
            Self::new()
        }
    }
}
