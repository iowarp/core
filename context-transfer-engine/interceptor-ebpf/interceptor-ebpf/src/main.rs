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

//! eBPF kernel program for I/O syscall interception.
//!
//! This program hooks into tracepoints for openat, read, write, and close syscalls
//! and sends events to a ring buffer for user-space processing.

#![no_main]
#![no_std]

use aya_ebpf::{
    macros::{map, tracepoint},
    maps::{LruHashMap, RingBuf},
    programs::TracePointContext,
    EbpfContext,
};
use interceptor_ebpf_common::{IoEvent, IoOp};

/// Ring buffer for sending events to user-space.
#[map]
static EVENTS: RingBuf = RingBuf::with_byte_size(1024 * 1024, 0);

/// Map for tracking PIDs in the process tree.
/// Key: PID (u32), Value: placeholder (1_u32)
/// Using LRU HashMap for automatic eviction of dead processes.
#[map]
static TRACKED_PIDS: LruHashMap<u32, u32> = LruHashMap::with_max_entries(1024, 0);

/// Maximum path length for openat syscall.
const MAX_PATH_LEN: usize = 256;

/// Maximum buffer size to capture for read/write operations.
const MAX_BUFFER_CAPTURE: usize = 64;

// ============================================================================
// Tracepoint Argument Structures
// ============================================================================
// These structures match the kernel tracepoint format.
// See: /sys/kernel/debug/tracing/events/syscalls/sys_enter_*/format

/// Argument structure for sys_enter_openat tracepoint.
/// Format: dfd:i32, filename:ptr, flags:i32, mode:u64
#[repr(C)]
struct SysEnterOpenatArgs {
    /// Directory fd (AT_FDCWD = -100 for relative paths)
    dfd: i32,
    /// Pointer to filename string
    filename: *const u8,
    /// Open flags (O_RDONLY, O_WRONLY, O_RDWR, etc.)
    flags: i32,
    /// Mode for file creation (permissions)
    mode: u64,
}

/// Argument structure for sys_exit_openat tracepoint.
/// Format: ret:i64
#[repr(C)]
struct SysExitOpenatArgs {
    /// Return value (fd on success, negative error on failure)
    ret: i64,
}

/// Argument structure for sys_enter_read tracepoint.
/// Format: fd:i32, buf:ptr, count:u64
#[repr(C)]
struct SysEnterReadArgs {
    /// File descriptor to read from
    fd: i32,
    /// Pointer to buffer for read data
    buf: *const u8,
    /// Number of bytes to read
    count: u64,
}

/// Argument structure for sys_exit_read tracepoint.
/// Format: ret:i64
#[repr(C)]
struct SysExitReadArgs {
    /// Return value (bytes read on success, negative error on failure)
    ret: i64,
}

/// Argument structure for sys_enter_write tracepoint.
/// Format: fd:i32, buf:ptr, count:u64
#[repr(C)]
struct SysEnterWriteArgs {
    /// File descriptor to write to
    fd: i32,
    /// Pointer to buffer with write data
    buf: *const u8,
    /// Number of bytes to write
    count: u64,
}

/// Argument structure for sys_exit_write tracepoint.
/// Format: ret:i64
#[repr(C)]
struct SysExitWriteArgs {
    /// Return value (bytes written on success, negative error on failure)
    ret: i64,
}

/// Argument structure for sys_enter_close tracepoint.
/// Format: fd:i32
#[repr(C)]
struct SysEnterCloseArgs {
    /// File descriptor to close
    fd: i32,
}

/// Argument structure for sys_exit_close tracepoint.
/// Format: ret:i64
#[repr(C)]
struct SysExitCloseArgs {
    /// Return value (0 on success, negative error on failure)
    ret: i64,
}

/// Argument structure for sched_process_fork tracepoint.
#[repr(C)]
struct SchedProcessForkArgs {
    parent_pid: i32,
    parent_comm: *const u8,
    child_pid: i32,
    child_comm: *const u8,
}

/// Argument structure for sched_process_exit tracepoint.
#[repr(C)]
struct SchedProcessExitArgs {
    pid: i32,
    prio: i32,
    exit_code: i32,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Reads a string from user-space memory with a maximum length.
///
/// # Arguments
/// * `ptr` - Pointer to user-space string.
/// * `buf` - Buffer to copy the string into.
///
/// # Returns
/// The number of bytes read.
unsafe fn read_user_string(ptr: *const u8, buf: &mut [u8]) -> usize {
    if ptr.is_null() {
        return 0;
    }

    let mut i = 0usize;
    while i < buf.len() {
        let byte = core::ptr::read_volatile(ptr.add(i));
        if byte == 0 {
            break;
        }
        buf[i] = byte;
        i += 1;
    }
    i
}

/// Reserve space in ring buffer and emit an event.
///
/// # Arguments
/// * `event` - The event to send.
#[inline(always)]
unsafe fn emit_event(event: IoEvent) {
    if let Some(mut record) = EVENTS.reserve::<IoEvent>(0) {
        record.write(event);
        record.submit(0);
    }
}

/// Get PID/TID from tracepoint context.
///
/// In eBPF tracepoints, the context contains process information.
/// We extract PID and TID from the current task.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall arguments.
///
/// # Returns
/// (pid, tid) tuple
#[inline(always)]
fn get_pid_tid() -> (u32, u32) {
    // In aya-ebpf, we use bpf_get_current_pid_tgid() helper
    // This returns (tgid << 32) | pid where tgid is the process ID
    // and pid is actually the thread ID in kernel terms
    let pid_tgid = aya_ebpf::helpers::bpf_get_current_pid_tgid();
    let pid = (pid_tgid >> 32) as u32;
    let tid = (pid_tgid & 0xFFFFFFFF) as u32;
    (pid, tid)
}

/// Check if a PID is in the tracked set.
#[inline(always)]
fn is_pid_tracked(pid: u32) -> bool {
    unsafe { TRACKED_PIDS.get(&pid).is_some() }
}

// ============================================================================
// Tracepoint Handlers
// ============================================================================
// Tracepoint Handlers
// ============================================================================

/// Tracepoint for sched_process_fork - track child PIDs for process tree.
#[tracepoint]
pub fn sched_process_fork(ctx: TracePointContext) -> Result<(), i32> {
    let args = unsafe { &*(ctx.as_ptr() as *const SchedProcessForkArgs) };

    let pid_tgid = aya_ebpf::helpers::bpf_get_current_pid_tgid();
    let parent_pid = (pid_tgid >> 32) as u32;
    let child_pid = args.child_pid as u32;

    // If parent is tracked, add child to tracked set
    if is_pid_tracked(parent_pid) {
        if let Err(ret) = unsafe { TRACKED_PIDS.insert(&child_pid, &1_u32, 0) } {
            return Err(ret);
        }
    }

    Ok(())
}

/// Tracepoint for sched_process_exit - clean up PIDs on process exit.
#[tracepoint]
pub fn sched_process_exit(ctx: TracePointContext) -> Result<(), i32> {
    let args = unsafe { &*(ctx.as_ptr() as *const SchedProcessExitArgs) };
    let pid = args.pid as u32;

    // Remove from tracked set if present (ignore errors)
    let _ = unsafe { TRACKED_PIDS.remove(&pid) };

    Ok(())
}

/// Tracepoint for sys_enter_openat - intercept file open operations.
///
/// This tracepoint fires when a process calls openat() syscall.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall arguments.
#[tracepoint]
pub fn sys_enter_openat(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // FILTER: Only process if PID is tracked
    if !is_pid_tracked(pid) {
        return;
    }

    // Read tracepoint args using proper structure cast
    let args = unsafe { &*(ctx.as_ptr() as *const SysEnterOpenatArgs) };

    let filename_ptr = args.filename;
    let mut event = IoEvent {
        op: IoOp::Open,
        pid,
        tid,
        fd: -1,
        size: 0,
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        read_user_string(filename_ptr, &mut event.path);
    }

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_exit_openat - capture the returned fd.
///
/// This tracepoint fires after openat() returns.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall return value.
#[tracepoint]
pub fn sys_exit_openat(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // FILTER: Only process if PID is tracked
    if !is_pid_tracked(pid) {
        return;
    }

    // Read return value from tracepoint args
    let args = unsafe { &*(ctx.as_ptr() as *const SysExitOpenatArgs) };
    let ret = args.ret as i32;

    // We use fd=-2 to indicate "return value" for exit tracepoints
    // The actual fd is encoded in the size field
    let event = IoEvent {
        op: IoOp::OpenReturn,
        pid,
        tid,
        fd: -2,           // Special marker for return value
        size: ret as u64, // Store return value/returned fd in size field
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_enter_read - intercept read operations.
///
/// This tracepoint fires when a process calls read() syscall.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall arguments.
#[tracepoint]
pub fn sys_enter_read(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // FILTER: Only process if PID is tracked
    if !is_pid_tracked(pid) {
        return;
    }

    // Read tracepoint args using proper structure cast
    let args = unsafe { &*(ctx.as_ptr() as *const SysEnterReadArgs) };

    let fd = args.fd;
    let count = args.count;

    let event = IoEvent {
        op: IoOp::Read,
        pid,
        tid,
        fd,
        size: count,
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_exit_read - capture read result.
///
/// This tracepoint fires after read() returns.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall return value.
#[tracepoint]
pub fn sys_exit_read(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // Read return value from tracepoint args
    let args = unsafe { &*(ctx.as_ptr() as *const SysExitReadArgs) };
    let ret = args.ret;

    // We don't know fd here - it was in the enter tracepoint
    // User-space will correlate using pid/tid
    let event = IoEvent {
        op: IoOp::ReadReturn,
        pid,
        tid,
        fd: -1,           // Unknown in exit
        size: ret as u64, // Store return value in size field
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_enter_write - intercept write operations.
///
/// This tracepoint fires when a process calls write() syscall.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall arguments.
#[tracepoint]
pub fn sys_enter_write(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // Read tracepoint args using proper structure cast
    let args = unsafe { &*(ctx.as_ptr() as *const SysEnterWriteArgs) };

    let fd = args.fd;
    let buf_ptr = args.buf;
    let count = args.count as usize;

    let mut event = IoEvent {
        op: IoOp::Write,
        pid,
        tid,
        fd,
        size: count as u64,
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    // Capture beginning of write buffer for analysis
    let capture_len = if count < MAX_BUFFER_CAPTURE {
        count
    } else {
        MAX_BUFFER_CAPTURE
    };
    if !buf_ptr.is_null() && capture_len > 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(buf_ptr, event.buffer.as_mut_ptr(), capture_len);
        }
    }

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_exit_write - capture write result.
///
/// This tracepoint fires after write() returns.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall return value.
#[tracepoint]
pub fn sys_exit_write(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // Read return value from tracepoint args
    let args = unsafe { &*(ctx.as_ptr() as *const SysExitWriteArgs) };
    let ret = args.ret;

    let event = IoEvent {
        op: IoOp::WriteReturn,
        pid,
        tid,
        fd: -1,           // Unknown in exit
        size: ret as u64, // Store return value in size field
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_enter_close - intercept close operations.
///
/// This tracepoint fires when a process calls close() syscall.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall arguments.
#[tracepoint]
pub fn sys_enter_close(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // Read tracepoint args using proper structure cast
    let args = unsafe { &*(ctx.as_ptr() as *const SysEnterCloseArgs) };

    let fd = args.fd;

    let event = IoEvent {
        op: IoOp::Close,
        pid,
        tid,
        fd,
        size: 0,
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        emit_event(event);
    }
}

/// Tracepoint for sys_exit_close - capture close result.
///
/// This tracepoint fires after close() returns.
///
/// # Arguments
/// * `ctx` - Tracepoint context containing syscall return value.
#[tracepoint]
pub fn sys_exit_close(ctx: TracePointContext) {
    let (pid, tid) = get_pid_tid();

    // Read return value from tracepoint args
    let args = unsafe { &*(ctx.as_ptr() as *const SysExitCloseArgs) };
    let ret = args.ret;

    let event = IoEvent {
        op: IoOp::CloseReturn,
        pid,
        tid,
        fd: -1,           // Unknown in exit
        size: ret as u64, // Store return value in size field
        path: [0u8; MAX_PATH_LEN],
        buffer: [0u8; MAX_BUFFER_CAPTURE],
    };

    unsafe {
        emit_event(event);
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}
