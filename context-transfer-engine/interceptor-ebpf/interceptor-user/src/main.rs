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

//! User-space controller for eBPF I/O interceptor.
//!
//! This program loads the eBPF kernel program, attaches tracepoints,
//! and processes events from perf arrays.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aya::programs::TracePoint;
use aya::{include_bytes_aligned, Ebpf};
use clap::Parser;
use interceptor_ebpf_common::IoEvent;
use interceptor_ebpf_common::IoOp;
use tokio::signal;
use tracing::info;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

/// Command-line arguments for the interceptor.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Filter by process ID (optional)
    #[arg(short, long)]
    pid: Option<u32>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Show buffer contents for read/write operations
    #[arg(short, long)]
    show_buffer: bool,
}

/// Load and attach eBPF programs.
///
/// # Arguments
/// * `bpf` - The loaded BPF object.
///
/// # Returns
/// Result indicating success or failure.
fn attach_tracepoints(bpf: &mut Ebpf) -> Result<(), anyhow::Error> {
    // Attach sys_enter_openat tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_enter_openat")
        .ok_or_else(|| anyhow::anyhow!("sys_enter_openat program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_enter_openat")?;
    info!("Attached sys_enter_openat tracepoint");

    // Attach sys_exit_openat tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_exit_openat")
        .ok_or_else(|| anyhow::anyhow!("sys_exit_openat program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_exit_openat")?;
    info!("Attached sys_exit_openat tracepoint");

    // Attach sys_enter_read tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_enter_read")
        .ok_or_else(|| anyhow::anyhow!("sys_enter_read program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_enter_read")?;
    info!("Attached sys_enter_read tracepoint");

    // Attach sys_exit_read tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_exit_read")
        .ok_or_else(|| anyhow::anyhow!("sys_exit_read program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_exit_read")?;
    info!("Attached sys_exit_read tracepoint");

    // Attach sys_enter_write tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_enter_write")
        .ok_or_else(|| anyhow::anyhow!("sys_enter_write program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_enter_write")?;
    info!("Attached sys_enter_write tracepoint");

    // Attach sys_exit_write tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_exit_write")
        .ok_or_else(|| anyhow::anyhow!("sys_exit_write program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_exit_write")?;
    info!("Attached sys_exit_write tracepoint");

    // Attach sys_enter_close tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_enter_close")
        .ok_or_else(|| anyhow::anyhow!("sys_enter_close program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_enter_close")?;
    info!("Attached sys_enter_close tracepoint");

    // Attach sys_exit_close tracepoint
    let program: &mut TracePoint = bpf
        .program_mut("sys_exit_close")
        .ok_or_else(|| anyhow::anyhow!("sys_exit_close program not found"))?
        .try_into()?;
    program.load()?;
    program.attach("syscalls", "sys_exit_close")?;
    info!("Attached sys_exit_close tracepoint");

    Ok(())
}

/// Format an I/O event for display.
///
/// # Arguments
/// * `event` - The I/O event to format.
/// * `show_buffer` - Whether to show buffer contents.
/// * `filter_pid` - Optional PID filter.
///
/// # Returns
/// Formatted string representation, or None if filtered out.
fn format_event(event: &IoEvent, show_buffer: bool, filter_pid: Option<u32>) -> Option<String> {
    // Filter by PID if specified
    if let Some(pid) = filter_pid {
        if event.pid != pid {
            return None;
        }
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    match event.op {
        IoOp::Open => {
            let path = event.path_str().unwrap_or("<invalid utf-8>");
            Some(format!(
                "[{}] OPEN(pid={}, tid={}, path={})",
                timestamp, event.pid, event.tid, path
            ))
        }
        IoOp::OpenReturn => {
            let ret = event.size as i64;
            let status = if ret >= 0 { "fd" } else { "error" };
            Some(format!(
                "[{}] OPEN_RETURN(pid={}, tid={}, {}={})",
                timestamp, event.pid, event.tid, status, ret
            ))
        }
        IoOp::Read => {
            let mut msg = format!(
                "[{}] READ(pid={}, tid={}, fd={}, size={})",
                timestamp, event.pid, event.tid, event.fd, event.size
            );
            if show_buffer && !event.buffer.iter().all(|&b| b == 0) {
                msg.push_str(&format!(" buffer={:?}", event.buffer_bytes()));
            }
            Some(msg)
        }
        IoOp::ReadReturn => {
            let ret = event.size as i64;
            Some(format!(
                "[{}] READ_RETURN(pid={}, tid={}, bytes_read={})",
                timestamp, event.pid, event.tid, ret
            ))
        }
        IoOp::Write => {
            let mut msg = format!(
                "[{}] WRITE(pid={}, tid={}, fd={}, size={})",
                timestamp, event.pid, event.tid, event.fd, event.size
            );
            if show_buffer {
                msg.push_str(&format!(" buffer={:?}", event.buffer_bytes()));
            }
            Some(msg)
        }
        IoOp::WriteReturn => {
            let ret = event.size as i64;
            Some(format!(
                "[{}] WRITE_RETURN(pid={}, tid={}, bytes_written={})",
                timestamp, event.pid, event.tid, ret
            ))
        }
        IoOp::Close => {
            Some(format!(
                "[{}] CLOSE(pid={}, tid={}, fd={})",
                timestamp, event.pid, event.tid, event.fd
            ))
        }
        IoOp::CloseReturn => {
            let ret = event.size as i64;
            let status = if ret == 0 { "success" } else { "error" };
            Some(format!(
                "[{}] CLOSE_RETURN(pid={}, tid={}, {}={})",
                timestamp, event.pid, event.tid, status, ret
            ))
        }
    }
}

/// Process events from the perf array.
///
/// # Arguments
/// * `event_data` - Raw event data from the perf buffer.
/// * `show_buffer` - Whether to show buffer contents.
/// * `filter_pid` - Optional PID filter.
fn handle_event(event_data: &[u8], show_buffer: bool, filter_pid: Option<u32>) {
    if event_data.len() < std::mem::size_of::<IoEvent>() {
        tracing::warn!("Received undersized event: {} bytes", event_data.len());
        return;
    }

    // SAFETY: We've verified the size
    let event: IoEvent = unsafe { std::ptr::read_unaligned(event_data.as_ptr() as *const IoEvent) };

    if let Some(formatted) = format_event(&event, show_buffer, filter_pid) {
        println!("{}", formatted);
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Parse command-line arguments
    let args = Args::parse();

    // Initialize logging
    let filter = if args.verbose {
        EnvFilter::from_default_env()
            .add_directive(tracing::Level::DEBUG.into())
    } else {
        EnvFilter::from_default_env()
            .add_directive(tracing::Level::INFO.into())
    };
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::ACTIVE)
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    info!("Starting eBPF I/O interceptor...");

    // Rlimit for memlock
    let rlimit = libc::rlimit {
        rlim_cur: 1024 * 1024 * 100, // 100 MB
        rlim_max: 1024 * 1024 * 100,
    };
    if unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &rlimit) } != 0 {
        tracing::warn!("Failed to set rlimit for memlock. This may be required for eBPF maps.");
    }

    // Load the eBPF program
    info!("Loading eBPF program...");
    // The eBPF binary path is relative to the workspace root target directory
    // When built from workspace: ws-target/bpfel-unknown-none/ebpf/interceptor-ebpf
    // When included from user/src/main.rs: ../../target/bpfel-unknown-none/ebpf/interceptor-ebpf
    let mut bpf = Ebpf::load(include_bytes_aligned!("../../target/bpfel-unknown-none/ebpf/interceptor-ebpf"))?;
    info!("eBPF program loaded successfully");

    // Attach tracepoints
    attach_tracepoints(&mut bpf)?;

    // Get the ring buffer for events
    // In Aya 0.13, take_map() returns Option<Map> directly
    let events_map = bpf
        .take_map("EVENTS")
        .ok_or_else(|| anyhow::anyhow!("EVENTS map not found"))?;
    let mut events_rb: aya::maps::RingBuf<_> = events_map.try_into()?;

    // Running flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Handle shutdown signals
    tokio::spawn(async move {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        info!("Shutdown signal received, stopping interceptor...");
        running_clone.store(false, Ordering::Relaxed);
    });

    // Process events from ring buffer
    info!("Starting event processing loop...");
    
    let show_buffer = args.show_buffer;
    let filter_pid = args.pid;
    
    // Read from the ring buffer in a loop
    // RingBuf requires us to poll for new events asynchronously
    loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        // In Aya 0.13, RingBuf::next() returns Option<RingBufItem<'_>>
        // We'll use a simple polling approach
        match events_rb.next() {
            Some(bytes) => {
                handle_event(&bytes, show_buffer, filter_pid);
            }
            None => {
                // No events available, yield to scheduler
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        }
    }

    info!("eBPF I/O interceptor stopped");
    Ok(())
}