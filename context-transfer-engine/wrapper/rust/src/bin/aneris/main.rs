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

//! Aneris Profiler - Combined Subprocess + Telemetry Capture
//!
//! This binary spawns a subprocess with CTE I/O interception and captures
//! telemetry in real-time, displaying results as they occur.
//!
//! Usage:
//!   LD_LIBRARY_PATH=~/clio-core/build/bin:$LD_LIBRARY_PATH \
//!     CHI_WITH_RUNTIME=1 \
//!     aneris-profiler <command> [args...]
//!
//! Example:
//!   aneris-profiler ior -t 1m -b 16m -s 16

use std::process::{Command, Stdio};
use std::time::Duration;
use wrp_cte::sync::init;
use wrp_cte::sync::Client;

fn main() {
    // Parse command line
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <command> [args...]", args[0]);
        eprintln!("Example: {} ior -t 1m -b 16m -s 16", args[0]);
        std::process::exit(1);
    }

    let executable = &args[1];
    let exec_args = &args[2..];

    println!("=== Aneris Profiler ===");
    println!("Executable: {}", executable);
    println!("");

    // Initialize CTE
    println!("[1/3] Initializing CTE runtime...");
    if let Err(e) = init("") {
        eprintln!("Failed to initialize CTE: {}", e);
        std::process::exit(1);
    }
    println!("      ✓ CTE runtime initialized\n");

    // Get build directory with multiple fallback strategies
    let build_dir = std::env::var("IOWARP_BUILD_DIR")
        .or_else(|_| std::env::var("CMAKE_BINARY_DIR"))
        .unwrap_or_else(|_| {
            // Try to detect from current executable path
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_string_lossy().to_string()))
                .unwrap_or_else(|| "/tmp".to_string())
        });

    // The POSIX adapter is in bin/ not lib/
    // But build_dir might already be the bin directory if detected from current_exe
    let posix_adapter = if build_dir.ends_with("/bin") || build_dir.ends_with("/bin/") {
        format!("{}/libwrp_cte_posix.so", build_dir)
    } else {
        format!("{}/bin/libwrp_cte_posix.so", build_dir)
    };

    // Check adapter
    if !std::path::Path::new(&posix_adapter).exists() {
        eprintln!("[!] Warning: POSIX adapter not found at {}", posix_adapter);
        eprintln!("    I/O interception will not work.");
    } else {
        println!("[✓] POSIX adapter found");
    }

    // Give runtime time to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Spawn subprocess with LD_PRELOAD
    println!("[2/3] Starting subprocess with I/O interception...");
    let mut child = Command::new(executable)
        .args(exec_args)
        .env("LD_PRELOAD", &posix_adapter)
        .env_remove("CHI_WITH_RUNTIME") // Child should NOT start its own runtime
        .env(
            "LD_LIBRARY_PATH",
            format!(
                "{}:{}",
                if build_dir.ends_with("/bin") || build_dir.ends_with("/bin/") {
                    build_dir.clone()
                } else {
                    format!("{}/bin", build_dir)
                },
                std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
            ),
        )
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn subprocess");

    println!("      ✓ Subprocess started (PID: {})\n", child.id());
    println!("=== Telemetry Capture Active ===\n");

    // Wait for subprocess
    let status = child.wait().expect("Failed to wait for subprocess");

    // Give runtime time to catch final operations
    std::thread::sleep(Duration::from_millis(500));

    // Create CTE client for telemetry polling
    let client = match Client::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Warning: Failed to create telemetry client: {}", e);
            println!("\nSubprocess exited with: {:?}", status.code());
            return;
        }
    };

    // Poll and display final results
    println!("\n=== Telemetry Summary ===");

    // Poll telemetry
    match client.poll_telemetry(0) {
        Ok(telemetry) => {
            if telemetry.is_empty() {
                println!("No telemetry entries captured.");
            } else {
                println!("Captured {} telemetry entries\n", telemetry.len());

                // Display telemetry table
                println!(
                    "{:<20} {:>12} {:>20}",
                    "Operation", "Size (bytes)", "Tag ID"
                );
                println!("{}", "-".repeat(60));

                let mut total_ops: u64 = 0;
                let mut total_bytes: u64 = 0;
                let mut write_bytes: u64 = 0;
                let mut read_bytes: u64 = 0;

                for entry in &telemetry {
                    println!(
                        "{:<20} {:>12} {:>20}",
                        format!("{:?}", entry.op),
                        entry.size,
                        format!("{}.{}", entry.tag_id.major, entry.tag_id.minor)
                    );

                    total_ops += 1;
                    total_bytes += entry.size;

                    // Track read/write separately
                    match entry.op {
                        wrp_cte::ffi::CteOp::PutBlob => write_bytes += entry.size,
                        wrp_cte::ffi::CteOp::GetBlob => read_bytes += entry.size,
                        _ => {}
                    }
                }

                // Summary statistics
                let avg_size = if total_ops > 0 {
                    total_bytes / total_ops
                } else {
                    0
                };

                println!("\n{}", "-".repeat(60));
                println!("=== Summary ===");
                println!("Total operations: {}", total_ops);
                println!(
                    "Total data transferred: {} bytes ({} MB)",
                    total_bytes,
                    total_bytes / (1024 * 1024)
                );
                println!(
                    "  - Writes: {} bytes ({} MB)",
                    write_bytes,
                    write_bytes / (1024 * 1024)
                );
                println!(
                    "  - Reads: {} bytes ({} MB)",
                    read_bytes,
                    read_bytes / (1024 * 1024)
                );
                println!("Average size: {} bytes", avg_size);
            }
        }
        Err(e) => {
            eprintln!("Telemetry poll failed: {}", e);
            eprintln!("This can happen if:");
            eprintln!("  - Telemetry collection is disabled");
            eprintln!("  - The runtime hasn't processed operations yet");
        }
    }

    println!("\nSubprocess exited with: {:?}", status.code());
}
