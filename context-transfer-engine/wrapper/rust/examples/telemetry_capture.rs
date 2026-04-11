/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

use std::thread;
use std::time::Duration;

/// Example: Capture CTE Telemetry
///
/// This example demonstrates:
/// 1. Initializing the CTE runtime
/// 2. Creating tags and writing blobs
/// 3. Capturing telemetry data
/// 4. Displaying operation statistics
///
/// Run with:
///   CHI_WITH_RUNTIME=1 LD_LIBRARY_PATH=/home/neeraj/clio-core/build/bin:$LD_LIBRARY_PATH cargo run --example telemetry_capture

fn main() {
    println!("=== IOWarp CTE Telemetry Capture Example ===\n");

    // Import sync API for simplicity (works in main)
    use wrp_cte::sync::{init, Client, Tag};

    // Step 1: Initialize CTE runtime
    println!("[1/4] Initializing CTE runtime...");
    init("").expect("Failed to initialize CTE");
    println!("      ✓ CTE runtime initialized\n");

    // Step 2: Create client
    println!("[2/4] Creating CTE client...");
    let client = Client::new().expect("Failed to create client");
    println!("      ✓ Client created\n");

    // Step 3: Create tag and write some data
    println!("[3/4] Creating workload (PutBlob operations)...");
    let tag = Tag::new("telemetry_test");

    // Write several blobs with different sizes
    let test_data_sizes = vec![
        ("small_blob.bin", 1024usize),   // 1 KB
        ("medium_blob.bin", 64 * 1024),  // 64 KB
        ("large_blob.bin", 1024 * 1024), // 1 MB
    ];

    for (name, size) in &test_data_sizes {
        let data = vec![0xABu8; *size];
        match tag.put_blob_with_options(name, &data, 0, 1.0) {
            Ok(_) => {
                println!("      ✓ Wrote {} ({} bytes)", name, size);
            }
            Err(e) => {
                eprintln!("      ✗ Failed to write {}: {}", name, e);
                eprintln!("");
                eprintln!(
                    "      Note: This error typically means no storage devices are configured."
                );
                eprintln!("      To configure storage, add devices to ~/.chimaera/chimaera.yaml:");
                eprintln!("");
                eprintln!("      devices:");
                eprintln!("        - name: ram");
                eprintln!("          type: ramfs");
                eprintln!("          capacity: 1g");
                eprintln!("");
            }
        }

        // Small delay to ensure telemetry is captured
        thread::sleep(Duration::from_millis(10));
    }
    println!();

    // Step 4: Capture telemetry
    println!("[4/4] Capturing telemetry...");

    // Give the runtime time to process and generate telemetry
    thread::sleep(Duration::from_millis(100));

    // Try to poll telemetry - this may fail if telemetry is not yet available
    match client.poll_telemetry(0, 5.0) {
        Ok(telemetry) => {
            println!("      ✓ Captured {} telemetry entries\n", telemetry.len());

            // Display telemetry
            if !telemetry.is_empty() {
                println!("=== Telemetry Data ===");
                println!(
                    "{:<20} {:>12} {:>20}",
                    "Operation", "Size (bytes)", "Tag ID"
                );
                println!("{}", "-".repeat(60));

                for entry in &telemetry {
                    println!(
                        "{:<20} {:>12} {:>20}",
                        format!("{:?}", entry.op),
                        entry.size,
                        format!("{}.{}", entry.tag_id.major, entry.tag_id.minor)
                    );
                }

                // Summary statistics
                let total_size: u64 = telemetry.iter().map(|t| t.size).sum();
                let avg_size = total_size / telemetry.len() as u64;

                println!("\n=== Summary ===");
                println!("Total operations: {}", telemetry.len());
                println!(
                    "Total data: {} bytes ({} MB)",
                    total_size,
                    total_size / (1024 * 1024)
                );
                println!("Average size: {} bytes", avg_size);
            } else {
                println!("      ! No telemetry entries captured yet");
                println!("        (This is normal for the first run)");
            }
        }
        Err(e) => {
            eprintln!("      ! Telemetry poll returned error: {}", e);
            eprintln!("        This can happen if:");
            eprintln!("        - Telemetry collection is disabled");
            eprintln!("        - The runtime hasn't processed operations yet");
            eprintln!("        - No operations have completed in the polling window");
        }
    }

    println!("\n=== Telemetry capture complete ===");
}
