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

//! Blob Monitor - Proof of Concept
//!
//! Monitors CTE blob access patterns and auto-adjusts scores based on frecency.
//!
//! Architecture (Two-Map Design):
//!
//! 1. **Telemetry Stats Map** - tracks access patterns by offset
//!    - Key: (tag_id, offset) from CTE telemetry
//!    - Value: TelemetryStats { access_count, bytes_read, bytes_written, timestamps }
//!    - Populated from telemetry polling
//!    - Purpose: Understand I/O patterns at offset granularity
//!
//! 2. **Blob Registry Map** - tracks blobs by name for score updates
//!    - Key: (tag_id, blob_name) from tag.get_contained_blobs()
//!    - Value: BlobInfo { score, last_checked, access_count }
//!    - Populated from tag blob discovery
//!    - Purpose: THIS is what we use for calling reorganize_blob()
//!
//! 3. **Main Logic:**
//!    - Telemetry task updates TelemetryStats by offset
//!    - Registry task updates BlobRegistry by name
//!    - Main loop: for each blob in registry, calculate frecency → score
//!    - Call tag.reorganize_blob(blob_name, new_score) for actual CTE updates
//!
//! # Per-Tag Frecency Limitation
//!
//! **IMPORTANT**: This implementation tracks frecency at the TAG level, not the
//! individual blob level. All blobs within a tag share the same access patterns
//! from telemetry, which is aggregated by tag_id. This is a fundamental limitation
//! of the current telemetry API, which provides offset-based data but no blob-to-offset
//! mapping.
//!
//! For per-blob frecency tracking, CTE would need to expose:
//! 1. Blob-to-offset mappings for each tag
//! 2. Telemetry entries that include blob identification
//!
//! Current behavior: All blobs in a tag inherit the tag's aggregate frecency score.
//!
//! # Lock Ordering
//!
//! To prevent deadlocks, locks must ALWAYS be acquired in this order:
//! 1. `telemetry_stats` (read or write)
//! 2. `blob_registry` (read or write)
//! 3. `known_tags` (read or write) - MUST NEVER be held with other locks
//!
//! Use `with_read_locks()` helper to ensure correct ordering.
//!
//! Usage:
//!   blob_monitor [REFRESH_MS]
//!
//! Environment:
//!   CHI_WITH_RUNTIME=1 - Start embedded CTE runtime

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, timeout};

use wrp_cte::{Client, CteOp, CteTagId, Tag};

/// Unique identifier for telemetry tracking (offset-based).
/// We track by offset because that's what CTE telemetry gives us.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct TelemetryKey {
    tag_id: CteTagId,
    offset: u64,
}

/// Statistics tracked from CTE telemetry at offset granularity.
#[derive(Clone, Debug, Default)]
struct TelemetryStats {
    access_count: u64,
    bytes_read: u64,
    bytes_written: u64,
    first_seen: u64,
    last_seen: u64,
}

/// Unique identifier for blob registry (name-based).
/// We track by name because that's what reorganize_blob() needs.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct BlobKey {
    tag_id: CteTagId,
    blob_name: String,
}

/// Information tracked for each named blob.
#[derive(Clone, Debug)]
struct BlobInfo {
    score: f32,
    last_checked: u64,
    /// Aggregate access count (from telemetry for this tag)
    access_count: u64,
}

impl Default for BlobInfo {
    fn default() -> Self {
        Self {
            score: 0.5,
            last_checked: 0,
            access_count: 0,
        }
    }
}

/// Error rate limiter for preventing log spam.
struct ErrorBackoff {
    consecutive: u32,
}

impl ErrorBackoff {
    fn new() -> Self {
        Self { consecutive: 0 }
    }

    /// Returns true if we should log this error.
    /// Logs first 3 errors, then every 10th consecutive error.
    fn should_log(&mut self) -> bool {
        self.consecutive += 1;
        self.consecutive <= 3 || self.consecutive % 10 == 0
    }

    /// Reset after successful operation.
    fn reset(&mut self) {
        self.consecutive = 0;
    }
}

/// Shutdown management using broadcast channel for coordinated shutdown.
struct ShutdownManager {
    tx: broadcast::Sender<()>,
}

impl ShutdownManager {
    fn new() -> Self {
        let (tx, _) = broadcast::channel(1);
        Self { tx }
    }

    fn subscriber(&self) -> broadcast::Receiver<()> {
        self.tx.subscribe()
    }

    fn shutdown(&self) {
        // Ignore send errors (no receivers)
        let _ = self.tx.send(());
    }
}

/// Blob monitor state with two separate tracking systems.
struct BlobMonitor {
    /// Telemetry statistics (offset-based) - for understanding I/O patterns
    telemetry_stats: Arc<RwLock<HashMap<TelemetryKey, TelemetryStats>>>,
    /// Blob registry (name-based) - for score updates via reorganize_blob()
    blob_registry: Arc<RwLock<HashMap<BlobKey, BlobInfo>>>,
    /// Known tags discovered from telemetry
    known_tags: Arc<RwLock<Vec<CteTagId>>>,
}

impl BlobMonitor {
    fn new() -> Self {
        Self {
            telemetry_stats: Arc::new(RwLock::new(HashMap::new())),
            blob_registry: Arc::new(RwLock::new(HashMap::new())),
            known_tags: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Acquires locks in documented order: telemetry_stats, then blob_registry.
    /// 
    /// CRITICAL: known_tags must NEVER be held with these two locks.
    /// Clone known_tags before calling this or outside of lock scope.
    ///
    /// # Lock Ordering
    /// 1. telemetry_stats (read)
    /// 2. blob_registry (read)
    async fn with_read_locks<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&HashMap<TelemetryKey, TelemetryStats>, &HashMap<BlobKey, BlobInfo>) -> R,
    {
        let telemetry = self.telemetry_stats.read().await;
        let registry = self.blob_registry.read().await;
        f(&telemetry, &registry)
    }

    /// Calculate frecency (frequency + recency) for a blob.
    ///
    /// Formula: frecency = access_count / (1 + (current_time - last_seen))
    fn calculate_frecency(access_count: u64, last_seen: u64, current_time: u64) -> f64 {
        if access_count == 0 {
            return 0.0;
        }
        let time_delta = current_time.saturating_sub(last_seen);
        access_count as f64 / (1.0 + time_delta as f64)
    }

    /// Map frecency to score.
    ///
    /// - frecency > 10 -> 0.9 (hot)
    /// - frecency < 2  -> 0.2 (cold)
    /// - else          -> 0.5 (neutral)
    fn frecency_to_score(frecency: f64) -> f32 {
        if frecency > 10.0 {
            0.9
        } else if frecency < 2.0 {
            0.2
        } else {
            0.5
        }
    }
}

/// Convert score to bucket (for hysteresis).
/// Returns: 0 (hot), 1 (neutral), 2 (cold)
fn score_to_bucket(score: f32) -> u8 {
    if score >= 0.8 {
        0 // hot
    } else if score >= 0.4 {
        1 // neutral
    } else {
        2 // cold
    }
}

/// Safely truncate a string to max_bytes respecting UTF-8 character boundaries.
fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    match s.get(..max_bytes) {
        Some(t) => t,
        None => {
            let byte_idx = s.char_indices()
                .take_while(|(i, _)| *i < max_bytes)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            &s[..byte_idx]
        }
    }
}

/// Telemetry streamer task.
///
/// Polls CTE telemetry every 500ms and updates offset-based statistics.
async fn telemetry_streamer(
    monitor: Arc<BlobMonitor>,
    mut shutdown: broadcast::Receiver<()>,
    mut error_backoff: ErrorBackoff,
) {
    let mut ticker = interval(Duration::from_millis(500));
    let client = match Client::new().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create client: {}", e);
            return;
        }
    };

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                match client.poll_telemetry(0, 5.0).await {
                    Ok(entries) => {
                        error_backoff.reset();
                        let mut stats = monitor.telemetry_stats.write().await;
                        let mut known_tags = monitor.known_tags.write().await;

                        for entry in entries {
                            // Track known tags
                            if !known_tags.contains(&entry.tag_id) {
                                known_tags.push(entry.tag_id);
                            }

                            // Update telemetry statistics (offset-based)
                            let key = TelemetryKey {
                                tag_id: entry.tag_id,
                                offset: entry.off,
                            };

                            let stats_entry = stats.entry(key).or_default();
                            stats_entry.first_seen = if stats_entry.first_seen == 0 {
                                entry.logical_time
                            } else {
                                stats_entry.first_seen
                            };
                            stats_entry.last_seen = entry.logical_time;
                            stats_entry.access_count += 1;

                            match entry.op {
                                CteOp::GetBlob => {
                                    stats_entry.bytes_read += entry.size;
                                }
                                CteOp::PutBlob => {
                                    stats_entry.bytes_written += entry.size;
                                }
                                _ => {}
                            }
                        }
                    }
                    Err(e) => {
                        if error_backoff.should_log() {
                            eprintln!("Telemetry poll error: {}", e);
                        }
                    }
                }
            }
            _ = shutdown.recv() => {
                println!("Telemetry streamer shutting down...");
                break;
            }
        }
    }
}

/// Blob registry builder task.
///
/// Polls tag.get_contained_blobs() every 5s to build name-based registry.
/// This registry is used for actual reorganize_blob() calls.
async fn blob_registry_builder(
    monitor: Arc<BlobMonitor>,
    mut shutdown: broadcast::Receiver<()>,
    mut error_backoff: ErrorBackoff,
) {
    let mut ticker = interval(Duration::from_secs(5));

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Clone known_tags while holding read lock
                let known_tags: Vec<CteTagId> = {
                    let tags = monitor.known_tags.read().await;
                    tags.clone()
                };

                // Acquire blob_registry write lock independently
                let mut blob_registry = monitor.blob_registry.write().await;

                for tag_id in known_tags {
                    match Tag::from_id(tag_id).await {
                        Ok(tag) => {
                            match tag.get_contained_blobs().await {
                                Ok(blobs) => {
                                    for blob_name in blobs {
                                        // Filter empty blob names
                                        if blob_name.is_empty() {
                                            eprintln!("Warning: Encountered empty blob name for tag {:?}", tag_id);
                                            continue;
                                        }

                                        let key = BlobKey {
                                            tag_id,
                                            blob_name,
                                        };
                                        // Initialize if not exists
                                        blob_registry.entry(key).or_default();
                                    }
                                }
                                Err(e) => {
                                    if error_backoff.should_log() {
                                        eprintln!("Failed to get blobs for tag {:?}: {}", tag_id, e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            if error_backoff.should_log() {
                                eprintln!("Failed to open tag {:?}: {}", tag_id, e);
                            }
                        }
                    }
                }
            }
            _ = shutdown.recv() => {
                println!("Blob registry builder shutting down...");
                break;
            }
        }
    }
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Print monitoring table with both telemetry and registry data.
fn print_table(
    telemetry_stats: &HashMap<TelemetryKey, TelemetryStats>,
    blob_registry: &HashMap<BlobKey, BlobInfo>,
    current_time: u64,
) {
    println!("\n{:=<100}", "");
    println!(
        "{:<30} | {:>8} | {:>10} | {:>10} | {:>8} | {:>6} | {:>8}",
        "Blob Name", "Accesses", "Bytes Read", "Bytes Writ", "Frecency", "Score", "State"
    );
    println!("{:=<100}", "");

    // Show blobs from registry (name-based) - these are what we actually manage
    let mut registry_entries: Vec<_> = blob_registry.iter().collect();
    registry_entries.sort_by(|a, b| b.1.access_count.cmp(&a.1.access_count));

    for (blob_key, info) in registry_entries {
        let frecency = BlobMonitor::calculate_frecency(info.access_count, info.last_checked, current_time);
        let state = if frecency > 10.0 {
            "HOT"
        } else if frecency < 2.0 {
            "COLD"
        } else {
            "NEUTRAL"
        };

        // Estimate bytes from telemetry (aggregate for this tag)
        let (total_read, total_written) = telemetry_stats
            .iter()
            .filter(|(k, _)| k.tag_id == blob_key.tag_id)
            .fold((0u64, 0u64), |acc, (_, s)| {
                (acc.0 + s.bytes_read, acc.1 + s.bytes_written)
            });

        let display_name = truncate_str(&blob_key.blob_name, 28);
        println!(
            "{:<30} | {:>8} | {:>10} | {:>10} | {:>8.2} | {:>6.2} | {:>8}",
            display_name,
            info.access_count,
            format_bytes(total_read),
            format_bytes(total_written),
            frecency,
            info.score,
            state
        );
    }

    println!("{:=<100}", "");
    println!("Named blobs in registry: {} | Telemetry offsets: {}", blob_registry.len(), telemetry_stats.len());
}

/// Aggregate telemetry stats per tag for registry updates.
/// Returns (total_access_count, last_seen_time).
fn aggregate_tag_telemetry(
    telemetry_stats: &HashMap<TelemetryKey, TelemetryStats>,
    tag_id: CteTagId,
    current_time: u64,
) -> (u64, u64) {
    let mut result = telemetry_stats
        .iter()
        .filter(|(k, _)| k.tag_id == tag_id)
        .fold((0u64, 0u64), |(count, last_seen), (_, s)| {
            // Use max() for last_seen to get most recent access
            (count + s.access_count, last_seen.max(s.last_seen))
        });

    // Fallback to current time if no telemetry
    if result.1 == 0 {
        result.1 = current_time;
    }

    result
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let refresh_ms: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    println!("Blob Monitor - Starting...");
    println!("Refresh interval: {} ms", refresh_ms);
    println!("Press Ctrl+C to shut down gracefully.\n");

    // Initialize CTE (keep client for reorganize_blob calls)
    let client = Arc::new(Client::new().await?);

    // Create monitor
    let monitor = Arc::new(BlobMonitor::new());

    // Set up shutdown manager (single broadcast channel)
    let shutdown_manager = Arc::new(ShutdownManager::new());

    // Spawn tasks with broadcast shutdown receivers
    let monitor1 = monitor.clone();
    let shutdown1 = shutdown_manager.subscriber();
    let error_backoff1 = ErrorBackoff::new();
    let telemetry_handle = tokio::spawn(async move {
        telemetry_streamer(monitor1, shutdown1, error_backoff1).await;
    });

    let monitor2 = monitor.clone();
    let shutdown2 = shutdown_manager.subscriber();
    let error_backoff2 = ErrorBackoff::new();
    let registry_handle = tokio::spawn(async move {
        blob_registry_builder(monitor2, shutdown2, error_backoff2).await;
    });

    // Main loop - calculate frecency and apply score updates
    let mut ticker = interval(Duration::from_millis(refresh_ms));
    let mut current_logical_time: u64 = 0;
    let mut shutdown_rx = shutdown_manager.subscriber();

    // Main monitoring loop
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Clone known_tags FIRST (outside lock scope) - kept for future use
                #[allow(unused_variables)]
                let known_tags: Vec<CteTagId> = {
                    let tags = monitor.known_tags.read().await;
                    tags.clone()
                };

                // Use with_read_locks helper to ensure correct lock ordering
                // Collect update proposals while holding read locks
                let updates: Vec<(BlobKey, f32, u64)> = monitor.with_read_locks(|telemetry_stats, blob_registry| {
                    // Update current logical time from telemetry, fallback to system time
                    current_logical_time = telemetry_stats.values()
                        .map(|s| s.last_seen)
                        .max()
                        .unwrap_or_else(|| {
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs()
                        });

                    // Print the table
                    print_table(telemetry_stats, blob_registry, current_logical_time);

                    // Calculate new scores for blobs in registry
                    let mut updates: Vec<(BlobKey, f32, u64)> = Vec::new();
                    for (blob_key, info) in blob_registry.iter() {
                        // Aggregate telemetry for this blob's tag
                        let (total_access, last_seen) = 
                            aggregate_tag_telemetry(telemetry_stats, blob_key.tag_id, current_logical_time);

                        let frecency = BlobMonitor::calculate_frecency(total_access, last_seen, current_logical_time);
                        let new_score = BlobMonitor::frecency_to_score(frecency);

                        // Hysteresis: Only update when crossing bucket boundaries
                        if score_to_bucket(info.score) != score_to_bucket(new_score) {
                            updates.push((blob_key.clone(), new_score, total_access));
                        }
                    }
                    updates
                }).await; // All read locks released here

                // Apply score updates AFTER releasing ALL locks
                if !updates.is_empty() {
                    // First, perform all reorganize_blob calls
                    let mut successful_updates: Vec<(BlobKey, f32, u64)> = Vec::new();
                    
                    for (blob_key, new_score, total_access) in updates {
                        // ACTUAL CTE UPDATE: call reorganize_blob()
                        match Tag::from_id(blob_key.tag_id).await {
                            Ok(tag) => {
                                match tag.reorganize_blob(blob_key.blob_name.clone(), new_score).await {
                                    Ok(_) => {
                                        println!("  Updated blob '{}' score to {:.2}", 
                                                 truncate_str(&blob_key.blob_name, 28), new_score);
                                        successful_updates.push((blob_key.clone(), new_score, total_access));
                                    }
                                    Err(e) => {
                                        eprintln!("  Failed to reorganize blob '{}': {}", 
                                                  truncate_str(&blob_key.blob_name, 28), e);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("  Failed to open tag {:?}: {}", blob_key.tag_id, e);
                            }
                        }
                    }

                    // Only update registry for successful reorganize_blob calls
                    if !successful_updates.is_empty() {
                        let mut blob_registry = monitor.blob_registry.write().await;
                        for (blob_key, new_score, total_access) in successful_updates {
                            if let Some(info) = blob_registry.get_mut(&blob_key) {
                                info.score = new_score;
                                info.access_count = total_access;
                                info.last_checked = current_logical_time;
                            } else {
                                eprintln!("Warning: Blob '{:?}' not in registry after successful reorganize", blob_key);
                            }
                        }
                    }
                }
            }
            _ = signal::ctrl_c() => {
                println!("\nShutdown signal received...");
                shutdown_manager.shutdown();
                break;
            }
        }
    }

    // Wait for tasks with timeout
    println!("Waiting for tasks to shut down (5s timeout)...");
    let shutdown_timeout = Duration::from_secs(5);

    match timeout(shutdown_timeout, telemetry_handle).await {
        Ok(_) => println!("Telemetry streamer stopped."),
        Err(_) => println!("Telemetry streamer timeout, terminated."),
    }

    match timeout(shutdown_timeout, registry_handle).await {
        Ok(_) => println!("Blob registry builder stopped."),
        Err(_) => println!("Blob registry builder timeout, terminated."),
    }

    println!("Blob monitor stopped.");
    Ok(())
}