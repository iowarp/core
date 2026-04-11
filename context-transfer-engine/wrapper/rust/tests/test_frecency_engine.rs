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

//! Standalone tests for frecency_engine module
//!
//! These tests can be run independently of the FFI/C++ layer.

use wrp_cte::{FrecencyEngine, HotSet, DECAY_FACTOR, HOT_SET_SIZE};

#[test]
fn test_hot_set_creation() {
    let hot = HotSet::new();
    assert_eq!(hot.active_count(), 0);
    assert_eq!(hot.free_slots.len(), HOT_SET_SIZE);
}

#[test]
fn test_hot_set_insert_and_find() {
    let mut hot = HotSet::new();

    let blob_id = 12345u64;
    let slot = hot.insert(blob_id).expect("Insert should succeed");

    assert!(slot < HOT_SET_SIZE);
    assert_eq!(hot.find(blob_id), Some(slot));
    assert_eq!(hot.active_count(), 1);
}

#[test]
fn test_hot_set_record_access() {
    let mut hot = HotSet::new();

    let blob_id = 12345u64;
    let slot = hot.insert(blob_id).unwrap();

    let score = hot.record_access(slot);
    assert!((score - 1.0).abs() < 0.01, "Initial score should be ~1.0");
    assert_eq!(hot.get_count(slot), 1);
}

#[test]
fn test_hot_set_remove() {
    let mut hot = HotSet::new();

    let blob_id = 12345u64;
    let slot = hot.insert(blob_id).unwrap();

    hot.remove(slot);

    assert!(hot.find(blob_id).is_none());
    assert_eq!(hot.active_count(), 0);
}

#[test]
fn test_hot_set_batch_decay_scalar() {
    let mut hot = HotSet::new();

    // Insert 3 entries
    let id1 = 100u64;
    let id2 = 200u64;
    let id3 = 300u64;

    hot.insert(id1);
    hot.insert(id2);
    hot.insert(id3);

    // Record accesses
    if let Some(slot) = hot.find(id1) {
        hot.record_access(slot);
    }
    if let Some(slot) = hot.find(id2) {
        hot.record_access(slot);
    }
    if let Some(slot) = hot.find(id3) {
        hot.record_access(slot);
    }

    // Get initial scores
    let score1_before = hot.get_score(hot.find(id1).unwrap());
    let score2_before = hot.get_score(hot.find(id2).unwrap());

    // Batch decay (force scalar for this test)
    hot.batch_decay_scalar();

    // Verify decay
    let score1_after = hot.get_score(hot.find(id1).unwrap());
    let score2_after = hot.get_score(hot.find(id2).unwrap());

    assert!((score1_after - score1_before * DECAY_FACTOR).abs() < 0.0001);
    assert!((score2_after - score2_before * DECAY_FACTOR).abs() < 0.0001);
}

#[test]
fn test_frecency_engine_creation() {
    let engine = FrecencyEngine::new();
    assert!(engine.is_empty());
    assert_eq!(engine.len(), 0);
}

#[test]
fn test_frecency_engine_record_access() {
    let mut engine = FrecencyEngine::new();

    let blob_id = 12345u64;
    let score = engine.record_access(blob_id);

    assert!((score - 1.0).abs() < 0.01);
    assert_eq!(engine.len(), 1);
    assert_eq!(engine.get_count(blob_id), Some(1));
}

#[test]
fn test_frecency_engine_multiple_accesses() {
    let mut engine = FrecencyEngine::new();

    let blob_id = 12345u64;

    // Multiple accesses should increase score and count
    engine.record_access(blob_id);
    engine.record_access(blob_id);
    let score = engine.record_access(blob_id);

    assert!(score > 2.0, "Score should be > 2.0 after 3 accesses");
    assert_eq!(engine.get_count(blob_id), Some(3));
}

#[test]
fn test_frecency_engine_batch_decay() {
    let mut engine = FrecencyEngine::new();

    // Insert entries
    let id1 = 100u64;
    let id2 = 200u64;

    engine.record_access(id1);
    engine.record_access(id2);

    let score_before = engine.get_score(id1).unwrap();

    // Batch decay
    let decayed = engine.batch_decay();

    let score_after = engine.get_score(id1).unwrap();

    assert!((score_after - score_before * DECAY_FACTOR).abs() < 0.0001);
    assert_eq!(decayed.len(), 2);
}

#[test]
fn test_frecency_engine_get_hot_candidates() {
    let mut engine = FrecencyEngine::new();

    // Create entries with different scores
    let id1 = 100u64;
    let id2 = 200u64;
    let id3 = 300u64;

    // Access id1 many times
    for _ in 0..10 {
        engine.record_access(id1);
    }

    // Access id2 moderately
    for _ in 0..5 {
        engine.record_access(id2);
    }

    // Access id3 rarely
    engine.record_access(id3);

    // Get candidates with threshold
    let threshold = 3.0;
    let candidates = engine.get_hot_candidates(threshold);

    assert!(candidates.contains(&id1), "id1 should be hot (score > 3)");
    assert!(candidates.contains(&id2), "id2 should be hot (score > 3)");
    assert!(
        !candidates.contains(&id3),
        "id3 should not be hot (score < 3)"
    );
}

#[test]
fn test_frecency_engine_remove() {
    let mut engine = FrecencyEngine::new();

    let blob_id = 12345u64;
    engine.record_access(blob_id);

    engine.remove(blob_id);

    assert!(engine.get_score(blob_id).is_none());
    assert_eq!(engine.len(), 0);
}

#[test]
fn test_frecency_engine_stats() {
    let mut engine = FrecencyEngine::new();

    // Add some entries
    for i in 0..5 {
        engine.record_access(i as u64);
    }

    let hot_stats = engine.hot_stats();
    assert_eq!(hot_stats.active_entries, 5);
    assert!(hot_stats.total_score > 0.0);

    let cold_stats = engine.cold_stats();
    assert_eq!(cold_stats.entry_count, 0);
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn test_simd_decay() {
    // This test only runs on x86/x86_64 with AVX2 support
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping SIMD test: AVX2 not available");
        return;
    }

    let mut hot = HotSet::new();

    // Fill all slots
    for i in 0..HOT_SET_SIZE {
        hot.insert(i as u64);
    }

    // Record some accesses
    for i in 0..HOT_SET_SIZE {
        let slot = hot.find(i as u64).unwrap();
        hot.record_access(slot);
    }

    // Get initial scores (all should be ~1.0)
    let scores_before: Vec<f64> = (0..HOT_SET_SIZE)
        .filter_map(|i| hot.find(i as u64).map(|s| hot.get_score(s)))
        .collect();

    // Batch decay with SIMD
    hot.increment_tick();
    unsafe {
        hot.batch_decay_simd();
    }

    // Verify decay applied correctly
    for i in 0..HOT_SET_SIZE {
        if let Some(slot) = hot.find(i as u64) {
            let score_after = hot.get_score(slot);
            let expected = scores_before[i] * DECAY_FACTOR;
            assert!(
                (score_after - expected).abs() < 0.0001,
                "Slot {} score mismatch: {} vs {}",
                i,
                score_after,
                expected
            );
        }
    }
}

#[test]
fn test_cold_set_promotion() {
    let mut engine = FrecencyEngine::new();

    // Fill hot set to capacity
    for i in 0..(HOT_SET_SIZE as u64) {
        engine.record_access(i);
    }

    // Add new blob (goes to cold set)
    let cold_blob = (HOT_SET_SIZE + 100) as u64;
    engine.record_access(cold_blob);

    // Verify it exists (could be in hot or cold set)
    assert!(engine.get_score(cold_blob).is_some());

    // Should be promoted to hot set (if space available)
    // Or stay in cold set if hot set is full
    let score = engine.get_score(cold_blob);
    assert!(
        score.is_some(),
        "Blob should be tracked in either hot or cold set"
    );
}

#[test]
fn test_alignment_requirements() {
    use std::mem;

    // Verify HotSet is cache line aligned
    assert!(
        mem::align_of::<HotSet>() >= 64,
        "HotSet should be cache line aligned"
    );
}

#[test]
fn test_many_sequential_accesses() {
    let mut engine = FrecencyEngine::new();

    // Test with many sequential blob IDs
    for i in 0..100 {
        let blob_id = i as u64;
        engine.record_access(blob_id);
    }

    // Verify all are tracked
    assert_eq!(engine.len(), 100);

    // Verify counts
    for i in 0..100u64 {
        assert_eq!(engine.get_count(i), Some(1));
    }
}

#[test]
fn test_decay_over_time() {
    let mut engine = FrecencyEngine::new();

    let blob_id = 100u64;

    // Record initial access
    let score1 = engine.record_access(blob_id);
    assert!((score1 - 1.0).abs() < 0.01);

    // Decay once
    engine.batch_decay();
    let score2 = engine.get_score(blob_id).unwrap();
    assert!((score2 - score1 * DECAY_FACTOR).abs() < 0.0001);

    // Decay again
    engine.batch_decay();
    let score3 = engine.get_score(blob_id).unwrap();
    assert!((score3 - score2 * DECAY_FACTOR).abs() < 0.0001);
}

#[test]
fn test_tick_tracking() {
    let mut engine = FrecencyEngine::new();

    assert_eq!(engine.current_tick(), 0);

    engine.batch_decay();
    assert_eq!(engine.current_tick(), 1);

    engine.batch_decay();
    assert_eq!(engine.current_tick(), 2);
}
