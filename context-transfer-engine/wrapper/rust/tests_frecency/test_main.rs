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

//! Standalone test binary for frecency_engine module
//! This test does not depend on FFI/C++ layer and can be compiled independently.

mod frecency_engine {
    use std::collections::HashMap;
    use std::hash::BuildHasherDefault;

    type FastHashMap<K, V> =
        HashMap<K, V, BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;

    pub const HOT_SET_SIZE: usize = 512;
    pub const DECAY_FACTOR: f64 = 0.999_999;
    pub const DEFAULT_SCORE: f64 = 0.0;

    #[derive(Debug, Clone)]
    struct ColdEntry {
        score: f64,
        count: u64,
        last_update: u64,
    }

    impl ColdEntry {
        fn new() -> Self {
            ColdEntry {
                score: DEFAULT_SCORE,
                count: 0,
                last_update: 0,
            }
        }
    }

    #[repr(align(64))]
    pub struct HotSet {
        scores: Vec<f64>,
        counts: Vec<u64>,
        last_updates: Vec<u64>,
        keys: Vec<u64>,
        key_to_slot: FastHashMap<u64, usize>,
        pub free_slots: Vec<usize>,
        current_tick: u64,
    }

    impl HotSet {
        pub fn new() -> Self {
            let mut free_slots: Vec<usize> = (0..HOT_SET_SIZE).collect();
            free_slots.reverse();
            HotSet {
                scores: vec![0.0; HOT_SET_SIZE],
                counts: vec![0; HOT_SET_SIZE],
                last_updates: vec![0; HOT_SET_SIZE],
                keys: vec![0; HOT_SET_SIZE],
                key_to_slot: FastHashMap::with_capacity_and_hasher(
                    HOT_SET_SIZE,
                    BuildHasherDefault::default(),
                ),
                free_slots,
                current_tick: 0,
            }
        }

        #[inline]
        fn has_avx2() -> bool {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                is_x86_feature_detected!("avx2")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }

        #[inline]
        pub fn find(&self, blob_id: u64) -> Option<usize> {
            self.key_to_slot.get(&blob_id).copied()
        }

        #[inline]
        pub fn record_access(&mut self, slot: usize) -> f64 {
            debug_assert!(slot < HOT_SET_SIZE);
            let missed_ticks = self.current_tick.saturating_sub(self.last_updates[slot]);
            if missed_ticks > 0 {
                self.scores[slot] *= DECAY_FACTOR.powi(missed_ticks as i32);
            }
            self.scores[slot] += 1.0;
            self.counts[slot] += 1;
            self.last_updates[slot] = self.current_tick;
            self.scores[slot]
        }

        pub fn insert(&mut self, blob_id: u64) -> Option<usize> {
            if let Some(slot) = self.key_to_slot.get(&blob_id) {
                return Some(*slot);
            }
            let slot = self.free_slots.pop()?;
            self.scores[slot] = DEFAULT_SCORE + 1.0;
            self.counts[slot] = 1;
            self.last_updates[slot] = self.current_tick;
            self.keys[slot] = blob_id;
            self.key_to_slot.insert(blob_id, slot);
            Some(slot)
        }

        #[inline]
        pub fn remove(&mut self, slot: usize) {
            debug_assert!(slot < HOT_SET_SIZE);
            let blob_id = self.keys[slot];
            self.key_to_slot.remove(&blob_id);
            self.scores[slot] = 0.0;
            self.counts[slot] = 0;
            self.last_updates[slot] = 0;
            self.keys[slot] = 0;
            self.free_slots.push(slot);
        }

        #[inline]
        pub fn get_score(&self, slot: usize) -> f64 {
            self.scores[slot]
        }
        #[inline]
        pub fn get_count(&self, slot: usize) -> u64 {
            self.counts[slot]
        }
        #[inline]
        pub fn get_key(&self, slot: usize) -> u64 {
            self.keys[slot]
        }
        #[inline]
        pub fn current_tick(&self) -> u64 {
            self.current_tick
        }
        #[inline]
        pub fn increment_tick(&mut self) {
            self.current_tick += 1;
        }
        #[inline]
        pub fn active_count(&self) -> usize {
            HOT_SET_SIZE - self.free_slots.len()
        }

        pub fn batch_decay(&mut self) -> Vec<(u64, f64)> {
            let mut decayed = Vec::with_capacity(self.active_count());
            self.increment_tick();
            if Self::has_avx2() {
                unsafe {
                    self.batch_decay_simd();
                }
            } else {
                self.batch_decay_scalar();
            }
            for (&blob_id, &slot) in self.key_to_slot.iter() {
                decayed.push((blob_id, self.scores[slot]));
            }
            decayed
        }

        pub fn batch_decay_scalar(&mut self) {
            for i in 0..HOT_SET_SIZE {
                if self.keys[i] != 0 {
                    self.scores[i] *= DECAY_FACTOR;
                }
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn batch_decay_simd(&mut self) {
            use std::arch::x86_64::*;
            let decay_vec = _mm256_set1_pd(DECAY_FACTOR);
            let chunks = HOT_SET_SIZE / 4;
            for chunk in 0..chunks {
                let offset = chunk * 4;
                let scores_ptr = self.scores.as_ptr().add(offset) as *const f64;
                let scores_vec = _mm256_load_pd(scores_ptr);
                let decayed = _mm256_mul_pd(scores_vec, decay_vec);
                let dest_ptr = self.scores.as_mut_ptr().add(offset) as *mut f64;
                _mm256_store_pd(dest_ptr, decayed);
            }
            let remainder_start = chunks * 4;
            for i in remainder_start..HOT_SET_SIZE {
                self.scores[i] *= DECAY_FACTOR;
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        pub fn batch_decay_simd(&mut self) {
            self.batch_decay_scalar();
        }
    }

    impl Default for HotSet {
        fn default() -> Self {
            Self::new()
        }
    }

    pub struct FrecencyEngine {
        hot: HotSet,
        cold: FastHashMap<u64, ColdEntry>,
        tick: u64,
    }

    impl FrecencyEngine {
        pub fn new() -> Self {
            FrecencyEngine {
                hot: HotSet::new(),
                cold: FastHashMap::with_hasher(BuildHasherDefault::default()),
                tick: 0,
            }
        }

        pub fn record_access(&mut self, blob_id: u64) -> f64 {
            if let Some(slot) = self.hot.find(blob_id) {
                return self.hot.record_access(slot);
            }

            if !self.cold.contains_key(&blob_id) {
                if self.hot.insert(blob_id).is_some() {
                    return self.hot.get_score(self.hot.find(blob_id).unwrap());
                }
                self.cold.insert(blob_id, ColdEntry::new());
                return 0.0;
            }

            let (score, count, should_promote) = {
                let entry = self.cold.get(&blob_id).unwrap();
                let mut score = entry.score;
                let missed_ticks = self.tick.saturating_sub(entry.last_update);
                if missed_ticks > 0 {
                    score *= DECAY_FACTOR.powi(missed_ticks as i32);
                }
                score += 1.0;
                (score, entry.count + 1, entry.count + 1 > 3)
            };

            if should_promote {
                self.cold.remove(&blob_id);
                if self.hot.insert(blob_id).is_some() {
                    if let Some(slot) = self.hot.find(blob_id) {
                        self.hot.scores[slot] = score;
                        self.hot.counts[slot] = count;
                        self.hot.last_updates[slot] = self.tick;
                    }
                    return score;
                }
                self.cold.insert(
                    blob_id,
                    ColdEntry {
                        score,
                        count,
                        last_update: self.tick,
                    },
                );
                return score;
            }

            let entry = self.cold.get_mut(&blob_id).unwrap();
            entry.score = score;
            entry.count = count;
            entry.last_update = self.tick;
            score
        }

        pub fn batch_decay(&mut self) -> Vec<(u64, f64)> {
            self.hot.increment_tick();
            self.tick += 1;
            if HotSet::has_avx2() {
                unsafe {
                    self.hot.batch_decay_simd();
                }
            } else {
                self.hot.batch_decay_scalar();
            }
            for entry in self.cold.values_mut() {
                entry.score *= DECAY_FACTOR;
            }
            let mut decayed = Vec::with_capacity(self.hot.active_count());
            for (&blob_id, &slot) in self.hot.key_to_slot.iter() {
                decayed.push((blob_id, self.hot.scores[slot]));
            }
            decayed
        }

        pub fn get_hot_candidates(&self, threshold: f64) -> Vec<u64> {
            self.hot
                .key_to_slot
                .iter()
                .filter(|(_, &slot)| self.hot.scores[slot] >= threshold)
                .map(|(&blob_id, _)| blob_id)
                .collect()
        }

        pub fn get_score(&self, blob_id: u64) -> Option<f64> {
            self.hot
                .find(blob_id)
                .map(|slot| self.hot.get_score(slot))
                .or_else(|| self.cold.get(&blob_id).map(|e| e.score))
        }

        pub fn get_count(&self, blob_id: u64) -> Option<u64> {
            self.hot
                .find(blob_id)
                .map(|slot| self.hot.get_count(slot))
                .or_else(|| self.cold.get(&blob_id).map(|e| e.count))
        }

        pub fn remove(&mut self, blob_id: u64) {
            if let Some(slot) = self.hot.find(blob_id) {
                self.hot.remove(slot);
            } else {
                self.cold.remove(&blob_id);
            }
        }

        pub fn len(&self) -> usize {
            self.hot.active_count() + self.cold.len()
        }
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
        pub fn current_tick(&self) -> u64 {
            self.tick
        }
    }

    impl Default for FrecencyEngine {
        fn default() -> Self {
            Self::new()
        }
    }
}

fn main() {
    use frecency_engine::*;

    println!("=== Frecency Engine Tests ===\n");

    // Test 1: Hot set creation
    print!("Test 1: Hot set creation... ");
    let hot = HotSet::new();
    assert_eq!(hot.active_count(), 0);
    assert_eq!(hot.free_slots.len(), HOT_SET_SIZE);
    println!("PASS");

    // Test 2: Hot set insert and find
    print!("Test 2: Hot set insert and find... ");
    let mut hot = HotSet::new();
    let blob_id = 12345u64;
    let slot = hot.insert(blob_id).expect("Insert should succeed");
    assert!(slot < HOT_SET_SIZE);
    assert_eq!(hot.find(blob_id), Some(slot));
    assert_eq!(hot.active_count(), 1);
    println!("PASS");

    // Test 3: Hot set record access
    print!("Test 3: Hot set record access... ");
    let mut hot = HotSet::new();
    let slot = hot.insert(12345u64).unwrap();
    let score_after_insert = hot.get_score(slot);
    assert!(
        (score_after_insert - 1.0).abs() < 0.01,
        "Score after insert should be ~1.0"
    );
    let score_after_access = hot.record_access(slot);
    assert!(
        (score_after_access - 2.0).abs() < 0.01,
        "Score after access should be ~2.0"
    );
    assert_eq!(hot.get_count(slot), 2);
    println!("PASS");

    // Test 4: Hot set remove
    print!("Test 4: Hot set remove... ");
    let mut hot = HotSet::new();
    let slot = hot.insert(12345u64).unwrap();
    hot.remove(slot);
    assert!(hot.find(12345u64).is_none());
    assert_eq!(hot.active_count(), 0);
    println!("PASS");

    // Test 5: Batch decay scalar
    print!("Test 5: Batch decay scalar... ");
    let mut hot = HotSet::new();
    hot.insert(100u64);
    hot.insert(200u64);
    hot.insert(300u64);
    if let Some(s) = hot.find(100) {
        hot.record_access(s);
    }
    if let Some(s) = hot.find(200) {
        hot.record_access(s);
    }
    let score_before = hot.get_score(hot.find(100).unwrap());
    hot.batch_decay_scalar();
    let score_after = hot.get_score(hot.find(100).unwrap());
    assert!((score_after - score_before * DECAY_FACTOR).abs() < 0.0001);
    println!("PASS");

    // Test 6: Frecency engine creation
    print!("Test 6: Frecency engine creation... ");
    let engine = FrecencyEngine::new();
    assert!(engine.is_empty());
    assert_eq!(engine.len(), 0);
    println!("PASS");

    // Test 7: Frecency engine record access
    print!("Test 7: Frecency engine record access... ");
    let mut engine = FrecencyEngine::new();
    let score = engine.record_access(12345u64);
    assert!((score - 1.0).abs() < 0.01);
    assert_eq!(engine.len(), 1);
    assert_eq!(engine.get_count(12345u64), Some(1));
    println!("PASS");

    // Test 8: Multiple accesses
    print!("Test 8: Multiple accesses... ");
    let mut engine = FrecencyEngine::new();
    engine.record_access(12345u64);
    engine.record_access(12345u64);
    let score = engine.record_access(12345u64);
    assert!(score > 2.0, "Score should be > 2.0 after 3 accesses");
    assert_eq!(engine.get_count(12345u64), Some(3));
    println!("PASS");

    // Test 9: Batch decay engine
    print!("Test 9: Batch decay engine... ");
    let mut engine = FrecencyEngine::new();
    engine.record_access(100u64);
    engine.record_access(200u64);
    let score_before = engine.get_score(100u64).unwrap();
    let decayed = engine.batch_decay();
    let score_after = engine.get_score(100u64).unwrap();
    assert!((score_after - score_before * DECAY_FACTOR).abs() < 0.0001);
    assert_eq!(decayed.len(), 2);
    println!("PASS");

    // Test 10: Get hot candidates
    print!("Test 10: Get hot candidates... ");
    let mut engine = FrecencyEngine::new();
    for _ in 0..10 {
        engine.record_access(100u64);
    }
    for _ in 0..5 {
        engine.record_access(200u64);
    }
    engine.record_access(300u64);
    let candidates = engine.get_hot_candidates(3.0);
    assert!(candidates.contains(&100u64));
    assert!(candidates.contains(&200u64));
    assert!(!candidates.contains(&300u64));
    println!("PASS");

    // Test 11: SIMD decay (if AVX2 available)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            print!("Test 11: SIMD decay... ");
            let mut hot = HotSet::new();
            for i in 0..HOT_SET_SIZE {
                hot.insert(i as u64);
            }
            for i in 0..HOT_SET_SIZE {
                if let Some(slot) = hot.find(i as u64) {
                    hot.record_access(slot);
                }
            }
            let scores_before: Vec<f64> = (0..HOT_SET_SIZE)
                .filter_map(|i| hot.find(i as u64).map(|s| hot.get_score(s)))
                .collect();
            hot.increment_tick();
            unsafe {
                hot.batch_decay_simd();
            }
            for i in 0..HOT_SET_SIZE {
                if let Some(slot) = hot.find(i as u64) {
                    let score_after = hot.get_score(slot);
                    let expected = scores_before[i] * DECAY_FACTOR;
                    assert!((score_after - expected).abs() < 0.0001);
                }
            }
            println!("PASS");
        } else {
            println!("Test 11: SIMD decay... SKIPPED (AVX2 not available)");
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        println!("Test 11: SIMD decay... SKIPPED (not x86/x86_64)");
    }

    // Test 12: Alignment
    print!("Test 12: Alignment requirements... ");
    assert!(std::mem::align_of::<HotSet>() >= 64);
    println!("PASS");

    // Test 13: Many sequential accesses
    print!("Test 13: Many sequential accesses... ");
    let mut engine = FrecencyEngine::new();
    for i in 0..100u64 {
        engine.record_access(i);
    }
    assert_eq!(engine.len(), 100);
    for i in 0..100u64 {
        assert_eq!(engine.get_count(i), Some(1));
    }
    println!("PASS");

    // Test 14: Decay over time (simplified)
    print!("Test 14: Decay over time... ");
    let mut hot = HotSet::new();
    let slot = hot.insert(100u64).unwrap();
    hot.record_access(slot);
    let score_before = hot.get_score(slot);
    hot.batch_decay_scalar();
    let score_after = hot.get_score(slot);
    assert!((score_after - score_before * DECAY_FACTOR).abs() < 0.0001);
    println!("PASS");

    // Test 15: Tick tracking
    print!("Test 15: Tick tracking... ");
    let mut engine = FrecencyEngine::new();
    assert_eq!(engine.current_tick(), 0);
    engine.batch_decay();
    assert_eq!(engine.current_tick(), 1);
    engine.batch_decay();
    assert_eq!(engine.current_tick(), 2);
    println!("PASS");

    println!("\n=== All tests passed! ===");
}
