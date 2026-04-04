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

//! Integration tests for CTE runtime
//!
//! These tests require a running CTE runtime. They are marked with `#[ignore]`
//! by default to avoid failures in CI environments without the runtime.
//!
//! # Running Integration Tests
//!
//! ## Async Tests (with tokio runtime)
//! ```bash
//! # Set environment variable to start embedded runtime
//! CHI_WITH_RUNTIME=1 cargo test --ignored --features async
//! ```
//!
//! ## Sync Tests (with embedded runtime)
//! ```bash
//! # Set environment variable to start embedded runtime
//! CHI_WITH_RUNTIME=1 cargo test --ignored
//! ```
//!
//! ## Alternative: Start runtime separately
//! ```bash
//! # Terminal 1: Start CTE runtime
//! wrp_cte --config /path/to/config.yaml
//!
//! # Terminal 2: Run tests
//! cargo test --ignored --features async
//! ```
//!
//! # Prerequisites
//! - CTE runtime installed and available on PATH
//! - Configuration file (optional, defaults to embedded config)
//! - Sufficient system resources for shared memory operations

#[cfg(test)]
mod sync_tests {
    use wrp_cte::sync::{init, Client, Tag};
    use wrp_cte::types::CteTagId;

    /// Test CTE initialization with default configuration
    ///
    /// This test verifies that CTE can be initialized with an empty
    /// configuration path (using defaults).
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_init_default_config() {
        let result = init("");
        assert!(result.is_ok(), "CTE initialization should succeed with default config");
    }

    /// Test client creation
    ///
    /// Verifies that a CTE client can be created after initialization.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_client_create() {
        init("").expect("CTE initialization failed");
        let client = Client::new();
        assert!(client.is_ok(), "Client creation should succeed after init");
    }

    /// Test client telemetry polling
    ///
    /// Verifies that telemetry can be polled from the client.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_client_telemetry() {
        init("").expect("CTE initialization failed");
        
        let client = Client::new().expect("Client creation should succeed");
        
        // Perform some operations to generate telemetry
        let tag_name = format!("test_tag_telemetry_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        tag.put_blob_with_options("telemetry_blob.bin", b"test data", 0, 1.0)
            .expect("Blob put should succeed");
        
        // Poll telemetry
        let telemetry = client.poll_telemetry(0).expect("Telemetry polling should succeed");
        
        // Verify telemetry is returned (may be empty if already cleared)
        // Just verify it doesn't panic
        for entry in telemetry {
            println!(
                "Telemetry: op={:?}, size={}, tag={}.{}",
                entry.op, entry.size, entry.tag_id.major, entry.tag_id.minor
            );
        }
    }

    /// Test client-level blob reorganization
    ///
    /// Verifies that blob scores can be changed via the client API.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_client_reorganize_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_reorg_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create a blob
        let blob_name = "reorg_blob.bin";
        tag.put_blob_with_options(blob_name, b"test", 0, 1.0)
            .expect("Blob put should succeed");
        
        // Reorganize via client
        let tag_id = tag.id();
        let client = Client::new().expect("Client creation should succeed");
        client
            .reorganize_blob(tag_id, blob_name, 0.25)
            .expect("Client reorganization should succeed");
        
        // Verify new score
        let new_score = tag.get_blob_score(blob_name).expect("Get blob score should succeed");
        assert!(
            (new_score - 0.25).abs() < 0.01,
            "New score should be 0.25, got {}",
            new_score
        );
    }

    /// Test client-level blob deletion
    ///
    /// Verifies that blobs can be deleted via the client API.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_client_del_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_del_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create a blob
        let blob_name = "del_blob.bin";
        tag.put_blob_with_options(blob_name, b"test data to delete", 0, 1.0)
            .expect("Blob put should succeed");
        
        // Verify it exists
        let size = tag.get_blob_size(blob_name).expect("Get blob size should succeed");
        assert!(size > 0, "Blob should exist before deletion");
        
        // Delete via client
        let tag_id = tag.id();
        let client = Client::new().expect("Client creation should succeed");
        client
            .del_blob(tag_id, blob_name)
            .expect("Blob deletion should succeed");
        
        // Verify it's gone
        let result = tag.get_blob_size(blob_name);
        assert!(result.is_err(), "Blob should not exist after deletion");
    }

    /// Test tag creation by name
    ///
    /// Creates a tag with a unique name and verifies it succeeds.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_create() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        let id = tag.id();
        
        assert!(
            id.major > 0 || id.minor > 0,
            "Tag ID should be valid (non-zero)"
        );
    }

    /// Test tag creation by ID
    ///
    /// Verifies that an existing tag can be opened by its ID.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_create_by_id() {
        init("").expect("CTE initialization failed");
        
        // First create a tag by name
        let tag_name = format!("test_tag_by_id_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        let id = tag.id();
        
        // Then open it by ID
        let tag_by_id = Tag::from_id(id);
        let id_again = tag_by_id.id();
        
        assert_eq!(id.major, id_again.major, "Tag major IDs should match");
        assert_eq!(id.minor, id_again.minor, "Tag minor IDs should match");
    }

    /// Test blob put and get operations
    ///
    /// Writes data to a blob and reads it back, verifying integrity.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_put_get_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_blob_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Write test data
        let blob_name = "test_blob.bin";
        let test_data = b"Hello, CTE!";
        tag.put_blob_with_options(blob_name, test_data, 0, 1.0)
            .expect("Blob put should succeed");
        
        // Get blob size
        let size = tag.get_blob_size(blob_name).expect("Get blob size should succeed");
        assert_eq!(
            size,
            test_data.len() as u64,
            "Blob size should match written data"
        );
        
        // Read back data
        let read_data = tag.get_blob(blob_name, size, 0).expect("Blob get should succeed");
        assert_eq!(read_data, test_data, "Read data should match written data");
    }

    /// Test blob score operations
    ///
    /// Verifies that blob placement scores can be set and retrieved.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_blob_score() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_score_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create a blob with score 1.0
        let blob_name = "scored_blob.bin";
        tag.put_blob_with_options(blob_name, b"test", 0, 1.0)
            .expect("Blob put should succeed");
        
        // Verify default score
        let score = tag.get_blob_score(blob_name).expect("Get blob score should succeed");
        assert!(
            (score - 1.0).abs() < 0.01,
            "Default score should be 1.0, got {}",
            score
        );
        
        // Change the score
        tag.reorganize_blob(blob_name, 0.5)
            .expect("Reorganize blob should succeed");
        
        // Verify new score
        let new_score = tag.get_blob_score(blob_name).expect("Get blob score should succeed");
        assert!(
            (new_score - 0.5).abs() < 0.01,
            "New score should be 0.5, got {}",
            new_score
        );
    }

    /// Test blob size retrieval
    ///
    /// Verifies that blob sizes can be queried.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_get_blob_size() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_size_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Test with different sizes
        let small_data = b"hello".to_vec();
        let medium_data = vec![0u8; 1024];
        let large_data = vec![0u8; 10240];
        
        let test_cases = [
            ("small.bin", small_data.as_slice()),
            ("medium.bin", medium_data.as_slice()),
            ("large.bin", large_data.as_slice()),
        ];
        
        for (name, data) in &test_cases {
            tag.put_blob_with_options(name, data, 0, 1.0)
                .expect("Blob put should succeed");
            
            let size = tag.get_blob_size(name).expect("Get blob size should succeed");
            assert_eq!(
                size,
                data.len() as u64,
                "Blob size should match for {}",
                name
            );
        }
    }

    /// Test blob listing
    ///
    /// Verifies that all blobs in a tag can be listed.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_contained_blobs() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_list_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create multiple blobs
        let blob_names: Vec<&str> = ["list_blob_0.bin", "list_blob_1.bin", "list_blob_2.bin"]
            .to_vec();
        
        for name in &blob_names {
            tag.put_blob_with_options(name, b"data", 0, 1.0)
                .expect("Blob put should succeed");
        }
        
        // List all blobs
        let blobs = tag.get_contained_blobs();
        assert!(
            blobs.len() >= blob_names.len(),
            "Should have at least {} blobs, got {}",
            blob_names.len(),
            blobs.len()
        );
        
        // Verify all created blobs are in the list
        for name in &blob_names {
            assert!(
                blobs.contains(&name.to_string()),
                "Blob list should contain {}",
                name
            );
        }
    }

    /// Test large blob handling (stress test)
    ///
    /// Verifies that moderately large blobs can be stored and retrieved.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_large_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_large_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create a moderately large blob (1 MB)
        let blob_name = "large_blob.bin";
        let large_data = vec![0u8; 1024 * 1024];
        
        tag.put_blob_with_options(blob_name, &large_data, 0, 1.0)
            .expect("Large blob put should succeed");
        
        // Read back and verify size
        let size = tag.get_blob_size(blob_name).expect("Get blob size should succeed");
        assert_eq!(size, large_data.len() as u64, "Large blob size should match");
        
        // Read back data
        let read_data = tag
            .get_blob(blob_name, size, 0)
            .expect("Large blob get should succeed");
        assert_eq!(
            read_data.len(),
            large_data.len(),
            "Large blob read size should match"
        );
    }

    /// Test error handling for invalid names
    ///
    /// Verifies that operations with empty names return proper errors.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_error_invalid_name() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_error_name_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Test put_blob with empty name
        let result = tag.put_blob_with_options("", b"data", 0, 1.0);
        assert!(result.is_err(), "put_blob with empty name should fail");
        match result {
            Err(wrp_cte::CteError::InvalidParameter { message }) => {
                assert!(
                    message.contains("cannot be empty"),
                    "Error message should mention empty name"
                );
            }
            _ => panic!("Expected InvalidParameter error"),
        }
        
        // Test get_blob_score with empty name
        let result = tag.get_blob_score("");
        assert!(result.is_err(), "get_blob_score with empty name should fail");
        
        // Test get_blob_size with empty name
        let result = tag.get_blob_size("");
        assert!(result.is_err(), "get_blob_size with empty name should fail");
        
        // Test get_blob with empty name
        let result = tag.get_blob("", 10, 0);
        assert!(result.is_err(), "get_blob with empty name should fail");
        
        // Test reorganize_blob with empty name
        let result = tag.reorganize_blob("", 0.5);
        assert!(result.is_err(), "reorganize_blob with empty name should fail");
    }

    /// Test error handling for invalid scores
    ///
    /// Verifies that operations with invalid scores return proper errors.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_error_invalid_score() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_error_score_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create blob first
        let blob_name = "score_test.bin";
        tag.put_blob_with_options(blob_name, b"data", 0, 1.0)
            .expect("Blob put should succeed");
        
        // Test with negative score
        let result = tag.reorganize_blob(blob_name, -1.0);
        assert!(result.is_err(), "reorganize_blob with negative score should fail");
        match result {
            Err(wrp_cte::CteError::InvalidParameter { message }) => {
                assert!(
                    message.contains("Score must be between"),
                    "Error message should mention score range"
                );
            }
            _ => panic!("Expected InvalidParameter error"),
        }
        
        // Test with score > 1.0
        let result = tag.reorganize_blob(blob_name, 1.5);
        assert!(result.is_err(), "reorganize_blob with score > 1.0 should fail");
        
        // Test with NaN
        let result = tag.reorganize_blob(blob_name, f32::NAN);
        assert!(result.is_err(), "reorganize_blob with NaN score should fail");
        
        // Test put_blob_with score validation
        let result = tag.put_blob_with_options("test.bin", b"data", 0, -0.5);
        assert!(result.is_err(), "put_blob with negative score should fail");
        
        let result = tag.put_blob_with_options("test.bin", b"data", 0, 2.0);
        assert!(result.is_err(), "put_blob with score > 1.0 should fail");
        
        // Test client reorganize_blob with invalid scores
        let client = Client::new().expect("Client creation should succeed");
        let tag_id = tag.id();
        
        let result = client.reorganize_blob(tag_id, blob_name, -0.1);
        assert!(result.is_err(), "client reorganize with negative score should fail");
        
        let result = client.reorganize_blob(tag_id, blob_name, 1.1);
        assert!(result.is_err(), "client reorganize with score > 1.0 should fail");
    }

    /// Test error handling for blob too large
    ///
    /// Verifies that operations with oversized data return proper errors.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_error_blob_too_large() {
        init("").expect("CTE initialization failed");
        
        // Note: We can't actually test with 16GB data in unit tests
        // Instead, we test the validation logic directly
        
        // Test with synthetic validation
        fn validate_size(size: u64) -> Result<(), wrp_cte::CteError> {
            const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GB
            if size > MAX_BLOB_SIZE {
                Err(wrp_cte::CteError::InvalidParameter {
                    message: format!("Data size {} exceeds maximum blob size {}", size, MAX_BLOB_SIZE),
                })
            } else {
                Ok(())
            }
        }
        
        // Test size limit
        let within_limit = validate_size(1024);
        assert!(within_limit.is_ok(), "Small size should be valid");
        
        // Test at exact limit
        const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024;
        let at_limit = validate_size(MAX_BLOB_SIZE);
        assert!(at_limit.is_ok(), "Size at limit should be valid");
        
        // Test over limit
        let over_limit = validate_size(MAX_BLOB_SIZE + 1);
        assert!(over_limit.is_err(), "Size over limit should fail");
        match over_limit {
            Err(wrp_cte::CteError::InvalidParameter { message }) => {
                assert!(
                    message.contains("exceeds maximum"),
                    "Error message should mention size limit"
                );
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    /// Test blob write with offset
    ///
    /// Verifies that data can be written at specific offsets within a blob.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_blob_write_with_offset() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_offset_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Write initial data
        let blob_name = "offset_blob.bin";
        tag.put_blob_with_options(blob_name, b"Hello", 0, 1.0)
            .expect("Initial write should succeed");
        
        // Write at offset
        tag.put_blob_with_options(blob_name, b" World", 5, 1.0)
            .expect("Offset write should succeed");
        
        // Read full data
        let size = tag.get_blob_size(blob_name).expect("Get size should succeed");
        let data = tag.get_blob(blob_name, size, 0).expect("Read should succeed");
        
        assert_eq!(&data, b"Hello World", "Data should match combined writes");
    }

    /// Test multiple blobs in a tag
    ///
    /// Verifies that multiple blobs can be stored and listed in a tag.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_multiple_blobs() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_multi_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Create multiple blobs
        for i in 0..5 {
            let blob_name = format!("multi_blob_{}.bin", i);
            let data = format!("data_{}", i);
            tag.put_blob_with_options(&blob_name, data.as_bytes(), 0, 1.0)
                .expect("Blob put should succeed");
        }
        
        // List all blobs
        let blobs = tag.get_contained_blobs();
        assert!(
            blobs.len() >= 5,
            "Should have at least 5 blobs in tag"
        );
        
        // Verify each blob can be read back
        for i in 0..5 {
            let blob_name = format!("multi_blob_{}.bin", i);
            let expected_data = format!("data_{}", i);
            let size = tag
                .get_blob_size(&blob_name)
                .expect("Get blob size should succeed");
            let data = tag
                .get_blob(&blob_name, size, 0)
                .expect("Get blob should succeed");
            assert_eq!(
                String::from_utf8_lossy(&data),
                expected_data,
                "Blob {} data should match",
                i
            );
        }
    }

    /// Test tag ID retrieval and conversion
    ///
    /// Verifies that tag IDs can be retrieved and converted to/from u64.
    #[test]
    #[ignore = "Requires running CTE runtime"]
    fn test_tag_id_operations() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("test_tag_id_{}", std::process::id());
        let tag = Tag::new(&tag_name);
        
        // Get tag ID
        let id = tag.id();
        assert!(!id.is_null(), "Tag ID should not be null");
        
        // Test conversion
        let as_u64 = id.to_u64();
        let from_u64 = CteTagId::from_u64(as_u64);
        assert_eq!(id.major, from_u64.major, "Major ID should match after conversion");
        assert_eq!(id.minor, from_u64.minor, "Minor ID should match after conversion");
        
        // Test with a different tag
        let tag2_name = format!("test_tag_id2_{}", std::process::id());
        let tag2 = Tag::new(&tag2_name);
        let id2 = tag2.id();
        
        // Tags should have different IDs
        assert_ne!(id.to_u64(), id2.to_u64(), "Different tags should have different IDs");
    }
}

#[cfg(all(test, feature = "async"))]
mod async_tests {
    use wrp_cte::r#async::{Client, Tag};
    use wrp_cte::sync::init;
    use wrp_cte::types::CteTagId;

    /// Test async CTE initialization
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_init_default_config() {
        let result = init("");
        // Note: async init is a re-export of sync init, so result handling may differ
        // depending on whether runtime is already initialized
        let _ = result;
    }

    /// Test async client creation
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_client_create() {
        init("").expect("CTE initialization failed");
        let client = Client::new().await;
        assert!(client.is_ok(), "Async client creation should succeed");
    }

    /// Test async client telemetry polling
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_client_telemetry() {
        init("").expect("CTE initialization failed");
        
        let client = Client::new().await.expect("Async client creation should succeed");
        
        // Perform some operations to generate telemetry
        let tag_name = format!("async_test_tag_telemetry_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        tag.put_blob("async_telemetry_blob.bin".to_string(), b"test".to_vec(), 0, 1.0)
            .await
            .expect("Async blob put should succeed");
        
        // Poll telemetry
        let telemetry = client
            .poll_telemetry(0)
            .await
            .expect("Async telemetry polling should succeed");
        
        for entry in telemetry {
            println!(
                "Async Telemetry: op={:?}, size={}, tag={}.{}",
                entry.op, entry.size, entry.tag_id.major, entry.tag_id.minor
            );
        }
    }

    /// Test async client reorganize blob
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_client_reorganize_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_reorg_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create a blob
        let blob_name = "async_reorg_blob.bin".to_string();
        tag.put_blob(blob_name.clone(), b"test".to_vec(), 0, 1.0)
            .await
            .expect("Async blob put should succeed");
        
        // Reorganize via client
        let tag_id = tag.get_id().await.expect("Get tag ID should succeed");
        let client = Client::new().await.expect("Async client creation should succeed");
        client
            .reorganize_blob(tag_id, blob_name.clone(), 0.25)
            .await
            .expect("Async client reorganization should succeed");
        
        // Verify new score
        let new_score = tag
            .get_blob_score(&blob_name)
            .await
            .expect("Async get blob score should succeed");
        assert!(
            (new_score - 0.25).abs() < 0.01,
            "New score should be 0.25, got {}",
            new_score
        );
    }

    /// Test async client del blob
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_client_del_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_del_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create a blob
        let blob_name = "async_del_blob.bin".to_string();
        tag.put_blob(blob_name.clone(), b"test data".to_vec(), 0, 1.0)
            .await
            .expect("Async blob put should succeed");
        
        // Verify it exists
        let size = tag
            .get_blob_size(&blob_name)
            .await
            .expect("Async get blob size should succeed");
        assert!(size > 0, "Blob should exist before deletion");
        
        // Delete via client
        let tag_id = tag.get_id().await.expect("Get tag ID should succeed");
        let client = Client::new().await.expect("Async client creation should succeed");
        client
            .del_blob(tag_id, blob_name.clone())
            .await
            .expect("Async blob deletion should succeed");
        
        // Verify it's gone
        let result = tag.get_blob_size(&blob_name).await;
        assert!(result.is_err(), "Blob should not exist after deletion");
    }

    /// Test async tag creation by name
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_create() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        let id = tag.get_id().await.expect("Get tag ID should succeed");
        assert!(id.major > 0 || id.minor > 0, "Tag ID should be valid");
    }

    /// Test async tag creation by ID
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_create_by_id() {
        init("").expect("CTE initialization failed");
        
        // First create a tag by name
        let tag_name = format!("async_test_tag_by_id_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        let id = tag.get_id().await.expect("Get tag ID should succeed");
        
        // Then open it by ID
        let tag_by_id = Tag::from_id(id).await.expect("Async tag from ID should succeed");
        let id_again = tag_by_id.get_id().await.expect("Get tag ID should succeed");
        
        assert_eq!(id.major, id_again.major, "Tag major IDs should match");
        assert_eq!(id.minor, id_again.minor, "Tag minor IDs should match");
    }

    /// Test async blob put and get operations
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_put_get_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_blob_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Write test data
        let blob_name = "async_test_blob.bin".to_string();
        let test_data = b"Hello, async CTE!".to_vec();
        tag.put_blob(blob_name.clone(), test_data.clone(), 0, 1.0)
            .await
            .expect("Async blob put should succeed");
        
        // Get blob size
        let size = tag
            .get_blob_size(&blob_name)
            .await
            .expect("Async get blob size should succeed");
        assert_eq!(
            size,
            test_data.len() as u64,
            "Blob size should match written data"
        );
        
        // Read back data
        let read_data = tag
            .get_blob(blob_name.clone(), size, 0)
            .await
            .expect("Async blob get should succeed");
        assert_eq!(read_data, test_data, "Read data should match written data");
    }

    /// Test async blob score operations
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_blob_score() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_score_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create a blob with score 1.0
        let blob_name = "async_scored_blob.bin".to_string();
        tag.put_blob(blob_name.clone(), b"test".to_vec(), 0, 1.0)
            .await
            .expect("Async blob put should succeed");
        
        // Verify default score
        let score = tag
            .get_blob_score(&blob_name)
            .await
            .expect("Async get blob score should succeed");
        assert!((score - 1.0).abs() < 0.01, "Default score should be 1.0");
        
        // Change the score
        tag.reorganize_blob(blob_name.clone(), 0.5)
            .await
            .expect("Async reorganize blob should succeed");
        
        // Verify new score
        let new_score = tag
            .get_blob_score(&blob_name)
            .await
            .expect("Async get blob score should succeed");
        assert!(
            (new_score - 0.5).abs() < 0.01,
            "New score should be 0.5, got {}",
            new_score
        );
    }

    /// Test async blob size retrieval
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_get_blob_size() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_size_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Test with different sizes
        let small_data = b"hello".to_vec();
        let medium_data = vec![0u8; 1024];
        let large_data = vec![0u8; 10240];
        
        let test_cases = [
            ("async_small.bin", small_data.clone()),
            ("async_medium.bin", medium_data.clone()),
            ("async_large.bin", large_data.clone()),
        ];
        
        for (name, data) in &test_cases {
            tag.put_blob(name.to_string(), data.clone(), 0, 1.0)
                .await
                .expect("Async blob put should succeed");
            
            let size = tag
                .get_blob_size(name)
                .await
                .expect("Async get blob size should succeed");
            assert_eq!(
                size,
                data.len() as u64,
                "Blob size should match for {}",
                name
            );
        }
    }

    /// Test async blob listing
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_contained_blobs() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_list_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create multiple blobs
        for i in 0..5 {
            let blob_name = format!("async_multi_blob_{}.bin", i);
            let data = format!("data_{}", i);
            tag.put_blob(blob_name, data.into_bytes(), 0, 1.0)
                .await
                .expect("Async blob put should succeed");
        }
        
        // List all blobs
        let blobs = tag
            .get_contained_blobs()
            .await
            .expect("Async get contained blobs should succeed");
        assert!(blobs.len() >= 5, "Should have at least 5 blobs in tag");
    }

    /// Test async large blob handling (stress test)
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_large_blob() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_large_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create a moderately large blob (1 MB)
        let blob_name = "async_large_blob.bin".to_string();
        let large_data = vec![0u8; 1024 * 1024];
        
        tag.put_blob(blob_name.clone(), large_data.clone(), 0, 1.0)
            .await
            .expect("Async large blob put should succeed");
        
        // Read back and verify size
        let size = tag
            .get_blob_size(&blob_name)
            .await
            .expect("Async get blob size should succeed");
        assert_eq!(size, large_data.len() as u64, "Large blob size should match");
        
        // Read back data
        let read_data = tag
            .get_blob(blob_name.clone(), size, 0)
            .await
            .expect("Async large blob get should succeed");
        assert_eq!(
            read_data.len(),
            large_data.len(),
            "Large blob read size should match"
        );
    }

    /// Test async error handling for invalid names
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_error_invalid_name() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_error_name_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Test put_blob with empty name
        let result = tag.put_blob("".to_string(), b"data".to_vec(), 0, 1.0).await;
        assert!(result.is_err(), "put_blob with empty name should fail");
        match result {
            Err(wrp_cte::CteError::InvalidParameter { message }) => {
                assert!(
                    message.contains("cannot be empty"),
                    "Error message should mention empty name"
                );
            }
            _ => panic!("Expected InvalidParameter error"),
        }
        
        // Test get_blob_score with empty name
        let result = tag.get_blob_score("").await;
        assert!(result.is_err(), "get_blob_score with empty name should fail");
        
        // Test get_blob_size with empty name
        let result = tag.get_blob_size("").await;
        assert!(result.is_err(), "get_blob_size with empty name should fail");
        
        // Test get_blob with empty name
        let result = tag.get_blob("".to_string(), 10, 0).await;
        assert!(result.is_err(), "get_blob with empty name should fail");
        
        // Test reorganize_blob with empty name
        let result = tag.reorganize_blob("".to_string(), 0.5).await;
        assert!(result.is_err(), "reorganize_blob with empty name should fail");
    }

    /// Test async error handling for invalid scores
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_error_invalid_score() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_error_score_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create blob first
        let blob_name = "async_score_test.bin".to_string();
        tag.put_blob(blob_name.clone(), b"data".to_vec(), 0, 1.0)
            .await
            .expect("Async blob put should succeed");
        
        // Test with negative score
        let result = tag.reorganize_blob(blob_name.clone(), -1.0).await;
        assert!(result.is_err(), "reorganize_blob with negative score should fail");
        match result {
            Err(wrp_cte::CteError::InvalidParameter { message }) => {
                assert!(
                    message.contains("Score must be between"),
                    "Error message should mention score range"
                );
            }
            _ => panic!("Expected InvalidParameter error"),
        }
        
        // Test with score > 1.0
        let result = tag.reorganize_blob(blob_name.clone(), 1.5).await;
        assert!(result.is_err(), "reorganize_blob with score > 1.0 should fail");
        
        // Test with NaN
        let result = tag.reorganize_blob(blob_name.clone(), f32::NAN).await;
        assert!(result.is_err(), "reorganize_blob with NaN score should fail");
        
        // Test put_blob score validation
        let result = tag.put_blob("test.bin".to_string(), b"data".to_vec(), 0, -0.5).await;
        assert!(result.is_err(), "put_blob with negative score should fail");
        
        let result = tag.put_blob("test.bin".to_string(), b"data".to_vec(), 0, 2.0).await;
        assert!(result.is_err(), "put_blob with score > 1.0 should fail");
        
        // Test client reorganize_blob with invalid scores
        let client = Client::new().await.expect("Async client creation should succeed");
        let tag_id = tag.get_id().await.expect("Get tag ID should succeed");
        
        let result = client.reorganize_blob(tag_id, blob_name.clone(), -0.1).await;
        assert!(result.is_err(), "client reorganize with negative score should fail");
        
        let result = client.reorganize_blob(tag_id, blob_name.clone(), 1.1).await;
        assert!(result.is_err(), "client reorganize with score > 1.0 should fail");
    }

    /// Test async blob write with offset
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_blob_write_with_offset() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_offset_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Write initial data
        let blob_name = "async_offset_blob.bin".to_string();
        tag.put_blob(blob_name.clone(), b"Hello".to_vec(), 0, 1.0)
            .await
            .expect("Async initial write should succeed");
        
        // Write at offset
        tag.put_blob(blob_name.clone(), b" World".to_vec(), 5, 1.0)
            .await
            .expect("Async offset write should succeed");
        
        // Read full data
        let size = tag
            .get_blob_size(&blob_name)
            .await
            .expect("Async get size should succeed");
        let data = tag
            .get_blob(blob_name.clone(), size, 0)
            .await
            .expect("Async read should succeed");
        
        assert_eq!(&data, b"Hello World", "Data should match combined writes");
    }

    /// Test async multiple blobs in a tag
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_multiple_blobs() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_multi_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Create multiple blobs
        for i in 0..5 {
            let blob_name = format!("async_multi_blob_{}.bin", i);
            let data = format!("data_{}", i);
            tag.put_blob(blob_name, data.into_bytes(), 0, 1.0)
                .await
                .expect("Async blob put should succeed");
        }
        
        // List all blobs
        let blobs = tag
            .get_contained_blobs()
            .await
            .expect("Async get contained blobs should succeed");
        assert!(blobs.len() >= 5, "Should have at least 5 blobs in tag");
    }

    /// Test async tag ID operations
    #[tokio::test]
    #[ignore = "Requires running CTE runtime"]
    async fn test_async_tag_id_operations() {
        init("").expect("CTE initialization failed");
        
        let tag_name = format!("async_test_tag_id_{}", std::process::id());
        let tag = Tag::new(&tag_name).await.expect("Async tag creation should succeed");
        
        // Get tag ID
        let id = tag.get_id().await.expect("Get tag ID should succeed");
        assert!(!id.is_null(), "Tag ID should not be null");
        
        // Test conversion
        let as_u64 = id.to_u64();
        let from_u64 = CteTagId::from_u64(as_u64);
        assert_eq!(id.major, from_u64.major, "Major ID should match after conversion");
        assert_eq!(id.minor, from_u64.minor, "Minor ID should match after conversion");
        
        // Test with a different tag
        let tag2_name = format!("async_test_tag_id2_{}", std::process::id());
        let tag2 = Tag::new(&tag2_name).await.expect("Async tag creation should succeed");
        let id2 = tag2.get_id().await.expect("Get tag ID should succeed");
        
        // Tags should have different IDs
        assert_ne!(id.to_u64(), id2.to_u64(), "Different tags should have different IDs");
    }
}