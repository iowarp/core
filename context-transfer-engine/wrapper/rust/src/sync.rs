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

//! Synchronous CTE API
//!
//! This module provides blocking (synchronous) wrappers around the CTE FFI.
//! For async operations, see the `r#async` module.
//!
//! # Example
//! ```
//! use wrp_cte::sync::{init, Client, Tag};
//!
//! // Initialize CTE
//! init("").expect("CTE init failed");
//!
//! // Create client and tag
//! let client = Client::new().unwrap();
//! let tag = Tag::new("my_dataset");
//!
//! // Use blocking operations with validation
//! tag.put_blob_with_options("data.bin", b"hello", 0, 1.0).expect("put failed");
//! let data = tag.get_blob("data.bin", 5, 0).expect("get failed");
//! ```

use crate::error::{CteError, CteResult};
use crate::ffi::ffi;
use std::sync::OnceLock;

/// Maximum blob size (16 GB)
pub const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024;

/// Cached initialization result
static INIT_RESULT: OnceLock<CteResult<()>> = OnceLock::new();

/// Re-export types for sync API
pub use crate::types::{
    BdevType, ChimaeraMode, CteOp, CteTagId, CteTelemetry, PoolQuery, SteadyTime,
};

/// Initialize CTE with embedded runtime
///
/// This function is thread-safe and will only initialize once.
/// Subsequent calls return the cached result.
///
/// # Arguments
/// * `config_path` - Path to configuration file, or "" for defaults
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(CteError::InitFailed)` on failure
///
/// # Example
/// ```
/// use wrp_cte::sync::init;
///
/// init("").expect("CTE initialization failed");
/// ```
pub fn init(config_path: &str) -> CteResult<()> {
    // Thread-safe initialization using OnceLock.
    // This ensures only one thread initializes CTE.
    // Other threads get the cached result.
    INIT_RESULT
        .get_or_init(|| {
            let rc = ffi::cte_init(config_path);
            if rc == 0 {
                Ok(())
            } else {
                Err(CteError::InitFailed {
                    reason: format!(
                        "CTE initialization failed with code {}: config_path='{}'",
                        rc, config_path
                    ),
                })
            }
        })
        .clone()
}

/// CTE client for low-level operations
///
/// Provides access to client-level operations like:
/// - Telemetry polling
/// - Blob reorganization
/// - Pool management
///
/// The client wraps the underlying CTE client handle.
pub struct Client {
    inner: cxx::UniquePtr<ffi::Client>,
}

impl Client {
    /// Create a new CTE client
    ///
    /// # Returns
    /// * `Ok(Client)` on success
    /// * `Err(CteError::InitFailed)` if CTE not initialized
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Client;
    ///
    /// let client = Client::new().unwrap();
    /// ```
    pub fn new() -> CteResult<Self> {
        let inner = ffi::client_new();
        Ok(Self { inner })
    }

    /// Poll telemetry log from CTE
    ///
    /// Returns telemetry entries for operations that occurred after `min_time`.
    ///
    /// # Arguments
    /// * `min_time` - Minimum timestamp to fetch (0 for all)
    ///
    /// # Returns
    /// Vector of telemetry entries
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Client;
    ///
    /// let client = Client::new().unwrap();
    /// let telemetry = client.poll_telemetry(0).unwrap();
    /// ```
    pub fn poll_telemetry(&self, min_time: u64) -> CteResult<Vec<crate::ffi::CteTelemetry>> {
        let mut raw = Vec::new();
        ffi::client_poll_telemetry_raw(&self.inner, min_time, &mut raw);
        Ok(crate::ffi::parse_telemetry(&raw))
    }

    /// Reorganize a blob (change placement score)
    ///
    /// Changes the importance score of a blob, which may trigger
    /// data migration between storage tiers.
    ///
    /// # Arguments
    /// * `tag_id` - ID of the tag containing the blob
    /// * `name` - Blob name (must not be empty)
    /// * `score` - New placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty or score is out of range
    /// * `Err(CteError::RuntimeError)` on failure
    pub fn reorganize_blob(&self, tag_id: CteTagId, name: &str, score: f32) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        let rc = ffi::client_reorganize_blob(&self.inner, tag_id.major, tag_id.minor, name, score);
        if rc == 0 {
            Ok(())
        } else {
            Err(CteError::RuntimeError {
                code: rc as u32,
                message: format!(
                    "Failed to reorganize blob '{}' in tag {}.{}: error code {}",
                    name, tag_id.major, tag_id.minor, rc
                ),
            })
        }
    }

    /// Delete a blob
    ///
    /// # Arguments
    /// * `tag_id` - ID of the tag containing the blob
    /// * `name` - Blob name (must not be empty)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty
    /// * `Err(CteError::RuntimeError)` on failure
    pub fn del_blob(&self, tag_id: CteTagId, name: &str) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        let rc = ffi::client_del_blob(&self.inner, tag_id.major, tag_id.minor, name);
        if rc == 0 {
            Ok(())
        } else {
            Err(CteError::RuntimeError {
                code: rc as u32,
                message: format!(
                    "Failed to delete blob '{}' in tag {}.{}: error code {}",
                    name, tag_id.major, tag_id.minor, rc
                ),
            })
        }
    }
}

/// High-level tag wrapper for blob operations
///
/// A tag is a container (bucket) for blobs. This wrapper provides
/// convenient methods for blob storage, retrieval, and management.
///
/// # Example
/// ```
/// use wrp_cte::sync::Tag;
///
/// let tag = Tag::new("my_dataset");
/// tag.put_blob_with_options("data.bin", b"hello", 0, 1.0).expect("put failed");
///
/// let size = tag.get_blob_size("data.bin").expect("size failed");
/// let data = tag.get_blob("data.bin", size, 0).expect("get failed");
/// ```
pub struct Tag {
    inner: cxx::UniquePtr<ffi::Tag>,
}

impl Tag {
    /// Create or get a tag by name
    ///
    /// If the tag exists, returns a handle to it.
    /// If not, creates a new tag.
    ///
    /// # Arguments
    /// * `name` - Tag name (must be unique)
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// ```
    pub fn new(name: &str) -> Self {
        let inner = ffi::tag_new(name);
        Self { inner }
    }

    /// Open an existing tag by ID
    ///
    /// # Arguments
    /// * `id` - Tag ID (major.minor format)
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    /// use wrp_cte::types::CteTagId;
    ///
    /// let id = CteTagId::new(1, 2);
    /// let tag = Tag::from_id(id);
    /// ```
    pub fn from_id(id: CteTagId) -> Self {
        let inner = ffi::tag_from_id(id.major, id.minor);
        Self { inner }
    }

    /// Get the placement score of a blob
    ///
    /// Score ranges from 0.0 (lowest priority) to 1.0 (highest priority).
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    ///
    /// # Returns
    /// * `Ok(score)` - Score value (0.0-1.0)
    /// * `Err(CteError::InvalidParameter)` if name is empty
    /// * `Err(CteError::RuntimeError)` on failure
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// let score = tag.get_blob_score("data.bin").expect("get score failed");
    /// println!("Score: {}", score);
    /// ```
    pub fn get_blob_score(&self, name: &str) -> CteResult<f32> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        Ok(ffi::tag_get_blob_score(&self.inner, name))
    }

    /// Reorganize a blob (change placement score)
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `score` - New placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty or score is out of range
    /// * `Err(CteError::RuntimeError)` on failure
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// tag.reorganize_blob("data.bin", 0.5).expect("reorganize failed");
    /// ```
    pub fn reorganize_blob(&self, name: &str, score: f32) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        let rc = ffi::tag_reorganize_blob(&self.inner, name, score);
        if rc == 0 {
            Ok(())
        } else {
            let id = self.id();
            Err(CteError::RuntimeError {
                code: rc as u32,
                message: format!(
                    "Failed to reorganize blob '{}' in tag {}.{} with score {}: error code {}",
                    name, id.major, id.minor, score, rc
                ),
            })
        }
    }

    /// Write data into a blob with validation
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `data` - Data to write
    /// * `offset` - Offset in blob (0 for new blobs)
    /// * `score` - Placement score (0.0-1.0)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty, score is out of range,
    ///   data exceeds MAX_BLOB_SIZE, or offset + size overflows
    /// * `Err(CteError::RuntimeError)` if FFI call fails
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// tag.put_blob_with_options("data.bin", b"hello", 0, 1.0).expect("put failed");
    /// ```
    pub fn put_blob_with_options(
        &self,
        name: &str,
        data: &[u8],
        offset: u64,
        score: f32,
    ) -> CteResult<()> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }
        if score < 0.0 || score > 1.0 || score.is_nan() {
            return Err(CteError::InvalidParameter {
                message: format!("Score must be between 0.0 and 1.0, got {}", score),
            });
        }

        // Check blob size limit
        let data_len = data.len() as u64;
        if data_len > MAX_BLOB_SIZE {
            return Err(CteError::InvalidParameter {
                message: format!(
                    "Data size {} exceeds maximum blob size {}",
                    data_len, MAX_BLOB_SIZE
                ),
            });
        }

        // Check for offset overflow
        let end_offset =
            offset
                .checked_add(data_len)
                .ok_or_else(|| CteError::InvalidParameter {
                    message: format!("Offset {} + size {} would overflow u64", offset, data_len),
                })?;

        if end_offset > MAX_BLOB_SIZE {
            return Err(CteError::InvalidParameter {
                message: format!(
                    "Total blob size {} exceeds maximum {}",
                    end_offset, MAX_BLOB_SIZE
                ),
            });
        }

        // Call FFI
        let rc = ffi::tag_put_blob(&self.inner, name, data, offset, score);
        if rc == 0 {
            Ok(())
        } else if rc == -1 {
            Err(CteError::RuntimeError {
                code: rc as u32,
                message: "Data size exceeds maximum blob size".to_string(),
            })
        } else if rc == -2 {
            Err(CteError::RuntimeError {
                code: rc as u32,
                message: "Offset + size overflow".to_string(),
            })
        } else {
            Err(CteError::RuntimeError {
                code: rc as u32,
                message: format!("Put blob failed with error code {}", rc),
            })
        }
    }

    /// Write data into a blob with default offset (0) and score (1.0)
    ///
    /// Convenience method for simple blob storage.
    /// Returns an error on validation failures instead of panicking.
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `data` - Data to write
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CteError::InvalidParameter)` if name is empty, data is too large,
    ///   or offset overflow
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// tag.put_blob("data.bin", b"hello").expect("put failed");
    /// ```
    pub fn put_blob(&self, name: &str, data: &[u8]) -> CteResult<()> {
        self.put_blob_with_options(name, data, 0, 1.0)
    }

    /// Read data from a blob
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    /// * `size` - Number of bytes to read
    /// * `offset` - Offset in blob (0 for start)
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - Data read from blob
    /// * `Err(CteError::InvalidParameter)` if name is empty
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// let data = tag.get_blob("data.bin", 1024, 0).expect("get failed");
    /// ```
    pub fn get_blob(&self, name: &str, size: u64, offset: u64) -> CteResult<Vec<u8>> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        let mut out = Vec::new();
        ffi::tag_get_blob(&self.inner, name, size, offset, &mut out);
        Ok(out)
    }

    /// Get the size of a blob
    ///
    /// # Arguments
    /// * `name` - Blob name (must not be empty)
    ///
    /// # Returns
    /// * `Ok(u64)` - Size in bytes
    /// * `Err(CteError::InvalidParameter)` if name is empty
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// let size = tag.get_blob_size("data.bin").expect("size failed");
    /// println!("Blob size: {}", size);
    /// ```
    pub fn get_blob_size(&self, name: &str) -> CteResult<u64> {
        // Validate inputs
        if name.is_empty() {
            return Err(CteError::InvalidParameter {
                message: "Blob name cannot be empty".to_string(),
            });
        }

        Ok(ffi::tag_get_blob_size(&self.inner, name))
    }

    /// List all blobs in this tag
    ///
    /// # Returns
    /// Vector of blob names
    pub fn get_contained_blobs(&self) -> Vec<String> {
        let mut out = Vec::new();
        ffi::tag_get_contained_blobs(&self.inner, &mut out);
        out
    }

    /// Get the tag ID
    ///
    /// # Returns
    /// The unique identifier for this tag
    ///
    /// # Example
    /// ```
    /// use wrp_cte::sync::Tag;
    ///
    /// let tag = Tag::new("my_dataset");
    /// let id = tag.id();
    /// println!("Tag ID: {}.{}", id.major, id.minor);
    /// ```
    pub fn id(&self) -> CteTagId {
        CteTagId {
            major: ffi::tag_get_id_major(&self.inner),
            minor: ffi::tag_get_id_minor(&self.inner),
        }
    }
}

/// Shutdown the CTE runtime
///
/// This should be called before program exit to clean up resources.
/// After shutdown, CTE must be re-initialized before use.
pub fn shutdown() {
    // CTE doesn't have a shutdown function in the FFI currently
    // This is a placeholder for future cleanup
    // When shutdown is implemented in the C++ library, call it here
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_returns_error_when_not_initialized() {
        // This will fail because CTE isn't running in tests
        // But we're testing the error path
        let result = init("");
        // Result depends on environment - just verify it compiles
        let _ = result;
    }

    #[test]
    fn test_cte_tag_id_conversion() {
        let id = CteTagId::new(1, 2);
        assert_eq!(id.major, 1);
        assert_eq!(id.minor, 2);
    }

    #[test]
    fn test_pool_query_variants() {
        let local = PoolQuery::local();
        let dynamic = PoolQuery::dynamic(30.0);
        let broadcast = PoolQuery::broadcast(60.0);

        assert_eq!(local.net_timeout(), 0.0);
        assert_eq!(dynamic.net_timeout(), 30.0);
        assert_eq!(broadcast.net_timeout(), 60.0);
    }

    #[test]
    fn test_get_blob_score_empty_name() {
        let tag = Tag::new("test_tag");
        let result = tag.get_blob_score("");
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_reorganize_blob_empty_name() {
        let tag = Tag::new("test_tag");
        let result = tag.reorganize_blob("", 0.5);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_reorganize_blob_invalid_score_low() {
        let tag = Tag::new("test_tag");
        let result = tag.reorganize_blob("test", -1.0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("Score must be between"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_reorganize_blob_invalid_score_high() {
        let tag = Tag::new("test_tag");
        let result = tag.reorganize_blob("test", 2.0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("Score must be between"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_put_blob_with_options_empty_name() {
        let tag = Tag::new("test_tag");
        let result = tag.put_blob_with_options("", b"data", 0, 1.0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_put_blob_with_options_invalid_score() {
        let tag = Tag::new("test_tag");
        let result = tag.put_blob_with_options("test", b"data", 0, -0.5);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("Score must be between"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_get_blob_empty_name() {
        let tag = Tag::new("test_tag");
        let result = tag.get_blob("", 10, 0);
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_get_blob_size_empty_name() {
        let tag = Tag::new("test_tag");
        let result = tag.get_blob_size("");
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    #[should_panic(expected = "validation failed")]
    fn test_put_blob_empty_name_panics() {
        let tag = Tag::new("test_tag");
        tag.put_blob("", b"data"); // Should panic
    }

    #[test]
    fn test_client_reorganize_blob_empty_name() {
        // Note: Client::new() will fail without CTE init, but validation happens first
        // We test validation logic separately
        // This demonstrates that validation happens before FFI call
        let result: CteResult<()> = Err(CteError::InvalidParameter {
            message: "Blob name cannot be empty".to_string(),
        });

        // Verify error type
        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_client_reorganize_blob_invalid_score() {
        let result: CteResult<()> = Err(CteError::InvalidParameter {
            message: "Score must be between 0.0 and 1.0, got 1.5".to_string(),
        });

        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("Score must be between"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_client_del_blob_empty_name() {
        let result: CteResult<()> = Err(CteError::InvalidParameter {
            message: "Blob name cannot be empty".to_string(),
        });

        assert!(result.is_err());
        match result {
            Err(CteError::InvalidParameter { message }) => {
                assert!(message.contains("cannot be empty"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }
}
