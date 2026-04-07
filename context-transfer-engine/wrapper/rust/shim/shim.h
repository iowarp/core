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

#pragma once

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "rust/cxx.h"

namespace cte_ffi {

// Thread-safe initialization
extern std::once_flag g_init_flag;
extern bool g_init_done;

// Opaque wrapper types - shared across FFI boundary
struct Client {
  mutable wrp_cte::core::Client inner;
};

struct Tag {
  mutable wrp_cte::core::Tag inner;

  explicit Tag(const std::string& name) : inner(name) {}
  explicit Tag(const chi::UniqueId& id) : inner(id) {}
};

// Initialization
int32_t cte_init(rust::Str config_path);

// Client operations
std::unique_ptr<Client> client_new();

// Tag factory functions
std::unique_ptr<Tag> tag_new(rust::Str name);
std::unique_ptr<Tag> tag_from_id(uint32_t major, uint32_t minor);

// Tag blob operations - simple scalar returns only
float tag_get_blob_score(const Tag& tag, rust::Str name);
int32_t tag_reorganize_blob(const Tag& tag, rust::Str name, float score);
uint64_t tag_get_blob_size(const Tag& tag, rust::Str name);

// Operations with buffers - avoid shared struct returns
// Returns 0 on success, negative on error
// -1 = size limit exceeded, -2 = offset overflow
int32_t client_reorganize_blob(const Client& client, uint32_t major,
                               uint32_t minor, rust::Str name, float score);

int32_t client_del_blob(const Client& client, uint32_t major, uint32_t minor,
                        rust::Str name);

int32_t tag_put_blob(const Tag& tag, rust::Str name,
                     rust::Slice<const uint8_t> data, uint64_t offset,
                     float score);

void tag_get_blob(const Tag& tag, rust::Str name, uint64_t size,
                  uint64_t offset, rust::Vec<uint8_t>& out);

void tag_get_contained_blobs(const Tag& tag, rust::Vec<rust::String>& out);

// Telemetry - returns flat array: each entry is (op:u32, off:u64, size:u64,
// tag_major:u32, tag_minor:u32, mod_time_nanos:i64, read_time_nanos:i64,
// logical_time:u64) Total 52 bytes per entry. Caller interprets the byte
// buffer.
void client_poll_telemetry_raw(const Client& client, uint64_t min_time,
                               rust::Vec<uint8_t>& out);

// GetBlobInfo - returns blob metadata with block placement
// Format: score(f32) + total_size(u64) + blocks_count(u32) + blocks[...]
// Each block: pool_id(u64) + block_size(u64) + block_offset(u64) = 24 bytes
int32_t client_get_blob_info_raw(const Client& client, uint32_t major,
                                 uint32_t minor, rust::Str name,
                                 rust::Vec<uint8_t>& out);

// Tag ID helpers (exposed for Rust-side conversion)
uint32_t tag_get_id_major(const Tag& tag);
uint32_t tag_get_id_minor(const Tag& tag);

}  // namespace cte_ffi
