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

#include "shim/shim.h"

#include <chimaera/bdev/bdev_client.h>
#include <wrp_cte/core/content_transfer_engine.h>
#include <wrp_cte/core/core_client.h>

namespace cte_ffi {

// Maximum blob size (16 GB) - must match Rust constant
constexpr uint64_t MAX_BLOB_SIZE = 16ULL * 1024ULL * 1024ULL * 1024ULL;

// Thread-safe initialization globals
std::once_flag g_init_flag;
bool g_init_done = false;

// Initialization with thread safety
int32_t cte_init(rust::Str config_path) {
  std::call_once(g_init_flag, [&]() {
    std::string path(config_path.data(), config_path.size());
    bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!ok) {
      g_init_done = false;
      return;
    }
    // WRP_CTE_CLIENT_INIT is in wrp_cte::core namespace (inside the namespace
    // block)
    g_init_done = ::wrp_cte::core::WRP_CTE_CLIENT_INIT(path);
  });
  return g_init_done ? 0 : -1;
}

// Client factory
std::unique_ptr<Client> client_new() { return std::make_unique<Client>(); }

// Tag factory functions
std::unique_ptr<Tag> tag_new(rust::Str name) {
  std::string n(name.data(), name.size());
  return std::make_unique<Tag>(n);
}

std::unique_ptr<Tag> tag_from_id(uint32_t major, uint32_t minor) {
  chi::UniqueId id(major, minor);
  return std::make_unique<Tag>(id);
}

// Tag ID helpers
uint32_t tag_get_id_major(const Tag& tag) {
  return tag.inner.GetTagId().major_;
}
uint32_t tag_get_id_minor(const Tag& tag) {
  return tag.inner.GetTagId().minor_;
}

// Tag operations - simple scalars
float tag_get_blob_score(const Tag& tag, rust::Str name) {
  std::string n(name.data(), name.size());
  return tag.inner.GetBlobScore(n);
}

int32_t tag_reorganize_blob(const Tag& tag, rust::Str name, float score) {
  std::string n(name.data(), name.size());
  tag.inner.ReorganizeBlob(n, score);
  // Tag::ReorganizeBlob is synchronous and returns void
  // Return 0 for success (no error detection available from sync API)
  return 0;
}

uint64_t tag_get_blob_size(const Tag& tag, rust::Str name) {
  std::string n(name.data(), name.size());
  return tag.inner.GetBlobSize(n);
}

// Client operations with simple returns
int32_t client_reorganize_blob(const Client& client, uint32_t major,
                               uint32_t minor, rust::Str name, float score) {
  chi::UniqueId tag_id(major, minor);
  std::string blob_name(name.data(), name.size());
  auto task = client.inner.AsyncReorganizeBlob(tag_id, blob_name, score);
  task.Wait();
  return task->GetReturnCode();
}

int32_t client_del_blob(const Client& client, uint32_t major, uint32_t minor,
                        rust::Str name) {
  chi::UniqueId tag_id(major, minor);
  std::string blob_name(name.data(), name.size());
  auto task = client.inner.AsyncDelBlob(tag_id, blob_name);
  task.Wait();
  return task->GetReturnCode();
}

// Tag operations with buffers
int32_t tag_put_blob(const Tag& tag, rust::Str name,
                     rust::Slice<const uint8_t> data, uint64_t offset,
                     float score) {
  std::string n(name.data(), name.size());

  // Validate blob size
  uint64_t data_size = data.size();
  if (data_size > MAX_BLOB_SIZE) {
    return -1;  // Error: data too large
  }

  // Check for offset overflow
  if (offset > MAX_BLOB_SIZE - data_size) {
    return -2;  // Error: offset + size overflow
  }

  tag.inner.PutBlob(n, reinterpret_cast<const char*>(data.data()), data.size(),
                    static_cast<size_t>(offset), score);
  return 0;  // Success
}

void tag_get_blob(const Tag& tag, rust::Str name, uint64_t size,
                  uint64_t offset, rust::Vec<uint8_t>& out) {
  std::string n(name.data(), name.size());
  auto buf = std::vector<uint8_t>(size);
  tag.inner.GetBlob(n, reinterpret_cast<char*>(buf.data()),
                    static_cast<size_t>(size), static_cast<size_t>(offset));
  out.clear();
  out.reserve(buf.size());
  for (auto b : buf) {
    out.push_back(b);
  }
}

void tag_get_contained_blobs(const Tag& tag, rust::Vec<rust::String>& out) {
  auto blobs = tag.inner.GetContainedBlobs();
  out.clear();
  out.reserve(blobs.size());
  for (const auto& b : blobs) {
    out.push_back(rust::String(b));
  }
}

// Telemetry - encoded as raw bytes for Rust to decode
// Each entry: op(u32) + off(u64) + size(u64) + tag_major(u32) + tag_minor(u32)
// +
//             mod_time_nanos(i64) + read_time_nanos(i64) + logical_time(u64)
// = 4 + 8 + 8 + 4 + 4 + 8 + 8 + 8 = 52 bytes per entry
void client_poll_telemetry_raw(const Client& client, uint64_t min_time,
                               rust::Vec<uint8_t>& out) {
  auto task = client.inner.AsyncPollTelemetryLog(min_time);
  task.Wait();

  out.clear();
  out.reserve(task->entries_.size() * 52);

  for (const auto& entry : task->entries_) {
    // op (u32)
    uint32_t op = static_cast<uint32_t>(entry.op_);
    out.push_back(static_cast<uint8_t>((op >> 0) & 0xFF));
    out.push_back(static_cast<uint8_t>((op >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((op >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((op >> 24) & 0xFF));

    // off (u64)
    uint64_t off = entry.off_;
    for (int i = 0; i < 8; ++i) {
      out.push_back(static_cast<uint8_t>((off >> (i * 8)) & 0xFF));
    }

    // size (u64)
    uint64_t sz = entry.size_;
    for (int i = 0; i < 8; ++i) {
      out.push_back(static_cast<uint8_t>((sz >> (i * 8)) & 0xFF));
    }

    // tag_major (u32)
    uint32_t major = entry.tag_id_.major_;
    out.push_back(static_cast<uint8_t>((major >> 0) & 0xFF));
    out.push_back(static_cast<uint8_t>((major >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((major >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((major >> 24) & 0xFF));

    // tag_minor (u32)
    uint32_t minor = entry.tag_id_.minor_;
    out.push_back(static_cast<uint8_t>((minor >> 0) & 0xFF));
    out.push_back(static_cast<uint8_t>((minor >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((minor >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((minor >> 24) & 0xFF));

    // mod_time_nanos (i64)
    int64_t mod_time = entry.mod_time_.time_since_epoch().count();
    for (int i = 0; i < 8; ++i) {
      out.push_back(static_cast<uint8_t>((mod_time >> (i * 8)) & 0xFF));
    }

    // read_time_nanos (i64)
    int64_t read_time = entry.read_time_.time_since_epoch().count();
    for (int i = 0; i < 8; ++i) {
      out.push_back(static_cast<uint8_t>((read_time >> (i * 8)) & 0xFF));
    }

    // logical_time (u64)
    uint64_t logical = entry.logical_time_;
    for (int i = 0; i < 8; ++i) {
      out.push_back(static_cast<uint8_t>((logical >> (i * 8)) & 0xFF));
    }
  }
}

}  // namespace cte_ffi