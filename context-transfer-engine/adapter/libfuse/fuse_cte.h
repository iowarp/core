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

#ifndef WRP_CTE_ADAPTER_LIBFUSE_FUSE_CTE_H_
#define WRP_CTE_ADAPTER_LIBFUSE_FUSE_CTE_H_

#ifdef WRP_CTE_FUSE_ENABLED
#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 35
#endif
#include <fuse3/fuse.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "wrp_cte/core/core_client.h"
#include "wrp_cte/core/core_tasks.h"

namespace wrp::cae::fuse {

/** Default page size for blob I/O */
static constexpr size_t kDefaultPageSize = 1024 * 1024;  // 1 MB

/**
 * CTE-backed filesystem helpers.
 *
 * Design:
 *   - Each file is a CTE Tag whose name is the absolute FUSE path.
 *   - Directories are implicit: a directory "/a/b" exists if any tag
 *     starts with "/a/b/".
 *   - readdir uses AsyncTagQuery with a regex to find direct children.
 *   - File data is stored as page-indexed blobs ("0", "1", "2", ...).
 *
 * No custom DirectoryTree or FsNode — all metadata lives in CTE.
 */

/** Per-open-file handle stored in fuse_file_info::fh */
struct FuseFileHandle {
  wrp_cte::core::TagId tag_id;
  std::string path;
  int flags;
};

// ============================================================================
// CTE helper functions (async API wrappers)
// ============================================================================

/** Query CTE for the authoritative tag size */
static inline size_t CteGetTagSize(const wrp_cte::core::TagId &tag_id) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetTagSize(tag_id);
  task.Wait();
  if (task->GetReturnCode() != 0) return 0;
  return task->tag_size_;
}

/** Delete a CTE tag by name */
static inline void CteDelTag(const std::string &tag_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncDelTag(tag_name);
  task.Wait();
}

/** Get or create a CTE tag, returning its TagId. Returns null id on failure. */
static inline wrp_cte::core::TagId CteGetOrCreateTag(const std::string &name) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncGetOrCreateTag(name);
  task.Wait();
  if (task->GetReturnCode() != 0) return wrp_cte::core::TagId::GetNull();
  return task->tag_id_;
}

/** Check if a tag exists by name using TagQuery with exact match */
static inline bool CteTagExists(const std::string &tag_name) {
  auto *cte_client = WRP_CTE_CLIENT;
  // Escape regex special chars and do exact match
  std::string escaped;
  for (char c : tag_name) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  auto task = cte_client->AsyncTagQuery(escaped, 1);
  task.Wait();
  return task->GetReturnCode() == 0 && !task->results_.empty();
}

/**
 * Query CTE for tags that are direct children of a directory path.
 * For directory "/a/b", finds tags matching "^/a/b/[^/]+$".
 * Returns just the basenames (not full paths).
 */
static inline std::vector<std::string>
CteListDirectChildren(const std::string &dir_path) {
  auto *cte_client = WRP_CTE_CLIENT;

  // Build regex: escape dir_path, then match one path component
  std::string escaped;
  for (char c : dir_path) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  // Ensure trailing slash
  if (!escaped.empty() && escaped.back() != '/') escaped += '/';
  std::string regex = "^" + escaped + "[^/]+$";

  auto task = cte_client->AsyncTagQuery(regex);
  task.Wait();

  std::vector<std::string> basenames;
  if (task->GetReturnCode() != 0) return basenames;

  // Extract basenames from full paths
  size_t prefix_len = dir_path.size();
  if (!dir_path.empty() && dir_path.back() != '/') prefix_len++;
  for (const auto &full_path : task->results_) {
    if (full_path.size() > prefix_len) {
      basenames.push_back(full_path.substr(prefix_len));
    }
  }
  return basenames;
}

/**
 * Find all unique immediate subdirectory names under dir_path.
 * For dir "/a", if tags "/a/b/c.txt" and "/a/b/d.txt" and "/a/e/f.txt" exist,
 * returns {"b", "e"}.
 */
static inline std::vector<std::string>
CteListSubdirs(const std::string &dir_path) {
  auto *cte_client = WRP_CTE_CLIENT;

  // Match any tag that has at least two more path components after dir_path
  std::string escaped;
  for (char c : dir_path) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  if (!escaped.empty() && escaped.back() != '/') escaped += '/';
  // Match tags with at least one more slash after the child component
  std::string regex = "^" + escaped + "[^/]+/.*";

  auto task = cte_client->AsyncTagQuery(regex);
  task.Wait();

  // Extract unique immediate subdirectory names
  std::vector<std::string> subdirs;
  size_t prefix_len = dir_path.size();
  if (!dir_path.empty() && dir_path.back() != '/') prefix_len++;

  for (const auto &full_path : task->results_) {
    if (full_path.size() <= prefix_len) continue;
    std::string remainder = full_path.substr(prefix_len);
    size_t slash_pos = remainder.find('/');
    if (slash_pos != std::string::npos) {
      std::string subdir = remainder.substr(0, slash_pos);
      // Deduplicate
      if (std::find(subdirs.begin(), subdirs.end(), subdir) == subdirs.end()) {
        subdirs.push_back(subdir);
      }
    }
  }
  return subdirs;
}

/**
 * Check if a directory path has any tags underneath it.
 * A directory exists if any tag starts with "dir_path/".
 */
static inline bool CteDirExists(const std::string &dir_path) {
  auto *cte_client = WRP_CTE_CLIENT;
  std::string escaped;
  for (char c : dir_path) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  if (!escaped.empty() && escaped.back() != '/') escaped += '/';
  std::string regex = "^" + escaped + ".*";
  auto task = cte_client->AsyncTagQuery(regex, 1);
  task.Wait();
  return task->GetReturnCode() == 0 && !task->results_.empty();
}

/**
 * Page-based PutBlob: allocate SHM, copy data, async put, wait, free.
 */
static inline bool CtePutBlob(const wrp_cte::core::TagId &tag_id,
                              const std::string &blob_name, const char *data,
                              size_t data_size, size_t blob_off) {
  auto *ipc_manager = CHI_IPC;
  auto *cte_client = WRP_CTE_CLIENT;
  hipc::FullPtr<char> shm_buf = ipc_manager->AllocateBuffer(data_size);
  if (shm_buf.IsNull()) return false;
  memcpy(shm_buf.ptr_, data, data_size);
  hipc::ShmPtr<> shm_ptr(shm_buf.shm_);
  auto task = cte_client->AsyncPutBlob(tag_id, blob_name, blob_off, data_size,
                                       shm_ptr);
  task.Wait();
  ipc_manager->FreeBuffer(shm_buf);
  return task->GetReturnCode() == 0;
}

/**
 * Page-based GetBlob: allocate SHM, async get, wait, copy out, free.
 */
static inline bool CteGetBlob(const wrp_cte::core::TagId &tag_id,
                              const std::string &blob_name, char *data,
                              size_t data_size, size_t blob_off) {
  auto *ipc_manager = CHI_IPC;
  auto *cte_client = WRP_CTE_CLIENT;
  hipc::FullPtr<char> shm_buf = ipc_manager->AllocateBuffer(data_size);
  if (shm_buf.IsNull()) return false;
  hipc::ShmPtr<> shm_ptr(shm_buf.shm_);
  auto task = cte_client->AsyncGetBlob(tag_id, blob_name, blob_off, data_size,
                                       0, shm_ptr);
  task.Wait();
  bool ok = (task->GetReturnCode() == 0);
  if (ok) memcpy(data, shm_buf.ptr_, data_size);
  ipc_manager->FreeBuffer(shm_buf);
  return ok;
}

/** Escape a string for use as a literal in std::regex */
static inline std::string RegexEscape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      out += '\\';
    }
    out += c;
  }
  return out;
}

}  // namespace wrp::cae::fuse

#endif  // WRP_CTE_ADAPTER_LIBFUSE_FUSE_CTE_H_
