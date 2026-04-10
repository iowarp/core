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

#include "fuse_cte.h"

#include <climits>
#include <set>

#include "chimaera/chimaera.h"
#include "wrp_cte/core/content_transfer_engine.h"

using namespace wrp::cae::fuse;

// ============================================================================
// Helpers
// ============================================================================

static FuseFileHandle* GetHandle(struct fuse_file_info* fi) {
  return reinterpret_cast<FuseFileHandle*>(fi->fh);
}

// ============================================================================
// FUSE lifecycle
// ============================================================================

static void* cte_fuse_init(struct fuse_conn_info* conn,
                           struct fuse_config* cfg) {
  (void)conn;
  cfg->use_ino = 0;
  cfg->direct_io = 1;

  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!success) {
    fprintf(stderr, "ERROR: CHIMAERA_INIT failed\n");
    return nullptr;
  }
  wrp_cte::core::WRP_CTE_CLIENT_INIT();
  return nullptr;
}

static void cte_fuse_destroy(void* private_data) {
  (void)private_data;
  chi::CHIMAERA_FINALIZE();
}

// ============================================================================
// Metadata
// ============================================================================

static int cte_fuse_getattr(const char* path, struct stat* stbuf,
                            struct fuse_file_info* fi) {
  (void)fi;
  memset(stbuf, 0, sizeof(struct stat));

  std::string p(path);

  // Root is always a directory
  if (p == "/") {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    return 0;
  }

  // Check if path is a file (tag exists with this exact name)
  if (CteTagExists(p)) {
    auto tag_id = CteGetOrCreateTag(p);
    stbuf->st_mode = S_IFREG | 0644;
    stbuf->st_nlink = 1;
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    stbuf->st_size = static_cast<off_t>(CteGetTagSize(tag_id));
    return 0;
  }

  // Check if path is an implicit directory (any tags under this prefix)
  if (CteDirExists(p)) {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    return 0;
  }

  return -ENOENT;
}

static int cte_fuse_utimens(const char* path, const struct timespec tv[2],
                            struct fuse_file_info* fi) {
  (void)path;
  (void)tv;
  (void)fi;
  // CTE timestamps are managed internally; accept silently
  return 0;
}

// ============================================================================
// Directory operations
// ============================================================================

static int cte_fuse_readdir(const char* path, void* buf, fuse_fill_dir_t filler,
                            off_t offset, struct fuse_file_info* fi,
                            enum fuse_readdir_flags flags) {
  (void)offset;
  (void)fi;
  (void)flags;

  std::string p(path);

  filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));

  // Track all entries to avoid duplicates
  std::set<std::string> entries;

  // List direct file children (excluding marker tags)
  auto files = CteListDirectChildren(p);
  for (const auto& name : files) {
    if (name.find(".cte_dir:") == std::string::npos) {
      entries.insert(name);
    }
  }

  // List implicit subdirectories
  auto subdirs = CteListSubdirs(p);
  for (const auto& name : subdirs) {
    entries.insert(name);
  }

  // List explicit directories (markers)
  auto explicit_dirs = CteListExplicitDirs(p);
  for (const auto& name : explicit_dirs) {
    entries.insert(name);
  }

  // Fill directory listing (sorted automatically by std::set)
  for (const auto& name : entries) {
    filler(buf, name.c_str(), nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  }

  return 0;
}

static int cte_fuse_mkdir(const char* path, mode_t mode) {
  (void)mode;
  std::string p(path);

  // Check if already a file (POSIX: EEXIST)
  if (CteTagExists(p)) return -EEXIST;

  // Check if already exists as explicit directory
  if (CteIsExplicitDir(p)) return -EEXIST;  // Already explicit
  // Implicit directories are OK to "promote" to explicit

  // Create directory marker
  if (!CteMakeDir(p)) return -EIO;
  return 0;
}

static int cte_fuse_rmdir(const char* path) {
  std::string p(path);

  // Check if directory exists at all
  if (!CteDirExists(p)) return -ENOENT;

  // Check if directory is empty
  if (!CteIsDirEmpty(p)) return -ENOTEMPTY;

  // Remove explicit marker if present (no-op if implicit only)
  CteRemoveDir(p);

  return 0;
}

// ============================================================================
// File lifecycle
// ============================================================================

static int cte_fuse_create(const char* path, mode_t mode,
                           struct fuse_file_info* fi) {
  (void)mode;
  std::string p(path);

  auto tag_id = CteGetOrCreateTag(p);
  if (tag_id.IsNull()) return -EIO;

  auto* handle = new FuseFileHandle();
  handle->tag_id = tag_id;
  handle->path = p;
  handle->flags = fi->flags;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  return 0;
}

static int cte_fuse_open(const char* path, struct fuse_file_info* fi) {
  std::string p(path);

  if (!CteTagExists(p)) return -ENOENT;

  auto tag_id = CteGetOrCreateTag(p);
  if (tag_id.IsNull()) return -EIO;

  auto* handle = new FuseFileHandle();
  handle->tag_id = tag_id;
  handle->path = p;
  handle->flags = fi->flags;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  return 0;
}

static int cte_fuse_release(const char* path, struct fuse_file_info* fi) {
  (void)path;
  delete GetHandle(fi);
  fi->fh = 0;
  return 0;
}

// ============================================================================
// Read / Write — page-based I/O
// ============================================================================

static int cte_fuse_read(const char* path, char* buf, size_t size, off_t offset,
                         struct fuse_file_info* fi) {
  (void)path;
  auto* handle = GetHandle(fi);

  if (size > static_cast<size_t>(INT_MAX)) size = static_cast<size_t>(INT_MAX);

  size_t file_size = CteGetTagSize(handle->tag_id);
  if (static_cast<size_t>(offset) >= file_size) return 0;
  if (static_cast<size_t>(offset) + size > file_size) size = file_size - offset;

  size_t bytes_read = 0;
  size_t cur = static_cast<size_t>(offset);

  while (bytes_read < size) {
    size_t page = cur / kDefaultPageSize;
    size_t poff = cur % kDefaultPageSize;
    size_t to_read = std::min(kDefaultPageSize - poff, size - bytes_read);

    if (!CteGetBlob(handle->tag_id, std::to_string(page), buf + bytes_read,
                    to_read, poff))
      break;

    bytes_read += to_read;
    cur += to_read;
  }
  return static_cast<int>(bytes_read);
}

static int cte_fuse_write(const char* path, const char* buf, size_t size,
                          off_t offset, struct fuse_file_info* fi) {
  (void)path;
  auto* handle = GetHandle(fi);

  if (size > static_cast<size_t>(INT_MAX)) size = static_cast<size_t>(INT_MAX);

  size_t bytes_written = 0;
  size_t cur = static_cast<size_t>(offset);

  while (bytes_written < size) {
    size_t page = cur / kDefaultPageSize;
    size_t poff = cur % kDefaultPageSize;
    size_t to_write = std::min(kDefaultPageSize - poff, size - bytes_written);

    if (!CtePutBlob(handle->tag_id, std::to_string(page), buf + bytes_written,
                    to_write, poff)) {
      if (bytes_written == 0) return -EIO;
      break;
    }

    bytes_written += to_write;
    cur += to_write;
  }
  return static_cast<int>(bytes_written);
}

// ============================================================================
// Unlink / Truncate
// ============================================================================

static int cte_fuse_unlink(const char* path) {
  std::string p(path);
  if (!CteTagExists(p)) return -ENOENT;
  CteDelTag(p);
  return 0;
}

static int cte_fuse_truncate(const char* path, off_t size,
                             struct fuse_file_info* fi) {
  (void)fi;
  (void)size;
  std::string p(path);
  if (!CteTagExists(p)) return -ENOENT;
  // CTE does not yet support blob truncation.
  return 0;
}

// ============================================================================
// Main
// ============================================================================

static const struct fuse_operations cte_fuse_ops = {
    .getattr = cte_fuse_getattr,
    .mkdir = cte_fuse_mkdir,
    .unlink = cte_fuse_unlink,
    .rmdir = cte_fuse_rmdir,
    .truncate = cte_fuse_truncate,
    .open = cte_fuse_open,
    .read = cte_fuse_read,
    .write = cte_fuse_write,
    .release = cte_fuse_release,
    .readdir = cte_fuse_readdir,
    .init = cte_fuse_init,
    .destroy = cte_fuse_destroy,
    .create = cte_fuse_create,
    .utimens = cte_fuse_utimens,
};

int main(int argc, char* argv[]) {
  return fuse_main(argc, argv, &cte_fuse_ops, nullptr);
}
