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
#include <cstring>
#include <string>
#include <vector>

#include <fuse3/fuse_lowlevel.h>  // fuse_session_custom_io, struct fuse_custom_io
#include <sys/uio.h>              // struct iovec, writev
#include <sys/mount.h>            // mount syscall
#include <unistd.h>               // read, getuid, getgid
#include <cerrno>                 // errno
#include <cstdio>                 // snprintf, fprintf

#include "chimaera/chimaera.h"
#include "wrp_cte/core/content_transfer_engine.h"

using namespace wrp::cae::fuse;

// ============================================================================
// Helpers
// ============================================================================

static FuseFileHandle *GetHandle(struct fuse_file_info *fi) {
  return reinterpret_cast<FuseFileHandle *>(fi->fh);
}

// ============================================================================
// FUSE lifecycle
// ============================================================================

static void *cte_fuse_init(struct fuse_conn_info *conn,
                           struct fuse_config *cfg) {
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

static void cte_fuse_destroy(void *private_data) {
  (void)private_data;
  chi::CHIMAERA_FINALIZE();
}

// ============================================================================
// Metadata
// ============================================================================

static int cte_fuse_getattr(const char *path, struct stat *stbuf,
                            struct fuse_file_info *fi) {
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

  // Check if path is an explicit directory (sentinel tag with trailing /)
  // or an implicit directory (any tags under this prefix).
  // Check directories BEFORE files so that "mkdir foo && touch foo" doesn't
  // shadow the directory with a file of the same name.
  if (CteTagExists(p + "/") || CteDirExists(p)) {
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

  return -ENOENT;
}

static int cte_fuse_utimens(const char *path, const struct timespec tv[2],
                            struct fuse_file_info *fi) {
  (void)path;
  (void)tv;
  (void)fi;
  // CTE timestamps are managed internally; accept silently
  return 0;
}

// ============================================================================
// Directory operations
// ============================================================================

static int cte_fuse_readdir(const char *path, void *buf,
                            fuse_fill_dir_t filler, off_t offset,
                            struct fuse_file_info *fi,
                            enum fuse_readdir_flags flags) {
  (void)offset;
  (void)fi;
  (void)flags;

  std::string p(path);

  filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));

  // List direct file children (tags whose full path is dir/name with no further slashes)
  auto files = CteListDirectChildren(p);
  for (const auto &name : files) {
    filler(buf, name.c_str(), nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  }

  // List implicit subdirectories (dirs with files beneath them)
  auto subdirs = CteListSubdirs(p);

  // Also find explicit empty directories (sentinel tags ending with /).
  // Sentinel tags under "/a/b" look like "/a/b/childdir/" — match with
  // regex "^/a/b/[^/]+/$" to find direct child sentinels.
  {
    auto *cte_client = WRP_CTE_CLIENT;
    std::string escaped = RegexEscape(p);
    if (!escaped.empty() && escaped.back() != '/') escaped += '/';
    std::string regex = "^" + escaped + "[^/]+/$";
    auto task = cte_client->AsyncTagQuery(regex);
    task.Wait();
    if (task->GetReturnCode() == 0) {
      size_t prefix_len = p.size();
      if (!p.empty() && p.back() != '/') prefix_len++;
      for (const auto &sentinel : task->results_) {
        // Extract dirname: strip prefix and trailing /
        if (sentinel.size() > prefix_len + 1) {
          std::string dirname = sentinel.substr(prefix_len,
                                                sentinel.size() - prefix_len - 1);
          if (std::find(subdirs.begin(), subdirs.end(), dirname) == subdirs.end()) {
            subdirs.push_back(dirname);
          }
        }
      }
    }
  }

  for (const auto &name : subdirs) {
    // Avoid duplicates if a file and subdir have the same name
    if (std::find(files.begin(), files.end(), name) == files.end()) {
      filler(buf, name.c_str(), nullptr, 0,
             static_cast<fuse_fill_dir_flags>(0));
    }
  }

  return 0;
}

static int cte_fuse_mkdir(const char *path, mode_t mode) {
  (void)mode;
  std::string p(path);
  // Create a sentinel tag with a trailing "/" to mark an explicit directory.
  // This lets getattr report the directory before any files exist under it.
  std::string sentinel = p + "/";
  auto tag_id = CteGetOrCreateTag(sentinel);
  if (tag_id.IsNull()) return -EIO;
  return 0;
}

static int cte_fuse_rmdir(const char *path) {
  std::string p(path);
  // A directory can only be removed if it has no children
  auto files = CteListDirectChildren(p);
  auto subdirs = CteListSubdirs(p);
  if (!files.empty() || !subdirs.empty()) return -ENOTEMPTY;
  // Delete the sentinel tag if it exists
  std::string sentinel = p + "/";
  if (CteTagExists(sentinel)) {
    CteDelTag(sentinel);
  }
  return 0;
}

// ============================================================================
// File lifecycle
// ============================================================================

static int cte_fuse_create(const char *path, mode_t mode,
                           struct fuse_file_info *fi) {
  (void)mode;
  std::string p(path);

  auto tag_id = CteGetOrCreateTag(p);
  if (tag_id.IsNull()) return -EIO;

  auto *handle = new FuseFileHandle();
  handle->tag_id = tag_id;
  handle->path = p;
  handle->flags = fi->flags;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  return 0;
}

static int cte_fuse_open(const char *path, struct fuse_file_info *fi) {
  std::string p(path);

  if (!CteTagExists(p)) return -ENOENT;

  auto tag_id = CteGetOrCreateTag(p);
  if (tag_id.IsNull()) return -EIO;

  auto *handle = new FuseFileHandle();
  handle->tag_id = tag_id;
  handle->path = p;
  handle->flags = fi->flags;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  return 0;
}

static int cte_fuse_release(const char *path, struct fuse_file_info *fi) {
  (void)path;
  delete GetHandle(fi);
  fi->fh = 0;
  return 0;
}

// ============================================================================
// Read / Write — page-based I/O
// ============================================================================

static int cte_fuse_read(const char *path, char *buf, size_t size,
                         off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);

  size_t file_size = CteGetTagSize(handle->tag_id);
  if (static_cast<size_t>(offset) >= file_size) return 0;
  if (static_cast<size_t>(offset) + size > file_size)
    size = file_size - offset;

  size_t bytes_read = 0;
  size_t cur = static_cast<size_t>(offset);

  while (bytes_read < size) {
    size_t page = cur / kDefaultPageSize;
    size_t poff = cur % kDefaultPageSize;
    size_t to_read = std::min(kDefaultPageSize - poff, size - bytes_read);

    if (!CteGetBlob(handle->tag_id, std::to_string(page),
                    buf + bytes_read, to_read, poff))
      break;

    bytes_read += to_read;
    cur += to_read;
  }
  return static_cast<int>(bytes_read);
}

static int cte_fuse_write(const char *path, const char *buf, size_t size,
                          off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);

  size_t bytes_written = 0;
  size_t cur = static_cast<size_t>(offset);

  while (bytes_written < size) {
    size_t page = cur / kDefaultPageSize;
    size_t poff = cur % kDefaultPageSize;
    size_t to_write = std::min(kDefaultPageSize - poff, size - bytes_written);

    if (!CtePutBlob(handle->tag_id, std::to_string(page),
                    buf + bytes_written, to_write, poff)) {
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

static int cte_fuse_unlink(const char *path) {
  std::string p(path);
  if (!CteTagExists(p)) return -ENOENT;
  CteDelTag(p);
  return 0;
}

static int cte_fuse_truncate(const char *path, off_t size,
                             struct fuse_file_info *fi) {
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

// custom_io callbacks: when apptainer hands us an already-mounted
// /dev/fuse fd, we need to drive the FUSE protocol on that fd directly
// instead of having libfuse mount its own. Plain read/writev syscalls
// suffice; splice is optional.
static ssize_t cte_custom_writev(int fd, struct iovec *iov, int count,
                                 void * /*userdata*/) {
  return writev(fd, iov, count);
}
static ssize_t cte_custom_read(int fd, void *buf, size_t buf_len,
                               void * /*userdata*/) {
  return read(fd, buf, buf_len);
}

int main(int argc, char *argv[]) {
  // Apptainer's --fusemount opens /dev/fuse on the host, performs the
  // kernel mount, and passes the fd to the FUSE binary as the last
  // argv ("/dev/fd/<N>"). libfuse 3's high-level argv parser doesn't
  // recognize this token (it's a libfuse2-era convention) and aborts
  // with "fuse: invalid argument '/dev/fd/N'". apptainer also strips
  // the mountpoint from the argv since it has already mounted, so
  // there's no fuse_main path that works.
  //
  // libfuse 3.14 added fuse_session_custom_io() which lets us drive
  // the protocol on an existing fd. When we detect /dev/fd/<N> as the
  // last argv, take the custom-io path; otherwise fall back to the
  // normal fuse_main mount-and-serve flow (host bare-metal use).
  int prefd = -1;
  int new_argc = argc;
  if (argc >= 2 && std::strncmp(argv[argc - 1], "/dev/fd/", 8) == 0) {
    prefd = std::atoi(argv[argc - 1] + 8);
    if (prefd < 0) {
      prefd = -1;
    } else {
      new_argc = argc - 1;  // drop /dev/fd/N from argv
    }
  }

  if (prefd == -1) {
    return fuse_main(argc, argv, &cte_fuse_ops, nullptr);
  }

  // Apptainer 1.2.5 (unprivileged, no starter-suid) doesn't actually
  // call mount(2) when --fusemount kicks in for a user-supplied
  // binary — it only opens /dev/fuse and hands us the fd. We have to
  // bind that fd to a mountpoint ourselves (this requires CAP_SYS_ADMIN
  // in the current user_ns, which we have via apptainer's userns
  // mapping). The mountpoint isn't communicated to us through argv or
  // env by apptainer, so the caller MUST set WRP_CTE_FUSE_MOUNTPOINT
  // before exec'ing the FUSE binary via --fusemount.
  const char *mountpoint = std::getenv("WRP_CTE_FUSE_MOUNTPOINT");
  if (mountpoint == nullptr) {
    std::fprintf(stderr,
                 "wrp_cte_fuse: got pre-opened fd %d but "
                 "WRP_CTE_FUSE_MOUNTPOINT env var is not set\n", prefd);
    return 1;
  }
  char mount_opts[256];
  std::snprintf(mount_opts, sizeof(mount_opts),
                "fd=%d,rootmode=040000,user_id=%u,group_id=%u",
                prefd, (unsigned)getuid(), (unsigned)getgid());
  if (mount("nodev", mountpoint, "fuse", MS_NODEV | MS_NOSUID,
            mount_opts) != 0) {
    std::fprintf(stderr, "wrp_cte_fuse: mount(\"%s\", fuse) failed: %s\n",
                 mountpoint, std::strerror(errno));
    return 1;
  }
  std::fprintf(stderr, "wrp_cte_fuse: mounted FUSE at %s with fd=%d\n",
               mountpoint, prefd);

  struct fuse_args args = FUSE_ARGS_INIT(new_argc, argv);
  struct fuse *fuse =
      fuse_new(&args, &cte_fuse_ops, sizeof(cte_fuse_ops), nullptr);
  if (!fuse) {
    fuse_opt_free_args(&args);
    return 1;
  }

  struct fuse_session *se = fuse_get_session(fuse);
  static const struct fuse_custom_io custom_io = {
      .writev = cte_custom_writev,
      .read = cte_custom_read,
      .splice_receive = nullptr,
      .splice_send = nullptr,
  };

  if (fuse_session_custom_io(se, &custom_io, prefd) != 0) {
    fuse_destroy(fuse);
    fuse_opt_free_args(&args);
    return 1;
  }

  // Single-threaded loop: simpler, sufficient for adapter testing.
  int ret = fuse_loop(fuse);
  fuse_destroy(fuse);
  fuse_opt_free_args(&args);
  return ret;
}
