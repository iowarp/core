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

#ifndef WRP_CTE_ADAPTER_FILESYSTEM_FILESYSTEM_IO_CLIENT_H_
#define WRP_CTE_ADAPTER_FILESYSTEM_FILESYSTEM_IO_CLIENT_H_

#include <mpi.h>

#include <filesystem>
#include <future>
#include <limits>

#include "adapter/adapter_types.h"
#include "adapter/mapper/balanced_mapper.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/types/bitfield.h"
#include "wrp_cte/core/core_client.h"
#include "wrp_cte/core/core_tasks.h"

namespace stdfs = std::filesystem;

namespace wrp::cae {

/** Put or get data directly from I/O client */
#define WRP_CTE_IO_CLIENT_BYPASS BIT_OPT(uint32_t, 0)
/** Only put or get data from a Hermes buffer; no fallback to I/O client */
#define WRP_CTE_IO_CLIENT_NO_FALLBACK BIT_OPT(uint32_t, 1)
/** Whether to perform seek */
#define WRP_CTE_FS_SEEK BIT_OPT(uint32_t, 2)
/** Whether to perform create */
#define WRP_CTE_FS_CREATE BIT_OPT(uint32_t, 3)
/** Whether in append mode */
#define WRP_CTE_FS_APPEND BIT_OPT(uint32_t, 4)
/** Whether to perform truncate */
#define WRP_CTE_FS_TRUNC BIT_OPT(uint32_t, 5)
/** Whether the file was found on-disk */
#define WRP_CTE_FS_EXISTS BIT_OPT(uint32_t, 6)
/** Whether the file supports reading */
#define WRP_CTE_FS_READ BIT_OPT(uint32_t, 7)
/** Whether the file supports writing */
#define WRP_CTE_FS_WRITE BIT_OPT(uint32_t, 8)

/** A structure to represent IO status */
struct IoStatus {
  size_t size_;                /**< POSIX/STDIO return value */
  int mpi_ret_;                /**< MPI return value */
  MPI_Status mpi_status_;      /**< MPI status */
  MPI_Status* mpi_status_ptr_; /**< MPI status pointer */
  bool success_;               /**< Whether the I/O succeeded */

  /** Default constructor */
  IoStatus()
      : size_(0),
        mpi_ret_(MPI_SUCCESS),
        mpi_status_ptr_(&mpi_status_),
        success_(true) {}

  /** Copy constructor */
  void Copy(const IoStatus& other) {
    size_ = other.size_;
    mpi_ret_ = other.mpi_ret_;
    mpi_status_ = other.mpi_status_;
    if (other.mpi_status_ptr_ == &other.mpi_status_) {
      mpi_status_ptr_ = &mpi_status_;
    } else {
      mpi_status_ptr_ = other.mpi_status_ptr_;
    }
    success_ = other.success_;
  }
};

/**
 * A structure to represent IO options for FS adapter.
 * For now, nothing additional than the typical FsIoOptions.
 * */
struct FsIoOptions {
  hshm::bitfield32_t flags_; /**< various I/O flags */
  MPI_Datatype mpi_type_;    /**< MPI data type */
  int mpi_count_;            /**< The number of types */
  int type_size_;            /**< The size of type */
  size_t backend_off_;       /**< Offset in the backend to begin I/O */
  size_t backend_size_;      /**< Size of I/O to perform at backend */

  /** Default constructor */
  FsIoOptions()
      : flags_(),
        mpi_type_(MPI_CHAR),
        mpi_count_(0),
        type_size_(1),
        backend_off_(0),
        backend_size_(0) {
    SetSeek();
  }

  /** Enable seek for this I/O */
  void SetSeek() { flags_.SetBits(WRP_CTE_FS_SEEK); }

  /** Disable seek for this I/O */
  void UnsetSeek() { flags_.UnsetBits(WRP_CTE_FS_SEEK); }

  /** Whether or not to perform seek in FS adapter */
  bool DoSeek() const { return flags_.Any(WRP_CTE_FS_SEEK); }

  /** Marks the file as truncated */
  void MarkTruncated() { flags_.SetBits(WRP_CTE_FS_TRUNC); }

  /** Whether a file is marked truncated */
  bool IsTruncated() const { return flags_.Any(WRP_CTE_FS_TRUNC); }

  /** return IO options with \a mpi_type MPI data type */
  static FsIoOptions DataType(MPI_Datatype mpi_type, bool seek = true) {
    FsIoOptions opts;
    opts.mpi_type_ = mpi_type;
    if (!seek) {
      opts.UnsetSeek();
    }
    return opts;
  }
};

/** The get task */
struct GetBlobAsyncTask {
  chi::Future<wrp_cte::core::GetBlobTask> future_;
  char* orig_data_;
  size_t orig_size_;
};

/** A structure to represent Hermes request */
struct FsAsyncTask {
  std::vector<chi::Future<wrp_cte::core::PutBlobTask>> put_futures_;
  std::vector<GetBlobAsyncTask> get_tasks_;
  IoStatus io_status_;
  FsIoOptions opts_;
};

/** Represents an object in the I/O client (e.g., a file) */
struct File {
  AdapterType type_;     /**< Client to forward I/O request to */
  std::string filename_; /**< Filename to read from */

  int hermes_fd_;          /**< fake file descriptor (SCRATCH MODE) */
  FILE* hermes_fh_;        /**< fake file handler (SCRATCH MODE) */
  MPI_File hermes_mpi_fh_; /**< fake MPI file handler (SCRATCH MODE) */

  bool status_;    /**< status */
  int mpi_status_; /**< MPI status */

  /** Default constructor */
  File()
      : type_(AdapterType::kNone),
        filename_(),
        hermes_fd_(-1),
        hermes_fh_(nullptr),
        hermes_mpi_fh_(nullptr),
        status_(true),
        mpi_status_(MPI_SUCCESS) {}

  /** file constructor that copies \a old file */
  File(const File& old) { Copy(old); }

  /** file assignment operator that copies \a old file */
  File& operator=(const File& old) {
    Copy(old);
    return *this;
  }

  /** copy \a old file */
  void Copy(const File& old) {
    filename_ = old.filename_;
    hermes_fd_ = old.hermes_fd_;
    hermes_fh_ = old.hermes_fh_;
    hermes_mpi_fh_ = old.hermes_mpi_fh_;
    status_ = old.status_;
  }

  /** file comparison operator */
  bool operator==(const File& other) const {
    return (hermes_fd_ == other.hermes_fd_) &&
           (hermes_fh_ == other.hermes_fh_) &&
           (hermes_mpi_fh_ == other.hermes_mpi_fh_);
  }

  /** return hash value of this class  */
  std::size_t hash() const {
    std::size_t result;
    std::size_t h1 = std::hash<int>{}(hermes_fd_);
    std::size_t h2 = std::hash<void*>{}(hermes_fh_);
    std::size_t h3 = std::hash<void*>{}(hermes_mpi_fh_);
    result = h1 ^ h2 ^ h3;
    return result;
  }
};

/** Any relevant statistics from the I/O client */
struct AdapterStat {
  std::string path_;          /**< The URL of this file */
  int flags_;                 /**< open() flags for POSIX */
  hshm::bitfield32_t hflags_; /**< Flags used by FS adapter */
  mode_t st_mode_;            /**< protection */
  uid_t st_uid_;              /**< user ID of owner */
  gid_t st_gid_;              /**< group ID of owner */
  size_t st_ptr_;             /**< current ptr of FILE */
  size_t file_size_;          /**< Size of file at backend at time of open */
  timespec st_atim_;          /**< time of last access */
  timespec st_mtim_;          /**< time of last modification */
  timespec st_ctim_;          /**< time of last status change */
  std::string mode_str_;      /**< mode used for fopen() */
  AdapterMode adapter_mode_;  /**< Mode used for adapter */

  int fd_;          /**< real file descriptor */
  FILE* fh_;        /**< real STDIO file handler */
  MPI_File mpi_fh_; /**< real MPI file handler */

  int amode_;      /**< access mode (MPI) */
  MPI_Info info_;  /**< Info object (handle) */
  MPI_Comm comm_;  /**< Communicator for the file.*/
  bool atomicity_; /**< Consistency semantics for data-access */

  wrp_cte::core::TagId tag_id_; /**< tag associated with the file */
  /** Page size used for file */
  size_t page_size_;

  /** Default constructor */
  AdapterStat()
      : flags_(0),
        hflags_(),
        st_mode_(),
        st_ptr_(0),
        file_size_(0),
        st_atim_(),
        st_mtim_(),
        st_ctim_(),
        adapter_mode_(AdapterMode::kNone),
        fd_(-1),
        fh_(nullptr),
        mpi_fh_(nullptr),
        amode_(0),
        comm_(MPI_COMM_SELF),
        atomicity_(false) {}

  /** Update to the current time */
  void UpdateTime() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    st_mtim_ = ts;
    st_ctim_ = ts;
  }

  /** compare \a a BLOB and \a b BLOB.*/
  static bool CompareBlobs(const std::string& a, const std::string& b) {
    return std::stol(a) < std::stol(b);
  }
};

/**
 * Metadta required by Filesystem I/O clients to perform a HermesOpen
 * */
struct FsIoClientMetadata {
  int hermes_fd_min_, hermes_fd_max_; /**< Min and max fd values (inclusive)*/
  std::atomic<int> hermes_fd_cur_;    /**< Current fd */

  /** Default constructor */
  FsIoClientMetadata() {
    hermes_fd_min_ = 8192;  // TODO(llogan): don't assume 8192
    hermes_fd_cur_ = hermes_fd_min_;
    hermes_fd_max_ = std::numeric_limits<int>::max();
  }

  /** Allocate a Hermes FD */
  int AllocateFd() {
    int cur = hermes_fd_cur_.fetch_add(1);
    return cur;
  }

  /** Release a Hermes FD */
  void ReleaseFd(int hermes_fd) {
    // TODO(llogan): recycle instead of ignore
    (void)hermes_fd;
  }
};

/**
 * State required by Filesystem I/O clients to perform a HermesOpen
 * */
struct FilesystemIoClientState {
  /**
   * A pointer to the FsIoClientMetadata stored in the Filesystem
   * */
  FsIoClientMetadata* mdm_;

  /**
   * A pointer to the Adapter Stat object. Used by STDIO + MPI-IO to
   * represent the hermes_fh_ and hermes_mpi_fh_ fields.
   * */
  void* stat_;

  /** Default constructor */
  FilesystemIoClientState(FsIoClientMetadata* mdm, void* stat)
      : mdm_(mdm), stat_(stat) {}
};

/**
 * Defines I/O clients which are compatible with the filesystem
 * base class.
 * */
class FilesystemIoClient {
 public:
  /** virtual destructor */
  virtual ~FilesystemIoClient() = default;

  /** Get initial statistics from the backend */
  virtual size_t GetBackendSize(const std::string& bkt_name) = 0;

  /** Write blob to backend */
  virtual void WriteBlob(const std::string& bkt_name, const void* data,
                         size_t size, const FsIoOptions& opts,
                         IoStatus& status) = 0;

  /** Read blob from the backend */
  virtual void ReadBlob(const std::string& bkt_name, void* data, size_t size,
                        const FsIoOptions& opts, IoStatus& status) = 0;

  /** real open */
  virtual void RealOpen(File& f, AdapterStat& stat,
                        const std::string& path) = 0;

  /**
   * Called after real open. Allocates the Hermes representation of
   * identifying file information, such as a hermes file descriptor
   * and hermes file handler. These are not the same as POSIX file
   * descriptor and STDIO file handler.
   * */
  virtual void HermesOpen(File& f, const AdapterStat& stat,
                          FilesystemIoClientState& fs_mdm) = 0;

  /** real sync */
  virtual int RealSync(const File& f, const AdapterStat& stat) = 0;

  /** real close */
  virtual int RealClose(const File& f, AdapterStat& stat) = 0;

  /** real remove */
  virtual int RealRemove(const std::string& path) = 0;

  /**
   * Called before RealClose. Releases information provisioned during
   * the allocation phase.
   * */
  virtual void HermesClose(File& f, const AdapterStat& stat,
                           FilesystemIoClientState& fs_mdm) = 0;

  /** Updates I/O status after read/write operations */
  virtual void UpdateIoStatus(const FsIoOptions& opts, IoStatus& status) = 0;
};

}  // namespace wrp::cae

namespace std {
/** A structure to represent hash */
template <>
struct hash<::wrp::cae::File> {
  /** hash creator functor */
  std::size_t operator()(const wrp::cae::File& key) const { return key.hash(); }
};
}  // namespace std

#endif  // WRP_CTE_ADAPTER_FILESYSTEM_FILESYSTEM_IO_CLIENT_H_
