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

#ifndef HSHM_SYSINFO_INFO_H_
#define HSHM_SYSINFO_INFO_H_

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
#include <sys/sysinfo.h>
#endif
#include <unistd.h>
#include <sys/types.h>
#endif

#ifdef _WIN32
#include <BaseTsd.h>
#include <sys/types.h>
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
using ssize_t = SSIZE_T;
#endif
using pid_t = int;
using mode_t = unsigned int;
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/util/formatter.h"
#include "hermes_shm/util/singleton.h"

#define HSHM_SYSTEM_INFO \
  hshm::LockfreeCrossSingleton<hshm::SystemInfo>::GetInstance()
#define HSHM_SYSTEM_INFO_T hshm::SystemInfo *

namespace hshm {

/** Dynamically load shared libraries */
struct SharedLibrary {
  void *handle_;

  SharedLibrary() = default;
  SharedLibrary(const std::string &name);
  ~SharedLibrary();

  // Delete copy operations
  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  // Move operations
  SharedLibrary(SharedLibrary &&other) noexcept;
  SharedLibrary &operator=(SharedLibrary &&other) noexcept;

  void Load(const std::string &name);
  void *GetSymbol(const std::string &name);
  std::string GetError() const;

  bool IsNull() { return handle_ == nullptr; }
};

/** File wrapper */
union File {
  int posix_fd_;
  HANDLE windows_fd_;
};

/** Handle to a spawned child process */
struct ProcessHandle {
#ifdef _WIN32
  HANDLE hProcess;
  HANDLE hThread;
  DWORD pid;
#else
  pid_t pid;
#endif
};

/** A unification of certain OS system calls */
class SystemInfo {
 public:
  int pid_;
  int ncpu_;
  int page_size_;
  int uid_;
  int gid_;
  size_t ram_size_;
#if HSHM_IS_HOST
  std::vector<size_t> cur_cpu_freq_;
#endif

 public:
  HSHM_CROSS_FUN
  SystemInfo() { RefreshInfo(); }

  HSHM_CROSS_FUN
  void RefreshInfo() {
#if HSHM_IS_HOST
    pid_ = GetPid();
    ncpu_ = GetCpuCount();
    page_size_ = GetPageSize();
    uid_ = GetUid();
    gid_ = GetGid();
    ram_size_ = GetRamCapacity();
    cur_cpu_freq_.resize(ncpu_);
    RefreshCpuFreqKhz();
#endif
  }

  void RefreshCpuFreqKhz();

  size_t GetCpuFreqKhz(int cpu);

  size_t GetCpuMaxFreqKhz(int cpu);

  size_t GetCpuMinFreqKhz(int cpu);

  size_t GetCpuMinFreqMhz(int cpu);

  size_t GetCpuMaxFreqMhz(int cpu);

  void SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz);

  void SetCpuFreqKhz(int cpu, size_t cpu_freq_khz);

  void SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz);

  void SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz);

  static int GetCpuCount();

  static int GetPageSize();

  static int GetTid();

  static int GetPid();

  static int GetUid();

  static int GetGid();

  static size_t GetRamCapacity();

  static void YieldThread();

  static bool CreateTls(ThreadLocalKey &key, void *data);

  static bool SetTls(const ThreadLocalKey &key, void *data);

  static void *GetTls(const ThreadLocalKey &key);

  static bool CreateNewSharedMemory(File &fd, const std::string &name,
                                             size_t size);

  static bool OpenSharedMemory(File &fd, const std::string &name);

  static void CloseSharedMemory(File &file);

  static void DestroySharedMemory(const std::string &name);

  static void *MapPrivateMemory(size_t size);

  static void *MapSharedMemory(const File &fd, size_t size, i64 off);

  /**
   * Map a contiguous memory region with mixed private/shared mapping
   *
   * Creates a contiguous virtual memory region where:
   * - First private_size bytes: MAP_PRIVATE | MAP_ANONYMOUS (process-local)
   * - Remaining shared_size bytes: MAP_SHARED from fd (inter-process)
   *
   * @param fd File descriptor for shared memory (from shm_open)
   * @param private_size Size of private region at the beginning
   * @param shared_size Size of shared region following the private region
   * @param shared_offset Offset into fd for the shared mapping (usually 0)
   * @return Pointer to the beginning of the contiguous region (private region start),
   *         or nullptr on failure
   *
   * Note: The entire region must be unmapped with a single UnmapMemory call
   *       using total_size = private_size + shared_size
   */
  static void *MapMixedMemory(const File &fd, size_t private_size,
                                        size_t shared_size, i64 shared_offset = 0);

  static void UnmapMemory(void *ptr, size_t size);

  static void *AlignedAlloc(size_t alignment, size_t size);

  static std::string Getenv(
      const char *name, size_t max_size = hshm::Unit<size_t>::Megabytes(1));

  static std::string Getenv(
      const std::string &name,
      size_t max_size = hshm::Unit<size_t>::Megabytes(1)) {
    return Getenv(name.c_str(), max_size);
  }

  static void Setenv(const char *name, const std::string &value,
                              int overwrite);

  static void Unsetenv(const char *name);

  /** Get the per-user chimaera tmp directory path (/tmp/chimaera_$USER) */
  HSHM_DLL static std::string GetMemfdDir();

  /** Get the full path for a named file in the memfd directory */
  HSHM_DLL static std::string GetMemfdPath(const std::string &name);

  /** Ensure the per-user memfd directory exists */
  HSHM_DLL static void EnsureMemfdDir();

  static bool IsProcessAlive(int pid);

  static std::string GetModuleDirectory();

  static std::string GetLibrarySearchPathVar();

  static char GetPathListSeparator();

  static std::string GetSharedLibExtension();

  static ProcessHandle SpawnProcess(
      const std::string &exe_path,
      const std::vector<std::string> &args,
      const std::vector<std::pair<std::string, std::string>> &env = {});

  static void KillProcess(ProcessHandle &proc);

  static int WaitProcess(ProcessHandle &proc);

  static std::string GetSelfExePath();

  /**
   * Suppress OS error/crash dialog popups.
   * On Windows: redirects CRT asserts/errors to stderr and disables
   * Windows Error Reporting dialogs so tests don't block on popups.
   * No-op on other platforms.
   */
  static void SuppressErrorDialogs();
};

}  // namespace hshm

#undef WIN32_LEAN_AND_MEAN

#endif  // HSHM_SYSINFO_INFO_H_
