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

#ifndef BDEV_CLIENT_H_
#define BDEV_CLIENT_H_

#include <chimaera/chimaera.h>

#include "bdev_tasks.h"

/**
 * Client API for bdev ChiMod
 *
 * Provides simple interface for block device operations with async I/O
 */

namespace chimaera::bdev {


class Client : public chi::ContainerClient {
 public:
  HSHM_CROSS_FUN Client() = default;
  HSHM_CROSS_FUN explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create bdev container - asynchronous
   * For file-based bdev, pool_name is the file path; for RAM, pool_name is a
   * unique identifier
   * @param custom_pool_id Explicit pool ID for the pool being created
   * @param perf_metrics Optional user-defined performance characteristics (uses defaults if not provided)
   */
  chi::Future<chimaera::bdev::CreateTask> AsyncCreate(
      const chi::PoolQuery& pool_query,
      const std::string& pool_name, const chi::PoolId& custom_pool_id,
      BdevType bdev_type, chi::u64 total_size = 0,
      chi::u32 io_depth = 32, chi::u32 alignment = 4096,
      const PerfMetrics* perf_metrics = nullptr) {
    auto* ipc_manager = CHI_IPC;

    // CreateTask should always use admin pool, never the client's pool_id_
    // Pass all arguments directly to NewTask constructor including CreateParams
    // arguments
    chi::u32 safe_alignment =
        (alignment == 0) ? 4096 : alignment;  // Ensure non-zero alignment

    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<chimaera::bdev::CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Send to admin pool for GetOrCreatePool processing
        pool_query,
        CreateParams::chimod_lib_name,  // chimod name from CreateParams
        pool_name,  // user-provided pool name (file path for files, unique name
                    // for RAM)
        custom_pool_id,   // target pool ID to create (explicit from user)
        this,             // Client pointer for PostWait
        // CreateParams arguments (perf_metrics is optional, defaults used if nullptr):
        bdev_type, total_size, io_depth, safe_alignment, perf_metrics);

    // Submit to runtime
    return ipc_manager->Send(task);
  }

  /**
   * Allocate data blocks - asynchronous
   * @param pool_query Pool query for routing
   * @param size Requested total size to allocate
   * @return Future for the allocation task
   */
  chi::Future<AllocateBlocksTask> AsyncAllocateBlocks(
      const chi::PoolQuery& pool_query,
      chi::u64 size) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<AllocateBlocksTask>(
        chi::CreateTaskId(), pool_id_, pool_query, size);

    return ipc_manager->Send(task);
  }

  /**
   * Free multiple blocks - asynchronous (host, std::vector)
   */
  chi::Future<chimaera::bdev::FreeBlocksTask> AsyncFreeBlocks(
      const chi::PoolQuery& pool_query,
      const std::vector<Block>& blocks) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<chimaera::bdev::FreeBlocksTask>(
        chi::CreateTaskId(), pool_id_, pool_query, blocks);

    return ipc_manager->Send(task);
  }

  /**
   * Free multiple blocks - asynchronous (priv::vector)
   */
  chi::Future<chimaera::bdev::FreeBlocksTask> AsyncFreeBlocks(
      const chi::PoolQuery& pool_query,
      const chi::priv::vector<Block>& blocks) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<chimaera::bdev::FreeBlocksTask>(
        chi::CreateTaskId(), pool_id_, pool_query, blocks);

    return ipc_manager->Send(task);
  }

  /**
   * Write data to blocks - asynchronous
   * @param pool_query Pool query for routing
   * @param blocks Blocks to write to
   * @param data ShmPtr to data buffer
   * @param length Size of data to write
   * @return Future for the write task
   */
  chi::Future<chimaera::bdev::WriteTask> AsyncWrite(
      const chi::PoolQuery& pool_query,
      const chi::priv::vector<Block>& blocks, hipc::ShmPtr<> data, size_t length) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<chimaera::bdev::WriteTask>(
        chi::CreateTaskId(), pool_id_, pool_query, blocks, data, length);

    return ipc_manager->Send(task);
  }

  /**
   * Read data from blocks - asynchronous
   * @param pool_query Pool query for routing
   * @param blocks Blocks to read from
   * @param data ShmPtr to output data buffer
   * @param buffer_size Size of the output buffer
   * @return Future for the read task
   */
  chi::Future<chimaera::bdev::ReadTask> AsyncRead(
      const chi::PoolQuery& pool_query,
      const chi::priv::vector<Block>& blocks, hipc::ShmPtr<> data,
      size_t buffer_size) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<chimaera::bdev::ReadTask>(
        chi::CreateTaskId(), pool_id_, pool_query, blocks, data, buffer_size);

    return ipc_manager->Send(task);
  }

  /**
   * Update GPU container with device/pinned memory pointers - asynchronous.
   * The CPU runtime fills in the actual pointers; callers pass zeros.
   * Primarily used to explicitly trigger GPU container initialization after
   * creating a kHbm or kPinned bdev pool.
   */
  chi::Future<UpdateTask> AsyncUpdate(const chi::PoolQuery &pool_query) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<UpdateTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        /*hbm_ptr=*/0, /*pinned_ptr=*/0,
        /*hbm_size=*/0, /*pinned_size=*/0,
        /*total_size=*/0, /*bdev_type=*/0, /*alignment=*/0);
    return ipc_manager->Send(task);
  }

  /**
   * Monitor container state - asynchronous
   */
  chi::Future<MonitorTask> AsyncMonitor(const chi::PoolQuery &pool_query,
                                        const std::string &query) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<MonitorTask>(
        chi::CreateTaskId(), pool_id_, pool_query, query);
    return ipc_manager->Send(task);
  }

  /**
   * Get performance statistics - asynchronous
   */
  chi::Future<chimaera::bdev::GetStatsTask> AsyncGetStats() {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<chimaera::bdev::GetStatsTask>(
        chi::CreateTaskId(), pool_id_, chi::PoolQuery());

    return ipc_manager->Send(task);
  }

};

}  // namespace chimaera::bdev

#endif  // BDEV_CLIENT_H_