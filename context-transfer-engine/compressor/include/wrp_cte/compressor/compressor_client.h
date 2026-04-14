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

#ifndef WRP_CTE_COMPRESSOR_CLIENT_H_
#define WRP_CTE_COMPRESSOR_CLIENT_H_

#include <wrp_cte/core/core_client.h>
#include <wrp_cte/compressor/compressor_tasks.h>

namespace wrp_cte::compressor {

/**
 * Transparent compression client.
 *
 * Inherits the full CTE core client API so that user code can swap
 * between wrp_cte::core::Client and wrp_cte::compressor::Client
 * without any changes. AsyncPutBlob is intercepted and routed through
 * the compressor chimod (DynamicSchedule), and AsyncGetBlob is routed
 * through decompression. All other methods (target management, tag
 * management, metadata, etc.) pass through to the CTE core directly.
 */
class Client : public wrp_cte::core::Client {
 public:
  Client() = default;

  /**
   * Construct a compressor client (runtime-internal use).
   * Only the compressor pool is known; core pool is resolved per-task.
   */
  explicit Client(const chi::PoolId &compressor_pool_id)
      : compressor_pool_id_(compressor_pool_id) {
    wrp_cte::core::Client::Init(compressor_pool_id);
  }

  /**
   * Construct a compressor client (user-facing).
   * @param compressor_pool_id Pool ID of the compressor chimod
   * @param core_pool_id Pool ID of the CTE core chimod
   */
  Client(const chi::PoolId &compressor_pool_id,
         const chi::PoolId &core_pool_id)
      : compressor_pool_id_(compressor_pool_id) {
    wrp_cte::core::Client::Init(core_pool_id);
  }

  /**
   * Create the compressor container.
   */
  chi::Future<CreateTask> AsyncCreateCompressor(
      const chi::PoolQuery &pool_query, const std::string &pool_name,
      const chi::PoolId &custom_pool_id) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, pool_query,
        "wrp_cte_compressor", pool_name, custom_pool_id, this);
    return ipc_manager->Send(task);
  }

  // ------------------------------------------------------------------
  // Overridden data-path methods: route through compressor
  // ------------------------------------------------------------------

  /**
   * AsyncPutBlob override: routes data through compression before storage.
   * The compressor runtime will analyze, compress, then call core PutBlob.
   */
  chi::Future<DynamicScheduleTask> AsyncPutBlob(
      const wrp_cte::core::TagId &tag_id, const char *blob_name,
      chi::u64 offset, chi::u64 size, hipc::ShmPtr<> blob_data,
      float score = -1.0f,
      const wrp_cte::core::Context &context = wrp_cte::core::Context(),
      chi::u32 flags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;
    // core_pool_id is a fallback — the runtime uses next_pool_id from
    // compose config when available.
    auto task = ipc_manager->NewTask<DynamicScheduleTask>(
        chi::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags,
        chi::PoolId::GetNull());
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  chi::Future<DynamicScheduleTask> AsyncPutBlob(
      const wrp_cte::core::TagId &tag_id, const std::string &blob_name,
      chi::u64 offset, chi::u64 size, hipc::ShmPtr<> blob_data,
      float score = -1.0f,
      const wrp_cte::core::Context &context = wrp_cte::core::Context(),
      chi::u32 flags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    return AsyncPutBlob(tag_id, blob_name.c_str(), offset, size, blob_data,
                        score, context, flags, pool_query);
  }

  /**
   * AsyncGetBlob override: retrieves and decompresses data transparently.
   * The compressor runtime will call core GetBlob then decompress.
   */
  chi::Future<DecompressTask> AsyncGetBlob(
      const wrp_cte::core::TagId &tag_id, const char *blob_name,
      chi::u64 offset, chi::u64 size, chi::u32 flags,
      hipc::ShmPtr<> blob_data,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<DecompressTask>(
        chi::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data,
        chi::PoolId::GetNull());
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  chi::Future<DecompressTask> AsyncGetBlob(
      const wrp_cte::core::TagId &tag_id, const std::string &blob_name,
      chi::u64 offset, chi::u64 size, chi::u32 flags,
      hipc::ShmPtr<> blob_data,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    return AsyncGetBlob(tag_id, blob_name.c_str(), offset, size, flags,
                        blob_data, pool_query);
  }

  // ------------------------------------------------------------------
  // Compressor-specific methods (used internally by compressor runtime)
  // ------------------------------------------------------------------

  /**
   * Explicit dynamic scheduling — analyzes data, picks best compressor,
   * compresses, and stores via core PutBlob.
   */
  chi::Future<DynamicScheduleTask> AsyncDynamicSchedule(
      const chi::PoolQuery &pool_query,
      const wrp_cte::core::TagId &tag_id, const std::string &blob_name,
      chi::u64 offset, chi::u64 size, hipc::ShmPtr<> blob_data,
      float score, const wrp_cte::core::Context &context,
      chi::u32 flags, const chi::PoolId &core_pool_id) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<DynamicScheduleTask>(
        chi::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags,
        core_pool_id);
    return ipc_manager->Send(task);
  }

  /**
   * Explicit compression — compresses data and stores via core PutBlob.
   */
  chi::Future<CompressTask> AsyncCompress(
      const chi::PoolQuery &pool_query,
      const wrp_cte::core::TagId &tag_id, const std::string &blob_name,
      chi::u64 offset, chi::u64 size, hipc::ShmPtr<> blob_data,
      float score, const wrp_cte::core::Context &context,
      chi::u32 flags, const chi::PoolId &core_pool_id) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CompressTask>(
        chi::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags,
        core_pool_id);
    return ipc_manager->Send(task);
  }

  /**
   * Explicit decompression — retrieves via core GetBlob and decompresses.
   */
  chi::Future<DecompressTask> AsyncDecompressExplicit(
      const chi::PoolQuery &pool_query,
      const wrp_cte::core::TagId &tag_id, const std::string &blob_name,
      chi::u64 offset, chi::u64 size, chi::u32 flags,
      hipc::ShmPtr<> blob_data, const chi::PoolId &core_pool_id) {
    auto *ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<DecompressTask>(
        chi::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data, core_pool_id);
    return ipc_manager->Send(task);
  }

 private:
  chi::PoolId compressor_pool_id_;
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_CLIENT_H_
