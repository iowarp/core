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

#include <wrp_cae/core/core_runtime.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <wrp_cae/core/factory/assimilator_factory.h>
#ifdef WRP_CAE_ENABLE_HDF5
#include <hdf5.h>
#include <wrp_cae/core/factory/hdf5_file_assimilator.h>
#endif

#include "hermes_shm/data_structures/serialization/global_serialize.h"
#include <fstream>
#include <vector>

// Include wrp_cte headers before opening namespace to avoid Method namespace
// collision
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cae::core::Runtime)

namespace wrp_cae::core {

chi::TaskResume Runtime::Monitor(hipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &rctx) {
  CHI_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  (void)rctx;
  CHI_CO_RETURN;
  CHI_TASK_BODY_END
}

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CHI_TASK_BODY_BEGIN
  // Container is already initialized via Init() before Create is called
  // Do NOT call Init() here

  // Initialize CTE client using the CTE pool ID
  cte_client_ =
      std::make_shared<wrp_cte::core::Client>(wrp_cte::core::kCtePoolId);

  // Additional container-specific initialization logic here
  HLOG(kInfo, "Core container created and initialized for pool: {} (ID: {})",
       pool_name_, pool_id_);
  CHI_CO_RETURN;
  CHI_TASK_BODY_END
}

chi::u64 Runtime::GetWorkRemaining() const {
  // CAE doesn't currently track work remaining
  // Return 0 to indicate no pending work
  return 0;
}

chi::TaskResume Runtime::ParseOmni(hipc::FullPtr<ParseOmniTask> task,
                                   chi::RunContext& ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CHI_TASK_BODY_BEGIN
  HLOG(kInfo, "ParseOmni called with {} bytes of serialized data",
       task->serialized_ctx_.size());

  // Deserialize the vector of AssimilationCtx
  std::vector<AssimilationCtx> assimilation_contexts;
  try {
    std::string data = task->serialized_ctx_.str();
    std::vector<char> buf(data.begin(), data.end());
    hshm::ipc::GlobalDeserialize<std::vector<char>> ar(buf);
    ar(assimilation_contexts);
  } catch (const std::exception& e) {
    HLOG(kError, "ParseOmni: Failed to deserialize AssimilationCtx vector: {}",
         e.what());
    task->result_code_ = -1;
    task->error_message_ = e.what();
    task->num_tasks_scheduled_ = 0;
    CHI_CO_RETURN;
  }

  HLOG(kInfo, "ParseOmni: Processing {} assimilation contexts",
       assimilation_contexts.size());

  // Process each assimilation context
  chi::u32 tasks_scheduled = 0;
  AssimilatorFactory factory(cte_client_);

  for (size_t i = 0; i < assimilation_contexts.size(); ++i) {
    const auto& assimilation_ctx = assimilation_contexts[i];

    HLOG(kInfo,
         "ParseOmni: Processing context {}/{} - src: {}, dst: {}, format: {}",
         i + 1, assimilation_contexts.size(), assimilation_ctx.src,
         assimilation_ctx.dst, assimilation_ctx.format);

    // Get appropriate assimilator for this context
    auto assimilator = factory.Get(assimilation_ctx.src);

    if (!assimilator) {
      HLOG(kError, "ParseOmni: No assimilator found for source: {}",
           assimilation_ctx.src);
      task->result_code_ = -2;
      task->error_message_ =
          "No assimilator found for source: " + assimilation_ctx.src;
      task->num_tasks_scheduled_ = tasks_scheduled;
      CHI_CO_RETURN;
    }

    // Schedule the assimilation using co_await
    int result = 0;
    CHI_CO_AWAIT(assimilator->Schedule(assimilation_ctx, result));
    if (result != 0) {
      HLOG(
          kError,
          "ParseOmni: Assimilator failed for context {}/{} with error code: {}",
          i + 1, assimilation_contexts.size(), result);
      task->result_code_ = result;
      task->error_message_ = std::string("Assimilator failed");
      task->num_tasks_scheduled_ = tasks_scheduled;
      CHI_CO_RETURN;
    }

    tasks_scheduled++;
  }

  // Success
  task->result_code_ = 0;
  task->error_message_ = "";
  task->num_tasks_scheduled_ = tasks_scheduled;

  HLOG(kInfo, "ParseOmni: Successfully scheduled {} assimilations",
       tasks_scheduled);
  CHI_CO_RETURN;
  CHI_TASK_BODY_END
}

chi::TaskResume Runtime::ProcessHdf5Dataset(
    hipc::FullPtr<ProcessHdf5DatasetTask> task, chi::RunContext& ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CHI_TASK_BODY_BEGIN
#ifdef WRP_CAE_ENABLE_HDF5
  HLOG(kInfo, "ProcessHdf5Dataset: file='{}', dataset='{}', tag_prefix='{}'",
       task->file_path_.str(), task->dataset_path_.str(),
       task->tag_prefix_.str());

  // Open the HDF5 file
  hid_t file_id =
      H5Fopen(task->file_path_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    HLOG(kError, "ProcessHdf5Dataset: Failed to open HDF5 file: {}",
         task->file_path_.str());
    task->result_code_ = -1;
    task->error_message_ =
        chi::priv::string("Failed to open HDF5 file", HSHM_MALLOC);
    CHI_CO_RETURN;
  }

  // Create assimilator and process the dataset
  wrp_cae::core::Hdf5FileAssimilator assimilator(cte_client_);
  int result = 0;
  CHI_CO_AWAIT(assimilator.ProcessDataset(file_id, task->dataset_path_.str(),
                                      task->tag_prefix_.str(), result));

  // Close the HDF5 file
  H5Fclose(file_id);

  if (result != 0) {
    HLOG(kError,
         "ProcessHdf5Dataset: Failed to process dataset '{}' (error: {})",
         task->dataset_path_.str(), result);
    task->result_code_ = result;
    task->error_message_ =
        chi::priv::string("Dataset processing failed", HSHM_MALLOC);
  } else {
    HLOG(kInfo, "ProcessHdf5Dataset: Successfully processed dataset '{}'",
         task->dataset_path_.str());
    task->result_code_ = 0;
  }
#else
  task->result_code_ = -1;
  task->error_message_ =
      chi::priv::string("HDF5 support not compiled in", HSHM_MALLOC);
#endif
  CHI_CO_RETURN;
  CHI_TASK_BODY_END
}

chi::TaskResume Runtime::ExportData(hipc::FullPtr<ExportDataTask> task,
                                    chi::RunContext& ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CHI_TASK_BODY_BEGIN
  task->result_code_ = 0;
  task->bytes_exported_ = 0;

  const std::string tag_name = task->tag_name_.str();
  const std::string output_path = task->output_path_.str();
  const std::string format = task->format_.str();

  HLOG(kInfo, "ExportData: tag='{}', output='{}', format='{}'",
       tag_name, output_path, format);

  // Step 1: resolve the tag ID
  auto tag_future = cte_client_->AsyncGetOrCreateTag(tag_name);
  CHI_CO_AWAIT(tag_future);
  const auto &tag_id = tag_future->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "ExportData: tag '{}' not found", tag_name);
    task->result_code_ = -1;
    task->error_message_ = chi::priv::string("Tag not found", HSHM_MALLOC);
    CHI_CO_RETURN;
  }

  // Step 2: list all blobs in the tag
  auto blobs_future = cte_client_->AsyncGetContainedBlobs(tag_id);
  CHI_CO_AWAIT(blobs_future);
  const auto &blob_names = blobs_future->blob_names_;

  if (blob_names.empty()) {
    HLOG(kInfo, "ExportData: tag '{}' has no blobs", tag_name);
    CHI_CO_RETURN;
  }

  if (format == "hdf5") {
#ifdef WRP_CAE_ENABLE_HDF5
    hid_t file_id = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
      HLOG(kError, "ExportData: failed to create HDF5 file '{}'", output_path);
      task->result_code_ = -2;
      task->error_message_ =
          chi::priv::string("Failed to create HDF5 file", HSHM_MALLOC);
      CHI_CO_RETURN;
    }

    for (const auto &blob_name : blob_names) {
      // Get blob size
      auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
      CHI_CO_AWAIT(size_future);
      chi::u64 blob_size = size_future->size_;
      if (blob_size == 0) continue;

      // Allocate buffer and read blob
      auto *ipc_manager = CHI_IPC;
      hipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(blob_size);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed for blob '{}'", blob_name);
        continue;
      }
      hipc::ShmPtr<> shm_ptr(buf.shm_);
      auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                   blob_size, 0, shm_ptr);
      CHI_CO_AWAIT(get_future);

      if (get_future->GetReturnCode() == 0) {
        hsize_t dims[1] = {static_cast<hsize_t>(blob_size)};
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(file_id, blob_name.c_str(), H5T_NATIVE_UINT8,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds >= 0) {
          H5Dwrite(ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   buf.ptr_);
          H5Dclose(ds);
          task->bytes_exported_ += blob_size;
        }
        H5Sclose(space);
      }
      ipc_manager->FreeBuffer(buf);
    }

    H5Fclose(file_id);
    HLOG(kInfo, "ExportData: wrote {} bytes to HDF5 '{}'",
         task->bytes_exported_, output_path);
#else
    task->result_code_ = -3;
    task->error_message_ =
        chi::priv::string("HDF5 support not compiled in", HSHM_MALLOC);
#endif
  } else {
    // Binary format: sequential blob data with a simple header per blob
    std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "ExportData: failed to open '{}' for writing", output_path);
      task->result_code_ = -2;
      task->error_message_ =
          chi::priv::string("Failed to open output file", HSHM_MALLOC);
      CHI_CO_RETURN;
    }

    for (const auto &blob_name : blob_names) {
      auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
      CHI_CO_AWAIT(size_future);
      chi::u64 blob_size = size_future->size_;
      if (blob_size == 0) continue;

      auto *ipc_manager = CHI_IPC;
      hipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(blob_size);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed for blob '{}'", blob_name);
        continue;
      }
      hipc::ShmPtr<> shm_ptr(buf.shm_);
      auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                   blob_size, 0, shm_ptr);
      CHI_CO_AWAIT(get_future);

      if (get_future->GetReturnCode() == 0) {
        // Header: name length (u32) + name + data length (u64) + data
        uint32_t name_len = static_cast<uint32_t>(blob_name.size());
        ofs.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
        ofs.write(blob_name.data(), name_len);
        ofs.write(reinterpret_cast<const char *>(&blob_size), sizeof(blob_size));
        ofs.write(buf.ptr_, static_cast<std::streamsize>(blob_size));
        task->bytes_exported_ += blob_size;
      }
      ipc_manager->FreeBuffer(buf);
    }

    HLOG(kInfo, "ExportData: wrote {} bytes to binary '{}'",
         task->bytes_exported_, output_path);
  }

  CHI_CO_RETURN;
  CHI_TASK_BODY_END
}

}  // namespace wrp_cae::core
