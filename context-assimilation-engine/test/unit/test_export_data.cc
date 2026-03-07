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

/**
 * test_export_data.cc
 *
 * Unit tests for the CAE ExportData feature.
 *
 * Coverage targets (>95% of ExportData implementation):
 *   - ExportDataTask struct: default ctor, emplace ctor, Copy, Aggregate,
 *     SerializeIn, SerializeOut
 *   - Runtime::ExportData: empty-tag path, binary success, binary bad path,
 *     HDF5 success (if enabled) or HDF5-not-compiled path, HDF5 bad path
 *   - autogen core_lib_exec.cc: kExportData cases in Run, SaveTask, LoadTask,
 *     LocalLoadTask, LocalSaveTask, NewCopyTask, NewTask, Aggregate, DelTask
 *     (exercised implicitly via AsyncExportData)
 */

#include "simple_test.h"

#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/core_tasks.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

#include <cereal/archives/binary.hpp>
#include <fstream>
#include <sstream>
#include <thread>
#include <cstring>
#include <vector>
#include <cstdio>

#ifdef WRP_CAE_ENABLE_HDF5
#include <hdf5.h>
#endif

using namespace wrp_cae::core;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class ExportDataFixture {
 public:
  static inline bool g_initialized = false;

  ExportDataFixture() {
    if (g_initialized) return;

    // Step 1: Chimaera client init
    bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!ok) throw std::runtime_error("CHIMAERA_INIT failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Step 2: CTE client + pool
    ok = wrp_cte::core::WRP_CTE_CLIENT_INIT();
    if (!ok) throw std::runtime_error("WRP_CTE_CLIENT_INIT failed");
    auto *cte = WRP_CTE_CLIENT;
    cte->Init(wrp_cte::core::kCtePoolId);

    wrp_cte::core::CreateParams cte_params;
    auto cte_fut = cte->AsyncCreate(chi::PoolQuery::Dynamic(),
                                    wrp_cte::core::kCtePoolName,
                                    wrp_cte::core::kCtePoolId, cte_params);
    cte_fut.Wait();

    // Step 3: CAE client + pool
    WRP_CAE_CLIENT_INIT();
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams cae_params;
    auto cae_fut = cae_client.AsyncCreate(chi::PoolQuery::Local(),
                                          "test_cae_pool",
                                          wrp_cae::core::kCaePoolId, cae_params);
    cae_fut.Wait();

    g_initialized = true;
  }

  /**
   * Put a blob with known data into CTE and return the tag ID.
   * Caller must ensure chimaera is initialised (fixture ctor handles this).
   */
  wrp_cte::core::TagId PutBlob(const std::string &tag_name,
                                const std::string &blob_name,
                                const std::vector<uint8_t> &data) {
    auto *cte = WRP_CTE_CLIENT;

    auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
    tag_fut.Wait();
    auto tag_id = tag_fut->tag_id_;

    auto buf = CHI_IPC->AllocateBuffer(data.size());
    std::memcpy(buf.ptr_, data.data(), data.size());
    hipc::ShmPtr<> shm_ptr = buf.shm_.template Cast<void>();

    auto put_fut = cte->AsyncPutBlob(tag_id, blob_name, 0,
                                     static_cast<chi::u64>(data.size()),
                                     shm_ptr);
    put_fut.Wait();
    CHI_IPC->FreeBuffer(buf);

    return tag_id;
  }
};

// ---------------------------------------------------------------------------
// ExportDataTask struct tests
// (require runtime for HSHM_MALLOC, so all use ExportDataFixture)
// ---------------------------------------------------------------------------

TEST_CASE("ExportData - Task default constructor", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CHI_IPC;
  auto task = ipc->NewTask<ExportDataTask>();
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->bytes_exported_ == 0);
  REQUIRE(task->tag_name_.str() == "");
  REQUIRE(task->output_path_.str() == "");
  REQUIRE(task->format_.str() == "");
  ipc->DelTask(task);

  INFO("ExportDataTask default constructor OK");
}

TEST_CASE("ExportData - Task emplace constructor", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CHI_IPC;
  chi::TaskId tid = chi::CreateTaskId();
  auto task = ipc->NewTask<ExportDataTask>(tid, wrp_cae::core::kCaePoolId,
                                           chi::PoolQuery::Local(),
                                           "my_tag", "/tmp/out.bin", "binary");

  REQUIRE(task->tag_name_.str() == "my_tag");
  REQUIRE(task->output_path_.str() == "/tmp/out.bin");
  REQUIRE(task->format_.str() == "binary");
  REQUIRE(task->result_code_ == 0);
  REQUIRE(task->bytes_exported_ == 0);
  REQUIRE(task->method_ == Method::kExportData);

  ipc->DelTask(task);
  INFO("ExportDataTask emplace constructor OK");
}

TEST_CASE("ExportData - Task Copy", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CHI_IPC;
  auto src = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                          wrp_cae::core::kCaePoolId,
                                          chi::PoolQuery::Local(),
                                          "tag_copy", "/tmp/copy.bin", "binary");
  src->bytes_exported_ = 42;
  src->result_code_ = 7;
  src->error_message_ = chi::priv::string("copy_err", HSHM_MALLOC);

  auto dst = ipc->NewTask<ExportDataTask>();
  dst->Copy(src);

  REQUIRE(dst->tag_name_.str() == "tag_copy");
  REQUIRE(dst->output_path_.str() == "/tmp/copy.bin");
  REQUIRE(dst->format_.str() == "binary");
  REQUIRE(dst->bytes_exported_ == 42);
  REQUIRE(dst->result_code_ == 7);
  REQUIRE(dst->error_message_.str() == "copy_err");

  ipc->DelTask(src);
  ipc->DelTask(dst);
  INFO("ExportDataTask Copy OK");
}

TEST_CASE("ExportData - Task Aggregate", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CHI_IPC;
  auto orig = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                           wrp_cae::core::kCaePoolId,
                                           chi::PoolQuery::Local(),
                                           "tag_agg", "/tmp/agg.bin", "binary");
  orig->bytes_exported_ = 100;
  orig->result_code_ = 1;

  auto replica = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                              wrp_cae::core::kCaePoolId,
                                              chi::PoolQuery::Local(),
                                              "tag_agg", "/tmp/agg.bin", "binary");
  replica->bytes_exported_ = 200;
  replica->result_code_ = 5;

  orig->Aggregate(replica.template Cast<chi::Task>());

  // Aggregate calls Copy, so orig should now have replica's values
  REQUIRE(orig->bytes_exported_ == 200);
  REQUIRE(orig->result_code_ == 5);

  ipc->DelTask(orig);
  ipc->DelTask(replica);
  INFO("ExportDataTask Aggregate OK");
}

TEST_CASE("ExportData - Task SerializeIn roundtrip", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CHI_IPC;
  auto task = ipc->NewTask<ExportDataTask>(chi::CreateTaskId(),
                                           wrp_cae::core::kCaePoolId,
                                           chi::PoolQuery::Local(),
                                           "ser_tag", "/tmp/ser.bin", "binary");

  // Write IN fields (tag_name_, output_path_, format_)
  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oa(ss);
    task->SerializeIn(oa);
  }

  // Read them back into a fresh task
  auto t2 = ipc->NewTask<ExportDataTask>();
  {
    cereal::BinaryInputArchive ia(ss);
    t2->SerializeIn(ia);
  }

  REQUIRE(t2->tag_name_.str() == "ser_tag");
  REQUIRE(t2->output_path_.str() == "/tmp/ser.bin");
  REQUIRE(t2->format_.str() == "binary");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ExportDataTask SerializeIn roundtrip OK");
}

TEST_CASE("ExportData - Task SerializeOut roundtrip", "[cae][export][task]") {
  ExportDataFixture f;

  auto *ipc = CHI_IPC;
  auto task = ipc->NewTask<ExportDataTask>();
  task->result_code_ = 3;
  task->bytes_exported_ = 777;
  task->error_message_ = chi::priv::string("err_msg", HSHM_MALLOC);

  // Write OUT fields (result_code_, error_message_, bytes_exported_)
  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oa(ss);
    task->SerializeOut(oa);
  }

  auto t2 = ipc->NewTask<ExportDataTask>();
  {
    cereal::BinaryInputArchive ia(ss);
    t2->SerializeOut(ia);
  }

  REQUIRE(t2->result_code_ == 3);
  REQUIRE(t2->bytes_exported_ == 777);
  REQUIRE(t2->error_message_.str() == "err_msg");

  ipc->DelTask(task);
  ipc->DelTask(t2);
  INFO("ExportDataTask SerializeOut roundtrip OK");
}

// ---------------------------------------------------------------------------
// Runtime integration tests — exercise Runtime::ExportData code paths
// ---------------------------------------------------------------------------

TEST_CASE("ExportData - Empty tag returns success with 0 bytes",
          "[cae][export][runtime]") {
  ExportDataFixture f;

  // Tag does not exist yet; GetOrCreateTag will create it with no blobs.
  wrp_cae::core::Client cae(wrp_cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData("export_empty_tag_xyz_001",
                                 "/tmp/cae_export_empty.bin", "binary");
  fut.Wait();

  // blob_names will be empty → success, 0 bytes exported
  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == 0);

  INFO("ExportData empty tag: OK");
}

TEST_CASE("ExportData - Binary export roundtrip", "[cae][export][runtime][binary]") {
  ExportDataFixture f;

  const std::string tag_name  = "export_binary_roundtrip_tag";
  const std::string out_path  = "/tmp/cae_export_binary_roundtrip.bin";

  // Put two blobs with known data
  std::vector<uint8_t> data_a = {0xDE, 0xAD, 0xBE, 0xEF};
  std::vector<uint8_t> data_b = {0x01, 0x02, 0x03, 0x04, 0x05};
  f.PutBlob(tag_name, "blob_a", data_a);
  f.PutBlob(tag_name, "blob_b", data_b);

  // Export to binary
  wrp_cae::core::Client cae(wrp_cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, out_path, "binary");
  fut.Wait();

  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == data_a.size() + data_b.size());

  // Parse the output file: [name_len(u32)][name][data_len(u64)][data] ...
  std::ifstream ifs(out_path, std::ios::binary);
  REQUIRE(ifs.is_open());

  size_t blobs_found = 0;
  size_t total_bytes = 0;
  while (ifs.good()) {
    uint32_t name_len = 0;
    ifs.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    if (ifs.gcount() < static_cast<std::streamsize>(sizeof(name_len))) break;

    std::string blob_name(name_len, '\0');
    ifs.read(blob_name.data(), name_len);

    uint64_t data_len = 0;
    ifs.read(reinterpret_cast<char *>(&data_len), sizeof(data_len));

    std::vector<char> blob_data(data_len);
    ifs.read(blob_data.data(), static_cast<std::streamsize>(data_len));

    REQUIRE(!blob_name.empty());
    REQUIRE(data_len > 0);
    total_bytes += data_len;
    blobs_found++;
  }

  REQUIRE(blobs_found == 2);
  REQUIRE(total_bytes == data_a.size() + data_b.size());

  std::remove(out_path.c_str());
  INFO("ExportData binary roundtrip: found " << blobs_found << " blobs, "
       << total_bytes << " bytes");
}

TEST_CASE("ExportData - Binary bad output path returns -2",
          "[cae][export][runtime][binary]") {
  ExportDataFixture f;

  // Put a blob so we pass the empty-tag check and reach the file-open branch
  const std::string tag_name = "export_bad_path_tag";
  f.PutBlob(tag_name, "blob_x", {1, 2, 3});

  wrp_cae::core::Client cae(wrp_cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name,
                                 "/nonexistent_dir_xyz_cae/out.bin", "binary");
  fut.Wait();

  REQUIRE(fut->result_code_ == -2);
  INFO("ExportData binary bad path: result_code=" << fut->result_code_);
}

#ifdef WRP_CAE_ENABLE_HDF5

TEST_CASE("ExportData - HDF5 export roundtrip", "[cae][export][runtime][hdf5]") {
  ExportDataFixture f;

  const std::string tag_name = "export_hdf5_roundtrip_tag";
  const std::string out_path = "/tmp/cae_export_hdf5_roundtrip.h5";

  std::vector<uint8_t> data1 = {10, 20, 30, 40};
  std::vector<uint8_t> data2 = {50, 60, 70};
  f.PutBlob(tag_name, "ds1", data1);
  f.PutBlob(tag_name, "ds2", data2);

  wrp_cae::core::Client cae(wrp_cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, out_path, "hdf5");
  fut.Wait();

  REQUIRE(fut->result_code_ == 0);
  REQUIRE(fut->bytes_exported_ == data1.size() + data2.size());

  // Verify the HDF5 file has the expected datasets
  hid_t fid = H5Fopen(out_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(fid >= 0);
  REQUIRE(H5Lexists(fid, "ds1", H5P_DEFAULT) > 0);
  REQUIRE(H5Lexists(fid, "ds2", H5P_DEFAULT) > 0);
  H5Fclose(fid);

  std::remove(out_path.c_str());
  INFO("ExportData HDF5 roundtrip OK");
}

TEST_CASE("ExportData - HDF5 bad output path returns -2",
          "[cae][export][runtime][hdf5]") {
  ExportDataFixture f;

  const std::string tag_name = "export_hdf5_bad_path_tag";
  f.PutBlob(tag_name, "blob_y", {1, 2, 3});

  wrp_cae::core::Client cae(wrp_cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name,
                                 "/nonexistent_dir_xyz_cae/out.h5", "hdf5");
  fut.Wait();

  REQUIRE(fut->result_code_ == -2);
  INFO("ExportData HDF5 bad path: result_code=" << fut->result_code_);
}

#else  // !WRP_CAE_ENABLE_HDF5

TEST_CASE("ExportData - HDF5 not compiled returns -3",
          "[cae][export][runtime][hdf5]") {
  ExportDataFixture f;

  const std::string tag_name = "export_hdf5_nocompile_tag";
  f.PutBlob(tag_name, "blob_z", {1, 2, 3});

  wrp_cae::core::Client cae(wrp_cae::core::kCaePoolId);
  auto fut = cae.AsyncExportData(tag_name, "/tmp/no_hdf5_test.h5", "hdf5");
  fut.Wait();

  REQUIRE(fut->result_code_ == -3);
  INFO("ExportData HDF5 not compiled: result_code=" << fut->result_code_);
}

#endif  // WRP_CAE_ENABLE_HDF5

SIMPLE_TEST_MAIN()
