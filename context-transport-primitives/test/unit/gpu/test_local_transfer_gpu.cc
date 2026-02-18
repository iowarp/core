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
 * GPU unit test for ShmTransport with GpuShmMmap backend
 *
 * This test verifies that data transfer works correctly with GPU-accessible
 * pinned memory using ShmTransferInfo and ShmTransport::WriteTransfer/ReadTransfer:
 * 1. Allocates pinned host memory using GpuShmMmap backend for copy space
 * 2. Uses ShmTransferInfo for SPSC ring buffer metadata
 * 3. GPU kernel writes data via ShmTransport::WriteTransfer
 * 4. CPU reads data via ShmTransport::ReadTransfer
 * 5. CPU verifies the transferred data
 */

#include <catch2/catch_all.hpp>

#include "hermes_shm/lightbeam/shm_transport.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#include "hermes_shm/util/gpu_api.h"

using hshm::ipc::ArenaAllocator;
using hshm::ipc::GpuShmMmap;
using hshm::ipc::MemoryBackendId;
using hshm::lbm::LbmContext;
using hshm::lbm::ShmTransferInfo;
using hshm::lbm::ShmTransport;

/**
 * GPU kernel to fill a buffer with a pattern
 */
__global__ void FillBufferKernel(char *buffer, size_t size, char pattern) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < size; i += stride) {
    buffer[i] = pattern;
  }
}

/**
 * GPU kernel that writes data to copy_space via ShmTransport::WriteTransfer
 * Uses the GPU-compatible SPSC ring buffer to transfer data.
 * Only thread 0 performs the transfer (single-producer).
 */
__global__ void GpuWriteTransferKernel(const char *src_buffer, size_t data_size,
                                        char *copy_space, ShmTransferInfo *shm_info) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::WriteTransfer(src_buffer, data_size, ctx);
  }
}

/**
 * GPU kernel to set a value at a specific location (for simple tests)
 */
__global__ void SetValueKernel(char *buffer, size_t index, char value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    buffer[index] = value;
  }
}

/**
 * Test GPU to CPU data transfer using ShmTransferInfo SPSC ring buffer
 */
TEST_CASE("ShmTransfer GPU", "[gpu][transfer]") {
  constexpr size_t kBackendSize = 16 * 1024 * 1024;  // 16MB
  constexpr size_t kCopySpaceSize = 16 * 1024;       // 16KB transfer granularity
  constexpr size_t kDataSize = 64 * 1024;            // 64KB buffer
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_local_transfer_gpu";

  SECTION("BasicGpuToCpuTransfer") {
    // Step 1: Create a GpuShmMmap backend for pinned host memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 0);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an ArenaAllocator on that backend
    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate copy space and ShmTransferInfo from pinned memory
    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Step 4: Allocate GPU source buffer (pinned so both GPU and CPU can access)
    char *gpu_buffer;
    cudaMallocHost(&gpu_buffer, kDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Step 5: Fill the buffer with pattern (value = 1) using GPU kernel
    constexpr char kPattern = 1;
    int blockSize = 256;
    int numBlocks = (kDataSize + blockSize - 1) / blockSize;
    FillBufferKernel<<<numBlocks, blockSize>>>(gpu_buffer, kDataSize, kPattern);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Step 6: GPU writes data via ShmTransport::WriteTransfer (SPSC ring buffer)
    // Launch with single thread since WriteTransfer is single-producer
    GpuWriteTransferKernel<<<1, 1>>>(gpu_buffer, kDataSize, copy_space, shm_info);

    // Step 7: CPU reads data via ShmTransport::ReadTransfer
    std::vector<char> received_data(kDataSize);
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::ReadTransfer(received_data.data(), kDataSize, ctx);

    // Wait for GPU kernel to complete
    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Step 8: Verify all data was transferred
    REQUIRE(received_data.size() == kDataSize);

    bool all_ones = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (received_data[i] != kPattern) {
        all_ones = false;
        break;
      }
    }
    REQUIRE(all_ones);

    // Cleanup
    cudaFreeHost(gpu_buffer);
  }

  SECTION("ChunkedTransferWithPattern") {
    // Test with a more complex pattern to verify data integrity
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_pattern", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Allocate and initialize GPU buffer with pattern
    char *gpu_buffer;
    cudaMallocHost(&gpu_buffer, kDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Initialize with pattern on CPU (index % 256)
    for (size_t i = 0; i < kDataSize; ++i) {
      gpu_buffer[i] = static_cast<char>(i % 256);
    }

    // GPU writes via SPSC ring buffer
    GpuWriteTransferKernel<<<1, 1>>>(gpu_buffer, kDataSize, copy_space, shm_info);

    // CPU reads via SPSC ring buffer
    std::vector<char> received_data(kDataSize);
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::ReadTransfer(received_data.data(), kDataSize, ctx);

    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Verify data integrity
    REQUIRE(received_data.size() == kDataSize);
    bool pattern_correct = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (received_data[i] != static_cast<char>(i % 256)) {
        pattern_correct = false;
        break;
      }
    }
    REQUIRE(pattern_correct);

    cudaFreeHost(gpu_buffer);
  }

  SECTION("DirectGpuMemoryAccess") {
    // Test that GPU can directly read/write to the GpuShmMmap memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 2);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_direct", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Allocate buffer directly from GpuShmMmap
    auto buffer_ptr = alloc_ptr->AllocateObjs<char>(1024);
    char *buffer = buffer_ptr.ptr_;
    REQUIRE(buffer != nullptr);

    // Initialize on CPU
    std::memset(buffer, 0, 1024);

    // GPU sets specific values
    SetValueKernel<<<1, 1>>>(buffer, 0, 'A');
    SetValueKernel<<<1, 1>>>(buffer, 100, 'B');
    SetValueKernel<<<1, 1>>>(buffer, 500, 'C');
    SetValueKernel<<<1, 1>>>(buffer, 1023, 'D');

    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // CPU reads and verifies
    REQUIRE(buffer[0] == 'A');
    REQUIRE(buffer[100] == 'B');
    REQUIRE(buffer[500] == 'C');
    REQUIRE(buffer[1023] == 'D');

    // Verify untouched locations are still 0
    REQUIRE(buffer[1] == 0);
    REQUIRE(buffer[50] == 0);
    REQUIRE(buffer[1022] == 0);
  }

  SECTION("LargeTransferPerformance") {
    // Test larger transfer (256KB) to verify performance
    constexpr size_t kLargeDataSize = 256 * 1024;  // 256KB

    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 3);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_large", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    auto shm_info_ptr = alloc_ptr->AllocateObjs<ShmTransferInfo>(1);
    ShmTransferInfo *shm_info = shm_info_ptr.ptr_;
    REQUIRE(shm_info != nullptr);
    new (shm_info) ShmTransferInfo();
    shm_info->copy_space_size_ = kCopySpaceSize;

    // Allocate GPU buffer
    char *gpu_buffer;
    cudaMallocHost(&gpu_buffer, kLargeDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Fill with pattern
    constexpr char kPattern = 0x55;
    int blockSize = 256;
    int numBlocks = (kLargeDataSize + blockSize - 1) / blockSize;
    FillBufferKernel<<<numBlocks, blockSize>>>(gpu_buffer, kLargeDataSize,
                                               kPattern);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // GPU writes via SPSC ring buffer
    GpuWriteTransferKernel<<<1, 1>>>(gpu_buffer, kLargeDataSize, copy_space, shm_info);

    // CPU reads via SPSC ring buffer
    std::vector<char> received_data(kLargeDataSize);
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = shm_info;
    ShmTransport::ReadTransfer(received_data.data(), kLargeDataSize, ctx);

    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Verify
    REQUIRE(received_data.size() == kLargeDataSize);

    bool pattern_correct = true;
    for (size_t i = 0; i < kLargeDataSize; ++i) {
      if (received_data[i] != kPattern) {
        pattern_correct = false;
        break;
      }
    }
    REQUIRE(pattern_correct);

    cudaFreeHost(gpu_buffer);
  }
}
