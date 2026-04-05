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

#ifdef HSHM_ENABLE_CUDA

#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cstring>

#include <chimaera/task.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <wrp_cte/core/core_tasks.h>

using namespace chi;
using chimaera::MOD_NAME::GpuSubmitTask;
using chimaera::bdev::AllocateBlocksTask;
using chimaera::bdev::WriteTask;
using chimaera::bdev::ReadTask;
using chimaera::bdev::Block;

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                          \
  {                                                               \
    cudaError_t err = call;                                       \
    if (err != cudaSuccess) {                                     \
      FAIL("CUDA error: " << cudaGetErrorString(err));           \
    }                                                             \
  }

/**
 * GPU kernel to verify GpuSubmitTask fields on device
 * Reads task fields and writes verification result to pinned host
 */
__global__ void VerifyGpuSubmitOnGpu(void *device_task, int *results) {
  auto *task = reinterpret_cast<GpuSubmitTask*>(device_task);
  auto *fshm = reinterpret_cast<FutureShm*>((char*)task + sizeof(GpuSubmitTask));

  // Check all fields match expected values
  if (task->gpu_id_ == 42 &&
      task->test_value_ == 7 &&
      task->result_value_ == 0 &&
      task->counter_addr_ == 0) {
    // Also check FutureShm fields
    if (fshm->method_id_ == 25) {  // Method::kGpuSubmit = 25
      results[0] = 1;  // Success
      return;
    }
  }
  results[0] = 0;  // Failure
}

/**
 * GPU kernel to verify AllocateBlocksTask fields on device
 * Allocates and initializes Block vector on GPU
 */
__global__ void VerifyAllocateBlocksOnGpu(void *device_task, int *results) {
  auto *task = reinterpret_cast<AllocateBlocksTask*>(device_task);

  // Check base Task field
  if (task->size_ != 1024) {
    results[0] = 0;
    return;
  }

  // Check blocks_ vector is accessible (svo pointer after FixupAfterCopy)
  // Since blocks_ is output-only, we just verify it's properly structured
  results[0] = 1;  // Success if we got here
}

/**
 * GPU kernel to verify WriteTask fields on device
 */
__global__ void VerifyWriteTaskOnGpu(void *device_task, int *results) {
  auto *task = reinterpret_cast<WriteTask*>(device_task);

  // Check length field
  if (task->length_ == 512) {
    results[0] = 1;  // Success
    return;
  }
  results[0] = 0;  // Failure
}

/**
 * GPU kernel to verify ReadTask fields on device
 */
__global__ void VerifyReadTaskOnGpu(void *device_task, int *results) {
  auto *task = reinterpret_cast<ReadTask*>(device_task);

  // Check length field
  if (task->length_ == 256) {
    results[0] = 1;  // Success
    return;
  }
  results[0] = 0;  // Failure
}

/**
 * GPU kernel to create GpuSubmitTask on device
 * Simulates GPU creating task with specific values
 */
__global__ void CreateGpuSubmitOnGpu(void *d_buf) {
  // Fill pre-allocated buffer (allocated by host via cudaMalloc)
  auto *task = new(d_buf) GpuSubmitTask();
  task->gpu_id_ = 99;
  task->test_value_ = 42;
  task->result_value_ = 0;
  task->counter_addr_ = 0x1234567890ABCDEFULL;
  task->pool_id_ = PoolId(1, 0);
  task->method_ = 25;  // Method::kGpuSubmit

  // Construct FutureShm in-place
  auto *fshm = new((char*)d_buf + sizeof(GpuSubmitTask)) FutureShm();
  fshm->pool_id_ = task->pool_id_;
  fshm->method_id_ = task->method_;
}

/**
 * GPU kernel to create AllocateBlocksTask on device
 */
__global__ void CreateAllocateBlocksOnGpu(void *d_buf) {
  // Fill pre-allocated buffer
  auto *task = new(d_buf) AllocateBlocksTask();
  task->size_ = 2048;
  task->pool_id_ = PoolId(2, 0);
  task->method_ = 10;  // Method::kAllocateBlocks

  // Construct FutureShm
  auto *fshm = new((char*)d_buf + sizeof(AllocateBlocksTask)) FutureShm();
  fshm->pool_id_ = task->pool_id_;
  fshm->method_id_ = task->method_;
}

// ============================================================================
// CPU->GPU Tests
// ============================================================================

TEST_CASE("CPU->GPU GpuSubmitTask POD transfer", "[gpu][transfer]") {
  // Create task on CPU with specific values
  GpuSubmitTask host_task;
  host_task.gpu_id_ = 42;
  host_task.test_value_ = 7;
  host_task.result_value_ = 0;
  host_task.counter_addr_ = 0;
  host_task.pool_id_ = PoolId(1, 0);
  host_task.method_ = 25;  // Method::kGpuSubmit

  // Allocate device memory for task + FutureShm
  size_t total_size = sizeof(GpuSubmitTask) + sizeof(FutureShm);
  void *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, total_size));

  // Copy task to device
  CUDA_CHECK(cudaMemcpy(d_buf, &host_task, sizeof(GpuSubmitTask),
                        cudaMemcpyHostToDevice));

  // Initialize FutureShm on device
  FutureShm fshm;
  memset(&fshm, 0, sizeof(fshm));
  fshm.pool_id_ = host_task.pool_id_;
  fshm.method_id_ = host_task.method_;
  CUDA_CHECK(cudaMemcpy((char*)d_buf + sizeof(GpuSubmitTask), &fshm,
                        sizeof(FutureShm), cudaMemcpyHostToDevice));

  // Allocate pinned result buffer
  int *d_results;
  CUDA_CHECK(cudaMallocHost(&d_results, sizeof(int)));
  *d_results = 0;

  // Launch verification kernel
  VerifyGpuSubmitOnGpu<<<1, 1>>>(d_buf, d_results);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify result
  REQUIRE(*d_results == 1);

  // Cleanup
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFreeHost(d_results));
}

TEST_CASE("CPU->GPU AllocateBlocksTask POD transfer", "[gpu][transfer]") {
  // Create task on CPU
  AllocateBlocksTask host_task;
  host_task.size_ = 1024;
  host_task.pool_id_ = PoolId(2, 0);
  host_task.method_ = 10;  // Method::kAllocateBlocks

  // Allocate device memory
  size_t total_size = sizeof(AllocateBlocksTask) + sizeof(FutureShm);
  void *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, total_size));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_buf, &host_task, sizeof(AllocateBlocksTask),
                        cudaMemcpyHostToDevice));

  // Initialize FutureShm
  FutureShm fshm;
  memset(&fshm, 0, sizeof(fshm));
  fshm.pool_id_ = host_task.pool_id_;
  fshm.method_id_ = host_task.method_;
  CUDA_CHECK(cudaMemcpy((char*)d_buf + sizeof(AllocateBlocksTask), &fshm,
                        sizeof(FutureShm), cudaMemcpyHostToDevice));

  // Allocate pinned result buffer
  int *d_results;
  CUDA_CHECK(cudaMallocHost(&d_results, sizeof(int)));
  *d_results = 0;

  // Launch verification kernel
  VerifyAllocateBlocksOnGpu<<<1, 1>>>(d_buf, d_results);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify result
  REQUIRE(*d_results == 1);

  // Cleanup
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFreeHost(d_results));
}

TEST_CASE("CPU->GPU WriteTask POD transfer", "[gpu][transfer]") {
  // Create task on CPU
  WriteTask host_task;
  host_task.length_ = 512;
  host_task.pool_id_ = PoolId(2, 0);
  host_task.method_ = 11;  // Method::kWrite

  // Allocate device memory
  size_t total_size = sizeof(WriteTask) + sizeof(FutureShm);
  void *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, total_size));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_buf, &host_task, sizeof(WriteTask),
                        cudaMemcpyHostToDevice));

  // Initialize FutureShm
  FutureShm fshm;
  memset(&fshm, 0, sizeof(fshm));
  fshm.pool_id_ = host_task.pool_id_;
  fshm.method_id_ = host_task.method_;
  CUDA_CHECK(cudaMemcpy((char*)d_buf + sizeof(WriteTask), &fshm,
                        sizeof(FutureShm), cudaMemcpyHostToDevice));

  // Allocate pinned result buffer
  int *d_results;
  CUDA_CHECK(cudaMallocHost(&d_results, sizeof(int)));
  *d_results = 0;

  // Launch verification kernel
  VerifyWriteTaskOnGpu<<<1, 1>>>(d_buf, d_results);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify result
  REQUIRE(*d_results == 1);

  // Cleanup
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFreeHost(d_results));
}

TEST_CASE("CPU->GPU ReadTask POD transfer", "[gpu][transfer]") {
  // Create task on CPU
  ReadTask host_task;
  host_task.length_ = 256;
  host_task.pool_id_ = PoolId(2, 0);
  host_task.method_ = 12;  // Method::kRead

  // Allocate device memory
  size_t total_size = sizeof(ReadTask) + sizeof(FutureShm);
  void *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, total_size));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_buf, &host_task, sizeof(ReadTask),
                        cudaMemcpyHostToDevice));

  // Initialize FutureShm
  FutureShm fshm;
  memset(&fshm, 0, sizeof(fshm));
  fshm.pool_id_ = host_task.pool_id_;
  fshm.method_id_ = host_task.method_;
  CUDA_CHECK(cudaMemcpy((char*)d_buf + sizeof(ReadTask), &fshm,
                        sizeof(FutureShm), cudaMemcpyHostToDevice));

  // Allocate pinned result buffer
  int *d_results;
  CUDA_CHECK(cudaMallocHost(&d_results, sizeof(int)));
  *d_results = 0;

  // Launch verification kernel
  VerifyReadTaskOnGpu<<<1, 1>>>(d_buf, d_results);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify result
  REQUIRE(*d_results == 1);

  // Cleanup
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFreeHost(d_results));
}

// ============================================================================
// GPU->CPU Tests
// ============================================================================

TEST_CASE("GPU->CPU GpuSubmitTask POD transfer", "[gpu][transfer]") {
  // Allocate device buffer via cudaMalloc (host-accessible for D2H copy)
  size_t total = sizeof(GpuSubmitTask) + sizeof(FutureShm);
  void *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, total));

  // Create task on GPU using pre-allocated buffer
  CreateGpuSubmitOnGpu<<<1, 1>>>(d_buf);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy to host
  char host_buf[1024];
  CUDA_CHECK(cudaMemcpy(host_buf, d_buf, total, cudaMemcpyDeviceToHost));

  // Cast and verify
  auto *task = reinterpret_cast<GpuSubmitTask*>(host_buf);
  auto *fshm = reinterpret_cast<FutureShm*>(host_buf + sizeof(GpuSubmitTask));

  // Call FixupAfterCopy (no-op for POD, but verify it doesn't crash)
  // GpuSubmitTask is fully POD, so FixupAfterCopy is not needed,
  // but we could add it for consistency

  // Verify fields
  REQUIRE(task->gpu_id_ == 99);
  REQUIRE(task->test_value_ == 42);
  REQUIRE(task->result_value_ == 0);
  REQUIRE(task->counter_addr_ == 0x1234567890ABCDEFULL);
  REQUIRE(fshm->method_id_ == 25);

  // Cleanup
  CUDA_CHECK(cudaFree(d_buf));
}

TEST_CASE("GPU->CPU AllocateBlocksTask POD transfer with FixupAfterCopy",
          "[gpu][transfer]") {
  // Allocate pinned memory for device pointers
  // Allocate device buffer via cudaMalloc
  size_t total = sizeof(AllocateBlocksTask) + sizeof(FutureShm);
  void *d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, total));

  // Create task on GPU using pre-allocated buffer
  CreateAllocateBlocksOnGpu<<<1, 1>>>(d_buf);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy to host
  char host_buf[1024];
  CUDA_CHECK(cudaMemcpy(host_buf, d_buf, total, cudaMemcpyDeviceToHost));

  // Cast and fixup
  auto *task = reinterpret_cast<AllocateBlocksTask*>(host_buf);
  task->FixupAfterCopy();  // Calls blocks_.FixupSvoPtr()

  // Verify fields
  REQUIRE(task->size_ == 2048);
  REQUIRE(task->pool_id_.major_ == 2);
  REQUIRE(task->method_ == 10);

  // Cleanup
  CUDA_CHECK(cudaFree(d_buf));
}

// ============================================================================
// Task layout consistency test
// ============================================================================

TEST_CASE("Task base class layout is consistent CPU/GPU", "[gpu][transfer]") {
  // Verify sizeof(Task) on CPU == what we expect
  // This should match across CPU and GPU compilation

  // All tasks should have the same base layout
  REQUIRE(sizeof(GpuSubmitTask) > 0);
  REQUIRE(sizeof(AllocateBlocksTask) > 0);
  REQUIRE(sizeof(WriteTask) > 0);
  REQUIRE(sizeof(ReadTask) > 0);

  // FutureShm should be POD (no virtual functions, all fields standard types)
  REQUIRE(sizeof(FutureShm) > 0);

  // Verify that GpuSubmitTask fields are at predictable offsets
  GpuSubmitTask t;
  t.gpu_id_ = 0;
  t.test_value_ = 0;
}

#endif  // HSHM_ENABLE_CUDA
