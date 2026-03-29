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
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Unit tests for GPU memory allocation in CHI_IPC
 * Tests GPU kernel memory allocation using BuddyAllocator
 * Only compiles when CUDA or HIP is enabled
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/local_task_archives.h>
#include <chimaera/pool_query.h>
#include <chimaera/task.h>
#include <hermes_shm/memory/backend/gpu_malloc.h>
#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/lightbeam/shm_transport.h>

#include <cstring>
#include <memory>
#include <vector>

#include "../simple_test.h"

namespace {

/**
 * Minimal GPU kernel to test basic execution (no CHIMAERA_GPU_INIT)
 */
__global__ void test_gpu_minimal_kernel(int *results) {
  int thread_id = threadIdx.x;
  results[thread_id] = thread_id + 100;  // Write a test value
}

/**
 * Test writing to backend.data_ without shm_init
 */
__global__ void test_gpu_backend_write_kernel(const hipc::MemoryBackend backend,
                                              int *results) {
  int thread_id = threadIdx.x;

  // Try to write a simple value to backend.data_
  if (thread_id == 0 && backend.data_ != nullptr) {
    char *test_ptr = backend.data_;
    test_ptr[0] = 42;                          // Simple write test
    results[0] = (test_ptr[0] == 42) ? 0 : 1;  // Verify
  }

  if (thread_id != 0) {
    results[thread_id] = 0;  // Other threads just pass
  }
}

/**
 * Test placement new on BuddyAllocator without shm_init
 */
__global__ void test_gpu_placement_new_kernel(const hipc::MemoryBackend backend,
                                              int *results) {
  int thread_id = threadIdx.x;

  if (thread_id == 0 && backend.data_ != nullptr) {
    // Try placement new without calling shm_init
    hipc::BuddyAllocator *alloc =
        reinterpret_cast<hipc::BuddyAllocator *>(backend.data_);
    new (alloc) hipc::BuddyAllocator();
    results[0] = 0;  // Success if we got here
  } else {
    results[thread_id] = 0;
  }
}

/**
 * Test placement new + shm_init
 */
__global__ void test_gpu_shm_init_kernel(const hipc::MemoryBackend backend,
                                         int *results) {
  int thread_id = threadIdx.x;

  if (thread_id == 0 && backend.data_ != nullptr) {
    hipc::BuddyAllocator *alloc =
        reinterpret_cast<hipc::BuddyAllocator *>(backend.data_);
    new (alloc) hipc::BuddyAllocator();
    results[0] = 1;  // Mark that we got past placement new
    alloc->shm_init(backend, backend.data_capacity_);
    results[0] = 0;  // Success if we got past shm_init
  } else {
    results[thread_id] = 0;
  }
}

/**
 * Test everything except IpcManager construction
 */
__global__ void test_gpu_alloc_no_ipc_kernel(const hipc::MemoryBackend backend,
                                             int *results) {
  __shared__ hipc::BuddyAllocator *g_arena_alloc;
  int thread_id = threadIdx.x;

  if (thread_id == 0) {
    g_arena_alloc =
        reinterpret_cast<hipc::BuddyAllocator *>(backend.data_);
    new (g_arena_alloc) hipc::BuddyAllocator();
    g_arena_alloc->shm_init(backend, backend.data_capacity_);
  }
  __syncthreads();

  results[thread_id] = 0;  // Success
}

/**
 * Test just IpcManager construction in __shared__ memory
 */
__global__ void test_gpu_ipc_construct_kernel(int *results) {
  chi::IpcManager *ipc = chi::IpcManager::GetBlockIpcManager();
  int thread_id = threadIdx.x;
  __syncthreads();

  results[thread_id] = (ipc != nullptr) ? 0 : 1;
}

/**
 * Simple GPU kernel for testing CHIMAERA_GPU_INIT without allocation
 * Just verifies initialization succeeds
 */
__global__ void test_gpu_init_only_kernel(
    chi::IpcManagerGpu gpu_info,
    int *results)
{
  CHIMAERA_GPU_INIT(gpu_info);

  // Just report success if initialization didn't crash
  results[thread_id] = 0;
  __syncthreads();
}

/**
 * GPU kernel for testing CHIMAERA_GPU_INIT and AllocateBuffer
 * Each thread allocates a buffer, writes data, and verifies it
 */
__global__ void test_gpu_allocate_buffer_kernel(
    chi::IpcManagerGpu gpu_info,
    int *results,             ///< Output: test results (0=pass, non-zero=fail)
    size_t *allocated_sizes,  ///< Output: sizes allocated per thread
    char **allocated_ptrs)    ///< Output: pointers allocated per thread
{
  CHIMAERA_GPU_INIT(gpu_info);

  // Warp-level allocation: only lane 0 allocates, all lanes verify.
  // Each warp allocates (warp_size * per_lane_size) bytes, then each lane
  // writes/verifies its own slice.
  chi::u32 lane_id = chi::IpcManager::GetLaneId();
  size_t per_lane_size = 64;
  size_t alloc_size = 32 * per_lane_size;  // One allocation per warp

  // Lane 0 allocates for the entire warp
  __shared__ char *s_warp_buffer;
  if (lane_id == 0) {
    hipc::FullPtr<char> buffer = CHI_IPC->AllocateBuffer(alloc_size);
    s_warp_buffer = buffer.IsNull() ? nullptr : buffer.ptr_;
  }
  __syncwarp();

  char *warp_buffer = s_warp_buffer;
  if (warp_buffer == nullptr) {
    results[thread_id] = 1;  // Allocation failed
    allocated_sizes[thread_id] = 0;
    allocated_ptrs[thread_id] = nullptr;
  } else {
    // Each lane writes its own slice
    char *my_slice = warp_buffer + lane_id * per_lane_size;
    char pattern = (char)(thread_id + 1);
    for (size_t i = 0; i < per_lane_size; ++i) {
      my_slice[i] = pattern;
    }

    // Verify pattern
    bool pattern_ok = true;
    for (size_t i = 0; i < per_lane_size; ++i) {
      if (my_slice[i] != pattern) {
        pattern_ok = false;
        break;
      }
    }

    results[thread_id] = pattern_ok ? 0 : 2;  // 2=verification failed
    allocated_sizes[thread_id] = per_lane_size;
    allocated_ptrs[thread_id] = my_slice;
  }

  __syncthreads();
}

/**
 * GPU kernel for testing ToFullPtr on GPU
 * Allocates a buffer, gets its FullPtr, and verifies it works
 */
__global__ void test_gpu_to_full_ptr_kernel(
    chi::IpcManagerGpu gpu_info,
    int *results)
{
  CHIMAERA_GPU_INIT(gpu_info);

  // Warp-level: only lane 0 allocates and tests ToFullPtr.
  // Other lanes just pass.
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (lane_id == 0) {
    // Allocate a buffer
    size_t alloc_size = 512;
    hipc::FullPtr<char> buffer = CHI_IPC->AllocateBuffer(alloc_size);

    if (buffer.IsNull()) {
      results[thread_id] = 1;  // Allocation failed
      __syncwarp();
      return;
    }

    // Write test data
    char test_value = (char)(thread_id + 42);
    for (size_t i = 0; i < alloc_size; ++i) {
      buffer.ptr_[i] = test_value;
    }

    // Get a ShmPtr and convert back to FullPtr
    hipc::ShmPtr<char> shm_ptr = buffer.shm_;

    // Convert back using ToFullPtr
    hipc::FullPtr<char> recovered = CHI_IPC->ToFullPtr(shm_ptr);

    if (recovered.IsNull()) {
      results[thread_id] = 3;  // ToFullPtr failed
      __syncwarp();
      return;
    }

    // Verify the recovered pointer works
    bool data_ok = true;
    for (size_t i = 0; i < alloc_size; ++i) {
      if (recovered.ptr_[i] != test_value) {
        data_ok = false;
        break;
      }
    }

    results[thread_id] = data_ok ? 0 : 4;  // 4=recovered data mismatch
  } else {
    results[thread_id] = 0;  // Non-lane-0 threads pass
  }
  __syncwarp();
}

/**
 * GPU kernel for testing multiple independent allocations per thread
 * Each thread makes multiple allocations and verifies they're independent
 */
__global__ void test_gpu_multiple_allocs_kernel(
    chi::IpcManagerGpu gpu_info,
    int *results)
{
  CHIMAERA_GPU_INIT(gpu_info);

  // Warp-level: only lane 0 allocates and verifies multiple buffers.
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (lane_id == 0) {
    const int num_allocs = 4;
    size_t alloc_sizes[] = {256, 512, 1024, 2048};

    // Use local array for thread-local pointers
    char *local_ptrs[4];

    // Allocate multiple buffers
    for (int i = 0; i < num_allocs; ++i) {
      hipc::FullPtr<char> buffer =
          CHI_IPC->AllocateBuffer(alloc_sizes[i]);

      if (buffer.IsNull()) {
        results[thread_id] = 10 + i;  // Allocation i failed
        return;
      }

      local_ptrs[i] = buffer.ptr_;

      // Initialize with unique pattern
      char pattern = (char)(thread_id * num_allocs + i);
      for (size_t j = 0; j < alloc_sizes[i]; ++j) {
        local_ptrs[i][j] = pattern;
      }
    }

    // Verify all allocations
    for (int i = 0; i < num_allocs; ++i) {
      char expected = (char)(thread_id * num_allocs + i);
      for (size_t j = 0; j < alloc_sizes[i]; ++j) {
        if (local_ptrs[i][j] != expected) {
          results[thread_id] = 20 + i;  // Verification i failed
          return;
        }
      }
    }
  }

  results[thread_id] = 0;  // All tests passed
}

/**
 * GPU kernel for testing NewTask from GPU
 * Tests that IpcManager::NewTask works from GPU kernel
 */
__global__ void test_gpu_new_task_kernel(chi::IpcManagerGpu gpu_info,
                                         int *results) {
  CHIMAERA_GPU_INIT(gpu_info);

  // Only thread 0 creates task
  if (thread_id == 0) {
    // Create task using NewTask
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(1000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 0;
    chi::u32 test_value = 123;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);

    if (task.IsNull()) {
      results[0] = 1;  // NewTask failed
    } else {
      // Verify task was created correctly
      if (task->gpu_id_ == gpu_id && task->test_value_ == test_value) {
        results[0] = 0;  // Success
      } else {
        results[0] = 2;  // Task created but values wrong
      }
    }
  }

  __syncthreads();
}

/**
 * GPU kernel for testing task serialization/deserialization on GPU
 * Creates a task, serializes it, then deserializes and verifies
 */
__global__ void test_gpu_serialize_deserialize_kernel(
    chi::IpcManagerGpu gpu_info, int *results) {
  CHIMAERA_GPU_INIT(gpu_info);

  // Only thread 0 tests serialization
  if (thread_id == 0) {
#if !HSHM_IS_HOST
    // Create a task using NewTask
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(2000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 7;
    chi::u32 test_value = 456;

    auto original_task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        task_id, pool_id, query, gpu_id, test_value);

    if (original_task.IsNull()) {
      results[0] = 1;  // NewTask failed
      __syncthreads();
      return;
    }

    // Serialize task using DefaultSaveArchive
    auto *alloc = CHI_PRIV_ALLOC;
    chi::DefaultSaveArchive save_ar(chi::LocalMsgType::kSerializeIn, alloc);
    original_task->SerializeIn(save_ar);
    size_t serialized_size = save_ar.GetSize();

    // Create a new task to deserialize into
    auto loaded_task =
        CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>();

    if (loaded_task.IsNull()) {
      results[0] = 4;  // Second NewTask failed
      __syncthreads();
      return;
    }

    // Deserialize using DefaultLoadArchive from save archive data
    chi::DefaultLoadArchive load_ar(save_ar.GetData());
    loaded_task->SerializeIn(load_ar);

    // Verify deserialized task matches original
    if (loaded_task->gpu_id_ == gpu_id &&
        loaded_task->test_value_ == test_value &&
        loaded_task->result_value_ == 0) {
      results[0] = 0;  // Success
    } else {
      results[0] = 3;  // Deserialization mismatch
    }
#endif  // !HSHM_IS_HOST
  }

  __syncthreads();
}

/**
 * GPU kernel for testing task serialization on GPU for CPU deserialization
 * Creates task, serializes with DefaultSaveArchive, copies to output buffer
 */
__global__ void test_gpu_serialize_for_cpu_kernel(
    chi::IpcManagerGpu gpu_info, char *output_buffer, size_t *output_size,
    int *results) {
  CHIMAERA_GPU_INIT(gpu_info);

  // Only thread 0 serializes
  if (thread_id == 0) {
#if !HSHM_IS_HOST
    // Create a task using NewTask
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(3000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 42;
    chi::u32 test_value = 99999;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);

    if (task.IsNull()) {
      results[0] = 1;  // NewTask failed
      *output_size = 0;
      __syncthreads();
      return;
    }

    // Serialize task using DefaultSaveArchive
    auto *alloc = CHI_PRIV_ALLOC;
    chi::DefaultSaveArchive save_ar(chi::LocalMsgType::kSerializeIn, alloc);
    task->SerializeIn(save_ar);

    // Copy serialized data to output buffer
    size_t sz = save_ar.GetSize();
    memcpy(output_buffer, save_ar.GetData().data(), sz);

    // Store serialized size
    *output_size = sz;
    results[0] = 0;  // Success
#endif  // !HSHM_IS_HOST
  }

  __syncthreads();
}

/**
 * GPU kernel that uses SendGpu to create+serialize+enqueue a task,
 * then blocks in Future::Wait until the CPU sets FUTURE_COMPLETE.
 */
__global__ void test_gpu_send_queue_wait_kernel(
    chi::IpcManagerGpu gpu_info,
    int *d_result) {
  CHIMAERA_GPU_INIT(gpu_info);

  if (thread_id == 0) {
    // 1. Create task on GPU
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(6000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 42;
    chi::u32 test_value = 77777;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);
    if (task.IsNull()) {
      *d_result = -1;
      return;
    }

    // 2. Use SendGpu (serializes into ring buffer + enqueues)
    auto future = CHI_IPC->SendGpu(task);
    if (future.IsNull()) {
      *d_result = -2;
      return;
    }

    // 3. Block until CPU sets FUTURE_COMPLETE
    future.Wait();
    *d_result = 0;
  }

  __syncthreads();
}

/**
 * Test 1: AllocateBuffer + NewTask with IpcManagerGpu per-warp allocators
 * Lane 0 allocates a buffer, all lanes write/verify their own slice,
 * then lane 0 also creates a task.
 */
__global__ void test_gpu_ipc_manager_gpu_alloc_kernel(
    chi::IpcManagerGpu gpu_info, int *results) {
  CHIMAERA_GPU_INIT(gpu_info);

  chi::u32 lane_id = chi::IpcManager::GetLaneId();
  size_t per_lane_size = 64;

  // Lane 0 allocates for the entire warp
  __shared__ char *s_warp_buffer;
  if (lane_id == 0) {
    hipc::FullPtr<char> buffer =
        CHI_IPC->AllocateBuffer(num_threads * per_lane_size);
    s_warp_buffer = buffer.IsNull() ? nullptr : buffer.ptr_;
  }
  __syncthreads();

  char *warp_buffer = s_warp_buffer;
  if (warp_buffer == nullptr) {
    results[thread_id] = 1;  // Allocation failed
    __syncthreads();
    return;
  }

  // Each lane writes/verifies its own slice
  char *my_slice = warp_buffer + thread_id * per_lane_size;
  char pattern = (char)(thread_id + 1);
  for (int i = 0; i < (int)per_lane_size; ++i) {
    my_slice[i] = pattern;
  }
  for (int i = 0; i < (int)per_lane_size; ++i) {
    if (my_slice[i] != pattern) {
      results[thread_id] = 2;
      __syncthreads();
      return;
    }
  }

  // Thread 0 also creates a task
  if (thread_id == 0) {
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(7000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        task_id, pool_id, query, (chi::u32)1, (chi::u32)42);
    if (task.IsNull()) {
      results[0] = 3;
      __syncthreads();
      return;
    }
    if (task->gpu_id_ != 1 || task->test_value_ != 42) {
      results[0] = 4;
      __syncthreads();
      return;
    }
  }

  results[thread_id] = 0;
  __syncthreads();
}

/**
 * Test 2: GPU SendGpu serialization via ring buffer
 * GPU creates task, calls SendGpu (which writes to ring buffer).
 * CPU thread pops from queue, reads input via ShmTransport::Recv.
 */
__global__ void test_gpu_send_serialization_kernel(
    chi::IpcManagerGpu gpu_info,
    int *d_result) {
  CHIMAERA_GPU_INIT(gpu_info);

  if (thread_id == 0) {
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(8000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 55;
    chi::u32 test_value = 12345;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        task_id, pool_id, query, gpu_id, test_value);
    if (task.IsNull()) {
      *d_result = -1;
      return;
    }

    auto future = CHI_IPC->SendGpu(task);
    if (future.IsNull()) {
      *d_result = -2;
      return;
    }

    // Wait for CPU to signal completion
    future.Wait();
    *d_result = 0;
  }

  __syncthreads();
}

/**
 * Test 4: GPU SendGpu + CPU process + GPU RecvGpu (round-trip)
 * GPU sends, CPU processes via ring buffer, GPU receives output via RecvGpu
 */
__global__ void test_gpu_send_recv_roundtrip_kernel(
    chi::IpcManagerGpu gpu_info,
    int *d_result) {
  CHIMAERA_GPU_INIT(gpu_info);

  if (thread_id == 0) {
    // Create and send task
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(9000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 10;
    chi::u32 test_value = 55555;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        task_id, pool_id, query, gpu_id, test_value);
    if (task.IsNull()) {
      *d_result = -1;
      return;
    }

    auto future = CHI_IPC->SendGpu(task);
    if (future.IsNull()) {
      *d_result = -2;
      return;
    }

    // Receive output via RecvGpu (reads ring buffer, then waits for FUTURE_COMPLETE)
    CHI_IPC->RecvGpu(future, task.ptr_);

    // Check deserialized output (CPU should have set result_value_ = test_value * 2)
    if (task->result_value_ == test_value * 2) {
      *d_result = 0;
    } else {
      *d_result = (int)task->result_value_;  // Return actual value for debug
    }
  }

  __syncthreads();
}

/**
 * Helper function to run GPU kernel and check results
 */
bool run_gpu_kernel_test(const std::string &kernel_name,
                         const hipc::MemoryBackend &backend, int block_size) {
  // Allocate result arrays on GPU
  int *d_results = hshm::GpuApi::Malloc<int>(sizeof(int) * block_size);

  // Initialize results to -1 (not run)
  std::vector<int> h_results(block_size, -1);
  hshm::GpuApi::Memcpy(d_results, h_results.data(), sizeof(int) * block_size);

  chi::IpcManagerGpu gpu_info(backend, nullptr);

  // Special test kernels (don't use IpcManagerGpu)
  if (kernel_name == "minimal") {
    test_gpu_minimal_kernel<<<1, block_size>>>(d_results);
  } else if (kernel_name == "backend_write") {
    test_gpu_backend_write_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "placement_new") {
    test_gpu_placement_new_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "shm_init") {
    test_gpu_shm_init_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "alloc_no_ipc") {
    test_gpu_alloc_no_ipc_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "ipc_construct") {
    test_gpu_ipc_construct_kernel<<<1, block_size>>>(d_results);
  } else if (kernel_name == "init_only") {
    test_gpu_init_only_kernel<<<1, block_size>>>(gpu_info, d_results);
  } else if (kernel_name == "allocate_buffer") {
    size_t *d_allocated_sizes =
        hshm::GpuApi::Malloc<size_t>(sizeof(size_t) * block_size);
    char **d_allocated_ptrs =
        hshm::GpuApi::Malloc<char *>(sizeof(char *) * block_size);

    test_gpu_allocate_buffer_kernel<<<1, block_size>>>(
        gpu_info, d_results, d_allocated_sizes, d_allocated_ptrs);

    hshm::GpuApi::Free(d_allocated_sizes);
    hshm::GpuApi::Free(d_allocated_ptrs);
  } else if (kernel_name == "to_full_ptr") {
    test_gpu_to_full_ptr_kernel<<<1, block_size>>>(gpu_info, d_results);
  } else if (kernel_name == "multiple_allocs") {
    test_gpu_multiple_allocs_kernel<<<1, block_size>>>(gpu_info, d_results);
  } else if (kernel_name == "new_task") {
    test_gpu_new_task_kernel<<<1, 1>>>(gpu_info, d_results);
  } else if (kernel_name == "serialize_deserialize") {
    test_gpu_serialize_deserialize_kernel<<<1, 1>>>(gpu_info, d_results);
  }

  // Synchronize to check for kernel errors
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    INFO("Kernel execution failed: " << cudaGetErrorString(sync_err));
    hshm::GpuApi::Free(d_results);
    return false;
  }

  // Copy results back
  cudaError_t memcpy_err =
      cudaMemcpy(h_results.data(), d_results, sizeof(int) * block_size,
                 cudaMemcpyDeviceToHost);
  if (memcpy_err != cudaSuccess) {
    INFO("Memcpy failed: " << cudaGetErrorString(memcpy_err));
    hshm::GpuApi::Free(d_results);
    return false;
  }
  hshm::GpuApi::Free(d_results);

  // Check results
  bool all_passed = true;
  for (int i = 0; i < block_size; ++i) {
    int expected = (kernel_name == "minimal") ? (i + 100) : 0;
    if (h_results[i] != expected) {
      INFO(kernel_name << " failed for thread " << i << ": result="
                       << h_results[i] << ", expected=" << expected);
      all_passed = false;
    }
  }

  return all_passed;
}

}  // namespace

TEST_CASE("GPU IPC AllocateBuffer basic functionality",
          "[gpu][ipc][allocate_buffer]") {
  // Create GPU memory backend
  hipc::MemoryBackendId backend_id(2, 0);     // Use ID 2.0 for GPU backend
  size_t gpu_memory_size = 10 * 1024 * 1024;  // 10MB GPU memory

  hipc::GpuShmMmap gpu_backend;
  REQUIRE(gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_test", 0));

  SECTION("GPU kernel minimal (no macro)") {
    int block_size = 32;
    bool passed = run_gpu_kernel_test("minimal", gpu_backend, block_size);
    if (!passed) {
      INFO("Basic GPU kernel execution failed - hardware/driver issue?");
    }
    REQUIRE(passed);
  }

  SECTION("GPU kernel backend write") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("backend_write", gpu_backend, block_size));
  }

  SECTION("GPU kernel placement new") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("placement_new", gpu_backend, block_size));
  }

  SECTION("GPU kernel shm_init") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("shm_init", gpu_backend, block_size));
  }

  SECTION("GPU kernel alloc without IpcManager") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("alloc_no_ipc", gpu_backend, block_size));
  }

  SECTION("GPU kernel IpcManager construct") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("ipc_construct", gpu_backend, block_size));
  }

  SECTION("GPU kernel init only") {
    int block_size = 32;  // Warp size
    REQUIRE(run_gpu_kernel_test("init_only", gpu_backend, block_size));
  }

  SECTION("GPU kernel allocate buffer") {
    int block_size = 32;  // Warp size
    REQUIRE(run_gpu_kernel_test("allocate_buffer", gpu_backend, block_size));
  }

  SECTION("GPU kernel NewTask") {
    INFO("Testing IpcManager::NewTask on GPU");
    REQUIRE(run_gpu_kernel_test("new_task", gpu_backend, 1));
  }

  SECTION("GPU kernel serialize/deserialize") {
    INFO("Testing GPU task serialization and deserialization");
    // This kernel uses DefaultSaveArchive which requires CHI_PRIV_ALLOC.
    // run_gpu_kernel_test creates IpcManagerGpu without a heap backend, so
    // we must set up a GpuMalloc heap backend and call the kernel directly.
    hipc::MemoryBackendId heap_backend_id(23, 0);
    hipc::GpuMalloc gpu_heap_backend;
    REQUIRE(gpu_heap_backend.shm_init(heap_backend_id, 4 * 1024 * 1024,
                                      "/gpu_test_heap_sd", 0));
    chi::IpcManagerGpu gpu_info(gpu_backend, nullptr);

    int *d_results = hshm::GpuApi::Malloc<int>(sizeof(int));
    int h_init = -1;
    hshm::GpuApi::Memcpy(d_results, &h_init, sizeof(int));

    test_gpu_serialize_deserialize_kernel<<<1, 1>>>(gpu_info, d_results);
    cudaError_t sync_err = cudaDeviceSynchronize();
    REQUIRE(sync_err == cudaSuccess);

    int h_result = -1;
    hshm::GpuApi::Memcpy(&h_result, d_results, sizeof(int));
    hshm::GpuApi::Free(d_results);
    REQUIRE(h_result == 0);
  }

  SECTION("GPU serialize -> CPU deserialize") {
    INFO(
        "Testing GPU task serialization -> ShmTransport -> CPU "
        "deserialization");

    hipc::MemoryBackendId heap_backend_id2(24, 0);
    hipc::GpuMalloc gpu_heap_backend2;
    REQUIRE(gpu_heap_backend2.shm_init(heap_backend_id2, 4 * 1024 * 1024,
                                       "/gpu_test_heap_sc", 0));
    chi::IpcManagerGpu gpu_info(gpu_backend, nullptr);

    // Allocate pinned host buffer for transfer
    size_t buffer_size = 1024;
    char *h_buffer = nullptr;
    cudaError_t err = cudaMallocHost(&h_buffer, buffer_size);
    REQUIRE(err == cudaSuccess);

    // Allocate GPU buffer
    char *d_buffer = hshm::GpuApi::Malloc<char>(buffer_size);
    size_t *d_output_size = hshm::GpuApi::Malloc<size_t>(sizeof(size_t));
    int *d_results = hshm::GpuApi::Malloc<int>(sizeof(int));

    // Run GPU kernel to serialize task using DefaultSaveArchive
    test_gpu_serialize_for_cpu_kernel<<<1, 1>>>(gpu_info, d_buffer,
                                                d_output_size, d_results);

    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Check GPU serialization result
    int h_result = -1;
    hshm::GpuApi::Memcpy(&h_result, d_results, sizeof(int));
    REQUIRE(h_result == 0);

    // Get serialized size
    size_t h_output_size = 0;
    hshm::GpuApi::Memcpy(&h_output_size, d_output_size, sizeof(size_t));
    INFO("Serialized task size: " + std::to_string(h_output_size) + " bytes");

    // ShmTransport: Copy serialized data from GPU to pinned host memory
    hshm::GpuApi::Memcpy(h_buffer, d_buffer, h_output_size);

    // Deserialize on CPU using DefaultLoadArchive
    std::vector<char> cpu_buffer(h_buffer, h_buffer + h_output_size);
    chi::DefaultLoadArchive load_ar(cpu_buffer);

    // Create a task to deserialize into
    chimaera::MOD_NAME::GpuSubmitTask cpu_task;
    cpu_task.SerializeIn(load_ar);

    // Verify deserialized task values
    REQUIRE(cpu_task.gpu_id_ == 42);
    REQUIRE(cpu_task.test_value_ == 99999);
    REQUIRE(cpu_task.result_value_ == 0);

    // Cleanup
    cudaFreeHost(h_buffer);
    hshm::GpuApi::Free(d_buffer);
    hshm::GpuApi::Free(d_output_size);
    hshm::GpuApi::Free(d_results);
  }
}

TEST_CASE("GPU IPC IpcManagerGpu per-thread allocators",
          "[gpu][ipc][ipc_manager_gpu]") {
  hipc::MemoryBackendId backend_id(2, 0);
  size_t gpu_memory_size = 10 * 1024 * 1024;

  hipc::GpuShmMmap gpu_backend;
  REQUIRE(gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_test_ipc_mgr", 0));

  SECTION("Test 1: AllocateBuffer + NewTask with IpcManagerGpu") {
    INFO("Testing per-thread allocator table with AllocateBuffer and NewTask");
    chi::IpcManagerGpu gpu_info(gpu_backend, nullptr);

    int block_size = 4;
    int *d_results = hshm::GpuApi::Malloc<int>(sizeof(int) * block_size);
    std::vector<int> h_results(block_size, -1);
    hshm::GpuApi::Memcpy(d_results, h_results.data(), sizeof(int) * block_size);

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    test_gpu_ipc_manager_gpu_alloc_kernel<<<1, block_size>>>(gpu_info, d_results);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      INFO("CUDA error: " << cudaGetErrorString(err));
    }
    REQUIRE(err == cudaSuccess);

    hshm::GpuApi::Memcpy(h_results.data(), d_results, sizeof(int) * block_size);
    for (int i = 0; i < block_size; ++i) {
      INFO("Thread " << i << " result: " << h_results[i]);
      REQUIRE(h_results[i] == 0);
    }

    hshm::GpuApi::Free(d_results);
  }

  SECTION("Test 2: SendGpu serialization via ring buffer") {
    INFO("Testing SendGpu with ShmTransport ring buffer + CPU Recv");

    // Create queue backend
    hipc::MemoryBackendId queue_backend_id(3, 0);
    size_t queue_memory_size = 64 * 1024 * 1024;
    hipc::GpuShmMmap queue_backend;
    REQUIRE(queue_backend.shm_init(queue_backend_id, queue_memory_size,
                                   "/gpu_queue_test2", 0));

    auto *queue_allocator = reinterpret_cast<hipc::BuddyAllocator *>(
        queue_backend.data_);
    new (queue_allocator) hipc::BuddyAllocator();
    queue_allocator->shm_init(queue_backend, queue_backend.data_capacity_);

    auto gpu_queue = queue_allocator->template NewObj<chi::TaskQueue>(
        queue_allocator, 1, 1, 256);
    REQUIRE(!gpu_queue.IsNull());

    // SendGpu uses CHI_PRIV_ALLOC for DefaultSaveArchive — needs a heap backend
    hipc::MemoryBackendId heap_id2(30, 0);
    hipc::GpuMalloc heap_backend2;
    REQUIRE(heap_backend2.shm_init(heap_id2, 4 * 1024 * 1024,
                                   "/gpu_heap_test2", 0));

    chi::IpcManagerGpu gpu_info(gpu_backend, gpu_queue.ptr_);

    int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
    int h_result_init = -999;
    hshm::GpuApi::Memcpy(d_result, &h_result_init, sizeof(int));

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    // Launch kernel async — GPU writes to ring buffer + enqueues + waits
    test_gpu_send_serialization_kernel<<<1, 1>>>(gpu_info, d_result);

    // CPU polls queue until a Future is available
    auto &lane = gpu_queue.ptr_->GetLane(0, 0);
    chi::Future<chi::Task> popped_future;
    while (!lane.Pop(popped_future)) {
      // Spin until GPU pushes the future
    }
    INFO("Popped future from queue");

    // Resolve FutureShm: SendGpu stores an absolute UVA pointer in off_.
    // (See ipc_manager.h SendGpu: fshmptr.off_ = reinterpret_cast<size_t>(buffer.ptr_))
    hipc::ShmPtr<chi::FutureShm> future_shm_ptr =
        popped_future.GetFutureShmPtr();
    REQUIRE(!future_shm_ptr.IsNull());
    chi::FutureShm *future_shm =
        reinterpret_cast<chi::FutureShm *>(future_shm_ptr.off_.load());

    // Verify FUTURE_COPY_FROM_CLIENT flag
    REQUIRE(future_shm->flags_.Any(chi::FutureShm::FUTURE_COPY_FROM_CLIENT));

    // CPU reads from ring buffer using ShmTransport::Recv with DefaultLoadArchive
    hshm::lbm::LbmContext recv_ctx;
    recv_ctx.copy_space = future_shm->copy_space;
    recv_ctx.shm_info_ = &future_shm->input_;

    chi::DefaultLoadArchive load_ar;
    hshm::lbm::ShmTransport::Recv(load_ar, recv_ctx);

    // Deserialize task and verify fields
    chimaera::MOD_NAME::GpuSubmitTask deserialized_task;
    deserialized_task.SerializeIn(load_ar);

    // Set FUTURE_COMPLETE to unblock the GPU kernel BEFORE checking results
    future_shm->flags_.SetBits(chi::FutureShm::FUTURE_COMPLETE);

    cudaError_t err = cudaDeviceSynchronize();

    int h_result = -999;
    hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));

    // Now check results after GPU is done (so GPU printf is visible)
    REQUIRE(err == cudaSuccess);
    REQUIRE(deserialized_task.gpu_id_ == 55);
    REQUIRE(deserialized_task.test_value_ == 12345);
    REQUIRE(deserialized_task.result_value_ == 0);
    REQUIRE(h_result == 0);

    hshm::GpuApi::Free(d_result);
  }

  SECTION("Test 3: GPU SendGpu + CPU process + GPU RecvGpu (round-trip)") {
    INFO("Testing full round-trip: GPU SendGpu -> CPU process -> GPU RecvGpu");

    // Create queue backend
    hipc::MemoryBackendId queue_backend_id(4, 0);
    size_t queue_memory_size = 64 * 1024 * 1024;
    hipc::GpuShmMmap queue_backend;
    REQUIRE(queue_backend.shm_init(queue_backend_id, queue_memory_size,
                                   "/gpu_queue_test4", 0));

    auto *queue_allocator = reinterpret_cast<hipc::BuddyAllocator *>(
        queue_backend.data_);
    new (queue_allocator) hipc::BuddyAllocator();
    queue_allocator->shm_init(queue_backend, queue_backend.data_capacity_);

    auto gpu_queue = queue_allocator->template NewObj<chi::TaskQueue>(
        queue_allocator, 1, 1, 256);
    REQUIRE(!gpu_queue.IsNull());

    // SendGpu uses CHI_PRIV_ALLOC for DefaultSaveArchive — needs a heap backend
    hipc::MemoryBackendId heap_id3(31, 0);
    hipc::GpuMalloc heap_backend3;
    REQUIRE(heap_backend3.shm_init(heap_id3, 4 * 1024 * 1024,
                                   "/gpu_heap_test3", 0));

    chi::IpcManagerGpu gpu_info(gpu_backend, gpu_queue.ptr_);

    int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
    int h_result_init = -999;
    hshm::GpuApi::Memcpy(d_result, &h_result_init, sizeof(int));

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    // Launch kernel async — GPU writes input to ring buffer, then blocks in RecvGpu
    test_gpu_send_recv_roundtrip_kernel<<<1, 1>>>(gpu_info, d_result);

    // CPU polls queue
    auto &lane = gpu_queue.ptr_->GetLane(0, 0);
    chi::Future<chi::Task> popped_future;
    while (!lane.Pop(popped_future)) {}

    hipc::ShmPtr<chi::FutureShm> future_shm_ptr =
        popped_future.GetFutureShmPtr();
    REQUIRE(!future_shm_ptr.IsNull());
    // SendGpu stores absolute UVA pointer in off_ — cast directly
    chi::FutureShm *future_shm =
        reinterpret_cast<chi::FutureShm *>(future_shm_ptr.off_.load());

    // CPU reads input from ring buffer
    hshm::lbm::LbmContext recv_ctx;
    recv_ctx.copy_space = future_shm->copy_space;
    recv_ctx.shm_info_ = &future_shm->input_;

    chi::DefaultLoadArchive load_ar;
    hshm::lbm::ShmTransport::Recv(load_ar, recv_ctx);

    chimaera::MOD_NAME::GpuSubmitTask deserialized_task;
    deserialized_task.SerializeIn(load_ar);

    // "Process" the task: set result_value_ = test_value_ * 2
    deserialized_task.result_value_ = deserialized_task.test_value_ * 2;

    // CPU writes output to ring buffer using ShmTransport::Send
    hshm::lbm::LbmContext send_ctx;
    send_ctx.copy_space = future_shm->copy_space;
    send_ctx.shm_info_ = &future_shm->output_;

    // Serialize output via DefaultSaveArchive + ShmTransport::Send
    chi::DefaultSaveArchive save_ar(chi::LocalMsgType::kSerializeOut);
    deserialized_task.SerializeOut(save_ar);
    hshm::lbm::ShmTransport::Send(save_ar, send_ctx);

    // Set FUTURE_COMPLETE
    future_shm->flags_.SetBits(chi::FutureShm::FUTURE_COMPLETE);

    // Wait for kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      INFO("CUDA error: " << cudaGetErrorString(err));
    }
    REQUIRE(err == cudaSuccess);

    int h_result = -999;
    hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
    INFO("GPU kernel result: " << h_result);
    REQUIRE(h_result == 0);

    hshm::GpuApi::Free(d_result);
  }
}

SIMPLE_TEST_MAIN()

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
