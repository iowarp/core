/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <catch2/catch_all.hpp>

#include "hermes_shm/data_structures/ipc/ring_buffer.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/util/gpu_api.h"

using hshm::ipc::ArenaAllocator;
using hshm::ipc::GpuMalloc;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendId;
using hshm::ipc::mpsc_ring_buffer;

/**
 * GPU kernel to create and initialize an allocator
 *
 * @tparam AllocT The allocator type to create
 * @param backend Pointer to the memory backend (must be in GPU-accessible
 * memory)
 * @param result Output pointer to store the created allocator pointer
 */
template <typename AllocT>
__global__ void MakeAllocKernel(MemoryBackend *backend, AllocT **result) {
  // Use MakeAlloc to create and initialize the allocator
  *result = backend->MakeAlloc<AllocT>();
  printf("MakeAllocKernel: Allocator created: %p\n", *result);
}

/**
 * GPU kernel to allocate and construct a ring buffer from the allocator
 *
 * @tparam AllocT The allocator type
 * @tparam T The ring buffer element type
 * @param alloc Pointer to the allocator
 * @param depth Depth (capacity + 1) of the ring buffer
 * @param result Output pointer to store the allocated ring buffer pointer
 */
template <typename AllocT, typename T>
__global__ void AllocateRingBufferKernel(AllocT *alloc, size_t capacity,
                                         mpsc_ring_buffer<T, AllocT> **result) {
  // Use NewObj to allocate and construct the ring buffer
  auto ring_ptr =
      alloc->template NewObj<mpsc_ring_buffer<T, AllocT>>(alloc, capacity);
  printf("AllocateRingBufferKernel: Ring buffer created: %p %p\n", alloc,
         ring_ptr.ptr_);
  // Return the ring buffer pointer
  *result = ring_ptr.ptr_;
}

/**
 * GPU kernel to push elements onto ring buffer
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type
 * @param ring Pointer to the ring buffer
 * @param values Array of values to push
 * @param count Number of elements to push
 */
template <typename T, typename AllocT>
__global__ void PushElementsKernel(mpsc_ring_buffer<T, AllocT> *ring, T *values,
                                   size_t count) {
  for (size_t i = 0; i < count; ++i) {
    ring->Emplace(values[i]);
  }
}

/**
 * GPU kernel to pop elements from ring buffer and verify values
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type
 * @param ring Pointer to the ring buffer
 * @param output Array to store popped values
 * @param count Number of elements to pop
 * @param success Output flag - set to 1 if all pops succeeded, 0 otherwise
 */
template <typename T, typename AllocT>
__global__ void PopElementsKernel(mpsc_ring_buffer<T, AllocT> *ring, T *output,
                                  size_t count, int *success) {
  *success = 1;  // Assume success
  for (size_t i = 0; i < count; ++i) {
    T value;
    bool popped = ring->Pop(value);
    if (!popped) {
      *success = 0;  // Pop failed
      return;
    }
    output[i] = value;
  }
}

/**
 * Test GpuMalloc backend with ring buffer
 *
 * Steps:
 * 1. Create a GpuMalloc backend
 * 2. Create an allocator on that backend
 * 3. Allocate a ring_buffer on that backend
 * 4. Pass the ring_buffer to the kernel
 * 5. Verify that we can place 10 elements on the ring buffer
 * 6. Verify the runtime can pop the 10 elements
 */
TEST_CASE("GpuMalloc", "[gpu][backend]") {
  constexpr size_t kDataSize = 64 * 1024 * 1024;  // 64MB
  constexpr size_t kNumElements = 10;
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_gpu_malloc";

  SECTION("RingBufferGpuAccess") {
    // Step 1: Create a GpuMalloc backend
    GpuMalloc backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success = backend.shm_init(backend_id, kDataSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an allocator on that backend (using GPU kernel)
    using AllocT = hipc::BuddyAllocator;
    AllocT **alloc_result_dev;
    cudaMalloc(&alloc_result_dev, sizeof(AllocT *));

    MemoryBackend *backend_dev;
    cudaMalloc(&backend_dev, sizeof(GpuMalloc));
    cudaMemcpy(backend_dev, &backend, sizeof(GpuMalloc),
               cudaMemcpyHostToDevice);

    MakeAllocKernel<AllocT><<<1, 1>>>(backend_dev, alloc_result_dev);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK(cudaGetLastError());

    AllocT *alloc_ptr;
    cudaMemcpy(&alloc_ptr, alloc_result_dev, sizeof(AllocT *),
               cudaMemcpyDeviceToHost);
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate a ring_buffer on that backend (using GPU kernel)
    using RingBuffer = mpsc_ring_buffer<int, AllocT>;
    RingBuffer **ring_result_dev;
    cudaMalloc(&ring_result_dev, sizeof(RingBuffer *));

    AllocateRingBufferKernel<AllocT, int>
        <<<1, 1>>>(alloc_ptr, kNumElements, ring_result_dev);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK(cudaGetLastError());

    RingBuffer *ring_ptr;
    cudaMemcpy(&ring_ptr, ring_result_dev, sizeof(RingBuffer *),
               cudaMemcpyDeviceToHost);
    REQUIRE(ring_ptr != nullptr);

    // Step 4 & 5: Pass the ring_buffer to the kernel and push 10 elements
    // Prepare values to push on the host
    int host_values[kNumElements];
    for (size_t i = 0; i < kNumElements; ++i) {
      host_values[i] = static_cast<int>(i);
    }

    // Copy values to GPU
    int *dev_values;
    cudaMalloc(&dev_values, kNumElements * sizeof(int));
    cudaMemcpy(dev_values, host_values, kNumElements * sizeof(int),
               cudaMemcpyHostToDevice);

    // Launch kernel to push elements
    PushElementsKernel<int, AllocT>
        <<<1, 1>>>(ring_ptr, dev_values, kNumElements);
    cudaDeviceSynchronize();

    // Step 6: Verify the runtime can pop the 10 elements
    // Allocate device memory for output values and success flag
    int *dev_output;
    cudaMalloc(&dev_output, kNumElements * sizeof(int));

    int *dev_success;
    cudaMalloc(&dev_success, sizeof(int));

    // Launch kernel to pop elements from ring buffer
    PopElementsKernel<int, AllocT>
        <<<1, 1>>>(ring_ptr, dev_output, kNumElements, dev_success);
    cudaDeviceSynchronize();

    // Copy results back to host
    int host_output[kNumElements];
    int success_flag;
    cudaMemcpy(host_output, dev_output, kNumElements * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&success_flag, dev_success, sizeof(int), cudaMemcpyDeviceToHost);

    // Verify all pops succeeded
    REQUIRE(success_flag == 1);

    // Verify the popped values match what we pushed
    for (size_t i = 0; i < kNumElements; ++i) {
      REQUIRE(host_output[i] == host_values[i]);
    }

    // Cleanup GPU temporary allocations
    cudaFree(dev_success);
    cudaFree(dev_output);
    cudaFree(dev_values);
    cudaFree(ring_result_dev);
    cudaFree(alloc_result_dev);
    cudaFree(backend_dev);

    // Backend cleanup handled automatically by destructor
  }
}
