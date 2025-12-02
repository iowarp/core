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

#ifndef HSHM_TEST_UNIT_ALLOCATOR_ALLOCATOR_TEST_H_
#define HSHM_TEST_UNIT_ALLOCATOR_ALLOCATOR_TEST_H_

#include <random>
#include <thread>
#include <vector>
#include "hermes_shm/memory/allocator/allocator.h"

namespace hshm::testing {

/**
 * Templated allocator test class
 * Tests all allocator APIs for a given allocator type
 */
template<typename AllocT>
class AllocatorTest {
 private:
  AllocT *alloc_;
  std::mt19937 rng_;

 public:
  /**
   * Constructor
   * @param alloc The allocator to test
   */
  explicit AllocatorTest(AllocT *alloc)
    : alloc_(alloc), rng_(std::random_device{}()) {}

  /**
   * Test 1: Allocate and free immediately in a loop
   * Same memory size for each allocation
   *
   * @param iterations Number of iterations
   * @param alloc_size Size of each allocation
   */
  void TestAllocFreeImmediate(size_t iterations, size_t alloc_size) {
    for (size_t i = 0; i < iterations; ++i) {
      auto ptr = alloc_->template AlignedAllocate<void>(alloc_size, 64);
      if (ptr.IsNull()) {
        throw std::runtime_error("Allocation failed in TestAllocFreeImmediate");
      }
      alloc_->Free(ptr);
    }
  }

  /**
   * Test 2: Allocate a bunch, then free the bunch
   * Iteratively in a loop. Same memory size per alloc
   *
   * @param iterations Number of iterations
   * @param batch_size Number of allocations per batch
   * @param alloc_size Size of each allocation
   */
  void TestAllocFreeBatch(size_t iterations, size_t batch_size, size_t alloc_size) {
    std::vector<hipc::FullPtr<void>> ptrs;
    ptrs.reserve(batch_size);

    for (size_t iter = 0; iter < iterations; ++iter) {
      // Allocate batch
      for (size_t i = 0; i < batch_size; ++i) {
        auto ptr = alloc_->template AlignedAllocate<void>(alloc_size, 64);
        if (ptr.IsNull()) {
          // Clean up already allocated pointers
          for (auto &p : ptrs) {
            alloc_->Free(p);
          }
          throw std::runtime_error("Allocation failed in TestAllocFreeBatch");
        }
        ptrs.push_back(ptr);
      }

      // Free batch
      for (auto &ptr : ptrs) {
        alloc_->Free(ptr);
      }
      ptrs.clear();
    }
  }

  /**
   * Test 3: Random allocation with random sizes
   * Random sizes between 0 and 1MB
   * Up to a total of 64MB or 5000 allocations
   * After all allocations, free. Do this iteratively.
   *
   * @param iterations Number of iterations
   */
  void TestRandomAllocation(size_t iterations) {
    const size_t kMaxAllocSize = 1024 * 1024;  // 1 MB
    const size_t kMaxTotalSize = 64 * 1024 * 1024;  // 64 MB
    const size_t kMaxAllocations = 5000;

    std::uniform_int_distribution<size_t> size_dist(1, kMaxAllocSize);
    std::vector<hipc::FullPtr<void>> ptrs;
    ptrs.reserve(kMaxAllocations);

    for (size_t iter = 0; iter < iterations; ++iter) {
      size_t total_allocated = 0;

      // Random allocations
      while (total_allocated < kMaxTotalSize && ptrs.size() < kMaxAllocations) {
        size_t alloc_size = size_dist(rng_);

        // Stop if this allocation would exceed the limit
        if (total_allocated + alloc_size > kMaxTotalSize) {
          break;
        }

        auto ptr = alloc_->template AlignedAllocate<void>(alloc_size, 64);
        if (ptr.IsNull()) {
          // Allocation failed - clean up and break
          break;
        }

        ptrs.push_back(ptr);
        total_allocated += alloc_size;
      }

      // Free all allocations
      for (auto &ptr : ptrs) {
        alloc_->Free(ptr);
      }
      ptrs.clear();
    }
  }

  /**
   * Test 4: Multi-threaded random allocation test
   * 8 threads calling the random allocation test
   *
   * @param num_threads Number of threads to spawn
   * @param iterations_per_thread Number of iterations per thread
   */
  void TestMultiThreadedRandom(size_t num_threads, size_t iterations_per_thread) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back([this, iterations_per_thread]() {
        TestRandomAllocation(iterations_per_thread);
      });
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }
  }

  /**
   * Run all tests with default parameters
   */
  void RunAllTests() {
    // Test 1: Allocate and free immediately
    TestAllocFreeImmediate(10000, 1024);

    // Test 2: Batch allocations
    TestAllocFreeBatch(100, 100, 4096);

    // Test 3: Random allocations
    TestRandomAllocation(16);

    // Test 4: Multi-threaded
    TestMultiThreadedRandom(8, 2);
  }
};

}  // namespace hshm::testing

#endif  // HSHM_TEST_UNIT_ALLOCATOR_ALLOCATOR_TEST_H_
