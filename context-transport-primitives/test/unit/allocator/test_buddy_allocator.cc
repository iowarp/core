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

#include <catch2/catch_test_macros.hpp>
#include <string>
#include "allocator_test.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"

using hshm::testing::AllocatorTest;

TEST_CASE("BuddyAllocator - Allocate and Free Immediate", "[BuddyAllocator]") {
  hipc::MallocBackend backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(hipc::BuddyAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::BuddyAllocator>();

  AllocatorTest<hipc::BuddyAllocator> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(10000, 1024));
  }

  SECTION("Medium allocations (64KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(1000, 64 * 1024));
  }

  SECTION("Large allocations (1MB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024 * 1024));
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Batch Allocate and Free", "[BuddyAllocator]") {
  hipc::MallocBackend backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(hipc::BuddyAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::BuddyAllocator>();

  AllocatorTest<hipc::BuddyAllocator> tester(alloc);

  SECTION("Small batches (10 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(1000, 10, 4096));
  }

  SECTION("Medium batches (100 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(100, 100, 4096));
  }

  SECTION("Large batches (1000 allocations of 1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(10, 1000, 1024));
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Random Allocation", "[BuddyAllocator]") {
  hipc::MallocBackend backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(hipc::BuddyAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::BuddyAllocator>();

  AllocatorTest<hipc::BuddyAllocator> tester(alloc);

  SECTION("16 iterations of random allocations"){
    try {
      tester.TestRandomAllocation(16);
    }
    catch (const std::exception &e) {
      std::cout << ("TestRandomAllocation(16) failed: " + std::string(e.what()));
    }
    catch (const hshm::Error &e) {
      std::cout << ("TestRandomAllocation(16) failed: " + std::string(e.what()));
    }
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Large Then Small", "[BuddyAllocator]") {
  hipc::MallocBackend backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(hipc::BuddyAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::BuddyAllocator>();

  AllocatorTest<hipc::BuddyAllocator> tester(alloc);

  SECTION("10 iterations: 100 x 1MB then 1000 x 128B") {
    REQUIRE_NOTHROW(tester.TestLargeThenSmall(10, 100, 1024 * 1024, 1000, 128));
  }

  SECTION("5 iterations: 50 x 512KB then 500 x 256B") {
    REQUIRE_NOTHROW(tester.TestLargeThenSmall(5, 50, 512 * 1024, 500, 256));
  }

  backend.shm_destroy();
}
