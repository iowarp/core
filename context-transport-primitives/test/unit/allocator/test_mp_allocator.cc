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
#include "allocator_test.h"
#include "hermes_shm/memory/backend/posix_mmap.h"
#include "hermes_shm/memory/allocator/mp_allocator.h"

using hshm::testing::AllocatorTest;

TEST_CASE("MultiProcessAllocator - Allocate and Free Immediate", "[MultiProcessAllocator]") {
  hipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(hipc::MultiProcessAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::MultiProcessAllocator>();

  AllocatorTest<hipc::MultiProcessAllocator> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(10000, 1024));
  }

  SECTION("Medium allocations (64KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(1000, 64 * 1024));
  }

  SECTION("Large allocations (1MB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024 * 1024));
  }

  alloc->shm_detach();
  backend.shm_destroy();
}

TEST_CASE("MultiProcessAllocator - Batch Allocate and Free", "[MultiProcessAllocator]") {
  hipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(hipc::MultiProcessAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::MultiProcessAllocator>();

  AllocatorTest<hipc::MultiProcessAllocator> tester(alloc);

  SECTION("Small batches (10 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(1000, 10, 4096));
  }

  SECTION("Medium batches (100 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(100, 100, 4096));
  }

  SECTION("Large batches (1000 allocations of 1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(10, 1000, 1024));
  }

  alloc->shm_detach();
  backend.shm_destroy();
}

TEST_CASE("MultiProcessAllocator - Random Allocation", "[MultiProcessAllocator]") {
  hipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(hipc::MultiProcessAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::MultiProcessAllocator>();

  AllocatorTest<hipc::MultiProcessAllocator> tester(alloc);

  SECTION("16 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(16));
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  alloc->shm_detach();
  backend.shm_destroy();
}

TEST_CASE("MultiProcessAllocator - Multi-threaded Random", "[MultiProcessAllocator][multithread]") {
  hipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(hipc::MultiProcessAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::MultiProcessAllocator>();

  AllocatorTest<hipc::MultiProcessAllocator> tester(alloc);

  SECTION("8 threads, 2 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(8, 2));
  }

  SECTION("4 threads, 4 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(4, 256));
  }

  alloc->shm_detach();
  backend.shm_destroy();
}
