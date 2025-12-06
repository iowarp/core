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
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using hshm::testing::AllocatorTest;

TEST_CASE("ArenaAllocator<false> - Allocate and Free Immediate", "[ArenaAllocator<false>]") {
  hipc::MallocBackend backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(hipc::ArenaAllocator<false>);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<hipc::ArenaAllocator<false>>();

  AllocatorTest<hipc::ArenaAllocator<false>> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024));
  }
  
  backend.shm_destroy();
}

