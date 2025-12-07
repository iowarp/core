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

/**
 * Helper function to create a MallocBackend and ArenaAllocator<false>
 * Returns the allocator pointer (caller must manage backend lifetime)
 */
hipc::ArenaAllocator<false>* CreateArenaAllocator(hipc::MallocBackend &backend) {
  // Initialize backend with space for allocator + heap
  size_t heap_size = 256 * 1024 * 1024;  // 256 MB heap
  size_t alloc_size = sizeof(hipc::ArenaAllocator<false>);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  // Construct allocator at beginning of backend
  auto *alloc = backend.Cast<hipc::ArenaAllocator<false>>();
  new (alloc) hipc::ArenaAllocator<false>();

  // Initialize allocator with backend and region_size
  alloc->shm_init(backend, heap_size);

  return alloc;
}

TEST_CASE("SubAllocator - Basic Creation and Destruction", "[SubAllocator]") {
  hipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  SECTION("Create and destroy a single sub-allocator") {
    // Create a sub-allocator with 64 MB
    size_t sub_alloc_size = 64 * 1024 * 1024;
    auto *sub_alloc = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
        sub_alloc_size, 0);

    REQUIRE(sub_alloc != nullptr);
    REQUIRE(sub_alloc->GetId() == parent_alloc->GetId());

    // Free the sub-allocator
    parent_alloc->FreeSubAllocator( sub_alloc);
  }

  SECTION("Create multiple sub-allocators with different IDs") {
    size_t sub_alloc_size = 32 * 1024 * 1024;

    auto *sub_alloc1 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
        sub_alloc_size, 0);
    auto *sub_alloc2 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
        sub_alloc_size, 0);
    auto *sub_alloc3 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
        sub_alloc_size, 0);

    REQUIRE(sub_alloc1 != nullptr);
    REQUIRE(sub_alloc2 != nullptr);
    REQUIRE(sub_alloc3 != nullptr);


    // Free all sub-allocators
    parent_alloc->FreeSubAllocator( sub_alloc1);
    parent_alloc->FreeSubAllocator( sub_alloc2);
    parent_alloc->FreeSubAllocator( sub_alloc3);
  }

  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Allocations within SubAllocator", "[SubAllocator]") {
  hipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create a sub-allocator with 64 MB
  size_t sub_alloc_size = 64 * 1024 * 1024;
  auto *sub_alloc = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc_size, 0);

  REQUIRE(sub_alloc != nullptr);

  SECTION("Allocate and free immediately") {
    

    for (size_t i = 0; i < 1000; ++i) {
      auto ptr = sub_alloc->template Allocate<void>( 1024, 64);
      REQUIRE_FALSE(ptr.IsNull());
      sub_alloc->Free( ptr);
    }
  }

  SECTION("Batch allocations") {
    
    std::vector<hipc::FullPtr<void>> ptrs;

    // Allocate batch
    for (size_t i = 0; i < 100; ++i) {
      auto ptr = sub_alloc->template Allocate<void>( 4096, 64);
      REQUIRE_FALSE(ptr.IsNull());
      ptrs.push_back(ptr);
    }

    // Free batch
    for (auto &ptr : ptrs) {
      sub_alloc->Free( ptr);
    }
  }

  // Free the sub-allocator
  parent_alloc->FreeSubAllocator( sub_alloc);

  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Random Allocation Test", "[SubAllocator]") {
  hipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create a sub-allocator with 64 MB
  size_t sub_alloc_size = 64 * 1024 * 1024;
  auto *sub_alloc = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc_size, 0);

  REQUIRE(sub_alloc != nullptr);

  // Use the AllocatorTest framework to run random tests
  AllocatorTest<hipc::ArenaAllocator<false>> tester(sub_alloc);

  SECTION("16 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(16));
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  // Free the sub-allocator
  parent_alloc->FreeSubAllocator( sub_alloc);

  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Multiple SubAllocators with Random Tests", "[SubAllocator]") {
  hipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create 3 sub-allocators, each with 32 MB
  size_t sub_alloc_size = 32 * 1024 * 1024;
  auto *sub_alloc1 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc_size, 0);
  auto *sub_alloc2 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc_size, 0);
  auto *sub_alloc3 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc_size, 0);

  REQUIRE(sub_alloc1 != nullptr);
  REQUIRE(sub_alloc2 != nullptr);
  REQUIRE(sub_alloc3 != nullptr);

  SECTION("Run random tests on all three sub-allocators") {
    AllocatorTest<hipc::ArenaAllocator<false>> tester1(sub_alloc1);
    AllocatorTest<hipc::ArenaAllocator<false>> tester2(sub_alloc2);
    AllocatorTest<hipc::ArenaAllocator<false>> tester3(sub_alloc3);

    REQUIRE_NOTHROW(tester1.TestRandomAllocation(8));
    REQUIRE_NOTHROW(tester2.TestRandomAllocation(8));
    REQUIRE_NOTHROW(tester3.TestRandomAllocation(8));
  }

  // Free all sub-allocators
  parent_alloc->FreeSubAllocator( sub_alloc1);
  parent_alloc->FreeSubAllocator( sub_alloc2);
  parent_alloc->FreeSubAllocator( sub_alloc3);

  backend.shm_destroy();
}

TEST_CASE("SubAllocator - Nested SubAllocators", "[SubAllocator][nested]") {
  hipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create a sub-allocator from parent (64 MB)
  size_t sub_alloc1_size = 64 * 1024 * 1024;
  auto *sub_alloc1 = parent_alloc->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc1_size, 0);

  REQUIRE(sub_alloc1 != nullptr);

  // Create a nested sub-allocator from the first sub-allocator (16 MB)
  size_t sub_alloc2_size = 16 * 1024 * 1024;
  auto *sub_alloc2 = sub_alloc1->CreateSubAllocator<hipc::_ArenaAllocator<false>>(
      sub_alloc2_size, 0);

  REQUIRE(sub_alloc2 != nullptr);

  // Test allocations in the nested sub-allocator
  AllocatorTest<hipc::ArenaAllocator<false>> tester(sub_alloc2);
  REQUIRE_NOTHROW(tester.TestRandomAllocation(8));

  // Free nested sub-allocator first, then parent sub-allocator
  sub_alloc1->FreeSubAllocator( sub_alloc2);
  parent_alloc->FreeSubAllocator( sub_alloc1);

  backend.shm_destroy();
}
