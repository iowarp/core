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

TEST_CASE("ArenaAllocator - Basic Allocation", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;  // 1 MB
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<false> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  SECTION("Single allocation") {
    

    auto ptr = alloc->AllocateOffset( 100);
    REQUIRE_FALSE(ptr.IsNull());
    REQUIRE(ptr.off_.load() == 0);  // First allocation at offset 0
    REQUIRE(alloc->GetHeapOffset() == 100);
  }

  SECTION("Multiple allocations") {
    

    auto ptr1 = alloc->AllocateOffset( 100);
    auto ptr2 = alloc->AllocateOffset( 200);
    auto ptr3 = alloc->AllocateOffset( 300);

    REQUIRE(ptr1.off_.load() == 0);
    REQUIRE(ptr2.off_.load() == 100);
    REQUIRE(ptr3.off_.load() == 300);
    REQUIRE(alloc->GetHeapOffset() == 600);
  }

  // Note: Allocation tracking requires HSHM_ALLOC_TRACK_SIZE to be defined

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Aligned Allocation", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<false> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  SECTION("Aligned allocations") {
    

    // Allocate 100 bytes aligned to 64
    auto ptr1 = alloc->AllocateOffset( 100, 64);
    REQUIRE(ptr1.off_.load() % 64 == 0);

    // Next allocation should also be 64-byte aligned
    auto ptr2 = alloc->AllocateOffset( 50, 64);
    REQUIRE(ptr2.off_.load() % 64 == 0);
  }

  SECTION("Mixed alignment") {
    

    auto ptr1 = alloc->AllocateOffset( 1);  // 1 byte
    auto ptr2 = alloc->AllocateOffset( 1, 64);  // Align to 64

    REQUIRE(ptr1.off_.load() == 0);
    REQUIRE(ptr2.off_.load() == 64);  // Should skip to next 64-byte boundary
  }

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Reset", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<false> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  

  // Allocate some memory
  alloc->AllocateOffset( 100);
  alloc->AllocateOffset( 200);
  alloc->AllocateOffset( 300);

  REQUIRE(alloc->GetHeapOffset() == 600);

  // Reset the arena
  alloc->Reset();

  REQUIRE(alloc->GetHeapOffset() == 0);

  // Allocate again - should start from offset 0
  auto ptr = alloc->AllocateOffset( 50);
  REQUIRE(ptr.off_.load() == 0);
  REQUIRE(alloc->GetHeapOffset() == 50);

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Out of Memory", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024;  // Small arena - 1 KB
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<false> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  

  // Allocate most of the arena
  alloc->AllocateOffset( 512);
  alloc->AllocateOffset( 256);

  // This allocation should succeed (768 + 200 = 968 < 1024)
  REQUIRE_NOTHROW(alloc->AllocateOffset( 200));

  // This allocation should fail (968 + 100 = 1068 > 1024)
  REQUIRE_THROWS(alloc->AllocateOffset( 100));

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Free is No-op", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<false> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  

  auto ptr1 = alloc->Allocate<int>( 10);
  auto ptr2 = alloc->Allocate<int>( 20);

  size_t heap_before = alloc->GetHeapOffset();

  // Free should be a no-op
  REQUIRE_NOTHROW(alloc->Free( ptr1));
  REQUIRE_NOTHROW(alloc->Free( ptr2));

  // Heap offset should not change
  REQUIRE(alloc->GetHeapOffset() == heap_before);

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Remaining Space", "[ArenaAllocator]") {
  hipc::MallocBackend backend;
  size_t test_arena_size = 1000;
  backend.shm_init(hipc::MemoryBackendId(0, 0), test_arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<false> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, test_arena_size, backend);

  

  REQUIRE(alloc->GetRemainingSize() == test_arena_size);

  alloc->AllocateOffset( 300);
  REQUIRE(alloc->GetRemainingSize() == 700);

  alloc->AllocateOffset( 200);
  REQUIRE(alloc->GetRemainingSize() == 500);

  alloc->Reset();
  REQUIRE(alloc->GetRemainingSize() == test_arena_size);

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator - Atomic Version", "[ArenaAllocator][atomic]") {
  hipc::MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  backend.shm_init(hipc::MemoryBackendId(0, 0), arena_size);

  // ArenaAllocator allocates its header with malloc, not in the backend
  hipc::ArenaAllocator<true> alloc_obj;
  auto *alloc = &alloc_obj;
  alloc->shm_init(hipc::AllocatorId(hipc::MemoryBackendId(0, 0), 0), 0, arena_size, backend);

  SECTION("Basic atomic allocations") {
    

    auto ptr1 = alloc->AllocateOffset( 100);
    auto ptr2 = alloc->AllocateOffset( 200);

    REQUIRE(ptr1.off_.load() == 0);
    REQUIRE(ptr2.off_.load() == 100);
    REQUIRE(alloc->GetHeapOffset() == 300);
  }

  SECTION("Atomic reset") {
    

    alloc->AllocateOffset( 500);
    REQUIRE(alloc->GetHeapOffset() == 500);

    alloc->Reset();
    REQUIRE(alloc->GetHeapOffset() == 0);
  }

  backend.shm_destroy();
}

// Note: Type allocation tests are skipped because ArenaAllocator with MallocBackend
// doesn't provide a real memory buffer (MallocBackend has data_=nullptr).
// ArenaAllocator is designed to work with backends that provide actual buffers
// (like PosixShmMmap or ArrayBackend from sub-allocators).
