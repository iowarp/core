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

#include "../../../context-runtime/test/simple_test.h"
#include "hermes_shm/data_structures/ipc/ring_buffer.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using namespace hshm::ipc;

/**
 * Helper function to create an ArenaAllocator for testing
 */
ArenaAllocator<false>* CreateTestAllocator(MallocBackend &backend,
                                            size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<ArenaAllocator<false>>();
}

// ============================================================================
// Extensible Ring Buffer Tests (Dynamic Size)
// ============================================================================

TEST_CASE("Extensible RingBuffer: basic operations", "[ring_buffer][ext]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ext_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Fill and drain within capacity
  for (int i = 0; i < 16; ++i) {
    REQUIRE(rb.Push(i));
  }

  for (int i = 0; i < 16; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  REQUIRE(rb.Empty());

  backend.shm_destroy();
}

TEST_CASE("Extensible RingBuffer: multiple cycles", "[ring_buffer][ext]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ext_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Run multiple fill/drain cycles
  for (int cycle = 0; cycle < 3; ++cycle) {
    for (int i = 0; i < 10; ++i) {
      REQUIRE(rb.Push(cycle * 100 + i));
    }

    for (int i = 0; i < 10; ++i) {
      int val;
      REQUIRE(rb.Pop(val));
      REQUIRE(val == cycle * 100 + i);
    }

    REQUIRE(rb.Empty());
  }

  backend.shm_destroy();
}

TEST_CASE("Extensible RingBuffer: partial cycles", "[ring_buffer][ext]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ext_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Push some items
  for (int i = 0; i < 12; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Pop half
  for (int i = 0; i < 6; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  // Verify size
  REQUIRE(rb.Size() == 6);

  // Push more
  for (int i = 12; i < 18; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Pop all
  for (int i = 6; i < 18; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  REQUIRE(rb.Empty());

  backend.shm_destroy();
}

TEST_CASE("Extensible RingBuffer: FIFO ordering", "[ring_buffer][ext]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ext_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 8);

  // Verify strict FIFO ordering through multiple push/pop cycles
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 8; ++i) {
      REQUIRE(rb.Push(round * 1000 + i));
    }

    for (int i = 0; i < 8; ++i) {
      int val;
      REQUIRE(rb.Pop(val));
      REQUIRE(val == round * 1000 + i);
    }
  }

  backend.shm_destroy();
}

SIMPLE_TEST_MAIN()
