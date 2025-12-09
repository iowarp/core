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
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

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
// MPSC Ring Buffer Tests (Multiple Producer Single Consumer)
// ============================================================================

TEST_CASE("MPSC RingBuffer: single producer baseline", "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 64);

  // Baseline: single "producer" pushes 32 items
  for (int i = 0; i < 32; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Consumer pops all items
  for (int i = 0; i < 32; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  REQUIRE(rb.Empty());

  backend.shm_destroy();
}

TEST_CASE("MPSC RingBuffer: concurrent producers", "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 128);

  // 4 producer threads, each pushing 25 items
  std::vector<std::thread> producers;
  std::atomic<int> push_count(0);

  for (int producer_id = 0; producer_id < 4; ++producer_id) {
    producers.emplace_back([&rb, &push_count, producer_id]() {
      for (int i = 0; i < 25; ++i) {
        int value = producer_id * 1000 + i;
        if (rb.Push(value)) {
          push_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Wait for all producers to complete
  for (auto &t : producers) {
    t.join();
  }

  // Verify that all 100 items were pushed
  REQUIRE(push_count.load() == 100);

  // Consumer pops all items (order may vary due to concurrent access)
  std::vector<int> popped_values;
  int val;
  while (rb.Pop(val)) {
    popped_values.push_back(val);
  }

  REQUIRE(popped_values.size() == 100);
  REQUIRE(rb.Empty());

  backend.shm_destroy();
}

TEST_CASE("MPSC RingBuffer: producer/consumer coordination",
          "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 32);

  std::atomic<bool> producer_done(false);
  std::atomic<int> consumed_count(0);

  // Producer thread: continuously push items until stopped
  std::thread producer([&rb, &producer_done]() {
    for (int i = 0; i < 100; ++i) {
      rb.Push(i);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    producer_done.store(true, std::memory_order_release);
  });

  // Consumer thread: continuously pop items until producer is done
  std::thread consumer([&rb, &producer_done, &consumed_count]() {
    while (true) {
      int val;
      if (rb.Pop(val)) {
        consumed_count.fetch_add(1, std::memory_order_relaxed);
      } else if (producer_done.load(std::memory_order_acquire)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  producer.join();
  consumer.join();

  // Verify all items were consumed
  REQUIRE(consumed_count.load() == 100);
  REQUIRE(rb.Empty());

  backend.shm_destroy();
}

TEST_CASE("MPSC RingBuffer: contention under capacity limit",
          "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  // Small buffer to induce contention
  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  std::atomic<int> successful_pushes(0);
  std::atomic<int> failed_pushes(0);

  // 3 producer threads try to push as many items as possible
  std::vector<std::thread> producers;
  for (int producer_id = 0; producer_id < 3; ++producer_id) {
    producers.emplace_back(
        [&rb, &successful_pushes, &failed_pushes, producer_id]() {
          for (int i = 0; i < 50; ++i) {
            int value = producer_id * 1000 + i;
            if (rb.Push(value)) {
              successful_pushes.fetch_add(1, std::memory_order_relaxed);
            } else {
              failed_pushes.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
          }
        });
  }

  for (auto &t : producers) {
    t.join();
  }

  // With ERROR_ON_NO_SPACE flag, some pushes should fail
  int total_attempts = 3 * 50;
  int total_succeeded = successful_pushes.load();
  int total_failed = failed_pushes.load();

  REQUIRE(total_succeeded + total_failed == total_attempts);
  REQUIRE(total_succeeded <= 16);  // At most capacity items in buffer

  backend.shm_destroy();
}

TEST_CASE("MPSC RingBuffer: stress test with varying producer count",
          "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  // Larger buffer for stress test to accommodate all producers
  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 512);

  std::atomic<int> total_pushed(0);
  std::vector<std::thread> producers;

  // Launch 8 producer threads pushing in parallel
  for (int producer_id = 0; producer_id < 8; ++producer_id) {
    producers.emplace_back([&rb, &total_pushed, producer_id]() {
      for (int i = 0; i < 50; ++i) {
        int value = producer_id * 1000 + i;
        if (rb.Push(value)) {
          total_pushed.fetch_add(1, std::memory_order_relaxed);
        } else {
          // Retry on failure
          std::this_thread::sleep_for(std::chrono::microseconds(10));
          if (rb.Push(value)) {
            total_pushed.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }

  // Wait for all producers
  for (auto &t : producers) {
    t.join();
  }

  // Consumer drains the entire buffer
  int consumed = 0;
  int val;
  while (rb.Pop(val)) {
    consumed++;
  }

  // Verify all pushes eventually succeeded
  REQUIRE(total_pushed.load() == 400);
  REQUIRE(consumed == 400);
  REQUIRE(rb.Empty());

  backend.shm_destroy();
}

SIMPLE_TEST_MAIN()
