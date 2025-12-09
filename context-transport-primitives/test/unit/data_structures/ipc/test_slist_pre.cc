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
#include <vector>
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using namespace hshm::ipc;

/**
 * Test node structure that embeds slist_node
 */
struct TestNode {
  pre::slist_node link_;  // List linkage
  int value_;             // Test data

  TestNode() : value_(0) {}
  explicit TestNode(int val) : value_(val) {}
};

/**
 * Helper function to create an ArenaAllocator for testing
 */
template<bool ATOMIC>
ArenaAllocator<ATOMIC>* CreateTestAllocator(MallocBackend &backend, size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);

  // Use MakeAlloc to create the allocator properly
  return backend.MakeAlloc<ArenaAllocator<ATOMIC>>();
}

TEST_CASE("slist_pre - Basic Operations", "[slist_pre]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;  // 1 MB
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("Initialization") {
    pre::slist<false> list;
    list.Init();

    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());
    REQUIRE(list.GetHead().IsNull());
  }

  SECTION("Single element emplace and pop") {
    pre::slist<false> list;
    list.Init();

    // Allocate a test node
    auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
    node_ptr.ptr_->value_ = 42;

    // Emplace the node (cast to slist_node*)
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
    FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    REQUIRE(list.size() == 1);
    REQUIRE_FALSE(list.empty());
    REQUIRE_FALSE(list.GetHead().IsNull());

    // Pop the node
    auto popped = list.pop(alloc);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());

    // Verify the data
    auto popped_node = reinterpret_cast<TestNode*>(popped.ptr_);
    REQUIRE(popped_node->value_ == 42);
  }

  SECTION("Multiple elements - LIFO order") {
    pre::slist<false> list;
    list.Init();

    // Allocate and emplace 5 nodes
    const int NUM_NODES = 5;
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;

      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    REQUIRE(list.size() == NUM_NODES);

    // Pop all nodes - should come out in reverse order (LIFO)
    for (int i = NUM_NODES - 1; i >= 0; --i) {
      auto popped = list.pop(alloc);
      REQUIRE_FALSE(popped.IsNull());

      auto popped_node = reinterpret_cast<TestNode*>(popped.ptr_);
      REQUIRE(popped_node->value_ == i);
    }

    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());
  }

  SECTION("Pop from empty list") {
    pre::slist<false> list;
    list.Init();

    auto popped = list.pop(alloc);
    REQUIRE(popped.IsNull());
    REQUIRE(list.size() == 0);
  }

  SECTION("Peek operation") {
    pre::slist<false> list;
    list.Init();

    // Peek empty list
    auto peeked = list.peek(alloc);
    REQUIRE(peeked.IsNull());

    // Add a node
    auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
    node_ptr.ptr_->value_ = 100;
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    // Peek should return the head without removing it
    peeked = list.peek(alloc);
    REQUIRE_FALSE(peeked.IsNull());
    REQUIRE(list.size() == 1);  // Size unchanged

    auto peeked_node = reinterpret_cast<TestNode*>(peeked.ptr_);
    REQUIRE(peeked_node->value_ == 100);
  }

  SECTION("Interleaved emplace and pop") {
    pre::slist<false> list;
    list.Init();

    // Emplace 3 nodes
    for (int i = 0; i < 3; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }
    REQUIRE(list.size() == 3);

    // Pop 2 nodes
    list.pop(alloc);
    list.pop(alloc);
    REQUIRE(list.size() == 1);

    // Emplace 2 more nodes
    for (int i = 10; i < 12; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }
    REQUIRE(list.size() == 3);

    // Pop all and verify order
    auto n1 = list.pop(alloc);
    auto n2 = list.pop(alloc);
    auto n3 = list.pop(alloc);

    REQUIRE(reinterpret_cast<TestNode*>(n1.ptr_)->value_ == 11);
    REQUIRE(reinterpret_cast<TestNode*>(n2.ptr_)->value_ == 10);
    REQUIRE(reinterpret_cast<TestNode*>(n3.ptr_)->value_ == 0);
  }

  backend.shm_destroy();
}

TEST_CASE("slist_pre - Atomic Version", "[slist_pre][atomic]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  auto *alloc = CreateTestAllocator<true>(backend, arena_size);
  

  SECTION("Basic atomic operations") {
    pre::slist<true> list;
    list.Init();

    // Allocate and emplace nodes
    const int NUM_NODES = 10;
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    REQUIRE(list.size() == NUM_NODES);

    // Pop all nodes
    for (int i = NUM_NODES - 1; i >= 0; --i) {
      auto popped = list.pop(alloc);
      REQUIRE_FALSE(popped.IsNull());
      REQUIRE(reinterpret_cast<TestNode*>(popped.ptr_)->value_ == i);
    }

    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());
  }

  backend.shm_destroy();
}

TEST_CASE("slist_pre - Node Reuse", "[slist_pre]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("Reuse popped nodes") {
    pre::slist<false> list;
    list.Init();

    // Allocate a node
    auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
    node_ptr.ptr_->value_ = 1;

    // Emplace, pop, modify, and re-emplace
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    auto popped = list.pop(alloc);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(list.size() == 0);

    // Modify the node
    auto reused_node = reinterpret_cast<TestNode*>(popped.ptr_);
    reused_node->value_ = 999;

    // Re-emplace the same node
    list.emplace(alloc, popped);
    REQUIRE(list.size() == 1);

    // Pop and verify
    auto final = list.pop(alloc);
    REQUIRE(reinterpret_cast<TestNode*>(final.ptr_)->value_ == 999);
  }

  backend.shm_destroy();
}

TEST_CASE("slist_pre - Large List", "[slist_pre]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;  // 10 MB for large test
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);


  SECTION("1000 elements") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 1000;

    // Emplace many nodes
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    REQUIRE(list.size() == NUM_NODES);

    // Pop and verify all
    int count = 0;
    while (!list.empty()) {
      auto popped = list.pop(alloc);
      REQUIRE_FALSE(popped.IsNull());
      count++;
    }

    REQUIRE(count == NUM_NODES);
    REQUIRE(list.size() == 0);
  }

  backend.shm_destroy();
}

TEST_CASE("slist_pre - Iterator Forward Traversal", "[slist_pre][iterator]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;  // 1 MB
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);


  SECTION("Empty list iteration") {
    pre::slist<false> list;
    list.Init();

    auto it = list.begin(alloc);
    REQUIRE(it == list.end());
    REQUIRE(it.IsNull());
  }

  SECTION("Single element iteration") {
    pre::slist<false> list;
    list.Init();

    // Allocate and emplace a single node
    auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
    node_ptr.ptr_->value_ = 42;
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
    FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    // Iterate and verify
    auto it = list.begin(alloc);
    REQUIRE_FALSE(it.IsNull());
    REQUIRE(it != list.end());

    auto node = FullPtr<TestNode>(alloc, OffsetPtr<TestNode>(it.GetCurrent().load()));
    REQUIRE(node.ptr_->value_ == 42);

    // Advance to next (should be end)
    auto next_it_copy = it;
    ++next_it_copy;
    REQUIRE(next_it_copy.IsNull());
    REQUIRE(next_it_copy == list.end());
  }

  SECTION("Multiple elements forward iteration") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 5;
    // Emplace nodes in order 0, 1, 2, 3, 4 (but LIFO means list order is 4, 3, 2, 1, 0)
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    // Iterate through list using operator++()
    int expected_values[] = {4, 3, 2, 1, 0};  // LIFO order
    int idx = 0;
    for (auto it = list.begin(alloc); it != list.end(); ++it) {
      auto node = FullPtr<TestNode>(alloc, OffsetPtr<TestNode>(it.GetCurrent().load()));
      REQUIRE(node.ptr_->value_ == expected_values[idx]);
      idx++;
    }
    REQUIRE(idx == NUM_NODES);
  }

  SECTION("Iterator equality comparison") {
    pre::slist<false> list;
    list.Init();

    // Allocate and emplace two nodes
    for (int i = 0; i < 2; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    auto it1 = list.begin(alloc);
    auto it2 = list.begin(alloc);
    REQUIRE(it1 == it2);

    auto it_next_copy = it1;
    ++it_next_copy;
    REQUIRE(it1 != it_next_copy);
    REQUIRE(it1 != list.end());
    REQUIRE(it_next_copy != list.end());
  }

  SECTION("Iterator position tracking") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 3;
    // Emplace nodes
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    // Test IsAtHead for first iterator
    auto it = list.begin(alloc);
    REQUIRE(it.IsAtHead());

    // Test non-head positions
    auto it_next_copy = it;
    ++it_next_copy;
    REQUIRE_FALSE(it_next_copy.IsAtHead());
    REQUIRE_FALSE(it_next_copy.GetPrev().IsNull());

    // Move to third element
    auto it_next_next_copy = it_next_copy;
    ++it_next_next_copy;
    REQUIRE_FALSE(it_next_next_copy.IsAtHead());
    REQUIRE_FALSE(it_next_next_copy.GetPrev().IsNull());
  }

  SECTION("Empty list loop behavior") {
    pre::slist<false> list;
    list.Init();

    // Verify loop with empty list doesn't execute
    int count = 0;
    for (auto it = list.begin(alloc); it != list.end(); ++it) {
      count++;
    }
    REQUIRE(count == 0);
  }

  SECTION("Single element Next() returns null") {
    pre::slist<false> list;
    list.Init();

    // Allocate and emplace a single node
    auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
    node_ptr.ptr_->value_ = 100;
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
    FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    auto it = list.begin(alloc);
    REQUIRE_FALSE(it.IsNull());

    // Increment on last element should result in null iterator
    auto next_it_copy = it;
    ++next_it_copy;
    REQUIRE(next_it_copy.IsNull());
    REQUIRE(next_it_copy == list.end());

    // Verify calling ++ on null iterator returns null
    auto null_next_copy = next_it_copy;
    ++null_next_copy;
    REQUIRE(null_next_copy.IsNull());
  }

  SECTION("Iterator traversal with exact count") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 10;
    // Emplace nodes
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    // Count elements during traversal
    int count = 0;
    for (auto it = list.begin(alloc); it != list.end(); ++it) {
      count++;
      REQUIRE_FALSE(it.IsNull());
      REQUIRE_FALSE(it.GetCurrent().IsNull());
    }

    REQUIRE(count == NUM_NODES);
  }

  SECTION("Iterator GetCurrent() consistency") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 5;
    // Emplace nodes with distinct values
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i * 10;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    // Verify GetCurrent() returns correct offset at each position
    std::vector<size_t> offsets;
    for (auto it = list.begin(alloc); it != list.end(); ++it) {
      size_t offset = it.GetCurrent().load();
      REQUIRE(offset != 0);  // Should not be null offset

      // Verify no duplicate offsets (each node has unique offset)
      for (size_t prev_offset : offsets) {
        REQUIRE(offset != prev_offset);
      }
      offsets.push_back(offset);
    }
    REQUIRE(offsets.size() == NUM_NODES);
  }

  SECTION("Iterator comparison - null iterators") {
    pre::slist<false> list;
    list.Init();

    auto it1 = list.end();
    auto it2 = list.end();

    // Two null iterators should be equal
    REQUIRE(it1 == it2);
    REQUIRE_FALSE(it1 != it2);

    // Null iterator should equal default constructed iterator
    pre::slist<false>::Iterator it3;
    REQUIRE(it1 == it3);
  }

  SECTION("Iterator comparison - null vs non-null") {
    pre::slist<false> list;
    list.Init();

    // Add a node
    auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
    node_ptr.ptr_->value_ = 1;
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
    FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    auto it_begin = list.begin(alloc);
    auto it_end = list.end();

    // Non-null iterator should not equal null iterator
    REQUIRE(it_begin != it_end);
    REQUIRE_FALSE(it_begin == it_end);
  }

  SECTION("Large list iteration - 1000 elements") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 1000;

    // Emplace many nodes
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    // Iterate and verify all elements are visited
    int count = 0;
    int expected_value = NUM_NODES - 1;  // LIFO order
    for (auto it = list.begin(alloc); it != list.end(); ++it) {
      auto node = FullPtr<TestNode>(alloc, OffsetPtr<TestNode>(it.GetCurrent().load()));
      REQUIRE(node.ptr_->value_ == expected_value);
      expected_value--;
      count++;
    }

    REQUIRE(count == NUM_NODES);
    REQUIRE(expected_value == -1);  // All values visited
  }

  SECTION("Iterator IsAtHead tracking through iteration") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 5;
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    int position = 0;
    for (auto it = list.begin(alloc); it != list.end(); ++it) {
      if (position == 0) {
        REQUIRE(it.IsAtHead());
        REQUIRE(it.GetPrev().IsNull());
      } else {
        REQUIRE_FALSE(it.IsAtHead());
        REQUIRE_FALSE(it.GetPrev().IsNull());
      }
      position++;
    }

    REQUIRE(position == NUM_NODES);
  }

  SECTION("Iterator with interleaved operations") {
    pre::slist<false> list;
    list.Init();

    // Add initial nodes
    for (int i = 0; i < 3; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>(sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    // Take iterator snapshot
    auto it = list.begin(alloc);
    auto first_value = FullPtr<TestNode>(alloc, OffsetPtr<TestNode>(it.GetCurrent().load()))->value_;

    // Pop head (iterator still points to old head position, but it's no longer in list)
    list.pop(alloc);

    // New iteration should start from new head
    auto new_it = list.begin(alloc);
    auto new_first_value = FullPtr<TestNode>(alloc, OffsetPtr<TestNode>(new_it.GetCurrent().load()))->value_;

    REQUIRE(new_first_value != first_value);  // Different head after pop
    REQUIRE(list.size() == 2);
  }

  backend.shm_destroy();
}
