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
#include "hermes_shm/memory/allocator/malloc_allocator.h"
#include <iostream>

using namespace hshm::ipc;

/**
 * Test basic allocation and deallocation with HSHM_MALLOC
 */
TEST_CASE("HSHM_MALLOC: basic allocate and free", "[hshm_malloc][basic]") {
  std::cout << "\n=== Test 1: Basic Allocate and Free ===" << std::endl;

  // Allocate a buffer
  size_t size = 1024;
  auto buffer = HSHM_MALLOC->AllocateObjs<char>(size);

  std::cout << "Allocated " << size << " bytes" << std::endl;
  std::cout << "  ptr_ = " << (void*)buffer.ptr_ << std::endl;
  std::cout << "  off_ = " << buffer.shm_.off_.load() << std::endl;
  std::cout << "  alloc_id = (" << buffer.shm_.alloc_id_.major_
            << "." << buffer.shm_.alloc_id_.minor_ << ")" << std::endl;

  REQUIRE(!buffer.IsNull());
  REQUIRE(buffer.ptr_ != nullptr);

  // Write to the buffer to ensure it's valid
  for (size_t i = 0; i < size; i++) {
    buffer.ptr_[i] = static_cast<char>(i % 256);
  }

  // Verify the data
  for (size_t i = 0; i < size; i++) {
    REQUIRE(buffer.ptr_[i] == static_cast<char>(i % 256));
  }

  std::cout << "Buffer is valid and writable" << std::endl;

  // Free the buffer
  std::cout << "Calling HSHM_MALLOC->Free()..." << std::endl;
  HSHM_MALLOC->Free(buffer);
  std::cout << "Free completed successfully" << std::endl;
}

/**
 * Test multiple allocations and deallocations
 */
TEST_CASE("HSHM_MALLOC: multiple allocations", "[hshm_malloc][multiple]") {
  std::cout << "\n=== Test 2: Multiple Allocations ===" << std::endl;

  const int num_buffers = 10;
  std::vector<FullPtr<char>> buffers;

  // Allocate multiple buffers
  for (int i = 0; i < num_buffers; i++) {
    size_t size = 512 + i * 128;
    auto buffer = HSHM_MALLOC->AllocateObjs<char>(size);

    std::cout << "Allocated buffer " << i << ": " << size << " bytes at ptr_="
              << (void*)buffer.ptr_ << std::endl;

    REQUIRE(!buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Write unique pattern
    for (size_t j = 0; j < size; j++) {
      buffer.ptr_[j] = static_cast<char>((i * 100 + j) % 256);
    }

    buffers.push_back(buffer);
  }

  // Verify all buffers
  for (int i = 0; i < num_buffers; i++) {
    size_t size = 512 + i * 128;
    for (size_t j = 0; j < size; j++) {
      REQUIRE(buffers[i].ptr_[j] == static_cast<char>((i * 100 + j) % 256));
    }
  }

  std::cout << "All buffers verified successfully" << std::endl;

  // Free all buffers
  for (int i = 0; i < num_buffers; i++) {
    std::cout << "Freeing buffer " << i << " at ptr_=" << (void*)buffers[i].ptr_ << std::endl;
    HSHM_MALLOC->Free(buffers[i]);
  }

  std::cout << "All buffers freed successfully" << std::endl;
}

/**
 * Test allocation with NULL allocator ID
 */
TEST_CASE("HSHM_MALLOC: allocator ID check", "[hshm_malloc][allocator_id]") {
  std::cout << "\n=== Test 3: Allocator ID Check ===" << std::endl;

  auto buffer = HSHM_MALLOC->AllocateObjs<char>(2048);

  REQUIRE(!buffer.IsNull());

  // Check that HSHM_MALLOC uses NULL allocator ID
  AllocatorId null_id = AllocatorId::GetNull();
  std::cout << "Expected NULL allocator ID: (" << null_id.major_ << "." << null_id.minor_ << ")" << std::endl;
  std::cout << "Actual allocator ID: (" << buffer.shm_.alloc_id_.major_
            << "." << buffer.shm_.alloc_id_.minor_ << ")" << std::endl;

  REQUIRE(buffer.shm_.alloc_id_ == null_id);

  std::cout << "Allocator ID is correctly NULL" << std::endl;

  HSHM_MALLOC->Free(buffer);
  std::cout << "Buffer freed successfully" << std::endl;
}

/**
 * Test the relationship between ptr_ and off_ in FullPtr
 */
TEST_CASE("HSHM_MALLOC: ptr and offset relationship", "[hshm_malloc][ptr_offset]") {
  std::cout << "\n=== Test 4: Ptr and Offset Relationship ===" << std::endl;

  auto buffer = HSHM_MALLOC->AllocateObjs<char>(1024);

  REQUIRE(!buffer.IsNull());

  std::cout << "ptr_ = " << (void*)buffer.ptr_ << std::endl;
  std::cout << "off_ = " << buffer.shm_.off_.load() << std::endl;

  // For HSHM_MALLOC with MallocPage header:
  // - The off_ should point to the data (after MallocPage header)
  // - The ptr_ should also point to the data
  // - They should be the same value

  uintptr_t ptr_value = reinterpret_cast<uintptr_t>(buffer.ptr_);
  size_t off_value = buffer.shm_.off_.load();

  std::cout << "ptr_ as uintptr_t = " << ptr_value << std::endl;
  std::cout << "off_ as size_t = " << off_value << std::endl;

  REQUIRE(ptr_value == off_value);
  std::cout << "ptr_ and off_ are equal (correct for HSHM_MALLOC)" << std::endl;

  HSHM_MALLOC->Free(buffer);
  std::cout << "Buffer freed successfully" << std::endl;
}

/**
 * Test allocating a FullPtr similar to how IpcManager does it
 */
TEST_CASE("HSHM_MALLOC: simulate IpcManager allocation", "[hshm_malloc][ipc_manager]") {
  std::cout << "\n=== Test 5: Simulate IpcManager Allocation Pattern ===" << std::endl;

  // Simulate the pattern used in IpcManager::AllocateBuffer (RUNTIME mode)
  size_t size = sizeof(int) + 4096;  // Simulate FutureShm + copy_space

  std::cout << "Allocating " << size << " bytes (simulating FutureShm)" << std::endl;
  FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(size);

  REQUIRE(!buffer.IsNull());
  REQUIRE(buffer.ptr_ != nullptr);
  REQUIRE(buffer.shm_.alloc_id_ == AllocatorId::GetNull());

  std::cout << "Allocation successful:" << std::endl;
  std::cout << "  ptr_ = " << (void*)buffer.ptr_ << std::endl;
  std::cout << "  off_ = " << buffer.shm_.off_.load() << std::endl;
  std::cout << "  alloc_id = NULL (as expected)" << std::endl;

  // Write some data
  std::memset(buffer.ptr_, 0xAB, size);

  // Verify
  for (size_t i = 0; i < size; i++) {
    REQUIRE(buffer.ptr_[i] == static_cast<char>(0xAB));
  }

  std::cout << "Data written and verified" << std::endl;

  // Now free it the same way IpcManager::FreeBuffer does
  std::cout << "Freeing buffer with HSHM_MALLOC->Free()..." << std::endl;

  // This is the exact call that IpcManager makes
  HSHM_MALLOC->Free(buffer);

  std::cout << "Free completed successfully!" << std::endl;
}

/**
 * Test double-free detection (should fail if attempted)
 */
TEST_CASE("HSHM_MALLOC: verify single free only", "[hshm_malloc][single_free]") {
  std::cout << "\n=== Test 6: Verify Single Free Only ===" << std::endl;

  auto buffer = HSHM_MALLOC->AllocateObjs<char>(512);
  REQUIRE(!buffer.IsNull());

  std::cout << "Allocated buffer at ptr_=" << (void*)buffer.ptr_ << std::endl;

  // Free once (should work)
  std::cout << "Freeing buffer (first time)..." << std::endl;
  HSHM_MALLOC->Free(buffer);
  std::cout << "First free completed" << std::endl;

  // Note: We do NOT attempt a second free here because that would crash the test
  // The point is to verify that a single free works correctly

  std::cout << "Single free verified - not attempting double free" << std::endl;
}

SIMPLE_TEST_MAIN()
