/**
 * Multi-process unit test for MultiProcessAllocator
 *
 * Usage: test_mp_allocator_multiprocess <rank> <duration_sec> <nthreads>
 *
 * rank 0: Initializes shared memory and optionally runs for duration_sec
 * rank 1+: Attaches to shared memory and runs for duration_sec
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "hermes_shm/memory/allocator/mp_allocator.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"
#include "allocator_test.h"

using namespace hshm::ipc;
using namespace hshm::testing;

// Shared memory configuration
constexpr size_t kShmSize = 512 * 1024 * 1024;  // 512 MB
const std::string kShmUrl = "/mp_allocator_multiprocess_test";

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <rank> <duration_sec> <nthreads>" << std::endl;
    return 1;
  }

  int rank = std::atoi(argv[1]);
  int duration_sec = std::atoi(argv[2]);
  int nthreads = std::atoi(argv[3]);

  std::cout << "Rank " << rank << ": Starting test with " << nthreads
            << " threads for " << duration_sec << " seconds" << std::endl;

  // Create or attach to shared memory
  PosixShmMmap backend;
  bool success = false;

  if (rank == 0) {
    // Rank 0 initializes
    std::cout << "Rank 0: Initializing shared memory" << std::endl;
    success = backend.shm_init(MemoryBackendId(0, 0), kShmSize, kShmUrl);
    if (!success) {
      std::cerr << "Rank 0: Failed to initialize shared memory" << std::endl;
      return 1;
    }
    // Memset backend.data_ to 11 before allocator construction
    std::memset(backend.data_, 11, backend.data_capacity_);
  } else {
    // Other ranks attach to existing shared memory
    std::cout << "Rank " << rank << ": Attaching to shared memory" << std::endl;
    success = backend.shm_attach(kShmUrl);
    if (!success) {
      std::cerr << "Rank " << rank << ": Failed to attach to shared memory" << std::endl;
      return 1;
    }
  }

  // Initialize or attach allocator
  MultiProcessAllocator *allocator = nullptr;

  if (rank == 0) {
    std::cout << "Rank 0: Initializing allocator" << std::endl;
    std::cout << "  Backend data capacity: " << backend.data_capacity_ << std::endl;
    allocator = backend.MakeAlloc<MultiProcessAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank 0: Failed to initialize allocator" << std::endl;
      return 1;
    }
    std::cout << "Rank 0: Allocator initialized successfully" << std::endl;
    std::cout << "  Allocator size: " << sizeof(MultiProcessAllocator) << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Attaching to allocator" << std::endl;

    // Give rank 0 time to fully initialize before we try to attach
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Attach to existing allocator without reinitializing
    allocator = backend.AttachAlloc<MultiProcessAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank " << rank << ": Failed to attach to allocator" << std::endl;
      return 1;
    }
    std::cout << "Rank " << rank << ": Attached to allocator successfully" << std::endl;
  }

  // Run test if duration > 0
  if (duration_sec > 0) {
    std::cout << "Rank " << rank << ": Starting timed workload test with " << nthreads
              << " threads for " << duration_sec << " seconds" << std::endl;
    std::cout << "Rank " << rank << ": Testing SMALL allocations only (1 byte to 16KB)" << std::endl;

    // Create allocator tester and run timed workload with SMALL allocations only
    AllocatorTest<MultiProcessAllocator> tester(allocator);
    constexpr size_t kSmallMin = 1;           // 1 byte
    constexpr size_t kSmallMax = 16 * 1024;   // 16 KB
    tester.TestTimedMultiThreadedWorkload(nthreads, duration_sec, kSmallMin, kSmallMax);
    std::cout << "Rank " << rank << ": TEST PASSED" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Initialization complete, exiting" << std::endl;
  }

  // Only rank 0 should clean up shared memory, and only if it ran the test
  // (if duration was 0, other ranks may still be using it)
  if (rank > 0 || (rank == 0 && duration_sec > 0)) {
    std::cout << "Rank 0: Cleaning up shared memory" << std::endl;
    backend.shm_destroy();
  }

  return 0;
}
