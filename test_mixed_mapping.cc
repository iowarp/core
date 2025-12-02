/**
 * Test program to verify PosixShmMmap mixed private/shared mapping
 *
 * This test verifies that:
 * 1. The first 16KB is private (process-local, not shared)
 * 2. The remaining region is shared (visible across processes)
 * 3. The regions are contiguous in virtual memory
 */

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include "hermes_shm/memory/backend/posix_shm_mmap.h"

using namespace hshm::ipc;

int main() {
  const std::string url = "/test_mixed_mapping";
  constexpr size_t kSharedSize = 1024 * 1024;  // 1MB shared region

  std::cout << "Testing PosixShmMmap with mixed private/shared mapping\n";
  std::cout << "Private region size: " << PosixShmMmap::GetPrivateRegionSize() << " bytes\n";
  std::cout << "Shared region size: " << kSharedSize << " bytes\n\n";

  // Parent process: Initialize
  PosixShmMmap backend;
  if (!backend.shm_init(MemoryBackendId(0, 0), kSharedSize, url)) {
    std::cerr << "Failed to initialize backend\n";
    return 1;
  }

  char *private_region = backend.GetPrivateRegion();
  char *shared_region = backend.data_;

  std::cout << "Parent: Initialized backend\n";
  std::cout << "  Private region: " << (void*)private_region << "\n";
  std::cout << "  Shared region:  " << (void*)shared_region << "\n";
  std::cout << "  Offset between: " << (shared_region - private_region) << " bytes\n";

  // Verify regions are contiguous
  if (shared_region - private_region != PosixShmMmap::GetPrivateRegionSize()) {
    std::cerr << "ERROR: Regions are not contiguous!\n";
    return 1;
  }
  std::cout << "✓ Regions are contiguous\n\n";

  // Write to private region
  strcpy(private_region, "PARENT_PRIVATE");
  std::cout << "Parent: Wrote to private region: " << private_region << "\n";

  // Write to shared region
  strcpy(shared_region, "PARENT_SHARED");
  std::cout << "Parent: Wrote to shared region: " << shared_region << "\n\n";

  // Fork child process
  pid_t pid = fork();

  if (pid == 0) {
    // Child process: Attach
    sleep(1);  // Give parent time

    PosixShmMmap child_backend;
    if (!child_backend.shm_attach(url)) {
      std::cerr << "Child: Failed to attach\n";
      return 1;
    }

    char *child_private = child_backend.GetPrivateRegion();
    char *child_shared = child_backend.data_;

    std::cout << "Child: Attached to backend\n";
    std::cout << "  Private region: " << (void*)child_private << "\n";
    std::cout << "  Shared region:  " << (void*)child_shared << "\n\n";

    // Check private region (should be zeroed, not parent's data)
    std::cout << "Child: Reading private region: '" << child_private << "'\n";
    if (strcmp(child_private, "PARENT_PRIVATE") == 0) {
      std::cerr << "ERROR: Private region is shared (should be independent)!\n";
      return 1;
    }
    std::cout << "✓ Private region is NOT shared (correct)\n\n";

    // Check shared region (should have parent's data)
    std::cout << "Child: Reading shared region: '" << child_shared << "'\n";
    if (strcmp(child_shared, "PARENT_SHARED") != 0) {
      std::cerr << "ERROR: Shared region does not contain parent's data!\n";
      return 1;
    }
    std::cout << "✓ Shared region IS shared (correct)\n\n";

    // Write to child's private region
    strcpy(child_private, "CHILD_PRIVATE");
    std::cout << "Child: Wrote to private region: " << child_private << "\n";

    // Write to shared region
    strcpy(child_shared, "CHILD_SHARED");
    std::cout << "Child: Wrote to shared region: " << child_shared << "\n";

    return 0;
  }

  // Parent: Wait for child
  int status;
  waitpid(pid, &status, 0);

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    std::cout << "\n✓ Child process completed successfully\n\n";
  } else {
    std::cerr << "\n✗ Child process failed\n";
    return 1;
  }

  // Check parent's private region (should still have parent's data)
  std::cout << "Parent: Re-reading private region: '" << private_region << "'\n";
  if (strcmp(private_region, "PARENT_PRIVATE") != 0) {
    std::cerr << "ERROR: Private region was modified by child!\n";
    return 1;
  }
  std::cout << "✓ Private region unchanged by child (correct)\n\n";

  // Check shared region (should have child's data)
  std::cout << "Parent: Re-reading shared region: '" << shared_region << "'\n";
  if (strcmp(shared_region, "CHILD_SHARED") != 0) {
    std::cerr << "ERROR: Shared region does not contain child's data!\n";
    return 1;
  }
  std::cout << "✓ Shared region contains child's data (correct)\n\n";

  backend.shm_destroy();

  std::cout << "===========================================\n";
  std::cout << "ALL TESTS PASSED\n";
  std::cout << "===========================================\n";

  return 0;
}
