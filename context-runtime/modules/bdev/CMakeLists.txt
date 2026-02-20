cmake_minimum_required(VERSION 3.10)

# Client library — hshm::aio provides async I/O headers via its INTERFACE includes
add_chimod_client(
  SOURCES
    src/bdev_client.cc
  LINK_LIBRARIES
    chimaera_admin_client
    hshm::aio
)

# Runtime library
add_chimod_runtime(
  SOURCES
    src/bdev_runtime.cc
    src/autogen/bdev_lib_exec.cc
  LINK_LIBRARIES
    chimaera_admin_runtime
    hshm::aio
)

# Unit tests
if(WRP_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
