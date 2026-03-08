cmake_minimum_required(VERSION 3.10)

# Collect optional NIXL dependency
set(BDEV_OPTIONAL_LIBS "")
if(WRP_CORE_ENABLE_NIXL)
  list(APPEND BDEV_OPTIONAL_LIBS hshm::nixl)
endif()

# Client library — hshm::aio provides async I/O headers via its INTERFACE includes
add_chimod_client(
  SOURCES
    src/bdev_client.cc
  LINK_LIBRARIES
    chimaera_admin_client
    hshm::aio
    ${BDEV_OPTIONAL_LIBS}
)

# Runtime library
add_chimod_runtime(
  SOURCES
    src/bdev_runtime.cc
    src/autogen/bdev_lib_exec.cc
  LINK_LIBRARIES
    chimaera_admin_runtime
    hshm::aio
    ${BDEV_OPTIONAL_LIBS}
)

# Unit tests
if(WRP_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
