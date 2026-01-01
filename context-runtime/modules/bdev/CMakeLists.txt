cmake_minimum_required(VERSION 3.10)

# Find libaio for Linux AIO support in per-worker I/O contexts
find_library(LIBAIO_LIBRARY NAMES aio libaio)
find_path(LIBAIO_INCLUDE_DIR NAMES libaio.h)

if(NOT LIBAIO_LIBRARY OR NOT LIBAIO_INCLUDE_DIR)
  message(FATAL_ERROR "libaio library not found. Please install libaio-dev (apt) or libaio (conda)")
endif()

message(STATUS "Found libaio: ${LIBAIO_LIBRARY}")
message(STATUS "Found libaio headers: ${LIBAIO_INCLUDE_DIR}")

# Create client library
# Note: Admin headers are automatically available via chimaera_admin_client link dependency
add_chimod_client(
  SOURCES
    src/bdev_client.cc
  LINK_LIBRARIES
    chimaera_admin_client  # Provides admin headers via transitive includes
)

# Create runtime library
# Note: Admin headers are automatically available via chimaera_admin_runtime link dependency
add_chimod_runtime(
  SOURCES
    src/bdev_runtime.cc
    src/autogen/bdev_lib_exec.cc
  LINK_LIBRARIES
    chimaera_admin_runtime  # Provides admin headers via transitive includes
    ${LIBAIO_LIBRARY}  # libaio for Linux AIO support
  INCLUDE_DIRECTORIES
    ${LIBAIO_INCLUDE_DIR}
)