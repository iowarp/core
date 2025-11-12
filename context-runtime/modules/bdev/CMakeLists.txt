cmake_minimum_required(VERSION 3.10)


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
)