# ------------------------------------------------------------------------------
# IOWarp HDF5 VOL Connector
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)

# Require HDF5 >= 1.12 for VOL connector support (H5VLconnector.h)
if(NOT HDF5_FOUND)
  find_package(HDF5 COMPONENTS C QUIET)
  if(NOT HDF5_FOUND)
    message(WARNING "HDF5 not found. HDF5 VOL adapter will not be built.")
    return()
  endif()
endif()

# Check for H5VLconnector.h (only in HDF5 >= 1.12)
include(CheckIncludeFile)
set(CMAKE_REQUIRED_INCLUDES ${HDF5_INCLUDE_DIRS} ${HDF5_C_INCLUDE_DIRS})
check_include_file("H5VLconnector.h" HAS_H5VL_CONNECTOR)
if(NOT HAS_H5VL_CONNECTOR)
  message(WARNING "HDF5 VOL connector API not found (requires HDF5 >= 1.12). "
                  "HDF5 VOL adapter will not be built.")
  return()
endif()

message(STATUS "Building HDF5 VOL adapter (iowarp_hdf5_vol)")

add_library(iowarp_hdf5_vol SHARED
    iowarp_vol.cc
)

target_include_directories(iowarp_hdf5_vol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${HDF5_C_INCLUDE_DIRS}
)

target_link_libraries(iowarp_hdf5_vol
    wrp_cte_core_client
    wrp_cte_core_runtime
    chimaera::admin_runtime
    chimaera::admin_client
    chimaera::bdev_runtime
    chimaera::bdev_client
    hshm::cxx
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(iowarp_hdf5_vol PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS iowarp_hdf5_vol
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

message(STATUS "HDF5 VOL adapter configured successfully")
