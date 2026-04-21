# ------------------------------------------------------------------------------
# IOWarp HDF5 VOL Connector
# Requires: WRP_CTE_ENABLE_HDF5_VOL=ON and HDF5 >= 2.0
# ------------------------------------------------------------------------------

if(NOT HDF5_FOUND)
  message(WARNING "HDF5 not found. HDF5 VOL adapter will not be built.")
  return()
endif()

if(HDF5_VERSION VERSION_LESS "2.0.0")
  message(WARNING "HDF5 ${HDF5_VERSION} found but >= 2.0.0 required for VOL connector. "
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
