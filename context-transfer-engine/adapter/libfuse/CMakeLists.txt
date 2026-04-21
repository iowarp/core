# Find FUSE3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 QUIET fuse3)

if(NOT FUSE3_FOUND)
  message(WARNING "FUSE3 not found. FUSE adapter will not be built.")
  message(STATUS "To install FUSE3: sudo apt install libfuse3-dev")
  return()
endif()

message(STATUS "Found FUSE3: ${FUSE3_VERSION}")

add_executable(wrp_cte_fuse
    fuse_cte.cc)

target_compile_definitions(wrp_cte_fuse PRIVATE
    FUSE_USE_VERSION=35
    WRP_CTE_FUSE_ENABLED)

target_include_directories(wrp_cte_fuse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${WRP_CTE_ROOT}
    ${FUSE3_INCLUDE_DIRS})

target_link_libraries(wrp_cte_fuse
    ${FUSE3_LIBRARIES}
    wrp_cte_core_client
    wrp_cte_cae_config
    hshm::cxx)

install(TARGETS wrp_cte_fuse
    RUNTIME DESTINATION ${WRP_CTE_INSTALL_BIN_DIR})
