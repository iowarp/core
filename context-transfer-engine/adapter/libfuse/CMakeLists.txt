# Find FUSE3 from multiple sources (spack, apt, custom paths)
# Method 1: Try pkg-config first (works with spack)
find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(FUSE3 QUIET fuse3)
endif()

# Method 2: If pkg-config fails, try direct find (works with apt)
if(NOT FUSE3_FOUND)
  # Find the library directly
  find_library(FUSE3_LIBRARY 
    NAMES fuse3
    PATHS 
      /usr/lib/x86_64-linux-gnu  # Ubuntu/Debian apt location
      /usr/lib64                 # RHEL/CentOS
      /usr/local/lib
      ${FUSE3_ROOT}/lib          # Custom path
    PATH_SUFFIXES fuse3
  )
  
  # Find headers
  find_path(FUSE3_INCLUDE_DIR
    NAMES fuse3/fuse.h
    PATHS
      /usr/include/fuse3         # Ubuntu/Debian
      /usr/local/include/fuse3
      ${FUSE3_ROOT}/include
  )
  
  if(FUSE3_LIBRARY AND FUSE3_INCLUDE_DIR)
    set(FUSE3_FOUND TRUE)
    set(FUSE3_LIBRARIES ${FUSE3_LIBRARY})
    set(FUSE3_INCLUDE_DIRS ${FUSE3_INCLUDE_DIR})
    get_filename_component(FUSE3_LIBDIR ${FUSE3_LIBRARY} DIRECTORY)
  endif()
endif()

# Method 3: Allow explicit FUSE3_ROOT
if(FUSE3_ROOT AND NOT FUSE3_FOUND)
  find_library(FUSE3_LIBRARY fuse3 PATHS ${FUSE3_ROOT}/lib NO_DEFAULT_PATH)
  find_path(FUSE3_INCLUDE_DIR fuse3/fuse.h PATHS ${FUSE3_ROOT}/include NO_DEFAULT_PATH)
  if(FUSE3_LIBRARY AND FUSE3_INCLUDE_DIR)
    set(FUSE3_FOUND TRUE)
    set(FUSE3_LIBRARIES ${FUSE3_LIBRARY})
    set(FUSE3_INCLUDE_DIRS ${FUSE3_INCLUDE_DIR})
  endif()
endif()

# Verify found or fail
if(NOT FUSE3_FOUND)
  message(FATAL_ERROR
    "FUSE3 not found but WRP_CTE_ENABLE_FUSE_ADAPTER=ON.\n"
    "Searched:\n"
    "  - pkg-config (set PKG_CONFIG_PATH for spack)\n"
    "  - /usr/lib/x86_64-linux-gnu (apt)\n"
    "  - /usr/local/lib\n"
    "Install: sudo apt install libfuse3-dev\n"
    "Or set:  cmake -DFUSE3_ROOT=/path/to/fuse3 .."
  )
endif()

message(STATUS "Found FUSE3:")
message(STATUS "  Libraries: ${FUSE3_LIBRARIES}")
message(STATUS "  Includes:  ${FUSE3_INCLUDE_DIRS}")

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