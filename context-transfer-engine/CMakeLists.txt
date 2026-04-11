cmake_minimum_required(VERSION 3.20)
project(content_transfer_engine)

# Set root directory for this component
set(WRP_CTE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Note: WRP_CTE_ENABLE_COMPRESS is defined in root CMakeLists.txt
# It controls both compressor chimod building and HSHM compression support

# Python bindings are controlled by WRP_CORE_ENABLE_PYTHON in root CMakeLists.txt
# CTE_ENABLE_ASAN is set by root CMakeLists.txt

# C++ standard, compile commands, RPATH, ASAN, and output directories are all set by root CMakeLists.txt

# CTE_ENABLE_CMAKE_DOTENV is set by root CMakeLists.txt
# DOTENV handling is done by root CMakeLists.txt

# CMake modules and ChiMod utilities are now provided by IowarpCoreCommon.cmake
# All find_package calls are handled by root CMakeLists.txt
# Targets chimaera::cxx, chimaera_admin_client, etc. are already available

# Python bindings configuration
# nanobind is configured by root CMakeLists.txt when WRP_CORE_ENABLE_PYTHON=ON
# No local nanobind setup needed - use the root configuration
if(WRP_CORE_ENABLE_PYTHON)
  if(NOT COMMAND nanobind_add_module)
    message(WARNING "WRP_CORE_ENABLE_PYTHON is ON but nanobind_add_module command not found. Python bindings may fail.")
  else()
    message(STATUS "Python bindings available - nanobind configured by root CMakeLists.txt")
  endif()
endif()

message(STATUS "Found Chimaera framework - building full ChiMod implementation")

# Add the core ChiMod
add_subdirectory(core)

# Add the compressor ChiMod (optional - only if compression is enabled)
if(WRP_CTE_ENABLE_COMPRESS)
    message(STATUS "CTE: Compression enabled - building compressor chimod")
    add_subdirectory(compressor)
else()
    message(STATUS "CTE: Compression disabled - skipping compressor chimod (use -DWRP_CTE_ENABLE_COMPRESS=ON to enable)")
endif()

# Add the bdev ChiMod (when it exists)
# add_subdirectory(chimods/bdev)

# Add adapters
add_subdirectory(adapter)

# Add eBPF I/O interceptor adapter (if enabled)
if(WRP_CTE_ENABLE_EBPF_ADAPTER)
    message(STATUS "CTE: eBPF adapter enabled - building eBPF I/O interceptor")
    add_subdirectory(interceptor-ebpf)
else()
    message(STATUS "CTE: eBPF adapter disabled (use -DWRP_CTE_ENABLE_EBPF_ADAPTER=ON to enable)")
endif()

# Add UVM (GPU virtual memory manager) -- requires CUDA
add_subdirectory(uvm)

# Enable testing
# CTest is included by root CMakeLists.txt
include(CTest)

# Add tests if enabled
# Tests are controlled by WRP_CORE_ENABLE_TESTS in root CMakeLists.txt
if(WRP_CORE_ENABLE_TESTS)
  add_subdirectory(test)
  message(STATUS "CTE tests enabled - added test subdirectory")
endif()

# Add wrapper components (includes Python bindings if enabled)
add_subdirectory(wrapper)

# Add benchmarks if enabled
# Benchmarks are controlled by WRP_CORE_ENABLE_BENCHMARKS
if(WRP_CORE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
  message(STATUS "CTE benchmarks enabled - added benchmark subdirectory")
endif()

message(STATUS "CTE Core and Bdev ChiMods configured for compilation")

if(WRP_CORE_ENABLE_JARVIS)
    jarvis_repo_add(${WRP_CTE_ROOT}/test/jarvis_iowarp)
endif()

#------------------------------------------------------------------------------
# Rust Bindings Configuration
#------------------------------------------------------------------------------
if(WRP_CORE_ENABLE_RUST)
    if(NOT COMMAND corrosion_import_crate)
        message(WARNING "WRP_CORE_ENABLE_RUST is ON but Corrosion not found. Ensure CMakeLists.txt root fetched Corrosion.")
    else()
        message(STATUS "Building CTE Rust bindings - Corrosion configured")
        
         # Export paths for build.rs
         # Include paths: hermes_shm, chimaera core, chimaera modules, cereal, and CTE headers
         message(STATUS "Setting HSHM_ROOT from CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
         get_filename_component(HSHM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../context-transport-primitives" ABSOLUTE)
        get_filename_component(CHIMAERA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../context-runtime" ABSOLUTE)
        get_filename_component(CTE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
        get_filename_component(CHIMODS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../context-runtime/modules" ABSOLUTE)
        
         # Get cereal include directory from CMake target
         get_target_property(CEREAL_INCLUDE_DIR cereal::cereal INTERFACE_INCLUDE_DIRECTORIES)
         if(NOT CEREAL_INCLUDE_DIR OR CEREAL_INCLUDE_DIR STREQUAL "CEREAL_INCLUDE_DIR-NOTFOUND")
             # Fallback: find cereal include from cereal_DIR
             # cereal_DIR points to cmake config, e.g., /path/to/cereal/lib64/cmake/cereal
             # Include dir is typically at /path/to/cereal/include
             if(cereal_DIR)
                 get_filename_component(CEREAL_ROOT "${cereal_DIR}/../../.." ABSOLUTE)
                 set(CEREAL_INCLUDE_DIR "${CEREAL_ROOT}/include")
             else()
                 # Last resort fallback to common locations
                 if(EXISTS "/usr/local/include/cereal")
                     set(CEREAL_INCLUDE_DIR "/usr/local/include")
                 elseif(EXISTS "/usr/include/cereal")
                     set(CEREAL_INCLUDE_DIR "/usr/include")
                 else()
                     message(WARNING "Cereal include directory not found - Rust build may fail")
                     set(CEREAL_INCLUDE_DIR "")
                 endif()
             endif()
         endif()
        
        message(STATUS "CTE Rust include paths:")
        message(STATUS "  CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
        message(STATUS "  HSHM_ROOT (before expansion): ${HSHM_ROOT}")
        message(STATUS "  HSHM_ROOT: ${HSHM_ROOT}/include")
        message(STATUS "  CHIMAERA_ROOT: ${CHIMAERA_ROOT}/include")
        message(STATUS "  CTE_ROOT: ${CTE_ROOT}/core/include")
        message(STATUS "  CHIMODS_ROOT: ${CHIMODS_ROOT}")
        message(STATUS "  CEREAL_INCLUDE: ${CEREAL_INCLUDE_DIR}")
        
        # Validate ZMQ was found before configuring Rust bindings
        if(WRP_CORE_ENABLE_ZMQ AND NOT ZMQ_LIBS)
            message(FATAL_ERROR "ZeroMQ not found but required for Rust bindings. Install with: sudo apt-get install libzmq3-dev")
        endif()

        # Include rust subdirectory to get ANERIS_RUSTFLAGS variable
        include(${CMAKE_CURRENT_SOURCE_DIR}/wrapper/rust/CMakeLists.txt)

         # Set RUSTFLAGS for corrosion
         set(ENV{RUSTFLAGS} ${ANERIS_RUSTFLAGS})

         # Import Rust crate using Corrosion
         # Note: Requires tokio features: rt, rt-multi-thread, signal, time, sync, macros
         # These features are specified in Cargo.toml and picked up during CMake configuration
         corrosion_import_crate(
             MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/rust/Cargo.toml
             FEATURES async
         )
         
         # Check variable values (for debug, can be removed after verification)
         
         # Set environment variables for aneris-profiler and aneris-rescorer as well
         # These are bin targets in the same Cargo.toml as wrp_cte but need the same env vars
         foreach(aneris_target aneris-profiler aneris-rescorer aneris-tune)
             message(STATUS "DEBUG: Checking if TARGET _cargo-build_${aneris_target} exists")
             if(TARGET _cargo-build_${aneris_target})
                 message(STATUS "DEBUG: Setting environment variables for _cargo-build_${aneris_target}")
                 corrosion_set_env_vars(_cargo-build_${aneris_target}
                     "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                     "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                     "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                     "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                     "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
                 )
             else()
                 message(STATUS "DEBUG: TARGET _cargo-build_${aneris_target} does not exist")
             endif()
         endforeach()
         
          # Set environment variables for build.rs to find C++ headers and libraries
           # Note: Corrosion uses crate name from Cargo.toml: wrp_cte (from wrp-cte-rs package)
           # The internal build target is _cargo-build_wrp_cte
           message(STATUS "Setting environment variables for Rust build targets")
           message(STATUS "  HSHM_ROOT: ${HSHM_ROOT}/include")
           message(STATUS "  CHIMAERA_ROOT: ${CHIMAERA_ROOT}/include")
           message(STATUS "  CTE_ROOT: ${CTE_ROOT}/core/include")
           message(STATUS "  CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}/bin")
           
            # Set for the main library target (crate name: wrp_cte)
            message(STATUS "DEBUG: IOWARP_EXTRA_INCLUDES value = ${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}")
            if(TARGET wrp_cte)
                corrosion_set_env_vars(wrp_cte 
                   "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                   "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                   "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                   "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                   "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
                )
                message(STATUS "  Set env vars for target: wrp_cte")
            else()
                message(WARNING "Target wrp_cte not found")
            endif()
           
           # Also set for the internal cargo-build target
           if(TARGET _cargo-build_wrp_cte)
               corrosion_set_env_vars(_cargo-build_wrp_cte 
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
               )
               message(STATUS "  Set env vars for target: _cargo-build_wrp_cte")
           else()
               message(WARNING "Target _cargo-build_wrp_cte not found")
           endif()
          
          # Also set environment variables for aneris-profiler which is a bin target in the same crate
          if(TARGET aneris-profiler)
              corrosion_set_env_vars(aneris-profiler
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
              )
              message(STATUS "  Set env vars for target: aneris-profiler")
          endif()
          
          if(TARGET _cargo-build_aneris-profiler)
              corrosion_set_env_vars(_cargo-build_aneris-profiler
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
              )
              message(STATUS "  Set env vars for target: _cargo-build_aneris-profiler")
          endif()
          
          # Also set environment variables for aneris-rescorer which is a bin target in the same crate
          if(TARGET aneris-rescorer)
              corrosion_set_env_vars(aneris-rescorer
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
              )
              message(STATUS "  Set env vars for target: aneris-rescorer")
          endif()
          
          if(TARGET _cargo-build_aneris-rescorer)
              corrosion_set_env_vars(_cargo-build_aneris-rescorer
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
              )
              message(STATUS "  Set env vars for target: _cargo-build_aneris-rescorer")
          endif()
          
          # Also set environment variables for aneris-tune which is a bin target in the same crate
          if(TARGET aneris-tune)
              corrosion_set_env_vars(aneris-tune
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
              )
              message(STATUS "  Set env vars for target: aneris-tune")
          endif()
          
          if(TARGET _cargo-build_aneris-tune)
              corrosion_set_env_vars(_cargo-build_aneris-tune
                  "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
                  "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
                  "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
                  "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
                  "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
              )
              message(STATUS "  Set env vars for target: _cargo-build_aneris-tune")
          endif()

         # Debug: Check if target property is set (try both possible property names)
         get_target_property(test_props wrp_cte CORROSION_ENVIRONMENT_VARIABLES)
         if(test_props STREQUAL "test_props-NOTFOUND")
             get_target_property(test_props wrp_cte INTERFACE_CORROSION_ENVIRONMENT_VARIABLES)
         endif()
         message(STATUS "DEBUG: wrp_cte environment variables = ${test_props}")
        
         # Ensure C++ client library is built before Rust library
         # Corrosion creates internal targets like _cargo-build_<crate>
         # We need to make the cargo build depend on the C++ library
         # Target names use crate name: wrp_cte
         if(TARGET _cargo-build_wrp_cte)
             add_dependencies(_cargo-build_wrp_cte wrp_cte_core_client)
             message(STATUS "Added dependency: _cargo-build_wrp_cte -> wrp_cte_core_client")
         else()
             message(STATUS "  Note: _cargo-build_wrp_cte not found yet (will be created by Corrosion)")
         endif()
         if(TARGET wrp_cte)
             add_dependencies(wrp_cte wrp_cte_core_client)
             message(STATUS "Added dependency: wrp_cte -> wrp_cte_core_client")
         else()
             message(STATUS "  Note: wrp_cte not found yet (will be created by Corrosion)")
         endif()
         
         # Serialize Rust builds to prevent Cargo target directory conflicts
         # All binaries share the same Cargo.toml workspace, so they must build sequentially
         if(TARGET _cargo-build_wrp_cte AND TARGET _cargo-build_aneris-profiler)
             add_dependencies(_cargo-build_aneris-profiler _cargo-build_wrp_cte)
             message(STATUS "Rust build serialization: aneris-profiler -> wrp_cte")
         endif()

        if(TARGET _cargo-build_aneris-rescorer AND TARGET _cargo-build_aneris-profiler)
            add_dependencies(_cargo-build_aneris-rescorer _cargo-build_aneris-profiler)
            message(STATUS "Rust build serialization: aneris-rescorer -> aneris-profiler")
        endif()
        
        # Add dependency chain for aneris-tune (must build after other binaries)
        if(TARGET _cargo-build_aneris-tune AND TARGET _cargo-build_aneris-rescorer)
            add_dependencies(_cargo-build_aneris-tune _cargo-build_aneris-rescorer)
            message(STATUS "Rust build serialization: aneris-tune -> aneris-rescorer")
        endif()
        
        # Also ensure C++ libraries are built before aneris-tune
        if(TARGET _cargo-build_aneris-tune)
            add_dependencies(_cargo-build_aneris-tune wrp_cte_core_client)
            message(STATUS "Added dependency: _cargo-build_aneris-tune -> wrp_cte_core_client")
        endif()
        
        message(STATUS "CTE Rust bindings configured: wrp_cte (from package wrp-cte-rs)")
        
        #------------------------------------------------------------------------------
        # Rust Example/Test Targets
        #------------------------------------------------------------------------------
        # These targets allow running Rust examples and tests via CMake with
        # proper environment setup
        
         # Get Cereal include directory for build.rs
         get_target_property(CEREAL_INCLUDE_DIR cereal::cereal INTERFACE_INCLUDE_DIRECTORIES)
         if(NOT CEREAL_INCLUDE_DIR OR CEREAL_INCLUDE_DIR STREQUAL "CEREAL_INCLUDE_DIR-NOTFOUND")
             if(cereal_DIR)
                 get_filename_component(CEREAL_ROOT "${cereal_DIR}/../../.." ABSOLUTE)
                 set(CEREAL_INCLUDE_DIR "${CEREAL_ROOT}/include")
             else()
                 if(EXISTS "/usr/local/include/cereal")
                     set(CEREAL_INCLUDE_DIR "/usr/local/include")
                 elseif(EXISTS "/usr/include/cereal")
                     set(CEREAL_INCLUDE_DIR "/usr/include")
                 else()
                     set(CEREAL_INCLUDE_DIR "")
                 endif()
             endif()
         endif()
        
        # Export variables for Rust build scripts
        set(RUST_ENV_VARS
            "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
            "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_ROOT}/core/include:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include:${CEREAL_INCLUDE_DIR}"
            "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
            "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
            "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
            "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
            "CHI_WITH_RUNTIME=1"
        )
        
        # Add custom target to run Rust telemetry example
        add_custom_target(run_rust_telemetry
            COMMAND ${CMAKE_COMMAND} -E env ${RUST_ENV_VARS}
                    ${Rust_CARGO} run --example telemetry_capture
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/rust
            COMMENT "Running Rust telemetry capture example..."
            DEPENDS wrp_cte
        )
        
        # Add custom target to run Rust unit tests
        add_custom_target(run_rust_tests
            COMMAND ${CMAKE_COMMAND} -E env ${RUST_ENV_VARS}
                    ${Rust_CARGO} test --lib
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/rust
            COMMENT "Running Rust unit tests..."
            DEPENDS wrp_cte
        )
        
        # Add custom target to run Rust integration tests
        add_custom_target(run_rust_integration_tests
            COMMAND ${CMAKE_COMMAND} -E env ${RUST_ENV_VARS}
                    ${Rust_CARGO} test -- --include-ignored
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/rust
            COMMENT "Running Rust integration tests..."
            DEPENDS wrp_cte
        )
        
        message(STATUS "  Added CMake targets:")
        message(STATUS "    - run_rust_telemetry: Run telemetry capture example")
        message(STATUS "    - run_rust_tests: Run Rust unit tests")
        message(STATUS "    - run_rust_integration_tests: Run Rust integration tests")
        
        #------------------------------------------------------------------------------
        # Aneris Telemetry Script
        #------------------------------------------------------------------------------
        # Generate Aneris_telemetry script with all paths properly configured
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/wrapper/rust/Aneris_telemetry.in"
            "${CMAKE_BINARY_DIR}/bin/Aneris_telemetry"
            @ONLY
        )
        
        # Make it executable
        file(CHMOD "${CMAKE_BINARY_DIR}/bin/Aneris_telemetry"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
        )
        
        message(STATUS "Generated Aneris_telemetry script: ${CMAKE_BINARY_DIR}/bin/Aneris_telemetry")
    endif()
endif()
