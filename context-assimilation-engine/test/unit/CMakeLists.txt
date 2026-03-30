#------------------------------------------------------------------------------
# Unit Tests CMakeLists.txt
#------------------------------------------------------------------------------
# This file defines all unit tests for the CAE (Content Assimilation Engine)
# and configures them as CTest targets for automated testing.
#
# Tests include:
# - Binary file assimilation
# - Range-based partial transfers
# - Error handling and validation
# - HDF5 format assimilation
# - Globus transfer integration (conditional)
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

# Enable testing for CTest
enable_testing()

# Find required dependencies
# yaml-cpp is already found by root CMakeLists.txt

# Conditionally find HDF5 if enabled
if(WRP_CAE_ENABLE_HDF5)
  # HDF5 is already found by root CMakeLists.txt
  if(NOT HDF5_FOUND)
    message(WARNING "WRP_CAE_ENABLE_HDF5 is ON but HDF5 not found. HDF5 tests will be disabled.")
    set(HDF5_ENABLED FALSE)
  else()
    set(HDF5_ENABLED TRUE)
  endif()
else()
  set(HDF5_ENABLED FALSE)
endif()

#------------------------------------------------------------------------------
# Helper function to create a unit test
#------------------------------------------------------------------------------
# Creates an executable, links dependencies, and registers with CTest
#
# Arguments:
#   TEST_NAME - Name of the test executable
#   TEST_SOURCE - Path to the test source file (relative to unit/ directory)
#   EXTRA_LIBS - Additional libraries to link (optional)
#   EXTRA_INCLUDES - Additional include directories (optional)
#------------------------------------------------------------------------------
function(add_cae_unit_test TEST_NAME TEST_SOURCE)
  # Parse optional arguments
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs EXTRA_LIBS EXTRA_INCLUDES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Create test executable
  add_executable(${TEST_NAME} ${TEST_SOURCE})

  # Link common dependencies
  target_link_libraries(${TEST_NAME}
    PRIVATE
      wrp_cae_core_client
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      ${ARG_EXTRA_LIBS}
  )

  # Add common include directories
  target_include_directories(${TEST_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${ARG_EXTRA_INCLUDES}
  )

  # Install test binary
  install(TARGETS ${TEST_NAME} DESTINATION bin)

  # Register with CTest
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

  # Set test environment variables
  set_tests_properties(${TEST_NAME} PROPERTIES
    ENVIRONMENT "CHI_WITH_RUNTIME=1;CHI_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/wrp_config.yaml;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 300  # 5 minute timeout
  )
endfunction()

#------------------------------------------------------------------------------
# Test 1: Binary File Assimilation
#------------------------------------------------------------------------------
# Tests basic binary file transfer functionality
# - Creates test binary file with patterned data
# - Loads OMNI configuration from YAML
# - Calls ParseOmni to transfer to CTE
# - Validates data integrity in CTE
#------------------------------------------------------------------------------
add_cae_unit_test(cae_binary_assim
  ${CMAKE_CURRENT_SOURCE_DIR}/binary_assim/test_binary_assim.cc
)

# Install test configuration
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/binary_assim/binary_assim_omni.yaml
  DESTINATION share/wrp_cae/test/binary_assim
)

#------------------------------------------------------------------------------
# Test 2: Range-Based Partial Transfers
#------------------------------------------------------------------------------
# Tests partial file transfers using range_off and range_size
# - Creates large test file
# - Transfers specific byte ranges
# - Validates correct data and size in CTE
#------------------------------------------------------------------------------
add_cae_unit_test(cae_range_assim
  ${CMAKE_CURRENT_SOURCE_DIR}/range_test/test_range_assim.cc
)

# Set smaller file size for range test to speed up CI
set_tests_properties(cae_range_assim PROPERTIES
  ENVIRONMENT "CHI_WITH_RUNTIME=1;CHI_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/wrp_config.yaml;TEST_FILE_SIZE=10;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"  # 10MB instead of default
)

#------------------------------------------------------------------------------
# Test 3: Error Handling and Validation
#------------------------------------------------------------------------------
# Tests error cases and validation logic
# - Invalid source paths
# - Invalid format types
# - Missing required fields
# - Malformed URIs
#------------------------------------------------------------------------------
add_cae_unit_test(cae_error_handling
  ${CMAKE_CURRENT_SOURCE_DIR}/error_test/test_error_handling.cc
)

#------------------------------------------------------------------------------
# Test 4: CAE Comprehensive Coverage Tests
#------------------------------------------------------------------------------
# Comprehensive tests for CAE Core APIs to improve code coverage
# - CAE client initialization and pool creation
# - AssimilationCtx constructors and serialization
# - ParseOmni API (binary files, ranges, multiple contexts, error cases)
# - HDF5 dataset processing (basic and error cases)
# - URL parsing and format detection
# - Integration with CTE blob storage
#------------------------------------------------------------------------------
add_executable(test_cae_comprehensive
  ${CMAKE_CURRENT_SOURCE_DIR}/test_cae_comprehensive.cc
)

target_include_directories(test_cae_comprehensive
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
)

target_link_libraries(test_cae_comprehensive
  PRIVATE
    wrp_cae_core_client
    wrp_cte_core_client
    chimaera::admin_runtime
    chimaera::bdev_runtime
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Install test binary
install(TARGETS test_cae_comprehensive DESTINATION bin)

# Register individual test cases with CTest
add_test(NAME CAE_ClientInit COMMAND test_cae_comprehensive "[cae][core][init]")
add_test(NAME CAE_AssimilationCtx_Default COMMAND test_cae_comprehensive "[cae][ctx]")
add_test(NAME CAE_AssimilationCtx_Full COMMAND test_cae_comprehensive "[cae][ctx]")
add_test(NAME CAE_AssimilationCtx_Patterns COMMAND test_cae_comprehensive "[cae][ctx]")
add_test(NAME CAE_ParseOmni_Binary COMMAND test_cae_comprehensive "[cae][parseomni][binary]")
add_test(NAME CAE_ParseOmni_BinaryRange COMMAND test_cae_comprehensive "[cae][parseomni][binary][range]")
add_test(NAME CAE_ParseOmni_Multi COMMAND test_cae_comprehensive "[cae][parseomni][multi]")
add_test(NAME CAE_ParseOmni_Empty COMMAND test_cae_comprehensive "[cae][parseomni][error]")
add_test(NAME CAE_ProcessHdf5_Basic COMMAND test_cae_comprehensive "[cae][hdf5]")
add_test(NAME CAE_ProcessHdf5_NonExistent COMMAND test_cae_comprehensive "[cae][hdf5][error]")
add_test(NAME CAE_ProcessHdf5_EmptyPath COMMAND test_cae_comprehensive "[cae][hdf5][error]")
add_test(NAME CAE_Serialization COMMAND test_cae_comprehensive "[cae][ctx][serialization]")
add_test(NAME CAE_VerifyBinaryData COMMAND test_cae_comprehensive "[cae][integration][binary]")
add_test(NAME CAE_FileURL COMMAND test_cae_comprehensive "[cae][url]")
add_test(NAME CAE_IOWarpURL COMMAND test_cae_comprehensive "[cae][url]")
add_test(NAME CAE_BinaryFormat COMMAND test_cae_comprehensive "[cae][format]")
add_test(NAME CAE_HDF5Format COMMAND test_cae_comprehensive "[cae][format]")
add_test(NAME CAE_UnknownFormat COMMAND test_cae_comprehensive "[cae][format]")

# Set test properties
set_tests_properties(
  CAE_ClientInit CAE_AssimilationCtx_Default CAE_AssimilationCtx_Full
  CAE_AssimilationCtx_Patterns CAE_ParseOmni_Binary CAE_ParseOmni_BinaryRange
  CAE_ParseOmni_Multi CAE_ParseOmni_Empty CAE_ProcessHdf5_Basic
  CAE_ProcessHdf5_NonExistent CAE_ProcessHdf5_EmptyPath CAE_Serialization
  CAE_VerifyBinaryData CAE_FileURL CAE_IOWarpURL CAE_BinaryFormat
  CAE_HDF5Format CAE_UnknownFormat
  PROPERTIES
    TIMEOUT 120
    LABELS "coverage;comprehensive;msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

#------------------------------------------------------------------------------
# Test 5: HDF5 Format Assimilation (Conditional)
#------------------------------------------------------------------------------
# Tests HDF5 hierarchical data format support
# - Creates HDF5 file with multiple datasets
# - Tests dataset discovery and transfer
# - Validates metadata and tensor information
# - Tests hierarchical group structures
# Note: Requires WRP_CORE_ENABLE_HDF5=ON
#------------------------------------------------------------------------------
if(HDF5_ENABLED)
  add_cae_unit_test(cae_hdf5_assim
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_assim/test_hdf5_assim.cc
    EXTRA_LIBS ${HDF5_LIBRARIES}
    EXTRA_INCLUDES ${HDF5_INCLUDE_DIRS}
  )

  # Set test labels
  set_tests_properties(cae_hdf5_assim PROPERTIES LABELS "format;hdf5;msan_skip")
endif()

#------------------------------------------------------------------------------
# Test 6: Summary Operator (Conditional)
#------------------------------------------------------------------------------
# Tests the SummaryOperator which reads a "description" blob and
# generates a 4-8 word summary via a local LLM inference endpoint.
# Requires: WRP_CAE_ENABLE_SUMMARY_OP=ON, CURL, nlohmann_json
# Runtime: CAE_SUMMARY_ENDPOINT and CAE_SUMMARY_MODEL env vars
# Note: Test SKIPs gracefully if env vars are not set
#------------------------------------------------------------------------------
if(SUMMARY_OP_ENABLED)
  # Compile summary_operator.cc directly to avoid pulling HDF5 transitive deps
  # from the runtime library (HDF5's libcurl has symbol version conflicts)
  add_executable(cae_summary_op
    ${CMAKE_CURRENT_SOURCE_DIR}/summary_op/test_summary_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/factory/summary_operator.cc
  )
  target_link_libraries(cae_summary_op
    PRIVATE
      wrp_cae_core_client
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      CURL::libcurl
      nlohmann_json::nlohmann_json
  )
  target_include_directories(cae_summary_op PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS cae_summary_op DESTINATION bin)
  add_test(NAME cae_summary_op COMMAND cae_summary_op)
  set_tests_properties(cae_summary_op PROPERTIES
    ENVIRONMENT "CHI_WITH_RUNTIME=1;CHI_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/wrp_config.yaml;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 300
    LABELS "operator;summary;msan_skip"
  )

  # Benchmark: SummaryOperator I/O + LLM latency (run manually)
  add_executable(cae_summary_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/summary_op/bench_summary_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/factory/summary_operator.cc
  )
  target_link_libraries(cae_summary_bench
    PRIVATE
      wrp_cae_core_client
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      CURL::libcurl
      nlohmann_json::nlohmann_json
  )
  target_include_directories(cae_summary_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS cae_summary_bench DESTINATION bin)

  # Benchmark: CTE Agent Memory — CTE Raw vs CTE+Summary (run manually)
  add_executable(cae_memory_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/summary_op/bench_cte_memory.cc
  )
  target_link_libraries(cae_memory_bench
    PRIVATE
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      CURL::libcurl
      nlohmann_json::nlohmann_json
  )
  target_include_directories(cae_memory_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS cae_memory_bench DESTINATION bin)

  # Benchmark: Knowledge Graph Pipeline — full end-to-end (run manually)
  # Requires both SUMMARY_OP and KNOWLEDGE_GRAPH enabled
  if(WRP_CTE_ENABLE_KNOWLEDGE_GRAPH)
    add_executable(cae_kg_bench
      ${CMAKE_CURRENT_SOURCE_DIR}/summary_op/bench_kg_pipeline.cc
    )
    target_link_libraries(cae_kg_bench
      PRIVATE
        wrp_cte_core_client
        chimaera::cxx
        hshm::configure
        CURL::libcurl
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(cae_kg_bench PRIVATE WRP_CTE_ENABLE_KNOWLEDGE_GRAPH)
    target_include_directories(cae_kg_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    install(TARGETS cae_kg_bench DESTINATION bin)
  endif()
endif()

# Benchmark: HDF5 Dataset Discovery via Knowledge Graph (run manually)
# Requires KNOWLEDGE_GRAPH enabled
if(WRP_CTE_ENABLE_KNOWLEDGE_GRAPH)
  add_executable(cae_hdf5_discovery_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_assim/bench_hdf5_discovery.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/factory/summary_operator.cc
  )
  target_link_libraries(cae_hdf5_discovery_bench
    PRIVATE
      wrp_cae_core_client
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      nlohmann_json::nlohmann_json
      CURL::libcurl
  )
  target_compile_definitions(cae_hdf5_discovery_bench PRIVATE
    WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
    WRP_CAE_ENABLE_SUMMARY_OP
  )
  target_include_directories(cae_hdf5_discovery_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  install(TARGETS cae_hdf5_discovery_bench DESTINATION bin)

  # KG Backend comparison test (standalone, no CTE runtime needed)
  add_executable(test_kg_backends
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_assim/test_kg_backends.cc
  )
  target_link_libraries(test_kg_backends
    PRIVATE
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      nlohmann_json::nlohmann_json
  )
  target_compile_definitions(test_kg_backends PRIVATE
    WRP_CTE_ENABLE_KNOWLEDGE_GRAPH
  )
  target_include_directories(test_kg_backends PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  # Optional backends
  option(WRP_CTE_KG_ELASTICSEARCH "Enable Elasticsearch KG backend" OFF)
  option(WRP_CTE_KG_NEO4J "Enable Neo4j KG backend" OFF)
  option(WRP_CTE_KG_REDIS "Enable Redis KG backend" OFF)
  option(WRP_CTE_KG_QDRANT "Enable Qdrant KG backend" OFF)

  if(WRP_CTE_KG_ELASTICSEARCH OR WRP_CTE_KG_NEO4J OR WRP_CTE_KG_QDRANT)
    target_include_directories(test_kg_backends PRIVATE
      ${CMAKE_SOURCE_DIR}/external/httplib
    )
  endif()

  if(WRP_CTE_KG_ELASTICSEARCH)
    target_compile_definitions(test_kg_backends PRIVATE WRP_CTE_KG_ELASTICSEARCH)
  endif()
  if(WRP_CTE_KG_NEO4J)
    target_compile_definitions(test_kg_backends PRIVATE WRP_CTE_KG_NEO4J)
  endif()
  if(WRP_CTE_KG_REDIS)
    find_library(HIREDIS_LIB NAMES hiredis
      PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
      PATH_SUFFIXES hiredis)
    if(NOT HIREDIS_LIB)
      # Try versioned .so on systems without symlink
      find_file(HIREDIS_LIB NAMES libhiredis.so.0.14 libhiredis.so.1
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
    endif()
    if(HIREDIS_LIB)
      target_compile_definitions(test_kg_backends PRIVATE WRP_CTE_KG_REDIS)
      target_include_directories(test_kg_backends PRIVATE
        ${CMAKE_SOURCE_DIR}/external
      )
      target_link_libraries(test_kg_backends PRIVATE ${HIREDIS_LIB})
      message(STATUS "Found hiredis: ${HIREDIS_LIB}")
    else()
      message(WARNING "hiredis library not found, Redis backend disabled")
    endif()
  endif()
  if(WRP_CTE_KG_QDRANT)
    target_compile_definitions(test_kg_backends PRIVATE WRP_CTE_KG_QDRANT)
  endif()

  option(WRP_CTE_KG_ZEP "Enable Zep KG backend" OFF)
  if(WRP_CTE_KG_ZEP)
    target_compile_definitions(test_kg_backends PRIVATE WRP_CTE_KG_ZEP)
  endif()

  install(TARGETS test_kg_backends DESTINATION bin)
endif()

#------------------------------------------------------------------------------
# Globus Transfer Integration (Conditional) - REMOVED
#------------------------------------------------------------------------------
# Globus integration testing has been moved to integration tests only
# due to the complexity of OAuth2 authentication and collection-specific scopes.
# See test/integration/globus_matsci/ for Globus integration testing.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Custom CTest Configuration
#------------------------------------------------------------------------------

# Create test labels for selective execution
# Skip under MSan: CAE tests use uninstrumented HDF5/yaml-cpp/ZMQ deps and
# the chimaera runtime which involves shared memory and forked processes.
set_tests_properties(cae_binary_assim PROPERTIES LABELS "core;basic;msan_skip")
set_tests_properties(cae_range_assim PROPERTIES LABELS "core;advanced;msan_skip")
set_tests_properties(cae_error_handling PROPERTIES LABELS "core;validation;msan_skip")

#------------------------------------------------------------------------------
# Test Summary Target
#------------------------------------------------------------------------------
# Custom target to run all tests with verbose output
add_custom_target(run_unit_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running all CAE unit tests with verbose output"
  USES_TERMINAL
)

# Custom target to run only core tests
add_custom_target(run_core_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose -L core
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running core CAE unit tests"
  USES_TERMINAL
)

#------------------------------------------------------------------------------
# Print Configuration Summary
#------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "CAE Unit Tests Configuration:")
message(STATUS "  Binary Assimilation Test: ENABLED")
message(STATUS "  Range Transfer Test:      ENABLED")
message(STATUS "  Error Handling Test:      ENABLED")
if(HDF5_ENABLED)
  message(STATUS "  HDF5 Format Test:         ENABLED")
else()
  message(STATUS "  HDF5 Format Test:         DISABLED (set -DWRP_CORE_ENABLE_HDF5=ON to enable)")
endif()
if(SUMMARY_OP_ENABLED)
  message(STATUS "  Summary Operator Test:    ENABLED")
else()
  message(STATUS "  Summary Operator Test:    DISABLED (set -DWRP_CAE_ENABLE_SUMMARY_OP=ON to enable)")
endif()
message(STATUS "")
message(STATUS "Run tests with:")
message(STATUS "  make test                 - Run all tests")
message(STATUS "  make run_unit_tests       - Run all tests with verbose output")
message(STATUS "  make run_core_tests       - Run only core tests")
message(STATUS "  ctest -L core             - Run tests with 'core' label")
if(HDF5_ENABLED)
  message(STATUS "  ctest -L hdf5             - Run tests with 'hdf5' label")
endif()
message(STATUS "")
