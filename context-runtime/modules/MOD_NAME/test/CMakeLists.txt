# CMakeLists.txt for MOD_NAME module unit tests
cmake_minimum_required(VERSION 3.10)

# Flush correctness test executable
set(FLUSH_TEST_TARGET chimaera_flush_correctness_tests)
set(FLUSH_TEST_SOURCES
  test_flush_correctness.cc
)

# CoMutex test executable
set(COMUTEX_TEST_TARGET chimaera_comutex_tests)
set(COMUTEX_TEST_SOURCES
  test_comutex.cc
)

# Wait Functionality test executable
set(WAIT_FUNCTIONALITY_TEST_TARGET chimaera_wait_functionality_tests)
set(WAIT_FUNCTIONALITY_TEST_SOURCES
  test_wait_functionality.cc
)

# Streaming test executable
set(STREAMING_TEST_TARGET chimaera_streaming_tests)
set(STREAMING_TEST_SOURCES
  test_streaming.cc
)

# GPU Client Process test executable
set(GPU_CLIENT_PROCESS_TEST_TARGET chimaera_gpu_client_process_tests)
set(GPU_CLIENT_PROCESS_TEST_SOURCES
  test_gpu_client_process.cc
)

# GPU Submission test executable (Part 3)
set(GPU_SUBMISSION_TEST_TARGET chimaera_gpu_submission_tests)
set(GPU_SUBMISSION_TEST_CPU_SOURCES
  test_gpu_submission_cpu.cc
)
set(GPU_SUBMISSION_TEST_GPU_SOURCES
  test_gpu_submission_gpu.cc
)

# GPU Queue Stress test executable
set(GPU_QUEUE_STRESS_TEST_TARGET chimaera_gpu_queue_stress_tests)
set(GPU_QUEUE_STRESS_TEST_CPU_SOURCES
  test_gpu_queue_stress_cpu.cc
)
set(GPU_QUEUE_STRESS_TEST_GPU_SOURCES
  test_gpu_queue_stress_gpu.cc
)

# Create flush correctness test executable
add_executable(${FLUSH_TEST_TARGET} ${FLUSH_TEST_SOURCES})

target_include_directories(${FLUSH_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks
)

target_link_libraries(${FLUSH_TEST_TARGET}
  cxx                             # Core Chimaera runtime library
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for work tracking
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${FLUSH_TEST_TARGET} PRIVATE CHIMAERA_RUNTIME=1)

set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create CoMutex test executable
add_executable(${COMUTEX_TEST_TARGET} ${COMUTEX_TEST_SOURCES})

target_include_directories(${COMUTEX_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${COMUTEX_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for CoMutex/CoRwLock
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Wait Functionality test executable
add_executable(${WAIT_FUNCTIONALITY_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_SOURCES})

target_include_directories(${WAIT_FUNCTIONALITY_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${WAIT_FUNCTIONALITY_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for WaitTest tasks
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Streaming test executable
add_executable(${STREAMING_TEST_TARGET} ${STREAMING_TEST_SOURCES})

target_include_directories(${STREAMING_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${STREAMING_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for streaming tasks
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest integration if testing is enabled
if(WRP_CORE_ENABLE_TESTS)
  # Flush Correctness Tests
  add_test(
    NAME cr_flush_basic_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][admin]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_flush_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_flush_work_orchestrator_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][mod_name]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_flush_work_orchestrator_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_flush_stress_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_flush_stress_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Comprehensive flush test (runs all tests, no filter)
  add_test(
    NAME cr_all_flush_tests
    COMMAND ${FLUSH_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_flush_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # CoMutex Tests
  add_test(
    NAME cr_comutex_basic_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_comutex_concurrent_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][concurrent]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_concurrent_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_corwlock_basic_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_corwlock_readers_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][readers]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_readers_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_corwlock_writers_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][writers]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_writers_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_corwlock_interaction_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][interaction]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_interaction_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_tasknode_grouping_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[tasknode][grouping]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_tasknode_grouping_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_comutex_error_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][error]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_comutex_performance_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][performance]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_comutex_integration_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[integration]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_integration_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Comprehensive CoMutex test (runs all tests, no filter)
  add_test(
    NAME cr_all_comutex_tests
    COMMAND ${COMUTEX_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_comutex_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Wait Functionality Tests
  add_test(
    NAME cr_wait_test_basic_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_wait_test_recursive_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][recursive]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_recursive_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_wait_test_async_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][async]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_async_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_wait_test_edge_cases_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][edge_cases]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_edge_cases_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Comprehensive Wait functionality test (runs all tests, no filter)
  add_test(
    NAME cr_all_wait_functionality_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_wait_functionality_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Streaming Tests
  add_test(
    NAME cr_streaming_small_tests
    COMMAND ${STREAMING_TEST_TARGET} "[streaming][small]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_small_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_streaming_large_tests
    COMMAND ${STREAMING_TEST_TARGET} "[streaming][large]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_large_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(
    NAME cr_streaming_concurrent_tests
    COMMAND ${STREAMING_TEST_TARGET} "[streaming][concurrent]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_concurrent_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Comprehensive Streaming test (runs all tests, no filter)
  add_test(
    NAME cr_all_streaming_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_streaming_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # Set test properties
  set_tests_properties(
    cr_flush_basic_tests
    cr_flush_work_orchestrator_tests
    cr_flush_stress_tests
    cr_all_flush_tests
    cr_comutex_basic_tests
    cr_comutex_concurrent_tests
    cr_corwlock_basic_tests
    cr_corwlock_readers_tests
    cr_corwlock_writers_tests
    cr_corwlock_interaction_tests
    cr_tasknode_grouping_tests
    cr_comutex_error_tests
    cr_comutex_performance_tests
    cr_comutex_integration_tests
    cr_all_comutex_tests
    cr_wait_test_basic_tests
    cr_wait_test_recursive_tests
    cr_wait_test_async_tests
    cr_wait_test_edge_cases_tests
    cr_all_wait_functionality_tests
    cr_streaming_small_tests
    cr_streaming_large_tests
    cr_streaming_concurrent_tests
    cr_all_streaming_tests
    PROPERTIES
      TIMEOUT 180
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1"
  )

  # Mark all MOD_NAME tests as msan_skip (they use ZMQ which is uninstrumented)
  set_tests_properties(
    cr_flush_basic_tests
    cr_flush_work_orchestrator_tests
    cr_flush_stress_tests
    cr_all_flush_tests
    cr_comutex_basic_tests
    cr_comutex_concurrent_tests
    cr_corwlock_basic_tests
    cr_corwlock_readers_tests
    cr_corwlock_writers_tests
    cr_corwlock_interaction_tests
    cr_tasknode_grouping_tests
    cr_comutex_error_tests
    cr_comutex_performance_tests
    cr_comutex_integration_tests
    cr_all_comutex_tests
    cr_wait_test_basic_tests
    cr_wait_test_recursive_tests
    cr_wait_test_async_tests
    cr_wait_test_edge_cases_tests
    cr_all_wait_functionality_tests
    cr_streaming_small_tests
    cr_streaming_large_tests
    cr_streaming_concurrent_tests
    cr_all_streaming_tests
    PROPERTIES LABELS "msan_skip;asan_skip"
  )

  # GPU Client Process Tests
  add_test(
    NAME cr_gpu_client_process_tests
    COMMAND ${GPU_CLIENT_PROCESS_TEST_TARGET} "[gpu][client_process]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_gpu_client_process_tests PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1"
    LABELS "msan_skip;asan_skip"
  )
endif()

# Custom targets for MOD_NAME tests
add_custom_target(test_flush
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush correctness tests"
)

add_custom_target(test_flush_basic
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush basic tests"
)

add_custom_target(test_flush_work_orchestrator
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush WorkOrchestrator integration tests"
)

add_custom_target(test_flush_stress
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush stress tests"
)

add_custom_target(test_comutex
  COMMAND ${COMUTEX_TEST_TARGET}
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera CoMutex and CoRwLock tests"
)

add_custom_target(test_comutex_basic
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][basic]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera CoMutex basic tests"
)

add_custom_target(test_corwlock_basic
  COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][basic]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera CoRwLock basic tests"
)

add_custom_target(test_comutex_concurrent
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][concurrent]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera concurrency tests"
)

add_custom_target(test_comutex_performance
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][performance]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera synchronization performance tests"
)

add_custom_target(test_streaming
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera output streaming tests"
)

add_custom_target(test_streaming_small
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera small output tests"
)

add_custom_target(test_streaming_large
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera large output streaming tests"
)

add_custom_target(test_streaming_concurrent
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera concurrent streaming tests"
)

# GPU Client Process test executable (fork-based, CPU-only)
add_executable(${GPU_CLIENT_PROCESS_TEST_TARGET} ${GPU_CLIENT_PROCESS_TEST_SOURCES})

target_include_directories(${GPU_CLIENT_PROCESS_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${GPU_CLIENT_PROCESS_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)
if(WRP_CORE_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  target_link_libraries(${GPU_CLIENT_PROCESS_TEST_TARGET} CUDA::cudart)
endif()

set_target_properties(${GPU_CLIENT_PROCESS_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create GPU Submission test executable
# Configure for CUDA if available, otherwise build CPU-only version
if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
  # Create object library for GPU kernels (compiled as CUDA)
  set(GPU_SUBMISSION_TEST_CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${GPU_SUBMISSION_TEST_GPU_SOURCES})
  configure_file(${GPU_SUBMISSION_TEST_GPU_SOURCES} ${GPU_SUBMISSION_TEST_CUDA_SOURCE} COPYONLY)
  set_source_files_properties(${GPU_SUBMISSION_TEST_CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)

  add_library(${GPU_SUBMISSION_TEST_TARGET}_gpu_kernels OBJECT ${GPU_SUBMISSION_TEST_CUDA_SOURCE})
  target_include_directories(${GPU_SUBMISSION_TEST_TARGET}_gpu_kernels PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
  )
  target_link_libraries(${GPU_SUBMISSION_TEST_TARGET}_gpu_kernels PRIVATE hshm::cuda_cxx)
  set_target_properties(${GPU_SUBMISSION_TEST_TARGET}_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
  )
  target_compile_options(${GPU_SUBMISSION_TEST_TARGET}_gpu_kernels PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  # Create main executable with CPU sources (compiled as C++)
  add_executable(${GPU_SUBMISSION_TEST_TARGET}
    ${GPU_SUBMISSION_TEST_CPU_SOURCES}
    $<TARGET_OBJECTS:${GPU_SUBMISSION_TEST_TARGET}_gpu_kernels>
  )

  # CPU sources need CUDA disabled to avoid __device__ errors from CXX compiler.
  # Also strip --coverage: these targets link with CUDA linker (g++) which can't
  # resolve llvm_gc* coverage symbols from clang-compiled objects.
  set_source_files_properties(${GPU_SUBMISSION_TEST_CPU_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-UHSHM_ENABLE_CUDA;-DHSHM_ENABLE_CUDA=0"
  )

  target_include_directories(${GPU_SUBMISSION_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test  # For simple_test.h
    ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
    ${CMAKE_CURRENT_SOURCE_DIR}  # For accessing original source directory
  )

  target_link_libraries(${GPU_SUBMISSION_TEST_TARGET}
    chimaera_admin_runtime          # Admin module runtime
    chimaera_admin_client           # Admin module client
    chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for GpuSubmit tasks
    chimaera_MOD_NAME_client        # MOD_NAME module client
    hshm::cxx                       # HermesShm library (CPU-only for main executable)
    hshm::cuda_cxx                  # HermesShm CUDA library (for GPU kernels via object library)
    ${CMAKE_THREAD_LIBS_INIT}       # Threading support
  )

  set_target_properties(${GPU_SUBMISSION_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
  )

  set_target_properties(${GPU_SUBMISSION_TEST_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  if(WRP_CORE_ENABLE_COVERAGE)
    set_property(TARGET ${GPU_SUBMISSION_TEST_TARGET} PROPERTY COMPILE_OPTIONS "")
    set_property(TARGET ${GPU_SUBMISSION_TEST_TARGET} PROPERTY LINK_OPTIONS "")
  endif()

  message(STATUS "Building GPU submission tests with CUDA support")
else()
  # CPU-only version (just the CPU test file, GPU kernels won't be compiled)
  add_executable(${GPU_SUBMISSION_TEST_TARGET} ${GPU_SUBMISSION_TEST_CPU_SOURCES})

  target_include_directories(${GPU_SUBMISSION_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test  # For simple_test.h
    ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
  )

  target_link_libraries(${GPU_SUBMISSION_TEST_TARGET}
    chimaera_admin_runtime          # Admin module runtime
    chimaera_admin_client           # Admin module client
    chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for GpuSubmit tasks
    chimaera_MOD_NAME_client        # MOD_NAME module client
    hshm::cxx                       # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}       # Threading support
  )

  set_target_properties(${GPU_SUBMISSION_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  set_target_properties(${GPU_SUBMISSION_TEST_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  message(STATUS "Building GPU submission tests without CUDA support (CPU-only)")
endif()

add_custom_target(test_gpu_submission
  COMMAND ${GPU_SUBMISSION_TEST_TARGET}
  DEPENDS ${GPU_SUBMISSION_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera Part 3 GPU submission tests"
)

# GPU Queue Stress test build rules
if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
  # GPU kernels compiled as CUDA
  set(GPU_QUEUE_STRESS_CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${GPU_QUEUE_STRESS_TEST_GPU_SOURCES})
  configure_file(${GPU_QUEUE_STRESS_TEST_GPU_SOURCES} ${GPU_QUEUE_STRESS_CUDA_SOURCE} COPYONLY)
  set_source_files_properties(${GPU_QUEUE_STRESS_CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)

  add_library(${GPU_QUEUE_STRESS_TEST_TARGET}_gpu_kernels OBJECT ${GPU_QUEUE_STRESS_CUDA_SOURCE})
  target_include_directories(${GPU_QUEUE_STRESS_TEST_TARGET}_gpu_kernels PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
  )
  target_link_libraries(${GPU_QUEUE_STRESS_TEST_TARGET}_gpu_kernels PRIVATE hshm::cuda_cxx)
  set_target_properties(${GPU_QUEUE_STRESS_TEST_TARGET}_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
  )
  target_compile_options(${GPU_QUEUE_STRESS_TEST_TARGET}_gpu_kernels PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  add_executable(${GPU_QUEUE_STRESS_TEST_TARGET}
    ${GPU_QUEUE_STRESS_TEST_CPU_SOURCES}
    $<TARGET_OBJECTS:${GPU_QUEUE_STRESS_TEST_TARGET}_gpu_kernels>
  )

  set_source_files_properties(${GPU_QUEUE_STRESS_TEST_CPU_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-UHSHM_ENABLE_CUDA;-DHSHM_ENABLE_CUDA=0"
  )

  target_include_directories(${GPU_QUEUE_STRESS_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${GPU_QUEUE_STRESS_TEST_TARGET}
    chimaera_admin_runtime
    chimaera_admin_client
    chimaera_MOD_NAME_runtime
    chimaera_MOD_NAME_client
    hshm::cxx
    hshm::cuda_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${GPU_QUEUE_STRESS_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  if(WRP_CORE_ENABLE_COVERAGE)
    set_property(TARGET ${GPU_QUEUE_STRESS_TEST_TARGET} PROPERTY COMPILE_OPTIONS "")
    set_property(TARGET ${GPU_QUEUE_STRESS_TEST_TARGET} PROPERTY LINK_OPTIONS "")
  endif()
  message(STATUS "Building GPU queue stress tests with CUDA support")
else()
  add_executable(${GPU_QUEUE_STRESS_TEST_TARGET} ${GPU_QUEUE_STRESS_TEST_CPU_SOURCES})

  target_include_directories(${GPU_QUEUE_STRESS_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
  )

  target_link_libraries(${GPU_QUEUE_STRESS_TEST_TARGET}
    chimaera_admin_runtime
    chimaera_admin_client
    chimaera_MOD_NAME_runtime
    chimaera_MOD_NAME_client
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${GPU_QUEUE_STRESS_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  message(STATUS "Building GPU queue stress tests without CUDA support (CPU-only)")
endif()

add_custom_target(test_gpu_queue_stress
  COMMAND ${GPU_QUEUE_STRESS_TEST_TARGET}
  DEPENDS ${GPU_QUEUE_STRESS_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running GPU queue stress tests (64 client warps vs 1 RT warp)"
)

add_custom_target(test_gpu_client_process
  COMMAND ${GPU_CLIENT_PROCESS_TEST_TARGET}
  DEPENDS ${GPU_CLIENT_PROCESS_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera GPU client process tests"
)

# GPU Parallelism test executable (cross-warp validation)
set(GPU_PARALLELISM_TEST_TARGET chimaera_gpu_parallelism_tests)
set(GPU_PARALLELISM_TEST_CPU_SOURCES
  test_gpu_parallelism_cpu.cc
)
set(GPU_PARALLELISM_TEST_GPU_SOURCES
  test_gpu_parallelism_gpu.cc
)

if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
  set(GPU_PARALLELISM_CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${GPU_PARALLELISM_TEST_GPU_SOURCES})
  configure_file(${GPU_PARALLELISM_TEST_GPU_SOURCES} ${GPU_PARALLELISM_CUDA_SOURCE} COPYONLY)
  set_source_files_properties(${GPU_PARALLELISM_CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)

  add_library(${GPU_PARALLELISM_TEST_TARGET}_gpu_kernels OBJECT ${GPU_PARALLELISM_CUDA_SOURCE})
  target_include_directories(${GPU_PARALLELISM_TEST_TARGET}_gpu_kernels PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
  )
  target_link_libraries(${GPU_PARALLELISM_TEST_TARGET}_gpu_kernels PRIVATE hshm::cuda_cxx)
  set_target_properties(${GPU_PARALLELISM_TEST_TARGET}_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
  )
  target_compile_options(${GPU_PARALLELISM_TEST_TARGET}_gpu_kernels PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  add_executable(${GPU_PARALLELISM_TEST_TARGET}
    ${GPU_PARALLELISM_TEST_CPU_SOURCES}
    $<TARGET_OBJECTS:${GPU_PARALLELISM_TEST_TARGET}_gpu_kernels>
  )

  set_source_files_properties(${GPU_PARALLELISM_TEST_CPU_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-UHSHM_ENABLE_CUDA;-DHSHM_ENABLE_CUDA=0"
  )

  target_include_directories(${GPU_PARALLELISM_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${GPU_PARALLELISM_TEST_TARGET}
    chimaera_admin_runtime
    chimaera_admin_client
    chimaera_MOD_NAME_runtime
    chimaera_MOD_NAME_client
    hshm::cxx
    hshm::cuda_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${GPU_PARALLELISM_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  if(WRP_CORE_ENABLE_COVERAGE)
    set_property(TARGET ${GPU_PARALLELISM_TEST_TARGET} PROPERTY COMPILE_OPTIONS "")
    set_property(TARGET ${GPU_PARALLELISM_TEST_TARGET} PROPERTY LINK_OPTIONS "")
  endif()
  message(STATUS "Building GPU parallelism tests with CUDA support")
else()
  add_executable(${GPU_PARALLELISM_TEST_TARGET} ${GPU_PARALLELISM_TEST_CPU_SOURCES})

  target_include_directories(${GPU_PARALLELISM_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
  )

  target_link_libraries(${GPU_PARALLELISM_TEST_TARGET}
    chimaera_admin_runtime
    chimaera_admin_client
    chimaera_MOD_NAME_runtime
    chimaera_MOD_NAME_client
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${GPU_PARALLELISM_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  message(STATUS "Building GPU parallelism tests without CUDA support (CPU-only)")
endif()

add_custom_target(test_gpu_parallelism
  COMMAND ${GPU_PARALLELISM_TEST_TARGET}
  DEPENDS ${GPU_PARALLELISM_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running GPU cross-warp parallelism validation tests"
)

if(WRP_CORE_ENABLE_TESTS)
  # Run cross-warp and single-warp tests as separate processes to avoid
  # GPU state corruption between test cases (cross-warp dispatch leaves
  # persistent kernel state that affects subsequent GPU operations).
  add_test(
    NAME cr_gpu_parallelism_cross_warp
    COMMAND ${GPU_PARALLELISM_TEST_TARGET} "[cross_warp]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_gpu_parallelism_cross_warp PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CHI_GPU_BLOCKS=64;CHI_GPU_THREADS=32"
    LABELS "msan_skip;asan_skip"
    WILL_FAIL TRUE
  )
  add_test(
    NAME cr_gpu_parallelism_single_warp
    COMMAND ${GPU_PARALLELISM_TEST_TARGET} "[single_warp]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_gpu_parallelism_single_warp PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CHI_GPU_BLOCKS=64;CHI_GPU_THREADS=32"
    LABELS "msan_skip;asan_skip"
  )
endif()

# Install GPU parallelism test
install(TARGETS ${GPU_PARALLELISM_TEST_TARGET}
  RUNTIME DESTINATION bin
)

# Migrate test executable (integration test binary for container migration)
set(MIGRATE_TEST_TARGET chimaera_migrate_tests)
set(MIGRATE_TEST_SOURCES
  ${CHIMAERA_ROOT}/test/integration/migrate/test_migrate.cc
)

add_executable(${MIGRATE_TEST_TARGET} ${MIGRATE_TEST_SOURCES})

target_include_directories(${MIGRATE_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${MIGRATE_TEST_TARGET}
  chimaera_admin_runtime
  chimaera_admin_client
  chimaera_MOD_NAME_runtime
  chimaera_MOD_NAME_client
  hshm::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${MIGRATE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Recovery test executable (integration test binary for node failure recovery)
set(RECOVERY_TEST_TARGET chimaera_recovery_tests)
set(RECOVERY_TEST_SOURCES
  ${CHIMAERA_ROOT}/test/integration/recovery/test_recovery.cc
)

add_executable(${RECOVERY_TEST_TARGET} ${RECOVERY_TEST_SOURCES})

target_include_directories(${RECOVERY_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${RECOVERY_TEST_TARGET}
  chimaera_admin_runtime
  chimaera_admin_client
  chimaera_MOD_NAME_runtime
  chimaera_MOD_NAME_client
  hshm::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${RECOVERY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Reconnect test executable (integration test binary for client failover)
set(RECONNECT_TEST_TARGET chimaera_reconnect_tests)
set(RECONNECT_TEST_SOURCES
  ${CHIMAERA_ROOT}/test/integration/reconnect/test_reconnect.cc
)

add_executable(${RECONNECT_TEST_TARGET} ${RECONNECT_TEST_SOURCES})

target_include_directories(${RECONNECT_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${RECONNECT_TEST_TARGET}
  chimaera_admin_runtime
  chimaera_admin_client
  chimaera_MOD_NAME_runtime
  chimaera_MOD_NAME_client
  hshm::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${RECONNECT_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install test executables
install(TARGETS ${FLUSH_TEST_TARGET} ${COMUTEX_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_TARGET} ${STREAMING_TEST_TARGET} ${GPU_SUBMISSION_TEST_TARGET} ${GPU_CLIENT_PROCESS_TEST_TARGET} ${MIGRATE_TEST_TARGET} ${RECOVERY_TEST_TARGET} ${RECONNECT_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "MOD_NAME module tests configured:")
message(STATUS "  Flush test target: ${FLUSH_TEST_TARGET}")
message(STATUS "  CoMutex test target: ${COMUTEX_TEST_TARGET}")
message(STATUS "  Wait Functionality test target: ${WAIT_FUNCTIONALITY_TEST_TARGET}")
message(STATUS "  Streaming test target: ${STREAMING_TEST_TARGET}")
message(STATUS "  GPU Submission test target: ${GPU_SUBMISSION_TEST_TARGET}")
message(STATUS "  GPU Client Process test target: ${GPU_CLIENT_PROCESS_TEST_TARGET}")
message(STATUS "  Migrate test target: ${MIGRATE_TEST_TARGET}")
message(STATUS "  Recovery test target: ${RECOVERY_TEST_TARGET}")
message(STATUS "  Reconnect test target: ${RECONNECT_TEST_TARGET}")
