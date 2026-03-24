# CMakeLists.txt for bdev module unit tests
cmake_minimum_required(VERSION 3.10)

# Bdev ChiMod test executable
set(BDEV_TEST_TARGET chimaera_bdev_chimod_tests)
set(BDEV_TEST_SOURCES
  test_bdev_chimod.cc
)

# Bdev GPU test executable
if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
  set(BDEV_GPU_TEST_TARGET chimaera_bdev_gpu_tests)
  set(BDEV_GPU_TEST_SOURCES test_bdev_gpucache.cc)
endif()

# Create bdev test executable
add_executable(${BDEV_TEST_TARGET} ${BDEV_TEST_SOURCES})

# Include directories for bdev tests
target_include_directories(${BDEV_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/bdev/include   # For bdev tasks and client
)

# Link against required libraries for bdev tests
target_link_libraries(${BDEV_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_bdev_runtime           # Bdev module runtime
  chimaera_bdev_client            # Bdev module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for bdev tests
set_target_properties(${BDEV_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# Copy bdev test executable to bin directory
set_target_properties(${BDEV_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# GPU bdev test executable
if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
  add_executable(${BDEV_GPU_TEST_TARGET} ${BDEV_GPU_TEST_SOURCES})
  target_include_directories(${BDEV_GPU_TEST_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/test
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/bdev/include
  )
  target_link_libraries(${BDEV_GPU_TEST_TARGET}
    chimaera_admin_runtime
    chimaera_admin_client
    chimaera_bdev_runtime
    chimaera_bdev_client
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  if(WRP_CORE_ENABLE_CUDA)
    target_link_libraries(${BDEV_GPU_TEST_TARGET} hshm::cuda_cxx)
    if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
      target_include_directories(${BDEV_GPU_TEST_TARGET} PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    # Need to link cudart explicitly since the test calls CUDA runtime APIs
    # through hshm::GpuApi wrappers (cudaMallocManaged, cudaFree, etc.)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      target_link_libraries(${BDEV_GPU_TEST_TARGET} CUDA::cudart)
    else()
      target_link_libraries(${BDEV_GPU_TEST_TARGET} cudart)
    endif()
  endif()
  set_target_properties(${BDEV_GPU_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endif()

# Enable CTest integration if testing is enabled
if(WRP_CORE_ENABLE_TESTS)
  # GPU bdev tests
  if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
    add_test(
      NAME cr_bdev_gpu_tests
      COMMAND ${BDEV_GPU_TEST_TARGET}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cr_bdev_gpu_tests PROPERTIES
      TIMEOUT 120
      LABELS "gpu;msan_skip"
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1"
    )
  endif()

  # Comprehensive bdev test (runs all tests; explicit TCP/IPC auto-skip
  # because default IPC mode is SHM and the skip guards filter them out)
  add_test(
    NAME cr_all_bdev_tests
    COMMAND ${BDEV_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_bdev_tests PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1"
  )

  # Dedicated explicit backend tests — each sets CHI_IPC_MODE so the
  # corresponding TEST_CASE runs while the other two auto-skip.
  add_test(
    NAME cr_bdev_explicit_backend_shm
    COMMAND ${BDEV_TEST_TARGET} "bdev_file_explicit_backend_shm"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_explicit_backend_shm PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CHI_IPC_MODE=SHM"
  )

  add_test(
    NAME cr_bdev_explicit_backend_tcp
    COMMAND ${BDEV_TEST_TARGET} "bdev_file_explicit_backend_tcp"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_explicit_backend_tcp PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CHI_IPC_MODE=TCP"
  )

  add_test(
    NAME cr_bdev_explicit_backend_ipc
    COMMAND ${BDEV_TEST_TARGET} "bdev_file_explicit_backend_ipc"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_explicit_backend_ipc PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CHI_IPC_MODE=IPC"
  )

  # Set test properties
  set_tests_properties(
    cr_all_bdev_tests
    cr_bdev_explicit_backend_shm
    cr_bdev_explicit_backend_tcp
    cr_bdev_explicit_backend_ipc
    PROPERTIES
      TIMEOUT 300
      LABELS "msan_skip"
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1"
  )
endif()

# Custom targets for bdev tests
add_custom_target(test_bdev
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev ChiMod tests"
)

add_custom_target(test_bdev_create
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev container creation tests"
)

add_custom_target(test_bdev_allocate
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev block allocation tests"
)

add_custom_target(test_bdev_io
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev I/O operation tests"
)

add_custom_target(test_bdev_async
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev asynchronous I/O tests"
)

add_custom_target(test_bdev_performance
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev performance metrics tests"
)

add_custom_target(test_bdev_error
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev error handling tests"
)

# Install test executable
install(TARGETS ${BDEV_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "Bdev module tests configured:")
message(STATUS "  Test target: ${BDEV_TEST_TARGET}")
message(STATUS "  Test sources: ${BDEV_TEST_SOURCES}")
