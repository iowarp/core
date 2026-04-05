# CMakeLists.txt for the GPU Runtime Latency Benchmark
#
# GPU kernels compiled with NVCC (via add_cuda_executable).

cmake_minimum_required(VERSION 3.18)

set(BENCH_TARGET bench_gpu_runtime)
set(BENCH_CPU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bench_gpu_runtime.cc)
set(BENCH_GPU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bench_gpu_runtime_gpu.cc)

if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
  find_package(CUDAToolkit QUIET)

  # Build as a single CUDA executable so device symbols resolve across
  # the GPU kernel file and chimaera_cxx_gpu.
  # Include module GPU runtime sources directly so device symbols resolve
  # at device-link time (NVCC can't device-link against shared lib symbols)
  set(BENCH_MODULE_GPU_SOURCES
    ${CHIMAERA_ROOT}/modules/MOD_NAME/src/MOD_NAME_runtime_gpu.cc
    ${CHIMAERA_ROOT}/modules/bdev/src/bdev_runtime_gpu.cc
    ${CHIMAERA_ROOT}/src/gpu/work_orchestrator_gpu.cc
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/src/core_runtime_gpu.cc
  )
  add_cuda_executable(${BENCH_TARGET} FALSE
    ${BENCH_GPU_SOURCE}
    ${BENCH_MODULE_GPU_SOURCES}
    INCLUDE_DIRS
      ${CHIMAERA_ROOT}/include
      ${CHIMAERA_ROOT}/modules/admin/include
      ${CHIMAERA_ROOT}/modules/bdev/include
      ${CHIMAERA_ROOT}/modules/MOD_NAME/include
      ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
      ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBS
      chimaera_cxx_gpu
      hshm::cuda_cxx
      chimaera::cxx
      chimaera_MOD_NAME_client
      $<$<BOOL:${WRP_CORE_ENABLE_CTE}>:wrp_cte::core_client>
      hshm::cxx
      CUDA::cudart
      ${CMAKE_THREAD_LIBS_INIT}
    DEFS
      BENCH_GPU_KERNELS_COMPILED=1
  )

  # Add the CPU source as a non-CUDA source
  target_sources(${BENCH_TARGET} PRIVATE ${BENCH_CPU_SOURCE})
  set_source_files_properties(${BENCH_CPU_SOURCE} PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS "-UHSHM_ENABLE_CUDA;-DHSHM_ENABLE_CUDA=0;-DBENCH_GPU_KERNELS_COMPILED=1"
  )

  target_include_directories(${BENCH_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/bdev/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  if(UNIX AND NOT APPLE)
    target_link_options(${BENCH_TARGET} PRIVATE "-Wl,-rpath,\\$ORIGIN")
  endif()

  set_target_properties(${BENCH_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "$ORIGIN"
  )

  message(STATUS "Building ${BENCH_TARGET} with CUDA support")
else()
  add_executable(${BENCH_TARGET} ${BENCH_CPU_SOURCE})
  target_include_directories(${BENCH_TARGET} PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/bdev/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
  )
  target_link_libraries(${BENCH_TARGET}
    chimaera::cxx
    chimaera_MOD_NAME_client
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(${BENCH_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  message(STATUS "Building ${BENCH_TARGET} without GPU support (CPU-only stub)")
endif()

install(TARGETS ${BENCH_TARGET} DESTINATION bin)
