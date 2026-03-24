# CMakeLists.txt for the GPU Runtime Latency Benchmark
#
# Builds bench_gpu_runtime, a benchmark that measures round-trip latency
# and throughput of the Chimaera GPU work orchestrator.
#
# When CUDA/ROCm is available, the GPU client kernel is compiled separately
# as a CUDA object library and linked into the final executable.

cmake_minimum_required(VERSION 3.18)

set(BENCH_TARGET bench_gpu_runtime)
set(BENCH_CPU_SOURCE bench_gpu_runtime.cc)
set(BENCH_GPU_SOURCE bench_gpu_runtime_gpu.cc)

if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
  # Compile GPU kernel source as CUDA by copying to a cuda/ subdir
  set(BENCH_GPU_CUDA_SOURCE
      ${CMAKE_CURRENT_BINARY_DIR}/cuda/${BENCH_GPU_SOURCE})
  configure_file(${BENCH_GPU_SOURCE} ${BENCH_GPU_CUDA_SOURCE} COPYONLY)
  set_source_files_properties(${BENCH_GPU_CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)

  # Object library for GPU kernels (CUDA compilation)
  add_library(${BENCH_TARGET}_gpu_kernels OBJECT ${BENCH_GPU_CUDA_SOURCE})
  target_include_directories(${BENCH_TARGET}_gpu_kernels PRIVATE
    ${CHIMAERA_ROOT}/include
    ${CHIMAERA_ROOT}/modules/admin/include
    ${CHIMAERA_ROOT}/modules/bdev/include
    ${CHIMAERA_ROOT}/modules/MOD_NAME/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
  )
  target_link_libraries(${BENCH_TARGET}_gpu_kernels PRIVATE hshm::cuda_cxx)
  set_target_properties(${BENCH_TARGET}_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
  )
  target_compile_options(${BENCH_TARGET}_gpu_kernels PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  # Main executable: CPU source + GPU object library
  add_executable(${BENCH_TARGET}
    ${BENCH_CPU_SOURCE}
    $<TARGET_OBJECTS:${BENCH_TARGET}_gpu_kernels>
  )

  # Disable CUDA in the CPU source to avoid __device__ errors from CXX compiler.
  # Define BENCH_GPU_KERNELS_COMPILED so the CPU driver knows GPU kernels are linked in.
  set_source_files_properties(${BENCH_CPU_SOURCE} PROPERTIES
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

  target_link_libraries(${BENCH_TARGET}
    chimaera::cxx                     # Core Chimaera runtime + admin
    chimaera_cxx_gpu                  # GPU work orchestrator (WorkOrchestrator::Resume)
    chimaera_MOD_NAME_client          # MOD_NAME module client
    wrp_cte::core_client              # CTE core client (PutBlob benchmark)
    hshm::cxx                         # HermesShm shared memory framework (CPU)
    hshm::cuda_cxx                    # HermesShm CUDA extensions (GPU kernels)
    ${CMAKE_THREAD_LIBS_INIT}
  )

  # Ensure the executable prefers in-tree shared libraries in ${CMAKE_BINARY_DIR}/bin.
  # Some CUDA link flows drop CMake's BUILD_RPATH; force an explicit $ORIGIN rpath.
  if(UNIX AND NOT APPLE)
    target_link_options(${BENCH_TARGET} PRIVATE "-Wl,-rpath,\\$ORIGIN")
  endif()

  set_target_properties(${BENCH_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "$ORIGIN"
  )

  message(STATUS "Building ${BENCH_TARGET} with CUDA support")

else()
  # Non-GPU build: compile CPU file only; GPU wrapper returns -200 (weak stub)
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
    wrp_cte::core_client
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

# Install benchmark binary
install(TARGETS ${BENCH_TARGET}
  DESTINATION bin
)

add_custom_target(run_bench_gpu_runtime
  COMMAND ${BENCH_TARGET} --test-case latency
  DEPENDS ${BENCH_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running GPU runtime latency benchmark with default parameters"
)
