# Benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Create wrp_cte_bench executable
add_executable(wrp_cte_bench
  wrp_cte_bench.cc
)

# Link required libraries (MPI no longer required - uses worker threads instead)
target_link_libraries(wrp_cte_bench
  wrp_cte::core_client
  chimaera::cxx
)

# Set C++ standard
target_compile_features(wrp_cte_bench PRIVATE cxx_std_17)

# Install the benchmark executable
install(TARGETS wrp_cte_bench
  RUNTIME DESTINATION bin
)

# Create wrp_cte_score_bench executable (MPI-based score/demotion benchmark)
# Only build if MPI is enabled
if(WRP_CORE_ENABLE_MPI)
  add_executable(wrp_cte_score_bench
    wrp_cte_score_bench.cc
  )

  # Link required libraries including MPI
  target_link_libraries(wrp_cte_score_bench
    wrp_cte::core_client
    chimaera::cxx
    MPI::MPI_CXX
  )

  # Set C++ standard
  target_compile_features(wrp_cte_score_bench PRIVATE cxx_std_17)

  # Install the score benchmark executable
  install(TARGETS wrp_cte_score_bench
    RUNTIME DESTINATION bin
  )
else()
  message(STATUS "Skipping wrp_cte_score_bench - MPI not enabled (set WRP_CORE_ENABLE_MPI=ON)")
endif()

# GPU benchmark (requires CUDA) — single-file build via add_cuda_executable
if(WRP_CORE_ENABLE_CUDA)
  # Collect transitive include dirs from dependencies for Clang CUDA
  set(_CTE_GPU_BENCH_EXTRA_INCLUDES "")
  foreach(_dep yaml-cpp::yaml-cpp ZeroMQ::libzmq)
    if(TARGET "${_dep}")
      get_target_property(_dep_incs "${_dep}" INTERFACE_INCLUDE_DIRECTORIES)
      if(_dep_incs)
        list(APPEND _CTE_GPU_BENCH_EXTRA_INCLUDES ${_dep_incs})
      endif()
    endif()
  endforeach()
  # BaM include dirs and link libs (conditional)
  set(_BAM_INCLUDE_DIRS "")
  set(_BAM_LINK_LIBS "")
  set(_BAM_DEFS "")
  if(WRP_CORE_ENABLE_BAM AND TARGET bam)
    set(_BAM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/external/bam/include)
    set(_BAM_LINK_LIBS bam)
    set(_BAM_DEFS WRP_CORE_ENABLE_BAM)
  endif()

  # Host driver compiled by g++; uses CHI_CPU_IPC (not CHI_IPC).
  # GPU workload kernels are in test_gpu_initiated_kernels (shared CUDA
  # device-link unit with chimaera_cxx_gpu for completion flag visibility).
  add_executable(wrp_cte_gpu_bench wrp_cte_gpu_bench.cc)
  set_target_properties(wrp_cte_gpu_bench PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      LINKER_LANGUAGE CUDA
      POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(wrp_cte_gpu_bench PRIVATE
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/benchmark
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/benchmark/gpu
      ${CMAKE_SOURCE_DIR}/context-runtime/include
      ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
      ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
      ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
      ${CMAKE_SOURCE_DIR}/build/context-transport-primitives/src/include
      ${_CTE_GPU_BENCH_EXTRA_INCLUDES}
      ${_BAM_INCLUDE_DIRS}
  )
  target_link_libraries(wrp_cte_gpu_bench
      test_gpu_initiated_kernels
      wrp_cte_core_runtime
      wrp_cte_core_client
      chimaera::admin_runtime
      chimaera::admin_client
      chimaera::bdev_runtime
      chimaera::bdev_client
      hshm::cxx
      hermes_shm_cuda
      chimaera_cxx_gpu
      ${_BAM_LINK_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
  )
  if(WRP_CORE_ENABLE_CUDA AND CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    find_package(CUDAToolkit QUIET)
    set_property(TARGET wrp_cte_gpu_bench APPEND PROPERTY
      LINK_LIBRARIES "CUDA::cudart")
  endif()

  if(_BAM_DEFS)
    target_compile_definitions(wrp_cte_gpu_bench PRIVATE ${_BAM_DEFS})
  endif()

  install(TARGETS wrp_cte_gpu_bench RUNTIME DESTINATION bin)

  # Regression ctests — quick smoke tests for each GPU benchmark mode.
  # Use small data (1 warp, 4KB I/O, few iterations) to keep runtime < 30s.

  # Client overhead: single warp, 1 iteration (GPU-initiated PutBlob)
  add_test(NAME cte_gpu_bench_client_overhead
    COMMAND wrp_cte_gpu_bench
      --test-case client_overhead
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 1)
  set_tests_properties(cte_gpu_bench_client_overhead PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  # Client overhead: multi-warp (16 warps)
  add_test(NAME cte_gpu_bench_client_overhead_multiblock
    COMMAND wrp_cte_gpu_bench
      --test-case client_overhead
      --client-blocks 1 --client-threads 512
      --io-size 4k --iterations 2)
  set_tests_properties(cte_gpu_bench_client_overhead_multiblock PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  # Client overhead: CPU-initiated path
  add_test(NAME cte_gpu_bench_client_overhead_cpu
    COMMAND wrp_cte_gpu_bench
      --test-case client_overhead_cpu
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 1)
  set_tests_properties(cte_gpu_bench_client_overhead_cpu PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  # CUDA malloc baseline
  add_test(NAME cte_gpu_bench_cuda_malloc
    COMMAND wrp_cte_gpu_bench
      --test-case cuda_malloc
      --iterations 10)
  set_tests_properties(cte_gpu_bench_cuda_malloc PROPERTIES
    TIMEOUT 120 LABELS "msan_skip")

  # Client overhead: stress (many iterations, larger I/O)
  add_test(NAME cte_gpu_bench_stress
    COMMAND wrp_cte_gpu_bench
      --test-case client_overhead
      --client-blocks 1 --client-threads 128
      --io-size 16k --iterations 64)
  set_tests_properties(cte_gpu_bench_stress PROPERTIES
    TIMEOUT 120 LABELS "msan_skip")

  # BaM regression tests (only when BaM is enabled)
  set(_BAM_TEST_NAMES "")

  # Synthetic I/O benchmarks
  add_test(NAME cte_gpu_synthetic_hbm
    COMMAND wrp_cte_gpu_bench
      --test-case synthetic --workload-mode hbm
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 10)
  set_tests_properties(cte_gpu_synthetic_hbm PROPERTIES
    TIMEOUT 60 LABELS "gpu;msan_skip")

  add_test(NAME cte_gpu_synthetic_direct
    COMMAND wrp_cte_gpu_bench
      --test-case synthetic --workload-mode direct
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 10)
  set_tests_properties(cte_gpu_synthetic_direct PROPERTIES
    TIMEOUT 60 LABELS "gpu;msan_skip")

  add_test(NAME cte_gpu_synthetic_hbm_validate
    COMMAND wrp_cte_gpu_bench
      --test-case synthetic --workload-mode hbm
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 5 --validate)
  set_tests_properties(cte_gpu_synthetic_hbm_validate PROPERTIES
    TIMEOUT 60 LABELS "gpu;msan_skip")

  add_test(NAME cte_gpu_synthetic_cte
    COMMAND wrp_cte_gpu_bench
      --test-case synthetic --workload-mode cte
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 1)
  set_tests_properties(cte_gpu_synthetic_cte PROPERTIES
    TIMEOUT 60 LABELS "gpu;msan_skip")

  add_test(NAME cte_gpu_synthetic_cte_validate
    COMMAND wrp_cte_gpu_bench
      --test-case synthetic --workload-mode cte
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 1 --validate)
  set_tests_properties(cte_gpu_synthetic_cte_validate PROPERTIES
    TIMEOUT 60 LABELS "gpu;msan_skip")

  # BaM synthetic benchmark: only register as ctest when BaM NVMe is enabled
  # (BaM library may be compiled but the driver/device may not be available)
  if(WRP_CORE_ENABLE_BAM AND WRP_CORE_ENABLE_BAM_NVME AND TARGET bam)
    add_test(NAME cte_gpu_synthetic_bam
      COMMAND wrp_cte_gpu_bench
        --test-case synthetic --workload-mode bam
        --client-blocks 1 --client-threads 32
        --io-size 4k --iterations 10
        --bam-page-size 4k --bam-cache-pages 16)
    set_tests_properties(cte_gpu_synthetic_bam PROPERTIES
      TIMEOUT 60 LABELS "gpu;msan_skip")
    list(APPEND _BAM_TEST_NAMES cte_gpu_synthetic_bam)
  endif()

  # Set CHI_REPO_PATH so the ChiMod plugin loader finds module .so files
  set_tests_properties(
    cte_gpu_bench_client_overhead
    cte_gpu_bench_client_overhead_multiblock
    cte_gpu_bench_client_overhead_cpu
    cte_gpu_bench_cuda_malloc
    cte_gpu_bench_stress
    cte_gpu_synthetic_hbm
    cte_gpu_synthetic_direct
    cte_gpu_synthetic_hbm_validate
    cte_gpu_synthetic_cte
    cte_gpu_synthetic_cte_validate
    ${_BAM_TEST_NAMES}
    PROPERTIES
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )
else()
  message(STATUS "Skipping wrp_cte_gpu_bench - CUDA not enabled")
endif()

# NOTE: Compression benchmark (wrp_cte_compress_bench) has been moved to
# context-transfer-engine/compressor/test/
