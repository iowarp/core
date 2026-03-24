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
  add_cuda_executable(wrp_cte_gpu_bench TRUE
      wrp_cte_gpu_bench.cc
      INCLUDE_DIRS
          ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
          ${CMAKE_SOURCE_DIR}/context-runtime/include
          ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
          ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
          ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
          ${CMAKE_SOURCE_DIR}/context-runtime/compiler/include
          ${_CTE_GPU_BENCH_EXTRA_INCLUDES}
      LINK_LIBS
          wrp_cte::core_client
          chimaera::cxx
          hshm::cuda_cxx
  )

  install(TARGETS wrp_cte_gpu_bench RUNTIME DESTINATION bin)

  # Regression ctests — quick smoke tests for each GPU benchmark mode.
  # Use small data (1 warp, 4KB I/O, 1 iteration) to keep runtime < 30s.
  add_test(NAME cte_gpu_bench_putblob_gpu
    COMMAND wrp_cte_gpu_bench
      --test-case putblob_gpu
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 1)
  set_tests_properties(cte_gpu_bench_putblob_gpu PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_putblob_gpu_multiblock
    COMMAND wrp_cte_gpu_bench
      --test-case putblob_gpu
      --client-blocks 1 --client-threads 512
      --io-size 4k --iterations 2)
  set_tests_properties(cte_gpu_bench_putblob_gpu_multiblock PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_direct
    COMMAND wrp_cte_gpu_bench
      --test-case direct
      --client-blocks 1 --client-threads 32
      --io-size 4k --iterations 1)
  set_tests_properties(cte_gpu_bench_direct PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_cudamemcpy
    COMMAND wrp_cte_gpu_bench
      --test-case cudamemcpy
      --io-size 4k --iterations 1)
  set_tests_properties(cte_gpu_bench_cudamemcpy PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_putblob_gpu_stress
    COMMAND wrp_cte_gpu_bench
      --test-case putblob_gpu
      --rt-blocks 1 --rt-threads 128
      --client-blocks 1 --client-threads 128
      --io-size 16k --iterations 1024)
  set_tests_properties(cte_gpu_bench_putblob_gpu_stress PROPERTIES
    TIMEOUT 120 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_putblob_gpu_64warp
    COMMAND wrp_cte_gpu_bench
      --test-case putblob_gpu
      --rt-blocks 64 --rt-threads 32
      --client-blocks 8 --client-threads 256
      --io-size 1m --iterations 16)
  set_tests_properties(cte_gpu_bench_putblob_gpu_64warp PROPERTIES
    TIMEOUT 120 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_putblob_gpu_64warp_stress
    COMMAND wrp_cte_gpu_bench
      --test-case putblob_gpu
      --rt-blocks 64 --rt-threads 32
      --client-blocks 8 --client-threads 256
      --io-size 16k --iterations 1024)
  set_tests_properties(cte_gpu_bench_putblob_gpu_64warp_stress PROPERTIES
    TIMEOUT 600 LABELS "msan_skip")

  add_test(NAME cte_gpu_bench_alloc_test
    COMMAND wrp_cte_gpu_bench
      --test-case alloc_test
      --client-blocks 1 --client-threads 256)
  set_tests_properties(cte_gpu_bench_alloc_test PROPERTIES
    TIMEOUT 60 LABELS "msan_skip")

  # Set CHI_REPO_PATH so the ChiMod plugin loader finds module .so files
  set_tests_properties(
    cte_gpu_bench_putblob_gpu
    cte_gpu_bench_putblob_gpu_multiblock
    cte_gpu_bench_direct
    cte_gpu_bench_cudamemcpy
    cte_gpu_bench_putblob_gpu_stress
    cte_gpu_bench_putblob_gpu_64warp
    cte_gpu_bench_putblob_gpu_64warp_stress
    cte_gpu_bench_alloc_test
    PROPERTIES
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )
else()
  message(STATUS "Skipping wrp_cte_gpu_bench - CUDA not enabled")
endif()

# NOTE: Compression benchmark (wrp_cte_compress_bench) has been moved to
# context-transfer-engine/compressor/test/
