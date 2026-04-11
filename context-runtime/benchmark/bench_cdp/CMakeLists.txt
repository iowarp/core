# Standalone CDP benchmark — compiled with NVCC (not Clang CUDA)
# because CUDA Dynamic Parallelism requires NVCC for <<<>>> in device code.

cmake_minimum_required(VERSION 3.18)

if(HSHM_ENABLE_CUDA)
  # Use CMake's native CUDA language (NVCC) directly.
  # The .cu extension tells CMake to compile with nvcc.
  add_executable(bench_cdp bench_cdp.cu)

  set_target_properties(bench_cdp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  # -rdc=true enables device-side runtime (<<<>>> in __global__).
  # CUDA_FORCE_CDP1_IF_SUPPORTED enables cudaDeviceSynchronize in device code
  # (CDP2 removes it; CDP1 mode re-enables it for sm < 90).
  target_compile_options(bench_cdp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -rdc=true>)
  target_compile_definitions(bench_cdp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:CUDA_FORCE_CDP1_IF_SUPPORTED __CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING>)

  # CDP requires cudadevrt for device-side runtime
  target_link_libraries(bench_cdp PRIVATE cudart cudadevrt)

  install(TARGETS bench_cdp DESTINATION bin)
  message(STATUS "Building bench_cdp (NVCC, CDP)")
endif()
