cmake_minimum_required(VERSION 3.20)

# GPU workload benchmarks — standalone CUDA executables that compare
# BaM page cache vs direct GPU↔DRAM baselines on real-world workloads.
#
# Requires CUDA and BaM (WRP_CORE_ENABLE_BAM=ON).

if(NOT WRP_CORE_ENABLE_CUDA)
  message(STATUS "GPU workload benchmarks: skipping (CUDA not enabled)")
  return()
endif()

if(NOT TARGET bam)
  message(STATUS "GPU workload benchmarks: skipping (BaM not enabled)")
  return()
endif()

set(_GPU_BENCH_LIBS bam)
set(_GPU_BENCH_INCLUDES
  ${CMAKE_SOURCE_DIR}/external/bam/include
)

# Helper macro for standalone CUDA benchmark executables
macro(add_gpu_bench NAME)
  add_executable(${NAME} ${ARGN})
  target_link_libraries(${NAME} PRIVATE ${_GPU_BENCH_LIBS})
  target_include_directories(${NAME} PRIVATE ${_GPU_BENCH_INCLUDES})
  set_target_properties(${NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CXX_STANDARD 17
  )
  target_compile_options(${NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  install(TARGETS ${NAME} RUNTIME DESTINATION bin)
endmacro()

# 1. PageRank — iterative graph algorithm with irregular memory access
#    Inspired by BaM benchmarks/pagerank
add_gpu_bench(bench_gpu_pagerank pagerank.cu)

# 2. GNN feature loading — mini-batch feature gather from large store
#    Inspired by GIDS evaluation/homogenous_train
add_gpu_bench(bench_gpu_gnn_features gnn_features.cu)

# 3. Gray-Scott reaction-diffusion — stencil simulation with periodic I/O
#    Inspired by external/iowarp-gray-scott
add_gpu_bench(bench_gpu_gray_scott gray_scott.cu)
