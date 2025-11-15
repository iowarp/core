# Benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# MPI is found in root CMakeLists.txt - benchmarks require MPI
if(NOT WRP_CORE_ENABLE_MPI)
  message(WARNING "CTE benchmarks require MPI but WRP_CORE_ENABLE_MPI is OFF - skipping benchmarks")
  return()
endif()

# Create wrp_cte_bench executable
add_executable(wrp_cte_bench
  wrp_cte_bench.cc
)

# Link required libraries
target_link_libraries(wrp_cte_bench
  wrp_cte::core_client
  chimaera::cxx
  MPI::MPI_CXX
)

# Set C++ standard
target_compile_features(wrp_cte_bench PRIVATE cxx_std_17)

# Install the benchmark executable
install(TARGETS wrp_cte_bench
  RUNTIME DESTINATION bin
)
