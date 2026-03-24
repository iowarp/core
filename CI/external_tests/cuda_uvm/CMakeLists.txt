cmake_minimum_required(VERSION 3.20)
project(cuda_uvm_external_test LANGUAGES CXX CUDA)

# Find CUDA toolkit (provides CUDA::cuda_driver, CUDA::cudart)
find_package(CUDAToolkit REQUIRED)

# Find the installed wrp_cte_uvm library and headers.
# When installed via conda, these live under $CONDA_PREFIX.
find_library(WRP_CTE_UVM_LIB wrp_cte_uvm
    HINTS ENV CONDA_PREFIX
    PATH_SUFFIXES lib)

find_path(WRP_CTE_UVM_INCLUDE wrp_cte/uvm/gpu_vmm.h
    HINTS ENV CONDA_PREFIX
    PATH_SUFFIXES include)

if(NOT WRP_CTE_UVM_LIB)
    message(FATAL_ERROR "Could not find libwrp_cte_uvm. "
        "Set CMAKE_PREFIX_PATH or CONDA_PREFIX to the IOWarp install prefix.")
endif()

if(NOT WRP_CTE_UVM_INCLUDE)
    message(FATAL_ERROR "Could not find wrp_cte/uvm/gpu_vmm.h. "
        "Set CMAKE_PREFIX_PATH or CONDA_PREFIX to the IOWarp install prefix.")
endif()

message(STATUS "Found wrp_cte_uvm: ${WRP_CTE_UVM_LIB}")
message(STATUS "Found wrp_cte_uvm headers: ${WRP_CTE_UVM_INCLUDE}")

# Build a minimal CUDA program that links against wrp_cte_uvm
set_source_files_properties(test_link.cu PROPERTIES LANGUAGE CUDA)
add_executable(test_uvm_link test_link.cu)
set_target_properties(test_uvm_link PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared)
target_include_directories(test_uvm_link PRIVATE ${WRP_CTE_UVM_INCLUDE})
target_link_libraries(test_uvm_link PRIVATE
    ${WRP_CTE_UVM_LIB}
    CUDA::cuda_driver
    CUDA::cudart)
