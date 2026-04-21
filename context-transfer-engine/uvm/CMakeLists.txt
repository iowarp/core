cmake_minimum_required(VERSION 3.20)

# UVM module: Software-managed GPU demand paging
# Only builds when CUDA support is enabled
if(NOT WRP_CORE_ENABLE_CUDA)
    message(STATUS "CTE UVM: Skipping (WRP_CORE_ENABLE_CUDA is OFF)")
    return()
endif()

message(STATUS "CTE UVM: Building GPU virtual memory manager (CUDA enabled)")

# CUDAToolkit provides CUDA::cuda_driver (cuda.h) and CUDA::cudart
# (cuda_runtime.h) as imported targets with proper include directories.
# Linking them PUBLIC ensures any target that includes gpu_vmm.h transitively
# also gets the CUDA include paths (e.g., llama.cpp → weight_manager.h → gpu_vmm.h).
find_package(CUDAToolkit REQUIRED)

# Library: wrp_cte_uvm
# gpu_vmm.cu contains only host-side CUDA driver/runtime API calls — no
# __global__ or __device__ functions.  Compiling as CXX avoids any CUDA device
# compilation (nvcc -dc / nvlink) and the resulting FATBIN/PTX that NVHPC 25.5
# embeds with a PTX ISA version newer than what the Polaris CUDA driver supports.
set_source_files_properties(src/gpu_vmm.cu PROPERTIES LANGUAGE CXX)
add_library(wrp_cte_uvm SHARED src/gpu_vmm.cu)
set_target_properties(wrp_cte_uvm PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(wrp_cte_uvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(wrp_cte_uvm
    PUBLIC
        CUDA::cuda_driver   # cuda.h + libcuda.so (driver API)
        CUDA::cudart        # cuda_runtime.h + libcudart.so (runtime API)
)
target_link_options(wrp_cte_uvm PRIVATE "-Wl,-soname,libwrp_cte_uvm.so")

install(TARGETS wrp_cte_uvm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/wrp_cte/uvm
    DESTINATION include/wrp_cte
    FILES_MATCHING PATTERN "*.h"
)

# Test executable
if(WRP_CORE_ENABLE_TESTS)
    set_source_files_properties(test/test_gpu_vmm.cu PROPERTIES LANGUAGE CUDA)
    add_executable(test_gpu_vmm test/test_gpu_vmm.cu)
    # Do NOT use CUDA_SEPARABLE_COMPILATION ON here: that would invoke nvlink,
    # but nvlink uses a default sm_52 arch (the -gencode flags are not propagated
    # in this sub-project context), producing an empty device-link object and a
    # missing binary.
    #
    # Instead, compile without separable compilation so nvcc embeds the FATBIN
    # directly.  Inject the SASS-only architecture flag explicitly via
    # target_compile_options, bypassing the broken CUDA_ARCHITECTURES property.
    # With code=[sm_80] (no compute_80 virtual), nvcc embeds ONLY sm_80 SASS
    # — no PTX, no "PTX compiled with unsupported toolchain" error at runtime.
    set_target_properties(test_gpu_vmm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared
        BUILD_RPATH "${CMAKE_BINARY_DIR}/bin")
    target_compile_options(test_gpu_vmm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_80,code=[sm_80]>)
    # nvcc may not forward CMake's BUILD_RPATH; add it explicitly via the linker.
    target_link_options(test_gpu_vmm PRIVATE
        "-Wl,-rpath,${CMAKE_BINARY_DIR}/bin")
    target_link_libraries(test_gpu_vmm PRIVATE wrp_cte_uvm cuda)
    add_test(NAME test_gpu_vmm COMMAND test_gpu_vmm)
    message(STATUS "CTE UVM: Test target 'test_gpu_vmm' added")
endif()
