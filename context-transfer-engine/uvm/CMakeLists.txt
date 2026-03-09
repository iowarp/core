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
# gpu_vmm.cu is a single translation unit with one internal __global__ kernel
# (fillKernel).  CUDA_SEPARABLE_COMPILATION ON is required so CMake generates
# a proper host-link step for the shared library (otherwise the linker rule is
# a no-op and libwrp_cte_uvm.so is never written to disk).
set_source_files_properties(src/gpu_vmm.cu PROPERTIES LANGUAGE CUDA)
add_library(wrp_cte_uvm SHARED src/gpu_vmm.cu)
target_compile_options(wrp_cte_uvm PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-fPIC>)
# CUDA_SEPARABLE_COMPILATION ON is required so CMake generates a real host-link
# command for the shared library.  With it OFF, the CUDA_SHARED_LIBRARY_LINKER
# rule becomes a no-op (": && :") and libwrp_cte_uvm.so is never written.
set_target_properties(wrp_cte_uvm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared)
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
    set_target_properties(test_gpu_vmm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(test_gpu_vmm PRIVATE wrp_cte_uvm cuda)
    add_test(NAME test_gpu_vmm COMMAND test_gpu_vmm)
    message(STATUS "CTE UVM: Test target 'test_gpu_vmm' added")
endif()
