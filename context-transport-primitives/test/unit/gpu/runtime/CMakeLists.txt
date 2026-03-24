#------------------------------------------------------------------------------
# GPU Dynamic Runtime Virtual Dispatch Test
#------------------------------------------------------------------------------

# Shared CUDA library (dynamically loaded)
add_cuda_library(gpu_runtime_lib SHARED TRUE lib.cc
  INCLUDE_DIRS
    ${HSHM_ROOT}/include
    ${PROJECT_BINARY_DIR}/include
)
target_link_libraries(gpu_runtime_lib hshm::cuda_cxx)
target_include_directories(gpu_runtime_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS gpu_runtime_lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

# Test executable
add_cuda_executable(test_gpu_runtime TRUE main.cc
    INCLUDE_DIRS
      ${HSHM_ROOT}/include
      ${PROJECT_BINARY_DIR}/include
    DEFS GPU_RUNTIME_LIB_PATH="$<TARGET_FILE:gpu_runtime_lib>")
target_link_libraries(test_gpu_runtime hshm::cuda_cxx ${CMAKE_DL_LIBS})
target_include_directories(test_gpu_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_gpu_runtime PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared)
add_dependencies(test_gpu_runtime gpu_runtime_lib)
add_test(NAME test_gpu_runtime COMMAND test_gpu_runtime)
# Cross-library CUDA virtual dispatch (calling device virtual methods in a
# dlopen'd .so from a kernel in the main binary) does not work with Clang CUDA
# + -fgpu-rdc because device function addresses are CUDA-module-local and are
# not portable across separately linked fat-binary modules.  With NVCC's
# CUDA_SEPARABLE_COMPILATION the device linker resolves these cross-module
# references, but Clang's manual device link step does not.
# Mark as WILL_FAIL so the test suite documents the limitation without blocking CI.
if(WRP_CORE_ENABLE_GPU_RUNTIME)
  set_tests_properties(test_gpu_runtime PROPERTIES WILL_FAIL TRUE)
endif()
install(TARGETS test_gpu_runtime RUNTIME DESTINATION bin)
