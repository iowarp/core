# CMakeLists.txt for context-runtime/compiler
#
# This directory contains headers for Clang-compiled CUDA code that requires
# C++20 features (coroutines) not available in NVCC device code.
#
# The include path (compiler/include) is added to targets that need GPU
# coroutine support. No libraries are built here -- only headers.

# Make the compiler include directory available as an interface target
add_library(chimaera_gpu_coroutine INTERFACE)
target_include_directories(chimaera_gpu_coroutine INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# LLVM pass plugin for GPU coroutine threadIdx.x fix
if(WRP_CORE_ENABLE_GPU_RUNTIME)
  add_subdirectory(coro_tid_fix)
  # Register the plugin so add_cuda_library passes it to clang
  set(WRP_CORE_CLANG_CUDA_PASS_PLUGINS
    "-fpass-plugin=$<TARGET_FILE:CoroTidFix>"
    CACHE INTERNAL "Clang CUDA pass plugins")
endif()
