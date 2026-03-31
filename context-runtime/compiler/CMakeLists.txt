# CMakeLists.txt for context-runtime/compiler
#
# This directory contains GPU headers for the runtime.
# No libraries are built here -- only headers.

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
