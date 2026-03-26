# LLVM pass plugin: Fix threadIdx.x in GPU coroutine resume functions
#
# clang-cuda's CoroSplit outlines resume paths into separate functions.
# Inside these outlined functions, threadIdx.x (the %tid.x special register)
# reads incorrectly. This pass saves %tid.x before llvm.coro.resume calls
# and replaces %tid.x reads inside .resume functions with the saved value.
#
# Usage: clang++ -fpass-plugin=<path>/libCoroTidFix.so ...

cmake_minimum_required(VERSION 3.20)

find_package(LLVM REQUIRED CONFIG
  HINTS /usr/lib/llvm-18)
message(STATUS "CoroTidFix: Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})

add_llvm_pass_plugin(CoroTidFix
  CoroTidFix.cpp
  LINK_COMPONENTS Core Support
)
