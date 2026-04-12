cmake_minimum_required(VERSION 3.10)

# Create runtime library using modern ChiMod build functions
add_chimod_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_dpe.cc
    src/autogen/core_lib_exec.cc
    # core_runtime_gpu.cc is compiled into chimaera_cxx_gpu (the monolithic GPU
    # library) via the external-module glob in context-runtime/src/CMakeLists.txt.
    # Building it here would create a redundant GPU companion that fails to
    # device-link because nvcc -dlink cannot resolve device symbols across
    # shared library boundaries (e.g. GpuCoroAlloc from chimaera_cxx_gpu).
)

# Create client library using modern ChiMod build functions
add_chimod_client(
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
    src/tag.cc
)

# Note: Cross-namespace dependencies (chimaera admin and bdev) are automatically
# linked by add_chimod_* functions in ChimaeraCommon.cmake for non-chimaera namespaces
