cmake_minimum_required(VERSION 3.10)

# Create runtime library using modern ChiMod build functions
add_chimod_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_dpe.cc
    src/autogen/core_lib_exec.cc
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