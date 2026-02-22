cmake_minimum_required(VERSION 3.20)

#------------------------------------------------------------------------------
# iowarp-llm: IOWarp integration for LLM inference with llama.cpp
#
# Two integration points:
#   kvcache/  - KV cache tiered offloading via CTE (MOONCAKE-style prefix caching)
#               Standalone: only needs WRP_CORE_ENABLE_CTE=ON.
#   weights/  - Model weight offloading via GpuVMM (FlexGen-style demand paging)
#               Requires CUDA + llama.cpp (WRP_CORE_ENABLE_LLAMA_SERVER=ON) because
#               it links against the 'ggml' target from external/llama.cpp.
#               CUDA separable-compilation device-link fails during CMake generate
#               when 'ggml' is absent, so we guard this carefully.
#------------------------------------------------------------------------------

add_subdirectory(kvcache)

# weights/ links against 'ggml' (from external/llama.cpp) and uses CUDA separable
# compilation.  Only add it when llama.cpp is also being built to keep ggml target
# available for the device-link step; otherwise the CMake generate step aborts with
# "_CMAKE_CUDA_RDC_FLAG not set" because the unresolved 'ggml' target breaks CUDA
# property propagation.
if(WRP_CORE_ENABLE_CUDA AND WRP_CORE_ENABLE_LLAMA_SERVER)
    add_subdirectory(weights)
else()
    message(STATUS "iowarp-llm/weights: skipped (requires CUDA + WRP_CORE_ENABLE_LLAMA_SERVER=ON)")
endif()

# NOTE: Patching of server-context and llama targets with IOWARP_LLM_KVCACHE
# is done in the root CMakeLists.txt AFTER add_subdirectory(external/llama.cpp),
# because those targets do not exist yet when this file is processed.

if(WRP_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
