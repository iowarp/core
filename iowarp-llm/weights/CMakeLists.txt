cmake_minimum_required(VERSION 3.20)

# wrp_llm_weights — Model weight offloading library (FlexGen-style demand paging).
# Requires CUDA + llama.cpp (ggml target) — only built when LLAMA_SERVER=ON.

# CUDAToolkit provides CUDA::cudart / CUDA::cuda_driver imported targets and
# the CUDA include directories (cuda.h, cuda_runtime.h, etc.).
find_package(CUDAToolkit REQUIRED)

set(WEIGHTS_SOURCES
    src/weight_manager.cc
    src/ggml_iowarp_backend.cc
)

add_library(wrp_llm_weights SHARED ${WEIGHTS_SOURCES})

target_include_directories(wrp_llm_weights
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # ggml internal headers needed for ggml-backend-impl.h
        ${CMAKE_SOURCE_DIR}/external/llama.cpp/ggml/src
        ${CMAKE_SOURCE_DIR}/external/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/external/llama.cpp/include
)

target_link_libraries(wrp_llm_weights
    PUBLIC
        wrp_cte_core_client
        wrp_cte_uvm          # brings in wrp_cte/uvm/gpu_vmm.h include path + GpuVmm symbols
    PRIVATE
        ggml
        CUDA::cudart         # CUDA runtime headers (cuda_runtime.h, cuda.h)
        CUDA::cuda_driver    # CUDA driver API (cuMemAddressReserve etc.)
)

install(TARGETS wrp_llm_weights
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
