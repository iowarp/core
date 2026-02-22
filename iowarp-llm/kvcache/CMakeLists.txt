cmake_minimum_required(VERSION 3.20)

# wrp_llm_kvcache — KV cache tiered offloading library.
# Links against the CTE core client to use PutBlob/GetBlob for tiered storage.

set(KVCACHE_SOURCES
    src/kvcache_manager.cc
)

add_library(wrp_llm_kvcache SHARED ${KVCACHE_SOURCES})

target_include_directories(wrp_llm_kvcache
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# CTE core client provides PutBlob/GetBlob/GetOrCreateTag etc.
# It also brings in the CUDA runtime transitively when CUDA is enabled,
# so we do NOT add an explicit 'cuda' link here (avoids CUDA RDC linker
# errors from mixing CUDA-separable and non-CUDA targets).
target_link_libraries(wrp_llm_kvcache
    PUBLIC
        wrp_cte_core_client
)

# Enable D2H/H2D staging-copy code paths inside kvcache_manager.cc.
if(WRP_CORE_ENABLE_CUDA)
    target_compile_definitions(wrp_llm_kvcache PRIVATE IOWARP_LLM_ENABLE_CUDA)
endif()

install(TARGETS wrp_llm_kvcache
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
