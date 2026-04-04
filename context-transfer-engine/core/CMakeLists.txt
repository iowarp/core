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

# Knowledge graph compile definition (conditional)
if(WRP_CTE_ENABLE_KNOWLEDGE_GRAPH)
  message(STATUS "CTE Core: Knowledge graph ENABLED")
  target_compile_definitions(wrp_cte_core_runtime PUBLIC WRP_CTE_ENABLE_KNOWLEDGE_GRAPH)
  target_compile_definitions(wrp_cte_core_client PUBLIC WRP_CTE_ENABLE_KNOWLEDGE_GRAPH)

  # Pluggable KG backend compile definitions
  # These enable the factory to create the corresponding backend
  if(WRP_CTE_KG_ELASTICSEARCH OR WRP_CTE_KG_NEO4J OR WRP_CTE_KG_QDRANT)
    # cpp-httplib is header-only, stored in external/httplib/
    target_include_directories(wrp_cte_core_runtime PRIVATE
      ${CMAKE_SOURCE_DIR}/external/httplib)
    target_link_libraries(wrp_cte_core_runtime PRIVATE nlohmann_json::nlohmann_json)
  endif()
  if(WRP_CTE_KG_ELASTICSEARCH)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CTE_KG_ELASTICSEARCH)
  endif()
  if(WRP_CTE_KG_NEO4J)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CTE_KG_NEO4J)
  endif()
  if(WRP_CTE_KG_QDRANT)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CTE_KG_QDRANT)
  endif()
else()
  message(STATUS "CTE Core: Knowledge graph disabled (use -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON to enable)")
endif()

# Note: Cross-namespace dependencies (chimaera admin and bdev) are automatically
# linked by add_chimod_* functions in ChimaeraCommon.cmake for non-chimaera namespaces