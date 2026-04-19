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

  # Acropolis depth controller needs EmbeddingClient (httplib + nlohmann_json)
  # whenever KG is enabled, regardless of whether a semantic backend is built.
  # cpp-httplib is header-only; wrap in BUILD_INTERFACE so install export is clean.
  target_include_directories(wrp_cte_core_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/httplib>)
  target_include_directories(wrp_cte_core_client PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/httplib>)
  target_link_libraries(wrp_cte_core_runtime PUBLIC nlohmann_json::nlohmann_json)
  target_link_libraries(wrp_cte_core_client PUBLIC nlohmann_json::nlohmann_json)

  # Pluggable KG backend compile definitions
  if(WRP_CTE_KG_ELASTICSEARCH)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CTE_KG_ELASTICSEARCH)
  endif()
  if(WRP_CTE_KG_NEO4J)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CTE_KG_NEO4J)
  endif()
  if(WRP_CTE_KG_QDRANT)
    target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CTE_KG_QDRANT)
  endif()

  # Acropolis L2 HDF5 extractor — optional, only when HDF5 is available.
  if(WRP_CORE_ENABLE_HDF5)
    find_package(HDF5 QUIET COMPONENTS C)
    if(HDF5_FOUND)
      target_compile_definitions(wrp_cte_core_runtime PRIVATE WRP_CORE_ENABLE_HDF5)
      target_include_directories(wrp_cte_core_runtime PRIVATE ${HDF5_INCLUDE_DIRS})
      # Use PUBLIC so downstream test binaries that link the runtime .so also
      # resolve libhdf5's transitive references.
      target_link_libraries(wrp_cte_core_runtime PUBLIC ${HDF5_C_LIBRARIES})

      # HDF5's REST VOL needs libcurl with matching symbol versions. The system
      # libhdf5 (apt) was built against the system libcurl (CURL_OPENSSL_4
      # versioned symbols), while find_package(CURL) may pick the conda curl
      # which uses different versions. Link the system curl directly by path
      # when HDF5 came from /usr/lib.
      set(_system_curl "/usr/lib/x86_64-linux-gnu/libcurl.so.4")
      if(EXISTS "${_system_curl}" AND HDF5_C_LIBRARIES MATCHES "/usr/lib/")
        target_link_libraries(wrp_cte_core_runtime PUBLIC "${_system_curl}")
        message(STATUS "CTE Core: HDF5 L2 extractor enabled (system curl: ${_system_curl})")
      else()
        find_package(CURL)
        if(CURL_FOUND)
          target_link_libraries(wrp_cte_core_runtime PUBLIC CURL::libcurl)
          message(STATUS "CTE Core: HDF5 L2 extractor enabled (found curl: ${CURL_LIBRARIES})")
        else()
          message(WARNING "CTE Core: HDF5 L2 extractor enabled but curl not found")
        endif()
      endif()
    else()
      message(STATUS "CTE Core: HDF5 L2 extractor requested but HDF5 not found")
    endif()
  endif()
else()
  message(STATUS "CTE Core: Knowledge graph disabled (use -DWRP_CTE_ENABLE_KNOWLEDGE_GRAPH=ON to enable)")
endif()

# Note: Cross-namespace dependencies (chimaera admin and bdev) are automatically
# linked by add_chimod_* functions in ChimaeraCommon.cmake for non-chimaera namespaces