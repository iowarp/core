include_directories(
        ${WRP_CTE_ROOT}
        ${HERMES_SRC_DIR}
        ${HERMES_ADAPTER_DIR}
        ${HERMES_IO_CLIENT_DIR}
        .)

# Create the MPIIO interceptor
if(HERMES_MPICH)
        message(STATUS "Using MPICH")
        add_definitions(-DWRP_CTE_MPICH)
elseif(HERMES_OPENMPI)
        message(STATUS "Using OpenMPI")
        add_definitions(-DWRP_CTE_OPENMPI)
else()
        # Auto-detect based on MPI implementation
        if(MPI_C_INCLUDE_DIRS MATCHES "openmpi" OR MPI_CXX_INCLUDE_DIRS MATCHES "openmpi")
                message(STATUS "Auto-detected OpenMPI")
                add_definitions(-DWRP_CTE_OPENMPI)
        elseif(MPI_C_INCLUDE_DIRS MATCHES "mpich" OR MPI_CXX_INCLUDE_DIRS MATCHES "mpich")
                message(STATUS "Auto-detected MPICH")
                add_definitions(-DWRP_CTE_MPICH)
        else()
                message(WARNING "Could not auto-detect MPI implementation, defaulting to OpenMPI")
                add_definitions(-DWRP_CTE_OPENMPI)
        endif()
endif()

set(INTERCEPTOR_DEPS
        ${Hermes_CLIENT_DEPS} wrp_cte_fs_base)
add_library(wrp_cte_mpiio SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/mpiio_api.cc)
add_dependencies(wrp_cte_mpiio ${INTERCEPTOR_DEPS})
target_link_libraries(wrp_cte_mpiio
        MPI::MPI_CXX stdc++fs ${libelf_LIBRARIES} dl ${INTERCEPTOR_DEPS})

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        wrp_cte_mpiio
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# -----------------------------------------------------------------------------
# Install headers
# -----------------------------------------------------------------------------
file(GLOB_RECURSE WRP_CTE_HEADERS "*.h")
install(
        FILES
        ${WRP_CTE_HEADERS}
        DESTINATION
        include/adapter/mpiio
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
