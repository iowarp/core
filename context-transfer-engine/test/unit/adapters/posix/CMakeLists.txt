# POSIX Adapter Unit Tests

include_directories(
    ${WRP_CTE_ROOT}
    .
)

# Create the POSIX adapter unit test
add_executable(wrp_cte_posix_unit_tests
    test_posix_adapter.cc
)

target_link_libraries(wrp_cte_posix_unit_tests
    wrp_cte_posix
    wrp_cte_fs_base
    wrp_cte_core_client
    wrp_cte_cae_config
    hshm::cxx
    hshm::interceptor
    Catch2::Catch2WithMain
)

# Link MPI only if enabled
if(WRP_CORE_ENABLE_MPI)
    target_link_libraries(wrp_cte_posix_unit_tests MPI::MPI_CXX)
endif()

# Install the test executable
install(
    TARGETS wrp_cte_posix_unit_tests
    RUNTIME DESTINATION bin
)