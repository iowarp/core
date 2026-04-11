# ------------------------------------------------------------------------------
# FUSE Adapter Unit Tests
# ------------------------------------------------------------------------------

message(STATUS "Building FUSE adapter tests")

# Tests do NOT require libfuse3 — they test the CTE helper functions
# (tag lifecycle, implicit directories, page-based I/O) directly.
add_executable(test_fuse_adapter
    test_fuse_adapter.cc
)

target_include_directories(test_fuse_adapter PRIVATE
    ${WRP_CTE_ROOT}
)

target_link_libraries(test_fuse_adapter
    wrp_cte_core_runtime
    wrp_cte_core_client
    chimaera::admin_runtime
    chimaera::admin_client
    chimaera::bdev_runtime
    chimaera::bdev_client
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Tag lifecycle tests
add_test(NAME fuse_cte_tag_exists
    COMMAND test_fuse_adapter "Tag create and exists")
add_test(NAME fuse_cte_tag_delete
    COMMAND test_fuse_adapter "Tag deletion")

# Implicit directory tests
add_test(NAME fuse_cte_dir_exists
    COMMAND test_fuse_adapter "CteDirExists")
add_test(NAME fuse_cte_list_children
    COMMAND test_fuse_adapter "CteListDirectChildren")
add_test(NAME fuse_cte_list_subdirs
    COMMAND test_fuse_adapter "CteListSubdirs")

# Page-based I/O tests
add_test(NAME fuse_cte_small_rw
    COMMAND test_fuse_adapter "Small write and read")
add_test(NAME fuse_cte_multipage
    COMMAND test_fuse_adapter "Multi-page write and read")
add_test(NAME fuse_cte_partial_page
    COMMAND test_fuse_adapter "Partial page write")
add_test(NAME fuse_cte_cross_page
    COMMAND test_fuse_adapter "Cross-page write")

# Integration test
add_test(NAME fuse_integration
    COMMAND test_fuse_adapter "Full file lifecycle")

set_tests_properties(
    fuse_cte_tag_exists
    fuse_cte_tag_delete
    fuse_cte_dir_exists
    fuse_cte_list_children
    fuse_cte_list_subdirs
    fuse_cte_small_rw
    fuse_cte_multipage
    fuse_cte_partial_page
    fuse_cte_cross_page
    fuse_integration
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;adapter;fuse"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

install(TARGETS test_fuse_adapter RUNTIME DESTINATION bin)

message(STATUS "FUSE adapter tests configured successfully")
