# CMakeLists.txt for CTE FUSE Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker with FUSE support (--cap-add SYS_ADMIN, /dev/fuse)
# They mount a CTE-backed FUSE filesystem and run POSIX I/O operations against it.

set(FUSE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cte_fuse_integration
    COMMAND ${FUSE_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${FUSE_TEST_DIR}
)

set_tests_properties(cte_fuse_integration PROPERTIES
    LABELS "integration;docker;fuse"
    TIMEOUT 300
)

message(STATUS "CTE FUSE integration test configured")
message(STATUS "  Test directory: ${FUSE_TEST_DIR}")
message(STATUS "  Run with: ctest -L fuse")
