# CMakeLists.txt for CTE Distributed Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a multi-node cluster and run distributed CTE tests

# Get the directory containing the test scripts
set(DISTRIBUTED_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test that runs the distributed test suite
add_test(
    NAME cte_distributed_integration_docker
    COMMAND ${DISTRIBUTED_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${DISTRIBUTED_TEST_DIR}
)

# Set test properties
set_tests_properties(cte_distributed_integration_docker PROPERTIES
    LABELS "integration;docker;distributed"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
)

message(STATUS "CTE distributed integration test configured")
message(STATUS "  Test directory: ${DISTRIBUTED_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
