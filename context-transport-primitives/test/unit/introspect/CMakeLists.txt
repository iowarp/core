# Introspect tests

# Test for MapMixedMemory functionality
add_executable(test_mixed_mapping test_mixed_mapping.cc)
target_link_libraries(test_mixed_mapping hshm::cxx)

# Install to build/bin
install(TARGETS test_mixed_mapping
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Add CTest
add_test(
  NAME ctp_mixed_mapping
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_mixed_mapping_test.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
