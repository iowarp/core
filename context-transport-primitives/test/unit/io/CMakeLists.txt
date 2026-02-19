project(hermes_shm)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------

add_executable(test_async_io_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_async_io.cc)

#------------------------------------------------------------------------------
# Link Dependencies
#------------------------------------------------------------------------------

add_dependencies(test_async_io_exec hermes_shm_host)
target_link_libraries(test_async_io_exec
        hermes_shm_host hshm::aio Catch2::Catch2)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

add_test(NAME ctp_async_io COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_async_io_exec "~[error=FatalError]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------

install(TARGETS
        test_async_io_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
