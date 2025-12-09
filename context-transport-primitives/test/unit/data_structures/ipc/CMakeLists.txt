project(hermes_shm)

#------------------------------------------------------------------------------
# Build IPC Data Structure Tests
#------------------------------------------------------------------------------

add_executable(test_slist_pre_exec
        test_slist_pre.cc)
add_dependencies(test_slist_pre_exec hermes_shm_host)
target_link_libraries(test_slist_pre_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_rb_tree_pre_exec
        test_rb_tree_pre.cc)
add_dependencies(test_rb_tree_pre_exec hermes_shm_host)
target_link_libraries(test_rb_tree_pre_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_vector_exec
        vector.cc)
add_dependencies(test_vector_exec hermes_shm_host)
target_link_libraries(test_vector_exec
        hermes_shm_host)

add_executable(test_ring_buffer_exec
        test_ring_buffer.cc)
add_dependencies(test_ring_buffer_exec hermes_shm_host)
target_link_libraries(test_ring_buffer_exec
        hermes_shm_host)

add_executable(test_ring_buffer_spsc_exec
        test_ring_buffer_spsc.cc)
add_dependencies(test_ring_buffer_spsc_exec hermes_shm_host)
target_link_libraries(test_ring_buffer_spsc_exec
        hermes_shm_host)

add_executable(test_ring_buffer_ext_exec
        test_ring_buffer_ext.cc)
add_dependencies(test_ring_buffer_ext_exec hermes_shm_host)
target_link_libraries(test_ring_buffer_ext_exec
        hermes_shm_host)

add_executable(test_ring_buffer_mpsc_exec
        test_ring_buffer_mpsc.cc)
add_dependencies(test_ring_buffer_mpsc_exec hermes_shm_host)
target_link_libraries(test_ring_buffer_mpsc_exec
        hermes_shm_host)

add_executable(test_multi_ring_buffer_exec
        test_multi_ring_buffer.cc)
add_dependencies(test_multi_ring_buffer_exec hermes_shm_host)
target_link_libraries(test_multi_ring_buffer_exec
        hermes_shm_host)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_slist_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec "[slist_pre]~[atomic]")
add_test(NAME ctp_slist_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec "[slist_pre][atomic]")

add_test(NAME ctp_rb_tree_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_exec "[rb_tree_pre]~[atomic]")
add_test(NAME ctp_rb_tree_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_exec "[rb_tree_pre][atomic]")

add_test(NAME ctp_vector COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector]")
add_test(NAME ctp_vector_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][pod]")
add_test(NAME ctp_vector_non_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][non-pod]")
add_test(NAME ctp_vector_edge COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][edge]")
add_test(NAME ctp_vector_stress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec "[vector][stress]")

add_test(NAME ctp_ring_buffer COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_exec "[ring_buffer]")
add_test(NAME ctp_ring_buffer_spsc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_spsc_exec "[ring_buffer][spsc]")
add_test(NAME ctp_ring_buffer_ext COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_ext_exec "[ring_buffer][ext]")
add_test(NAME ctp_ring_buffer_mpsc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_mpsc_exec "[ring_buffer][mpsc]")

add_test(NAME ctp_multi_ring_buffer COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_multi_ring_buffer_exec "[multi_ring_buffer]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_slist_pre_exec
        test_rb_tree_pre_exec
        test_vector_exec
        test_ring_buffer_exec
        test_ring_buffer_spsc_exec
        test_ring_buffer_ext_exec
        test_ring_buffer_mpsc_exec
        test_multi_ring_buffer_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_slist_pre_exec)
    set_coverage_flags(test_rb_tree_pre_exec)
    set_coverage_flags(test_vector_exec)
    set_coverage_flags(test_ring_buffer_exec)
    set_coverage_flags(test_ring_buffer_spsc_exec)
    set_coverage_flags(test_ring_buffer_ext_exec)
    set_coverage_flags(test_ring_buffer_mpsc_exec)
    set_coverage_flags(test_multi_ring_buffer_exec)
endif()
