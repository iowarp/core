# Task Archive Serialization Tests

This document describes the comprehensive unit tests for the Chimaera task archive serialization system.

## Overview

The task archive tests verify the complete functionality of the Chimaera serialization system, including:

- **Archive Types**: TaskInputArchiveIN/OUT, TaskOutputArchiveIN/OUT
- **Task Serialization**: Automatic BaseSerializeIn/Out + SerializeIn/Out calls
- **Container Integration**: Save/Load methods for network transfer
- **Bulk Transfer**: Recording of bulk transfer metadata
- **Task Types**: Base Task class and admin module tasks
- **Error Handling**: Edge cases and invalid data scenarios

## Test File Structure

### Main Test File
- `test_task_archive.cc` - Comprehensive test suite with 13 major test cases

### Test Categories

1. **Basic Archive Tests**
   - Construction and basic functionality
   - Data retrieval and serialization

2. **Bulk Transfer Tests**
   - Recording bulk transfer metadata
   - CHI_WRITE and CHI_EXPOSE flag handling
   - Multiple bulk transfers per archive

3. **Non-Task Serialization Tests**
   - Custom structs and standard types
   - Bidirectional operator() functionality

4. **Task Base Class Tests**
   - BaseSerializeIn/Out automatic calls
   - Base task field preservation

5. **Admin Task Tests**
   - CreateTask, DestroyPoolTask, StopRuntimeTask
   - IN/OUT parameter serialization
   - INOUT parameter handling

6. **Archive Operator Tests**
   - Bidirectional functionality verification
   - Compatibility with standard cereal

7. **Container Integration Tests**
   - SaveIn/LoadIn methods
   - SaveOut/LoadOut methods
   - Switch-case task type handling

8. **Error Handling Tests**
   - Invalid serialization data
   - Empty data scenarios
   - Null pointer handling

9. **Performance Tests**
   - Large data serialization
   - Multiple task sequences
   - Memory efficiency

10. **Integration Tests**
    - Complete round-trip serialization flow
    - Remote execution simulation

## Running the Tests

### Using CTest (Recommended)

```bash
# Run all task archive tests
ctest -R task_archive

# Run specific test categories
ctest -R task_archive_basic_tests
ctest -R task_archive_bulk_transfer_tests  
ctest -R task_archive_admin_tasks_tests
ctest -R task_archive_performance_tests
ctest -R task_archive_integration_tests

# Run comprehensive test
ctest -R all_task_archive_tests
```

### Using CMake Targets

```bash
# Build and run all task archive tests
make test_task_archive

# Run basic functionality tests
make test_task_archive_basic

# Run serialization tests
make test_task_archive_serialization

# Run performance tests
make test_task_archive_performance
```

### Direct Execution

```bash
# Build the test executable
cd build
make chimaera_task_archive_tests

# Run all tests
./bin/chimaera_task_archive_tests

# Run specific test categories using tags
./bin/chimaera_task_archive_tests "[task_archive][bulk_transfer]"
./bin/chimaera_task_archive_tests "[task_archive][admin_tasks]"
./bin/chimaera_task_archive_tests "[task_archive][performance]"
```

## Test Coverage

### Archive Types Tested
- ✅ TaskInputArchiveIN - Deserializing input parameters
- ✅ TaskOutputArchiveIN - Serializing input parameters  
- ✅ TaskInputArchiveOUT - Deserializing output parameters
- ✅ TaskOutputArchiveOUT - Serializing output parameters

### Task Types Tested
- ✅ Base Task class (BaseSerializeIn/Out + SerializeIn/Out)
- ✅ admin::CreateTask (templated BaseCreateTask)
- ✅ admin::DestroyPoolTask 
- ✅ admin::StopRuntimeTask
- ✅ Network task types (ClientSendTaskInTask, ServerRecvTaskInTask, etc.)

### Serialization Features Tested
- ✅ Automatic Task detection and method dispatch
- ✅ BaseSerialize method calls for base task fields
- ✅ SerializeIn/Out method calls for derived task fields
- ✅ Non-Task object serialization via cereal
- ✅ Bidirectional operator() functionality
- ✅ Bulk transfer metadata recording
- ✅ Container Save/Load integration

### Error Scenarios Tested
- ✅ Invalid serialization data handling
- ✅ Empty data scenarios
- ✅ Null pointer bulk transfers
- ✅ Large data serialization performance

## Implementation Notes

### Key Features Verified

1. **Automatic Method Dispatch**: Archives automatically detect Task inheritance using `std::is_base_of_v<Task, T>` and call appropriate serialization methods.

2. **BaseSerialize Integration**: All Task-derived objects automatically have BaseSerializeIn/Out called to handle common task fields (pool_id_, method_, pool_query_, etc.).

3. **IN/OUT Parameter Separation**: 
   - IN archives handle input parameters for remote execution
   - OUT archives handle output parameters for result transfer
   - INOUT parameters appear in both serialization paths

4. **Bulk Transfer Recording**: Archives track bulk transfer metadata with CHI_WRITE/CHI_EXPOSE flags for large data transfers.

5. **Container Integration**: Container class Save/Load methods use switch-case statements to cast tasks to concrete types before calling archive operators.

### Memory Management

- Uses Chimaera's allocator system (hipc::CtxAllocator)
- Properly handles hipc::string types in task fields
- Tests memory cleanup and resource management

### Performance Considerations

- Tests serialization of large strings (10KB) and vectors (1000 elements)
- Verifies efficient serialization with minimal overhead
- Tests multiple task serialization sequences

## Dependencies

The tests require the following components:

- **Chimaera Core**: Main library with task and archive classes
- **Admin Module**: For testing concrete task types
- **HermesShm**: For shared memory allocator functionality
- **Cereal**: For binary serialization (included with Chimaera)
- **SimpleTest Framework**: Lightweight test framework

## Expected Results

All tests should pass, demonstrating:

1. **Correctness**: Serialized data matches original data after round-trip
2. **Completeness**: All task fields are preserved through serialization
3. **Performance**: Large data can be serialized efficiently
4. **Robustness**: System handles edge cases gracefully
5. **Integration**: Container methods work correctly with archives

## Troubleshooting

If tests fail:

1. **Compilation Errors**: Check that all required headers are included
2. **Linking Errors**: Verify chimaera_admin library is linked
3. **Runtime Errors**: Ensure CHIMAERA_RUNTIME=1 is defined
4. **Memory Errors**: Check allocator initialization in test setup

For detailed error information, run tests with verbose output:
```bash
ctest -R task_archive -V
```

## Future Enhancements

Potential areas for additional testing:

1. **More Task Types**: Test additional modules' task types
2. **Concurrent Serialization**: Multi-threaded serialization tests
3. **Network Integration**: End-to-end network transfer tests
4. **Compression**: Archive compression functionality
5. **Versioning**: Serialization format version compatibility