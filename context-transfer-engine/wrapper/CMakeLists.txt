# Python bindings are controlled by WRP_CORE_ENABLE_PYTHON
if(WRP_CORE_ENABLE_PYTHON)
    message(STATUS "CTE Python bindings enabled")
    add_subdirectory(python)
else()
    message(STATUS "CTE Python bindings disabled: WRP_CORE_ENABLE_PYTHON is OFF")
endif()

# Rust bindings are handled by parent CMakeLists.txt via Corrosion
# Tests can be run manually: cd wrapper/rust && ./run_tests.sh
