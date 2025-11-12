cmake_minimum_required(VERSION 3.20)
project(external_chimod_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Load environment variables from .env.cmake if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.env.cmake")
  message(STATUS "Loading environment from ${CMAKE_CURRENT_SOURCE_DIR}/.env.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/.env.cmake")
endif()

# Find required Chimaera packages
find_package(chimaera REQUIRED)
find_package(chimaera_admin REQUIRED)

# Create test executable
add_executable(test_external_chimod test_external_chimod.cc)

# Link against required libraries
target_link_libraries(test_external_chimod
  chimaera::admin_client                  # Admin client (required, includes chimaera::cxx)
  ${CMAKE_THREAD_LIBS_INIT}               # Threading support
)

message(STATUS "External ChiMod test configured successfully")
message(STATUS "  Target: test_external_chimod")
message(STATUS "  Links to: chimaera::admin_client (includes chimaera::cxx automatically)")

# Enable testing
enable_testing()
add_test(NAME external_chimod_test COMMAND test_external_chimod)