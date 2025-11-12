# Chimaera CMake Infrastructure Redesign

## Executive Summary

This document outlines a complete redesign of the Chimaera CMake build system to achieve simplicity, predictability, and external-project friendliness. The new design eliminates complex auto-magic behaviors in favor of clear, single-purpose functions with predictable naming conventions.

## 1. Problem Analysis

### Current Issues
- **Complexity**: The previous `add_chimod_both()` function combined too many responsibilities
- **Opacity**: Auto-generated targets and aliases are hard to understand
- **External Integration**: Difficult to use ChiMods from external projects
- **Naming Confusion**: Inconsistent target naming across modules
- **Maintenance Burden**: Complex CMake logic is hard to debug and maintain

### Design Principles
1. **Explicit over Implicit**: Clear function calls with visible parameters
2. **Single Responsibility**: Each function does one thing well
3. **Predictable Naming**: Consistent target naming across all modules
4. **External-First**: Design with external project usage as primary use case
5. **Minimal Magic**: Reduce auto-generation in favor of clarity

## 2. Architecture Overview

### Directory Structure
```
chimaera/
├── cmake/
│   ├── ChimaeraCommon.cmake       # Core shared functionality
│   └── ChimaeraConfig.cmake.in    # Export configuration template
├── chimods/
│   ├── chimaera_repo.yaml         # Repository configuration
│   ├── admin/
│   │   ├── chimaera_mod.yaml      # Module configuration
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   └── src/
│   └── bdev/
│       ├── chimaera_mod.yaml
│       ├── CMakeLists.txt
│       ├── include/
│       └── src/
└── CMakeLists.txt                  # Root CMake file
```

### Target Naming Convention

#### Physical Target Names
- Client: `<namespace>-<module>-client` (e.g., `chimaera-admin-client`)
- Runtime: `<namespace>-<module>-runtime` (e.g., `chimaera-admin-runtime`)

#### Alias Target Names (for external use)
- Client: `<namespace>::<module>-client` (e.g., `chimaera::admin-client`)
- Runtime: `<namespace>::<module>-runtime` (e.g., `chimaera::admin-runtime`)

## 3. Detailed Design

### 3.1 ChimaeraCommon.cmake

```cmake
# ChimaeraCommon.cmake - Core shared CMake functionality for Chimaera

# Guard against multiple inclusions
if(CHIMAERA_COMMON_INCLUDED)
  return()
endif()
set(CHIMAERA_COMMON_INCLUDED TRUE)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------

# Find HermesShm
find_package(hermes_shm REQUIRED)

# Find Boost components
find_package(Boost REQUIRED COMPONENTS fiber context system thread)

# Find cereal
find_package(cereal REQUIRED)

# Find MPI (optional)
find_package(MPI QUIET)

# Thread support
find_package(Threads REQUIRED)

#------------------------------------------------------------------------------
# Common compile definitions and flags
#------------------------------------------------------------------------------

# Set common compile features
set(CHIMAERA_CXX_STANDARD 17)

# Common compile definitions
set(CHIMAERA_COMMON_COMPILE_DEFS
  $<$<CONFIG:Debug>:DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)

# Common include directories
set(CHIMAERA_COMMON_INCLUDES
  ${Boost_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
)

# Common link libraries
set(CHIMAERA_COMMON_LIBS
  hermes_shm::cxx
  Boost::fiber
  Boost::context
  Boost::system
  Threads::Threads
)

#------------------------------------------------------------------------------
# Module configuration parsing
#------------------------------------------------------------------------------

# Function to read module configuration from chimaera_mod.yaml
function(chimaera_read_module_config MODULE_DIR)
  set(CONFIG_FILE "${MODULE_DIR}/chimaera_mod.yaml")
  
  if(NOT EXISTS ${CONFIG_FILE})
    message(FATAL_ERROR "Missing chimaera_mod.yaml in ${MODULE_DIR}")
  endif()
  
  # Parse YAML file (simple regex parsing for key: value pairs)
  file(READ ${CONFIG_FILE} CONFIG_CONTENT)
  
  # Extract module_name
  string(REGEX MATCH "module_name:[ ]*([^\n\r]*)" _ ${CONFIG_CONTENT})
  set(CHIMAERA_MODULE_NAME ${CMAKE_MATCH_1} PARENT_SCOPE)
  
  # Extract namespace
  string(REGEX MATCH "namespace:[ ]*([^\n\r]*)" _ ${CONFIG_CONTENT})
  set(CHIMAERA_NAMESPACE ${CMAKE_MATCH_1} PARENT_SCOPE)
  
  # Validate extracted values
  if(NOT CHIMAERA_MODULE_NAME)
    message(FATAL_ERROR "module_name not found in ${CONFIG_FILE}")
  endif()
  
  if(NOT CHIMAERA_NAMESPACE)
    message(FATAL_ERROR "namespace not found in ${CONFIG_FILE}")
  endif()
endfunction()

#------------------------------------------------------------------------------
# ChiMod Client Library Function
#------------------------------------------------------------------------------

# add_chimod_client - Create a ChiMod client library
#
# Parameters:
#   SOURCES             - Source files for the client library
#   COMPILE_DEFINITIONS - Additional compile definitions
#   LINK_LIBRARIES      - Additional libraries to link
#   LINK_DIRECTORIES    - Additional link directories
#   INCLUDE_LIBRARIES   - Libraries whose includes should be added
#   INCLUDE_DIRECTORIES - Additional include directories
#
function(add_chimod_client)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;COMPILE_DEFINITIONS;LINK_LIBRARIES;LINK_DIRECTORIES;INCLUDE_LIBRARIES;INCLUDE_DIRECTORIES"
    ${ARGN}
  )
  
  # Read module configuration
  chimaera_read_module_config(${CMAKE_CURRENT_SOURCE_DIR})
  
  # Create target name
  set(TARGET_NAME "${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-client")
  
  # Create the library
  add_library(${TARGET_NAME} ${ARG_SOURCES})
  
  # Set C++ standard
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})
  
  # Add compile definitions
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )
  
  # Add include directories
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${CHIMAERA_COMMON_INCLUDES}
      ${ARG_INCLUDE_DIRECTORIES}
  )
  
  # Add include directories from INCLUDE_LIBRARIES
  foreach(LIB ${ARG_INCLUDE_LIBRARIES})
    get_target_property(LIB_INCLUDES ${LIB} INTERFACE_INCLUDE_DIRECTORIES)
    if(LIB_INCLUDES)
      target_include_directories(${TARGET_NAME} PUBLIC ${LIB_INCLUDES})
    endif()
  endforeach()
  
  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()
  
  # Link libraries
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${CHIMAERA_COMMON_LIBS}
      ${ARG_LINK_LIBRARIES}
  )
  
  # Create alias for external use
  add_library(${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}-client ALIAS ${TARGET_NAME})
  
  # Set properties for installation
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CHIMAERA_MODULE_NAME}-client"
    OUTPUT_NAME "${CHIMAERA_MODULE_NAME}_client"
  )
  
  # Export module info to parent scope
  set(CHIMAERA_MODULE_CLIENT_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# ChiMod Runtime Library Function
#------------------------------------------------------------------------------

# add_chimod_runtime - Create a ChiMod runtime library
#
# Parameters:
#   SOURCES             - Source files for the runtime library
#   COMPILE_DEFINITIONS - Additional compile definitions
#   LINK_LIBRARIES      - Additional libraries to link
#   LINK_DIRECTORIES    - Additional link directories
#   INCLUDE_LIBRARIES   - Libraries whose includes should be added
#   INCLUDE_DIRECTORIES - Additional include directories
#
function(add_chimod_runtime)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;COMPILE_DEFINITIONS;LINK_LIBRARIES;LINK_DIRECTORIES;INCLUDE_LIBRARIES;INCLUDE_DIRECTORIES"
    ${ARGN}
  )
  
  # Read module configuration
  chimaera_read_module_config(${CMAKE_CURRENT_SOURCE_DIR})
  
  # Create target name
  set(TARGET_NAME "${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-runtime")
  
  # Create the library
  add_library(${TARGET_NAME} ${ARG_SOURCES})
  
  # Set C++ standard
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})
  
  # Add compile definitions (runtime always has CHIMAERA_RUNTIME=1)
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      CHIMAERA_RUNTIME=1
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )
  
  # Add include directories
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${CHIMAERA_COMMON_INCLUDES}
      ${ARG_INCLUDE_DIRECTORIES}
  )
  
  # Add include directories from INCLUDE_LIBRARIES
  foreach(LIB ${ARG_INCLUDE_LIBRARIES})
    get_target_property(LIB_INCLUDES ${LIB} INTERFACE_INCLUDE_DIRECTORIES)
    if(LIB_INCLUDES)
      target_include_directories(${TARGET_NAME} PUBLIC ${LIB_INCLUDES})
    endif()
  endforeach()
  
  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()
  
  # Link libraries
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${CHIMAERA_COMMON_LIBS}
      ${ARG_LINK_LIBRARIES}
  )
  
  # Create alias for external use
  add_library(${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}-runtime ALIAS ${TARGET_NAME})
  
  # Set properties for installation
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CHIMAERA_MODULE_NAME}-runtime"
    OUTPUT_NAME "${CHIMAERA_MODULE_NAME}_runtime"
  )
  
  # Export module info to parent scope
  set(CHIMAERA_MODULE_RUNTIME_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Installation Helpers
#------------------------------------------------------------------------------

# install_chimod - Install a ChiMod with proper exports
#
# This function should be called after add_chimod_client/runtime
#
function(install_chimod)
  # Use module info from parent scope
  if(NOT CHIMAERA_MODULE_NAME OR NOT CHIMAERA_NAMESPACE)
    message(FATAL_ERROR "install_chimod must be called after add_chimod_client or add_chimod_runtime")
  endif()
  
  # Install targets
  if(TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
    install(TARGETS ${CHIMAERA_MODULE_CLIENT_TARGET}
      EXPORT ${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-targets
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin
    )
  endif()
  
  if(TARGET ${CHIMAERA_MODULE_RUNTIME_TARGET})
    install(TARGETS ${CHIMAERA_MODULE_RUNTIME_TARGET}
      EXPORT ${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-targets
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin
    )
  endif()
  
  # Install headers
  install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
  
  # Generate and install package config files
  set(CONFIG_INSTALL_DIR "lib/cmake/${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}")
  
  # Create config file content
  set(CONFIG_CONTENT "
# ${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME} CMake Configuration

include(CMakeFindDependencyMacro)

# Find dependencies
find_dependency(hermes_shm REQUIRED)
find_dependency(Boost REQUIRED COMPONENTS fiber context system thread)
find_dependency(cereal REQUIRED)
find_dependency(Threads REQUIRED)

# Include targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-targets.cmake\")
")
  
  # Write config file
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-config.cmake"
    ${CONFIG_CONTENT}
  )
  
  # Install config file
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-config.cmake"
    DESTINATION ${CONFIG_INSTALL_DIR}
  )
  
  # Install targets file
  install(EXPORT ${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-targets
    FILE ${CHIMAERA_NAMESPACE}-${CHIMAERA_MODULE_NAME}-targets.cmake
    NAMESPACE ${CHIMAERA_NAMESPACE}::
    DESTINATION ${CONFIG_INSTALL_DIR}
  )
endfunction()
```

### 3.2 Module YAML Configuration

Each ChiMod directory contains `chimaera_mod.yaml`:

```yaml
# chimods/admin/chimaera_mod.yaml
module_name: admin
namespace: chimaera
version: 1.0.0
description: Admin module for Chimaera pool management
```

```yaml
# chimods/bdev/chimaera_mod.yaml
module_name: bdev
namespace: chimaera
version: 1.0.0
description: Block device ChiMod for storage operations
```

### 3.3 ChiMod CMakeLists.txt Example

```cmake
# chimods/admin/CMakeLists.txt

# Include common functionality
include(${CMAKE_SOURCE_DIR}/cmake/ChimaeraCommon.cmake)

# Create client library
add_chimod_client(
  SOURCES
    src/admin_client.cc
  LINK_LIBRARIES
    chimaera-core  # Core Chimaera library
  INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/include
)

# Create runtime library
add_chimod_runtime(
  SOURCES
    src/admin_runtime.cc
    src/autogen/admin_lib_exec.cc
  LINK_LIBRARIES
    chimaera-core
    ${CHIMAERA_MODULE_CLIENT_TARGET}  # Link to client lib
  INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/include
)

# Install the module
install_chimod()
```

### 3.4 External Project Usage

```cmake
# External project CMakeLists.txt

cmake_minimum_required(VERSION 3.16)
project(MyChimaeraApp)

# Find Chimaera core (provides ChimaeraCommon.cmake)
find_package(chimaera-core REQUIRED)

# Find specific ChiMods
find_package(chimaera-admin REQUIRED)
find_package(chimaera-bdev REQUIRED)

# Create application
add_executable(my_app src/main.cpp)

# Link to ChiMod client libraries
target_link_libraries(my_app
  PRIVATE
    chimaera::admin-client
    chimaera::bdev-client
    chimaera::cxx  # Core library
)
```

## 4. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
1. **Create new ChimaeraCommon.cmake**
   - Implement dependency finding
   - Create `add_chimod_client()` function
   - Create `add_chimod_runtime()` function
   - Implement `install_chimod()` function

2. **Create ChimaeraConfig.cmake.in template**
   - Package discovery configuration
   - Dependency propagation
   - Target export configuration

### Phase 2: Module Migration (Week 2)
1. **Update admin module**
   - Create `chimaera_mod.yaml`
   - Simplify CMakeLists.txt
   - Test build and installation

2. **Update bdev module**
   - Create `chimaera_mod.yaml`
   - Simplify CMakeLists.txt
   - Test build and installation

3. **Update other modules**
   - Apply same pattern to remaining ChiMods
   - Ensure consistent naming

### Phase 3: Testing and Documentation (Week 3)
1. **Create test external project**
   - Validate find_package works
   - Test linking and compilation
   - Verify runtime loading

2. **Update documentation**
   - Module development guide
   - External project integration guide
   - Migration guide from old system

3. **CI/CD updates**
   - Update build scripts
   - Add external project tests
   - Validate installation process

## 5. Migration Strategy

### For Existing ChiMods
1. Add `chimaera_mod.yaml` to each module directory
2. Replace previous `add_chimod_both()` calls with separate client/runtime calls
3. Update target references to use new naming convention
4. Test build and installation

### For External Projects
1. Update find_package calls to use new package names
2. Update target_link_libraries to use new target names
3. Remove any workarounds for old system complexity

## 6. Benefits of New Design

### Simplicity
- Clear, single-purpose functions
- Predictable target naming
- Minimal configuration required

### External-Friendly
- Standard CMake patterns
- Clear package discovery
- No hidden dependencies

### Maintainability
- Less CMake code to maintain
- Clear separation of concerns
- Easy to debug and extend

### Flexibility
- Easy to add new modules
- Simple to customize per-module
- Clear extension points

## 7. Example Implementations

### 7.1 Simple ChiMod (no dependencies)

```cmake
# chimods/simple/CMakeLists.txt
include(${CMAKE_SOURCE_DIR}/cmake/ChimaeraCommon.cmake)

add_chimod_client(
  SOURCES src/simple_client.cc
)

add_chimod_runtime(
  SOURCES src/simple_runtime.cc
  LINK_LIBRARIES ${CHIMAERA_MODULE_CLIENT_TARGET}
)

install_chimod()
```

### 7.2 Complex ChiMod (with dependencies)

```cmake
# chimods/complex/CMakeLists.txt
include(${CMAKE_SOURCE_DIR}/cmake/ChimaeraCommon.cmake)

# Find additional dependencies
find_package(OpenSSL REQUIRED)

add_chimod_client(
  SOURCES 
    src/complex_client.cc
    src/crypto.cc
  LINK_LIBRARIES
    OpenSSL::SSL
    OpenSSL::Crypto
  COMPILE_DEFINITIONS
    USE_OPENSSL=1
)

add_chimod_runtime(
  SOURCES 
    src/complex_runtime.cc
    src/autogen/complex_lib_exec.cc
  LINK_LIBRARIES
    ${CHIMAERA_MODULE_CLIENT_TARGET}
    chimaera-admin-client
  INCLUDE_LIBRARIES
    chimaera-admin-client
)

install_chimod()
```

## 8. Testing Strategy

### Unit Tests
- Test each CMake function in isolation
- Verify target creation and properties
- Validate installation paths

### Integration Tests
- Build all ChiMods with new system
- Test find_package from external project
- Verify runtime loading of modules

### Regression Tests
- Ensure all existing functionality works
- Compare with old system behavior
- Validate performance characteristics

## 9. Documentation Updates

### Files to Update
1. `doc/MODULE_DEVELOPMENT_GUIDE.md` - Complete rewrite for new system
2. `README.md` - Update build instructions
3. `doc/CMAKE_GUIDE.md` - New file documenting CMake infrastructure
4. `CLAUDE.md` - Update with new CMake patterns

### Key Documentation Topics
- Module creation walkthrough
- External project integration
- CMake function reference
- Migration from old system
- Troubleshooting guide

## 10. Risk Mitigation

### Potential Risks
1. **Breaking Changes**: Mitigate with clear migration guide
2. **Learning Curve**: Address with comprehensive documentation
3. **CI/CD Impact**: Update incrementally with fallback options
4. **Performance**: Ensure no runtime impact from changes

### Rollback Plan
- Keep old system in parallel during transition
- Tag stable version before migration
- Document rollback procedures

## 11. Success Metrics

### Quantitative
- Reduction in CMake code lines (target: 50% reduction)
- Build time improvement (target: 20% faster)
- External project setup time (target: < 5 minutes)

### Qualitative
- Developer feedback on simplicity
- Ease of debugging build issues
- External user adoption rate

## 12. Conclusion

This redesign fundamentally simplifies the Chimaera CMake infrastructure while improving external project integration. By following standard CMake patterns and reducing complexity, we create a more maintainable and user-friendly build system that scales with the project's growth.

The implementation roadmap provides a clear path forward with minimal disruption to existing users while delivering significant improvements in usability and maintainability.