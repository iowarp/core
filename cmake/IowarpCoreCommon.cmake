# IowarpCoreCommon.cmake - Common CMake functions for IOWarp Core and external repos
#
# This file provides shared utilities for both the IOWarp Core build and external
# repositories that depend on it.

# Guard against multiple inclusions
if(IOWARP_CORE_COMMON_INCLUDED)
  return()
endif()
set(IOWARP_CORE_COMMON_INCLUDED TRUE)

message(STATUS "Loading IowarpCoreCommon.cmake")

#------------------------------------------------------------------------------
# Dependency Target Resolution
#------------------------------------------------------------------------------

# Macro to resolve yaml-cpp target name across different versions
# Older versions use "yaml-cpp", newer versions use "yaml-cpp::yaml-cpp"
macro(resolve_yaml_cpp_target)
  if(NOT DEFINED YAML_CPP_LIBS)
    if(TARGET yaml-cpp::yaml-cpp)
      set(YAML_CPP_LIBS yaml-cpp::yaml-cpp)
      message(STATUS "Using yaml-cpp target: yaml-cpp::yaml-cpp")
    elseif(TARGET yaml-cpp)
      set(YAML_CPP_LIBS yaml-cpp)
      message(STATUS "Using yaml-cpp target: yaml-cpp")
    else()
      message(FATAL_ERROR "yaml-cpp target not found. Expected either 'yaml-cpp::yaml-cpp' or 'yaml-cpp'")
    endif()
  endif()
endmacro()

#------------------------------------------------------------------------------
# GPU Support Functions
#------------------------------------------------------------------------------

# Enable cuda boilerplate
macro(wrp_core_enable_cuda CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for" FORCE)
    endif()

    message(STATUS "USING CUDA ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler -diag-suppress=177,20014,20011,20012")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-format,-Wno-pedantic,-Wno-sign-compare,-Wno-unused-but-set-variable")
    enable_language(CUDA)

    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

    # Cache critical CUDA platform variables so they survive any nested
    # project() call (e.g. from external/llama.cpp) that may reset the
    # CMake variable scope.  Without caching, _CMAKE_CUDA_WHOLE_FLAG and
    # CMAKE_CUDA_COMPILE_OBJECT are silently lost and the generate step
    # fails with "Error required internal CMake variable not set."
    foreach(_cuda_var
            CMAKE_INCLUDE_FLAG_CUDA
            _CMAKE_CUDA_WHOLE_FLAG
            _CMAKE_CUDA_RDC_FLAG
            _CMAKE_CUDA_PTX_FLAG
            _CMAKE_CUDA_EXTRA_FLAGS
            _CMAKE_COMPILE_AS_CUDA_FLAG
            CMAKE_CUDA_COMPILE_OBJECT
            CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
            CMAKE_CUDA_LINK_EXECUTABLE
            CMAKE_CUDA_DEVICE_LINK_COMPILE_WHOLE_COMPILATION
            CMAKE_CUDA_COMPILER_HAS_DEVICE_LINK_PHASE
            CMAKE_CUDA_CREATE_SHARED_LIBRARY
            CMAKE_CUDA_CREATE_SHARED_MODULE
            CMAKE_CUDA_DEVICE_LINK_LIBRARY
            CMAKE_CUDA_DEVICE_LINK_EXECUTABLE
            CMAKE_CUDA_DEVICE_LINK_COMPILE
            CMAKE_CUDA_HOST_LINK_LAUNCHER
            CMAKE_SHARED_LIBRARY_CUDA_FLAGS
            CMAKE_SHARED_LIBRARY_CREATE_CUDA_FLAGS)
        if(DEFINED ${_cuda_var})
            set(${_cuda_var} "${${_cuda_var}}" CACHE INTERNAL "" FORCE)
        endif()
    endforeach()
endmacro()

# Enable rocm boilerplate
macro(wrp_core_enable_rocm GPU_RUNTIME CXX_STANDARD)
    set(GPU_RUNTIME ${GPU_RUNTIME})
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD ${CXX_STANDARD})
    set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
    set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )

    if(GPU_RUNTIME STREQUAL "CUDA")
        include_directories("${ROCM_ROOT}/include")
    endif()

    if(NOT HIP_FOUND)
        find_package(HIP REQUIRED)
    endif()
endmacro()

# Function for setting source files for rocm
function(set_rocm_sources MODE DO_COPY SRC_FILES ROCM_SOURCE_FILES_VAR)
    set(ROCM_SOURCE_FILES ${${ROCM_SOURCE_FILES_VAR}} PARENT_SCOPE)
    set(GPU_RUNTIME ${GPU_RUNTIME} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(ROCM_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/rocm_${MODE}/${SOURCE})
            configure_file(${SOURCE} ${ROCM_SOURCE} COPYONLY)
        else()
            set(ROCM_SOURCE ${SOURCE})
        endif()

        list(APPEND ROCM_SOURCE_FILES ${ROCM_SOURCE})
        set_source_files_properties(${ROCM_SOURCE} PROPERTIES LANGUAGE ${GPU_RUNTIME})
    endforeach()

    set(${ROCM_SOURCE_FILES_VAR} ${ROCM_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for setting source files for cuda
function(set_cuda_sources DO_COPY SRC_FILES CUDA_SOURCE_FILES_VAR)
    set(CUDA_SOURCE_FILES ${${CUDA_SOURCE_FILES_VAR}} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${SOURCE})
            configure_file(${SOURCE} ${CUDA_SOURCE} COPYONLY)
        else()
            set(CUDA_SOURCE ${SOURCE})
        endif()

        list(APPEND CUDA_SOURCE_FILES ${CUDA_SOURCE})
        set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    endforeach()

    set(${CUDA_SOURCE_FILES_VAR} ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for adding a ROCm library
function(add_rocm_gpu_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(gpu "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Function for adding a ROCm host-only library
function(add_rocm_host_library TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(host "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Function for adding a ROCm executable
function(add_rocm_host_executable TARGET)
    set(SRC_FILES ${ARGN})
    add_executable(${TARGET} ${SRC_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
endfunction()

# Function for adding a ROCm executable
function(add_rocm_gpu_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(exec "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_executable(${TARGET} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC amdhip64 amd_comgr)
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
endfunction()

# Helper: collect compile definitions for the Clang-CUDA custom command.
#
# The Clang custom-command path (add_custom_command) cannot inherit
# target_compile_definitions from linked interface libraries, so we
# must collect them explicitly.  This function:
#   1. Builds the base HSHM/Chimaera definitions (HSHM_ENABLE_CUDA, etc.)
#   2. Walks LINK_LIBS and recursively collects INTERFACE_COMPILE_DEFINITIONS
#   3. Returns the combined list via ${OUT_VAR}
#
# Usage (inside add_cuda_library / add_cuda_executable):
#   _wrp_core_collect_clang_cuda_defs("${CUDA_LINK_LIBS}" CLANG_CUDA_DEFS)
#
function(_wrp_core_collect_clang_cuda_defs LINK_LIBS OUT_VAR)
    set(_DEFS "")

    # --- Base definitions (mirroring hshm_target_compile_definitions) ---
    if(WRP_CORE_ENABLE_CUDA)
        list(APPEND _DEFS "-DHSHM_ENABLE_CUDA=1" "-DHSHM_ENABLE_ROCM=0")
    elseif(WRP_CORE_ENABLE_ROCM)
        list(APPEND _DEFS "-DHSHM_ENABLE_CUDA=0" "-DHSHM_ENABLE_ROCM=1")
    endif()
    if(HSHM_ENABLE_PTHREADS)
        list(APPEND _DEFS
            "-DHSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread"
            "-DHSHM_ENABLE_PTHREADS=1")
    else()
        list(APPEND _DEFS
            "-DHSHM_DEFAULT_THREAD_MODEL=hshm::thread::StdThread"
            "-DHSHM_ENABLE_PTHREADS=0")
    endif()
    if(WRP_CORE_ENABLE_CUDA)
        list(APPEND _DEFS
            "-DHSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda")
    elseif(WRP_CORE_ENABLE_ROCM)
        list(APPEND _DEFS
            "-DHSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Rocm")
    else()
        list(APPEND _DEFS
            "-DHSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::StdThread")
    endif()
    list(APPEND _DEFS
        "-DHSHM_DEFAULT_ALLOC_T=hipc::ThreadLocalAllocator"
        "-DHSHM_ENABLE_WINDOWS_THREADS=0"
        "-DHSHM_ENABLE_PROCFS_SYSINFO=1"
        "-DHSHM_ENABLE_WINDOWS_SYSINFO=0"
        "-DHSHM_COMPILER_MSVC=0"
        "-DHSHM_COMPILER_GNU=0"
        "-DHSHM_ENABLE_DOXYGEN=0"
        "-DHSHM_DEBUG_LOCK=0"
        "-DHSHM_LOG_LEVEL=${HSHM_LOG_LEVEL}"
        "-DHSHM_ENABLE_DLL_EXPORT=0"
        "-DHSHM_ENABLE_MPI=0"
    )

    # --- Collect transitive INTERFACE_COMPILE_DEFINITIONS from LINK_LIBS ---
    # Walk the linked targets and their transitive dependencies to pick up
    # definitions like HSHM_ENABLE_LIGHTBEAM, HSHM_ENABLE_ZMQ, etc. that
    # interface libraries propagate.
    set(_VISITED "")
    set(_QUEUE ${LINK_LIBS})
    while(_QUEUE)
        list(POP_FRONT _QUEUE _LIB)
        if(_LIB IN_LIST _VISITED)
            continue()
        endif()
        list(APPEND _VISITED "${_LIB}")
        if(NOT TARGET "${_LIB}")
            continue()
        endif()
        get_target_property(_LIB_DEFS "${_LIB}" INTERFACE_COMPILE_DEFINITIONS)
        if(_LIB_DEFS)
            foreach(_DEF IN LISTS _LIB_DEFS)
                # Skip generator expressions (they can't be evaluated here)
                string(FIND "${_DEF}" "$<" _GE_POS)
                if(_GE_POS EQUAL -1)
                    list(APPEND _DEFS "-D${_DEF}")
                endif()
            endforeach()
        endif()
        # Recurse into transitive link dependencies
        get_target_property(_LIB_LINK "${_LIB}" INTERFACE_LINK_LIBRARIES)
        if(_LIB_LINK)
            foreach(_DEP IN LISTS _LIB_LINK)
                if(TARGET "${_DEP}" AND NOT "${_DEP}" IN_LIST _VISITED)
                    list(APPEND _QUEUE "${_DEP}")
                endif()
            endforeach()
        endif()
    endwhile()

    list(REMOVE_DUPLICATES _DEFS)
    set(${OUT_VAR} ${_DEFS} PARENT_SCOPE)
endfunction()

# Function for adding a CUDA library
#
# When WRP_CORE_CLANG_CUDA_COMPILER is set (via wrp_core_find_clang_cuda()),
# compiles with Clang (C++20, coroutines in device code).  Otherwise uses
# NVCC via CMake's native CUDA language support.  The choice is transparent
# to callers -- source code should be portable across both compilers.
#
# Usage:
#   add_cuda_library(TARGET SHARED|STATIC DO_COPY source1.cu ...
#       [INCLUDE_DIRS dir1 dir2 ...]
#       [LINK_LIBS lib1 lib2 ...])
function(add_cuda_library TARGET SHARED DO_COPY)
    cmake_parse_arguments(CUDA "" "" "INCLUDE_DIRS;LINK_LIBS" ${ARGN})
    set(SRC_FILES ${CUDA_UNPARSED_ARGUMENTS})

    # Resolve "native" to the detected GPU architecture before add_library so
    # CMake does not attempt to re-detect the GPU at configure time for targets
    # created in subdirectories processed after the first enable_language(CUDA)
    # call.  CMAKE_CUDA_ARCHITECTURES_NATIVE is set by CMake during that first
    # language-enable pass (e.g. in context-transport-primitives or
    # context-runtime) and contains the concrete arch list (e.g. "89-real").
    # We must update the CACHE variable (not just a local variable) because
    # add_library reads the global CMAKE_CUDA_ARCHITECTURES value.
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
            CACHE STRING "CUDA architectures to compile for" FORCE)
    endif()

    if(WRP_CORE_CLANG_CUDA_COMPILER)
        # ---- Clang path ----
        _wrp_core_get_clang_gpu_arch(GPU_ARCH_FLAGS)

        set(INCLUDE_FLAGS "")
        foreach(DIR IN LISTS CUDA_INCLUDE_DIRS)
            list(APPEND INCLUDE_FLAGS "-I${DIR}")
        endforeach()

        # Collect all compile definitions (base + transitive from LINK_LIBS)
        _wrp_core_collect_clang_cuda_defs("${CUDA_LINK_LIBS}" CLANG_CUDA_DEFS)

        set(OBJECT_FILES "")
        foreach(SRC IN LISTS SRC_FILES)
            get_filename_component(SRC_NAME ${SRC} NAME_WE)
            get_filename_component(SRC_ABS ${SRC} ABSOLUTE)
            set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda/${SRC_NAME}.o")

            add_custom_command(
                OUTPUT ${OBJ_FILE}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda"
                COMMAND ${WRP_CORE_CLANG_CUDA_COMPILER}
                    -x cuda
                    -std=c++20
                    --cuda-path=${WRP_CORE_CLANG_CUDA_PATH}
                    ${GPU_ARCH_FLAGS}
                    -Wno-unknown-cuda-version
                    -fPIC
                    -fgpu-rdc
                    ${CLANG_CUDA_DEFS}
                    ${INCLUDE_FLAGS}
                    -c ${SRC_ABS}
                    -o ${OBJ_FILE}
                    "$<$<CONFIG:Debug>:-g>"
                    "$<$<CONFIG:Debug>:-O1>"
                    "$<$<CONFIG:Release>:-O2>"
                    "$<$<CONFIG:RelWithDebInfo>:-O2>"
                    "$<$<CONFIG:RelWithDebInfo>:-g>"
                DEPENDS ${SRC_ABS}
                COMMENT "Clang CUDA: Compiling ${SRC}"
                VERBATIM
            )
            list(APPEND OBJECT_FILES ${OBJ_FILE})
        endforeach()

        # Save source object files (before device link) for fatbin extraction
        set(_CUDA_SRC_OBJ_FILES ${OBJECT_FILES})

        # Device link step: nvcc -dlink resolves __cudaRegisterLinkedBinary__nv_*
        # symbols that clang-cuda -fgpu-rdc compilation leaves undefined.
        set(DEVICE_LINK_OBJ "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda/device_link.o")
        set(NVCC_ARCH_FLAGS "")
        if(CMAKE_CUDA_ARCHITECTURES)
            foreach(ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
                if(NOT "${ARCH}" STREQUAL "native")
                    string(REGEX REPLACE "-real|-virtual" "" ARCH_NUM "${ARCH}")
                    list(APPEND NVCC_ARCH_FLAGS
                        "--generate-code=arch=compute_${ARCH_NUM},code=[compute_${ARCH_NUM},sm_${ARCH_NUM}]")
                endif()
            endforeach()
        endif()
        if(NOT NVCC_ARCH_FLAGS)
            list(APPEND NVCC_ARCH_FLAGS "--generate-code=arch=compute_80,code=[compute_80,sm_80]")
        endif()
        add_custom_command(
            OUTPUT ${DEVICE_LINK_OBJ}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda"
            COMMAND ${CMAKE_CUDA_COMPILER}
                -dlink
                -shared
                ${NVCC_ARCH_FLAGS}
                -Xcompiler=-fPIC
                ${OBJECT_FILES}
                -o ${DEVICE_LINK_OBJ}
                -L${WRP_CORE_CLANG_CUDA_PATH}/lib64
                -lcudadevrt
                -lcudart
            DEPENDS ${OBJECT_FILES}
            COMMENT "NVCC: Device link for ${TARGET}"
            VERBATIM
        )
        list(APPEND OBJECT_FILES ${DEVICE_LINK_OBJ})

        add_library(${TARGET} ${SHARED} ${OBJECT_FILES})
        set_target_properties(${TARGET} PROPERTIES
            LINKER_LANGUAGE CXX
            POSITION_INDEPENDENT_CODE ON
        )
        set_property(TARGET ${TARGET} APPEND PROPERTY
            LINK_LIBRARIES "-L${WRP_CORE_CLANG_CUDA_PATH}/lib64;-lcudart")

        # Export the Clang-CUDA object files (excluding device_link.o) to parent scope
        # so embed_gpu_device_code() can extract fatbins from them.
        set(${TARGET}_CUDA_OBJ_FILES ${_CUDA_SRC_OBJ_FILES} PARENT_SCOPE)
    else()
        # ---- NVCC path ----
        set(CUDA_SOURCE_FILES "")
        set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)

        add_library(${TARGET} ${SHARED} ${CUDA_SOURCE_FILES})

        set_target_properties(${TARGET} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

        target_compile_options(${TARGET} PUBLIC
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

        if(SHARED STREQUAL "SHARED")
            set_target_properties(${TARGET} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                POSITION_INDEPENDENT_CODE ON
                CUDA_RUNTIME_LIBRARY Shared
            )
        else()
            set_target_properties(${TARGET} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                POSITION_INDEPENDENT_CODE ON
                CUDA_RUNTIME_LIBRARY Static
            )
        endif()

        if(CUDA_INCLUDE_DIRS)
            target_include_directories(${TARGET} PUBLIC ${CUDA_INCLUDE_DIRS})
        endif()
    endif()

    if(CUDA_LINK_LIBS)
        target_link_libraries(${TARGET} ${CUDA_LINK_LIBS})
    endif()
endfunction()

# Function for adding a CUDA executable
#
# When WRP_CORE_CLANG_CUDA_COMPILER is set (via wrp_core_find_clang_cuda()),
# compiles with Clang (C++20, coroutines in device code).  Otherwise uses
# NVCC via CMake's native CUDA language support.
#
# Usage:
#   add_cuda_executable(TARGET DO_COPY source1.cu ...
#       [INCLUDE_DIRS dir1 dir2 ...]
#       [LINK_LIBS lib1 lib2 ...])
function(add_cuda_executable TARGET DO_COPY)
    cmake_parse_arguments(CUDA "" "" "INCLUDE_DIRS;LINK_LIBS;DEFS" ${ARGN})
    set(SRC_FILES ${CUDA_UNPARSED_ARGUMENTS})

    if(WRP_CORE_CLANG_CUDA_COMPILER)
        # ---- Clang path ----
        _wrp_core_get_clang_gpu_arch(GPU_ARCH_FLAGS)

        set(INCLUDE_FLAGS "")
        foreach(DIR IN LISTS CUDA_INCLUDE_DIRS)
            list(APPEND INCLUDE_FLAGS "-I${DIR}")
        endforeach()

        # Collect extra per-target definitions passed via DEFS argument
        set(EXTRA_DEFS "")
        foreach(DEF IN LISTS CUDA_DEFS)
            list(APPEND EXTRA_DEFS "-D${DEF}")
        endforeach()

        # Collect all compile definitions (base + transitive from LINK_LIBS)
        _wrp_core_collect_clang_cuda_defs("${CUDA_LINK_LIBS}" CLANG_CUDA_DEFS)

        set(OBJECT_FILES "")
        foreach(SRC IN LISTS SRC_FILES)
            get_filename_component(SRC_NAME ${SRC} NAME_WE)
            get_filename_component(SRC_ABS ${SRC} ABSOLUTE)
            set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda/${SRC_NAME}.o")

            add_custom_command(
                OUTPUT ${OBJ_FILE}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda"
                COMMAND ${WRP_CORE_CLANG_CUDA_COMPILER}
                    -x cuda
                    -std=c++20
                    --cuda-path=${WRP_CORE_CLANG_CUDA_PATH}
                    ${GPU_ARCH_FLAGS}
                    -Wno-unknown-cuda-version
                    ${CLANG_CUDA_DEFS}
                    ${EXTRA_DEFS}
                    ${INCLUDE_FLAGS}
                    -c ${SRC_ABS}
                    -o ${OBJ_FILE}
                    "$<$<CONFIG:Debug>:-g>"
                    "$<$<CONFIG:Debug>:-O1>"
                    "$<$<CONFIG:Release>:-O2>"
                    "$<$<CONFIG:RelWithDebInfo>:-O2>"
                    "$<$<CONFIG:RelWithDebInfo>:-g>"
                DEPENDS ${SRC_ABS}
                COMMENT "Clang CUDA: Compiling ${SRC}"
                VERBATIM
            )
            list(APPEND OBJECT_FILES ${OBJ_FILE})
        endforeach()

        add_executable(${TARGET} ${OBJECT_FILES})
        set_target_properties(${TARGET} PROPERTIES
            LINKER_LANGUAGE CXX
        )
        set_property(TARGET ${TARGET} APPEND PROPERTY
            LINK_LIBRARIES "-L${WRP_CORE_CLANG_CUDA_PATH}/lib64;-lcudart")
    else()
        # ---- NVCC path ----
        set(CUDA_SOURCE_FILES "")
        set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
        add_executable(${TARGET} ${CUDA_SOURCE_FILES})
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )

        if(${DO_COPY})
            target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        endif()

        target_compile_options(${TARGET} PUBLIC
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

        if(CUDA_INCLUDE_DIRS)
            target_include_directories(${TARGET} PUBLIC ${CUDA_INCLUDE_DIRS})
        endif()
    endif()

    if(CUDA_LINK_LIBS)
        target_link_libraries(${TARGET} ${CUDA_LINK_LIBS})
    endif()
endfunction()

#------------------------------------------------------------------------------
# Clang CUDA Detection (for C++20 coroutines on GPU)
#------------------------------------------------------------------------------

# Find the Clang compiler for CUDA compilation.
# When found, add_cuda_library() and add_cuda_executable() will automatically
# use Clang instead of NVCC, enabling C++20 features (coroutines, concepts,
# etc.) in device code.
macro(wrp_core_find_clang_cuda)
    if(NOT WRP_CORE_CLANG_CUDA_COMPILER)
        find_program(WRP_CORE_CLANG_CUDA_COMPILER
            NAMES clang++-18 clang++-19 clang++-20 clang++
            PATHS /usr/bin /usr/local/bin
            DOC "Clang compiler for CUDA compilation"
        )
    endif()

    # Derive --cuda-path from CMAKE_CUDA_COMPILER (e.g., /usr/local/cuda-12.6/bin/nvcc -> /usr/local/cuda-12.6)
    if(NOT WRP_CORE_CLANG_CUDA_PATH)
        if(CMAKE_CUDA_COMPILER)
            get_filename_component(_cuda_bin_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
            get_filename_component(WRP_CORE_CLANG_CUDA_PATH "${_cuda_bin_dir}" DIRECTORY)
        elseif(CUDAToolkit_BIN_DIR)
            get_filename_component(WRP_CORE_CLANG_CUDA_PATH "${CUDAToolkit_BIN_DIR}" DIRECTORY)
        else()
            set(WRP_CORE_CLANG_CUDA_PATH "/usr/local/cuda")
        endif()
        set(WRP_CORE_CLANG_CUDA_PATH "${WRP_CORE_CLANG_CUDA_PATH}" CACHE PATH "CUDA toolkit root for Clang")
    endif()

    if(WRP_CORE_CLANG_CUDA_COMPILER)
        message(STATUS "Found Clang for CUDA: ${WRP_CORE_CLANG_CUDA_COMPILER}")
        message(STATUS "  CUDA path for Clang: ${WRP_CORE_CLANG_CUDA_PATH}")
    else()
        message(WARNING "Clang not found -- GPU coroutine targets will be unavailable")
    endif()
endmacro()

# Get the GPU architecture flag for Clang from CMAKE_CUDA_ARCHITECTURES.
# Converts CMake architecture numbers (e.g., 89) to Clang's --cuda-gpu-arch=sm_XX.
function(_wrp_core_get_clang_gpu_arch OUTPUT_VAR)
    set(ARCH_FLAGS "")
    if(CMAKE_CUDA_ARCHITECTURES)
        foreach(ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(NOT "${ARCH}" STREQUAL "native")
                list(APPEND ARCH_FLAGS "--cuda-gpu-arch=sm_${ARCH}")
            endif()
        endforeach()
    endif()
    # If no explicit architectures (or only "native"), default to sm_80
    if(NOT ARCH_FLAGS)
        list(APPEND ARCH_FLAGS "--cuda-gpu-arch=sm_80")
    endif()
    set(${OUTPUT_VAR} "${ARCH_FLAGS}" PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Jarvis Repo Management
#------------------------------------------------------------------------------

# Function for autoregistering a jarvis repo
macro(jarvis_repo_add REPO_PATH)
    # Get the file name of the source path
    get_filename_component(REPO_NAME ${REPO_PATH} NAME)

    # Install jarvis repo
    install(DIRECTORY ${REPO_PATH}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/jarvis)

    # Add jarvis repo after installation
    # Ensure install commands use env vars from host system, particularly PATH and PYTHONPATH
    install(CODE "execute_process(COMMAND env \"PATH=$ENV{PATH}\" \"PYTHONPATH=$ENV{PYTHONPATH}\" jarvis repo add ${CMAKE_INSTALL_PREFIX}/jarvis/${REPO_NAME})")
endmacro()

#------------------------------------------------------------------------------
# Doxygen Documentation
#------------------------------------------------------------------------------

function(add_doxygen_doc)
    set(options)
    set(oneValueArgs BUILD_DIR DOXY_FILE TARGET_NAME COMMENT)
    set(multiValueArgs)

    cmake_parse_arguments(DOXY_DOC
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    configure_file(
        ${DOXY_DOC_DOXY_FILE}
        ${DOXY_DOC_BUILD_DIR}/Doxyfile
        @ONLY
    )

    add_custom_target(${DOXY_DOC_TARGET_NAME}
        COMMAND
        ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY
        ${DOXY_DOC_BUILD_DIR}
        COMMENT
        "Building ${DOXY_DOC_COMMENT} with Doxygen"
        VERBATIM
    )

    message(STATUS "Added ${DOXY_DOC_TARGET_NAME} [Doxygen] target to build documentation")
endfunction()

#------------------------------------------------------------------------------
# Python Finding Utilities
#------------------------------------------------------------------------------

# FIND PYTHON
macro(find_first_path_python)
    # If scikit-build-core or the caller has already set Python3_EXECUTABLE
    # (e.g. pointing at the target interpreter with dev headers), respect it
    # and skip the PATH scan so we don't accidentally pick up a build-env
    # interpreter that lacks development headers.
    if(NOT Python3_EXECUTABLE AND DEFINED ENV{PATH})
        string(REPLACE ":" ";" PATH_LIST $ENV{PATH})

        foreach(PATH_ENTRY ${PATH_LIST})
            find_program(PYTHON_SCAN
                NAMES python3 python
                PATHS ${PATH_ENTRY}
                NO_DEFAULT_PATH
            )

            if(PYTHON_SCAN)
                message(STATUS "Found Python in PATH: ${PYTHON_SCAN}")
                set(Python_EXECUTABLE ${PYTHON_SCAN} CACHE FILEPATH "Python executable" FORCE)
                set(Python3_EXECUTABLE ${PYTHON_SCAN} CACHE FILEPATH "Python executable" FORCE)
                break()
            endif()
        endforeach()
    endif()

    set(Python_FIND_STRATEGY LOCATION)
    find_package(Python3 COMPONENTS Interpreter Development.Module)

    if(Python3_FOUND)
        message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Python3 not found")
    endif()
endmacro()

#------------------------------------------------------------------------------
# ChiMod Helper Functions
#------------------------------------------------------------------------------

# Helper function to link runtime to client library (called via DEFER)
# This allows linking to work regardless of which target is defined first
function(_chimaera_link_runtime_to_client RUNTIME_TARGET CLIENT_TARGET)
  if(TARGET ${CLIENT_TARGET})
    target_link_libraries(${RUNTIME_TARGET} PUBLIC ${CLIENT_TARGET})
    message(STATUS "Deferred linking: Runtime ${RUNTIME_TARGET} linked to client ${CLIENT_TARGET}")
  endif()
endfunction()

# Function to read repository namespace from chimaera_repo.yaml
# Searches up the directory tree from the given path to find chimaera_repo.yaml
function(read_repo_namespace output_var start_path)
  set(current_path "${start_path}")
  set(namespace "chimaera")  # Default fallback

  # Search up the directory tree for chimaera_repo.yaml
  while(NOT "${current_path}" STREQUAL "/" AND NOT "${current_path}" STREQUAL "")
    set(repo_file "${current_path}/chimaera_repo.yaml")
    if(EXISTS "${repo_file}")
      # Read and parse the YAML file
      file(READ "${repo_file}" REPO_YAML_CONTENT)
      string(REGEX MATCH "namespace: *([^\n\r]+)" NAMESPACE_MATCH "${REPO_YAML_CONTENT}")
      if(NAMESPACE_MATCH)
        string(REGEX REPLACE "namespace: *" "" namespace "${NAMESPACE_MATCH}")
        string(STRIP "${namespace}" namespace)
        break()
      endif()
    endif()

    # Move up one directory
    get_filename_component(current_path "${current_path}" DIRECTORY)
  endwhile()

  set(${output_var} "${namespace}" PARENT_SCOPE)
endfunction()

# Function to read module configuration from chimaera_mod.yaml
function(chimaera_read_module_config MODULE_DIR)
  set(CONFIG_FILE "${MODULE_DIR}/chimaera_mod.yaml")

  if(NOT EXISTS ${CONFIG_FILE})
    message(FATAL_ERROR "Missing chimaera_mod.yaml in ${MODULE_DIR}")
  endif()

  # Parse YAML file (simple regex parsing for key: value pairs)
  file(READ ${CONFIG_FILE} CONFIG_CONTENT)

  # Extract module_name
  string(REGEX MATCH "module_name:[ ]*([^\n\r]*)" MODULE_MATCH ${CONFIG_CONTENT})
  if(MODULE_MATCH)
    string(REGEX REPLACE "module_name:[ ]*" "" CHIMAERA_MODULE_NAME "${MODULE_MATCH}")
    string(STRIP "${CHIMAERA_MODULE_NAME}" CHIMAERA_MODULE_NAME)
  endif()
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)

  # Extract namespace
  string(REGEX MATCH "namespace:[ ]*([^\n\r]*)" NAMESPACE_MATCH ${CONFIG_CONTENT})
  if(NAMESPACE_MATCH)
    string(REGEX REPLACE "namespace:[ ]*" "" CHIMAERA_NAMESPACE "${NAMESPACE_MATCH}")
    string(STRIP "${CHIMAERA_NAMESPACE}" CHIMAERA_NAMESPACE)
  endif()
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)

  # Validate extracted values
  if(NOT CHIMAERA_MODULE_NAME)
    message(FATAL_ERROR "module_name not found in ${CONFIG_FILE}. Content preview: ${CONFIG_CONTENT}")
  endif()

  if(NOT CHIMAERA_NAMESPACE)
    message(FATAL_ERROR "namespace not found in ${CONFIG_FILE}. Content preview: ${CONFIG_CONTENT}")
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
# Automatic Cross-Namespace Dependencies (Unified Builds):
#   For non-chimaera namespaces (e.g., wrp_cte, wrp_cae), this function automatically
#   links chimaera admin and bdev client libraries if they are available as targets.
#   This enables wrp_* ChiMods to use chimaera ChiMod headers and functionality without
#   explicit dependency declarations in their CMakeLists.txt files.
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
  set(TARGET_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_client")

  # Create the library
  add_library(${TARGET_NAME} SHARED ${ARG_SOURCES})

  # Set C++ standard
  set(CHIMAERA_CXX_STANDARD 20)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})

  # Common compile definitions
  set(CHIMAERA_COMMON_COMPILE_DEFS
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )

  # Add compile definitions
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )

  # Add include directories with proper BUILD_INTERFACE and INSTALL_INTERFACE
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()

  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()

  # Link libraries - use chimaera::cxx for internal builds, hermes_shm::cxx for external
  set(CORE_LIB "")
  if(TARGET chimaera::cxx)
    set(CORE_LIB chimaera::cxx)
  elseif(TARGET hermes_shm::cxx)
    set(CORE_LIB hermes_shm::cxx)
  elseif(TARGET HermesShm::cxx)
    set(CORE_LIB HermesShm::cxx)
  elseif(TARGET cxx)
    set(CORE_LIB cxx)
  else()
    message(FATAL_ERROR "Neither chimaera::cxx, hermes_shm::cxx, HermesShm::cxx nor cxx target found")
  endif()

  # Automatically add chimaera ChiMod dependencies in unified builds
  set(CHIMAERA_CHIMOD_DEPS "")
  if(NOT "${CHIMAERA_NAMESPACE}" STREQUAL "chimaera")
    if(TARGET chimaera_admin_client)
      list(APPEND CHIMAERA_CHIMOD_DEPS chimaera_admin_client)
    endif()
    if(TARGET chimaera_bdev_client)
      list(APPEND CHIMAERA_CHIMOD_DEPS chimaera_bdev_client)
    endif()
  endif()

  # Clients only link to hshm::cxx (no Boost)
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${CORE_LIB}
      ${ARG_LINK_LIBRARIES}
      ${CHIMAERA_CHIMOD_DEPS}
  )

  # Create alias for external use
  add_library(${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_client ALIAS ${TARGET_NAME})

  # Set properties for installation
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CHIMAERA_MODULE_NAME}_client"
    OUTPUT_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_client"
  )

  # Install the client library
  set(MODULE_PACKAGE_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}")
  set(MODULE_EXPORT_NAME "${MODULE_PACKAGE_NAME}")

  install(TARGETS ${TARGET_NAME}
    EXPORT ${MODULE_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  # Install headers
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION include
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()

  # Precompiled headers for faster builds
  target_precompile_headers(${TARGET_NAME} PRIVATE
      <string> <vector> <memory> <unordered_map>
      <functional> <algorithm> <cstdint> <cstring> <iostream>
  )

  # Export module info to parent scope
  set(CHIMAERA_MODULE_CLIENT_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# GPU Device Code Embedding Function
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
# Automatic Cross-Namespace Dependencies (Unified Builds):
#   For non-chimaera namespaces (e.g., wrp_cte, wrp_cae), this function automatically
#   links chimaera admin and bdev runtime libraries if they are available as targets.
#   This enables wrp_* ChiMods to use chimaera ChiMod headers and functionality without
#   explicit dependency declarations in their CMakeLists.txt files.
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
  set(TARGET_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_runtime")

  # Separate _gpu.cc sources from regular sources
  set(CPU_SOURCES "")
  set(GPU_SOURCES "")
  foreach(SRC ${ARG_SOURCES})
    if(SRC MATCHES "_gpu\\.cc$")
      list(APPEND GPU_SOURCES ${SRC})
    else()
      list(APPEND CPU_SOURCES ${SRC})
    endif()
  endforeach()

  # Create the library (CPU sources only)
  add_library(${TARGET_NAME} SHARED ${CPU_SOURCES})

  # Build GPU companion library if GPU sources exist and GPU is enabled
  set(GPU_TARGET_NAME "${TARGET_NAME}_gpu")
  if(GPU_SOURCES)
    if(WRP_CORE_ENABLE_CUDA)
      add_cuda_library(${GPU_TARGET_NAME} SHARED TRUE ${GPU_SOURCES})
      target_link_libraries(${GPU_TARGET_NAME} PUBLIC ${TARGET_NAME} hshm::cuda_cxx)
      target_include_directories(${GPU_TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      )
      message(STATUS "GPU companion ${GPU_TARGET_NAME} created with CUDA for: ${GPU_SOURCES}")
    elseif(WRP_CORE_ENABLE_ROCM)
      add_rocm_gpu_library(${GPU_TARGET_NAME} SHARED TRUE ${GPU_SOURCES})
      target_link_libraries(${GPU_TARGET_NAME} PUBLIC ${TARGET_NAME} hshm::cxx)
      target_include_directories(${GPU_TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      )
      message(STATUS "GPU companion ${GPU_TARGET_NAME} created with ROCm for: ${GPU_SOURCES}")
    else()
      message(STATUS "GPU sources found but no GPU backend enabled, skipping: ${GPU_SOURCES}")
    endif()
  endif()

  # Set C++ standard
  set(CHIMAERA_CXX_STANDARD 20)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})

  # Common compile definitions
  set(CHIMAERA_COMMON_COMPILE_DEFS
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )

  # Add compile definitions (runtime always has CHIMAERA_RUNTIME=1)
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      CHIMAERA_RUNTIME=1
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )

  # Add include directories with proper BUILD_INTERFACE and INSTALL_INTERFACE
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()

  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()

  # Link libraries - use hermes_shm::cxx for internal builds, chimaera::cxx for external
  set(CORE_LIB "")
  if(TARGET chimaera::cxx)
    set(CORE_LIB chimaera::cxx)
  elseif(TARGET hermes_shm::cxx)
    set(CORE_LIB hermes_shm::cxx)
  elseif(TARGET HermesShm::cxx)
    set(CORE_LIB HermesShm::cxx)
  elseif(TARGET cxx)
    set(CORE_LIB cxx)
  else()
    message(FATAL_ERROR "Neither chimaera::cxx, hermes_shm::cxx, HermesShm::cxx nor cxx target found")
  endif()

  # Runtime-specific link libraries
  set(CHIMAERA_RUNTIME_LIBS
    Threads::Threads
  )

  # Automatically link to client library if it exists
  set(RUNTIME_LINK_LIBS ${CORE_LIB} ${CHIMAERA_RUNTIME_LIBS} ${ARG_LINK_LIBRARIES})

  # Try to find client target by name (handles cases where client was defined first)
  set(CLIENT_TARGET_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_client")
  if(TARGET ${CLIENT_TARGET_NAME})
    list(APPEND RUNTIME_LINK_LIBS ${CLIENT_TARGET_NAME})
    message(STATUS "Runtime ${TARGET_NAME} linking to client ${CLIENT_TARGET_NAME}")
  elseif(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
    # Fallback to variable-based approach for compatibility
    list(APPEND RUNTIME_LINK_LIBS ${CHIMAERA_MODULE_CLIENT_TARGET})
    message(STATUS "Runtime ${TARGET_NAME} linking to client ${CHIMAERA_MODULE_CLIENT_TARGET}")
  endif()

  # Automatically add chimaera ChiMod dependencies in unified builds
  if(NOT "${CHIMAERA_NAMESPACE}" STREQUAL "chimaera")
    if(TARGET chimaera_admin_runtime)
      list(APPEND RUNTIME_LINK_LIBS chimaera_admin_runtime)
    endif()
    if(TARGET chimaera_bdev_runtime)
      list(APPEND RUNTIME_LINK_LIBS chimaera_bdev_runtime)
    endif()
  endif()

  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${RUNTIME_LINK_LIBS} 
      rt  # POSIX real-time library for async I/O
  )

  # Create alias for external use
  add_library(${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_runtime ALIAS ${TARGET_NAME})

  # Set properties for installation
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CHIMAERA_MODULE_NAME}_runtime"
    OUTPUT_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_runtime"
  )

  # Use cmake_language(DEFER) to link to client after all targets are processed
  cmake_language(EVAL CODE "
    cmake_language(DEFER CALL _chimaera_link_runtime_to_client \"${TARGET_NAME}\" \"${CLIENT_TARGET_NAME}\")
  ")

  # Install the runtime library (add to existing export set if client exists)
  set(MODULE_PACKAGE_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}")
  set(MODULE_EXPORT_NAME "${MODULE_PACKAGE_NAME}")

  install(TARGETS ${TARGET_NAME}
    EXPORT ${MODULE_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  # Install headers (only if not already installed by client)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include" AND NOT CHIMAERA_MODULE_CLIENT_TARGET)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION include
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()

  # Generate and install package config files (only do this once per module)
  set(SHOULD_GENERATE_CONFIG FALSE)
  if(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
    set(SHOULD_GENERATE_CONFIG TRUE)
  elseif(NOT CHIMAERA_MODULE_CLIENT_TARGET)
    set(SHOULD_GENERATE_CONFIG TRUE)
  endif()

  if(SHOULD_GENERATE_CONFIG)
    # Export targets file
    install(EXPORT ${MODULE_EXPORT_NAME}
      FILE ${MODULE_EXPORT_NAME}.cmake
      NAMESPACE ${CHIMAERA_NAMESPACE}::
      DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Generate Config.cmake file
    set(CONFIG_CONTENT "
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Find the core Chimaera package (handles all other dependencies)
find_dependency(chimaera REQUIRED)

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/${MODULE_EXPORT_NAME}.cmake\")

# Provide components
check_required_components(${MODULE_PACKAGE_NAME})
")

    # Write Config.cmake template
    set(CONFIG_IN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake.in")
    file(WRITE "${CONFIG_IN_FILE}" "${CONFIG_CONTENT}")

    # Configure and install Config.cmake
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
      "${CONFIG_IN_FILE}"
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake"
      INSTALL_DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Generate ConfigVersion.cmake
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}ConfigVersion.cmake"
      VERSION 1.0.0
      COMPATIBILITY SameMajorVersion
    )

    # Install Config and ConfigVersion files
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}ConfigVersion.cmake"
      DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Collect targets for status message
    set(INSTALLED_TARGETS ${TARGET_NAME})
    if(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
      list(APPEND INSTALLED_TARGETS ${CHIMAERA_MODULE_CLIENT_TARGET})
    endif()

    message(STATUS "Created module package: ${MODULE_PACKAGE_NAME}")
    message(STATUS "  Targets: ${INSTALLED_TARGETS}")
    message(STATUS "  Aliases: ${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_client, ${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_runtime")
  endif()

  # Precompiled headers for faster builds
  target_precompile_headers(${TARGET_NAME} PRIVATE
      <string> <vector> <memory> <unordered_map>
      <functional> <algorithm> <cstdint> <cstring> <iostream>
  )

  # Export module info to parent scope
  set(CHIMAERA_MODULE_RUNTIME_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
endfunction()

message(STATUS "IowarpCoreCommon.cmake loaded successfully")
