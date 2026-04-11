# AutoConfigureRust.cmake
# Automatically finds and configures Rust toolchain for IOWarp Core
# This module should be included after WRP_CORE_ENABLE_RUST is set

if(WRP_CORE_ENABLE_RUST AND (NOT Rust_COMPILER OR NOT Rust_CARGO))
    message(STATUS "Auto-configuring Rust toolchain...")

    # Try to find rustup
    find_program(RUSTUP_EXE rustup
        PATHS
            "$ENV{HOME}/.cargo/bin"
            "$ENV{CARGO_HOME}/bin"
            "$ENV{RUSTUP_HOME}/bin"
        DOC "Rustup executable"
    )

    if(RUSTUP_EXE)
        message(STATUS "Found rustup: ${RUSTUP_EXE}")

        # Get the active toolchain
        execute_process(
            COMMAND ${RUSTUP_EXE} show active-toolchain
            OUTPUT_VARIABLE RUSTUP_ACTIVE_TOOLCHAIN
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(RUSTUP_ACTIVE_TOOLCHAIN)
            # Extract toolchain name (format: "toolchain-name (active)" or just "toolchain-name")
            string(REGEX MATCH "^[^ ]+" RUSTUP_TOOLCHAIN_NAME "${RUSTUP_ACTIVE_TOOLCHAIN}")
            message(STATUS "Active rustup toolchain: ${RUSTUP_TOOLCHAIN_NAME}")

            # Get toolchain path
            execute_process(
                COMMAND ${RUSTUP_EXE} which rustc
                OUTPUT_VARIABLE RUSTUP_RUSTC_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )

            if(RUSTUP_RUSTC_PATH)
                get_filename_component(RUSTUP_TOOLCHAIN_DIR "${RUSTUP_RUSTC_PATH}" DIRECTORY)
                message(STATUS "Rust toolchain directory: ${RUSTUP_TOOLCHAIN_DIR}")

                if(NOT Rust_COMPILER)
                    set(Rust_COMPILER "${RUSTUP_TOOLCHAIN_DIR}/rustc" CACHE FILEPATH "Rust compiler" FORCE)
                    message(STATUS "Auto-set Rust_COMPILER: ${Rust_COMPILER}")
                endif()

                if(NOT Rust_CARGO)
                    set(Rust_CARGO "${RUSTUP_TOOLCHAIN_DIR}/cargo" CACHE FILEPATH "Cargo" FORCE)
                    message(STATUS "Auto-set Rust_CARGO: ${Rust_CARGO}")
                endif()

                # Also set Rust_RUSTUP to help Corrosion's FindRust
                set(Rust_RUSTUP "${RUSTUP_EXE}" CACHE FILEPATH "Rustup executable" FORCE)
                message(STATUS "Auto-set Rust_RUSTUP: ${Rust_RUSTUP}")
            endif()
        else()
            message(WARNING "Could not determine active rustup toolchain")
        endif()
    else()
        message(WARNING "rustup not found - Rust auto-configuration will not work")
    endif()

    # Auto-find cereal if not already set
    if(NOT cereal_DIR)
        # Try spack locations
        file(GLOB SPACK_CEREAL_DIRS
            "$ENV{HOME}/spack/opt/spack/*/cereal*/lib*/cmake/cereal"
            "/opt/spack/opt/spack/*/cereal*/lib*/cmake/cereal"
        )
        if(SPACK_CEREAL_DIRS)
            list(GET SPACK_CEREAL_DIRS 0 SPACK_CEREAL_DIR)
            set(cereal_DIR "${SPACK_CEREAL_DIR}" CACHE PATH "cereal directory" FORCE)
            message(STATUS "Auto-found spack cereal: ${cereal_DIR}")
        endif()
    endif()
endif()
