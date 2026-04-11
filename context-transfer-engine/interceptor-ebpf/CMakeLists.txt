# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# All rights reserved.
#
# This file is part of IOWarp Core.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

message(STATUS "Configuring eBPF I/O interceptor adapter")

# Check if Rust/Corrosion is available
if(NOT COMMAND corrosion_import_crate)
    message(FATAL_ERROR "eBPF adapter requires Rust/Corrosion. Enable WRP_CORE_ENABLE_RUST.")
endif()

# Set up paths
set(EBPF_WORKSPACE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(EBPF_KERNEL_DIR ${EBPF_WORKSPACE_ROOT}/interceptor-ebpf)
set(EBPF_TARGET_DIR ${EBPF_WORKSPACE_ROOT}/target)
set(EBPF_KERNEL_BIN ${EBPF_TARGET_DIR}/bpfel-unknown-none/ebpf/interceptor-ebpf)

# Check for nightly Rust toolchain
execute_process(
    COMMAND rustup show active-toolchain
    OUTPUT_VARIABLE RUST_ACTIVE_TOOLCHAIN
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT RUST_ACTIVE_TOOLCHAIN MATCHES "nightly")
    message(WARNING "eBPF adapter requires nightly Rust toolchain. Found: ${RUST_ACTIVE_TOOLCHAIN}")
    message(STATUS "Install with: rustup default nightly")
endif()

# Check for rust-src component (required for -Z build-std)
execute_process(
    COMMAND rustup component list --toolchain nightly
    OUTPUT_VARIABLE RUST_COMPONENTS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT RUST_COMPONENTS MATCHES "rust-src")
    message(WARNING "eBPF adapter requires rust-src component. Install with: rustup component add rust-src --toolchain nightly")
else()
    message(STATUS "Found rust-src component for nightly toolchain")
endif()

# Note: bpfel-unknown-none is built from source via -Z build-std
# It is NOT a prebuilt target, so we don't use 'rustup target add'
message(STATUS "eBPF target bpfel-unknown-none will be built from source via -Z build-std")

# Custom command to build eBPF kernel code
# This must complete BEFORE userspace code is built
# Working directory is set to EBPF_KERNEL_DIR to pick up .cargo/config.toml
# which contains: target = "bpfel-unknown-none" and build-std settings
add_custom_command(
    OUTPUT ${EBPF_KERNEL_BIN}
    COMMAND rustup run nightly cargo build --profile ebpf
    WORKING_DIRECTORY ${EBPF_KERNEL_DIR}
    COMMENT "Building eBPF kernel code"
    VERBATIM
)

# Create a custom target for the eBPF build
add_custom_target(ebpf-kernel ALL
    DEPENDS ${EBPF_KERNEL_BIN}
    COMMENT "eBPF kernel target"
)

# Import the userspace crate using Corrosion
# interceptor-ebpf-common is a dependency that gets built automatically
corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    CRATES interceptor-user
)

# Ensure eBPF kernel is built before userspace
add_dependencies(interceptor-user ebpf-kernel)

# Install the userspace binary
# corrosion_import_crate doesn't create a CMake target immediately,
# so we use install(PROGRAMS) with the actual output path
install(PROGRAMS
    ${EBPF_TARGET_DIR}/release/interceptor-user
    DESTINATION bin
    RENAME interceptor-ebpf
)

message(STATUS "eBPF I/O interceptor adapter configured")
message(STATUS "  Binary: interceptor-user (installed to bin/)")
