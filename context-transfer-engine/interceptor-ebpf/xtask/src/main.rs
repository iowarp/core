/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

//! Build automation helper for eBPF interceptor project.
//!
//! This task runner provides commands for building and testing the eBPF
//! interceptor components.

use clap::Parser;
use std::process::Command;

/// Available commands for the build task runner.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
enum Commands {
    /// Build the eBPF program.
    BuildEbpf,
    /// Build the user-space program.
    BuildUser,
    /// Build all components.
    BuildAll,
    /// Clean build artifacts.
    Clean,
}

/// Run a cargo command and return its exit status.
fn run_cargo(args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("cargo")
        .args(args)
        .status()
        .map_err(|e| format!("Failed to run cargo: {}", e))?;

    if !status.success() {
        return Err(format!("Cargo command failed with {:?}", status).into());
    }

    Ok(())
}

/// Build the eBPF program for the target architecture.
fn build_ebpf() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building eBPF program...");
    run_cargo(&[
        "build",
        "--manifest-path",
        "interceptor-ebpf/Cargo.toml",
        "--release",
        "-Z",
        "build-std=core,alloc,compiler_builtins",
        "--target",
        "bpfel-unknown-none",
    ])?;
    println!("eBPF program built successfully");
    Ok(())
}

/// Build the user-space controller program.
fn build_user() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building user-space program...");
    run_cargo(&[
        "build",
        "--manifest-path",
        "interceptor-user/Cargo.toml",
        "--release",
    ])?;
    println!("User-space program built successfully");
    Ok(())
}

/// Clean all build artifacts.
fn clean() -> Result<(), Box<dyn std::error::Error>> {
    println!("Cleaning build artifacts...");
    run_cargo(&["clean"])?;
    println!("Build artifacts cleaned");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = Commands::parse();

    match command {
        Commands::BuildEbpf => build_ebpf()?,
        Commands::BuildUser => build_user()?,
        Commands::BuildAll => {
            build_ebpf()?;
            build_user()?;
        }
        Commands::Clean => clean()?,
    }

    Ok(())
}
