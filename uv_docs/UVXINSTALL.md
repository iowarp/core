# IOWarp uvx Installation Guide

This guide shows how to set up uvx-compatible installation for IOWarp Core.

## Overview

I've created two Python packages to enable uvx installation:

1. **`iowarp-core`** - Full package with source code (for development)
2. **`iowarp`** - Lightweight installer package (for users)

Both packages contain installation scripts that:
- Check system dependencies 
- Clone the repository from GitHub
- Build with CMake
- Install to a Python-compatible directory structure

## Quick Start with uvx

### Option 1: Install with the lightweight installer (Recommended)

```bash
# Install just the installer tool
uvx install iowarp

# Use it to install IOWarp Core
iowarp install

# Check status
iowarp status

# Set up environment
source ~/.local/bin/iowarp-env.sh
```

### Option 2: Install the full package

```bash
# Install the full package (includes source)
uvx install iowarp-core

# Use the included tools
iowarp-install
iowarp-config status
iowarp-test
```

## Installation Process

The installer performs these steps:

1. **Dependency Check**: Verifies required system packages
2. **Repository Clone**: Downloads IOWarp Core from GitHub with submodules
3. **CMake Configuration**: Sets up build with optimal settings
4. **Build**: Compiles all components with parallel jobs
5. **Installation**: Installs to `~/.local` (or custom prefix)
6. **Environment Setup**: Creates shell scripts for easy environment configuration

## System Requirements

### Required Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libboost-context-dev libboost-fiber-dev libboost-system-dev \
  libelf-dev libzmq3-dev

# RHEL/CentOS/Fedora  
sudo dnf install -y \
  gcc gcc-c++ cmake pkgconfig \
  boost-devel elfutils-libelf-devel zeromq-devel

# macOS (with Homebrew)
brew install cmake boost libelf zeromq pkg-config
```

### Optional Dependencies

```bash
# For MPI support
sudo apt-get install -y libmpich-dev  # or libopenmpi-dev

# For HDF5 support (scientific data formats)
sudo apt-get install -y libhdf5-dev

# For compression support
sudo apt-get install -y \
  libbz2-dev liblzo2-dev libzstd-dev liblz4-dev zlib1g-dev
```

## Usage Examples

### Basic Installation

```bash
# Install to default location (~/.local)
iowarp install

# Install to custom location
iowarp install --prefix /opt/iowarp

# Debug build for development
iowarp install --build-type Debug

# Keep build directory for running tests
iowarp install --keep-build
```

### Configuration Management

```bash
# Check installation status
iowarp status

# Show environment variables needed
iowarp config env

# Check what dependencies are available
iowarp install --check-deps-only
```

### Testing (requires --keep-build)

```bash
# Run all tests
iowarp test

# Run tests for specific component
iowarp test --component runtime
iowarp test --component cte

# List available tests
iowarp test --list
```

## Environment Setup

After installation, set up your environment:

```bash
# One-time setup
source ~/.local/bin/iowarp-env.sh

# Add to your shell profile for permanent setup
echo "source ~/.local/bin/iowarp-env.sh" >> ~/.bashrc
```

The environment script sets up:
- `PATH` for IOWarp binaries
- `LD_LIBRARY_PATH` for shared libraries
- `PKG_CONFIG_PATH` for pkg-config files
- `CMAKE_MODULE_PATH` for CMake modules
- `IOWARP_PREFIX` and `IOWARP_ROOT` variables

## Advanced Configuration

### Custom CMake Options

The installer uses these CMake settings by default:

```cmake
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=~/.local
-DWRP_CORE_ENABLE_RUNTIME=ON
-DWRP_CORE_ENABLE_CTE=ON
-DWRP_CORE_ENABLE_CAE=ON
-DWRP_CORE_ENABLE_CEE=ON
-DWRP_CORE_ENABLE_PYTHON=ON
-DBUILD_SHARED_LIBS=ON
```

### Directory Structure

After installation, you'll have:

```
~/.local/
├── bin/
│   ├── iowarp-env.sh          # Environment setup script
│   └── <iowarp executables>   # IOWarp binaries
├── lib/
│   ├── *.so                   # Shared libraries
│   ├── cmake/                 # CMake modules
│   └── pkgconfig/             # pkg-config files
├── include/
│   └── <header files>         # C++ headers
└── etc/
    └── iowarp/
        └── config.json        # Installation config
```

## Building the Packages

If you want to build and publish the packages yourself:

### Build Both Packages

```bash
# In the repository root
cd /path/to/iowarp-core

# Build the full package
python -m build

# Build the installer package
cd iowarp-installer
python -m build
```

### Local Testing

```bash
# Test the installer package locally
cd iowarp-installer
pip install -e .

# Test installation
iowarp install --check-deps-only
```

### Publishing to PyPI

```bash
# Install publishing tools
pip install twine

# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# Upload to production PyPI
python -m twine upload dist/*
```

## Comparison with Other Installation Methods

| Method | Pros | Cons |
|--------|------|------|
| **uvx** | ✅ Easy to use<br>✅ Isolated environment<br>✅ Good for users | ❌ Requires Python packaging |
| **Docker** | ✅ Complete environment<br>✅ Reproducible | ❌ Heavyweight<br>❌ Container overhead |
| **Spack** | ✅ Great for HPC<br>✅ Dependency management | ❌ Complex for simple use cases |
| **Manual CMake** | ✅ Full control<br>✅ Optimal performance | ❌ Complex dependencies<br>❌ Manual environment setup |

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Check what's missing
   iowarp install --check-deps-only
   
   # Install missing packages (Ubuntu)
   sudo apt-get install <missing-packages>
   ```

2. **Build Failures**
   ```bash
   # Clean and retry
   rm -rf ~/.cache/iowarp-core
   iowarp install
   ```

3. **Environment Issues**
   ```bash
   # Re-source the environment
   source ~/.local/bin/iowarp-env.sh
   
   # Check paths
   iowarp config env
   ```

4. **Permission Issues**
   ```bash
   # Install to a different prefix
   iowarp install --prefix ~/iowarp
   ```

### Getting Help

- Check the [IOWarp documentation](https://grc.iit.edu/docs/category/iowarp)
- Visit the [GitHub repository](https://github.com/iowarp/iowarp-core)
- Report issues on [GitHub Issues](https://github.com/iowarp/iowarp-core/issues)

## Next Steps

Once installed, you can:

1. **Explore the Documentation**: Read component-specific READMEs
2. **Run Examples**: Try the example code in the repository
3. **Develop Applications**: Use IOWarp Core in your scientific computing projects
4. **Contribute**: Help improve IOWarp Core

The uvx installation provides a user-friendly way to get started with IOWarp Core while maintaining the full power and flexibility of the underlying CMake-based build system.