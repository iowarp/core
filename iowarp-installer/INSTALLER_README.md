# IOWarp Installer

IOWarp is a comprehensive platform for context management in scientific computing. This package provides an easy way to install IOWarp Core using uvx.

## Quick Start

```bash
# Install the installer with uvx
uvx install iowarp

# Install IOWarp Core
iowarp install

# Check installation status
iowarp status

# Set up environment
source ~/.local/bin/iowarp-env.sh
```

## What This Does

This installer will:

1. **Check Dependencies**: Verify that required system dependencies are installed
2. **Clone Repository**: Download the IOWarp Core source code from GitHub
3. **Build**: Compile IOWarp Core using CMake with optimized settings
4. **Install**: Install binaries, libraries, and headers to `~/.local` (or custom prefix)
5. **Configure**: Set up environment scripts and configuration files

## Components Installed

IOWarp Core includes five integrated components:

- **Context Transport Primitives**: High-performance shared memory data structures
- **Chimaera Runtime**: Modular runtime system with coroutine-based task execution
- **Context Transfer Engine**: Multi-tiered I/O buffering system
- **Context Assimilation Engine**: Data ingestion and processing engine
- **Context Exploration Engine**: Interactive tools for data exploration

## System Requirements

### Required Dependencies

- C++17 compatible compiler (GCC >= 9, Clang >= 10)
- CMake >= 3.20
- pkg-config
- Boost >= 1.70 (context, fiber, system components)
- libelf (ELF binary parsing)
- ZeroMQ (libzmq)
- POSIX threads

### Optional Dependencies

- MPI (MPICH, OpenMPI) for distributed computing
- CUDA Toolkit >= 11.0 for GPU support
- ROCm/HIP >= 4.0 for AMD GPU support
- HDF5 for scientific data formats
- Various compression libraries (zstd, lz4, etc.)

### Installation on Ubuntu/Debian

```bash
# Required dependencies
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libboost-context-dev libboost-fiber-dev libboost-system-dev \
  libelf-dev libzmq3-dev

# Optional dependencies
sudo apt-get install -y \
  libhdf5-dev libmpich-dev \
  libbz2-dev liblzo2-dev libzstd-dev liblz4-dev
```

## Usage

### Basic Installation

```bash
# Install to default location (~/.local)
iowarp install

# Install to custom location
iowarp install --prefix /opt/iowarp

# Debug build
iowarp install --build-type Debug
```

### Configuration Management

```bash
# Check installation status
iowarp status

# Get configuration values
iowarp config get version
iowarp config get prefix

# Set configuration values
iowarp config set custom_setting "value"

# Show environment variables
iowarp config env
```

### Testing

```bash
# Run all tests
iowarp test

# Run tests for specific component
iowarp test --component runtime
iowarp test --component cte

# Run tests matching pattern
iowarp test --filter "*bdev*"

# List available tests
iowarp test --list
```

### Environment Setup

After installation, set up your environment:

```bash
# Source the environment script
source ~/.local/bin/iowarp-env.sh

# Or add to your shell profile
echo "source ~/.local/bin/iowarp-env.sh" >> ~/.bashrc
```

## Advanced Usage

### Dependency Checking

```bash
# Check system dependencies without installing
iowarp install --check-deps-only
```

### Custom Build Options

The installer supports various build configurations:

- `--build-type`: Debug, Release, RelWithDebInfo
- `--prefix`: Installation prefix
- `--source-dir`: Custom source directory
- `--keep-build`: Keep build directory after installation

### Docker Alternative

If you prefer Docker:

```bash
docker pull iowarp/iowarp-build:latest
docker run -it iowarp/iowarp-build:latest
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Use `iowarp install --check-deps-only` to identify missing packages
2. **Build Failures**: Check that you have enough disk space (5GB) and memory
3. **Permission Issues**: Make sure you have write access to the installation prefix

### Getting Help

- Visit the [IOWarp documentation](https://grc.iit.edu/docs/category/iowarp)
- Check the [GitHub repository](https://github.com/iowarp/iowarp-core)
- Report issues on [GitHub Issues](https://github.com/iowarp/iowarp-core/issues)

## License

IOWarp Core is licensed under the BSD 3-Clause License.

## About

IOWarp Core is developed at the GRC lab at Illinois Institute of Technology as part of the IOWarp project, supported by the National Science Foundation (NSF).