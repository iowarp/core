# IOWarp Core - Quick Start Installation

## One-Command Installation

```bash
# Install to ~/.local (recommended for development)
INSTALL_PREFIX=$HOME/.local python3 setup.py install
```

After installation, add to your `~/.bashrc`:

```bash
export INSTALL_PREFIX=$HOME/.local
export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export PATH="$INSTALL_PREFIX/bin:$PATH"
```

## What This Does

The installation system:

1. **Detects** which dependencies are already installed on your system
2. **Builds** only the missing dependencies from submodules
3. **Installs** everything to a single prefix location
4. **Configures** IOWarp Core to use the installed dependencies

## Installation Time

Typical installation times (on a 16-core system with `BUILD_JOBS=16`):

- **All dependencies missing**: ~15-20 minutes
- **Some dependencies present**: ~5-10 minutes
- **All dependencies present**: ~2-3 minutes (only builds IOWarp Core)

## Customization

### Install to a different location
```bash
INSTALL_PREFIX=/opt/iowarp python3 setup.py install
```

### Use more build jobs (faster on multi-core systems)
```bash
BUILD_JOBS=32 INSTALL_PREFIX=$HOME/.local python3 setup.py install
```

### Skip HDF5 (if you don't need CAE component)
```bash
BUILD_HDF5=no INSTALL_PREFIX=$HOME/.local python3 setup.py install
```

## Verify Installation

After installation, verify IOWarp Core is properly installed:

```bash
# Check that libraries are installed
ls $INSTALL_PREFIX/lib/lib*iowarp*
ls $INSTALL_PREFIX/lib/libchimaera*
ls $INSTALL_PREFIX/lib/libhshm*

# Check that CMake config is installed
ls $INSTALL_PREFIX/lib/cmake/iowarp-core/

# Test that pkg-config can find dependencies
pkg-config --modversion libzmq
pkg-config --modversion yaml-cpp
```

## Next Steps

See [INSTALL.md](INSTALL.md) for detailed installation documentation and troubleshooting.

See [CLAUDE.md](CLAUDE.md) for development guidelines and build configuration.
