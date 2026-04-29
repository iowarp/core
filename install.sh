#!/bin/bash
# install.sh - Install IOWarp Core using conda-build
# This script builds and installs IOWarp Core from source
# It will automatically install Miniconda if conda is not detected
#
# Usage:
#   ./install.sh              # Build with default (release) preset
#   ./install.sh release      # Build with release preset
#   ./install.sh release-fuse # Build with FUSE adapter enabled
#   ./install.sh debug        # Build with debug preset
#   ./install.sh conda        # Build with conda-optimized preset
#   ./install.sh cuda         # Build with CUDA preset
#   ./install.sh rocm         # Build with ROCm preset

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse preset argument (default to release)
PRESET="${1:-release}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================"
echo -e "IOWarp Core - Installation"
echo -e "======================================================================${NC}"
echo ""
echo -e "${BLUE}Preset: ${YELLOW}$PRESET${NC}"
echo ""

# Function to install Miniconda
install_miniconda() {
    echo -e "${YELLOW}Conda not detected. Installing Miniconda...${NC}"
    echo ""

    # Default Miniconda installation directory
    MINICONDA_DIR="$HOME/miniconda3"

    # Detect platform
    if [[ "$OSTYPE" == "linux"* ]]; then
        PLATFORM="Linux"
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$ARCH" == "aarch64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else
            echo -e "${RED}Error: Unsupported Linux architecture: $ARCH${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macOS"
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif [[ "$ARCH" == "arm64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            echo -e "${RED}Error: Unsupported macOS architecture: $ARCH${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: Unsupported operating system: $OSTYPE${NC}"
        exit 1
    fi

    echo -e "${BLUE}Detected platform: $PLATFORM ($ARCH)${NC}"
    echo -e "${BLUE}Installation directory: $MINICONDA_DIR${NC}"
    echo ""

    # Download Miniconda installer
    INSTALLER_SCRIPT="/tmp/miniconda_installer.sh"
    echo -e "${BLUE}Downloading Miniconda installer...${NC}"
    curl -L -o "$INSTALLER_SCRIPT" "$INSTALLER_URL"

    # Install Miniconda
    echo -e "${BLUE}Installing Miniconda...${NC}"
    bash "$INSTALLER_SCRIPT" -b -p "$MINICONDA_DIR"
    rm "$INSTALLER_SCRIPT"

    # Initialize conda for bash
    echo -e "${BLUE}Initializing conda for bash...${NC}"
    "$MINICONDA_DIR/bin/conda" init bash

    # Source conda to make it available in current shell
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"

    echo ""
    echo -e "${GREEN}Miniconda installed successfully!${NC}"
    echo ""
}

# Function to ensure conda is available
ensure_conda() {
    # Check if conda command is available
    if ! command -v conda &> /dev/null; then
        # Check if conda is installed but not in PATH
        if [ -f "$HOME/miniconda3/bin/conda" ]; then
            echo -e "${YELLOW}Conda found but not in PATH. Activating...${NC}"
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/bin/conda" ]; then
            echo -e "${YELLOW}Anaconda found but not in PATH. Activating...${NC}"
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        else
            # Install Miniconda
            install_miniconda
        fi
    else
        echo -e "${GREEN}Conda detected: $(conda --version)${NC}"
    fi
    echo ""
}

# Ensure conda is available
ensure_conda

# Accept Conda Terms of Service for Anaconda channels
echo -e "${BLUE}Accepting Conda Terms of Service...${NC}"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
echo -e "${GREEN}Conda ToS accepted${NC}"
echo ""

# Configure conda channels (add conda-forge if not already present)
echo -e "${BLUE}Configuring conda channels...${NC}"
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority flexible 2>/dev/null || true
echo -e "${GREEN}Conda channels configured${NC}"
echo ""

# Create and activate environment if not already in one
if [ -z "$CONDA_PREFIX" ]; then
    ENV_NAME="iowarp"
    echo -e "${BLUE}Creating conda environment: $ENV_NAME${NC}"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Using existing environment.${NC}"
    else
        conda create -n "$ENV_NAME" -y python
        echo -e "${GREEN}Environment created${NC}"
    fi

    echo -e "${BLUE}Activating environment: $ENV_NAME${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    echo ""
fi

echo -e "${GREEN}Active conda environment: $CONDA_PREFIX${NC}"
echo ""

# Check if conda-build is installed.
# conda-build registers its subcommand plugin against the *base* environment's
# conda CLI, so installing it into a non-base env leaves `conda build`
# unrecognized. Always install into base.
# conda-build is a *plugin* registered against the base env's conda CLI.
# When we invoke it from a non-base active env (e.g. `iowarp` created
# above), the active env's `conda` doesn't see the plugin. So always
# use base's conda binary explicitly. This was found in CI on a fresh
# Miniconda install: `conda install -n base conda-build` succeeds, but
# the subsequent `conda build` from iowarp's PATH errors out with
# "invalid choice: 'build'".
CONDA_BASE="$(conda info --base)"
CONDA_BIN="${CONDA_BASE}/bin/conda"

if ! "$CONDA_BIN" build --version &> /dev/null; then
    echo -e "${YELLOW}Installing conda-build into base environment...${NC}"
    "$CONDA_BIN" install -n base -y conda-build -c conda-forge
    echo ""
fi

if ! "$CONDA_BIN" build --version &> /dev/null; then
    echo -e "${RED}conda-build is still not available after install attempt.${NC}"
    echo -e "${YELLOW}Try manually:${NC} conda install -n base -y conda-build -c conda-forge"
    exit 1
fi

echo -e "${GREEN}conda-build detected: $("$CONDA_BIN" build --version)${NC}"
echo ""

# Initialize and update git submodules recursively (if in a git repository)
if [ -d ".git" ]; then
    echo -e "${BLUE}>>> Initializing git submodules...${NC}"
    git submodule update --init --recursive 2>/dev/null || {
        echo -e "${YELLOW}Some submodules failed to update (worktrees or optional repos). Continuing...${NC}"
    }
    echo ""
elif [ -d "context-transport-primitives" ] && [ "$(ls -A context-transport-primitives 2>/dev/null)" ]; then
    echo -e "${GREEN}>>> Submodules already present${NC}"
    echo ""
else
    echo -e "${RED}ERROR: Not a git repository and no submodule content found${NC}"
    echo "       Cannot proceed with build - missing dependencies"
    echo ""
    exit 1
fi

# Build the conda package
RECIPE_DIR="$SCRIPT_DIR/installers/conda"

echo -e "${BLUE}>>> Building conda package with conda-build...${NC}"
echo -e "${YELLOW}This may take 10-30 minutes depending on your system${NC}"
echo ""

OUTPUT_DIR="$SCRIPT_DIR/build/conda-output"
mkdir -p "$OUTPUT_DIR"

export IOWARP_PRESET="$PRESET"

# Match the build Python version to the target environment so that
# the built package's python pin is satisfiable at install time.
TARGET_PYTHON="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo -e "${BLUE}Target Python version: $TARGET_PYTHON${NC}"

if "$CONDA_BIN" build "$RECIPE_DIR" \
    --output-folder "$OUTPUT_DIR" \
    -c conda-forge \
    --python="$TARGET_PYTHON" \
    --no-anaconda-upload; then
    BUILD_SUCCESS=true
else
    BUILD_SUCCESS=false
fi

echo ""

if [ "$BUILD_SUCCESS" = true ]; then
    # Find the built package
    PACKAGE_PATH=$(find "$OUTPUT_DIR" -name "iowarp-core-*.tar.bz2" -o -name "iowarp-core-*.conda" | head -1)

    if [ -z "$PACKAGE_PATH" ]; then
        echo -e "${RED}Error: Could not find built package in $OUTPUT_DIR${NC}"
        exit 1
    fi

    echo -e "${GREEN}======================================================================"
    echo -e "Package built successfully!"
    echo -e "======================================================================${NC}"
    echo ""
    echo -e "${BLUE}Package location:${NC}"
    echo "  $PACKAGE_PATH"
    echo ""

    # Install using local output directory as a channel so that conda
    # resolves run dependencies (installing by file path skips dep resolution).
    echo -e "${BLUE}>>> Installing iowarp-core into current environment...${NC}"
    if conda install -y -c "$OUTPUT_DIR" -c conda-forge iowarp-core; then
        echo ""
        echo -e "${GREEN}======================================================================"
        echo -e "IOWarp Core installed successfully!"
        echo -e "======================================================================${NC}"
        echo ""
        echo -e "${BLUE}Installation prefix: $CONDA_PREFIX${NC}"
        echo ""
        echo -e "${BLUE}Verify installation:${NC}"
        echo "  conda list iowarp-core"
        echo ""
        echo -e "${YELLOW}NOTE: To use iowarp-core in a new terminal session, activate the environment:${NC}"
        echo "  conda activate $(basename $CONDA_PREFIX)"
        echo ""
    else
        echo ""
        echo -e "${RED}Installation failed.${NC}"
        echo ""
        echo -e "${YELLOW}You can try installing manually:${NC}"
        echo "  conda install \"$PACKAGE_PATH\""
        echo ""
        exit 1
    fi
else
    echo -e "${RED}======================================================================"
    echo -e "Build failed!"
    echo -e "======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo ""
    echo "1. Check that submodules are initialized:"
    echo "   git submodule update --init --recursive"
    echo ""
    echo "2. Verify conda-forge channel is configured:"
    echo "   conda config --show channels"
    echo ""
    echo "3. Try building with verbose output:"
    echo "   IOWARP_PRESET=$PRESET conda build $RECIPE_DIR -c conda-forge --no-anaconda-upload"
    echo ""
    exit 1
fi
