#!/bin/bash
# install.sh - Install IOWarp Core using conda
# This script builds and installs IOWarp Core from source using conda-build
# It will automatically install Miniconda if conda is not detected

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================"
echo -e "IOWarp Core - Conda Installation"
echo -e "======================================================================${NC}"
echo ""

# Function to install Miniconda
install_miniconda() {
    echo -e "${YELLOW}Conda not detected. Installing Miniconda...${NC}"
    echo ""

    # Default Miniconda installation directory
    MINICONDA_DIR="$HOME/miniconda3"

    # Detect platform
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
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
    echo -e "${GREEN}✓ Miniconda installed successfully!${NC}"
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
        echo -e "${GREEN}✓ Conda detected: $(conda --version)${NC}"
    fi
    echo ""
}

# Ensure conda is available
ensure_conda

# Configure conda channels (add conda-forge if not already present)
echo -e "${BLUE}Configuring conda channels...${NC}"
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority flexible 2>/dev/null || true
echo -e "${GREEN}✓ Conda channels configured${NC}"
echo ""

# Create and activate environment if not already in one
if [ -z "$CONDA_PREFIX" ]; then
    ENV_NAME="iowarp-build"
    echo -e "${BLUE}Creating conda environment: $ENV_NAME${NC}"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Using existing environment.${NC}"
    else
        conda create -n "$ENV_NAME" -y python=3.11
        echo -e "${GREEN}✓ Environment created${NC}"
    fi

    echo -e "${BLUE}Activating environment: $ENV_NAME${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    echo ""
fi

echo -e "${GREEN}✓ Active conda environment: $CONDA_PREFIX${NC}"
echo ""

# Check if conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo -e "${YELLOW}Installing conda-build...${NC}"
    conda install -y conda-build
    echo ""
fi

# Initialize and update git submodules recursively (if in a git repository)
if [ -d ".git" ]; then
    echo -e "${BLUE}>>> Initializing git submodules...${NC}"
    git submodule update --init --recursive
    echo ""
elif [ -d "external/yaml-cpp" ] && [ "$(ls -A external/yaml-cpp 2>/dev/null)" ]; then
    echo -e "${GREEN}>>> Using bundled submodule content (source distribution)${NC}"
    echo "    external/yaml-cpp: $(ls -1 external/yaml-cpp 2>/dev/null | wc -l) files"
    echo "    external/cereal: $(ls -1 external/cereal 2>/dev/null | wc -l) files"
    echo "    external/Catch2: $(ls -1 external/Catch2 2>/dev/null | wc -l) files"
    echo "    external/nanobind: $(ls -1 external/nanobind 2>/dev/null | wc -l) files"
    echo ""
else
    echo -e "${RED}ERROR: Not a git repository and no bundled submodule content found${NC}"
    echo "       Cannot proceed with build - missing external dependencies"
    echo ""
    exit 1
fi

# Build the conda package
echo -e "${BLUE}>>> Building conda package...${NC}"
echo -e "${YELLOW}This may take 10-30 minutes depending on your system${NC}"
echo ""

RECIPE_DIR="$SCRIPT_DIR/conda"

if ! conda build "$RECIPE_DIR" -c conda-forge; then
    echo ""
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
    echo "   conda build conda/ -c conda-forge --debug"
    echo ""
    echo "4. Check build logs in:"
    echo "   $CONDA_PREFIX/conda-bld/"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}======================================================================"
echo -e "Package built successfully!"
echo -e "======================================================================${NC}"
echo ""

# Get the output package path (optional - for informational purposes only)
# Note: This command may fail in some environments, but it's not critical
echo -e "${BLUE}Locating built package...${NC}"
set +e  # Temporarily disable exit-on-error for this non-critical operation
PACKAGE_PATH=$(conda build "$RECIPE_DIR" --output 2>&1)
PACKAGE_EXIT_CODE=$?
set -e  # Re-enable exit-on-error

if [ $PACKAGE_EXIT_CODE -eq 0 ] && [ -n "$PACKAGE_PATH" ]; then
    echo -e "${BLUE}Package location:${NC}"
    echo "  $PACKAGE_PATH"
    echo ""
else
    echo -e "${YELLOW}Note: Could not determine package path (this is non-critical)${NC}"
    echo ""
fi

# Install the package non-interactively
echo -e "${BLUE}>>> Installing iowarp-core...${NC}"
echo ""

# Ensure conda is configured for non-interactive operation and has conda-forge channel
conda config --set always_yes true 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority flexible 2>/dev/null || true

if conda install --use-local iowarp-core -c conda-forge -y 2>&1; then
    echo ""
    echo -e "${GREEN}======================================================================"
    echo -e "✓ IOWarp Core installed successfully!"
    echo -e "======================================================================${NC}"
    echo ""
    echo -e "${BLUE}Installation prefix: $CONDA_PREFIX${NC}"
    echo ""
    echo -e "${BLUE}Verify installation:${NC}"
    echo "  conda list iowarp-core"
    echo ""
    echo -e "${BLUE}Test the installation:${NC}"
    echo "  chimaera_start_runtime --help"
    echo "  wrp_cte --help"
    echo ""
    echo -e "${BLUE}Python bindings:${NC}"
    echo "  python -c 'import wrp_cte; print(wrp_cte.__version__)'"
    echo ""
    echo -e "${YELLOW}NOTE: To use iowarp-core in a new terminal session, activate the environment:${NC}"
    echo "  conda activate $(basename $CONDA_PREFIX)"
    echo ""
else
    echo ""
    echo -e "${RED}======================================================================"
    echo -e "Installation failed!"
    echo -e "======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}You can try installing manually:${NC}"
    echo "  conda config --add channels conda-forge"
    echo "  conda install --use-local iowarp-core -c conda-forge"
    echo ""
    echo -e "${YELLOW}Or check that conda-forge channel is available:${NC}"
    echo "  conda config --show channels"
    echo ""
    exit 1
fi
