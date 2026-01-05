#!/bin/bash

# Script to build and install the iowarp-core conda package locally
# This script builds IOWarp Core from your current local source code using rattler-build
#
# Usage:
#   ./conda-local.sh                    # Build with default (release) variant
#   ./conda-local.sh release            # Build with release preset
#   ./conda-local.sh cuda               # Build with CUDA preset
#   ./conda-local.sh custom             # Build with custom variant
#   ./conda-local.sh /path/to/variant.yaml  # Build with custom variant file

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RECIPE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IOWarp Core - Local Conda Package Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_PREFIX" ]; then
    echo -e "${RED}Error: No conda environment detected${NC}"
    echo "Please activate a conda environment first:"
    echo "  conda create -n iowarp-build"
    echo "  conda activate iowarp-build"
    exit 1
fi

echo -e "${GREEN}Conda environment detected: $CONDA_PREFIX${NC}"
echo ""

# Check if rattler-build is installed
if ! command -v rattler-build &> /dev/null; then
    echo -e "${YELLOW}rattler-build is not installed. Installing...${NC}"
    conda install -y rattler-build -c conda-forge
    echo ""
fi

# Determine variant file
VARIANT_ARG="$1"
if [ -z "$VARIANT_ARG" ]; then
    VARIANT_FILE="$RECIPE_DIR/variants/release.yaml"
    echo -e "${BLUE}Using default variant: release${NC}"
elif [ -f "$VARIANT_ARG" ]; then
    VARIANT_FILE="$VARIANT_ARG"
    echo -e "${BLUE}Using variant file: $VARIANT_FILE${NC}"
elif [ -f "$RECIPE_DIR/variants/${VARIANT_ARG}.yaml" ]; then
    VARIANT_FILE="$RECIPE_DIR/variants/${VARIANT_ARG}.yaml"
    echo -e "${BLUE}Using variant: $VARIANT_ARG${NC}"
else
    echo -e "${RED}Error: Variant not found: $VARIANT_ARG${NC}"
    echo ""
    echo "Available variants:"
    for f in "$RECIPE_DIR/variants"/*.yaml; do
        basename "$f" .yaml
    done
    echo ""
    echo "Usage:"
    echo "  $0                    # Use default (release) variant"
    echo "  $0 <variant-name>     # Use named variant (e.g., cuda, mpi)"
    echo "  $0 /path/to/file.yaml # Use custom variant file"
    exit 1
fi

# Navigate to repository root
cd "$REPO_ROOT"

# Check if submodules are initialized
echo -e "${BLUE}Checking git submodules...${NC}"
if [ ! -f "context-transport-primitives/CMakeLists.txt" ]; then
    echo -e "${YELLOW}Submodules not initialized. Initializing now...${NC}"
    git submodule update --init --recursive
    echo -e "${GREEN}Submodules initialized successfully${NC}"
else
    echo -e "${GREEN}Submodules are initialized${NC}"
fi
echo ""

# Show build configuration
echo -e "${BLUE}Build Configuration:${NC}"
echo "  Recipe directory:  $RECIPE_DIR"
echo "  Source directory:  $REPO_ROOT"
echo "  Variant file:      $VARIANT_FILE"
echo "  Conda environment: $CONDA_PREFIX"
echo ""

# Show variant contents
echo -e "${BLUE}Variant settings:${NC}"
cat "$VARIANT_FILE"
echo ""

# Build the package with rattler-build
echo -e "${BLUE}Building conda package...${NC}"
echo -e "${YELLOW}This may take 10-30 minutes depending on your system${NC}"
echo ""

OUTPUT_DIR="$REPO_ROOT/build/conda-output"
mkdir -p "$OUTPUT_DIR"

if rattler-build build \
    --recipe "$RECIPE_DIR" \
    --variant-config "$VARIANT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    -c conda-forge; then
    BUILD_SUCCESS=true
else
    BUILD_SUCCESS=false
fi

echo ""

if [ "$BUILD_SUCCESS" = true ]; then
    # Find the built package
    PACKAGE_PATH=$(find "$OUTPUT_DIR" -name "iowarp-core-*.conda" -o -name "iowarp-core-*.tar.bz2" | head -1)

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Package built successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Package location:${NC}"
    echo "  $PACKAGE_PATH"
    echo ""

    # Install directly into current environment
    echo -e "${BLUE}Installing iowarp-core into current environment...${NC}"
    if conda install "$PACKAGE_PATH" -y; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Installation successful!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${BLUE}Verify installation:${NC}"
        echo "  conda list iowarp-core"
        echo ""
        echo -e "${BLUE}Remove package:${NC}"
        echo "  conda remove iowarp-core"
        echo ""
    else
        echo ""
        echo -e "${RED}Installation failed.${NC}"
        echo ""
        echo -e "${YELLOW}You can try installing manually:${NC}"
        echo "  conda install \"$PACKAGE_PATH\""
        echo ""
    fi

else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
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
    echo "   rattler-build build --recipe $RECIPE_DIR --variant-config $VARIANT_FILE --verbose"
    echo ""
    echo "4. Check available variants:"
    echo "   ls $RECIPE_DIR/variants/"
    echo ""
    echo "For more help, see: $RECIPE_DIR/README.md"
    echo ""
    exit 1
fi
