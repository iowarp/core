#!/bin/bash

# CD into git workspace
cd ${GITHUB_WORKSPACE}

# This script will build and install them via Spack from source
# because Hermes requires a very specific version and configuration options
# for each package.

set -x
set -e
set -o pipefail

# Change this especially when your $HOME doesn't have enough disk space. 
INSTALL_DIR="${HOME}"
SPACK_DIR=${INSTALL_DIR}/spack
SPACK_VERSION=0.22.2

echo "Installing dependencies at ${INSTALL_DIR}"
mkdir -p ${INSTALL_DIR}

# Load Spack
git clone https://github.com/spack/spack ${SPACK_DIR}
cd ${SPACK_DIR}
git checkout v${SPACK_VERSION}
. ${SPACK_DIR}/share/spack/setup-env.sh

# Clone iowarp-install
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
spack repo add clio-core/installers/spack

# This will allow Spack to skip building some packages that are directly
spack external find

# Install hermes_shm (needed for dependencies)
spack install cte-hermes-shm +compress +encrypt +elf +cereal +mpiio +vfd +mochi +boost +nocompile 
