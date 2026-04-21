#!/bin/bash
# IOWarp runtime build script — installs all build deps, clones the IOWarp
# source at ##GIT_BRANCH##, and configures/builds via the ##CMAKE_PRESET##
# CMakePresets.json preset. Runs inside the jarvis pipeline build container
# (started from container_base, typically ubuntu:24.04).
set -e

export DEBIAN_FRONTEND=noninteractive

# --- System build deps ------------------------------------------------------
apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    cmake ninja-build pkg-config g++ make \
    python3-dev python3-pip python3-venv \
    libelf-dev libaio-dev liburing-dev \
    libfuse3-dev fuse3 \
    openmpi-bin libopenmpi-dev mpi-default-dev \
    libboost-all-dev catch2 libcurl4-openssl-dev libssl-dev \
    nlohmann-json3-dev \
    zlib1g-dev libbz2-dev liblzo2-dev libzstd-dev liblz4-dev liblzma-dev \
    libbrotli-dev libsnappy-dev libblosc2-dev libzfp-dev \
 && rm -rf /var/lib/apt/lists/*

# --- yaml-cpp 0.8.0 ---------------------------------------------------------
cd /tmp
curl -sL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz | tar xz
cmake -S yaml-cpp-0.8.0 -B yaml-cpp-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON \
   -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build yaml-cpp-build -j"$(nproc)"
cmake --install yaml-cpp-build
ldconfig
rm -rf /tmp/yaml-cpp-*

# --- cereal 1.3.2 (header-only) ---------------------------------------------
cd /tmp
curl -sL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz
cmake -S cereal-1.3.2 -B cereal-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DSKIP_PERFORMANCE_COMPARISON=ON \
   -DBUILD_TESTS=OFF -DBUILD_SANDBOX=OFF -DBUILD_DOC=OFF
cmake --install cereal-build
rm -rf /tmp/cereal-*

# --- msgpack-c 6.1.0 --------------------------------------------------------
cd /tmp
git clone --depth 1 --branch c-6.1.0 https://github.com/msgpack/msgpack-c.git
cmake -S msgpack-c -B msgpack-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
   -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF
cmake --build msgpack-build -j"$(nproc)"
cmake --install msgpack-build
rm -rf /tmp/msgpack-c /tmp/msgpack-build

# --- libsodium 1.0.20 -------------------------------------------------------
cd /tmp
curl -sL https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz | tar xz
cd libsodium-1.0.20
./configure --prefix=/usr/local --with-pic
make -j"$(nproc)"
make install
ldconfig
rm -rf /tmp/libsodium-*

# --- zeromq 4.3.5 -----------------------------------------------------------
cd /tmp
curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz | tar xz
cmake -S zeromq-4.3.5 -B zmq-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED=ON -DBUILD_STATIC=ON \
   -DBUILD_TESTS=OFF -DWITH_LIBSODIUM=ON -DWITH_DOCS=OFF \
   -DCMAKE_PREFIX_PATH=/usr/local
cmake --build zmq-build -j"$(nproc)"
cmake --install zmq-build
ldconfig
rm -rf /tmp/zeromq-* /tmp/zmq-build

# --- cppzmq 4.10.0 (header-only) --------------------------------------------
cd /tmp
curl -sL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz | tar xz
cmake -S cppzmq-4.10.0 -B cppzmq-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr/local \
   -DCPPZMQ_BUILD_TESTS=OFF
cmake --install cppzmq-build
rm -rf /tmp/cppzmq-*

# --- HDF5 2.1.1 -------------------------------------------------------------
# Ubuntu 24.04 apt only has 1.10; iowarp_hdf5_vol needs 2.x VOL API.
cd /tmp
wget -q https://github.com/HDFGroup/hdf5/releases/download/2.1.1/hdf5-2.1.1.tar.gz
tar xzf hdf5-2.1.1.tar.gz
cd hdf5-2.1.1
cmake -B build -S . \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
   -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_TOOLS=ON \
   -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
   -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_FORTRAN=OFF -DBUILD_TESTING=OFF
cmake --build build -j"$(nproc)"
cmake --install build
cd /tmp
rm -rf hdf5-2.1.1*

# --- ADIOS2 v2.11.0 ---------------------------------------------------------
cd /tmp
git clone --depth 1 --branch v2.11.0 https://github.com/ornladios/ADIOS2.git
cmake -S ADIOS2 -B adios2-build \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
   -DADIOS2_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
   -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON -DADIOS2_USE_ZeroMQ=ON \
   -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF -DADIOS2_USE_Fortran=OFF \
   -DCMAKE_CXX_STANDARD=17
make -C adios2-build -j"$(nproc)"
make -C adios2-build install
ldconfig
rm -rf /tmp/ADIOS2 /tmp/adios2-build

# --- Lossy compression: FPZIP, SZ3, std_compat, LibPressio ------------------
cd /tmp
git clone https://github.com/LLNL/fpzip.git
cmake -S fpzip -B fpzip-build \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
   -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DBUILD_UTILITIES=OFF
make -C fpzip-build -j"$(nproc)"
make -C fpzip-build install
ldconfig
rm -rf /tmp/fpzip*

cd /tmp
git clone https://github.com/szcompressor/SZ3.git
cmake -S SZ3 -B sz3-build \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
   -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF
make -C sz3-build -j"$(nproc)"
make -C sz3-build install
ldconfig
rm -rf /tmp/SZ3 /tmp/sz3-build

cd /tmp
git clone https://github.com/robertu94/std_compat.git
cmake -S std_compat -B std_compat-build \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTING=OFF
make -C std_compat-build -j"$(nproc)"
make -C std_compat-build install
ldconfig
rm -rf /tmp/std_compat*

cd /tmp
git clone https://github.com/robertu94/libpressio.git
cmake -S libpressio -B libpressio-build \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
   -DLIBPRESSIO_HAS_ZFP=ON -DLIBPRESSIO_HAS_SZ3=ON -DLIBPRESSIO_HAS_FPZIP=ON \
   -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF
make -C libpressio-build -j"$(nproc)"
make -C libpressio-build install
ldconfig
rm -rf /tmp/libpressio*

# --- Clone and build IOWarp -------------------------------------------------
# Submodules are NOT recursed at clone time: external/jarvis-cd pulls its own
# `awesome-scienctific-applications` submodule via an SSH URL that fails in
# containers, and the core build does not need it.
git clone --depth 1 --branch ##GIT_BRANCH## \
    https://github.com/iowarp/clio-core.git /opt/iowarp

cd /opt/iowarp
cmake --preset ##CMAKE_PRESET## -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j"$(nproc)"
cmake --install build
ldconfig

# --- Seed default chimaera config ------------------------------------------
mkdir -p /root/.chimaera
cp /opt/iowarp/context-runtime/config/chimaera_default.yaml \
   /root/.chimaera/chimaera.yaml
