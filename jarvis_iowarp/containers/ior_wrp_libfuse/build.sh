#!/bin/bash
# Combined builder-stage script for the ior_wrp_libfuse container.
# Produces:
#   - IOR (from hpc/ior 3.3.0) + Darshan runtime (3.4.4) under /opt
#   - IOWarp / chimaera runtime + dependencies under /usr/local
#   - libfuse3 build environment installed (runtime libfuse3 lives in
#     the deploy stage so the user-mode FUSE mount path works there too)
#
# Mirrors:
#   builtin/builtin/ior/build.sh
#   clio-core/jarvis_iowarp/jarvis_iowarp/wrp_runtime/build.sh
# Substitutions:
#   GIT_BRANCH    branch of iowarp/clio-core to clone (default: main)
#   CMAKE_PRESET  CMakePresets.json preset for the IOWarp build
#                 (default: release-adapter)
set -e
export DEBIAN_FRONTEND=noninteractive

GIT_BRANCH="${GIT_BRANCH:-main}"
CMAKE_PRESET="${CMAKE_PRESET:-release-adapter}"

# ---------------------------------------------------------------------------
# System build deps — union of IOR and wrp_runtime build deps, plus libfuse3
# ---------------------------------------------------------------------------
apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    build-essential autoconf automake libtool \
    cmake ninja-build pkg-config g++ make \
    python3-dev python3-pip python3-venv \
    libelf-dev libaio-dev liburing-dev \
    libfuse3-dev fuse3 \
    openmpi-bin libopenmpi-dev mpi-default-dev \
    openssh-server openssh-client \
    libboost-all-dev catch2 libcurl4-openssl-dev libssl-dev \
    nlohmann-json3-dev \
    zlib1g-dev libbz2-dev liblzo2-dev libzstd-dev liblz4-dev liblzma-dev \
    libbrotli-dev libsnappy-dev libblosc2-dev libzfp-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# SSH setup so MPI multi-container deployments can ssh between replicas of
# this image. Lifted verbatim from builtin/builtin/ior/build.sh.
# ---------------------------------------------------------------------------
mkdir -p /var/run/sshd /root/.ssh \
    && ssh-keygen -A \
    && ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 \
    && cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n" >> /etc/ssh/ssh_config

# ---------------------------------------------------------------------------
# yaml-cpp 0.8.0
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# cereal 1.3.2 (header-only)
# ---------------------------------------------------------------------------
cd /tmp
curl -sL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz
cmake -S cereal-1.3.2 -B cereal-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DSKIP_PERFORMANCE_COMPARISON=ON \
   -DBUILD_TESTS=OFF -DBUILD_SANDBOX=OFF -DBUILD_DOC=OFF
cmake --install cereal-build
rm -rf /tmp/cereal-*

# ---------------------------------------------------------------------------
# msgpack-c 6.1.0
# ---------------------------------------------------------------------------
cd /tmp
git clone --depth 1 --branch c-6.1.0 https://github.com/msgpack/msgpack-c.git
cmake -S msgpack-c -B msgpack-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
   -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF
cmake --build msgpack-build -j"$(nproc)"
cmake --install msgpack-build
rm -rf /tmp/msgpack-c /tmp/msgpack-build

# ---------------------------------------------------------------------------
# libsodium 1.0.20
# ---------------------------------------------------------------------------
cd /tmp
curl -sL https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz | tar xz
cd libsodium-1.0.20
./configure --prefix=/usr/local --with-pic
make -j"$(nproc)"
make install
ldconfig
rm -rf /tmp/libsodium-*

# ---------------------------------------------------------------------------
# zeromq 4.3.5
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# cppzmq 4.10.0 (header-only)
# ---------------------------------------------------------------------------
cd /tmp
curl -sL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz | tar xz
cmake -S cppzmq-4.10.0 -B cppzmq-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr/local \
   -DCPPZMQ_BUILD_TESTS=OFF
cmake --install cppzmq-build
rm -rf /tmp/cppzmq-*

# ---------------------------------------------------------------------------
# IOR 3.3.0 + Darshan 3.4.4 (mirrors builtin/builtin/ior/build.sh)
# Built without HDF5 — this image bundles only IOR + wrp_runtime + libfuse;
# layered HDF5/ADIOS2 stacks belong in the wider pipeline (see
# wrp_runtime/build.sh header).
# ---------------------------------------------------------------------------
mkdir -p /opt/ior && curl -sL https://github.com/hpc/ior/releases/download/3.3.0/ior-3.3.0.tar.gz \
    | tar xz --strip-components=1 -C /opt/ior

cd /opt/ior \
    && ./configure --prefix=/opt/ior/install \
        LDFLAGS="-L/usr/local/lib" CPPFLAGS="-I/usr/local/include" \
    && make -j"$(nproc)" \
    && make install

mkdir -p /opt/darshan && curl -sL https://github.com/darshan-hpc/darshan/archive/refs/tags/darshan-3.4.4.tar.gz \
    | tar xz --strip-components=1 -C /opt/darshan

cd /opt/darshan/darshan-runtime \
    && autoreconf -ivf \
    && ./configure --prefix=/opt/darshan/install \
        --with-log-path-by-env=DARSHAN_LOG_DIR \
        --with-jobid-env=PBS_JOBID \
        CC=mpicc \
    && make -j"$(nproc)" install

# ---------------------------------------------------------------------------
# IOWarp / chimaera runtime
# Submodules are NOT recursed at clone time — see header note in
# wrp_runtime/build.sh (external/jarvis-cd uses an SSH submodule URL that
# fails inside containers, and the core build does not need it).
# ---------------------------------------------------------------------------
git clone --depth 1 --branch "${GIT_BRANCH}" \
    https://github.com/iowarp/clio-core.git /opt/iowarp

cd /opt/iowarp
cmake --preset "${CMAKE_PRESET}" -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j"$(nproc)"
cmake --install build
ldconfig

# ---------------------------------------------------------------------------
# Seed default chimaera config so `chimaera runtime start` works without
# additional setup.
# ---------------------------------------------------------------------------
mkdir -p /root/.chimaera
cp /opt/iowarp/context-runtime/config/chimaera_default.yaml \
   /root/.chimaera/chimaera.yaml
