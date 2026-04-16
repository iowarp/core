"""
IOWarp Runtime Service Package

Manages the Chimaera runtime deployment. Supports both bare-metal (default)
and container deployment modes via deploy_mode configuration.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Kill, GdbServer
from jarvis_cd.util import SizeType
from jarvis_cd.util.logger import Color
import os
import subprocess
import time
import yaml

# Shared snippet: install all IOWarp build deps + build IOWarp from source
# on a bare ubuntu:24.04. Used by wrp_runtime and adios2_gray_scott build phases.
IOWARP_BUILD_DEPS = r"""
ARG DEBIAN_FRONTEND=noninteractive

# Build tools + system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# yaml-cpp 0.8.0
RUN cd /tmp \
    && curl -sL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz | tar xz \
    && cmake -S yaml-cpp-0.8.0 -B yaml-cpp-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON \
       -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF \
    && cmake --build yaml-cpp-build -j$(nproc) && cmake --install yaml-cpp-build \
    && ldconfig && rm -rf /tmp/yaml-cpp-*

# cereal 1.3.2 (header-only)
RUN cd /tmp \
    && curl -sL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz \
    && cmake -S cereal-1.3.2 -B cereal-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DSKIP_PERFORMANCE_COMPARISON=ON \
       -DBUILD_TESTS=OFF -DBUILD_SANDBOX=OFF -DBUILD_DOC=OFF \
    && cmake --install cereal-build && rm -rf /tmp/cereal-*

# msgpack-c 6.1.0
RUN cd /tmp \
    && git clone --depth 1 --branch c-6.1.0 https://github.com/msgpack/msgpack-c.git \
    && cmake -S msgpack-c -B msgpack-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF \
    && cmake --build msgpack-build -j$(nproc) && cmake --install msgpack-build \
    && rm -rf /tmp/msgpack-c /tmp/msgpack-build

# libsodium 1.0.20
RUN cd /tmp \
    && curl -sL https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz | tar xz \
    && cd libsodium-1.0.20 && ./configure --prefix=/usr/local --with-pic \
    && make -j$(nproc) && make install && ldconfig && rm -rf /tmp/libsodium-*

# zeromq 4.3.5
RUN cd /tmp \
    && curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz | tar xz \
    && cmake -S zeromq-4.3.5 -B zmq-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED=ON -DBUILD_STATIC=ON \
       -DBUILD_TESTS=OFF -DWITH_LIBSODIUM=ON -DWITH_DOCS=OFF \
       -DCMAKE_PREFIX_PATH=/usr/local \
    && cmake --build zmq-build -j$(nproc) && cmake --install zmq-build \
    && ldconfig && rm -rf /tmp/zeromq-* /tmp/zmq-build

# cppzmq 4.10.0 (header-only)
RUN cd /tmp \
    && curl -sL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz | tar xz \
    && cmake -S cppzmq-4.10.0 -B cppzmq-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr/local \
       -DCPPZMQ_BUILD_TESTS=OFF \
    && cmake --install cppzmq-build && rm -rf /tmp/cppzmq-*

# HDF5 2.1.1 (Ubuntu 24.04 apt only has 1.10)
RUN cd /tmp \
    && wget -q https://github.com/HDFGroup/hdf5/releases/download/2.1.1/hdf5-2.1.1.tar.gz \
    && tar xzf hdf5-2.1.1.tar.gz && cd hdf5-2.1.1 \
    && cmake -B build -S . \
       -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
       -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_TOOLS=ON \
       -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
       -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_FORTRAN=OFF -DBUILD_TESTING=OFF \
    && cmake --build build -j$(nproc) && cmake --install build \
    && cd /tmp && rm -rf hdf5-2.1.1*

# ADIOS2 v2.11.0
RUN cd /tmp \
    && git clone --depth 1 --branch v2.11.0 https://github.com/ornladios/ADIOS2.git \
    && cmake -S ADIOS2 -B adios2-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DADIOS2_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
       -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON -DADIOS2_USE_ZeroMQ=ON \
       -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF -DADIOS2_USE_Fortran=OFF \
       -DCMAKE_CXX_STANDARD=17 \
    && make -C adios2-build -j$(nproc) && make -C adios2-build install \
    && ldconfig && rm -rf /tmp/ADIOS2 /tmp/adios2-build

# Lossy compression: FPZIP, SZ3, std_compat, LibPressio
RUN cd /tmp \
    && git clone https://github.com/LLNL/fpzip.git \
    && cmake -S fpzip -B fpzip-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DBUILD_UTILITIES=OFF \
    && make -C fpzip-build -j$(nproc) && make -C fpzip-build install \
    && ldconfig && rm -rf /tmp/fpzip*

RUN cd /tmp \
    && git clone https://github.com/szcompressor/SZ3.git \
    && cmake -S SZ3 -B sz3-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
    && make -C sz3-build -j$(nproc) && make -C sz3-build install \
    && ldconfig && rm -rf /tmp/SZ3 /tmp/sz3-build

RUN cd /tmp \
    && git clone https://github.com/robertu94/std_compat.git \
    && cmake -S std_compat -B std_compat-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTING=OFF \
    && make -C std_compat-build -j$(nproc) && make -C std_compat-build install \
    && ldconfig && rm -rf /tmp/std_compat*

RUN cd /tmp \
    && git clone https://github.com/robertu94/libpressio.git \
    && cmake -S libpressio -B libpressio-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DLIBPRESSIO_HAS_ZFP=ON -DLIBPRESSIO_HAS_SZ3=ON -DLIBPRESSIO_HAS_FPZIP=ON \
       -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
    && make -C libpressio-build -j$(nproc) && make -C libpressio-build install \
    && ldconfig && rm -rf /tmp/libpressio*
"""

def iowarp_clone_and_build(branch: str, preset: str) -> str:
    """
    Render the Dockerfile snippet that clones IOWarp at the given branch and
    builds it via the given CMakePresets.json preset.

    :param branch: Git branch of iowarp/clio-core to check out (e.g., 'main',
                   'dev', 'transparent-compress').
    :param preset: Name of a preset declared in CMakePresets.json
                   (e.g., 'build-cpu-release', 'release-adapter').
    """
    return rf"""
# Clone and build IOWarp ({branch} @ preset: {preset})
RUN git clone --recurse-submodules --depth 1 --branch {branch} \
    https://github.com/iowarp/clio-core.git /opt/iowarp

WORKDIR /opt/iowarp
RUN cmake --preset {preset} -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    ldconfig

# Seed default chimaera config
RUN mkdir -p /root/.chimaera && \
    cp /opt/iowarp/context-runtime/config/chimaera_default.yaml \
       /root/.chimaera/chimaera.yaml
"""


# Default snippet preserved for backward compatibility with older imports
# (e.g., jarvis_iowarp.adios2_gray_scott). Uses the 'main' branch and the
# legacy 'build-cpu-release' preset.
IOWARP_CLONE_AND_BUILD = iowarp_clone_and_build('main', 'build-cpu-release')

IOWARP_DEPLOY_BASE = r"""
ARG DEBIAN_FRONTEND=noninteractive

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libelf1 \
    liburing2 \
    openmpi-bin \
    libopenmpi3t64 \
    openssh-server openssh-client \
    libblosc2-2t64 \
    libzstd1 liblz4-1 liblzma5 libbz2-1.0 libbrotli1 libsnappy1v5 \
    libzfp1 \
    libfuse3-3 fuse3 \
    && rm -rf /var/lib/apt/lists/*

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Copy IOWarp binaries and libraries from build container
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share /usr/local/share
COPY --from=builder /root/.chimaera /root/.chimaera

# Set up library paths and update cache
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu
ENV PATH=/usr/local/bin:${PATH}
RUN ldconfig
RUN mkdir -p /run/sshd

# Install Python and jarvis-cd
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages pyyaml pandas podman-compose
RUN cd /tmp \
    && git clone --depth 1 --branch dev https://github.com/grc-iit/jarvis-cd.git \
    && cd jarvis-cd \
    && git config submodule.awesome-scienctific-applications.update none \
    && pip3 install --break-system-packages . \
    && cd / && rm -rf /tmp/jarvis-cd
"""


class WrpRuntime(Service):
    """
    IOWarp Runtime Service supporting default and container deployment.
    """

    def _init(self):
        self.config_file = f'{self.shared_dir}/chimaera_config.yaml'

    def _configure_menu(self):
        return [
            {
                'name': 'num_threads',
                'msg': 'Number of worker threads for task execution',
                'type': int,
                'default': 4
            },
            {
                'name': 'process_reaper_workers',
                'msg': 'Number of process reaper worker threads',
                'type': int,
                'default': 1
            },
            {
                'name': 'main_segment_size',
                'msg': 'Main memory segment size (e.g., 1G, 512M, or "auto")',
                'type': str,
                'default': 'auto'
            },
            {
                'name': 'client_data_segment_size',
                'msg': 'Client data segment size (e.g., 512M, 256M)',
                'type': str,
                'default': '512M'
            },
            {
                'name': 'port',
                'msg': 'ZeroMQ port for networking',
                'type': int,
                'default': 9413
            },
            {
                'name': 'ipc_mode',
                'msg': 'IPC transport mode for client-server communication',
                'type': str,
                'choices': ['tcp', 'ipc', 'shm'],
                'default': 'tcp'
            },
            {
                'name': 'log_level',
                'msg': 'Logging level',
                'type': str,
                'choices': ['debug', 'info', 'warning', 'error'],
                'default': 'info'
            },
            {
                'name': 'queue_depth',
                'msg': 'Task queue depth per worker',
                'type': int,
                'default': 1024
            },
            {
                'name': 'local_sched',
                'msg': 'Local task scheduler',
                'type': str,
                'default': 'default'
            },
            {
                'name': 'heartbeat_interval',
                'msg': 'Runtime heartbeat interval (milliseconds)',
                'type': int,
                'default': 1000
            },
            {
                'name': 'first_busy_wait',
                'msg': 'Busy wait duration before sleeping (microseconds)',
                'type': int,
                'default': 50
            },
            {
                'name': 'max_sleep',
                'msg': 'Maximum sleep duration cap (microseconds)',
                'type': int,
                'default': 50000
            },
            {
                'name': 'git_branch',
                'msg': 'Branch of iowarp/clio-core to clone inside the container build',
                'type': str,
                'default': 'main'
            },
            {
                'name': 'cmake_preset',
                'msg': 'CMakePresets.json preset used to configure the IOWarp build',
                'type': str,
                'default': 'release-adapter'
            },
        ]

    # ------------------------------------------------------------------
    # Container Dockerfile generators
    # ------------------------------------------------------------------

    def _build_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        branch = self.config.get('git_branch', 'main')
        preset = self.config.get('cmake_preset', 'release-adapter')
        content = f"""FROM ubuntu:24.04
{IOWARP_BUILD_DEPS}
{iowarp_clone_and_build(branch, preset)}
"""
        return content, preset

    def _build_deploy_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        suffix = getattr(self, '_build_suffix', '')
        content = f"""FROM {self.build_image_name()} AS builder
FROM ubuntu:24.04
{IOWARP_DEPLOY_BASE}
CMD ["/bin/bash"]
"""
        return content, suffix

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

        self.config_file = f'{self.shared_dir}/chimaera_config.yaml'

        self.setenv('CHI_SERVER_CONF', self.config_file)
        self.setenv('HSHM_LOG_LEVEL', self.config['log_level'])
        self.setenv('CHI_IPC_MODE', self.config['ipc_mode'].upper())

        self._generate_config()

        self.log(f"IOWarp runtime configured")
        self.log(f"  Config file: {self.config_file}")

    def _generate_config(self):
        if self.config['main_segment_size'] == 'auto':
            main_size = 'auto'
        else:
            main_size = SizeType(self.config['main_segment_size']).bytes
        client_size = SizeType(self.config['client_data_segment_size']).bytes

        config_dict = {
            'memory': {
                'main_segment_size': main_size,
                'client_data_segment_size': client_size
            },
            'networking': {
                'port': self.config['port'],
                'hostfile': self.jarvis.hostfile.path if self.jarvis.hostfile.path else ''
            },
            'logging': {
                'level': self.config['log_level'],
                'file': f"{self.shared_dir}/chimaera.log"
            },
            'runtime': {
                'num_threads': self.config['num_threads'],
                'process_reaper_threads': self.config['process_reaper_workers'],
                'queue_depth': self.config['queue_depth'],
                'local_sched': self.config['local_sched'],
                'heartbeat_interval': self.config['heartbeat_interval'],
                'first_busy_wait': self.config['first_busy_wait'],
                'max_sleep': self.config['max_sleep']
            }
        }

        with open(self.config_file, 'w') as f:
            f.write('# Chimaera Runtime Configuration\n\n')
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.log("Starting IOWarp runtime")
        cmd = 'chimaera runtime start'

        if self.config.get('do_dbg', False):
            GdbServer(cmd, self.config['dbg_port'], PsshExecInfo(
                env=self.env,
                hostfile=self.jarvis.hostfile,
                exec_async=True,
                container=self._container_engine,
                container_image=self.deploy_image_name,
                private_dir=self.private_dir,
                bind_mounts=self.container_mounts,
            )).run()
        else:
            Exec(cmd, PsshExecInfo(
                env=self.env,
                hostfile=self.jarvis.hostfile,
                exec_async=True,
                container=self._container_engine,
                container_image=self.deploy_image_name,
                private_dir=self.private_dir,
                bind_mounts=self.container_mounts,
            )).run()

        self.sleep()

        port = self.config['port']
        host = self.jarvis.hostfile.hosts[0] if self.jarvis.hostfile.hosts else '127.0.0.1'
        self.log(f'Waiting for runtime on {host}:{port}', color=Color.YELLOW)
        for i in range(30):
            try:
                ret = subprocess.run(
                    ['bash', '-c', f'echo > /dev/tcp/{host}/{port}'],
                    capture_output=True, timeout=2)
                if ret.returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1)
        else:
            self.log(f'WARNING: Runtime did not respond on {host}:{port} after 30s',
                     color=Color.RED)

        self.log("IOWarp runtime started")

    def stop(self):
        self.log("Stopping IOWarp runtime")

        Exec('chimaera runtime stop', PsshExecInfo(
            env=self.env,
            hostfile=self.jarvis.hostfile,
            container=self._container_engine,
            container_image=self.deploy_image_name,
            private_dir=self.private_dir,
            bind_mounts=self.container_mounts,
        )).run()

        Kill('chimaera',
             PsshExecInfo(env=self.env,
                          hostfile=self.jarvis.hostfile),
             partial=False).run()

        port = self.config['port']
        host = self.jarvis.hostfile.hosts[0] if self.jarvis.hostfile.hosts else '127.0.0.1'
        for i in range(10):
            try:
                ret = subprocess.run(
                    ['bash', '-c', f'echo > /dev/tcp/{host}/{port}'],
                    capture_output=True, timeout=2)
                if ret.returncode != 0:
                    break
            except subprocess.TimeoutExpired:
                break
            time.sleep(1)
        time.sleep(1)

        self.log("IOWarp runtime stopped")

    def kill(self):
        self.log("Forcibly killing IOWarp runtime")
        Kill('chimaera', PsshExecInfo(
            hostfile=self.jarvis.hostfile
        )).run()

    def clean(self):
        self.log("Cleaning IOWarp runtime data")

        if self.config_file and os.path.exists(self.config_file):
            os.remove(self.config_file)

        log_file = f'{self.shared_dir}/chimaera.log'
        if os.path.exists(log_file):
            os.remove(log_file)

        Exec('rm -f /dev/shm/chi_*', PsshExecInfo(
            hostfile=self.jarvis.hostfile
        )).run()
