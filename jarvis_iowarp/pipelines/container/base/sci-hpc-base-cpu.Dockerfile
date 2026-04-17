# CPU-only variant of sci-hpc-base for hosts without an NVIDIA GPU.
# Provides: OpenMPI, libhdf5-dev, Python 3, SSH — matches the interface
# expected by the builtin.pyflextrkr_container / warpx / nyx / lammps
# Dockerfile.build files (which assume the base already has libopenmpi-dev).
#
# Build:
#   docker build -t sci-hpc-base-cpu -f sci-hpc-base-cpu.Dockerfile .

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl wget git \
        python3 python3-pip python3-venv \
        openmpi-bin libopenmpi-dev \
        libhdf5-dev \
        openssh-server openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV HDF5_ROOT=/usr

# SSH setup (same shared-key idea as upstream sci-hpc-base — simulation only).
RUN mkdir -p /var/run/sshd /root/.ssh \
    && ssh-keygen -A \
    && ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 \
    && cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n" >> /etc/ssh/ssh_config

EXPOSE 22
CMD ["/bin/bash"]
