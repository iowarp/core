# Combined IOR + wrp_runtime + libfuse container.
# Standalone two-stage build — no Jarvis pipeline required.
#
# Build:
#   docker build -t ior_wrp_libfuse \
#       --build-arg GIT_BRANCH=main \
#       --build-arg CMAKE_PRESET=release-adapter \
#       containers/ior_wrp_libfuse
#
# The deploy stage mirrors clio-core/jarvis_iowarp/jarvis_iowarp/wrp_runtime/
# Dockerfile.deploy plus IOR's runtime artifacts and libfuse3 user-space
# libraries.

ARG BASE_IMAGE=ubuntu:24.04
ARG DEPLOY_BASE=ubuntu:24.04

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
ARG GIT_BRANCH=main
ARG CMAKE_PRESET=release-adapter
ENV GIT_BRANCH=${GIT_BRANCH} \
    CMAKE_PRESET=${CMAKE_PRESET} \
    DEBIAN_FRONTEND=noninteractive

COPY build.sh /tmp/build.sh
RUN chmod +x /tmp/build.sh && bash /tmp/build.sh

# ---------------------------------------------------------------------------
# Stage 2: deploy
# Runtime deps = union of IOR's deploy + wrp_runtime's deploy, with libfuse3
# user-space libraries (libfuse3-3, fuse3) for FUSE mount support.
# ---------------------------------------------------------------------------
FROM ${DEPLOY_BASE}
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libelf1 \
        liburing2 \
        openmpi-bin libopenmpi3t64 \
        openssh-server openssh-client \
        libblosc2-2t64 \
        libzstd1 liblz4-1 liblzma5 libbz2-1.0 libbrotli1 libsnappy1v5 \
        libzfp1 \
        libfuse3-3 fuse3 \
        python3 python3-pip python3-venv git \
        wget curl \
    && rm -rf /var/lib/apt/lists/*

ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# IOWarp / chimaera runtime artifacts
COPY --from=builder /usr/local/lib     /usr/local/lib
COPY --from=builder /usr/local/bin     /usr/local/bin
COPY --from=builder /usr/local/share   /usr/local/share
COPY --from=builder /usr/local/include /usr/local/include
COPY --from=builder /root/.chimaera    /root/.chimaera

# IOR + Darshan artifacts (kept under /opt to match the IOR package layout)
COPY --from=builder /opt/ior/install/bin/ior          /usr/local/bin/ior
COPY --from=builder /opt/darshan/install/lib/libdarshan.so /opt/darshan/lib/libdarshan.so

# SSH host + root keys generated in the builder so all replicas share an
# identity (matches builtin/builtin/ior/Dockerfile.deploy).
COPY --from=builder /etc/ssh/ssh_host_*    /etc/ssh/
COPY --from=builder /root/.ssh             /root/.ssh

ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/darshan/lib:/usr/lib/x86_64-linux-gnu \
    PATH=/usr/local/bin:${PATH}

RUN ldconfig && mkdir -p /run/sshd /var/run/sshd \
    && printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n" \
        > /root/.ssh/config \
    && chmod 600 /root/.ssh/config

# Install jarvis-cd inside the deploy image so pipeline orchestration
# (SSH/MPI from a host-side jarvis) reaches into the container.
RUN pip3 install --break-system-packages pyyaml pandas podman-compose \
    && cd /tmp \
    && git clone --depth 1 --branch dev https://github.com/grc-iit/jarvis-cd.git \
    && cd jarvis-cd \
    && git config submodule.awesome-scienctific-applications.update none \
    && pip3 install --break-system-packages . \
    && cd / && rm -rf /tmp/jarvis-cd

EXPOSE 22
CMD ["/bin/bash"]
