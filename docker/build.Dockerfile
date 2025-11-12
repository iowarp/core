# Dockerfile for building the Content Transfer Engine (CTE)
# Inherits from iowarp/iowarp-cte-build:latest which contains all build dependencies

FROM iowarp/iowarp-deps:latest

# Set working directory
WORKDIR /workspace

# Copy the entire CTE source tree
COPY . /workspace/

# Initialize git submodules and build
# Install to both /usr/local and /iowarp-cte for flexibility
RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    git submodule update --init --recursive && \
    mkdir -p build && \
    cmake --preset release && \
    cmake --build build -j$(nproc) && \
    sudo cmake --install build --prefix /usr/local && \
    sudo cmake --install build --prefix /iowarp-cte && \
    sudo rm -rf /workspace


# Add iowarp-cte to Spack configuration
RUN echo "  iowarp-core:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: iowarp-core@main" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr/local" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml

# Create empty runtime configuration file
RUN sudo mkdir -p /etc/iowarp && \
    sudo touch /etc/iowarp/wrp_conf.yaml

# Set runtime configuration environment variable
ENV WRP_RUNTIME_CONF=/etc/iowarp/wrp_conf.yaml
