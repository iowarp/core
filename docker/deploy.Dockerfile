# Deployment Dockerfile for IOWarp Core
# Inherits from the build container
FROM iowarp/core-build:latest

# Create empty runtime configuration file (if not inherited from base)
RUN sudo mkdir -p /etc/iowarp && \
    sudo touch /etc/iowarp/wrp_conf.yaml

# Set runtime configuration environment variable
ENV WRP_RUNTIME_CONF=/etc/iowarp/wrp_conf.yaml
