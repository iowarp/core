# Dockerfile for CAE (Content Assimilation Engine) container
# Used to send omni files to the iowarp runtime

FROM iowarp/iowarp:latest

# Copy the omni file with updated path for file_assets.csv
# Build context is set to IOWARP_CORE_ROOT in docker-compose
COPY docker/test_external/file_assets_omni.yaml /binary_assim.yaml

# Copy the data directory
COPY context-assimilation-engine/data /workspace/context-assimilation-engine/data

WORKDIR /workspace

