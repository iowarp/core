#!/bin/bash

# Build iowarp-core Docker images

# Get the project root directory (parent of docker folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Build the core Docker images
docker build --no-cache -t iowarp/core-build:latest -f "${SCRIPT_DIR}/local.Dockerfile" "${PROJECT_ROOT}"

docker build --no-cache -t iowarp/core:latest -f "${SCRIPT_DIR}/deploy.Dockerfile" "${PROJECT_ROOT}"

# Build the minimal development image (Python + venv + build tools)
docker build --no-cache -t iowarp/minimal:latest -f "${SCRIPT_DIR}/minimal.Dockerfile" "${PROJECT_ROOT}"

# Build the benchmark Docker images
# docker build --no-cache -t iowarp/redis-bench:latest -f "${SCRIPT_DIR}/redis_bench/Dockerfile" "${PROJECT_ROOT}"

# docker build --no-cache -t iowarp/cte-bench:latest -f "${SCRIPT_DIR}/wrp_cte_bench/Dockerfile" "${PROJECT_ROOT}"
