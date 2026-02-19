# =============================================================================
# Multi-stage pip wheel build + test
# =============================================================================
# Stage 1 (builder): Build the wheel in deps-cpu with static archives
# Stage 2 (test):    Install in a clean Python image and verify runtime
#
# Usage (from repo root):
#   docker build -t iowarp/pip-test:latest -f installers/pip/test/Dockerfile .
#   docker run --rm --shm-size=256m iowarp/pip-test:latest
# =============================================================================

# ------------------------------------------------------------------
# Stage 1: Build the wheel
# ------------------------------------------------------------------
FROM iowarp/deps-cpu:latest AS builder

WORKDIR /build
COPY . /build/

RUN sudo chown -R $(whoami):$(whoami) /build

# Build the wheel using scikit-build-core (static deps via pyproject.toml)
RUN cd /build/installers/pip && \
    pip wheel -v -w /tmp/wheelhouse --no-build-isolation .

# Fix RPATHs so the wheel is self-contained ($ORIGIN-relative),
# then repack the wheel preserving Unix file permissions (execute bits).
RUN cd /tmp && mkdir -p unpack && \
    WHEEL=$(ls /tmp/wheelhouse/*.whl | head -1) && \
    python3 -m zipfile -e "$WHEEL" unpack/ && \
    chmod +x unpack/iowarp_core/bin/* && \
    bash /build/installers/pip/fix_rpaths.sh unpack/ && \
    rm "$WHEEL" && \
    WHEEL_NAME=$(basename "$WHEEL") && \
    python3 /build/installers/pip/test/repack_wheel.py \
        "/tmp/wheelhouse/$WHEEL_NAME" /tmp/unpack && \
    rm -rf /tmp/unpack

# Verify static linking — no external deps on yaml-cpp/zmq/sodium
RUN cd /tmp && mkdir -p verify && \
    python3 -m zipfile -e /tmp/wheelhouse/*.whl verify/ && \
    echo "=== Checking for dynamic deps that should be static ===" && \
    FAIL=0 && \
    for so in verify/iowarp_core/lib/*.so; do \
        [ -f "$so" ] || continue; \
        LEAKED=$(ldd "$so" 2>/dev/null | grep -E "yaml-cpp|libzmq|libsodium|libaio" || true); \
        if [ -n "$LEAKED" ]; then \
            echo "FAIL: $(basename $so) has leaked dynamic deps:"; \
            echo "$LEAKED"; \
            FAIL=1; \
        fi; \
    done && \
    if [ "$FAIL" = "1" ]; then \
        echo "ERROR: Wheel has dynamic dependencies that should be statically linked"; \
        exit 1; \
    fi && \
    echo "All external deps are statically linked" && \
    rm -rf verify

# ------------------------------------------------------------------
# Stage 2: Test in a clean container
# ------------------------------------------------------------------
FROM python:3.12-slim AS test

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    patchelf procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /test

# Copy wheel from builder
COPY --from=builder /tmp/wheelhouse/*.whl /test/wheelhouse/

# Copy test files
COPY installers/pip/fix_rpaths.sh /test/fix_rpaths.sh
COPY installers/pip/test/run_tests.sh /test/run_tests.sh
COPY installers/pip/test/chimaera_test.yaml /test/chimaera_test.yaml

RUN chmod +x /test/run_tests.sh

CMD ["/test/run_tests.sh"]
