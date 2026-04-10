#!/bin/bash
#
# run_test.sh - Integration test for Globus data assimilation
#
# This script:
# 1. Starts the Chimaera runtime (with CTE + CAE compose) in the background
# 2. Runs wrp_cae_omni to process the OMNI file
#
# Prerequisites:
# - GLOBUS_ACCESS_TOKEN environment variable must be set
# - Globus endpoint must be accessible
# - chimaera and wrp_cae_omni must be installed and in PATH
# - Built with -DCAE_ENABLE_GLOBUS=ON
#
# Usage:
#   export GLOBUS_ACCESS_TOKEN="your_token_here"
#   ./run_test.sh

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime configuration (includes compose section for CTE + CAE pools)
RUNTIME_CONF="${SCRIPT_DIR}/wrp_runtime_conf.yaml"

# OMNI file
OMNI_FILE="${SCRIPT_DIR}/matsci_globus_omni.yaml"

# Output directory for transferred files
OUTPUT_DIR="/tmp/globus_matsci"

echo "========================================="
echo "Globus Materials Science Integration Test"
echo "========================================="
echo ""

# Source saved tokens if not already set
if [ -z "${GLOBUS_ACCESS_TOKEN}" ] && [ -f /tmp/globus_tokens.sh ]; then
    source /tmp/globus_tokens.sh
fi

# Check for Globus access token
if [ -z "${GLOBUS_ACCESS_TOKEN}" ]; then
    echo "ERROR: GLOBUS_ACCESS_TOKEN environment variable is not set"
    echo ""
    echo "To obtain a Globus access token:"
    echo "1. Install globus-sdk: pip install globus-sdk"
    echo "2. Run: python3 ${SCRIPT_DIR}/get_oauth_token.py --client-id YOUR_CLIENT_ID COLLECTION_ID"
    echo "3. Load tokens: source /tmp/globus_tokens.sh"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Runtime Config: ${RUNTIME_CONF}"
echo "  OMNI File:      ${OMNI_FILE}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo "Created output directory: ${OUTPUT_DIR}"
echo ""

# Start Chimaera runtime in the background
# The runtime config contains a compose section that creates both
# CTE (pool 512.0) and CAE (pool 400.0) automatically on startup.
echo "Starting Chimaera runtime..."
export CHI_SERVER_CONF="${RUNTIME_CONF}"
chimaera runtime start &
CHIMAERA_PID=$!
echo "Chimaera runtime started (PID: ${CHIMAERA_PID})"
echo ""

# Wait for runtime to initialize and create pools
echo "Waiting for runtime to initialize..."
sleep 10
echo ""

# Process OMNI file
echo "Processing OMNI file..."
wrp_cae_omni "${OMNI_FILE}"
OMNI_STATUS=$?

echo ""
if [ ${OMNI_STATUS} -eq 0 ]; then
    echo "========================================="
    echo "Test PASSED"
    echo "========================================="
    echo ""
    echo "Transferred files should be in: ${OUTPUT_DIR}"
    ls -lh "${OUTPUT_DIR}" 2>/dev/null || echo "No files found (transfer may have failed)"
else
    echo "========================================="
    echo "Test FAILED"
    echo "========================================="
    echo ""
    echo "OMNI processing failed with exit code: ${OMNI_STATUS}"
fi

# Cleanup: Stop Chimaera runtime
echo ""
echo "Stopping Chimaera runtime..."
chimaera runtime stop 2>/dev/null || kill ${CHIMAERA_PID} 2>/dev/null || true
wait ${CHIMAERA_PID} 2>/dev/null || true
echo "Chimaera runtime stopped"

exit ${OMNI_STATUS}
