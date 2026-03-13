#!/bin/bash
#
# download_test.sh - Download from Globus collection via HTTPS server
#
# This script:
# 1) Uses GLOBUS_ACCESS_TOKEN (Transfer API token) to resolve https_server
# 2) Uses GLOBUS_HTTPS_ACCESS_TOKEN (collection token) to download/list data
#
# Usage:
#   export GLOBUS_ACCESS_TOKEN="transfer_token"
#   export GLOBUS_HTTPS_ACCESS_TOKEN="collection_token"
#   ./download_test.sh [--collection-id UUID] <remote_path> [output_file]
#
# If --collection-id is not provided, the script uses:
#   722751ce-1264-43b8-9160-a9272f746d78
#
# Examples:
#   ./download_test.sh /
#   ./download_test.sh /share/godata/file1.txt /tmp/file1.txt
#   ./download_test.sh --collection-id 00000000-0000-0000-0000-000000000000 /

set -e

COLLECTION_ID_DEFAULT="722751ce-1264-43b8-9160-a9272f746d78"

usage() {
    echo "Usage: $0 [--collection-id UUID] <remote_path> [output_file]" 1>&2
}

# Parse arguments (optional --collection-id first, then positionals)
PARSED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --collection-id)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --collection-id requires an argument" 1>&2
                usage
                exit 1
            fi
            COLLECTION_ID="$2"
            shift 2
            ;;
        --collection-id=*)
            COLLECTION_ID="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                PARSED_ARGS+=("$1")
                shift
            done
            ;;
        -*)
            echo "ERROR: Unknown option: $1" 1>&2
            usage
            exit 1
            ;;
        *)
            PARSED_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${PARSED_ARGS[@]}"

COLLECTION_ID="${COLLECTION_ID:-${COLLECTION_ID_DEFAULT}}"
REMOTE_PATH="${1:-${REMOTE_PATH:-}}"
OUTPUT_FILE="${2:-${OUTPUT_FILE:-/tmp/globus_download.bin}}"
TMP_RESPONSE_BODY="/tmp/globus_download_response.$$"

cleanup() {
    rm -f "${TMP_RESPONSE_BODY}"
}
trap cleanup EXIT

print_globus_error_hints() {
    local body_file="$1"
    local error_code
    error_code=$(jq -r '.code // empty' "${body_file}" 2>/dev/null || true)
    if [ -z "${error_code}" ]; then
        return
    fi

    echo ""
    echo "Globus error code: ${error_code}"
    case "${error_code}" in
        ConsentRequired)
            echo "Hint: re-authorize and include collection data access scope:"
            echo "  https://auth.globus.org/scopes/${COLLECTION_ID}/data_access"
            ;;
        InvalidToken)
            echo "Hint: collection token is invalid/expired for this collection."
            echo "Re-run get_oauth_token.py and source /tmp/globus_tokens.sh."
            ;;
        InsufficientScope)
            echo "Hint: token is missing required scopes."
            echo "Required collection scopes:"
            echo "  https://auth.globus.org/scopes/${COLLECTION_ID}/https"
            echo "  https://auth.globus.org/scopes/${COLLECTION_ID}/data_access"
            ;;
        InsufficientSession)
            echo "Hint: your current authenticated identity/session is not valid for this collection."
            echo "Use Globus login with the required identity/domain and re-consent."
            ;;
        AccessDenied)
            echo "Hint: your identity is not allowed to access this collection/path."
            ;;
    esac

    local required_scopes
    required_scopes=$(jq -r '.authorization_parameters.required_scopes[]? // empty' "${body_file}" 2>/dev/null || true)
    if [ -n "${required_scopes}" ]; then
        echo "Required scopes reported by Globus:"
        echo "${required_scopes}" | sed 's/^/  - /'
    fi
}

echo "========================================="
echo "Globus HTTPS Download Test"
echo "========================================="
echo ""

if [ -z "${GLOBUS_ACCESS_TOKEN:-}" ]; then
    echo "ERROR: GLOBUS_ACCESS_TOKEN is not set"
    echo "This must be the transfer.api.globus.org token."
    exit 1
fi

if [ -z "${GLOBUS_HTTPS_ACCESS_TOKEN:-}" ]; then
    echo "ERROR: GLOBUS_HTTPS_ACCESS_TOKEN is not set"
    echo "This must be the collection token for ${COLLECTION_ID}."
    exit 1
fi

if [ -z "${REMOTE_PATH}" ]; then
    echo "ERROR: Missing remote path"
    echo "Usage: ./download_test.sh <remote_path> [output_file]"
    exit 1
fi

echo "Configuration:"
echo "  Collection ID: ${COLLECTION_ID}"
echo "  Remote Path:   ${REMOTE_PATH}"
echo "  Output File:   ${OUTPUT_FILE}"
echo ""

echo "Step 1: Getting endpoint details..."
ENDPOINT_DETAILS=$(curl -sS -X GET \
  "https://transfer.api.globus.org/v0.10/endpoint/${COLLECTION_ID}" \
  -H "Authorization: Bearer ${GLOBUS_ACCESS_TOKEN}")

echo "Endpoint details response:"
echo "${ENDPOINT_DETAILS}" | jq '.' || echo "${ENDPOINT_DETAILS}"
echo ""

HTTPS_SERVER=$(echo "${ENDPOINT_DETAILS}" | jq -r '.https_server // empty')
if [ -z "${HTTPS_SERVER}" ]; then
    echo "ERROR: Could not find https_server in endpoint details"
    echo "This endpoint may not have HTTPS access enabled"
    exit 1
fi

echo "Step 2: Found HTTPS server: ${HTTPS_SERVER}"
echo ""

DOWNLOAD_URL="${HTTPS_SERVER%/}${REMOTE_PATH}"
echo "Step 3: Requesting ${DOWNLOAD_URL}"

# Directory listing mode: print JSON response
if [[ "${REMOTE_PATH}" == */ ]]; then
    echo "(directory mode)"
    HTTP_CODE=$(curl -sS -w "%{http_code}" -X GET \
      "${DOWNLOAD_URL}" \
      -H "Authorization: Bearer ${GLOBUS_HTTPS_ACCESS_TOKEN}" \
      -o "${TMP_RESPONSE_BODY}")
    if [[ "${HTTP_CODE}" != 2* ]]; then
        echo "ERROR: directory request failed with HTTP ${HTTP_CODE}"
        jq '.' "${TMP_RESPONSE_BODY}" 2>/dev/null || cat "${TMP_RESPONSE_BODY}"
        print_globus_error_hints "${TMP_RESPONSE_BODY}"
        exit 1
    fi
    jq '.' "${TMP_RESPONSE_BODY}" 2>/dev/null || cat "${TMP_RESPONSE_BODY}"
else
    echo "(file download mode)"
    HTTP_CODE=$(curl -sS -w "%{http_code}" -L -X GET \
      "${DOWNLOAD_URL}" \
      -H "Authorization: Bearer ${GLOBUS_HTTPS_ACCESS_TOKEN}" \
      -o "${TMP_RESPONSE_BODY}")
    if [[ "${HTTP_CODE}" != 2* ]]; then
        echo "ERROR: file download failed with HTTP ${HTTP_CODE}"
        jq '.' "${TMP_RESPONSE_BODY}" 2>/dev/null || cat "${TMP_RESPONSE_BODY}"
        print_globus_error_hints "${TMP_RESPONSE_BODY}"
        exit 1
    fi
    mv "${TMP_RESPONSE_BODY}" "${OUTPUT_FILE}"
    echo "Downloaded file to: ${OUTPUT_FILE}"
fi

echo ""
echo "========================================="
echo "Test Complete"
echo "========================================="
