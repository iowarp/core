#!/bin/bash
# Convenience wrapper for update_headers.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/update_headers.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: update_headers.py not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run the Python script with all arguments
python3 "$PYTHON_SCRIPT" "$@"
