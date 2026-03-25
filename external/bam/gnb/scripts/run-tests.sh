#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# Run GNB cooperative tests.
#
# Usage (inside VM or on bare metal):
#   ./scripts/run-tests.sh [/dev/nvme0]
#
# The NVMe device stays bound to the nvme driver.
# GNB creates extra queue pairs alongside the driver's queues.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NVME_DEV="${1:-/dev/nvme0}"

echo "=== GNB Cooperative Test Runner ==="
echo "NVMe controller: $NVME_DEV"
echo ""

# Check NVMe device exists
if [ ! -c "$NVME_DEV" ]; then
    echo "ERROR: $NVME_DEV not found"
    echo "Available NVMe controllers:"
    ls /dev/nvme[0-9]* 2>/dev/null || echo "  (none)"
    exit 1
fi

# Check nvme driver is loaded and managing the device
if ! lsmod | grep -q "^nvme "; then
    echo "ERROR: nvme driver not loaded"
    exit 1
fi
echo "nvme driver is loaded (good — we cooperate with it)"

# Build
echo ""
echo "--- Building module ---"
cd "$PROJECT_DIR"
make module

echo ""
echo "--- Building test ---"
make test

# Load GNB module
echo ""
echo "--- Loading gnb module ---"
if lsmod | grep -q gpu_nvme_bridge; then
    sudo rmmod gpu_nvme_bridge
fi
sudo insmod gpu_nvme_bridge.ko
sleep 0.5

if [ ! -c /dev/gnb ]; then
    echo "ERROR: /dev/gnb not found after loading module"
    dmesg | tail -5
    exit 1
fi
echo "/dev/gnb available"

# Run test
echo ""
echo "--- Running test ---"
sudo ./test_gnb_host "$NVME_DEV"

# Cleanup
echo ""
echo "--- Unloading module ---"
sudo rmmod gpu_nvme_bridge

echo ""
echo "--- Kernel log (last 30 lines) ---"
dmesg | tail -30

echo ""
echo "=== Done ==="
