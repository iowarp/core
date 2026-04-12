#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# Launch a QEMU VM for testing GNB cooperative mode.
#
# The VM has:
#   - Root filesystem on virtio-blk (/dev/vda)
#   - NVMe device (/dev/nvme0n1) with data, managed by nvme driver
#   - GNB creates extra queues on the same NVMe controller
#
# Prerequisites:
#   1. A root filesystem image with kernel headers + build-essential
#   2. QEMU with KVM support
#
# Usage:
#   ./scripts/qemu-launch.sh [rootfs.qcow2]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ROOTFS="${1:-${PROJECT_DIR}/vm/rootfs.qcow2}"
NVME_IMG="${PROJECT_DIR}/vm/nvme-test.img"
MEMORY="2G"
CPUS="2"  # Few CPUs = fewer nvme queues = more free QIDs for us
SSH_PORT="2222"

mkdir -p "$(dirname "$NVME_IMG")"

if [ ! -f "$NVME_IMG" ]; then
    echo "Creating 1GB NVMe test image..."
    qemu-img create -f raw "$NVME_IMG" 1G
fi

if [ ! -f "$ROOTFS" ]; then
    cat <<'HELP'
Root filesystem image not found.

Quick setup (Debian 12):
  mkdir -p vm
  wget https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2 \
       -O vm/rootfs.qcow2

  # Resize to have space for kernel source
  qemu-img resize vm/rootfs.qcow2 10G

  # Set root password and install build tools
  sudo virt-customize -a vm/rootfs.qcow2 \
    --root-password password:root \
    --install linux-headers-$(uname -r),build-essential,nvme-cli

  # Note: kernel headers must match the guest kernel version.
  # If the guest kernel differs, install matching headers inside the VM.
HELP
    exit 1
fi

echo "=== Launching QEMU VM ==="
echo "  RootFS:  $ROOTFS (virtio-blk, /dev/vda)"
echo "  NVMe:    $NVME_IMG (emulated NVMe, /dev/nvme0)"
echo "  SSH:     ssh -p $SSH_PORT root@localhost"
echo "  CPUs:    $CPUS (fewer = more free NVMe queue IDs for GNB)"
echo ""
echo "Inside VM:"
echo "  mount -t 9p -o trans=virtio gnb_src /mnt"
echo "  cd /mnt && ./scripts/run-tests.sh /dev/nvme0"
echo ""
echo "Note: The nvme driver stays loaded. GNB cooperates with it."
echo ""

# Use QEMU NVMe with max_ioqpairs to ensure we have spare queue IDs.
# The nvme driver creates 1 queue per CPU (2 here). QEMU's NVMe
# supports up to 64 queue pairs by default, so GNB gets IDs 3..64.

exec qemu-system-x86_64 \
    -enable-kvm \
    -m "$MEMORY" \
    -smp "$CPUS" \
    -cpu host \
    -drive file="$ROOTFS",format=qcow2,if=virtio \
    -drive file="$NVME_IMG",format=raw,if=none,id=nvme0 \
    -device nvme,serial=gnb-test,drive=nvme0 \
    -virtfs local,path="$PROJECT_DIR",mount_tag=gnb_src,security_model=mapped-xattr,id=gnb_src \
    -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 \
    -device virtio-net-pci,netdev=net0 \
    -nographic \
    -serial mon:stdio
