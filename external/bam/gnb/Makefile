# SPDX-License-Identifier: GPL-2.0
#
# gpu_nvme_bridge -- cooperative GPU-direct NVMe bridge
#
# Usage:
#   make                           # Build kernel module (host-only, no P2P)
#   make GPU_P2P=1                 # Build with GPU VRAM P2P DMA support
#   make KDIR=/path/to/kernel/src  # Build against specific kernel source
#   make test                      # Build host test program
#   make clean
#
# The module requires the nvme driver's internal headers.
# KDIR must point to the full kernel source tree (not just headers).
#
# For GPU P2P support (GPU_P2P=1), you also need NVIDIA driver headers:
#   NVIDIA_DIR = /usr/src/nvidia-<version>/nvidia/
# The open-source kernel modules work too:
#   NVIDIA_DIR = /usr/src/nvidia-open-<version>/kernel-open/nvidia/

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

obj-m := gpu_nvme_bridge.o
gpu_nvme_bridge-objs := gnb_main.o gnb_nvme.o gnb_fops.o

# Include nvme driver's internal header directory for struct nvme_ctrl
ccflags-y += -I$(KDIR)/drivers/nvme/host
ccflags-y += -Wall -Werror

# GPU P2P DMA support (requires NVIDIA driver headers)
ifdef GPU_P2P
    gpu_nvme_bridge-objs += gnb_p2p.o
    ccflags-y += -DGNB_GPU_P2P

    # Auto-detect NVIDIA header location
    NVIDIA_DIR ?= $(shell ls -d /usr/src/nvidia-*/nvidia 2>/dev/null | tail -1)
    ifeq ($(NVIDIA_DIR),)
        NVIDIA_DIR := $(shell ls -d /usr/src/nvidia-open-*/kernel-open/nvidia 2>/dev/null | tail -1)
    endif
    ifneq ($(NVIDIA_DIR),)
        ccflags-y += -I$(NVIDIA_DIR)
    else
        $(warning GPU_P2P=1 but NVIDIA headers not found. Set NVIDIA_DIR manually.)
    endif

    # nvidia_p2p_get_pages API changed over driver versions.
    # Older drivers (< 515) use 5 params: (p2p_token, va_space, gpu_va, size, &pt, cb, data)
    # Newer drivers use 3 params: (gpu_va, size, &pt, cb, data)
    # Try to detect from the header.
    NV_P2P_HEADER := $(NVIDIA_DIR)/nv-p2p.h
    ifneq ($(wildcard $(NV_P2P_HEADER)),)
        NV_P2P_5_PARAMS := $(shell grep -c 'p2p_token' $(NV_P2P_HEADER) 2>/dev/null)
        ifneq ($(NV_P2P_5_PARAMS),0)
            ccflags-y += -DNV_P2P_GET_PAGES_HAS_5_PARAMS
        endif
    endif
endif

all: module

module:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

test: test_gnb_host
	@echo "Built test_gnb_host. Run: sudo ./test_gnb_host"

test_gnb_host: test_gnb_host.c gnb_uapi.h
	$(CC) -O2 -Wall -o $@ $<

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f test_gnb_host

.PHONY: all module test clean
