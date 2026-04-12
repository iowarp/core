# gpu_nvme_bridge: Cooperative GPU-Direct NVMe I/O

## Executive Summary

`gpu_nvme_bridge` (GNB) is a Linux kernel module that creates additional
NVMe I/O queue pairs **alongside the existing nvme driver**. The nvme driver
continues to manage the device, filesystem, and its own queues. GNB adds
extra queues for GPU-direct (or userspace-direct) I/O submission.

Once queues are created, the GPU (or CPU test code) writes NVMe submission
queue entries, rings doorbells, and polls completion queue entries directly —
**zero CPU involvement in the I/O data path**.

### Key Properties

- **No driver unbinding**: The nvme driver stays loaded and manages the
  device normally. Filesystems remain mounted and functional.
- **No kernel thread**: Unlike io_uring or block-layer approaches, there is
  no CPU polling thread. The NVMe controller autonomously processes commands
  from all queue pairs.
- **Setup-only kernel involvement**: The module handles queue creation (via
  admin commands through the nvme driver) and memory management. After setup,
  the CPU is not in the I/O path.
- **Works in QEMU**: Host-memory fallback mode for testing without a GPU.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      nvme driver (unchanged) │
                    │                              │
                    │  Queues 1..N: filesystem I/O  │
                    │  Admin queue: shared          │
                    └──────────┬───────────────────┘
                               │ admin cmds via
                               │ __nvme_submit_sync_cmd
                    ┌──────────┴───────────────────┐
                    │      gnb module               │
                    │                              │
                    │  Create Queues N+1..M         │
                    │  Alloc DMA memory             │
                    │  Map BAR0 doorbells           │
                    │  mmap to userspace/GPU        │
                    └──────────┬───────────────────┘
                               │ mmap
                    ┌──────────┴───────────────────┐
                    │   Userspace / GPU kernel      │
                    │                              │
                    │  Write SQE → ring doorbell    │
                    │  Poll CQE → ring doorbell     │
                    │  (no CPU, no syscall)          │
                    └──────────────────────────────┘
```

---

## How It Works

### 1. Attach (`GNB_IOC_ATTACH`)

The module finds the NVMe controller via its character device path
(`/dev/nvme0`). It extracts `struct nvme_ctrl *` using
`container_of(inode->i_cdev, struct nvme_ctrl, cdev)`, then:

- Maps BAR0 independently (multiple ioremaps of the same BAR are safe)
- Reads CAP register for queue depth limits and doorbell stride
- Queries Get Features (Number of Queues) to find the allocated range
- Computes available queue IDs: `ctrl->queue_count` through `allocated_count`

### 2. Create Queue (`GNB_IOC_CREATE_QUEUE`)

Allocates DMA-coherent host memory for SQ and CQ, then submits
Create I/O CQ and Create I/O SQ admin commands through the nvme
driver's admin queue using the exported `__nvme_submit_sync_cmd()`.

The nvme driver handles admin queue locking and command ID management.
Our queues are invisible to the nvme driver — the NVMe controller
manages them independently.

### 3a. Allocate Host Buffer (`GNB_IOC_ALLOC_BUF`)

Allocates DMA-coherent host memory for data buffers. Returns:
- mmap offset (for userspace access)
- Bus/DMA address (for use as PRP entries in NVMe commands)

### 3b. Register GPU VRAM Buffer (`GNB_IOC_REG_GPU_BUF`)

For true GPU-direct I/O with zero host-memory involvement:

1. Userspace calls `cudaMalloc()` to allocate GPU VRAM (64KB-aligned)
2. Passes the CUDA device pointer to `GNB_IOC_REG_GPU_BUF`
3. Kernel module calls `nvidia_p2p_get_pages()` to pin GPU pages
   and obtain PCIe bus addresses
4. Userspace retrieves per-page bus addresses via `GNB_IOC_GET_GPU_PAGES`
5. GPU kernel uses these bus addresses as PRP1 in NVMe commands

The NVMe controller DMAs directly to/from GPU VRAM over PCIe —
no host memory copy, no CPU involvement.

**Requirements**: NVIDIA datacenter GPU (A100/V100/P100) with P2P DMA
support. GPU and NVMe must be on the same PCIe root complex (check
`nvidia-smi topo -m`). IOMMU must allow cross-device DMA.

Build with `make GPU_P2P=1` to enable this path.

### 4. mmap (doorbells, queues, buffers)

- **Doorbells**: BAR0 doorbell region mapped as uncacheable MMIO.
  For GPU: `cudaHostRegister(ptr, size, cudaHostRegisterIoMemory)`.
- **Queue memory**: DMA-coherent pages mapped to userspace.
  For GPU: `cudaHostRegister(ptr, size, cudaHostRegisterDefault)`.
- **Data buffers**: Same as queue memory.

### 5. I/O Submission (userspace or GPU)

```c
// Write NVMe command to SQ
sq[sq_tail].opcode = NVME_OP_WRITE;
sq[sq_tail].prp1 = buffer_bus_addr;
sq[sq_tail].cdw10 = lba_low;
sq[sq_tail].cdw12 = num_blocks - 1;
// ...

barrier();
sq_tail = (sq_tail + 1) % sq_depth;
*sq_doorbell = sq_tail;  // MMIO write → NVMe controller fetches SQE
```

### 6. Completion Polling

```c
// Poll CQ for phase bit
while ((cq[cq_head].status & 1) != expected_phase)
    ;  // spin

uint16_t cid = cq[cq_head].command_id;
int status = cq[cq_head].status >> 1;

cq_head = (cq_head + 1) % cq_depth;
if (cq_head == 0) expected_phase ^= 1;
*cq_doorbell = cq_head;  // MMIO write → releases CQE slot
```

---

## Queue ID Management

The nvme driver negotiates queue count with the controller during init
via Set Features (Number of Queues). Typically:

- Controller supports: 64+ queue pairs
- Driver requests: `num_online_cpus()` pairs
- Controller grants: `min(requested, supported)`

The driver creates queues 1..N (plus admin queue 0). GNB uses IDs
N+1..allocated_max. With 2 CPUs in QEMU, the driver uses queues 1-2,
leaving 3-64 for GNB.

If the driver reset the controller (error recovery), our queues would
be destroyed. A production version would need a notifier to detect
this and re-create queues.

---

## File Layout

```
gpu-nvme-bridge/
  DESIGN.md              This document
  gnb_uapi.h             Shared header (ioctl structs, NVMe types)
  gnb_common.h           Kernel-internal (includes nvme driver header)
  gnb_main.c             Module init, misc device registration
  gnb_nvme.c             Attach to ctrl, admin commands, queue mgmt
  gnb_fops.c             ioctl dispatch, mmap
  gnb_p2p.c             GPU VRAM P2P DMA (nvidia_p2p_get_pages)
  gnb_gpu.h              GPU-side NVMe helpers (CUDA __device__ fns)
  Makefile               Builds against kernel source tree
  test_gnb_host.c        CPU test (QEMU-compatible, no GPU needed)
  test_gnb_cuda.cu       GPU test (needs CUDA + real hardware)
  scripts/
    qemu-launch.sh       QEMU VM with emulated NVMe
    run-tests.sh         Build + load + test runner
```

---

## Building

Requires full kernel source tree (not just headers) because the module
includes `drivers/nvme/host/nvme.h` for `struct nvme_ctrl`.

```bash
# Point KDIR to kernel source
make KDIR=/usr/src/linux-source-6.x module

# With GPU VRAM P2P support (requires NVIDIA driver headers)
make GPU_P2P=1 KDIR=/usr/src/linux-source-6.x module

# Build userspace test
make test

# Build CUDA test
nvcc -O2 -o test_gnb_cuda test_gnb_cuda.cu
```

---

## QEMU Testing

```bash
# Launch VM with NVMe device (rootfs on virtio, NVMe separate)
./scripts/qemu-launch.sh vm/rootfs.qcow2

# Inside VM:
mount -t 9p -o trans=virtio gnb_src /mnt
cd /mnt
make KDIR=/lib/modules/$(uname -r)/build module
make test
sudo insmod gpu_nvme_bridge.ko
sudo ./test_gnb_host /dev/nvme0
sudo rmmod gpu_nvme_bridge
```

The nvme driver stays loaded. `/dev/nvme0n1` remains accessible.
GNB creates extra queue pairs that the nvme driver doesn't know about.

---

## Limitations

1. **Controller reset**: If the nvme driver resets the controller
   (error recovery, suspend/resume), our queues are destroyed without
   notification. Need a reset notifier for production use.

2. **Queue count**: Limited by what the nvme driver negotiated at boot.
   If it requested exactly N queues and the controller gave N, no spare
   IDs exist. Use fewer CPUs (QEMU: `-smp 2`) to free up queue IDs.

3. **Host memory queues**: SQ/CQ are in DMA-coherent host memory.
   Data buffers can be in GPU VRAM via `nvidia_p2p_get_pages` (build
   with `GPU_P2P=1`). Moving SQ/CQ to GPU VRAM would require NVMe
   Controller Memory Buffer (CMB) support, which few controllers have.

4. **Multi-page I/O**: Single-page I/Os only (PRP1 with PRP2=0).
   Multi-page I/O needs PRP lists, which requires allocating a PRP list
   page and setting PRP2 to its bus address.

---

## References

- [NVMe Base Specification](https://nvmexpress.org/specifications/)
- [BaM: GPU-Orchestrated Access to Storage](https://arxiv.org/abs/2203.04910)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Linux nvme driver source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/nvme/host/)
