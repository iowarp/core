# BaM — Big Accelerator Memory

A basic implementation of [BaM (ASPLOS'23)](https://github.com/ZaidQureshi/bam/)
for GPU-initiated on-demand storage access. NVMe is optional — the system
falls back to pinned host memory when NVMe hardware is unavailable.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  GPU Kernel                         │
│                                                     │
│   bam::ArrayDevice<T>::read(idx) / write(idx, val)  │
│          │                                          │
│          ▼                                          │
│   ┌─────────────────────┐                           │
│   │   GPU Page Cache    │  (VRAM, direct-mapped)    │
│   │   page_cache_acquire│                           │
│   └────────┬────────────┘                           │
│        hit │     │ miss                             │
│            │     ▼                                  │
│            │  ┌──────────────────┐                  │
│            │  │ Backend I/O      │                  │
│            │  │  NVMe: gnb_gpu.h │  (P2P DMA)      │
│            │  │  Host: memcpy    │  (pinned mem)    │
│            │  └──────────────────┘                  │
│            ▼                                        │
│     return data                                     │
└─────────────────────────────────────────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `include/bam/types.h` | Core types: `PageCacheConfig`, `PageCacheDeviceState`, `PageState` |
| `include/bam/controller.h` | Host-side NVMe controller wrapping `gpu-nvme-bridge` ioctls |
| `include/bam/page_cache_host.h` | Host-side page cache manager (GPU memory allocation, backend init) |
| `include/bam/page_cache.cuh` | Device-side page cache: acquire, release, mark_dirty |
| `include/bam/array.cuh` | `bam::ArrayDevice<T>` — transparent storage-backed array for GPU kernels |
| `include/bam/bam.h` | Convenience header (host-side) |
| `src/controller.cc` | NVMe controller implementation (conditional on `BAM_ENABLE_NVME`) |
| `src/page_cache_host.cc` | Page cache host-side init (CUDA allocations, NVMe/host setup) |
| `src/page_cache.cu` | GPU kernels: cache init, spinlocks, host-memory I/O |
| `src/array.cu` | `bam::Array<T>` host-side template instantiations |

## Building

```bash
# Host-memory backend only (no NVMe required)
cmake -DWRP_CORE_ENABLE_BAM=ON -DBAM_ENABLE_TESTS=ON ..
make bam test_bam_basic

# With NVMe backend via gpu-nvme-bridge
cmake -DWRP_CORE_ENABLE_BAM=ON -DWRP_CORE_ENABLE_BAM_NVME=ON ..
make bam
```

## Usage

### Host-side setup

```cpp
#include <bam/bam.h>
#include <bam/array.cuh>

// Configure page cache
bam::PageCacheConfig config;
config.page_size = 4096;
config.num_pages = 1024;
config.backend = bam::BackendType::kHostMemory;  // or kNvme
config.nvme_dev = nullptr;  // "/dev/nvme0" for NVMe

bam::PageCache cache(config);
bam::Array<float> arr(num_elements, cache);
arr.load_from_host(host_data, num_elements);

// Launch kernel
my_kernel<<<grid, block>>>(arr.device());
```

### GPU kernel

```cuda
__global__ void my_kernel(bam::ArrayDevice<float> arr) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = arr.read(idx);
    arr.write(idx, val * 2.0f);
}
```

## Design Decisions

- **Direct-mapped cache**: Each storage page maps to exactly one cache slot
  (`slot = (offset / page_size) % num_pages`). Simple and predictable, though
  lower hit rates than set-associative. Sufficient for this basic version.

- **Per-page spinlocks**: Each cache slot has its own lock. A thread holds the
  lock from acquire through release. This is simple but limits concurrency to
  one thread per cache slot at a time.

- **Host-memory fallback**: When NVMe is unavailable, data lives in
  `cudaMallocHost` pinned memory. GPU threads read/write via `uint4`-wide
  copies through the PCIe bus. This is slower than NVMe P2P but works
  everywhere with a CUDA GPU.

- **Single-page I/O**: Each NVMe command transfers exactly one page (PRP1 only,
  PRP2=0). Multi-page I/O with PRP lists is left for a future version.

## References

- [BaM: GPU-Orchestrated Access to Storage (ASPLOS'23)](https://arxiv.org/abs/2203.04910)
- [gpu-nvme-bridge](../gpu-nvme-bridge/DESIGN.md) — cooperative GPU-direct NVMe kernel module
