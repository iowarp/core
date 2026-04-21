// SPDX-License-Identifier: GPL-2.0
/*
 * test_gnb_cuda.cu -- GPU-direct NVMe test (cooperative mode)
 *
 * Two modes:
 *   1. Host-memory mode (default): Data buffers in DMA-coherent host RAM,
 *      GPU accesses via cudaHostRegister. Works in QEMU, any NVIDIA GPU.
 *
 *   2. GPU VRAM P2P mode (--p2p): Data buffers in GPU VRAM via cudaMalloc,
 *      registered with nvidia_p2p_get_pages for NVMe P2P DMA.
 *      Requires datacenter GPU (A100/V100/P100) + real NVMe hardware.
 *
 * SQ/CQ are always in host memory (NVMe spec requires controller-
 * accessible memory for queues; GPU VRAM queues would need CMB support).
 *
 * Build:
 *   nvcc -O2 -o test_gnb_cuda test_gnb_cuda.cu
 *
 * Usage:
 *   sudo ./test_gnb_cuda [/dev/nvme0]           # host-memory mode
 *   sudo ./test_gnb_cuda --p2p [/dev/nvme0]     # GPU VRAM P2P mode
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

#include "gnb_uapi.h"
#include "gnb_gpu.h"

#define QUEUE_DEPTH  64
#define NUM_IOS      8

#define CUDA_CHECK(call) do {                                         \
    cudaError_t e = (call);                                           \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e)); exit(1);                      \
    }                                                                 \
} while(0)

#ifndef PAGE_ALIGN
#define PAGE_ALIGN(x) (((x) + 4095UL) & ~4095UL)
#endif

/* ------------------------------------------------------------------ */
/* GPU kernel: P2P mode (per-page bus addresses)                       */
/* ------------------------------------------------------------------ */

/*
 * In P2P mode, each GPU page (64KB) has its own bus address.
 * For I/Os that fit within a single GPU page, we use that page's
 * bus address as PRP1. For cross-page I/Os we'd need PRP lists,
 * but for simplicity we align I/Os to GPU page boundaries.
 */
__global__ void gpu_io_kernel_p2p(
    volatile struct gnb_nvme_cmd *sq,
    volatile struct gnb_nvme_cqe *cq,
    volatile char *db_base,
    uint32_t sq_db_off,
    uint32_t cq_db_off,
    uint32_t sq_depth,
    uint32_t cq_depth,
    uint32_t nsid,
    uint32_t lba_size,
    uint64_t *page_bus_addrs,   /* Array of per-page bus addresses */
    uint32_t gpu_page_size,     /* Typically 64KB */
    char *data_buf,             /* GPU VRAM pointer for data access */
    uint32_t io_size,           /* Must be <= gpu_page_size */
    uint32_t num_ios,
    int *results)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    struct gnb_gpu_queue q;
    gnb_gpu_queue_init(&q, sq, cq, db_base,
                       sq_db_off, cq_db_off,
                       sq_depth, cq_depth, nsid, lba_size);

    /* Write pattern into GPU VRAM */
    for (uint32_t i = 0; i < num_ios * io_size; i++)
        data_buf[i] = (char)(i & 0xFF);
    __threadfence_system();

    /* Submit writes using per-page bus addresses */
    for (uint32_t i = 0; i < num_ios; i++) {
        uint32_t byte_off = i * io_size;
        uint32_t page_idx = byte_off / gpu_page_size;
        uint32_t page_off = byte_off % gpu_page_size;
        uint64_t bus = page_bus_addrs[page_idx] + page_off;
        uint64_t lba = (uint64_t)byte_off >> q.lba_shift;
        uint32_t nlb = (io_size >> q.lba_shift) - 1;

        gnb_gpu_submit_io(&q, GNB_NVME_OP_WRITE,
                          (uint16_t)(0x100 + i),
                          bus, 0, lba, nlb);
    }

    int write_errors = 0;
    for (uint32_t i = 0; i < num_ios; i++) {
        int32_t st;
        gnb_gpu_poll_cq(&q, &st);
        if (st != 0) write_errors++;
    }
    results[0] = write_errors;

    /* Clear GPU VRAM and read back from NVMe */
    for (uint32_t i = 0; i < num_ios * io_size; i++)
        data_buf[i] = 0;
    __threadfence_system();

    for (uint32_t i = 0; i < num_ios; i++) {
        uint32_t byte_off = i * io_size;
        uint32_t page_idx = byte_off / gpu_page_size;
        uint32_t page_off = byte_off % gpu_page_size;
        uint64_t bus = page_bus_addrs[page_idx] + page_off;
        uint64_t lba = (uint64_t)byte_off >> q.lba_shift;
        uint32_t nlb = (io_size >> q.lba_shift) - 1;

        gnb_gpu_submit_io(&q, GNB_NVME_OP_READ,
                          (uint16_t)(0x200 + i),
                          bus, 0, lba, nlb);
    }

    int read_errors = 0;
    for (uint32_t i = 0; i < num_ios; i++) {
        int32_t st;
        gnb_gpu_poll_cq(&q, &st);
        if (st != 0) read_errors++;
    }
    results[1] = read_errors;

    /* Verify data in GPU VRAM */
    __threadfence_system();
    int data_errors = 0;
    for (uint32_t i = 0; i < num_ios * io_size; i++) {
        if (data_buf[i] != (char)(i & 0xFF))
            data_errors++;
    }
    results[2] = data_errors;
}

/* ------------------------------------------------------------------ */
/* GPU kernel: host-memory mode (single contiguous bus address)        */
/* ------------------------------------------------------------------ */

__global__ void gpu_io_kernel_host(
    volatile struct gnb_nvme_cmd *sq,
    volatile struct gnb_nvme_cqe *cq,
    volatile char *db_base,
    uint32_t sq_db_off,
    uint32_t cq_db_off,
    uint32_t sq_depth,
    uint32_t cq_depth,
    uint32_t nsid,
    uint32_t lba_size,
    uint64_t buf_bus_addr,
    char *data_buf,
    uint32_t io_size,
    uint32_t num_ios,
    int *results)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    struct gnb_gpu_queue q;
    gnb_gpu_queue_init(&q, sq, cq, db_base,
                       sq_db_off, cq_db_off,
                       sq_depth, cq_depth, nsid, lba_size);

    /* Write pattern */
    for (uint32_t i = 0; i < num_ios * io_size; i++)
        data_buf[i] = (char)(i & 0xFF);
    __threadfence_system();

    /* Submit writes */
    for (uint32_t i = 0; i < num_ios; i++) {
        uint64_t bus = buf_bus_addr + i * io_size;
        gnb_gpu_write(&q, (uint16_t)(0x100 + i), bus,
                      (uint64_t)(i * io_size), io_size);
    }

    int write_errors = 0;
    for (uint32_t i = 0; i < num_ios; i++) {
        int32_t st;
        gnb_gpu_poll_cq(&q, &st);
        if (st != 0) write_errors++;
    }
    results[0] = write_errors;

    /* Clear and read back */
    for (uint32_t i = 0; i < num_ios * io_size; i++)
        data_buf[i] = 0;
    __threadfence_system();

    for (uint32_t i = 0; i < num_ios; i++) {
        uint64_t bus = buf_bus_addr + i * io_size;
        gnb_gpu_read(&q, (uint16_t)(0x200 + i), bus,
                     (uint64_t)(i * io_size), io_size);
    }

    int read_errors = 0;
    for (uint32_t i = 0; i < num_ios; i++) {
        int32_t st;
        gnb_gpu_poll_cq(&q, &st);
        if (st != 0) read_errors++;
    }
    results[1] = read_errors;

    /* Verify */
    __threadfence_system();
    int data_errors = 0;
    for (uint32_t i = 0; i < num_ios * io_size; i++) {
        if (data_buf[i] != (char)(i & 0xFF))
            data_errors++;
    }
    results[2] = data_errors;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    const char *nvme_dev = "/dev/nvme0";
    bool use_p2p = false;
    int fd, ret;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--p2p") == 0)
            use_p2p = true;
        else
            nvme_dev = argv[i];
    }

    /*
     * In P2P mode, I/O size must fit within one GPU page (64KB).
     * We use the full 64KB GPU page for maximum throughput.
     */
    uint32_t io_size = use_p2p ? GNB_GPU_PAGE_SIZE : 4096;
    uint64_t total_size = (uint64_t)io_size * NUM_IOS;

    printf("=== GNB GPU-Direct Test ===\n");
    printf("Mode: %s\n", use_p2p ? "GPU VRAM P2P" : "Host Memory");
    printf("I/O size: %u bytes x %d I/Os = %lu bytes total\n\n",
           io_size, NUM_IOS, (unsigned long)total_size);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n\n", prop.name);

    fd = open("/dev/gnb", O_RDWR);
    if (fd < 0) { perror("open /dev/gnb"); return 1; }

    /* Attach */
    struct gnb_attach att = {};
    strncpy(att.nvme_dev, nvme_dev, sizeof(att.nvme_dev) - 1);
    ret = ioctl(fd, GNB_IOC_ATTACH, &att);
    if (ret < 0) { perror("ATTACH"); return 1; }
    printf("NVMe: %lu blocks x %u bytes, queues %u..%u\n\n",
           (unsigned long)att.ns_size_blocks, att.lba_size,
           att.first_qid, att.max_qid - 1);

    /* Create queue */
    struct gnb_create_queue cqp = { .sq_depth = QUEUE_DEPTH,
                                    .cq_depth = QUEUE_DEPTH };
    ret = ioctl(fd, GNB_IOC_CREATE_QUEUE, &cqp);
    if (ret < 0) { perror("CREATE_QUEUE"); return 1; }
    printf("Queue %u created\n", cqp.qid);

    /* mmap doorbells */
    size_t db_size = 4096;
    void *db_mmap = mmap(NULL, db_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, GNB_MMAP_DOORBELLS);
    if (db_mmap == MAP_FAILED) { perror("mmap db"); return 1; }
    CUDA_CHECK(cudaHostRegister(db_mmap, db_size, cudaHostRegisterIoMemory));

    /* mmap SQ */
    size_t sq_sz = PAGE_ALIGN(QUEUE_DEPTH * sizeof(struct gnb_nvme_cmd));
    void *sq_mmap = mmap(NULL, sq_sz, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, cqp.sq_mmap_offset);
    if (sq_mmap == MAP_FAILED) { perror("mmap sq"); return 1; }
    CUDA_CHECK(cudaHostRegister(sq_mmap, sq_sz, cudaHostRegisterDefault));

    /* mmap CQ */
    size_t cq_sz = PAGE_ALIGN(QUEUE_DEPTH * sizeof(struct gnb_nvme_cqe));
    void *cq_mmap = mmap(NULL, cq_sz, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, cqp.cq_mmap_offset);
    if (cq_mmap == MAP_FAILED) { perror("mmap cq"); return 1; }
    CUDA_CHECK(cudaHostRegister(cq_mmap, cq_sz, cudaHostRegisterDefault));

    /* Results */
    int *d_results;
    CUDA_CHECK(cudaMallocManaged(&d_results, 3 * sizeof(int)));
    d_results[0] = d_results[1] = d_results[2] = -1;

    if (use_p2p) {
        /* ---- P2P MODE: data in GPU VRAM ---- */

        /*
         * Allocate GPU VRAM. Must be 64KB-aligned for P2P.
         * cudaMalloc returns 256-byte aligned at minimum;
         * for 64KB alignment we allocate extra and align manually.
         */
        char *gpu_buf_raw, *gpu_buf;
        uint64_t aligned_size = total_size + GNB_GPU_PAGE_SIZE;
        CUDA_CHECK(cudaMalloc(&gpu_buf_raw, aligned_size));
        gpu_buf = (char *)(((uintptr_t)gpu_buf_raw + GNB_GPU_PAGE_SIZE - 1)
                           & ~(GNB_GPU_PAGE_SIZE - 1));
        uint64_t gpu_va = (uint64_t)(uintptr_t)gpu_buf;

        printf("GPU VRAM buffer: va=0x%lx (raw=0x%lx)\n",
               (unsigned long)gpu_va, (unsigned long)(uintptr_t)gpu_buf_raw);

        /* Register GPU buffer with kernel module for P2P DMA */
        struct gnb_reg_gpu_buffer reg = {};
        reg.gpu_va = gpu_va;
        reg.size = total_size;
        ret = ioctl(fd, GNB_IOC_REG_GPU_BUF, &reg);
        if (ret < 0) {
            perror("REG_GPU_BUF");
            fprintf(stderr, "\nP2P registration failed. Possible causes:\n"
                    "  - Module not built with GPU_P2P=1\n"
                    "  - GPU doesn't support P2P (need datacenter GPU)\n"
                    "  - IOMMU blocking P2P DMA (try intel_iommu=off)\n"
                    "  - NVIDIA driver too old\n");
            return 1;
        }
        printf("Registered: %u pages x %u bytes (buf_id=%u)\n",
               reg.num_pages, reg.page_size, reg.buf_id);

        /* Get per-page bus addresses */
        uint64_t *bus_addrs = (uint64_t *)malloc(reg.num_pages * sizeof(uint64_t));
        struct gnb_get_gpu_pages gp = {};
        gp.buf_id = reg.buf_id;
        gp.max_pages = reg.num_pages;
        gp.bus_addrs_ptr = (uint64_t)(uintptr_t)bus_addrs;
        ret = ioctl(fd, GNB_IOC_GET_GPU_PAGES, &gp);
        if (ret < 0) { perror("GET_GPU_PAGES"); return 1; }

        printf("Bus addresses:\n");
        for (uint32_t i = 0; i < gp.num_pages && i < 4; i++)
            printf("  page[%u] = 0x%lx\n", i, (unsigned long)bus_addrs[i]);
        if (gp.num_pages > 4)
            printf("  ... (%u more)\n", gp.num_pages - 4);
        printf("\n");

        /* Copy bus addresses to GPU-accessible memory */
        uint64_t *d_bus_addrs;
        CUDA_CHECK(cudaMalloc(&d_bus_addrs, gp.num_pages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_bus_addrs, bus_addrs,
                              gp.num_pages * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        /* Launch P2P kernel */
        printf("Launching GPU P2P kernel...\n");
        gpu_io_kernel_p2p<<<1, 32>>>(
            (volatile struct gnb_nvme_cmd *)sq_mmap,
            (volatile struct gnb_nvme_cqe *)cq_mmap,
            (volatile char *)db_mmap,
            cqp.sq_db_offset - 0x1000,
            cqp.cq_db_offset - 0x1000,
            QUEUE_DEPTH, QUEUE_DEPTH,
            att.nsid, att.lba_size,
            d_bus_addrs, reg.page_size,
            gpu_buf, io_size, NUM_IOS, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Cleanup P2P */
        struct gnb_unreg_gpu_buffer unreg = { .buf_id = reg.buf_id };
        ioctl(fd, GNB_IOC_UNREG_GPU_BUF, &unreg);
        cudaFree(d_bus_addrs);
        free(bus_addrs);
        cudaFree(gpu_buf_raw);

    } else {
        /* ---- HOST MEMORY MODE ---- */

        struct gnb_alloc_buffer abuf = { .size = total_size };
        ret = ioctl(fd, GNB_IOC_ALLOC_BUF, &abuf);
        if (ret < 0) { perror("ALLOC_BUF"); return 1; }
        printf("Buffer: bus=0x%lx\n\n", (unsigned long)abuf.bus_addr);

        void *data_mmap = mmap(NULL, abuf.size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, abuf.mmap_offset);
        if (data_mmap == MAP_FAILED) { perror("mmap data"); return 1; }
        CUDA_CHECK(cudaHostRegister(data_mmap, abuf.size,
                                    cudaHostRegisterDefault));

        char *d_data;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_data, data_mmap, 0));

        printf("Launching GPU host-memory kernel...\n");
        gpu_io_kernel_host<<<1, 32>>>(
            (volatile struct gnb_nvme_cmd *)sq_mmap,
            (volatile struct gnb_nvme_cqe *)cq_mmap,
            (volatile char *)db_mmap,
            cqp.sq_db_offset - 0x1000,
            cqp.cq_db_offset - 0x1000,
            QUEUE_DEPTH, QUEUE_DEPTH,
            att.nsid, att.lba_size,
            abuf.bus_addr,
            d_data, io_size, NUM_IOS, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaHostUnregister(data_mmap);
        struct gnb_free_buffer fb = { .mmap_offset = abuf.mmap_offset };
        ioctl(fd, GNB_IOC_FREE_BUF, &fb);
        munmap(data_mmap, abuf.size);
    }

    /* Print results */
    printf("\nWrite errors: %d\n", d_results[0]);
    printf("Read errors:  %d\n", d_results[1]);
    printf("Data errors:  %d\n", d_results[2]);
    printf("Overall:      %s\n",
           (d_results[0] == 0 && d_results[1] == 0 && d_results[2] == 0)
           ? "PASSED" : "FAILED");

    /* Cleanup */
    cudaHostUnregister(cq_mmap);
    cudaHostUnregister(sq_mmap);
    cudaHostUnregister(db_mmap);

    struct gnb_destroy_queue dq = { .qid = cqp.qid };
    ioctl(fd, GNB_IOC_DESTROY_QUEUE, &dq);
    ioctl(fd, GNB_IOC_DETACH);

    munmap(cq_mmap, cq_sz);
    munmap(sq_mmap, sq_sz);
    munmap(db_mmap, db_size);
    cudaFree(d_results);
    close(fd);

    printf("\n=== Done ===\n");
    return 0;
}
