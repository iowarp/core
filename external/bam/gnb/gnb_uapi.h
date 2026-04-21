/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * gpu_nvme_bridge -- GPU-direct NVMe I/O (cooperative mode)
 *
 * Shared userspace/kernelspace header.
 *
 * This module cooperates with the existing nvme driver. It does NOT
 * unbind or replace the nvme driver. Instead, it creates additional
 * I/O queue pairs on the same NVMe controller, using queue IDs above
 * those used by the nvme driver.
 */
#ifndef GNB_UAPI_H
#define GNB_UAPI_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
#endif

/* ------------------------------------------------------------------ */
/* NVMe command structures (NVMe spec, for GPU-side use)               */
/* ------------------------------------------------------------------ */

struct gnb_nvme_cmd {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t command_id;
    uint32_t nsid;
    uint64_t rsvd2;
    uint64_t metadata;
    uint64_t prp1;          /* Bus address of data buffer */
    uint64_t prp2;          /* PRP2 or PRP list pointer */
    uint32_t cdw10;         /* Starting LBA (low) */
    uint32_t cdw11;         /* Starting LBA (high) */
    uint32_t cdw12;         /* Number of logical blocks - 1 */
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
};

struct gnb_nvme_cqe {
    uint32_t result;
    uint32_t rsvd;
    uint16_t sq_head;
    uint16_t sq_id;
    uint16_t command_id;
    uint16_t status;        /* Bit 0 = phase tag */
};

#define GNB_NVME_OP_WRITE  0x01
#define GNB_NVME_OP_READ   0x02

/* ------------------------------------------------------------------ */
/* ioctl interface                                                     */
/* ------------------------------------------------------------------ */

#define GNB_IOC_MAGIC  'G'

/**
 * Attach to an existing NVMe controller (managed by the nvme driver).
 * The module finds the controller, discovers available queue IDs,
 * and maps BAR0 for doorbell access.
 */
struct gnb_attach {
    /* Input */
    char     nvme_dev[64];      /* e.g. "/dev/nvme0"                  */

    /* Output */
    uint64_t ns_size_blocks;    /* Namespace size (ns1)                */
    uint32_t lba_size;          /* Logical block size                  */
    uint32_t max_queue_depth;   /* From CAP.MQES                      */
    uint32_t doorbell_stride;   /* Doorbell stride in bytes            */
    uint32_t nsid;              /* Namespace ID (usually 1)            */
    uint32_t first_qid;         /* First available queue ID for us     */
    uint32_t max_qid;           /* Max queue ID we can use             */
};

#define GNB_IOC_ATTACH  _IOWR(GNB_IOC_MAGIC, 1, struct gnb_attach)

/**
 * Detach from the NVMe controller.
 */
#define GNB_IOC_DETACH  _IO(GNB_IOC_MAGIC, 2)

/**
 * Create an I/O queue pair using DMA-coherent host memory.
 * The module allocates memory, creates the queues via the nvme
 * driver's admin queue, and returns mmap offsets.
 */
struct gnb_create_queue {
    /* Input */
    uint32_t sq_depth;          /* Power of 2, <= max_queue_depth      */
    uint32_t cq_depth;          /* Power of 2, <= max_queue_depth      */

    /* Output */
    uint32_t qid;               /* Assigned queue ID                   */
    uint32_t sq_db_offset;      /* SQ tail doorbell offset in BAR0     */
    uint32_t cq_db_offset;      /* CQ head doorbell offset in BAR0     */
    uint32_t _pad;
    uint64_t sq_mmap_offset;    /* mmap offset for SQ memory           */
    uint64_t cq_mmap_offset;    /* mmap offset for CQ memory           */
};

#define GNB_IOC_CREATE_QUEUE  _IOWR(GNB_IOC_MAGIC, 3, struct gnb_create_queue)

/**
 * Destroy an I/O queue pair.
 */
struct gnb_destroy_queue {
    uint32_t qid;
    uint32_t _pad;
};

#define GNB_IOC_DESTROY_QUEUE _IOW(GNB_IOC_MAGIC, 4, struct gnb_destroy_queue)

/**
 * Allocate a DMA-coherent data buffer.
 * Returns mmap offset and bus address (for use as PRP in NVMe cmds).
 */
struct gnb_alloc_buffer {
    /* Input */
    uint64_t size;              /* Requested size (page-aligned)       */

    /* Output */
    uint64_t mmap_offset;       /* mmap offset for this buffer         */
    uint64_t bus_addr;          /* DMA bus address (for PRP entries)    */
};

#define GNB_IOC_ALLOC_BUF  _IOWR(GNB_IOC_MAGIC, 5, struct gnb_alloc_buffer)

/**
 * Free a DMA-coherent data buffer.
 */
struct gnb_free_buffer {
    uint64_t mmap_offset;       /* Identifies the buffer               */
};

#define GNB_IOC_FREE_BUF  _IOW(GNB_IOC_MAGIC, 6, struct gnb_free_buffer)

/**
 * Register a GPU VRAM buffer for P2P DMA with NVMe.
 *
 * The userspace process passes a CUDA device pointer and size.
 * The kernel module calls nvidia_p2p_get_pages() to pin the GPU
 * memory and obtain bus addresses that the NVMe controller can
 * DMA to/from directly (GPU VRAM <-> NVMe, no host RAM involved).
 *
 * Returns an array of bus addresses (one per GPU page, typically 64KB).
 * The GPU kernel uses these as PRP entries in NVMe commands.
 */
struct gnb_reg_gpu_buffer {
    /* Input */
    uint64_t gpu_va;            /* CUDA device pointer (cudaMalloc'd)  */
    uint64_t size;              /* Size in bytes (64KB-aligned)        */

    /* Output */
    uint32_t buf_id;            /* Buffer ID for later unreg           */
    uint32_t page_size;         /* GPU page size (usually 64KB)        */
    uint32_t num_pages;         /* Number of GPU pages pinned          */
    uint32_t _pad;
    /*
     * Bus addresses are returned via a separate ioctl
     * (GNB_IOC_GET_GPU_PAGES) to avoid variable-length structs.
     */
};

#define GNB_IOC_REG_GPU_BUF  _IOWR(GNB_IOC_MAGIC, 7, struct gnb_reg_gpu_buffer)

/**
 * Unregister a GPU VRAM buffer.
 */
struct gnb_unreg_gpu_buffer {
    uint32_t buf_id;
    uint32_t _pad;
};

#define GNB_IOC_UNREG_GPU_BUF _IOW(GNB_IOC_MAGIC, 8, struct gnb_unreg_gpu_buffer)

/**
 * Get bus addresses of pinned GPU pages.
 *
 * After GNB_IOC_REG_GPU_BUF, call this to retrieve the per-page
 * bus addresses. The GPU kernel uses these as PRP1 entries.
 */
struct gnb_get_gpu_pages {
    /* Input */
    uint32_t buf_id;
    uint32_t max_pages;         /* Size of user's bus_addrs array      */

    /* Output (written to user-supplied array) */
    uint64_t bus_addrs_ptr;     /* Pointer to uint64_t[] in userspace  */
    uint32_t num_pages;         /* Actual pages returned               */
    uint32_t page_size;         /* GPU page size                       */
};

#define GNB_IOC_GET_GPU_PAGES _IOWR(GNB_IOC_MAGIC, 9, struct gnb_get_gpu_pages)

/* ------------------------------------------------------------------ */
/* mmap offset encoding                                                */
/* ------------------------------------------------------------------ */

/*
 * mmap offset layout (in the pgoff field):
 *
 *   0x00000  Doorbell BAR region (4KB or more)
 *   0x10000 + qid * 0x10000          Queue pair memory (SQ then CQ)
 *   0x100000 + buf_idx * 0x100000    Data buffer
 *
 * All offsets are page-aligned. The actual mmap size must match
 * the allocation size.
 */
#define GNB_MMAP_DOORBELLS      0x00000ULL
#define GNB_MMAP_QUEUE_BASE     0x10000ULL
#define GNB_MMAP_QUEUE_STRIDE   0x10000ULL
#define GNB_MMAP_BUF_BASE       0x100000ULL
#define GNB_MMAP_BUF_STRIDE     0x100000ULL

/* ------------------------------------------------------------------ */
/* Limits                                                              */
/* ------------------------------------------------------------------ */

#define GNB_MAX_QUEUES      16
#define GNB_MAX_QUEUE_DEPTH 4096
#define GNB_MAX_BUFFERS     32
#define GNB_MAX_GPU_BUFFERS 32
#define GNB_GPU_PAGE_SIZE   (64ULL * 1024)  /* nvidia_p2p uses 64KB pages */

#endif /* GNB_UAPI_H */
