/* SPDX-License-Identifier: GPL-2.0 */
/*
 * gpu_nvme_bridge -- kernel-internal definitions (cooperative mode)
 */
#ifndef GNB_COMMON_H
#define GNB_COMMON_H

#include <linux/types.h>
#include <linux/pci.h>
#include <linux/miscdevice.h>
#include <linux/mutex.h>

/*
 * Include the nvme driver's internal header for struct nvme_ctrl.
 * The Makefile adds -I$(KDIR)/drivers/nvme/host to make this work.
 *
 * We need:
 *   - struct nvme_ctrl (for admin_q, device, queue_count)
 *   - struct nvme_command (for Create CQ/SQ admin commands)
 *   - __nvme_submit_sync_cmd (exported by nvme-core)
 *
 * This couples us to the nvme driver's internals, which is acceptable
 * for a research prototype. Production code would need a stable API.
 */
#include "nvme.h"

#include "gnb_uapi.h"

/* ------------------------------------------------------------------ */
/* Queue state                                                         */
/* ------------------------------------------------------------------ */

struct gnb_queue {
    uint32_t            qid;
    uint32_t            sq_depth;
    uint32_t            cq_depth;

    /* DMA-coherent queue memory (host RAM) */
    void                *sq_vaddr;
    dma_addr_t          sq_dma;
    void                *cq_vaddr;
    dma_addr_t          cq_dma;

    /* Doorbell offsets within BAR0 */
    uint32_t            sq_db_offset;
    uint32_t            cq_db_offset;

    /* mmap tracking */
    uint64_t            sq_mmap_offset;
    uint64_t            cq_mmap_offset;

    bool                active;
};

/* ------------------------------------------------------------------ */
/* Data buffer (host DMA-coherent)                                     */
/* ------------------------------------------------------------------ */

struct gnb_buffer {
    void                *vaddr;
    dma_addr_t          dma_addr;
    uint64_t            size;
    uint64_t            mmap_offset;
    bool                active;
};

/* ------------------------------------------------------------------ */
/* GPU VRAM buffer (P2P DMA via nvidia_p2p_get_pages)                  */
/* ------------------------------------------------------------------ */

struct gnb_gpu_buffer {
    uint64_t            gpu_va;         /* CUDA device pointer          */
    uint64_t            size;
    uint32_t            buf_id;
    uint32_t            num_pages;
    uint32_t            page_size;
    bool                active;

    /*
     * nvidia_p2p_page_table_t from nvidia_p2p_get_pages.
     * Opaque here; gnb_p2p.c casts to the real type.
     * We use void* to avoid requiring nv-p2p.h in this header.
     */
    void                *p2p_page_table;

    /*
     * Bus addresses extracted from the page table.
     * One per GPU page (64KB each). Used as PRP addresses
     * in NVMe commands for true GPU-direct I/O.
     */
    uint64_t            *bus_addrs;
};

/* ------------------------------------------------------------------ */
/* Per-open-file device state                                          */
/* ------------------------------------------------------------------ */

struct gnb_device {
    struct miscdevice   misc;
    struct mutex        lock;

    /* NVMe controller (owned by nvme driver, we just hold a ref) */
    struct nvme_ctrl    *ctrl;          /* nvme driver's ctrl struct    */
    struct pci_dev      *pdev;          /* NVMe PCI device             */

    /* BAR0 mapping (our own ioremap, independent of nvme driver's) */
    void __iomem        *bar;
    resource_size_t     bar_phys;
    size_t              bar_size;

    /* Controller parameters (cached from CAP + Identify) */
    uint32_t            doorbell_stride;
    uint32_t            max_queue_depth;
    uint32_t            db_offset;      /* 0x1000 (doorbell region start) */

    /* Namespace info */
    uint32_t            nsid;
    uint64_t            ns_size_blocks;
    uint32_t            lba_size;

    /* Queue ID range available to us */
    uint32_t            first_qid;      /* First QID above nvme driver's */
    uint32_t            max_qid;        /* Max QID from Set Features     */
    uint32_t            next_qid;       /* Next QID to allocate          */

    /* Our I/O queues */
    struct gnb_queue    queues[GNB_MAX_QUEUES];

    /* Host DMA-coherent data buffers */
    struct gnb_buffer   buffers[GNB_MAX_BUFFERS];

    /* GPU VRAM P2P buffers */
    struct gnb_gpu_buffer gpu_buffers[GNB_MAX_GPU_BUFFERS];
    uint32_t            next_gpu_buf_id;

    bool                attached;
};

/* ------------------------------------------------------------------ */
/* Doorbell offset helpers                                             */
/* ------------------------------------------------------------------ */

static inline uint32_t gnb_sq_db_off(struct gnb_device *dev, uint32_t qid)
{
    return dev->db_offset + 2 * qid * dev->doorbell_stride;
}

static inline uint32_t gnb_cq_db_off(struct gnb_device *dev, uint32_t qid)
{
    return dev->db_offset + (2 * qid + 1) * dev->doorbell_stride;
}

/* ------------------------------------------------------------------ */
/* Function declarations                                               */
/* ------------------------------------------------------------------ */

/* gnb_nvme.c */
int  gnb_attach_ctrl(struct gnb_device *dev, const char *nvme_path);
void gnb_detach_ctrl(struct gnb_device *dev);
int  gnb_create_io_queue(struct gnb_device *dev, struct gnb_queue *q);
void gnb_destroy_io_queue(struct gnb_device *dev, struct gnb_queue *q);

/* gnb_p2p.c */
int  gnb_reg_gpu_buffer(struct gnb_device *dev, struct gnb_gpu_buffer *gbuf);
void gnb_unreg_gpu_buffer(struct gnb_device *dev, struct gnb_gpu_buffer *gbuf);
void gnb_unreg_all_gpu_buffers(struct gnb_device *dev);

/* gnb_fops.c */
extern const struct file_operations gnb_fops;

#define GNB_NAME "gnb"

#endif /* GNB_COMMON_H */
