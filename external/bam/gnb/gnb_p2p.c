// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_nvme_bridge -- GPU P2P DMA support via nvidia_p2p_get_pages
 *
 * Pins GPU VRAM pages and obtains their bus (physical) addresses
 * so the NVMe controller can DMA directly to/from GPU memory.
 *
 * This file requires the NVIDIA driver headers:
 *   -I/usr/src/nvidia-<version>/nvidia/
 * or the open-source GPU kernel modules:
 *   -I/usr/src/nvidia-open-<version>/kernel-open/nvidia/
 *
 * The key API:
 *   nvidia_p2p_get_pages(gpu_va, size, &page_table, free_callback, data)
 *     - Pins GPU pages and returns bus addresses
 *     - page_table->pages[i]->physical_address is the PCIe bus addr
 *     - GPU pages are typically 64KB (not 4KB like CPU pages)
 *
 *   nvidia_p2p_put_pages(gpu_va, page_table)
 *     - Unpins GPU pages
 *
 *   nvidia_p2p_free_page_table(page_table)
 *     - Called in the free_callback (when GPU memory is freed
 *       while pages are still pinned)
 */

#include <linux/kernel.h>
#include <linux/slab.h>

#include "gnb_common.h"

/*
 * NVIDIA P2P header. Location varies by driver version:
 *   - Proprietary: /usr/src/nvidia-XXX/nvidia/nv-p2p.h
 *   - Open kernel modules: nvidia-open/kernel-open/nvidia/nv-p2p.h
 *
 * The Makefile adds the appropriate -I path.
 */
#include <nv-p2p.h>

/* ------------------------------------------------------------------ */
/* Free callback                                                       */
/* ------------------------------------------------------------------ */

/*
 * Called by the NVIDIA driver when the GPU memory backing our pinned
 * pages is freed (e.g., cudaFree before we unreg). We must release
 * the page table and mark the buffer inactive.
 */
static void gnb_p2p_free_callback(void *data)
{
    struct gnb_gpu_buffer *gbuf = data;
    nvidia_p2p_page_table_t *pt;

    pr_warn("gnb: GPU memory freed while P2P pages pinned (buf_id=%u)\n",
            gbuf->buf_id);

    pt = (nvidia_p2p_page_table_t *)gbuf->p2p_page_table;
    if (pt)
        nvidia_p2p_free_page_table(pt);

    gbuf->p2p_page_table = NULL;
    kfree(gbuf->bus_addrs);
    gbuf->bus_addrs = NULL;
    gbuf->active = false;
}

/* ------------------------------------------------------------------ */
/* Register GPU buffer                                                 */
/* ------------------------------------------------------------------ */

/**
 * gnb_reg_gpu_buffer - Pin GPU VRAM and get bus addresses for P2P DMA.
 *
 * @dev:  GNB device
 * @gbuf: GPU buffer struct (gpu_va and size must be set)
 *
 * On success, gbuf->bus_addrs[] contains one bus address per GPU page.
 * These addresses can be used as PRP entries in NVMe commands.
 */
int gnb_reg_gpu_buffer(struct gnb_device *dev, struct gnb_gpu_buffer *gbuf)
{
    nvidia_p2p_page_table_t *page_table = NULL;
    int ret;
    uint32_t i;

    /*
     * nvidia_p2p_get_pages pins the GPU memory and returns a page
     * table with bus addresses. The NVIDIA driver handles the GPU
     * MMU and ensures the pages remain resident.
     *
     * The free_callback is invoked if the GPU memory is freed
     * (cudaFree) while our pages are still pinned.
     */
    ret = nvidia_p2p_get_pages(
#ifdef NV_P2P_GET_PAGES_HAS_5_PARAMS
        0,  /* p2p_token (deprecated, pass 0) */
        0,  /* va_space (deprecated, pass 0) */
#endif
        gbuf->gpu_va,
        gbuf->size,
        &page_table,
        gnb_p2p_free_callback,
        gbuf);

    if (ret) {
        pr_err("gnb: nvidia_p2p_get_pages failed: %d "
               "(gpu_va=0x%llx size=%llu)\n",
               ret, gbuf->gpu_va, gbuf->size);
        return ret;
    }

    if (!page_table || page_table->entries == 0) {
        pr_err("gnb: nvidia_p2p_get_pages returned empty table\n");
        if (page_table)
            nvidia_p2p_free_page_table(page_table);
        return -EINVAL;
    }

    gbuf->num_pages = page_table->entries;
    gbuf->page_size = page_table->page_size;
    gbuf->p2p_page_table = page_table;

    /* Extract bus addresses */
    gbuf->bus_addrs = kcalloc(gbuf->num_pages, sizeof(uint64_t), GFP_KERNEL);
    if (!gbuf->bus_addrs) {
        nvidia_p2p_put_pages(
#ifdef NV_P2P_GET_PAGES_HAS_5_PARAMS
            0, 0,
#endif
            gbuf->gpu_va, page_table);
        gbuf->p2p_page_table = NULL;
        return -ENOMEM;
    }

    for (i = 0; i < gbuf->num_pages; i++) {
        gbuf->bus_addrs[i] = page_table->pages[i]->physical_address;
    }

    /* Translate nvidia page_size enum to bytes */
    switch (gbuf->page_size) {
    case NVIDIA_P2P_PAGE_SIZE_4KB:   gbuf->page_size = 4096; break;
    case NVIDIA_P2P_PAGE_SIZE_64KB:  gbuf->page_size = 65536; break;
    case NVIDIA_P2P_PAGE_SIZE_128KB: gbuf->page_size = 131072; break;
    default:
        pr_warn("gnb: unknown GPU page size enum %u, assuming 64KB\n",
                gbuf->page_size);
        gbuf->page_size = 65536;
    }

    gbuf->active = true;

    pr_info("gnb: pinned GPU buffer: gpu_va=0x%llx size=%llu "
            "pages=%u page_size=%u\n",
            gbuf->gpu_va, gbuf->size,
            gbuf->num_pages, gbuf->page_size);

    for (i = 0; i < min(gbuf->num_pages, 4u); i++) {
        pr_info("gnb:   page[%u] bus=0x%llx\n", i, gbuf->bus_addrs[i]);
    }
    if (gbuf->num_pages > 4)
        pr_info("gnb:   ... (%u more pages)\n", gbuf->num_pages - 4);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Unregister GPU buffer                                               */
/* ------------------------------------------------------------------ */

void gnb_unreg_gpu_buffer(struct gnb_device *dev, struct gnb_gpu_buffer *gbuf)
{
    nvidia_p2p_page_table_t *pt;

    if (!gbuf->active)
        return;

    pt = (nvidia_p2p_page_table_t *)gbuf->p2p_page_table;
    if (pt) {
        /*
         * nvidia_p2p_put_pages unpins the GPU memory and invalidates
         * the bus addresses. After this, the NVMe controller must NOT
         * access these addresses.
         */
        nvidia_p2p_put_pages(
#ifdef NV_P2P_GET_PAGES_HAS_5_PARAMS
            0, 0,
#endif
            gbuf->gpu_va, pt);
        gbuf->p2p_page_table = NULL;
    }

    kfree(gbuf->bus_addrs);
    gbuf->bus_addrs = NULL;
    gbuf->active = false;

    pr_info("gnb: unpinned GPU buffer buf_id=%u\n", gbuf->buf_id);
}

void gnb_unreg_all_gpu_buffers(struct gnb_device *dev)
{
    int i;
    for (i = 0; i < GNB_MAX_GPU_BUFFERS; i++) {
        if (dev->gpu_buffers[i].active)
            gnb_unreg_gpu_buffer(dev, &dev->gpu_buffers[i]);
    }
}
