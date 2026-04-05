// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_nvme_bridge -- NVMe controller interaction (cooperative mode)
 *
 * Attaches to an NVMe controller managed by the standard nvme driver.
 * Creates additional I/O queue pairs using the nvme driver's admin
 * queue (via __nvme_submit_sync_cmd). Does NOT replace or unbind the
 * nvme driver.
 *
 * Finding the nvme_ctrl:
 *   /dev/nvme0 → inode → i_cdev → container_of → struct nvme_ctrl
 *
 * The nvme_ctrl has:
 *   - admin_q: request_queue for admin commands
 *   - dev: struct device * (the PCI device)
 *   - queue_count: number of queues the driver created
 */

#include <linux/fs.h>
#include <linux/namei.h>
#include <linux/pci.h>
#include <linux/dma-mapping.h>

#include "gnb_common.h"

/* ------------------------------------------------------------------ */
/* NVMe CAP register fields                                            */
/* ------------------------------------------------------------------ */

#define NVME_REG_CAP   0x0000
#define NVME_CAP_MQES(cap)   ((cap) & 0xFFFF)
#define NVME_CAP_DSTRD(cap)  (((cap) >> 32) & 0xF)

/* ------------------------------------------------------------------ */
/* Attach to NVMe controller                                           */
/* ------------------------------------------------------------------ */

/**
 * gnb_attach_ctrl - Find and attach to an NVMe controller.
 *
 * @dev:       GNB device state
 * @nvme_path: Path to NVMe char device (e.g., "/dev/nvme0")
 *
 * Opens the NVMe character device, extracts struct nvme_ctrl via
 * container_of on the cdev, maps BAR0 for doorbell access, and
 * determines the available queue ID range.
 */
int gnb_attach_ctrl(struct gnb_device *dev, const char *nvme_path)
{
    struct path path;
    struct inode *inode;
    struct nvme_ctrl *ctrl;
    struct pci_dev *pdev;
    uint64_t cap;
    uint32_t num_queues_allocated;
    int ret;

    /* Look up the NVMe character device */
    ret = kern_path(nvme_path, LOOKUP_FOLLOW, &path);
    if (ret) {
        pr_err("gnb: cannot find %s: %d\n", nvme_path, ret);
        return ret;
    }

    inode = d_inode(path.dentry);
    if (!inode || !inode->i_cdev) {
        pr_err("gnb: %s is not a character device\n", nvme_path);
        path_put(&path);
        return -EINVAL;
    }

    /*
     * The nvme driver's char device open does:
     *   ctrl = container_of(inode->i_cdev, struct nvme_ctrl, cdev);
     * We do the same. This works because nvme_ctrl embeds a struct cdev.
     */
    ctrl = container_of(inode->i_cdev, struct nvme_ctrl, cdev);
    path_put(&path);

    if (!ctrl || !ctrl->admin_q) {
        pr_err("gnb: failed to get nvme_ctrl from %s\n", nvme_path);
        return -ENODEV;
    }

    /* Hold a reference to prevent ctrl from being freed */
    nvme_get_ctrl(ctrl);
    dev->ctrl = ctrl;

    /* Get PCI device */
    pdev = to_pci_dev(ctrl->dev);
    dev->pdev = pdev;

    /*
     * Map BAR0 independently from the nvme driver.
     * Multiple ioremap calls on the same physical region are safe.
     * We only need the doorbell region (offset 0x1000+), but mapping
     * the full BAR lets us read CAP and other registers too.
     */
    dev->bar_phys = pci_resource_start(pdev, 0);
    dev->bar_size = pci_resource_len(pdev, 0);
    dev->bar = ioremap(dev->bar_phys, dev->bar_size);
    if (!dev->bar) {
        pr_err("gnb: failed to map BAR0\n");
        ret = -ENOMEM;
        goto err_put_ctrl;
    }

    /* Read controller capabilities */
    cap = readl(dev->bar) | ((uint64_t)readl(dev->bar + 4) << 32);
    dev->max_queue_depth = NVME_CAP_MQES(cap) + 1;
    dev->doorbell_stride = 4 << NVME_CAP_DSTRD(cap);
    dev->db_offset = 0x1000;

    pr_info("gnb: attached to %s: MQES=%u DSTRD=%u stride=%u\n",
            nvme_path, dev->max_queue_depth,
            (uint32_t)NVME_CAP_DSTRD(cap), dev->doorbell_stride);

    /*
     * Determine available queue IDs.
     *
     * The nvme driver's ctrl->queue_count = admin(1) + I/O queues.
     * We start our queue IDs just above the driver's last one.
     *
     * The total number of allocated queues was negotiated by the nvme
     * driver via Set Features (Number of Queues). We query this to
     * know our upper bound.
     */
    dev->first_qid = ctrl->queue_count;

    /*
     * Query the allocated queue count via Get Features.
     * Feature ID 0x07 = Number of Queues.
     * CDW11 response: [31:16] = NCQA (0-based), [15:0] = NSQA (0-based)
     */
    {
        struct nvme_command cmd = {};
        union nvme_result result;

        cmd.features.opcode = nvme_admin_get_features;
        cmd.features.fid = cpu_to_le32(NVME_FEAT_NUM_QUEUES);

        ret = __nvme_submit_sync_cmd(ctrl->admin_q, &cmd, &result,
                                     NULL, 0, 0, 0);
        if (ret) {
            pr_warn("gnb: Get Features (Num Queues) failed: %d, "
                    "assuming max_qid = first_qid + 16\n", ret);
            num_queues_allocated = dev->first_qid + 16;
        } else {
            uint32_t nsqa = (le32_to_cpu(result.u32) & 0xFFFF) + 1;
            uint32_t ncqa = ((le32_to_cpu(result.u32) >> 16) & 0xFFFF) + 1;
            num_queues_allocated = min(nsqa, ncqa);
            pr_info("gnb: allocated queues: SQ=%u CQ=%u\n", nsqa, ncqa);
        }
    }

    dev->max_qid = num_queues_allocated;
    dev->next_qid = dev->first_qid;

    if (dev->first_qid >= dev->max_qid) {
        pr_err("gnb: no free queue IDs (driver uses %u, allocated %u)\n",
               dev->first_qid, dev->max_qid);
        ret = -ENOSPC;
        goto err_unmap;
    }

    pr_info("gnb: queue ID range: %u..%u (%u available)\n",
            dev->first_qid, dev->max_qid - 1,
            dev->max_qid - dev->first_qid);

    /* Get namespace info from ctrl (the nvme driver already identified it) */
    dev->nsid = 1;
    if (ctrl->vs >= NVME_VS(1, 0, 0)) {
        /*
         * Read ns info via Identify Namespace admin command.
         * We submit through the nvme driver's admin queue.
         */
        struct nvme_command cmd = {};
        union nvme_result result;
        void *id_buf;
        dma_addr_t id_dma;

        id_buf = dma_alloc_coherent(&pdev->dev, 4096, &id_dma, GFP_KERNEL);
        if (id_buf) {
            cmd.identify.opcode = nvme_admin_identify;
            cmd.identify.nsid = cpu_to_le32(dev->nsid);
            cmd.identify.cns = 0; /* Identify Namespace */
            cmd.identify.prp1 = cpu_to_le64(id_dma);

            ret = __nvme_submit_sync_cmd(ctrl->admin_q, &cmd, &result,
                                         NULL, 0, 0, 0);
            if (ret == 0) {
                uint8_t *id = id_buf;
                uint32_t flbas, lbads;

                memcpy(&dev->ns_size_blocks, id, 8);
                dev->ns_size_blocks = le64_to_cpu(dev->ns_size_blocks);

                flbas = id[26] & 0x0F;
                lbads = id[128 + flbas * 4 + 2];
                dev->lba_size = 1 << lbads;

                pr_info("gnb: ns%u: %llu blocks x %u bytes\n",
                        dev->nsid, dev->ns_size_blocks, dev->lba_size);
            } else {
                pr_warn("gnb: Identify Namespace failed: %d\n", ret);
                dev->ns_size_blocks = 0;
                dev->lba_size = 512;
            }
            dma_free_coherent(&pdev->dev, 4096, id_buf, id_dma);
        }
    }

    dev->attached = true;
    return 0;

err_unmap:
    iounmap(dev->bar);
    dev->bar = NULL;
err_put_ctrl:
    nvme_put_ctrl(ctrl);
    dev->ctrl = NULL;
    return ret;
}

/**
 * gnb_detach_ctrl - Detach from NVMe controller.
 */
void gnb_detach_ctrl(struct gnb_device *dev)
{
    int i;

    if (!dev->attached)
        return;

    /* Destroy all our I/O queues */
    for (i = 0; i < GNB_MAX_QUEUES; i++) {
        if (dev->queues[i].active)
            gnb_destroy_io_queue(dev, &dev->queues[i]);
    }

    /* Unpin GPU VRAM buffers */
    gnb_unreg_all_gpu_buffers(dev);

    /* Free host DMA data buffers */
    for (i = 0; i < GNB_MAX_BUFFERS; i++) {
        if (dev->buffers[i].active) {
            dma_free_coherent(&dev->pdev->dev,
                              dev->buffers[i].size,
                              dev->buffers[i].vaddr,
                              dev->buffers[i].dma_addr);
            dev->buffers[i].active = false;
        }
    }

    if (dev->bar) {
        iounmap(dev->bar);
        dev->bar = NULL;
    }

    if (dev->ctrl) {
        nvme_put_ctrl(dev->ctrl);
        dev->ctrl = NULL;
    }

    dev->attached = false;
    pr_info("gnb: detached\n");
}

/* ------------------------------------------------------------------ */
/* I/O Queue Creation / Deletion                                       */
/* ------------------------------------------------------------------ */

/**
 * gnb_create_io_queue - Create an I/O queue pair via admin commands.
 *
 * Allocates DMA-coherent memory for SQ and CQ, then submits
 * Create I/O CQ and Create I/O SQ admin commands through the
 * nvme driver's admin queue.
 */
int gnb_create_io_queue(struct gnb_device *dev, struct gnb_queue *q)
{
    struct nvme_command cmd;
    union nvme_result result;
    int ret;

    /* Allocate CQ memory */
    q->cq_vaddr = dma_alloc_coherent(&dev->pdev->dev,
        q->cq_depth * sizeof(struct gnb_nvme_cqe),
        &q->cq_dma, GFP_KERNEL);
    if (!q->cq_vaddr)
        return -ENOMEM;
    memset(q->cq_vaddr, 0, q->cq_depth * sizeof(struct gnb_nvme_cqe));

    /* Allocate SQ memory */
    q->sq_vaddr = dma_alloc_coherent(&dev->pdev->dev,
        q->sq_depth * sizeof(struct gnb_nvme_cmd),
        &q->sq_dma, GFP_KERNEL);
    if (!q->sq_vaddr) {
        ret = -ENOMEM;
        goto err_free_cq;
    }
    memset(q->sq_vaddr, 0, q->sq_depth * sizeof(struct gnb_nvme_cmd));

    /*
     * Create I/O CQ via admin command.
     * We submit through the nvme driver's admin queue, which handles
     * synchronization and command ID management for us.
     */
    memset(&cmd, 0, sizeof(cmd));
    cmd.create_cq.opcode = nvme_admin_create_cq;
    cmd.create_cq.prp1 = cpu_to_le64(q->cq_dma);
    cmd.create_cq.cqid = cpu_to_le16(q->qid);
    cmd.create_cq.qsize = cpu_to_le16(q->cq_depth - 1);
    cmd.create_cq.cq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
    cmd.create_cq.irq_vector = 0;  /* No interrupts, we poll */

    ret = __nvme_submit_sync_cmd(dev->ctrl->admin_q, &cmd, &result,
                                 NULL, 0, 0, 0);
    if (ret) {
        pr_err("gnb: Create CQ qid=%u failed: %d\n", q->qid, ret);
        goto err_free_sq;
    }

    /*
     * Create I/O SQ via admin command.
     * Paired with the CQ we just created (same QID).
     */
    memset(&cmd, 0, sizeof(cmd));
    cmd.create_sq.opcode = nvme_admin_create_sq;
    cmd.create_sq.prp1 = cpu_to_le64(q->sq_dma);
    cmd.create_sq.sqid = cpu_to_le16(q->qid);
    cmd.create_sq.qsize = cpu_to_le16(q->sq_depth - 1);
    cmd.create_sq.sq_flags = cpu_to_le16(NVME_SQ_PRIO_MEDIUM |
                                          NVME_QUEUE_PHYS_CONTIG);
    cmd.create_sq.cqid = cpu_to_le16(q->qid);

    ret = __nvme_submit_sync_cmd(dev->ctrl->admin_q, &cmd, &result,
                                 NULL, 0, 0, 0);
    if (ret) {
        pr_err("gnb: Create SQ qid=%u failed: %d\n", q->qid, ret);
        goto err_del_cq;
    }

    /* Compute doorbell offsets */
    q->sq_db_offset = gnb_sq_db_off(dev, q->qid);
    q->cq_db_offset = gnb_cq_db_off(dev, q->qid);

    q->active = true;

    pr_info("gnb: created queue pair qid=%u "
            "sq_depth=%u cq_depth=%u "
            "sq_dma=0x%llx cq_dma=0x%llx "
            "sq_db=0x%x cq_db=0x%x\n",
            q->qid, q->sq_depth, q->cq_depth,
            (unsigned long long)q->sq_dma,
            (unsigned long long)q->cq_dma,
            q->sq_db_offset, q->cq_db_offset);

    return 0;

err_del_cq:
    {
        struct nvme_command del = {};
        del.delete_queue.opcode = nvme_admin_delete_cq;
        del.delete_queue.qid = cpu_to_le16(q->qid);
        __nvme_submit_sync_cmd(dev->ctrl->admin_q, &del, NULL,
                               NULL, 0, 0, 0);
    }
err_free_sq:
    dma_free_coherent(&dev->pdev->dev,
        q->sq_depth * sizeof(struct gnb_nvme_cmd),
        q->sq_vaddr, q->sq_dma);
    q->sq_vaddr = NULL;
err_free_cq:
    dma_free_coherent(&dev->pdev->dev,
        q->cq_depth * sizeof(struct gnb_nvme_cqe),
        q->cq_vaddr, q->cq_dma);
    q->cq_vaddr = NULL;
    return ret;
}

/**
 * gnb_destroy_io_queue - Destroy an I/O queue pair.
 *
 * Sends Delete I/O SQ then Delete I/O CQ admin commands,
 * then frees the DMA memory.
 */
void gnb_destroy_io_queue(struct gnb_device *dev, struct gnb_queue *q)
{
    struct nvme_command cmd;

    if (!q->active)
        return;

    /* Delete SQ first (NVMe spec: SQ must be deleted before its CQ) */
    memset(&cmd, 0, sizeof(cmd));
    cmd.delete_queue.opcode = nvme_admin_delete_sq;
    cmd.delete_queue.qid = cpu_to_le16(q->qid);
    __nvme_submit_sync_cmd(dev->ctrl->admin_q, &cmd, NULL,
                           NULL, 0, 0, 0);

    /* Delete CQ */
    memset(&cmd, 0, sizeof(cmd));
    cmd.delete_queue.opcode = nvme_admin_delete_cq;
    cmd.delete_queue.qid = cpu_to_le16(q->qid);
    __nvme_submit_sync_cmd(dev->ctrl->admin_q, &cmd, NULL,
                           NULL, 0, 0, 0);

    /* Free DMA memory */
    if (q->sq_vaddr) {
        dma_free_coherent(&dev->pdev->dev,
            q->sq_depth * sizeof(struct gnb_nvme_cmd),
            q->sq_vaddr, q->sq_dma);
        q->sq_vaddr = NULL;
    }
    if (q->cq_vaddr) {
        dma_free_coherent(&dev->pdev->dev,
            q->cq_depth * sizeof(struct gnb_nvme_cqe),
            q->cq_vaddr, q->cq_dma);
        q->cq_vaddr = NULL;
    }

    q->active = false;
    pr_info("gnb: destroyed queue pair qid=%u\n", q->qid);
}
