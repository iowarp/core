// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_nvme_bridge -- file operations (cooperative mode)
 *
 * Character device for /dev/gnb0: ioctl + mmap.
 */

#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/dma-mapping.h>

#include "gnb_common.h"

/* ------------------------------------------------------------------ */
/* GNB_IOC_ATTACH                                                      */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_attach(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_attach params;
    int ret;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;
    params.nvme_dev[sizeof(params.nvme_dev) - 1] = '\0';

    mutex_lock(&dev->lock);
    if (dev->attached) {
        mutex_unlock(&dev->lock);
        return -EBUSY;
    }

    ret = gnb_attach_ctrl(dev, params.nvme_dev);
    if (ret) {
        mutex_unlock(&dev->lock);
        return ret;
    }

    /* Fill output */
    params.ns_size_blocks = dev->ns_size_blocks;
    params.lba_size = dev->lba_size;
    params.max_queue_depth = dev->max_queue_depth;
    params.doorbell_stride = dev->doorbell_stride;
    params.nsid = dev->nsid;
    params.first_qid = dev->first_qid;
    params.max_qid = dev->max_qid;

    mutex_unlock(&dev->lock);

    if (copy_to_user((void __user *)arg, &params, sizeof(params)))
        return -EFAULT;

    return 0;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_DETACH                                                      */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_detach(struct gnb_device *dev)
{
    mutex_lock(&dev->lock);
    gnb_detach_ctrl(dev);
    mutex_unlock(&dev->lock);
    return 0;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_CREATE_QUEUE                                                */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_create_queue(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_create_queue params;
    struct gnb_queue *q;
    uint32_t qid;
    int idx, ret;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    if (!is_power_of_2(params.sq_depth) || !is_power_of_2(params.cq_depth))
        return -EINVAL;
    if (params.sq_depth > GNB_MAX_QUEUE_DEPTH ||
        params.cq_depth > GNB_MAX_QUEUE_DEPTH)
        return -EINVAL;

    mutex_lock(&dev->lock);
    if (!dev->attached) {
        mutex_unlock(&dev->lock);
        return -ENODEV;
    }

    /* Allocate queue ID */
    qid = dev->next_qid;
    if (qid >= dev->max_qid) {
        mutex_unlock(&dev->lock);
        pr_err("gnb: no more queue IDs available\n");
        return -ENOSPC;
    }

    /* Find free queue slot */
    idx = -1;
    for (int i = 0; i < GNB_MAX_QUEUES; i++) {
        if (!dev->queues[i].active) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        mutex_unlock(&dev->lock);
        return -ENOSPC;
    }

    q = &dev->queues[idx];
    memset(q, 0, sizeof(*q));
    q->qid = qid;
    q->sq_depth = min(params.sq_depth, dev->max_queue_depth);
    q->cq_depth = min(params.cq_depth, dev->max_queue_depth);

    /* Compute mmap offsets */
    q->sq_mmap_offset = GNB_MMAP_QUEUE_BASE + (uint64_t)idx * GNB_MMAP_QUEUE_STRIDE;
    q->cq_mmap_offset = q->sq_mmap_offset +
        PAGE_ALIGN(q->sq_depth * sizeof(struct gnb_nvme_cmd));

    ret = gnb_create_io_queue(dev, q);
    if (ret) {
        mutex_unlock(&dev->lock);
        return ret;
    }

    dev->next_qid = qid + 1;
    mutex_unlock(&dev->lock);

    /* Fill output */
    params.qid = qid;
    params.sq_db_offset = q->sq_db_offset;
    params.cq_db_offset = q->cq_db_offset;
    params.sq_mmap_offset = q->sq_mmap_offset;
    params.cq_mmap_offset = q->cq_mmap_offset;

    if (copy_to_user((void __user *)arg, &params, sizeof(params)))
        return -EFAULT;

    return 0;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_DESTROY_QUEUE                                               */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_destroy_queue(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_destroy_queue params;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    mutex_lock(&dev->lock);

    for (int i = 0; i < GNB_MAX_QUEUES; i++) {
        if (dev->queues[i].active && dev->queues[i].qid == params.qid) {
            gnb_destroy_io_queue(dev, &dev->queues[i]);
            mutex_unlock(&dev->lock);
            return 0;
        }
    }

    mutex_unlock(&dev->lock);
    return -ENOENT;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_ALLOC_BUF                                                   */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_alloc_buf(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_alloc_buffer params;
    struct gnb_buffer *buf;
    int idx;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    params.size = PAGE_ALIGN(params.size);
    if (params.size == 0 || params.size > (64ULL << 20)) /* 64MB max */
        return -EINVAL;

    mutex_lock(&dev->lock);
    if (!dev->attached) {
        mutex_unlock(&dev->lock);
        return -ENODEV;
    }

    idx = -1;
    for (int i = 0; i < GNB_MAX_BUFFERS; i++) {
        if (!dev->buffers[i].active) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        mutex_unlock(&dev->lock);
        return -ENOSPC;
    }

    buf = &dev->buffers[idx];
    buf->size = params.size;
    buf->mmap_offset = GNB_MMAP_BUF_BASE +
                       (uint64_t)idx * GNB_MMAP_BUF_STRIDE;

    buf->vaddr = dma_alloc_coherent(&dev->pdev->dev, buf->size,
                                     &buf->dma_addr, GFP_KERNEL);
    if (!buf->vaddr) {
        mutex_unlock(&dev->lock);
        return -ENOMEM;
    }

    buf->active = true;
    mutex_unlock(&dev->lock);

    params.mmap_offset = buf->mmap_offset;
    params.bus_addr = buf->dma_addr;

    pr_info("gnb: allocated buffer %d: size=%llu dma=0x%llx\n",
            idx, buf->size, (unsigned long long)buf->dma_addr);

    if (copy_to_user((void __user *)arg, &params, sizeof(params)))
        return -EFAULT;

    return 0;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_FREE_BUF                                                    */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_free_buf(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_free_buffer params;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    mutex_lock(&dev->lock);

    for (int i = 0; i < GNB_MAX_BUFFERS; i++) {
        if (dev->buffers[i].active &&
            dev->buffers[i].mmap_offset == params.mmap_offset) {
            dma_free_coherent(&dev->pdev->dev,
                              dev->buffers[i].size,
                              dev->buffers[i].vaddr,
                              dev->buffers[i].dma_addr);
            dev->buffers[i].active = false;
            mutex_unlock(&dev->lock);
            return 0;
        }
    }

    mutex_unlock(&dev->lock);
    return -ENOENT;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_REG_GPU_BUF                                                 */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_reg_gpu_buf(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_reg_gpu_buffer params;
    struct gnb_gpu_buffer *gbuf;
    int idx, ret;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    /* GPU P2P pages are 64KB-aligned */
    if (params.gpu_va == 0 || params.size == 0)
        return -EINVAL;
    if (params.gpu_va & (GNB_GPU_PAGE_SIZE - 1))
        return -EINVAL;
    if (params.size & (GNB_GPU_PAGE_SIZE - 1))
        return -EINVAL;

    mutex_lock(&dev->lock);
    if (!dev->attached) {
        mutex_unlock(&dev->lock);
        return -ENODEV;
    }

    /* Find free slot */
    idx = -1;
    for (int i = 0; i < GNB_MAX_GPU_BUFFERS; i++) {
        if (!dev->gpu_buffers[i].active) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        mutex_unlock(&dev->lock);
        return -ENOSPC;
    }

    gbuf = &dev->gpu_buffers[idx];
    memset(gbuf, 0, sizeof(*gbuf));
    gbuf->gpu_va = params.gpu_va;
    gbuf->size = params.size;
    gbuf->buf_id = dev->next_gpu_buf_id++;

    ret = gnb_reg_gpu_buffer(dev, gbuf);
    mutex_unlock(&dev->lock);

    if (ret)
        return ret;

    params.buf_id = gbuf->buf_id;
    params.page_size = gbuf->page_size;
    params.num_pages = gbuf->num_pages;

    if (copy_to_user((void __user *)arg, &params, sizeof(params)))
        return -EFAULT;

    return 0;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_UNREG_GPU_BUF                                               */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_unreg_gpu_buf(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_unreg_gpu_buffer params;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    mutex_lock(&dev->lock);
    for (int i = 0; i < GNB_MAX_GPU_BUFFERS; i++) {
        if (dev->gpu_buffers[i].active &&
            dev->gpu_buffers[i].buf_id == params.buf_id) {
            gnb_unreg_gpu_buffer(dev, &dev->gpu_buffers[i]);
            mutex_unlock(&dev->lock);
            return 0;
        }
    }
    mutex_unlock(&dev->lock);
    return -ENOENT;
}

/* ------------------------------------------------------------------ */
/* GNB_IOC_GET_GPU_PAGES                                               */
/* ------------------------------------------------------------------ */

static int gnb_ioctl_get_gpu_pages(struct gnb_device *dev, unsigned long arg)
{
    struct gnb_get_gpu_pages params;
    struct gnb_gpu_buffer *gbuf = NULL;
    uint32_t copy_count;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    mutex_lock(&dev->lock);
    for (int i = 0; i < GNB_MAX_GPU_BUFFERS; i++) {
        if (dev->gpu_buffers[i].active &&
            dev->gpu_buffers[i].buf_id == params.buf_id) {
            gbuf = &dev->gpu_buffers[i];
            break;
        }
    }

    if (!gbuf) {
        mutex_unlock(&dev->lock);
        return -ENOENT;
    }

    copy_count = min(params.max_pages, gbuf->num_pages);

    if (copy_count > 0 && params.bus_addrs_ptr) {
        if (copy_to_user((void __user *)params.bus_addrs_ptr,
                         gbuf->bus_addrs,
                         copy_count * sizeof(uint64_t))) {
            mutex_unlock(&dev->lock);
            return -EFAULT;
        }
    }

    params.num_pages = gbuf->num_pages;
    params.page_size = gbuf->page_size;
    mutex_unlock(&dev->lock);

    if (copy_to_user((void __user *)arg, &params, sizeof(params)))
        return -EFAULT;

    return 0;
}

/* ------------------------------------------------------------------ */
/* ioctl dispatch                                                      */
/* ------------------------------------------------------------------ */

static long gnb_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct gnb_device *dev = file->private_data;

    switch (cmd) {
    case GNB_IOC_ATTACH:
        return gnb_ioctl_attach(dev, arg);
    case GNB_IOC_DETACH:
        return gnb_ioctl_detach(dev);
    case GNB_IOC_CREATE_QUEUE:
        return gnb_ioctl_create_queue(dev, arg);
    case GNB_IOC_DESTROY_QUEUE:
        return gnb_ioctl_destroy_queue(dev, arg);
    case GNB_IOC_ALLOC_BUF:
        return gnb_ioctl_alloc_buf(dev, arg);
    case GNB_IOC_FREE_BUF:
        return gnb_ioctl_free_buf(dev, arg);
    case GNB_IOC_REG_GPU_BUF:
        return gnb_ioctl_reg_gpu_buf(dev, arg);
    case GNB_IOC_UNREG_GPU_BUF:
        return gnb_ioctl_unreg_gpu_buf(dev, arg);
    case GNB_IOC_GET_GPU_PAGES:
        return gnb_ioctl_get_gpu_pages(dev, arg);
    default:
        return -ENOTTY;
    }
}

/* ------------------------------------------------------------------ */
/* mmap                                                                */
/* ------------------------------------------------------------------ */

static int gnb_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct gnb_device *dev = file->private_data;
    uint64_t offset = (uint64_t)vma->vm_pgoff << PAGE_SHIFT;
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long pfn;

    mutex_lock(&dev->lock);
    if (!dev->attached) {
        mutex_unlock(&dev->lock);
        return -ENODEV;
    }

    /* Doorbell region */
    if (offset == GNB_MMAP_DOORBELLS) {
        pfn = (dev->bar_phys + dev->db_offset) >> PAGE_SHIFT;
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);

        if (io_remap_pfn_range(vma, vma->vm_start, pfn, size,
                               vma->vm_page_prot)) {
            mutex_unlock(&dev->lock);
            return -EAGAIN;
        }
        mutex_unlock(&dev->lock);
        return 0;
    }

    /* Queue memory */
    if (offset >= GNB_MMAP_QUEUE_BASE && offset < GNB_MMAP_BUF_BASE) {
        /* Find matching queue by offset */
        for (int i = 0; i < GNB_MAX_QUEUES; i++) {
            struct gnb_queue *q = &dev->queues[i];
            if (!q->active)
                continue;

            if (offset == q->sq_mmap_offset && q->sq_vaddr) {
                pfn = virt_to_phys(q->sq_vaddr) >> PAGE_SHIFT;
                vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
                vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP);
                if (remap_pfn_range(vma, vma->vm_start, pfn, size,
                                    vma->vm_page_prot)) {
                    mutex_unlock(&dev->lock);
                    return -EAGAIN;
                }
                mutex_unlock(&dev->lock);
                return 0;
            }

            if (offset == q->cq_mmap_offset && q->cq_vaddr) {
                pfn = virt_to_phys(q->cq_vaddr) >> PAGE_SHIFT;
                vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
                vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP);
                if (remap_pfn_range(vma, vma->vm_start, pfn, size,
                                    vma->vm_page_prot)) {
                    mutex_unlock(&dev->lock);
                    return -EAGAIN;
                }
                mutex_unlock(&dev->lock);
                return 0;
            }
        }
    }

    /* Data buffer */
    if (offset >= GNB_MMAP_BUF_BASE) {
        for (int i = 0; i < GNB_MAX_BUFFERS; i++) {
            struct gnb_buffer *buf = &dev->buffers[i];
            if (buf->active && buf->mmap_offset == offset) {
                pfn = virt_to_phys(buf->vaddr) >> PAGE_SHIFT;
                vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
                vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP);
                if (remap_pfn_range(vma, vma->vm_start, pfn, size,
                                    vma->vm_page_prot)) {
                    mutex_unlock(&dev->lock);
                    return -EAGAIN;
                }
                mutex_unlock(&dev->lock);
                return 0;
            }
        }
    }

    mutex_unlock(&dev->lock);
    return -EINVAL;
}

/* ------------------------------------------------------------------ */
/* open / release                                                      */
/* ------------------------------------------------------------------ */

static int gnb_open(struct inode *inode, struct file *file)
{
    struct gnb_device *dev = container_of(file->private_data,
                                          struct gnb_device, misc);
    file->private_data = dev;
    return 0;
}

static int gnb_release(struct inode *inode, struct file *file)
{
    struct gnb_device *dev = file->private_data;

    mutex_lock(&dev->lock);
    if (dev->attached)
        gnb_detach_ctrl(dev);
    mutex_unlock(&dev->lock);

    return 0;
}

const struct file_operations gnb_fops = {
    .owner          = THIS_MODULE,
    .open           = gnb_open,
    .release        = gnb_release,
    .unlocked_ioctl = gnb_ioctl,
    .compat_ioctl   = compat_ptr_ioctl,
    .mmap           = gnb_mmap,
};
