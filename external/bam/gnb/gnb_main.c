// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_nvme_bridge -- module init/exit (cooperative mode)
 *
 * Creates a single /dev/gnb0 misc device. NOT a PCI driver.
 * The nvme driver continues to own the NVMe PCI device.
 */

#include <linux/module.h>
#include <linux/init.h>

#include "gnb_common.h"

static struct gnb_device gnb_dev;

static int __init gnb_init(void)
{
    int ret;

    memset(&gnb_dev, 0, sizeof(gnb_dev));
    mutex_init(&gnb_dev.lock);

    gnb_dev.misc.minor = MISC_DYNAMIC_MINOR;
    gnb_dev.misc.name = GNB_NAME;
    gnb_dev.misc.fops = &gnb_fops;

    ret = misc_register(&gnb_dev.misc);
    if (ret) {
        pr_err("gnb: misc_register failed: %d\n", ret);
        return ret;
    }

    pr_info("gnb: /dev/%s registered (cooperative NVMe bridge)\n",
            GNB_NAME);
    return 0;
}

static void __exit gnb_exit(void)
{
    mutex_lock(&gnb_dev.lock);
    if (gnb_dev.attached)
        gnb_detach_ctrl(&gnb_dev);
    mutex_unlock(&gnb_dev.lock);

    misc_deregister(&gnb_dev.misc);
    pr_info("gnb: unloaded\n");
}

module_init(gnb_init);
module_exit(gnb_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("IOWarp");
MODULE_DESCRIPTION("GPU-direct NVMe I/O bridge (cooperative with nvme driver)");
MODULE_VERSION("0.2");
MODULE_SOFTDEP("pre: nvme");
