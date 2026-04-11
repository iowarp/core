/**
 * bam/bam.h -- Convenience header for BaM (Big Accelerator Memory)
 *
 * Include this single header from host-side code to get access to
 * the full BaM API: page cache, array abstraction, and NVMe controller.
 *
 * For GPU kernel code (.cu files), include bam/array.cuh directly.
 */
#ifndef BAM_BAM_H
#define BAM_BAM_H

#include <bam/types.h>
#include <bam/controller.h>
#include <bam/page_cache_host.h>

#endif  // BAM_BAM_H
