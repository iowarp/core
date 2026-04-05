/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * gpu_nvme_bridge -- GPU-side NVMe command helpers
 *
 * Include this from CUDA .cu files. Provides device functions for
 * writing NVMe SQEs, ringing doorbells, and polling CQEs.
 *
 * Usage:
 *   #include "gnb_gpu.h"
 *
 * All functions are __device__ and operate on GPU-accessible memory:
 *   - SQ/CQ arrays in GPU VRAM (or mmap'd host DMA memory)
 *   - Doorbell registers mmap'd from NVMe BAR0
 *   - Data buffers in GPU VRAM with pre-registered bus addresses
 */
#ifndef GNB_GPU_H
#define GNB_GPU_H

#include <stdint.h>

/* NVMe I/O opcodes */
#define GNB_NVME_OP_WRITE   0x01
#define GNB_NVME_OP_READ    0x02

/* ------------------------------------------------------------------ */
/* NVMe command structures (same layout as gnb_uapi.h)                 */
/* ------------------------------------------------------------------ */

struct gnb_nvme_cmd {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t command_id;
    uint32_t nsid;
    uint64_t rsvd2;
    uint64_t metadata;
    uint64_t prp1;
    uint64_t prp2;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
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

/* ------------------------------------------------------------------ */
/* GPU Queue Context                                                   */
/* ------------------------------------------------------------------ */

/**
 * struct gnb_gpu_queue - Per-queue state for GPU-side NVMe I/O.
 *
 * Initialize from the output of GNB_IOC_CREATE_QUEUE and mmap.
 * Each GPU thread/warp that submits I/O needs its own instance
 * (or coordinated access with atomics).
 */
struct gnb_gpu_queue {
    volatile struct gnb_nvme_cmd *sq;    /* SQ entries (VRAM or mmap'd) */
    volatile struct gnb_nvme_cqe *cq;    /* CQ entries (VRAM or mmap'd) */
    volatile uint32_t *sq_doorbell;      /* SQ tail doorbell (MMIO)     */
    volatile uint32_t *cq_doorbell;      /* CQ head doorbell (MMIO)     */
    uint32_t sq_depth;
    uint32_t cq_depth;
    uint32_t sq_tail;                    /* Local SQ tail               */
    uint32_t cq_head;                    /* Local CQ head               */
    uint32_t cq_phase;                   /* Expected phase bit          */
    uint32_t nsid;                       /* Namespace ID                */
    uint32_t lba_shift;                  /* log2(lba_size) for addr calc */
};

/* ------------------------------------------------------------------ */
/* Initialization                                                      */
/* ------------------------------------------------------------------ */

/**
 * gnb_gpu_queue_init - Initialize a GPU queue context.
 *
 * @q            Queue context to initialize
 * @sq           SQ entries array (in VRAM or mmap'd host)
 * @cq           CQ entries array
 * @db_base      Base of mmap'd doorbell region
 * @sq_db_off    SQ doorbell offset (from GNB_IOC_CREATE_QUEUE)
 * @cq_db_off    CQ doorbell offset
 * @sq_depth     SQ depth
 * @cq_depth     CQ depth
 * @nsid         Namespace ID
 * @lba_size     Logical block size (512 or 4096)
 */
__device__ __host__ inline void gnb_gpu_queue_init(
    struct gnb_gpu_queue *q,
    volatile struct gnb_nvme_cmd *sq,
    volatile struct gnb_nvme_cqe *cq,
    volatile char *db_base,
    uint32_t sq_db_off,
    uint32_t cq_db_off,
    uint32_t sq_depth,
    uint32_t cq_depth,
    uint32_t nsid,
    uint32_t lba_size)
{
    q->sq = sq;
    q->cq = cq;
    q->sq_doorbell = (volatile uint32_t *)(db_base + sq_db_off);
    q->cq_doorbell = (volatile uint32_t *)(db_base + cq_db_off);
    q->sq_depth = sq_depth;
    q->cq_depth = cq_depth;
    q->sq_tail = 0;
    q->cq_head = 0;
    q->cq_phase = 1;  /* NVMe phase starts at 1 */
    q->nsid = nsid;

    /* Compute lba_shift */
    q->lba_shift = 0;
    uint32_t s = lba_size;
    while (s > 1) { s >>= 1; q->lba_shift++; }
}

/* ------------------------------------------------------------------ */
/* Command Submission                                                  */
/* ------------------------------------------------------------------ */

/**
 * gnb_gpu_submit_io - Submit an NVMe read or write command.
 *
 * @q            Queue context
 * @opcode       GNB_NVME_OP_READ or GNB_NVME_OP_WRITE
 * @command_id   Unique tag (returned in CQE for correlation)
 * @prp1         Bus address of data buffer (from GNB_IOC_REG_BUFFER)
 * @prp2         PRP2 bus address (0 for single-page I/O, PRP list for multi)
 * @lba          Starting logical block address
 * @num_blocks   Number of logical blocks minus 1 (NVMe convention)
 *
 * After this call, the NVMe controller will fetch the SQE via P2P DMA
 * and process the I/O. Use gnb_gpu_poll_cq to wait for completion.
 */
__device__ inline void gnb_gpu_submit_io(
    struct gnb_gpu_queue *q,
    uint8_t opcode,
    uint16_t command_id,
    uint64_t prp1,
    uint64_t prp2,
    uint64_t lba,
    uint32_t num_blocks)
{
    uint32_t idx = q->sq_tail;
    volatile struct gnb_nvme_cmd *cmd = &q->sq[idx];

    /* Build NVMe command */
    cmd->opcode = opcode;
    cmd->flags = 0;
    cmd->command_id = command_id;
    cmd->nsid = q->nsid;
    cmd->rsvd2 = 0;
    cmd->metadata = 0;
    cmd->prp1 = prp1;
    cmd->prp2 = prp2;
    cmd->cdw10 = (uint32_t)(lba & 0xFFFFFFFF);
    cmd->cdw11 = (uint32_t)(lba >> 32);
    cmd->cdw12 = num_blocks;
    cmd->cdw13 = 0;
    cmd->cdw14 = 0;
    cmd->cdw15 = 0;

    /* Ensure SQE writes are visible before ringing doorbell */
    __threadfence_system();

    /* Advance tail */
    q->sq_tail = (idx + 1) % q->sq_depth;

    /* Ring SQ tail doorbell (MMIO write to NVMe BAR0) */
    *(q->sq_doorbell) = q->sq_tail;
    __threadfence_system();
}

/* ------------------------------------------------------------------ */
/* Completion Polling                                                  */
/* ------------------------------------------------------------------ */

/**
 * gnb_gpu_poll_cq - Poll for one completion.
 *
 * Spins until a CQE with matching phase bit appears.
 * Returns the command_id from the CQE.
 *
 * @q            Queue context
 * @status_out   Output: NVMe status code (0 = success)
 *
 * After processing, rings the CQ head doorbell to release the CQE slot.
 */
__device__ inline uint16_t gnb_gpu_poll_cq(
    struct gnb_gpu_queue *q,
    int32_t *status_out)
{
    uint32_t idx = q->cq_head;
    volatile struct gnb_nvme_cqe *cqe = &q->cq[idx];
    uint16_t status;

    /* Spin until phase bit matches expected */
    do {
        __threadfence_system();
        status = cqe->status;
    } while ((status & 1) != q->cq_phase);

    uint16_t cid = cqe->command_id;
    *status_out = (int32_t)(status >> 1);  /* Status code (0 = success) */

    /* Advance CQ head */
    q->cq_head = (idx + 1) % q->cq_depth;
    if (q->cq_head == 0)
        q->cq_phase ^= 1;  /* Flip phase on wrap */

    /* Ring CQ head doorbell */
    *(q->cq_doorbell) = q->cq_head;
    __threadfence_system();

    return cid;
}

/**
 * gnb_gpu_try_poll_cq - Non-blocking completion poll.
 *
 * Returns true if a completion was found, false otherwise.
 */
__device__ inline bool gnb_gpu_try_poll_cq(
    struct gnb_gpu_queue *q,
    uint16_t *cid_out,
    int32_t *status_out)
{
    uint32_t idx = q->cq_head;
    volatile struct gnb_nvme_cqe *cqe = &q->cq[idx];
    uint16_t status;

    __threadfence_system();
    status = cqe->status;

    if ((status & 1) != q->cq_phase)
        return false;

    *cid_out = cqe->command_id;
    *status_out = (int32_t)(status >> 1);

    q->cq_head = (idx + 1) % q->cq_depth;
    if (q->cq_head == 0)
        q->cq_phase ^= 1;

    *(q->cq_doorbell) = q->cq_head;
    __threadfence_system();

    return true;
}

/* ------------------------------------------------------------------ */
/* Convenience: byte-addressed I/O                                     */
/* ------------------------------------------------------------------ */

/**
 * gnb_gpu_read - Read bytes from NVMe to a GPU buffer.
 *
 * @q            Queue context
 * @command_id   Unique tag
 * @buf_bus_addr Bus address of GPU data buffer
 * @byte_offset  Byte offset on device (must be LBA-aligned)
 * @byte_len     Bytes to read (must be multiple of LBA size)
 */
__device__ inline void gnb_gpu_read(
    struct gnb_gpu_queue *q,
    uint16_t command_id,
    uint64_t buf_bus_addr,
    uint64_t byte_offset,
    uint32_t byte_len)
{
    uint64_t lba = byte_offset >> q->lba_shift;
    uint32_t num_blocks = (byte_len >> q->lba_shift) - 1;

    gnb_gpu_submit_io(q, GNB_NVME_OP_READ, command_id,
                      buf_bus_addr, 0, lba, num_blocks);
}

/**
 * gnb_gpu_write - Write bytes from a GPU buffer to NVMe.
 */
__device__ inline void gnb_gpu_write(
    struct gnb_gpu_queue *q,
    uint16_t command_id,
    uint64_t buf_bus_addr,
    uint64_t byte_offset,
    uint32_t byte_len)
{
    uint64_t lba = byte_offset >> q->lba_shift;
    uint32_t num_blocks = (byte_len >> q->lba_shift) - 1;

    gnb_gpu_submit_io(q, GNB_NVME_OP_WRITE, command_id,
                      buf_bus_addr, 0, lba, num_blocks);
}

#endif /* GNB_GPU_H */
