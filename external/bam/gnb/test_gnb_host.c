// SPDX-License-Identifier: GPL-2.0
/*
 * test_gnb_host.c -- Host-memory test for cooperative GNB
 *
 * Tests the full path: attach → create queue → alloc buffer →
 * mmap everything → write NVMe SQEs → ring doorbells → poll CQEs →
 * verify data.
 *
 * No GPU required. Validates NVMe command submission, doorbell
 * access, and completion polling using DMA-coherent host memory.
 *
 * Build: gcc -O2 -o test_gnb_host test_gnb_host.c
 * Usage: sudo ./test_gnb_host [/dev/nvme0]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "gnb_uapi.h"

#define QUEUE_DEPTH  64
#define IO_SIZE      4096     /* Must be >= lba_size */
#define NUM_IOS      8

static void die(const char *msg)
{
    perror(msg);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* NVMe CQ polling (phase-bit based)                                   */
/* ------------------------------------------------------------------ */

static uint16_t poll_completion(
    volatile struct gnb_nvme_cqe *cq,
    volatile uint32_t *cq_doorbell,
    uint32_t cq_depth,
    uint32_t *head,
    uint32_t *phase,
    int32_t *status_out)
{
    volatile struct gnb_nvme_cqe *cqe = &cq[*head];
    uint16_t status;

    /* Spin until phase bit matches */
    int spins = 0;
    do {
        __sync_synchronize();
        status = cqe->status;
        if (++spins > 100000000) {
            fprintf(stderr, "  CQ poll timeout at head=%u phase=%u "
                    "status=0x%04x\n", *head, *phase, status);
            *status_out = -1;
            return 0xFFFF;
        }
    } while ((status & 1) != *phase);

    uint16_t cid = cqe->command_id;
    *status_out = (int32_t)(status >> 1);

    *head = (*head + 1) % cq_depth;
    if (*head == 0)
        *phase ^= 1;

    __sync_synchronize();
    *cq_doorbell = *head;

    return cid;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    const char *nvme_dev = argc > 1 ? argv[1] : "/dev/nvme0";
    const char *gnb_dev = "/dev/gnb";
    int fd, ret;

    printf("=== GNB Cooperative Test ===\n");
    printf("NVMe controller: %s\n", nvme_dev);
    printf("GNB device:      %s\n\n", gnb_dev);

    /* Open GNB */
    fd = open(gnb_dev, O_RDWR);
    if (fd < 0) die("open /dev/gnb");

    /* ---- Step 1: Attach to NVMe controller ---- */
    struct gnb_attach att = {};
    strncpy(att.nvme_dev, nvme_dev, sizeof(att.nvme_dev) - 1);

    ret = ioctl(fd, GNB_IOC_ATTACH, &att);
    if (ret < 0) die("GNB_IOC_ATTACH");

    printf("Attached to %s:\n", nvme_dev);
    printf("  NS size:    %lu blocks x %u bytes\n",
           (unsigned long)att.ns_size_blocks, att.lba_size);
    printf("  Max QD:     %u\n", att.max_queue_depth);
    printf("  DB stride:  %u bytes\n", att.doorbell_stride);
    printf("  Queue IDs:  %u..%u\n", att.first_qid, att.max_qid - 1);
    printf("\n");

    /* ---- Step 2: Create I/O queue pair ---- */
    struct gnb_create_queue cqp = {
        .sq_depth = QUEUE_DEPTH,
        .cq_depth = QUEUE_DEPTH,
    };

    ret = ioctl(fd, GNB_IOC_CREATE_QUEUE, &cqp);
    if (ret < 0) die("GNB_IOC_CREATE_QUEUE");

    printf("Queue created: qid=%u\n", cqp.qid);
    printf("  SQ doorbell offset: 0x%x\n", cqp.sq_db_offset);
    printf("  CQ doorbell offset: 0x%x\n", cqp.cq_db_offset);
    printf("  SQ mmap offset:     0x%lx\n", (unsigned long)cqp.sq_mmap_offset);
    printf("  CQ mmap offset:     0x%lx\n", (unsigned long)cqp.cq_mmap_offset);
    printf("\n");

    /* ---- Step 3: Allocate data buffer ---- */
    struct gnb_alloc_buffer abuf = {
        .size = IO_SIZE * NUM_IOS,
    };

    ret = ioctl(fd, GNB_IOC_ALLOC_BUF, &abuf);
    if (ret < 0) die("GNB_IOC_ALLOC_BUF");

    printf("Buffer allocated: bus_addr=0x%lx mmap=0x%lx size=%lu\n\n",
           (unsigned long)abuf.bus_addr,
           (unsigned long)abuf.mmap_offset,
           (unsigned long)abuf.size);

    /* ---- Step 4: mmap doorbells ---- */
    /*
     * The doorbell region starts at BAR0 + 0x1000.
     * We mmap enough to cover our queue's doorbells.
     */
    size_t db_size = 4096;  /* One page covers many doorbells */
    volatile char *db_base = mmap(NULL, db_size,
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd,
                                   GNB_MMAP_DOORBELLS);
    if (db_base == MAP_FAILED) die("mmap doorbells");

    /*
     * Doorbell offsets are relative to BAR0 + 0x1000.
     * Our mmap starts at 0x1000, so subtract db_offset base.
     */
    volatile uint32_t *sq_db = (volatile uint32_t *)
        (db_base + cqp.sq_db_offset - 0x1000);
    volatile uint32_t *cq_db = (volatile uint32_t *)
        (db_base + cqp.cq_db_offset - 0x1000);

    printf("Doorbells mmap'd at %p (SQ db=%p, CQ db=%p)\n",
           db_base, sq_db, cq_db);

    /* ---- Step 5: mmap SQ and CQ ---- */
    size_t sq_size = PAGE_ALIGN(QUEUE_DEPTH * sizeof(struct gnb_nvme_cmd));
    volatile struct gnb_nvme_cmd *sq = mmap(NULL, sq_size,
        PROT_READ | PROT_WRITE, MAP_SHARED, fd, cqp.sq_mmap_offset);
    if (sq == MAP_FAILED) die("mmap SQ");

    size_t cq_size = PAGE_ALIGN(QUEUE_DEPTH * sizeof(struct gnb_nvme_cqe));
    volatile struct gnb_nvme_cqe *cq = mmap(NULL, cq_size,
        PROT_READ | PROT_WRITE, MAP_SHARED, fd, cqp.cq_mmap_offset);
    if (cq == MAP_FAILED) die("mmap CQ");

    printf("SQ mmap'd at %p, CQ mmap'd at %p\n", sq, cq);

    /* ---- Step 6: mmap data buffer ---- */
    char *data = mmap(NULL, abuf.size,
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                      abuf.mmap_offset);
    if (data == MAP_FAILED) die("mmap data");

    printf("Data buffer mmap'd at %p\n\n", data);

    /* ---- Compute LBA shift ---- */
    uint32_t lba_shift = 0;
    { uint32_t s = att.lba_size; while (s > 1) { s >>= 1; lba_shift++; } }

    /* ================================================================ */
    /* Phase 1: Write                                                    */
    /* ================================================================ */
    printf("--- Phase 1: Write %d x %d bytes ---\n", NUM_IOS, IO_SIZE);

    /* Fill data buffer with pattern */
    for (int i = 0; i < IO_SIZE * NUM_IOS; i++)
        data[i] = (char)(i & 0xFF);
    __sync_synchronize();

    uint32_t sq_tail = 0;
    uint32_t cq_head = 0;
    uint32_t cq_phase = 1;

    /* Submit writes */
    for (int i = 0; i < NUM_IOS; i++) {
        uint32_t idx = sq_tail;
        volatile struct gnb_nvme_cmd *cmd = &sq[idx];

        uint64_t buf_bus = abuf.bus_addr + i * IO_SIZE;
        uint64_t lba = (uint64_t)(i * IO_SIZE) >> lba_shift;
        uint32_t nlb = (IO_SIZE >> lba_shift) - 1;

        cmd->opcode = GNB_NVME_OP_WRITE;
        cmd->flags = 0;
        cmd->command_id = (uint16_t)(0x100 + i);
        cmd->nsid = att.nsid;
        cmd->rsvd2 = 0;
        cmd->metadata = 0;
        cmd->prp1 = buf_bus;
        cmd->prp2 = 0;
        cmd->cdw10 = (uint32_t)(lba & 0xFFFFFFFF);
        cmd->cdw11 = (uint32_t)(lba >> 32);
        cmd->cdw12 = nlb;
        cmd->cdw13 = 0;
        cmd->cdw14 = 0;
        cmd->cdw15 = 0;

        __sync_synchronize();
        sq_tail = (sq_tail + 1) % QUEUE_DEPTH;
        *sq_db = sq_tail;
    }

    /* Poll completions */
    for (int i = 0; i < NUM_IOS; i++) {
        int32_t st;
        uint16_t cid = poll_completion(cq, cq_db, QUEUE_DEPTH,
                                        &cq_head, &cq_phase, &st);
        printf("  CQE: cid=0x%04x status=%d\n", cid, st);
        if (st != 0)
            fprintf(stderr, "  WRITE FAILED cid=0x%04x\n", cid);
    }

    /* ================================================================ */
    /* Phase 2: Read back                                                */
    /* ================================================================ */
    printf("\n--- Phase 2: Read %d x %d bytes ---\n", NUM_IOS, IO_SIZE);

    memset(data, 0, IO_SIZE * NUM_IOS);
    __sync_synchronize();

    for (int i = 0; i < NUM_IOS; i++) {
        uint32_t idx = sq_tail;
        volatile struct gnb_nvme_cmd *cmd = &sq[idx];

        uint64_t buf_bus = abuf.bus_addr + i * IO_SIZE;
        uint64_t lba = (uint64_t)(i * IO_SIZE) >> lba_shift;
        uint32_t nlb = (IO_SIZE >> lba_shift) - 1;

        cmd->opcode = GNB_NVME_OP_READ;
        cmd->flags = 0;
        cmd->command_id = (uint16_t)(0x200 + i);
        cmd->nsid = att.nsid;
        cmd->rsvd2 = 0;
        cmd->metadata = 0;
        cmd->prp1 = buf_bus;
        cmd->prp2 = 0;
        cmd->cdw10 = (uint32_t)(lba & 0xFFFFFFFF);
        cmd->cdw11 = (uint32_t)(lba >> 32);
        cmd->cdw12 = nlb;
        cmd->cdw13 = 0;
        cmd->cdw14 = 0;
        cmd->cdw15 = 0;

        __sync_synchronize();
        sq_tail = (sq_tail + 1) % QUEUE_DEPTH;
        *sq_db = sq_tail;
    }

    for (int i = 0; i < NUM_IOS; i++) {
        int32_t st;
        uint16_t cid = poll_completion(cq, cq_db, QUEUE_DEPTH,
                                        &cq_head, &cq_phase, &st);
        printf("  CQE: cid=0x%04x status=%d\n", cid, st);
        if (st != 0)
            fprintf(stderr, "  READ FAILED cid=0x%04x\n", cid);
    }

    /* ================================================================ */
    /* Phase 3: Verify                                                   */
    /* ================================================================ */
    printf("\n--- Verifying data ---\n");
    __sync_synchronize();
    {
        int errors = 0;
        for (int i = 0; i < IO_SIZE * NUM_IOS; i++) {
            if (data[i] != (char)(i & 0xFF)) {
                if (errors < 10)
                    fprintf(stderr, "  Mismatch byte %d: "
                            "expected 0x%02x got 0x%02x\n",
                            i, (unsigned char)(i & 0xFF),
                            (unsigned char)data[i]);
                errors++;
            }
        }
        printf("  %s (%d errors)\n",
               errors == 0 ? "PASSED" : "FAILED", errors);
    }

    /* ---- Cleanup ---- */
    struct gnb_free_buffer fb = { .mmap_offset = abuf.mmap_offset };
    ioctl(fd, GNB_IOC_FREE_BUF, &fb);

    struct gnb_destroy_queue dq = { .qid = cqp.qid };
    ioctl(fd, GNB_IOC_DESTROY_QUEUE, &dq);

    ioctl(fd, GNB_IOC_DETACH);

    munmap(data, abuf.size);
    munmap((void *)cq, cq_size);
    munmap((void *)sq, sq_size);
    munmap((void *)db_base, db_size);
    close(fd);

    printf("\n=== Test complete ===\n");
    return 0;
}

/* Helper for PAGE_ALIGN in userspace */
#ifndef PAGE_ALIGN
#define PAGE_ALIGN(x) (((x) + 4095) & ~4095UL)
#endif
