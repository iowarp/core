/**
 * controller.cc -- NvmeController implementation wrapping gpu-nvme-bridge
 *
 * When BAM_ENABLE_NVME is defined (cmake option), this uses the
 * gpu-nvme-bridge kernel module via ioctls. Otherwise, all methods
 * return errors.
 */
#include <bam/controller.h>

#include <cstring>
#include <cstdio>

#ifdef BAM_ENABLE_NVME
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "gnb_uapi.h"
#endif

namespace bam {

NvmeController::NvmeController()
    : fd_(-1),
      attached_(false),
      lba_size_(0),
      nsid_(0),
      doorbell_stride_(0),
      max_queue_depth_(0),
      ns_size_blocks_(0),
      db_base_(nullptr),
      db_mmap_size_(0) {}

NvmeController::~NvmeController() {
  detach();
}

int NvmeController::attach(const std::string &nvme_dev) {
#ifdef BAM_ENABLE_NVME
  if (attached_) {
    fprintf(stderr, "bam::NvmeController: already attached\n");
    return -1;
  }

  fd_ = open("/dev/gnb", O_RDWR);
  if (fd_ < 0) {
    perror("bam::NvmeController: open /dev/gnb");
    return -1;
  }

  struct gnb_attach att;
  memset(&att, 0, sizeof(att));
  strncpy(att.nvme_dev, nvme_dev.c_str(), sizeof(att.nvme_dev) - 1);

  if (ioctl(fd_, GNB_IOC_ATTACH, &att) < 0) {
    perror("bam::NvmeController: GNB_IOC_ATTACH");
    close(fd_);
    fd_ = -1;
    return -1;
  }

  lba_size_ = att.lba_size;
  nsid_ = att.nsid;
  doorbell_stride_ = att.doorbell_stride;
  max_queue_depth_ = att.max_queue_depth;
  ns_size_blocks_ = att.ns_size_blocks;
  attached_ = true;

  return 0;
#else
  (void)nvme_dev;
  fprintf(stderr, "bam::NvmeController: NVMe support not compiled (BAM_ENABLE_NVME)\n");
  return -1;
#endif
}

void NvmeController::detach() {
#ifdef BAM_ENABLE_NVME
  if (!attached_) return;

  // Free all buffers
  for (auto &buf : buffers_) {
    if (buf.host_ptr) {
      munmap(buf.host_ptr, buf.size);
    }
    struct gnb_free_buffer fb;
    fb.mmap_offset = buf.mmap_offset;
    ioctl(fd_, GNB_IOC_FREE_BUF, &fb);
  }
  buffers_.clear();

  // Destroy all queues
  for (auto &qp : queues_) {
    if (qp.sq_ptr) {
      munmap(qp.sq_ptr, qp.sq_depth * 64);  // sizeof(gnb_nvme_cmd) = 64
    }
    if (qp.cq_ptr) {
      munmap(qp.cq_ptr, qp.cq_depth * 16);  // sizeof(gnb_nvme_cqe) = 16
    }
    struct gnb_destroy_queue dq;
    dq.qid = qp.qid;
    ioctl(fd_, GNB_IOC_DESTROY_QUEUE, &dq);
  }
  queues_.clear();

  // Unmap doorbells
  if (db_base_) {
    munmap(db_base_, db_mmap_size_);
    db_base_ = nullptr;
  }

  // Detach
  ioctl(fd_, GNB_IOC_DETACH);
  close(fd_);
  fd_ = -1;
  attached_ = false;
#endif
}

int NvmeController::create_queue(uint32_t sq_depth, uint32_t cq_depth) {
#ifdef BAM_ENABLE_NVME
  if (!attached_) return -1;

  struct gnb_create_queue cq;
  memset(&cq, 0, sizeof(cq));
  cq.sq_depth = sq_depth;
  cq.cq_depth = cq_depth;

  if (ioctl(fd_, GNB_IOC_CREATE_QUEUE, &cq) < 0) {
    perror("bam::NvmeController: GNB_IOC_CREATE_QUEUE");
    return -1;
  }

  QueuePairHost qp;
  qp.qid = cq.qid;
  qp.sq_depth = sq_depth;
  qp.cq_depth = cq_depth;
  qp.sq_db_offset = cq.sq_db_offset;
  qp.cq_db_offset = cq.cq_db_offset;
  qp.sq_mmap_offset = cq.sq_mmap_offset;
  qp.cq_mmap_offset = cq.cq_mmap_offset;

  // mmap SQ
  size_t sq_size = sq_depth * sizeof(struct gnb_nvme_cmd);
  qp.sq_ptr = mmap(nullptr, sq_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd_, cq.sq_mmap_offset);
  if (qp.sq_ptr == MAP_FAILED) {
    perror("bam::NvmeController: mmap SQ");
    return -1;
  }

  // mmap CQ
  size_t cq_size = cq_depth * sizeof(struct gnb_nvme_cqe);
  qp.cq_ptr = mmap(nullptr, cq_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd_, cq.cq_mmap_offset);
  if (qp.cq_ptr == MAP_FAILED) {
    perror("bam::NvmeController: mmap CQ");
    munmap(qp.sq_ptr, sq_size);
    return -1;
  }

  int idx = static_cast<int>(queues_.size());
  queues_.push_back(qp);
  return idx;
#else
  (void)sq_depth; (void)cq_depth;
  return -1;
#endif
}

int NvmeController::alloc_buffer(size_t size) {
#ifdef BAM_ENABLE_NVME
  if (!attached_) return -1;

  struct gnb_alloc_buffer ab;
  memset(&ab, 0, sizeof(ab));
  ab.size = size;

  if (ioctl(fd_, GNB_IOC_ALLOC_BUF, &ab) < 0) {
    perror("bam::NvmeController: GNB_IOC_ALLOC_BUF");
    return -1;
  }

  DmaBuffer buf;
  buf.mmap_offset = ab.mmap_offset;
  buf.bus_addr = ab.bus_addr;
  buf.size = size;

  buf.host_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd_, ab.mmap_offset);
  if (buf.host_ptr == MAP_FAILED) {
    perror("bam::NvmeController: mmap buffer");
    return -1;
  }

  int idx = static_cast<int>(buffers_.size());
  buffers_.push_back(buf);
  return idx;
#else
  (void)size;
  return -1;
#endif
}

void NvmeController::free_buffer(int buf_idx) {
#ifdef BAM_ENABLE_NVME
  if (!attached_ || buf_idx < 0 ||
      buf_idx >= static_cast<int>(buffers_.size())) return;

  auto &buf = buffers_[buf_idx];
  if (buf.host_ptr) {
    munmap(buf.host_ptr, buf.size);
    buf.host_ptr = nullptr;
  }

  struct gnb_free_buffer fb;
  fb.mmap_offset = buf.mmap_offset;
  ioctl(fd_, GNB_IOC_FREE_BUF, &fb);
#else
  (void)buf_idx;
#endif
}

int NvmeController::mmap_doorbells() {
#ifdef BAM_ENABLE_NVME
  if (!attached_ || db_base_) return (db_base_ ? 0 : -1);

  // Doorbell region is at offset 0 in mmap space
  db_mmap_size_ = 4096;  // Minimum page size
  db_base_ = mmap(nullptr, db_mmap_size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd_, GNB_MMAP_DOORBELLS);
  if (db_base_ == MAP_FAILED) {
    perror("bam::NvmeController: mmap doorbells");
    db_base_ = nullptr;
    return -1;
  }
  return 0;
#else
  return -1;
#endif
}

}  // namespace bam
