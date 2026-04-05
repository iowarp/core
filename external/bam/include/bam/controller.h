/**
 * bam/controller.h -- NVMe controller abstraction wrapping gpu-nvme-bridge
 *
 * Host-side class that manages the lifecycle of the gpu-nvme-bridge
 * connection: attach, queue creation, buffer registration, and mmap.
 *
 * When BAM_ENABLE_NVME is not defined, this file provides a stub
 * implementation that returns errors for all NVMe operations.
 */
#ifndef BAM_CONTROLLER_H
#define BAM_CONTROLLER_H

#include <bam/types.h>
#include <string>
#include <vector>
#include <memory>

namespace bam {

struct QueuePairHost {
  uint32_t qid;
  uint32_t sq_depth;
  uint32_t cq_depth;
  void    *sq_ptr;           // mmap'd SQ
  void    *cq_ptr;           // mmap'd CQ
  uint32_t sq_db_offset;     // Doorbell offsets within BAR0 mmap
  uint32_t cq_db_offset;
  uint64_t sq_mmap_offset;
  uint64_t cq_mmap_offset;
};

struct DmaBuffer {
  uint64_t mmap_offset;
  uint64_t bus_addr;         // PCIe bus address (for NVMe PRP entries)
  void    *host_ptr;         // mmap'd pointer
  size_t   size;
};

/**
 * NvmeController -- Manages gpu-nvme-bridge file descriptor and resources.
 *
 * Lifecycle:
 *   1. Construct with device path
 *   2. attach() — opens /dev/gnb and issues GNB_IOC_ATTACH
 *   3. create_queue() — creates I/O queue pairs
 *   4. alloc_buffer() — allocates DMA data buffers
 *   5. mmap_doorbells() — maps BAR0 doorbell region
 *   6. Use queues and buffers for I/O
 *   7. Destructor cleans up
 */
class NvmeController {
 public:
  NvmeController();
  ~NvmeController();

  /** Attach to NVMe controller via gpu-nvme-bridge. */
  int attach(const std::string &nvme_dev);

  /** Detach and release all resources. */
  void detach();

  /** Create an I/O queue pair. Returns queue index on success, -1 on error. */
  int create_queue(uint32_t sq_depth, uint32_t cq_depth);

  /** Allocate a DMA-coherent data buffer. Returns buffer index, -1 on error. */
  int alloc_buffer(size_t size);

  /** Free a DMA buffer by index. */
  void free_buffer(int buf_idx);

  /** Map the doorbell BAR0 region. Must be called after attach. */
  int mmap_doorbells();

  /** Accessors */
  bool is_attached() const { return fd_ >= 0 && attached_; }
  uint32_t lba_size() const { return lba_size_; }
  uint32_t nsid() const { return nsid_; }
  void *doorbell_base() const { return db_base_; }
  const std::vector<QueuePairHost> &queues() const { return queues_; }
  const std::vector<DmaBuffer> &buffers() const { return buffers_; }

 private:
  int fd_;
  bool attached_;
  uint32_t lba_size_;
  uint32_t nsid_;
  uint32_t doorbell_stride_;
  uint32_t max_queue_depth_;
  uint64_t ns_size_blocks_;
  void *db_base_;
  size_t db_mmap_size_;
  std::vector<QueuePairHost> queues_;
  std::vector<DmaBuffer> buffers_;
};

}  // namespace bam

#endif  // BAM_CONTROLLER_H
