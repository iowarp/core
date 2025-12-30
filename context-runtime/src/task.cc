/**
 * Task implementation
 */

#include "chimaera/task.h"

namespace chi {

size_t Task::EstCpuTime() const {
  // Calculate: io_size / 4GBps + compute + 5
  // 4 GBps = 4 * 1024 * 1024 * 1024 bytes/second = 4294967296 bytes/second
  // Convert to microseconds: (io_size / 4294967296) * 1000000
  size_t io_time_us = (stat_.io_size_ * 1000000) / 4294967296ULL;
  return io_time_us + stat_.compute_ + 5;
}

}  // namespace chi
