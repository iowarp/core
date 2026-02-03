/**
 * LocalTransfer - Chunked data transfer between worker and client
 *
 * This class encapsulates the streaming transfer protocol for moving
 * serialized task data through FutureShm's copy_space buffer.
 *
 * The transfer protocol uses FUTURE_NEW_DATA flag for synchronization:
 * - Sender sets flag when data is available
 * - Receiver unsets flag when data is consumed
 * - Both sides respect time budgets and can be called multiple times
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TRANSFER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TRANSFER_H_

#include <atomic>
#include <vector>

#include "chimaera/task.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/timer.h"

namespace chi {

// Forward declaration
class Container;

/**
 * LocalTransfer - Manages chunked data transfer via FutureShm copy_space
 *
 * Used for:
 * - Worker::CopyTaskOutputToClient (sender side)
 * - IpcManager::Recv in client mode (receiver side)
 *
 * The class maintains transfer state across multiple Send()/Recv() calls,
 * allowing the transfer to be interleaved with other work.
 */
class LocalTransfer {
 public:
  /**
   * Construct a sender-side LocalTransfer
   *
   * @param data Serialized data to send (moved into LocalTransfer)
   * @param future_shm FullPtr to the FutureShm structure
   * @param task_ptr Task pointer for cleanup (optional, nullptr by default)
   * @param method_id Method ID for task deletion (optional, 0 by default)
   * @param container Container for task deletion (optional, nullptr by default)
   */
  LocalTransfer(std::vector<char>&& data, hipc::FullPtr<FutureShm> future_shm,
                hipc::FullPtr<Task> task_ptr = hipc::FullPtr<Task>(),
                u32 method_id = 0,
                Container* container = nullptr);

  /**
   * Construct a receiver-side LocalTransfer
   *
   * @param future_shm FullPtr to the FutureShm structure
   * @param total_size Total expected size of data to receive
   */
  LocalTransfer(hipc::FullPtr<FutureShm> future_shm, size_t total_size);

  /**
   * Default constructor for container usage
   */
  LocalTransfer() = default;

  /**
   * Move constructor
   */
  LocalTransfer(LocalTransfer&& other) noexcept;

  /**
   * Move assignment operator
   */
  LocalTransfer& operator=(LocalTransfer&& other) noexcept;

  // Disable copy operations
  LocalTransfer(const LocalTransfer&) = delete;
  LocalTransfer& operator=(const LocalTransfer&) = delete;

  /**
   * Send data to client within time budget
   *
   * Transfers as much data as possible to the client within the specified
   * time frame. Can be called multiple times until IsComplete() returns true.
   *
   * Protocol:
   * 1. If FUTURE_NEW_DATA is set, wait briefly for client to consume
   * 2. Copy next chunk to copy_space
   * 3. Set FUTURE_NEW_DATA flag
   * 4. Repeat until time budget exhausted or transfer complete
   *
   * @param max_xfer_time_us Maximum time to spend transferring in microseconds (default 5000us = 5ms)
   * @return true if transfer is complete, false if more calls needed
   */
  bool Send(u32 max_xfer_time_us = 5000);

  /**
   * Receive data from worker
   *
   * Waits for data to arrive in copy_space and assembles it into the
   * internal buffer. Blocks until all data is received.
   *
   * Protocol:
   * 1. Wait for FUTURE_NEW_DATA flag to be set
   * 2. Copy data from copy_space to internal buffer
   * 3. Unset FUTURE_NEW_DATA flag to signal consumption
   * 4. Repeat until all data received
   *
   * @return true if transfer is complete, false on error
   */
  bool Recv();

  /**
   * Check if the transfer is complete
   *
   * @return true if all data has been transferred
   */
  bool IsComplete() const { return bytes_transferred_ >= total_size_; }

  /**
   * Get the received data buffer
   *
   * Only valid for receiver-side LocalTransfer after Recv() completes.
   *
   * @return Reference to the received data buffer
   */
  std::vector<char>& GetData() { return data_; }

  /**
   * Get the received data buffer (const)
   *
   * @return Const reference to the received data buffer
   */
  const std::vector<char>& GetData() const { return data_; }

  /**
   * Get number of bytes transferred so far
   *
   * @return Number of bytes transferred
   */
  size_t GetBytesTransferred() const { return bytes_transferred_; }

  /**
   * Get total size of data to transfer
   *
   * @return Total data size in bytes
   */
  size_t GetTotalSize() const { return total_size_; }

  /**
   * Check if this is a sender-side transfer
   *
   * @return true if sender, false if receiver
   */
  bool IsSender() const { return is_sender_; }

  /**
   * Set the FUTURE_COMPLETE flag on the FutureShm
   *
   * Called automatically when Send() completes, but can be called
   * manually if needed.
   */
  void SetComplete();

 private:
  /** Serialized data buffer (sender holds data to send, receiver accumulates) */
  std::vector<char> data_;

  /** FullPtr to the FutureShm structure */
  hipc::FullPtr<FutureShm> future_shm_;

  /** Task pointer for cleanup on completion (sender only) */
  hipc::FullPtr<Task> task_ptr_;

  /** Method ID for task deletion (sender only) */
  u32 method_id_ = 0;

  /** Container for task deletion (sender only, nullptr if no cleanup needed) */
  Container* container_ = nullptr;

  /** Number of bytes transferred so far */
  size_t bytes_transferred_ = 0;

  /** Total size of data to transfer */
  size_t total_size_ = 0;

  /** Flag indicating sender (true) or receiver (false) mode */
  bool is_sender_ = false;

  /** Flag indicating if this object has been initialized */
  bool is_initialized_ = false;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TRANSFER_H_
