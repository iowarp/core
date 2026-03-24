/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_
#define CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_

#include <type_traits>
#include <utility>
#include <vector>

// Include LocalSerialize for serialization
#include <hermes_shm/data_structures/priv/vector.h>
#include <hermes_shm/data_structures/serialization/local_serialize.h>
#include <hermes_shm/lightbeam/lightbeam.h>

#include "chimaera/types.h"

namespace chi {

// Forward declaration
class Task;

/**
 * Message type enum for local task archives
 * Defines the type of message being sent/received
 */
enum class LocalMsgType : uint8_t {
  kSerializeIn = 0,  /**< Serialize task inputs for execution */
  kSerializeOut = 1, /**< Serialize task outputs */
};

/**
 * Common task information structure used by both LocalSaveTaskArchive and
 * LocalLoadTaskArchive
 */
struct LocalTaskInfo {
  TaskId task_id_;
  PoolId pool_id_;
  u32 method_id_;

  /**
   * Cereal serialization support for network transfer
   * @tparam Archive Archive type
   * @param ar Archive instance
   */
  template <class Archive>
  HSHM_CROSS_FUN void serialize(Archive &ar) {
    ar(task_id_.pid_, task_id_.tid_, task_id_.major_, task_id_.replica_id_,
       task_id_.unique_, task_id_.node_id_, task_id_.net_key_);
    ar(pool_id_.major_, pool_id_.minor_);
    ar(method_id_);
  }
};

}  // namespace chi

// Add local serialization support for LocalTaskInfo in hshm::ipc namespace
namespace hshm::ipc {

/**
 * Save LocalTaskInfo to local archive
 * @param ar Archive to save to
 * @param info LocalTaskInfo to serialize
 */
template <typename Ar>
HSHM_CROSS_FUN void save(Ar &ar, const chi::LocalTaskInfo &info) {
  // Serialize TaskId fields
  ar << info.task_id_.pid_;
  ar << info.task_id_.tid_;
  ar << info.task_id_.major_;
  ar << info.task_id_.replica_id_;
  ar << info.task_id_.unique_;
  ar << info.task_id_.node_id_;
  ar << info.task_id_.net_key_;
  // Serialize PoolId (UniqueId) fields
  ar << info.pool_id_.major_;
  ar << info.pool_id_.minor_;
  // Serialize method_id
  ar << info.method_id_;
}

/**
 * Load LocalTaskInfo from local archive
 * @param ar Archive to load from
 * @param info LocalTaskInfo to deserialize into
 */
template <typename Ar>
HSHM_CROSS_FUN void load(Ar &ar, chi::LocalTaskInfo &info) {
  // Deserialize TaskId fields
  ar >> info.task_id_.pid_;
  ar >> info.task_id_.tid_;
  ar >> info.task_id_.major_;
  ar >> info.task_id_.replica_id_;
  ar >> info.task_id_.unique_;
  ar >> info.task_id_.node_id_;
  ar >> info.task_id_.net_key_;
  // Deserialize PoolId (UniqueId) fields
  ar >> info.pool_id_.major_;
  ar >> info.pool_id_.minor_;
  // Deserialize method_id
  ar >> info.method_id_;
}

}  // namespace hshm::ipc

namespace chi {

// Base type for LbmMeta inheritance: use CHI_PRIV_ALLOC_T so that
// ShmTransport::Recv can allocate internal buffers on GPU (BuddyAllocator)
// and on host (MallocAllocator).
using LocalLbmBase = hshm::lbm::LbmMeta<CHI_PRIV_ALLOC_T>;
using LocalTaskInfoVec = chi::priv::vector<LocalTaskInfo>;

/**
 * Dry-run archive that computes serialized size for a task without copying.
 * Implements the same API surface as LocalSaveTaskArchive so that
 * SerializeIn / SerializeOut code paths work unchanged.
 * Used by LocalSaveTaskArchive to pre-size the buffer before serialization.
 */
class CalculateSizeTaskArchive {
 public:
  using is_saving = std::true_type;
  using is_loading = std::false_type;
  using supports_range_ops = std::true_type;

 private:
  hshm::ipc::CalculateSizeArchive calc_;
  LocalMsgType msg_type_;

 public:
  HSHM_CROSS_FUN explicit CalculateSizeTaskArchive(LocalMsgType msg_type)
      : msg_type_(msg_type) {}

  /** Get the total computed size */
  HSHM_INLINE_CROSS_FUN size_t size() const { return calc_.size(); }

  /** Serialize operator — for non-Task types, delegates to CalculateSizeArchive */
  template <typename T>
  HSHM_CROSS_FUN CalculateSizeTaskArchive &operator<<(const T &obj) {
    calc_ << obj;
    return *this;
  }

  /** & operator */
  template <typename T>
  HSHM_CROSS_FUN CalculateSizeTaskArchive &operator&(const T &obj) {
    calc_ << obj;
    return *this;
  }

  /** Call operator */
  template <typename... Args>
  HSHM_CROSS_FUN void operator()(Args &...args) {
    (calc_.base(args), ...);
  }

  /** Write raw binary data — just accumulate size */
  HSHM_CROSS_FUN void write_binary(const char *data, size_t size) {
    calc_.write_binary(data, size);
  }

  /** write_range — compute span size */
  template <typename FirstT, typename LastT>
  HSHM_INLINE_CROSS_FUN void write_range(const FirstT *first,
                                          const LastT *last) {
    calc_.write_range(first, last);
  }

  /** range() — compute span size of contiguous POD fields */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void range(Args &...args) {
    calc_.range(args...);
  }

  /** Fused string save — just accumulate size */
  HSHM_CROSS_FUN void save_string_fused(const char *str_data, size_t len) {
    calc_.save_string_fused(str_data, len);
  }

  /** Bulk transfer size for ShmPtr */
  template <typename T>
  HSHM_CROSS_FUN void bulk(hipc::ShmPtr<T> ptr, size_t size, uint32_t flags) {
    if (!ptr.alloc_id_.IsNull()) {
      // mode=0: uint8_t + size_t + u32 + u32
      calc_.cur_off_ += sizeof(uint8_t) + sizeof(size_t) +
                         sizeof(uint32_t) + sizeof(uint32_t);
    } else if (ptr.off_.load() != 0) {
      // mode=3: uint8_t + size_t
      calc_.cur_off_ += sizeof(uint8_t) + sizeof(size_t);
    } else if (flags & BULK_XFER) {
      // mode=1: uint8_t + data bytes
      calc_.cur_off_ += sizeof(uint8_t) + size;
    } else {
      // mode=2: uint8_t
      calc_.cur_off_ += sizeof(uint8_t);
    }
  }

  /** Bulk transfer size for FullPtr */
  template <typename T>
  HSHM_CROSS_FUN void bulk(const hipc::FullPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    if (!ptr.shm_.alloc_id_.IsNull()) {
      calc_.cur_off_ += sizeof(uint8_t) + sizeof(size_t) +
                         sizeof(uint32_t) + sizeof(uint32_t);
    } else if (flags & BULK_XFER) {
      calc_.cur_off_ += sizeof(uint8_t) + size;
    } else {
      calc_.cur_off_ += sizeof(uint8_t);
    }
  }

  /** Bulk transfer size for raw pointer */
  template <typename T>
  HSHM_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)ptr; (void)flags;
    calc_.cur_off_ += size;
  }
};

/**
 * Archive for saving tasks (inputs or outputs) using LocalSerialize
 * Local version that uses hshm::ipc::LocalSerialize instead of cereal
 * GPU version uses priv::vector with GPU allocator
 * Inherits from LbmMeta for ShmTransport::Send compatibility
 */
class LocalSaveTaskArchive : public LocalLbmBase {
  using Base = LocalLbmBase;

 public:
  using is_saving = std::true_type;
  using is_loading = std::false_type;
  using supports_range_ops = std::true_type;
  LocalTaskInfoVec task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

 private:
  chi::priv::vector<char> buffer_;
  hshm::ipc::LocalSerialize<chi::priv::vector<char>> serializer_;

 public:
  /**
   * Serialize for ShmTransport compatibility.
   * Wire-compatible with SaveTaskArchive::serialize /
   * LoadTaskArchive::serialize.
   */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    if constexpr (Ar::is_saving::value) {
      serializer_.Finalize();
    }
    ar(this->send, this->recv, this->send_bulks, this->recv_bulks);
    ar(task_infos_, msg_type_);
    ar(buffer_);
  }

  /**
   * Constructor with message type.
   *
   * @param msg_type Message type (kSerializeIn or kSerializeOut)
   */
  HSHM_CROSS_FUN explicit LocalSaveTaskArchive(LocalMsgType msg_type)
      : Base(CHI_PRIV_ALLOC),
        task_infos_(CHI_PRIV_ALLOC),
        msg_type_(msg_type),
        buffer_(CHI_PRIV_ALLOC),
        serializer_(buffer_) {
    // Pre-allocate to avoid repeated allocator calls during serialization.
    // Without this, the vector grows from 0 through ~8 capacity doublings,
    // each triggering a BuddyAllocator alloc+free on GPU (~1.5M clocks).
    buffer_.reserve(256);
    task_infos_.reserve(4);
  }

  /**
   * Constructor with external buffer and allocator.
   * Used by GPU worker where buffer is pre-allocated from GPU heap.
   */
  template <typename AllocPtrT>
  HSHM_CROSS_FUN LocalSaveTaskArchive(LocalMsgType msg_type,
                                       AllocPtrT *alloc)
      : Base(alloc),
        task_infos_(alloc),
        msg_type_(msg_type),
        buffer_(alloc),
        serializer_(buffer_) {
    buffer_.reserve(256);
    task_infos_.reserve(4);
  }

  /** Move constructor */
  HSHM_CROSS_FUN LocalSaveTaskArchive(LocalSaveTaskArchive &&other) noexcept
      : Base(std::move(other)),
        task_infos_(std::move(other.task_infos_)),
        msg_type_(other.msg_type_),
        buffer_(std::move(other.buffer_)),
        serializer_(buffer_, true) {}  // Use non-clearing constructor

  /** Move assignment operator - not supported due to reference member in
   * serializer */
  LocalSaveTaskArchive &operator=(LocalSaveTaskArchive &&other) noexcept =
      delete;

  /** Delete copy constructor and assignment */
  LocalSaveTaskArchive(const LocalSaveTaskArchive &) = delete;
  LocalSaveTaskArchive &operator=(const LocalSaveTaskArchive &) = delete;

 public:
  /**
   * Serialize operator - handles Task-derived types specially
   *
   * @tparam T Type to serialize
   * @param value Value to serialize
   * @return Reference to this archive for chaining
   */
  template <typename T>
  HSHM_CROSS_FUN LocalSaveTaskArchive &operator<<(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Record task information
      LocalTaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);

      // Serialize task based on mode
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
    } else {
      serializer_ << value;
    }
    return *this;
  }

  /**
   * Bidirectional serialization operator - forwards to operator<<
   * Used by types like bitfield that use ar & value syntax
   *
   * @tparam T Type to serialize
   * @param value Value to serialize
   * @return Reference to this archive for chaining
   */
  template <typename T>
  HSHM_CROSS_FUN LocalSaveTaskArchive &operator&(T &value) {
    return *this << value;
  }

  /**
   * Bidirectional serialization - acts as output for this archive type
   *
   * @tparam Args Types to serialize
   * @param args Values to serialize
   */
  template <typename... Args>
  HSHM_CROSS_FUN void operator()(Args &...args) {
    (SerializeArg(args), ...);
  }

 private:
  /** Helper to serialize individual arguments - handles Tasks specially */
  template <typename T>
  HSHM_CROSS_FUN void SerializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      *this << arg;
    } else {
      serializer_ << arg;
    }
  }

 public:
  /** Write raw binary data to the serializer */
  HSHM_CROSS_FUN void write_binary(const char *data, size_t size) {
    serializer_.write_binary(data, size);
  }

  /** Fused string save — delegates to serializer */
  HSHM_CROSS_FUN void save_string_fused(const char *str_data, size_t len) {
    serializer_.save_string_fused(str_data, len);
  }

  /** Batch-serialize a contiguous range of POD fields in one memcpy */
  template <typename FirstT, typename LastT>
  HSHM_INLINE_CROSS_FUN void write_range(const FirstT *first,
                                          const LastT *last) {
    serializer_.write_range(first, last);
  }

  /** range() — batch-serialize contiguous POD fields */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void range(Args &...args) {
    serializer_.range(args...);
  }

  /**
   * Bulk transfer support for ShmPtr - just serialize the pointer value
   *
   * @tparam T Type of data
   * @param ptr Shared memory pointer
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  HSHM_CROSS_FUN void bulk(hipc::ShmPtr<T> ptr, size_t size, uint32_t flags) {
    if (!ptr.alloc_id_.IsNull()) {
      // mode=0: SHM-offset-based pointer
      uint8_t mode = 0;
      serializer_ << mode;
      size_t off = ptr.off_.load();
      serializer_ << off << ptr.alloc_id_.major_ << ptr.alloc_id_.minor_;
    } else if (ptr.off_.load() != 0) {
      // mode=3: Raw UVA pointer (e.g., UVM buffer accessible to CPU and GPU).
      // Stores only the pointer address — avoids copying bulk data through the
      // ring buffer. The receiver resolves the address directly.
      uint8_t mode = 3;
      serializer_ << mode;
      size_t raw_ptr = ptr.off_.load();
      serializer_ << raw_ptr;
    } else if (flags & BULK_XFER) {
      // mode=1: null-alloc_id null-ptr with BULK_XFER — copy zero bytes
      uint8_t mode = 1;
      serializer_ << mode;
      char *raw_ptr = reinterpret_cast<char *>(ptr.off_.load());
      serializer_.write_binary(raw_ptr, size);
    } else {
      // mode=2: null pointer, BULK_EXPOSE only
      uint8_t mode = 2;
      serializer_ << mode;
    }
  }

  /**
   * Bulk transfer support for FullPtr - serialize only the shm_ part
   *
   * @tparam T Type of data
   * @param ptr Full pointer
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  HSHM_CROSS_FUN void bulk(const hipc::FullPtr<T> &ptr, size_t size,
                           uint32_t flags) {
    if (!ptr.shm_.alloc_id_.IsNull()) {
      uint8_t mode = 0;
      serializer_ << mode;
      size_t off = ptr.shm_.off_.load();
      serializer_ << off << ptr.shm_.alloc_id_.major_
                  << ptr.shm_.alloc_id_.minor_;
    } else if (flags & BULK_XFER) {
      uint8_t mode = 1;
      serializer_ << mode;
      serializer_.write_binary(reinterpret_cast<const char *>(ptr.ptr_), size);
    } else {
      uint8_t mode = 2;
      serializer_ << mode;
    }
  }

  /**
   * Bulk transfer support for raw pointer - full memory copy
   *
   * @tparam T Type of data
   * @param ptr Raw pointer
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  HSHM_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    serializer_.write_binary(reinterpret_cast<const char *>(ptr), size);
  }

  /**
   * Get task information
   *
   * @return Vector of task information
   */
  const LocalTaskInfoVec &GetTaskInfos() const { return task_infos_; }

  /**
   * Get message type
   *
   * @return Message type
   */
  LocalMsgType GetMsgType() const { return msg_type_; }

  /**
   * Get serialized data size
   *
   * @return Size of serialized data
   */
  HSHM_CROSS_FUN size_t GetSize() {
    serializer_.Finalize();
    return buffer_.size();
  }
  /**
   * Get serialized data
   *
   * @return Reference to buffer containing serialized data
   */
  HSHM_CROSS_FUN const chi::priv::vector<char> &GetData() {
    serializer_.Finalize();
    return buffer_;
  }

  /**
   * Move serialized data out of the archive
   *
   * @return Moved buffer containing serialized data
   */
  chi::priv::vector<char> MoveData() {
    serializer_.Finalize();
    return std::move(buffer_);
  }
};

/**
 * Archive for loading tasks (inputs or outputs) using LocalDeserialize
 * Local version that uses hshm::ipc::LocalDeserialize instead of cereal
 * GPU version uses priv::vector with GPU allocator
 * Inherits from LbmMeta for ShmTransport::Recv compatibility
 */
class LocalLoadTaskArchive : public LocalLbmBase {
  using Base = LocalLbmBase;

 public:
  using is_saving = std::false_type;
  using is_loading = std::true_type;
  using supports_range_ops = std::true_type;
  LocalTaskInfoVec task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

 private:
  chi::priv::vector<char> owned_data_;
  const chi::priv::vector<char> *data_;
  hshm::ipc::LocalDeserialize<chi::priv::vector<char>> deserializer_;
  size_t current_task_index_;

 public:
  /**
   * Serialize for ShmTransport compatibility.
   * Wire-compatible with SaveTaskArchive::serialize /
   * LoadTaskArchive::serialize. Populates data_ from ring buffer metadata, then
   * deserializer_ reads from it.
   */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(this->send, this->recv, this->send_bulks, this->recv_bulks);
    ar(task_infos_, msg_type_);
    ar(owned_data_);
    data_ = &owned_data_;
    new (&deserializer_)
        hshm::ipc::LocalDeserialize<chi::priv::vector<char>>(owned_data_);
  }

  /**
   * Default constructor
   */
  HSHM_CROSS_FUN LocalLoadTaskArchive()
      : Base(CHI_PRIV_ALLOC),
        task_infos_(CHI_PRIV_ALLOC),
        msg_type_(LocalMsgType::kSerializeIn),
        owned_data_(CHI_PRIV_ALLOC),
        data_(nullptr),
        deserializer_(GetEmptyBuffer()),
        current_task_index_(0) {
    owned_data_.reserve(256);
    task_infos_.reserve(4);
  }

  /**
   * Constructor with explicit allocator pointer.
   * Used by GPU worker where allocator comes from GPU heap.
   */
  HSHM_CROSS_FUN explicit LocalLoadTaskArchive(CHI_PRIV_ALLOC_T *alloc)
      : Base(alloc),
        task_infos_(alloc),
        msg_type_(LocalMsgType::kSerializeIn),
        owned_data_(alloc),
        data_(nullptr),
        deserializer_(GetEmptyBuffer()),
        current_task_index_(0) {
    owned_data_.reserve(256);
    task_infos_.reserve(4);
  }

  /**
   * Constructor from serialized data (uses chi::priv::vector)
   *
   * @param data Buffer containing serialized data
   */
  HSHM_CROSS_FUN explicit LocalLoadTaskArchive(
      const chi::priv::vector<char> &data)
      : Base(CHI_PRIV_ALLOC),
        task_infos_(CHI_PRIV_ALLOC),
        msg_type_(LocalMsgType::kSerializeIn),
        owned_data_(CHI_PRIV_ALLOC),
        data_(&data),
        deserializer_(data),
        current_task_index_(0) {}

  /**
   * Constructor from std::vector serialized data.
   * Copies the data into owned_data_ for lifetime management.
   *
   * @param data Buffer containing serialized data
   */
  explicit LocalLoadTaskArchive(const std::vector<char> &data)
      : Base(),
        task_infos_(CHI_PRIV_ALLOC),
        msg_type_(LocalMsgType::kSerializeIn),
        owned_data_(CHI_PRIV_ALLOC),
        data_(nullptr),
        deserializer_(GetEmptyBuffer()),
        current_task_index_(0) {
    owned_data_.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      owned_data_.push_back(data[i]);
    }
    data_ = &owned_data_;
    new (&deserializer_)
        hshm::ipc::LocalDeserialize<chi::priv::vector<char>>(owned_data_);
  }

  /** Move constructor */
  HSHM_CROSS_FUN LocalLoadTaskArchive(LocalLoadTaskArchive &&other) noexcept
      : Base(std::move(other)),
        task_infos_(std::move(other.task_infos_)),
        msg_type_(other.msg_type_),
        owned_data_(std::move(other.owned_data_)),
        data_(other.data_),
        deserializer_(other.data_ ? *other.data_ : GetEmptyBuffer()),
        current_task_index_(other.current_task_index_) {
    other.data_ = nullptr;
  }

 public:
  /** Move assignment operator - not supported due to reference member in
   * deserializer */
  LocalLoadTaskArchive &operator=(LocalLoadTaskArchive &&other) noexcept =
      delete;

  /** Delete copy constructor and assignment */
  LocalLoadTaskArchive(const LocalLoadTaskArchive &) = delete;
  LocalLoadTaskArchive &operator=(const LocalLoadTaskArchive &) = delete;

 public:
  /**
   * Deserialize operator - handles Task-derived types specially
   *
   * @tparam T Type to deserialize
   * @param value Value to deserialize into
   * @return Reference to this archive for chaining
   */
  template <typename T>
  HSHM_CROSS_FUN LocalLoadTaskArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        value.SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        value.SerializeOut(*this);
      }
    } else {
      deserializer_ >> value;
    }
    return *this;
  }

  /**
   * Bidirectional serialization operator - forwards to operator>>
   * Used by types like bitfield that use ar & value syntax
   *
   * @tparam T Type to deserialize
   * @param value Value to deserialize into
   * @return Reference to this archive for chaining
   */
  template <typename T>
  HSHM_CROSS_FUN LocalLoadTaskArchive &operator&(T &value) {
    return *this >> value;
  }

  /**
   * Deserialize task pointers
   *
   * @tparam T Task type
   * @param value Pointer to deserialize into (must be pre-allocated)
   * @return Reference to this archive for chaining
   */
  template <typename T>
  LocalLoadTaskArchive &operator>>(T *&value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        value->SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        value->SerializeOut(*this);
      }
#if HSHM_IS_HOST
      current_task_index_++;
#endif
    } else {
      deserializer_ >> value;
    }
    return *this;
  }

  /**
   * Bidirectional serialization - acts as input for this archive type
   *
   * @tparam Args Types to deserialize
   * @param args Values to deserialize into
   */
  template <typename... Args>
  HSHM_CROSS_FUN void operator()(Args &...args) {
    (DeserializeArg(args), ...);
  }

 private:
  /** Helper to deserialize individual arguments - handles Tasks specially */
  template <typename T>
  HSHM_CROSS_FUN void DeserializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      *this >> arg;
    } else {
      deserializer_ >> arg;
    }
  }

 public:
  /** Read raw binary data from the deserializer */
  HSHM_CROSS_FUN void read_binary(char *data, size_t size) {
    deserializer_.read_binary(data, size);
  }

  /** Batch-deserialize a contiguous range of POD fields in one memcpy */
  template <typename FirstT, typename LastT>
  HSHM_INLINE_CROSS_FUN void read_range(FirstT *first, LastT *last) {
    deserializer_.read_range(first, last);
  }

  /** range() — batch-deserialize contiguous POD fields */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void range(Args &...args) {
    deserializer_.range(args...);
  }

  /**
   * Bulk transfer support for ShmPtr - deserialize the pointer value
   *
   * @tparam T Type of data
   * @param ptr Shared memory pointer to deserialize into
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  HSHM_CROSS_FUN void bulk(hipc::ShmPtr<T> &ptr, size_t size, uint32_t flags) {
    (void)flags;
    uint8_t mode = 0;
    deserializer_ >> mode;
    if (mode == 1) {
      // BULK_XFER: allocate buffer and copy bytes from archive.
      // On GPU this path is not reached: LocalSaveTaskArchive uses mode=3
      // for any non-null UVA pointer, so mode=1 only occurs for null ptrs.
#if HSHM_IS_GPU
      ptr.alloc_id_.SetNull();
      ptr.off_ = 0;
#else
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      deserializer_.read_binary(buf.ptr_, size);
      ptr.off_ = buf.shm_.off_.load();
      ptr.alloc_id_ = buf.shm_.alloc_id_;
#endif
    } else if (mode == 2) {
      // BULK_EXPOSE: allocate buffer without copying.
      // On GPU this path is not reached (see mode=1 note above).
#if HSHM_IS_GPU
      ptr.alloc_id_.SetNull();
      ptr.off_ = 0;
#else
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      ptr.off_ = buf.shm_.off_.load();
      ptr.alloc_id_ = buf.shm_.alloc_id_;
#endif
    } else if (mode == 3) {
      // Raw UVA pointer — restore directly (UVM visible to both CPU and GPU)
      size_t raw_ptr = 0;
      deserializer_ >> raw_ptr;
      ptr.off_ = raw_ptr;
      ptr.alloc_id_.SetNull();
    } else {
      // mode == 0: SHM-offset-based pointer
      size_t off = 0;
      u32 major = 0, minor = 0;
      deserializer_ >> off >> major >> minor;
      ptr.off_ = off;
      ptr.alloc_id_ = hipc::AllocatorId(major, minor);
    }
  }

  /**
   * Bulk transfer support for FullPtr - deserialize only the shm_ part
   *
   * @tparam T Type of data
   * @param ptr Full pointer to deserialize into
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  void bulk(hipc::FullPtr<T> &ptr, size_t size, uint32_t flags) {
    (void)flags;
    uint8_t mode = 0;
    deserializer_ >> mode;
    if (mode == 1) {
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      deserializer_.read_binary(buf.ptr_, size);
      ptr.shm_.off_ = buf.shm_.off_.load();
      ptr.shm_.alloc_id_ = buf.shm_.alloc_id_;
      ptr.ptr_ = reinterpret_cast<T *>(buf.ptr_);
    } else if (mode == 2) {
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      ptr.shm_.off_ = buf.shm_.off_.load();
      ptr.shm_.alloc_id_ = buf.shm_.alloc_id_;
      ptr.ptr_ = reinterpret_cast<T *>(buf.ptr_);
    } else {
      size_t off = 0;
      u32 major = 0, minor = 0;
      deserializer_ >> off >> major >> minor;
      ptr.shm_.off_ = off;
      ptr.shm_.alloc_id_ = hipc::AllocatorId(major, minor);
    }
  }

  /**
   * Bulk transfer support for raw pointer - full memory copy
   *
   * @tparam T Type of data
   * @param ptr Raw pointer to deserialize into (must be pre-allocated)
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  HSHM_CROSS_FUN void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    deserializer_.read_binary(reinterpret_cast<char *>(ptr), size);
  }

  /**
   * Get task information (HOST only)
   *
   * @return Vector of task information
   */
  const LocalTaskInfoVec &GetTaskInfos() const { return task_infos_; }

  /**
   * Get current task info (HOST only)
   *
   * @return Current task information
   */
  const LocalTaskInfo &GetCurrentTaskInfo() const {
    return task_infos_[current_task_index_];
  }

  /**
   * Get message type
   *
   * @return Message type
   */
  LocalMsgType GetMsgType() const { return msg_type_; }

  /**
   * Reset task index for iteration (HOST only)
   */
  void ResetTaskIndex() { current_task_index_ = 0; }

  /**
   * Set message type
   *
   * @param msg_type Message type
   */
  HSHM_CROSS_FUN void SetMsgType(LocalMsgType msg_type) {
    msg_type_ = msg_type;
  }

 private:
  /**
   * Get a reference to an empty buffer for use as a placeholder.
   * Uses function-local static to be accessible from both host and device.
   *
   * @return Reference to an empty vector
   */
  HSHM_CROSS_FUN static chi::priv::vector<char> &GetEmptyBuffer() {
#if HSHM_IS_HOST
    static chi::priv::vector<char> buf(nullptr);
#else
    // On GPU, use shared memory for the empty buffer placeholder
    __shared__ char buf_storage[sizeof(chi::priv::vector<char>)];
    auto &buf = *reinterpret_cast<chi::priv::vector<char> *>(buf_storage);
#endif
    return buf;
  }
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_
