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

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// Include LocalSerialize for serialization
#include <hermes_shm/data_structures/serialization/local_serialize.h>
#include <hermes_shm/data_structures/priv/vector.h>
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
  kSerializeIn = 0,   /**< Serialize task inputs for execution */
  kSerializeOut = 1,  /**< Serialize task outputs */
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
    ar(task_id_.pid_, task_id_.tid_, task_id_.major_,
       task_id_.replica_id_, task_id_.unique_, task_id_.node_id_,
       task_id_.net_key_);
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

// Conditional base type for LbmMeta inheritance
#if HSHM_IS_HOST
using LocalLbmBase = hshm::lbm::LbmMeta<>;
using LocalTaskInfoVec = std::vector<LocalTaskInfo>;
#else
using LocalLbmBase = hshm::lbm::LbmMeta<HSHM_DEFAULT_ALLOC_GPU_T>;
using LocalTaskInfoVec = hshm::priv::vector<LocalTaskInfo, HSHM_DEFAULT_ALLOC_GPU_T>;
#endif

/**
 * Archive for saving tasks (inputs or outputs) using LocalSerialize
 * Local version that uses hshm::ipc::LocalSerialize instead of cereal
 * GPU version uses priv::vector with GPU allocator
 * Inherits from LbmMeta for ShmTransport::Send compatibility
 */
class LocalSaveTaskArchive : public LocalLbmBase {
  using Base = LocalLbmBase;
public:
  LocalTaskInfoVec task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

private:
#if HSHM_IS_HOST
  std::vector<char> buffer_;
  hshm::ipc::LocalSerialize<std::vector<char>> serializer_;
#else
  using GpuVec = hshm::priv::vector<char, HSHM_DEFAULT_ALLOC_GPU_T>;
  GpuVec &buffer_;
  hshm::ipc::LocalSerialize<GpuVec> serializer_;
#endif

public:
  /**
   * Serialize for ShmTransport compatibility.
   * Wire-compatible with SaveTaskArchive::serialize / LoadTaskArchive::serialize.
   */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(this->send, this->recv, this->send_bulks, this->recv_bulks);
    ar(task_infos_, msg_type_);
    ar(buffer_);
  }

  /**
   * Constructor with message type (HOST - uses std::vector buffer)
   *
   * @param msg_type Message type (kSerializeIn or kSerializeOut)
   */
#if HSHM_IS_HOST
  explicit LocalSaveTaskArchive(LocalMsgType msg_type)
      : Base(), msg_type_(msg_type), serializer_(buffer_) {}
#else
  HSHM_GPU_FUN explicit LocalSaveTaskArchive(LocalMsgType msg_type);  // Not implemented for GPU
#endif

#if HSHM_IS_GPU_COMPILER
  /**
   * Constructor with message type, pre-allocated vector, and allocator (GPU)
   *
   * @param msg_type Message type (kSerializeIn or kSerializeOut)
   * @param buffer Pre-allocated priv::vector for serialization
   * @param alloc Allocator for LbmMeta and task_infos_
   */
  HSHM_CROSS_FUN explicit LocalSaveTaskArchive(
      LocalMsgType msg_type,
      hshm::priv::vector<char, HSHM_DEFAULT_ALLOC_GPU_T> &buffer,
      HSHM_DEFAULT_ALLOC_GPU_T *alloc)
      :
#if HSHM_IS_GPU
      Base(alloc), task_infos_(alloc),
#else
      Base(),
#endif
      msg_type_(msg_type)
#if HSHM_IS_GPU
      , buffer_(buffer), serializer_(buffer_)
#else
      , serializer_(buffer_)
#endif
  { (void)buffer; (void)alloc; }
#endif

#if HSHM_IS_HOST
  /** Move constructor (HOST only) */
  LocalSaveTaskArchive(LocalSaveTaskArchive &&other) noexcept
      : Base(std::move(other)),
        task_infos_(std::move(other.task_infos_)), msg_type_(other.msg_type_),
        buffer_(std::move(other.buffer_)),
        serializer_(buffer_, true) {} // Use non-clearing constructor

  /** Move assignment operator - not supported due to reference member in serializer */
  LocalSaveTaskArchive &operator=(LocalSaveTaskArchive &&other) noexcept = delete;
#else
  /** Move constructor disabled for GPU */
  LocalSaveTaskArchive(LocalSaveTaskArchive &&other) = delete;
  LocalSaveTaskArchive &operator=(LocalSaveTaskArchive &&other) = delete;
#endif

  /** Delete copy constructor and assignment */
  LocalSaveTaskArchive(const LocalSaveTaskArchive &) = delete;
  LocalSaveTaskArchive &operator=(const LocalSaveTaskArchive &) = delete;

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
#if HSHM_IS_HOST
      // Record task information
      LocalTaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);
#endif

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
      uint8_t mode = 0;
      serializer_ << mode;
      size_t off = ptr.off_.load();
      serializer_ << off << ptr.alloc_id_.major_ << ptr.alloc_id_.minor_;
    } else if (flags & BULK_XFER) {
      uint8_t mode = 1;
      serializer_ << mode;
      char *raw_ptr = reinterpret_cast<char *>(ptr.off_.load());
      serializer_.write_binary(raw_ptr, size);
    } else {
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
  HSHM_CROSS_FUN void bulk(const hipc::FullPtr<T> &ptr, size_t size, uint32_t flags) {
    if (!ptr.shm_.alloc_id_.IsNull()) {
      uint8_t mode = 0;
      serializer_ << mode;
      size_t off = ptr.shm_.off_.load();
      serializer_ << off << ptr.shm_.alloc_id_.major_ << ptr.shm_.alloc_id_.minor_;
    } else if (flags & BULK_XFER) {
      uint8_t mode = 1;
      serializer_ << mode;
      serializer_.write_binary(reinterpret_cast<const char *>(ptr.get()), size);
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

#if HSHM_IS_HOST
  /**
   * Get task information (HOST only)
   *
   * @return Vector of task information
   */
  const std::vector<LocalTaskInfo> &GetTaskInfos() const { return task_infos_; }
#endif

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
  HSHM_CROSS_FUN size_t GetSize() const {
    return buffer_.size();
  }

#if HSHM_IS_HOST
  /**
   * Get serialized data
   *
   * @return Reference to buffer containing serialized data
   */
  const std::vector<char> &GetData() const { return buffer_; }

  /**
   * Move serialized data out of the archive
   *
   * @return Moved buffer containing serialized data
   */
  std::vector<char> MoveData() { return std::move(buffer_); }
#else
  /**
   * Get raw buffer pointer (GPU only)
   *
   * @return Pointer to buffer data
   */
  HSHM_GPU_FUN const char *GetData() const { return buffer_.data(); }
#endif
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
  LocalTaskInfoVec task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

private:
#if HSHM_IS_HOST
  std::vector<char> owned_data_;
  const std::vector<char> *data_;
  hshm::ipc::LocalDeserialize<std::vector<char>> deserializer_;
  size_t current_task_index_;
#else
  using GpuVec = hshm::priv::vector<char, HSHM_DEFAULT_ALLOC_GPU_T>;
  GpuVec data_;
  hshm::ipc::LocalDeserialize<GpuVec> deserializer_;
#endif

public:
  /**
   * Serialize for ShmTransport compatibility.
   * Wire-compatible with SaveTaskArchive::serialize / LoadTaskArchive::serialize.
   * Populates data_ from ring buffer metadata, then deserializer_ reads from it.
   */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(this->send, this->recv, this->send_bulks, this->recv_bulks);
    ar(task_infos_, msg_type_);
#if HSHM_IS_HOST
    ar(owned_data_);
    data_ = &owned_data_;
    new (&deserializer_) hshm::ipc::LocalDeserialize<std::vector<char>>(owned_data_);
#else
    ar(data_);
    new (&deserializer_) hshm::ipc::LocalDeserialize<GpuVec>(data_);
#endif
  }

#if HSHM_IS_HOST
  /**
   * Default constructor (HOST)
   */
  LocalLoadTaskArchive()
      : Base(), msg_type_(LocalMsgType::kSerializeIn), data_(nullptr),
        deserializer_(empty_buffer_), current_task_index_(0) {}

  /**
   * Constructor from serialized data (HOST - uses std::vector)
   *
   * @param data Buffer containing serialized data
   */
  explicit LocalLoadTaskArchive(const std::vector<char> &data)
      : Base(), msg_type_(LocalMsgType::kSerializeIn), data_(&data),
        deserializer_(data), current_task_index_(0) {}
#else
  HSHM_GPU_FUN LocalLoadTaskArchive();  // Not implemented for GPU
  HSHM_GPU_FUN explicit LocalLoadTaskArchive(const std::vector<char> &data);  // Not implemented for GPU
#endif

#if HSHM_IS_GPU_COMPILER
  /**
   * Constructor with allocator only (GPU - for ShmTransport::Recv path)
   * data_ starts empty; serialize() populates it from the ring buffer.
   *
   * @param alloc Allocator for LbmMeta, task_infos_, and data_
   */
  HSHM_CROSS_FUN explicit LocalLoadTaskArchive(
      HSHM_DEFAULT_ALLOC_GPU_T *alloc)
      :
#if HSHM_IS_GPU
      Base(alloc), task_infos_(alloc),
#else
      Base(),
#endif
      msg_type_(LocalMsgType::kSerializeIn)
#if HSHM_IS_GPU
      , data_(alloc), deserializer_(data_)
#else
      , data_(nullptr), deserializer_(empty_buffer_), current_task_index_(0)
#endif
  { (void)alloc; }

  /**
   * Constructor from pre-populated priv::vector (GPU)
   * Copies data from ext_data into owned data_ buffer.
   *
   * @param ext_data Pre-populated vector containing serialized data
   * @param alloc Allocator for LbmMeta, task_infos_, and data_
   */
  HSHM_CROSS_FUN explicit LocalLoadTaskArchive(
      const hshm::priv::vector<char, HSHM_DEFAULT_ALLOC_GPU_T> &ext_data,
      HSHM_DEFAULT_ALLOC_GPU_T *alloc)
      :
#if HSHM_IS_GPU
      Base(alloc), task_infos_(alloc),
#else
      Base(),
#endif
      msg_type_(LocalMsgType::kSerializeIn)
#if HSHM_IS_GPU
      , data_(ext_data.size(), alloc), deserializer_(data_)
#else
      , data_(nullptr), deserializer_(empty_buffer_), current_task_index_(0)
#endif
  {
#if HSHM_IS_GPU
    for (size_t i = 0; i < ext_data.size(); ++i) {
      data_[i] = ext_data[i];
    }
#endif
    (void)ext_data; (void)alloc;
  }
#endif

#if HSHM_IS_HOST
  /** Move constructor (HOST only) */
  LocalLoadTaskArchive(LocalLoadTaskArchive &&other) noexcept
      : Base(std::move(other)),
        task_infos_(std::move(other.task_infos_)), msg_type_(other.msg_type_),
        data_(other.data_), deserializer_(other.data_ ? *other.data_ : empty_buffer_),
        current_task_index_(other.current_task_index_) {
    other.data_ = nullptr;
  }

  /** Move assignment operator - not supported due to reference member in deserializer */
  LocalLoadTaskArchive &operator=(LocalLoadTaskArchive &&other) noexcept = delete;
#else
  /** Move constructor disabled for GPU */
  LocalLoadTaskArchive(LocalLoadTaskArchive &&other) = delete;
  LocalLoadTaskArchive &operator=(LocalLoadTaskArchive &&other) = delete;
#endif

  /** Delete copy constructor and assignment */
  LocalLoadTaskArchive(const LocalLoadTaskArchive &) = delete;
  LocalLoadTaskArchive &operator=(const LocalLoadTaskArchive &) = delete;

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
  template <typename T> LocalLoadTaskArchive &operator>>(T *&value) {
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
  /**
   * Bulk transfer support for ShmPtr - deserialize the pointer value
   *
   * @tparam T Type of data
   * @param ptr Shared memory pointer to deserialize into
   * @param size Size of data
   * @param flags Transfer flags
   */
  template <typename T>
  void bulk(hipc::ShmPtr<T> &ptr, size_t size, uint32_t flags) {
    (void)flags;
    uint8_t mode = 0;
    deserializer_ >> mode;
    if (mode == 1) {
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      deserializer_.read_binary(buf.get(), size);
      ptr.off_ = buf.shm_.off_.load();
      ptr.alloc_id_ = buf.shm_.alloc_id_;
    } else if (mode == 2) {
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      ptr.off_ = buf.shm_.off_.load();
      ptr.alloc_id_ = buf.shm_.alloc_id_;
    } else {
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
      deserializer_.read_binary(buf.get(), size);
      ptr.shm_.off_ = buf.shm_.off_.load();
      ptr.shm_.alloc_id_ = buf.shm_.alloc_id_;
      ptr.set_ptr(reinterpret_cast<T *>(buf.get()));
    } else if (mode == 2) {
      hipc::FullPtr<char> buf = HSHM_MALLOC->AllocateObjs<char>(size);
      ptr.shm_.off_ = buf.shm_.off_.load();
      ptr.shm_.alloc_id_ = buf.shm_.alloc_id_;
      ptr.set_ptr(reinterpret_cast<T *>(buf.get()));
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
  void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;
    deserializer_.read_binary(reinterpret_cast<char *>(ptr), size);
  }

#if HSHM_IS_HOST
  /**
   * Get task information (HOST only)
   *
   * @return Vector of task information
   */
  const std::vector<LocalTaskInfo> &GetTaskInfos() const { return task_infos_; }

  /**
   * Get current task info (HOST only)
   *
   * @return Current task information
   */
  const LocalTaskInfo &GetCurrentTaskInfo() const {
    return task_infos_[current_task_index_];
  }
#endif

  /**
   * Get message type
   *
   * @return Message type
   */
  LocalMsgType GetMsgType() const { return msg_type_; }

#if HSHM_IS_HOST
  /**
   * Reset task index for iteration (HOST only)
   */
  void ResetTaskIndex() { current_task_index_ = 0; }
#endif

  /**
   * Set message type
   *
   * @param msg_type Message type
   */
  HSHM_CROSS_FUN void SetMsgType(LocalMsgType msg_type) { msg_type_ = msg_type; }

private:
#if HSHM_IS_HOST
  static inline std::vector<char> empty_buffer_;
#endif
};

} // namespace chi

#endif // CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_
