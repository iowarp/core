#ifndef CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_
#define CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// Include LocalSerialize for serialization
#include <hermes_shm/data_structures/serialization/local_serialize.h>

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
  void serialize(Archive &ar) {
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
void save(Ar &ar, const chi::LocalTaskInfo &info) {
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
void load(Ar &ar, chi::LocalTaskInfo &info) {
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


/**
 * Archive for saving tasks (inputs or outputs) using LocalSerialize
 * Local version that uses hshm::ipc::LocalSerialize instead of cereal
 */
class LocalSaveTaskArchive {
public:
  std::vector<LocalTaskInfo> task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

private:
  std::vector<char> buffer_;
  hshm::ipc::LocalSerialize<std::vector<char>> serializer_;

public:
  /**
   * Constructor with message type
   *
   * @param msg_type Message type (kSerializeIn or kSerializeOut)
   */
  explicit LocalSaveTaskArchive(LocalMsgType msg_type)
      : msg_type_(msg_type), serializer_(buffer_) {}

  /** Move constructor */
  LocalSaveTaskArchive(LocalSaveTaskArchive &&other) noexcept
      : task_infos_(std::move(other.task_infos_)), msg_type_(other.msg_type_),
        buffer_(std::move(other.buffer_)),
        serializer_(buffer_, true) {} // Use non-clearing constructor

  /** Move assignment operator - not supported due to reference member in serializer */
  LocalSaveTaskArchive &operator=(LocalSaveTaskArchive &&other) noexcept = delete;

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
  template <typename T> LocalSaveTaskArchive &operator<<(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Record task information
      LocalTaskInfo info{value.task_id_, value.pool_id_, value.method_};
      task_infos_.push_back(info);

      // Serialize task based on mode
      // Task::SerializeIn/SerializeOut will handle base class fields
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        // SerializeIn mode - serialize input parameters
        value.SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        // SerializeOut mode - serialize output parameters
        value.SerializeOut(*this);
      }
    } else {
      serializer_ << value;
    }
    return *this;
  }

  /**
   * Bidirectional serialization - acts as output for this archive type
   *
   * @tparam Args Types to serialize
   * @param args Values to serialize
   */
  template <typename... Args> void operator()(Args &...args) {
    (SerializeArg(args), ...);
  }

private:
  /** Helper to serialize individual arguments - handles Tasks specially */
  template <typename T> void SerializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      // This is a Task or Task pointer - use operator<< which handles tasks
      *this << arg;
    } else {
      // Regular type - serialize directly
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
  void bulk(hipc::ShmPtr<T> ptr, size_t size, uint32_t flags) {
    (void)size;   // Unused for local serialization
    (void)flags;  // Unused for local serialization
    // Serialize the ShmPtr value directly (offset and allocator ID)
    serializer_ << ptr.off_.load() << ptr.alloc_id_.major_ << ptr.alloc_id_.minor_;
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
  void bulk(const hipc::FullPtr<T> &ptr, size_t size, uint32_t flags) {
    (void)size;   // Unused for local serialization
    (void)flags;  // Unused for local serialization
    // Serialize only the ShmPtr part (offset and allocator ID)
    serializer_ << ptr.shm_.off_.load() << ptr.shm_.alloc_id_.major_ << ptr.shm_.alloc_id_.minor_;
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
  void bulk(T *ptr, size_t size, uint32_t flags) {
    (void)flags;  // Unused for local serialization
    // Full memory copy for raw pointers
    serializer_.write_binary(reinterpret_cast<const char *>(ptr), size);
  }

  /**
   * Get task information
   *
   * @return Vector of task information
   */
  const std::vector<LocalTaskInfo> &GetTaskInfos() const { return task_infos_; }

  /**
   * Get message type
   *
   * @return Message type
   */
  LocalMsgType GetMsgType() const { return msg_type_; }

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
};

/**
 * Archive for loading tasks (inputs or outputs) using LocalDeserialize
 * Local version that uses hshm::ipc::LocalDeserialize instead of cereal
 */
class LocalLoadTaskArchive {
public:
  std::vector<LocalTaskInfo> task_infos_;
  LocalMsgType msg_type_; /**< Message type: kSerializeIn or kSerializeOut */

private:
  const std::vector<char> *data_;
  hshm::ipc::LocalDeserialize<std::vector<char>> deserializer_;
  size_t current_task_index_;

public:
  /**
   * Default constructor
   */
  LocalLoadTaskArchive()
      : msg_type_(LocalMsgType::kSerializeIn), data_(nullptr),
        deserializer_(empty_buffer_), current_task_index_(0) {}

  /**
   * Constructor from serialized data
   *
   * @param data Buffer containing serialized data
   */
  explicit LocalLoadTaskArchive(const std::vector<char> &data)
      : msg_type_(LocalMsgType::kSerializeIn), data_(&data),
        deserializer_(data), current_task_index_(0) {}

  /** Move constructor */
  LocalLoadTaskArchive(LocalLoadTaskArchive &&other) noexcept
      : task_infos_(std::move(other.task_infos_)), msg_type_(other.msg_type_),
        data_(other.data_), deserializer_(other.data_ ? *other.data_ : empty_buffer_),
        current_task_index_(other.current_task_index_) {
    other.data_ = nullptr;
  }

  /** Move assignment operator - not supported due to reference member in deserializer */
  LocalLoadTaskArchive &operator=(LocalLoadTaskArchive &&other) noexcept = delete;

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
  template <typename T> LocalLoadTaskArchive &operator>>(T &value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // Call Serialize* for Task-derived objects
      // Task::SerializeIn/SerializeOut will handle base class fields
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
   * Deserialize task pointers
   *
   * @tparam T Task type
   * @param value Pointer to deserialize into (must be pre-allocated)
   * @return Reference to this archive for chaining
   */
  template <typename T> LocalLoadTaskArchive &operator>>(T *&value) {
    if constexpr (std::is_base_of_v<Task, T>) {
      // value must be pre-allocated by caller using CHI_IPC->NewTask
      // Deserialize task based on mode
      // Task::SerializeIn/SerializeOut will handle base class fields
      if (msg_type_ == LocalMsgType::kSerializeIn) {
        // SerializeIn mode - deserialize input parameters
        value->SerializeIn(*this);
      } else if (msg_type_ == LocalMsgType::kSerializeOut) {
        // SerializeOut mode - deserialize output parameters
        value->SerializeOut(*this);
      }
      current_task_index_++;
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
  template <typename... Args> void operator()(Args &...args) {
    (DeserializeArg(args), ...);
  }

private:
  /** Helper to deserialize individual arguments - handles Tasks specially */
  template <typename T> void DeserializeArg(T &arg) {
    if constexpr (std::is_base_of_v<Task,
                                    std::remove_pointer_t<std::decay_t<T>>>) {
      // This is a Task or Task pointer - use operator>> which handles tasks
      *this >> arg;
    } else {
      // Regular type - deserialize directly
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
    (void)size;   // Unused for local deserialization
    (void)flags;  // Unused for local deserialization
    // Deserialize the ShmPtr value (offset and allocator ID)
    size_t off;
    u32 major, minor;
    deserializer_ >> off >> major >> minor;

    ptr.off_ = off;
    ptr.alloc_id_ = hipc::AllocatorId(major, minor);
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
    (void)size;   // Unused for local deserialization
    (void)flags;  // Unused for local deserialization
    // Deserialize only the ShmPtr part (offset and allocator ID)
    size_t off;
    u32 major, minor;
    deserializer_ >> off >> major >> minor;

    ptr.shm_.off_ = off;
    ptr.shm_.alloc_id_ = hipc::AllocatorId(major, minor);
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
    (void)flags;  // Unused for local deserialization
    // Full memory copy for raw pointers
    deserializer_.read_binary(reinterpret_cast<char *>(ptr), size);
  }

  /**
   * Get task information
   *
   * @return Vector of task information
   */
  const std::vector<LocalTaskInfo> &GetTaskInfos() const { return task_infos_; }

  /**
   * Get current task info
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
   * Reset task index for iteration
   */
  void ResetTaskIndex() { current_task_index_ = 0; }

  /**
   * Set message type
   *
   * @param msg_type Message type
   */
  void SetMsgType(LocalMsgType msg_type) { msg_type_ = msg_type; }

private:
  static inline std::vector<char> empty_buffer_;
};

} // namespace chi

#endif // CHIMAERA_INCLUDE_CHIMAERA_LOCAL_TASK_ARCHIVES_H_
