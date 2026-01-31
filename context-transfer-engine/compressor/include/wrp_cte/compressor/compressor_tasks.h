// Copyright 2024 IOWarp contributors
#ifndef WRP_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_
#define WRP_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/task.h>
#include <chimaera/admin/admin_tasks.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/compressor/autogen/compressor_methods.h>

namespace wrp_cte::compressor {

/** Import Context from core for compression operations */
using Context = wrp_cte::core::Context;
using CteOp = wrp_cte::core::CteOp;
using Timestamp = std::chrono::steady_clock::time_point;

/**
 * CreateParams - Configuration for compressor container creation
 */
struct CompressorConfig {
  static constexpr const char* chimod_lib_name = "wrp_cte_compressor";

  std::string qtable_model_path_;
  std::string linreg_model_path_;
  std::string distribution_model_path_;
  std::string dnn_model_weights_path_;
  std::string trace_folder_path_;

  CompressorConfig() = default;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(qtable_model_path_, linreg_model_path_, distribution_model_path_,
       dnn_model_weights_path_, trace_folder_path_);
  }
};

/**
 * CreateTask - Use GetOrCreatePoolTask for standard pool creation
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CompressorConfig>;

/**
 * DestroyTask - Cleanup the compressor container
 */
struct DestroyTask : public chi::Task {
  // No additional fields needed
  DestroyTask() : chi::Task() {}

  explicit DestroyTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query)
      : chi::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void Copy(const hipc::FullPtr<DestroyTask>& other) {
    // No additional fields to copy beyond chi::Task
  }

  template <typename Ar> void SerializeStart(Ar &ar) { task_serialize<Ar>(ar); }
  template <typename Ar> void SerializeEnd(Ar &ar) {}
};

/**
 * Target state - cached information about storage targets
 * Used by compressor to make intelligent compression/tiering decisions
 */
struct TargetState {
  std::string target_name_;      // Name of the target
  float target_score_;           // Target score (0-1, normalized log bandwidth)
  chi::u64 remaining_space_;     // Remaining allocatable space in bytes
  chi::u64 bytes_written_;       // Bytes written to target
  Timestamp last_updated_;       // When this state was last refreshed

  TargetState()
      : target_score_(0.0f), remaining_space_(0), bytes_written_(0),
        last_updated_(std::chrono::steady_clock::now()) {}

  TargetState(const std::string &name, float score, chi::u64 space, chi::u64 written)
      : target_name_(name), target_score_(score), remaining_space_(space),
        bytes_written_(written), last_updated_(std::chrono::steady_clock::now()) {}
};

/**
 * MonitorTask - Periodically poll core for target information
 * Updates cached target state (scores, capacities) for compression decisions
 */
struct MonitorTask : public chi::Task {
  IN chi::PoolId core_pool_id_;  // Pool ID of core chimod to monitor

  // SHM constructor
  MonitorTask() : chi::Task() {}

  // Emplace constructor
  explicit MonitorTask(const chi::TaskId &task_id,
                       const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query,
                       const chi::PoolId &core_pool_id)
      : chi::Task(task_id, pool_id, pool_query, Method::kMonitor),
        core_pool_id_(core_pool_id) {}

  void Copy(const hipc::FullPtr<MonitorTask>& other) {
    core_pool_id_ = other->core_pool_id_;
  }

  template <typename Ar> void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(core_pool_id_);
  }
  template <typename Ar> void SerializeEnd(Ar &ar) {}
};

/**
 * Compression telemetry data structure for performance monitoring
 * Tracks compression decisions and actual performance
 */
struct CompressionTelemetry {
  CteOp op_;                     // Operation type (kPutBlob or kGetBlob)
  int compress_lib_;             // Compression library used (0 = none)
  chi::u64 original_size_;       // Original data size in bytes
  chi::u64 compressed_size_;     // Compressed data size in bytes
  double compress_time_ms_;      // Actual compression time in milliseconds
  double decompress_time_ms_;    // Actual decompression time in milliseconds
  double psnr_db_;               // Actual PSNR for lossy compression
  Timestamp timestamp_;          // When operation occurred
  std::uint64_t logical_time_;   // Logical time for ordering

  CompressionTelemetry()
      : op_(CteOp::kPutBlob), compress_lib_(0), original_size_(0),
        compressed_size_(0), compress_time_ms_(0.0), decompress_time_ms_(0.0),
        psnr_db_(0.0), timestamp_(std::chrono::steady_clock::now()),
        logical_time_(0) {}

  CompressionTelemetry(CteOp op, int lib, chi::u64 orig_size, chi::u64 comp_size,
                       double comp_time, double decomp_time, double psnr,
                       const Timestamp &ts, std::uint64_t logical_time = 0)
      : op_(op), compress_lib_(lib), original_size_(orig_size),
        compressed_size_(comp_size), compress_time_ms_(comp_time),
        decompress_time_ms_(decomp_time), psnr_db_(psnr),
        timestamp_(ts), logical_time_(logical_time) {}

  // Calculate compression ratio
  double GetCompressionRatio() const {
    if (compressed_size_ == 0) return 1.0;
    return static_cast<double>(original_size_) / static_cast<double>(compressed_size_);
  }

  // Serialization support for cereal
  template <class Archive> void serialize(Archive &ar) {
    // Convert timestamps to duration counts for serialization
    auto ts_count = timestamp_.time_since_epoch().count();
    ar(op_, compress_lib_, original_size_, compressed_size_,
       compress_time_ms_, decompress_time_ms_, psnr_db_,
       ts_count, logical_time_);
    // Note: On deserialization, timestamps will be reconstructed from counts
    if (Archive::is_loading::value) {
      timestamp_ = Timestamp(Timestamp::duration(ts_count));
    }
  }
};

/**
 * DynamicSchedule task - Analyzes data and determines optimal compression strategy
 */
struct DynamicScheduleTask : public chi::Task {
  IN chi::u64 chunk_size_;           // Size of data chunk to analyze
  IN void *chunk_data_;               // Pointer to data chunk
  INOUT Context context_;             // Compression context (updated with predictions)
  OUT int return_code_;               // 0 on success, error code otherwise

  // SHM constructor
  DynamicScheduleTask()
      : chi::Task(), chunk_size_(0), chunk_data_(nullptr),
        context_(), return_code_(-1) {}

  // Emplace constructor
  explicit DynamicScheduleTask(const chi::TaskId &task_id,
                               const chi::PoolId &pool_id,
                               const chi::PoolQuery &pool_query,
                               chi::u64 chunk_size,
                               void *chunk_data,
                               const Context &context)
      : chi::Task(task_id, pool_id, pool_query, Method::kDynamicSchedule),
        chunk_size_(chunk_size), chunk_data_(chunk_data),
        context_(context), return_code_(-1) {}

  void Copy(const hipc::FullPtr<DynamicScheduleTask>& other) {
    chunk_size_ = other->chunk_size_;
    chunk_data_ = other->chunk_data_;
    context_ = other->context_;
    return_code_ = other->return_code_;
  }

  /** Serialize */
  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(chunk_size_, context_, return_code_);
    // Note: chunk_data_ is not serialized (local pointer)
  }

  /** Deserialize */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {}
};

/**
 * Compress task - Performs actual compression
 */
struct CompressTask : public chi::Task {
  IN void *input_data_;               // Input data pointer
  IN chi::u64 input_size_;            // Input data size
  IN Context context_;                // Compression context (library, preset, etc.)
  OUT void *output_data_;             // Output compressed data (allocated by task)
  OUT chi::u64 output_size_;          // Output compressed size
  OUT double compress_time_ms_;       // Compression time in milliseconds
  OUT int return_code_;               // 0 on success, error code otherwise

  // SHM constructor
  CompressTask()
      : chi::Task(), input_data_(nullptr), input_size_(0),
        context_(), output_data_(nullptr), output_size_(0),
        compress_time_ms_(0.0), return_code_(-1) {}

  // Emplace constructor
  explicit CompressTask(const chi::TaskId &task_id,
                        const chi::PoolId &pool_id,
                        const chi::PoolQuery &pool_query,
                        void *input_data,
                        chi::u64 input_size,
                        const Context &context)
      : chi::Task(task_id, pool_id, pool_query, Method::kCompress),
        input_data_(input_data), input_size_(input_size),
        context_(context), output_data_(nullptr), output_size_(0),
        compress_time_ms_(0.0), return_code_(-1) {}

  void Copy(const hipc::FullPtr<CompressTask>& other) {
    input_data_ = other->input_data_;
    input_size_ = other->input_size_;
    context_ = other->context_;
    output_data_ = other->output_data_;
    output_size_ = other->output_size_;
    compress_time_ms_ = other->compress_time_ms_;
    return_code_ = other->return_code_;
  }

  /** Serialize */
  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(input_size_, context_, output_size_, compress_time_ms_, return_code_);
    // Note: input_data_ and output_data_ are not serialized (local pointers)
  }

  /** Deserialize */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {}
};

/**
 * Decompress task - Performs decompression
 */
struct DecompressTask : public chi::Task {
  IN void *input_data_;               // Compressed data pointer
  IN chi::u64 input_size_;            // Compressed data size
  IN chi::u64 expected_output_size_;  // Expected decompressed size
  IN int compress_lib_;               // Compression library used
  IN int compress_preset_;            // Compression preset used
  OUT void *output_data_;             // Output decompressed data (allocated by task)
  OUT chi::u64 output_size_;          // Actual decompressed size
  OUT double decompress_time_ms_;     // Decompression time in milliseconds
  OUT int return_code_;               // 0 on success, error code otherwise

  // SHM constructor
  DecompressTask()
      : chi::Task(), input_data_(nullptr), input_size_(0),
        expected_output_size_(0), compress_lib_(0), compress_preset_(2),
        output_data_(nullptr), output_size_(0), decompress_time_ms_(0.0),
        return_code_(-1) {}

  // Emplace constructor
  explicit DecompressTask(const chi::TaskId &task_id,
                          const chi::PoolId &pool_id,
                          const chi::PoolQuery &pool_query,
                          void *input_data,
                          chi::u64 input_size,
                          chi::u64 expected_output_size,
                          int compress_lib,
                          int compress_preset)
      : chi::Task(task_id, pool_id, pool_query, Method::kDecompress),
        input_data_(input_data), input_size_(input_size),
        expected_output_size_(expected_output_size),
        compress_lib_(compress_lib), compress_preset_(compress_preset),
        output_data_(nullptr), output_size_(0), decompress_time_ms_(0.0),
        return_code_(-1) {}

  void Copy(const hipc::FullPtr<DecompressTask>& other) {
    input_data_ = other->input_data_;
    input_size_ = other->input_size_;
    expected_output_size_ = other->expected_output_size_;
    compress_lib_ = other->compress_lib_;
    compress_preset_ = other->compress_preset_;
    output_data_ = other->output_data_;
    output_size_ = other->output_size_;
    decompress_time_ms_ = other->decompress_time_ms_;
    return_code_ = other->return_code_;
  }

  /** Serialize */
  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(input_size_, expected_output_size_, compress_lib_, compress_preset_,
       output_size_, decompress_time_ms_, return_code_);
    // Note: input_data_ and output_data_ are not serialized (local pointers)
  }

  /** Deserialize */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {}
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_
