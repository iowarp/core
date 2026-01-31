// Copyright 2024 IOWarp contributors
#ifndef WRP_CTE_COMPRESSOR_CLIENT_H_
#define WRP_CTE_COMPRESSOR_CLIENT_H_

#include <chimaera/chimaera.h>
#include <hermes_shm/util/singleton.h>
#include <wrp_cte/compressor/compressor_tasks.h>

namespace wrp_cte::compressor {

/**
 * Compressor client for asynchronous compression operations
 * Provides async API for DynamicSchedule, Compress, and Decompress tasks
 */
class Client : public chi::ContainerClient {
 public:
  Client() = default;
  explicit Client(const chi::PoolId &pool_id) { Init(pool_id); }

  /**
   * Create the compressor container
   * @param pool_query Task routing strategy
   * @param pool_name Name of the pool
   * @param custom_pool_id Explicit pool ID for the container
   * @return Future for CreateTask
   */
  chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                       const std::string& pool_name,
                                       const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        "wrp_cte_compressor",  // ChiMod library name
        pool_name,
        custom_pool_id,
        this);  // Client pointer for PostWait callback
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous monitor - polls core for target information
   * @param core_pool_id Pool ID of core chimod to monitor
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future for MonitorTask
   */
  chi::Future<MonitorTask> AsyncMonitor(
      const chi::PoolId &core_pool_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<MonitorTask>(
        chi::CreateTaskId(), pool_id_, pool_query, core_pool_id);

    return ipc_manager->Send(task);
  }

  /**
   * Synchronous monitor - blocks until complete
   */
  void Monitor(
      const chi::PoolId &core_pool_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto future = AsyncMonitor(core_pool_id, pool_query);
    future.Wait();
  }

  /**
   * Asynchronous dynamic scheduling - analyzes data and determines optimal compression
   * @param chunk_size Size of data chunk to analyze
   * @param chunk_data Pointer to data chunk
   * @param context Compression context (updated with predictions)
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future for DynamicScheduleTask
   */
  chi::Future<DynamicScheduleTask> AsyncDynamicSchedule(
      chi::u64 chunk_size,
      void *chunk_data,
      const Context &context,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DynamicScheduleTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        chunk_size, chunk_data, context);

    return ipc_manager->Send(task);
  }

  /**
   * Synchronous dynamic scheduling - blocks until complete
   */
  void DynamicSchedule(
      chi::u64 chunk_size,
      void *chunk_data,
      Context &context,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto future = AsyncDynamicSchedule(chunk_size, chunk_data, context, pool_query);
    future.Wait();
    context = future->context_;
  }

  /**
   * Asynchronous compression - performs actual compression
   * @param input_data Input data pointer
   * @param input_size Input data size
   * @param context Compression context (library, preset, etc.)
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future for CompressTask
   */
  chi::Future<CompressTask> AsyncCompress(
      void *input_data,
      chi::u64 input_size,
      const Context &context,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<CompressTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        input_data, input_size, context);

    return ipc_manager->Send(task);
  }

  /**
   * Synchronous compression - blocks until complete
   * @param input_data Input data pointer
   * @param input_size Input data size
   * @param context Compression context (library, preset, etc.)
   * @param output_data Output compressed data (allocated by compressor)
   * @param output_size Output compressed size
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return 0 on success, error code otherwise
   */
  int Compress(
      void *input_data,
      chi::u64 input_size,
      const Context &context,
      void *&output_data,
      chi::u64 &output_size,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto future = AsyncCompress(input_data, input_size, context, pool_query);
    future.Wait();
    output_data = future->output_data_;
    output_size = future->output_size_;
    return future->return_code_;
  }

  /**
   * Asynchronous decompression - performs decompression
   * @param input_data Compressed data pointer
   * @param input_size Compressed data size
   * @param expected_output_size Expected decompressed size
   * @param compress_lib Compression library used
   * @param compress_preset Compression preset used
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future for DecompressTask
   */
  chi::Future<DecompressTask> AsyncDecompress(
      void *input_data,
      chi::u64 input_size,
      chi::u64 expected_output_size,
      int compress_lib,
      int compress_preset,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DecompressTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        input_data, input_size, expected_output_size,
        compress_lib, compress_preset);

    return ipc_manager->Send(task);
  }

  /**
   * Synchronous decompression - blocks until complete
   * @param input_data Compressed data pointer
   * @param input_size Compressed data size
   * @param expected_output_size Expected decompressed size
   * @param compress_lib Compression library used
   * @param compress_preset Compression preset used
   * @param output_data Output decompressed data (allocated by compressor)
   * @param output_size Actual decompressed size
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return 0 on success, error code otherwise
   */
  int Decompress(
      void *input_data,
      chi::u64 input_size,
      chi::u64 expected_output_size,
      int compress_lib,
      int compress_preset,
      void *&output_data,
      chi::u64 &output_size,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto future = AsyncDecompress(input_data, input_size, expected_output_size,
                                   compress_lib, compress_preset, pool_query);
    future.Wait();
    output_data = future->output_data_;
    output_size = future->output_size_;
    return future->return_code_;
  }
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_CLIENT_H_
