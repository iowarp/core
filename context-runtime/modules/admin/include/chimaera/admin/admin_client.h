#ifndef ADMIN_CLIENT_H_
#define ADMIN_CLIENT_H_

#include <chimaera/chimaera.h>
#include <chrono>
#include <unistd.h>

#include "admin_tasks.h"

/**
 * Client API for Admin ChiMod
 *
 * Critical ChiMod for managing ChiPools and runtime lifecycle.
 * Provides methods for external programs to create/destroy pools and stop
 * runtime.
 */

namespace chimaera::admin {

class Client : public chi::ContainerClient {
 public:
  /**
   * Default constructor
   */
  Client() = default;

  /**
   * Constructor with pool ID
   */
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the Admin container (synchronous)
   * @param mctx Memory context for the operation
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the admin pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   * @return true if creation succeeded, false if it failed
   */
  bool Create(const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
              const std::string& pool_name, const chi::PoolId& custom_pool_id) {
    auto task = AsyncCreate(mctx, pool_query, pool_name, custom_pool_id);
    task->Wait();

    // CRITICAL: Update client pool_id_ with the actual pool ID from the task
    pool_id_ = task->new_pool_id_;

    // Store the return code from the Create task in the client
    return_code_ = task->return_code_;

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);

    // Return true for success (return_code_ == 0), false for failure
    return return_code_ == 0;
  }

  /**
   * Create the Admin container (asynchronous)
   * @param mctx Memory context for the operation
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the admin pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   */
  hipc::FullPtr<CreateTask> AsyncCreate(const hipc::MemContext& mctx,
                                        const chi::PoolQuery& pool_query,
                                        const std::string& pool_name,
                                        const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;

    // Allocate CreateTask for admin container creation
    // Note: Admin uses BaseCreateTask pattern, not GetOrCreatePoolTask
    // The custom_pool_id is the ID for the pool being created (not the task pool)
    auto task = ipc_manager->NewTask<CreateTask>(chi::CreateTaskId(),
                                                 chi::kAdminPoolId, pool_query, "", pool_name, custom_pool_id);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Destroy an existing ChiPool (synchronous)
   */
  void DestroyPool(const hipc::MemContext& mctx,
                   const chi::PoolQuery& pool_query, chi::PoolId target_pool_id,
                   chi::u32 destruction_flags = 0) {
    auto task =
        AsyncDestroyPool(mctx, pool_query, target_pool_id, destruction_flags);
    task->Wait();

    // Check for errors
    if (task->return_code_ != 0) {
      std::string error = task->error_message_.str();
      auto* ipc_manager = CHI_IPC;
      ipc_manager->DelTask(task);
      throw std::runtime_error("Pool destruction failed: " + error);
    }

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
  }

  /**
   * Destroy an existing ChiPool (asynchronous)
   */
  hipc::FullPtr<DestroyPoolTask> AsyncDestroyPool(
      const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
      chi::PoolId target_pool_id, chi::u32 destruction_flags = 0) {
    auto* ipc_manager = CHI_IPC;

    // Allocate DestroyPoolTask
    auto task = ipc_manager->NewTask<DestroyPoolTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_pool_id,
        destruction_flags);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Send task to remote nodes (synchronous)
   * Can be used for both SerializeIn (sending inputs) and SerializeOut (sending outputs)
   */
  template <typename TaskType>
  void Send(const hipc::MemContext& mctx,
            chi::MsgType msg_type,
            const hipc::FullPtr<TaskType>& subtask,
            const std::vector<chi::PoolQuery>& pool_queries,
            chi::u32 transfer_flags = 0) {
    auto task = AsyncSend(mctx, msg_type, subtask, pool_queries, transfer_flags);
    task->Wait();

    // Check for errors
    if (task->GetReturnCode() != 0) {
      std::string error = task->error_message_.str();
      auto* ipc_manager = CHI_IPC;
      ipc_manager->DelTask(task);
      throw std::runtime_error("Send failed: " + error);
    }

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
  }

  /**
   * Send task to remote nodes (asynchronous)
   * Can be used for SerializeIn (sending inputs), SerializeOut (sending outputs), or Heartbeat
   */
  template <typename TaskType>
  hipc::FullPtr<SendTask> AsyncSend(
      const hipc::MemContext& mctx,
      chi::MsgType msg_type,
      const hipc::FullPtr<TaskType>& subtask,
      const std::vector<chi::PoolQuery>& pool_queries,
      chi::u32 transfer_flags = 0) {
    auto* ipc_manager = CHI_IPC;

    // Use local routing
    chi::PoolQuery local_pool_query = chi::PoolQuery::Local();

    // Cast subtask to base Task type
    hipc::FullPtr<chi::Task> base_subtask = subtask.template Cast<chi::Task>();

    // Allocate SendTask
    auto task = ipc_manager->NewTask<SendTask>(
        chi::CreateTaskId(), pool_id_, local_pool_query,
        msg_type, base_subtask, pool_queries, transfer_flags);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Receive tasks from network (synchronous)
   * Can be used for both SerializeIn (receiving inputs) and SerializeOut (receiving outputs)
   */
  void Recv(const hipc::MemContext& mctx,
            const chi::PoolQuery& pool_query,
            chi::u32 transfer_flags = 0) {
    auto task = AsyncRecv(mctx, pool_query, transfer_flags);
    task->Wait();

    // Check for errors
    if (task->GetReturnCode() != 0) {
      std::string error = task->error_message_.str();
      auto* ipc_manager = CHI_IPC;
      ipc_manager->DelTask(task);
      throw std::runtime_error("Recv failed: " + error);
    }

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
  }

  /**
   * Receive tasks from network (asynchronous)
   * Can be used for both SerializeIn (receiving inputs) and SerializeOut (receiving outputs)
   */
  hipc::FullPtr<RecvTask> AsyncRecv(
      const hipc::MemContext& mctx,
      const chi::PoolQuery& pool_query,
      chi::u32 transfer_flags = 0,
      double period_us = 25) {
    auto* ipc_manager = CHI_IPC;

    // Allocate RecvTask
    auto task = ipc_manager->NewTask<RecvTask>(
        chi::CreateTaskId(), pool_id_, pool_query, transfer_flags);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Flush administrative operations (synchronous)
   */
  void Flush(const hipc::MemContext& mctx, const chi::PoolQuery& pool_query) {
    auto task = AsyncFlush(mctx, pool_query);
    task->Wait();

    // Check for errors
    if (task->return_code_ != 0) {
      auto* ipc_manager = CHI_IPC;
      ipc_manager->DelTask(task);
      throw std::runtime_error("Flush failed with result code: " +
                               std::to_string(task->return_code_));
    }

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
  }

  /**
   * Flush administrative operations (asynchronous)
   */
  hipc::FullPtr<FlushTask> AsyncFlush(const hipc::MemContext& mctx,
                                      const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;

    // Allocate FlushTask
    auto task = ipc_manager->NewTask<FlushTask>(chi::CreateTaskId(), pool_id_,
                                                pool_query);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Stop the entire Chimaera runtime (asynchronous)
   */
  hipc::FullPtr<StopRuntimeTask> AsyncStopRuntime(
      const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
      chi::u32 shutdown_flags = 0, chi::u32 grace_period_ms = 5000) {
    auto* ipc_manager = CHI_IPC;

    // Allocate StopRuntimeTask
    auto task = ipc_manager->NewTask<StopRuntimeTask>(
        chi::CreateTaskId(), pool_id_, pool_query, shutdown_flags,
        grace_period_ms);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Compose - Create multiple pools from compose configuration
   * Iterates over pools and creates them one-by-one synchronously
   * @param compose_config Configuration with list of pools to create
   * @return true if all pools created successfully, false otherwise
   */
  bool Compose(const chi::ComposeConfig& compose_config) {
    auto* ipc_manager = CHI_IPC;

    // Iterate over each pool configuration
    for (const auto& pool_config : compose_config.pools_) {
      HILOG(kInfo, "Compose: Creating pool {} (module: {})",
            pool_config.pool_name_, pool_config.mod_name_);

      // Create ComposeTask with PoolConfig passed directly to constructor
      auto task = ipc_manager->NewTask<chimaera::admin::ComposeTask<chi::PoolConfig>>(
          chi::CreateTaskId(),
          chi::kAdminPoolId,
          pool_config.pool_query_,
          pool_config
      );

      // Submit and wait for completion
      ipc_manager->Enqueue(task);
      task->Wait();

      // Check return code
      chi::u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        HELOG(kError, "Compose: Failed to create pool {} (module: {}), return code: {}",
              pool_config.pool_name_, pool_config.mod_name_, return_code);
        ipc_manager->DelTask(task);
        return false;
      }

      HILOG(kInfo, "Compose: Successfully created pool {} (module: {})",
            pool_config.pool_name_, pool_config.mod_name_);

      // Cleanup task
      ipc_manager->DelTask(task);
    }

    HILOG(kInfo, "Compose: All {} pools created successfully", compose_config.pools_.size());
    return true;
  }

private:
  /**
   * Generate a unique pool name with a given prefix
   * Uses timestamp and process ID to ensure uniqueness
   */
  static std::string GeneratePoolName(const std::string& prefix) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    pid_t pid = getpid();
    return prefix + "_" + std::to_string(timestamp) + "_" + std::to_string(pid);
  }
};

}  // namespace chimaera::admin

#endif  // ADMIN_CLIENT_H_