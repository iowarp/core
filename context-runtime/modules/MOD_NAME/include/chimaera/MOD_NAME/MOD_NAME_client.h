#ifndef MOD_NAME_CLIENT_H_
#define MOD_NAME_CLIENT_H_

#include <chimaera/chimaera.h>
#include <chrono>
#include <unistd.h>

#include "MOD_NAME_tasks.h"

/**
 * Client API for MOD_NAME
 *
 * Provides methods for external programs to submit tasks to the runtime.
 */

namespace chimaera::MOD_NAME {

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
   * Create the container (synchronous)
   * @param mctx Memory context for the operation
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the pool (user-provided)
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
   * Create the container (asynchronous)
   * @param mctx Memory context for the operation
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   */
  hipc::FullPtr<CreateTask> AsyncCreate(const hipc::MemContext& mctx,
                                        const chi::PoolQuery& pool_query,
                                        const std::string& pool_name,
                                        const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;

    // CreateTask is a GetOrCreatePoolTask, which must be handled by admin pool
    // So we send it to admin pool (chi::kAdminPoolId), not to our target
    // pool_id_
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Send to admin pool for GetOrCreatePool processing
        pool_query,
        CreateParams::chimod_lib_name,  // chimod name from CreateParams
        pool_name,  // user-provided pool name
        custom_pool_id    // target pool ID to create (explicit from user)
    );

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Execute custom operation (synchronous)
   */
  chi::u32 Custom(const hipc::MemContext& mctx,
                  const chi::PoolQuery& pool_query,
                  const std::string& input_data, chi::u32 operation_id,
                  std::string& output_data) {
    auto task = AsyncCustom(mctx, pool_query, input_data, operation_id);
    task->Wait();

    // Get results
    output_data = task->data_.str();
    chi::u32 result_code = task->return_code_;

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);

    return result_code;
  }

  /**
   * Execute custom operation (asynchronous)
   */
  hipc::FullPtr<CustomTask> AsyncCustom(const hipc::MemContext& mctx,
                                        const chi::PoolQuery& pool_query,
                                        const std::string& input_data,
                                        chi::u32 operation_id) {
    auto* ipc_manager = CHI_IPC;

    // Allocate CustomTask
    auto task = ipc_manager->NewTask<CustomTask>(
        chi::CreateTaskId(), pool_id_, pool_query, input_data, operation_id);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Execute CoMutex test (synchronous)
   */
  chi::u32 CoMutexTest(const hipc::MemContext& mctx,
                       const chi::PoolQuery& pool_query, chi::u32 test_id,
                       chi::u32 hold_duration_ms) {
    auto task = AsyncCoMutexTest(mctx, pool_query, test_id, hold_duration_ms);
    task->Wait();

    // Get result
    chi::u32 result = task->return_code_;

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);

    return result;
  }

  /**
   * Execute CoMutex test (asynchronous)
   */
  hipc::FullPtr<CoMutexTestTask> AsyncCoMutexTest(
      const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
      chi::u32 test_id, chi::u32 hold_duration_ms) {
    auto* ipc_manager = CHI_IPC;

    // Allocate CoMutexTestTask
    auto task = ipc_manager->NewTask<CoMutexTestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, test_id, hold_duration_ms);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Execute CoRwLock test (synchronous)
   */
  chi::u32 CoRwLockTest(const hipc::MemContext& mctx,
                        const chi::PoolQuery& pool_query, chi::u32 test_id,
                        bool is_writer, chi::u32 hold_duration_ms) {
    auto task = AsyncCoRwLockTest(mctx, pool_query, test_id, is_writer,
                                  hold_duration_ms);
    task->Wait();

    // Get result
    chi::u32 result = task->return_code_;

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);

    return result;
  }

  /**
   * Execute CoRwLock test (asynchronous)
   */
  hipc::FullPtr<CoRwLockTestTask> AsyncCoRwLockTest(
      const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
      chi::u32 test_id, bool is_writer, chi::u32 hold_duration_ms) {
    auto* ipc_manager = CHI_IPC;

    // Allocate CoRwLockTestTask
    auto task = ipc_manager->NewTask<CoRwLockTestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, test_id, is_writer,
        hold_duration_ms);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Submit Wait test task (asynchronous)
   * Tests recursive task->Wait() functionality with specified depth
   * @param mctx Memory context for the operation
   * @param pool_query Pool routing information  
   * @param depth Number of recursive calls to make
   * @param test_id Test identifier for tracking
   * @return Task handle for waiting and result retrieval
   */
  hipc::FullPtr<WaitTestTask> AsyncWaitTest(const hipc::MemContext& mctx,
                                           const chi::PoolQuery& pool_query,
                                           chi::u32 depth,
                                           chi::u32 test_id) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<WaitTestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, depth, test_id);

    // Submit to runtime
    ipc_manager->Enqueue(task);
    return task;
  }

  /**
   * Submit Wait test task (synchronous)
   * Tests recursive task->Wait() functionality with specified depth
   * @param mctx Memory context for the operation
   * @param pool_query Pool routing information
   * @param depth Number of recursive calls to make  
   * @param test_id Test identifier for tracking
   * @return The final depth reached by the recursive calls
   */
  chi::u32 WaitTest(const hipc::MemContext& mctx,
                   const chi::PoolQuery& pool_query,
                   chi::u32 depth,
                   chi::u32 test_id) {
    auto task = AsyncWaitTest(mctx, pool_query, depth, test_id);
    task->Wait();
    
    chi::u32 final_depth = task->current_depth_;
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
    
    return final_depth;
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

}  // namespace chimaera::MOD_NAME

#endif  // MOD_NAME_CLIENT_H_