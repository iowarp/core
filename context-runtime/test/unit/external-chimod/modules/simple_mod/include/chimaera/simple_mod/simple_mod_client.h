#ifndef SIMPLE_MOD_CLIENT_H_
#define SIMPLE_MOD_CLIENT_H_

#include <chimaera/chimaera.h>

#include "simple_mod_tasks.h"

/**
 * Client API for Simple Mod ChiMod
 *
 * Minimal ChiMod for testing external development patterns.
 * Demonstrates basic client API structure for external ChiMod development.
 */

namespace external_test::simple_mod {

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
   * Create the Simple Mod container (synchronous)
   */
  void Create(const hipc::MemContext& mctx, const chi::PoolQuery& pool_query) {
    auto task = AsyncCreate(mctx, pool_query);
    task->Wait();

    // Check for errors
    if (task->return_code_ != 0) {
      std::string error = task->error_message_.str();
      auto* ipc_manager = CHI_IPC;
      ipc_manager->DelTask(task);
      throw std::runtime_error("Simple mod creation failed: " + error);
    }

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
  }

  /**
   * Create the Simple Mod container (asynchronous)
   */
  hipc::FullPtr<CreateTask> AsyncCreate(const hipc::MemContext& mctx,
                                        const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;

    // Use admin pool for CreateTask as per CLAUDE.md requirements
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, pool_query,
        "external_test_simple_mod", "simple_mod_pool", pool_id_);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Destroy the Simple Mod container (synchronous)
   */
  void Destroy(const hipc::MemContext& mctx, const chi::PoolQuery& pool_query) {
    auto task = AsyncDestroy(mctx, pool_query);
    task->Wait();

    // Check for errors
    if (task->return_code_ != 0) {
      std::string error = task->error_message_.str();
      auto* ipc_manager = CHI_IPC;
      ipc_manager->DelTask(task);
      throw std::runtime_error("Simple mod destruction failed: " + error);
    }

    // Clean up task
    auto* ipc_manager = CHI_IPC;
    ipc_manager->DelTask(task);
  }

  /**
   * Destroy the Simple Mod container (asynchronous)
   */
  hipc::FullPtr<DestroyTask> AsyncDestroy(const hipc::MemContext& mctx,
                                          const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;

    // Allocate DestroyTask
    auto task = ipc_manager->NewTask<DestroyTask>(chi::CreateTaskId(), pool_id_,
                                                  pool_query, pool_id_, 0);

    // Submit to runtime
    ipc_manager->Enqueue(task);

    return task;
  }

  /**
   * Flush simple mod operations (synchronous)
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
   * Flush simple mod operations (asynchronous)
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
};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_CLIENT_H_