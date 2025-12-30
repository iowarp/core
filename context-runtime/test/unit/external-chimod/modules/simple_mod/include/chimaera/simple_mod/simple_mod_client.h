#ifndef SIMPLE_MOD_CLIENT_H_
#define SIMPLE_MOD_CLIENT_H_

#include <chimaera/chimaera.h>

#include "simple_mod_tasks.h"

/**
 * Client API for Simple Mod ChiMod
 *
 * Minimal ChiMod for testing external development patterns.
 * All methods return Future objects - call Wait() to block for completion.
 * Task cleanup is automatic when Future goes out of scope after Wait().
 */

namespace external_test::simple_mod {

class Client : public chi::ContainerClient {
 public:
  /** Default constructor */
  Client() = default;

  /** Constructor with pool ID */
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the Simple Mod container (asynchronous)
   * @param mctx Memory context
   * @param pool_query Pool routing information
   * @return Future for the CreateTask
   */
  chi::Future<CreateTask> AsyncCreate(const hipc::MemContext& mctx,
                                       const chi::PoolQuery& pool_query) {
    (void)mctx;  // Memory context not needed for task creation
    auto* ipc_manager = CHI_IPC;

    // Use admin pool for CreateTask as per CLAUDE.md requirements
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, pool_query,
        "external_test_simple_mod", "simple_mod_pool", pool_id_,
        this);

    return ipc_manager->Send(task);
  }

  /**
   * Destroy the Simple Mod container (asynchronous)
   * @param mctx Memory context
   * @param pool_query Pool routing information
   * @return Future for the DestroyTask
   */
  chi::Future<DestroyTask> AsyncDestroy(const hipc::MemContext& mctx,
                                         const chi::PoolQuery& pool_query) {
    (void)mctx;  // Memory context not needed for task creation
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<DestroyTask>(chi::CreateTaskId(), pool_id_,
                                                  pool_query, pool_id_, 0);

    return ipc_manager->Send(task);
  }

  /**
   * Flush simple mod operations (asynchronous)
   * @param mctx Memory context
   * @param pool_query Pool routing information
   * @return Future for the FlushTask
   */
  chi::Future<FlushTask> AsyncFlush(const hipc::MemContext& mctx,
                                     const chi::PoolQuery& pool_query) {
    (void)mctx;  // Memory context not needed for task creation
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<FlushTask>(chi::CreateTaskId(), pool_id_,
                                                pool_query);

    return ipc_manager->Send(task);
  }
};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_CLIENT_H_
