#ifndef SIMPLE_MOD_TASKS_H_
#define SIMPLE_MOD_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_tasks.h>

#include "autogen/simple_mod_methods.h"

/**
 * Task struct definitions for Simple Mod ChiMod
 *
 * Minimal ChiMod for testing external development patterns.
 * Demonstrates basic task structure for external ChiMod development.
 */

namespace external_test::simple_mod {

/**
 * CreateParams for simple_mod chimod
 * Contains configuration parameters for simple_mod container creation
 */
struct CreateParams {
  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "external_test_simple_mod";

  // Default constructor
  CreateParams() = default;

  // Constructor with allocator
  explicit CreateParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc) {
    (void)alloc;  // Simple mod doesn't need allocator-based initialization
  }

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    // No additional fields to serialize for simple_mod
    (void)ar;
  }
};

/**
 * CreateTask - Simple mod container creation task
 * Uses the standard BaseCreateTask template from admin module
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * Standard DestroyTask for simple_mod
 * Uses the reusable DestroyTask from admin module
 */
using DestroyTask = chimaera::admin::DestroyTask;

/**
 * FlushTask - Simple flush task for simple_mod
 * Minimal task with no additional inputs beyond basic task parameters
 */
struct FlushTask : public chi::Task {
  // Output results
  OUT chi::u64 total_work_done_;  ///< Total amount of work completed

  /** SHM default constructor */
  explicit FlushTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), total_work_done_(0) {}

  /** Emplace constructor */
  explicit FlushTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                     const chi::TaskId &task_node, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query)
      : chi::Task(alloc, task_node, pool_id, pool_query, 10),
        total_work_done_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFlush;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * No additional parameters for FlushTask
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    // No parameters to serialize for flush
    (void)ar;
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: total_work_done_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    ar(total_work_done_);
  }
};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_TASKS_H_