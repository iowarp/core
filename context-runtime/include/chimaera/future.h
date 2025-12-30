#ifndef CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_
#define CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_

#include <atomic>
#include <coroutine>
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/data_structures/ipc/shm_container.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "chimaera/types.h"

namespace chi {

// Forward declarations
class Task;
class IpcManager;
struct RunContext;

/**
 * TaskResume - Coroutine return type for runtime methods
 *
 * This lightweight class serves as the return type for ChiMod Run methods
 * that use C++20 coroutines. It holds a coroutine handle and provides
 * methods to resume and check completion status.
 */
class TaskResume {
 public:
  /**
   * Promise type for C++20 coroutine machinery
   *
   * The promise_type defines how the coroutine behaves at various points:
   * - initial_suspend: suspend immediately (lazy start)
   * - final_suspend: resume caller if exists, else suspend
   * - return_void: coroutines return void
   */
  struct promise_type {
    /** Pointer to the RunContext for this coroutine */
    RunContext* run_ctx_ = nullptr;
    /** Handle to the caller coroutine (for nested coroutine support) */
    std::coroutine_handle<> caller_handle_ = nullptr;

    /**
     * Create the TaskResume object from this promise
     * @return TaskResume wrapping the coroutine handle
     */
    TaskResume get_return_object() {
      return TaskResume(
          std::coroutine_handle<promise_type>::from_promise(*this));
    }

    /**
     * Suspend immediately on coroutine start (lazy evaluation)
     * @return Always suspend
     */
    std::suspend_always initial_suspend() noexcept { return {}; }

    /**
     * Awaiter for final_suspend that resumes the caller coroutine
     * if one exists, enabling nested coroutine support.
     */
    struct FinalAwaiter {
      std::coroutine_handle<> caller_;

      bool await_ready() noexcept { return false; }

      /**
       * Resume the caller coroutine if it exists, otherwise use noop
       * @return Handle to resume (caller or noop)
       */
      std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept {
        return caller_ ? caller_ : std::noop_coroutine();
      }

      void await_resume() noexcept {}
    };

    /**
     * Suspend at final suspension point and resume caller if exists
     * @return FinalAwaiter that handles resuming the caller
     */
    FinalAwaiter final_suspend() noexcept {
      return FinalAwaiter{caller_handle_};
    }

    /**
     * Handle void return from coroutine
     */
    void return_void() {}

    /**
     * Handle unhandled exceptions by terminating
     */
    void unhandled_exception() { std::terminate(); }

    /**
     * Set the RunContext for this coroutine
     * @param ctx Pointer to RunContext
     */
    void set_run_context(RunContext* ctx) { run_ctx_ = ctx; }

    /**
     * Get the RunContext for this coroutine
     * @return Pointer to RunContext
     */
    RunContext* get_run_context() const { return run_ctx_; }

    /**
     * Set the caller coroutine handle
     * @param caller Handle to the caller coroutine
     */
    void set_caller(std::coroutine_handle<> caller) { caller_handle_ = caller; }
  };

  using handle_type = std::coroutine_handle<promise_type>;

 private:
  /** The coroutine handle */
  handle_type handle_;
  /** Stored caller handle for await_resume to update run_ctx */
  std::coroutine_handle<> caller_handle_ = nullptr;

 public:
  /**
   * Construct from coroutine handle
   * @param h The coroutine handle
   */
  explicit TaskResume(handle_type h) : handle_(h) {}

  /**
   * Default constructor - null handle
   */
  TaskResume() : handle_(nullptr) {}

  /**
   * Move constructor
   * @param other TaskResume to move from
   */
  TaskResume(TaskResume&& other) noexcept
      : handle_(other.handle_), caller_handle_(other.caller_handle_) {
    other.handle_ = nullptr;
    other.caller_handle_ = nullptr;
  }

  /**
   * Move assignment operator
   * @param other TaskResume to move from
   * @return Reference to this
   */
  TaskResume& operator=(TaskResume&& other) noexcept {
    if (this != &other) {
      if (handle_) {
        handle_.destroy();
      }
      handle_ = other.handle_;
      caller_handle_ = other.caller_handle_;
      other.handle_ = nullptr;
      other.caller_handle_ = nullptr;
    }
    return *this;
  }

  /** Disable copy constructor */
  TaskResume(const TaskResume&) = delete;

  /** Disable copy assignment */
  TaskResume& operator=(const TaskResume&) = delete;

  /**
   * Destructor - destroys the coroutine handle
   */
  ~TaskResume() {
    if (handle_) {
      handle_.destroy();
    }
  }

  /**
   * Get the coroutine handle
   * @return The coroutine handle
   */
  handle_type get_handle() const { return handle_; }

  /**
   * Check if coroutine is done
   * @return True if coroutine has completed
   */
  bool done() const { return handle_ && handle_.done(); }

  /**
   * Resume the coroutine
   */
  void resume() {
    if (handle_ && !handle_.done()) {
      handle_.resume();
    }
  }

  /**
   * Destroy the coroutine handle manually
   */
  void destroy() {
    if (handle_) {
      handle_.destroy();
      handle_ = nullptr;
    }
  }

  /**
   * Check if the handle is valid
   * @return True if handle is not null
   */
  explicit operator bool() const { return handle_ != nullptr; }

  /**
   * Release ownership of the handle without destroying it
   * @return The coroutine handle
   */
  handle_type release() {
    handle_type h = handle_;
    handle_ = nullptr;
    return h;
  }

  // ============================================================
  // Awaiter interface - allows TaskResume to be used with co_await
  // ============================================================

  /**
   * Check if the coroutine is already done
   * @return True if coroutine completed, false otherwise
   */
  bool await_ready() const noexcept {
    return handle_ && handle_.done();
  }

  /**
   * Suspend the calling coroutine and run this one to completion or suspension
   *
   * This runs the inner coroutine (TaskResume) until it either:
   * - Completes (co_return)
   * - Suspends at a co_await (is_yielded_ is true)
   *
   * IMPORTANT: Propagates the RunContext from the caller to the inner coroutine
   * so that nested co_await calls on Futures work correctly.
   *
   * When the inner coroutine suspends on a Future, the caller is also suspended.
   * When the awaited Future completes, the inner coroutine's handle (stored in
   * run_ctx->coro_handle_) is resumed. The await_resume of this TaskResume will
   * then continue running the inner coroutine to completion.
   *
   * @tparam PromiseT The promise type of the calling coroutine
   * @param caller_handle The coroutine handle of the caller
   * @return True if we should suspend (inner suspended), false if inner completed
   */
  template<typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> caller_handle) noexcept {
    if (!handle_) {
      return false;  // Nothing to run, don't suspend
    }

    // Store caller handle for await_resume to use when updating run_ctx
    caller_handle_ = caller_handle;

    // CRITICAL: Propagate RunContext from caller to inner coroutine
    // This allows nested co_await on Futures to properly suspend
    RunContext* caller_run_ctx = caller_handle.promise().get_run_context();
    if (caller_run_ctx) {
      handle_.promise().set_run_context(caller_run_ctx);
    }

    // NOTE: We do NOT set caller_handle in inner's promise yet!
    // If the inner coroutine completes synchronously during resume(),
    // final_suspend would try to resume caller while we're still inside await_suspend,
    // causing undefined behavior. We only set it after confirming suspension.

    // Resume the inner coroutine
    handle_.resume();

    // Check if inner coroutine is done
    if (handle_.done()) {
      // Inner completed synchronously, destroy it
      handle_.destroy();
      handle_ = nullptr;
      return false;  // Don't suspend caller
    }

    // Inner coroutine suspended (on co_await Future or yield)
    // NOW it's safe to set caller_handle - the inner will complete asynchronously
    // and final_suspend will properly resume the caller
    handle_.promise().set_caller(caller_handle);

    // The inner's handle is now stored in run_ctx->coro_handle_ by Future::await_suspend
    // When the awaited Future completes, worker will resume inner via run_ctx->coro_handle_
    // When inner eventually completes, final_suspend will resume the caller (this coroutine)
    return true;
  }

  /**
   * Resume after await - cleanup inner coroutine and update run_ctx
   *
   * This is called when the caller is resumed after the inner coroutine completes.
   * The inner's final_suspend resumes the caller, which triggers this method.
   * We need to:
   * 1. Get run_ctx from inner's promise (before destroying)
   * 2. Destroy inner's handle (it's done)
   * 3. Update run_ctx->coro_handle_ to caller's handle so subsequent events
   *    properly resume the caller (outer) coroutine
   *
   * Note: The implementation is in await_resume_impl<> to defer instantiation
   * until RunContext is fully defined (avoiding circular include issues).
   */
  void await_resume() noexcept {
    await_resume_impl<void>();
  }

 private:
  /**
   * Implementation of await_resume, templated to defer instantiation
   * @tparam T Unused template parameter for deferred instantiation
   */
  template<typename T = void>
  void await_resume_impl() noexcept {
    // Get run_ctx from inner's promise before destroying
    RunContext* run_ctx = nullptr;
    if (handle_) {
      run_ctx = handle_.promise().get_run_context();
      // Inner coroutine is done (final_suspend just resumed us), destroy it
      handle_.destroy();
      handle_ = nullptr;
    }

    // Update run_ctx->coro_handle_ to caller's handle
    // This ensures if caller suspends again on another co_await,
    // or if caller completes, the worker can properly handle it
    if (run_ctx && caller_handle_) {
      run_ctx->coro_handle_ = caller_handle_;
    }
  }
};

/**
 * FutureShm - Shared memory container for task future state
 *
 * This container holds the serialized task data and completion status
 * for asynchronous task operations.
 */
template<typename AllocT = CHI_MAIN_ALLOC_T>
class FutureShm : public hipc::ShmContainer<AllocT> {
 public:
  /** Pool ID for the task */
  PoolId pool_id_;

  /** Method ID for the task */
  u32 method_id_;

  /** Serialized task data */
  hipc::vector<char, AllocT> serialized_task_;

  /** Atomic completion flag (0=not complete, 1=complete) */
  std::atomic<u32> is_complete_;

  /**
   * SHM default constructor
   */
  explicit FutureShm(AllocT* alloc)
      : hipc::ShmContainer<AllocT>(alloc),
        serialized_task_(alloc) {
    pool_id_ = PoolId::GetNull();
    method_id_ = 0;
    is_complete_.store(0);
  }
};

/**
 * Future - Template class for asynchronous task operations
 *
 * Future provides a handle to an asynchronous task operation, allowing
 * the caller to check completion status and retrieve results.
 *
 * @tparam TaskT The task type (e.g., CreateTask, CustomTask)
 * @tparam AllocT The allocator type (defaults to CHI_MAIN_ALLOC_T)
 */
template<typename TaskT, typename AllocT = CHI_MAIN_ALLOC_T>
class Future {
 public:
  using FutureT = FutureShm<AllocT>;

  // Allow all Future instantiations to access each other's private members
  // This enables the Cast method to work across different task types
  template<typename OtherTaskT, typename OtherAllocT>
  friend class Future;

 private:
  /** FullPtr to the task (wraps private memory with null allocator) */
  hipc::FullPtr<TaskT> task_ptr_;

  /** FullPtr to the shared FutureShm object */
  hipc::FullPtr<FutureT> future_shm_;

  /** Parent task RunContext pointer (nullptr if no parent waiting) */
  RunContext* parent_task_;

  /** Flag indicating if this Future owns the task and should destroy it */
  bool is_owner_;

 public:
  /**
   * Constructor with allocator - allocates new FutureShm
   * @param alloc Allocator to use for FutureShm allocation
   * @param task_ptr FullPtr to the task (wraps private memory with null allocator)
   */
  Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr)
      : task_ptr_(task_ptr),
        parent_task_(nullptr),
        is_owner_(false) {
    // Allocate FutureShm object
    future_shm_ = alloc->template NewObj<FutureT>(alloc).template Cast<FutureT>();
    // Copy pool_id to FutureShm
    if (!task_ptr_.IsNull() && !future_shm_.IsNull()) {
      future_shm_->pool_id_ = task_ptr_->pool_id_;
      future_shm_->method_id_ = task_ptr_->method_;
    }
  }

  /**
   * Constructor with allocator and existing FutureShm
   * @param alloc Allocator (for FullPtr construction)
   * @param task_ptr FullPtr to the task (wraps private memory with null allocator)
   * @param future_shm ShmPtr to existing FutureShm object
   */
  Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr, hipc::ShmPtr<FutureT> future_shm)
      : task_ptr_(task_ptr),
        future_shm_(alloc, future_shm),
        parent_task_(nullptr),
        is_owner_(false) {}

  /**
   * Constructor from FullPtr<FutureShm> and FullPtr<Task>
   * @param future_shm FullPtr to existing FutureShm object
   * @param task_ptr FullPtr to the task (wraps private memory with null allocator)
   */
  Future(hipc::FullPtr<FutureT> future_shm, hipc::FullPtr<TaskT> task_ptr)
      : task_ptr_(task_ptr),
        future_shm_(future_shm),
        parent_task_(nullptr),
        is_owner_(false) {
    // No need to copy pool_id - FutureShm already has it
  }

  /**
   * Default constructor - creates null future
   */
  Future() : parent_task_(nullptr), is_owner_(false) {}

  /**
   * Constructor from ShmPtr<FutureShm> - used by ring buffer deserialization
   * Task pointer will be null and must be set later
   * @param future_shm_ptr ShmPtr to FutureShm object
   */
  explicit Future(const hipc::ShmPtr<FutureT>& future_shm_ptr)
      : future_shm_(nullptr, future_shm_ptr),
        parent_task_(nullptr),
        is_owner_(false) {
    // Task pointer starts null - will be set in ProcessNewTasks
    task_ptr_.SetNull();
  }

  /**
   * Fix the allocator pointer after construction from ShmPtr
   * Call this immediately after popping from ring buffer
   * @param alloc Allocator to use for FullPtr
   */
  void SetAllocator(AllocT* alloc) {
    // Reconstruct the FullPtr with the allocator
    future_shm_ = hipc::FullPtr<FutureT>(alloc, future_shm_.shm_);
  }

  /**
   * Destructor - destroys the task if this Future owns it
   */
  ~Future() {
    if (is_owner_) {
      Destroy();
    }
  }

  /**
   * Destroy the task using CHI_IPC->DelTask if not null
   * Sets the task pointer to null afterwards
   */
  void Destroy();

  /**
   * Copy constructor - does not transfer ownership
   * @param other Future to copy from
   */
  Future(const Future& other)
      : task_ptr_(other.task_ptr_),
        future_shm_(other.future_shm_),
        parent_task_(other.parent_task_),
        is_owner_(false) {}  // Copy does not transfer ownership

  /**
   * Copy assignment operator - does not transfer ownership
   * @param other Future to copy from
   * @return Reference to this future
   */
  Future& operator=(const Future& other) {
    if (this != &other) {
      // Destroy existing task if we own it
      if (is_owner_) {
        Destroy();
      }
      task_ptr_ = other.task_ptr_;
      future_shm_ = other.future_shm_;
      parent_task_ = other.parent_task_;
      is_owner_ = false;  // Copy does not transfer ownership
    }
    return *this;
  }

  /**
   * Move constructor - transfers ownership
   * @param other Future to move from
   */
  Future(Future&& other) noexcept
      : task_ptr_(std::move(other.task_ptr_)),
        future_shm_(std::move(other.future_shm_)),
        parent_task_(other.parent_task_),
        is_owner_(other.is_owner_) {  // Transfer ownership
    other.parent_task_ = nullptr;
    other.is_owner_ = false;  // Source no longer owns
  }

  /**
   * Move assignment operator - transfers ownership
   * @param other Future to move from
   * @return Reference to this future
   */
  Future& operator=(Future&& other) noexcept {
    if (this != &other) {
      // Destroy existing task if we own it
      if (is_owner_) {
        Destroy();
      }
      task_ptr_ = std::move(other.task_ptr_);
      future_shm_ = std::move(other.future_shm_);
      parent_task_ = other.parent_task_;
      is_owner_ = other.is_owner_;  // Transfer ownership
      other.parent_task_ = nullptr;
      other.is_owner_ = false;  // Source no longer owns
    }
    return *this;
  }

  /**
   * Get raw pointer to the task
   * @return Pointer to the task object
   */
  TaskT* get() const {
    return task_ptr_.ptr_;
  }

  /**
   * Get the FullPtr to the task (non-const version)
   * @return FullPtr to the task object
   */
  hipc::FullPtr<TaskT>& GetTaskPtr() {
    return task_ptr_;
  }

  /**
   * Get the FullPtr to the task (const version)
   * @return FullPtr to the task object
   */
  const hipc::FullPtr<TaskT>& GetTaskPtr() const {
    return task_ptr_;
  }

  /**
   * Dereference operator - access task members
   * @return Reference to the task object
   */
  TaskT& operator*() const {
    return *task_ptr_.ptr_;
  }

  /**
   * Arrow operator - access task members
   * @return Pointer to the task object
   */
  TaskT* operator->() const {
    return task_ptr_.ptr_;
  }

  /**
   * Check if the task is complete
   * @return True if task has completed, false otherwise
   */
  bool IsComplete() const {
    if (future_shm_.IsNull()) {
      return false;
    }
    return future_shm_->is_complete_.load() != 0;
  }

  /**
   * Wait for task completion (blocking)
   * Calls IpcManager::Recv() to handle task completion and deserialization
   */
  void Wait();

  /**
   * Mark the task as complete
   */
  void Complete() {
    if (!future_shm_.IsNull()) {
      future_shm_->is_complete_.store(1);
    }
  }

  /**
   * Mark the task as complete (alias for Complete)
   */
  void SetComplete() {
    Complete();
  }

  /**
   * Check if this future is null
   * @return True if future is null, false otherwise
   */
  bool IsNull() const {
    return task_ptr_.IsNull();
  }

  /**
   * Get the FutureShm FullPtr
   * @return FullPtr to the FutureShm object
   */
  hipc::FullPtr<FutureT>& GetFutureShm() {
    return future_shm_;
  }

  /**
   * Get the FutureShm FullPtr (const version)
   * @return FullPtr to the FutureShm object
   */
  const hipc::FullPtr<FutureT>& GetFutureShm() const {
    return future_shm_;
  }

  /**
   * Get the pool ID from the FutureShm
   * @return Pool ID for the task
   */
  PoolId GetPoolId() const {
    if (future_shm_.IsNull()) {
      return PoolId::GetNull();
    }
    return future_shm_->pool_id_;
  }

  /**
   * Set the pool ID in the FutureShm
   * @param pool_id Pool ID to set
   */
  void SetPoolId(const PoolId& pool_id) {
    if (!future_shm_.IsNull()) {
      future_shm_->pool_id_ = pool_id;
    }
  }

  /**
   * Cast this Future to a Future of a different task type
   *
   * This is a safe operation because Future<TaskT> and Future<NewTaskT>
   * have identical memory layouts - they both store the same underlying
   * pointers (task_ptr_, future_shm_, parent_task_).
   *
   * Note: Cast does not transfer ownership - the original Future retains it.
   *
   * @tparam NewTaskT The new task type to cast to
   * @return Future<NewTaskT> with the same underlying state (non-owning)
   */
  template<typename NewTaskT>
  Future<NewTaskT, AllocT> Cast() const {
    Future<NewTaskT, AllocT> result;
    // Use reinterpret_cast to copy the memory layout
    // This works because Future<TaskT> and Future<NewTaskT> have identical sizes
    result.task_ptr_ = task_ptr_.template Cast<NewTaskT>();
    result.future_shm_ = future_shm_;
    result.parent_task_ = parent_task_;
    result.is_owner_ = false;  // Cast does not transfer ownership
    return result;
  }

  /**
   * Get the parent task RunContext pointer
   * @return Pointer to parent RunContext or nullptr
   */
  RunContext* GetParentTask() const {
    return parent_task_;
  }

  /**
   * Set the parent task RunContext pointer
   * @param parent_task Pointer to parent RunContext
   */
  void SetParentTask(RunContext* parent_task) {
    parent_task_ = parent_task;
  }

  // =========================================================================
  // C++20 Coroutine Awaitable Interface
  // These methods allow `co_await future` in runtime coroutines
  // =========================================================================

  /**
   * Check if the awaitable is ready (coroutine await_ready)
   *
   * If the task is already complete, the coroutine won't suspend.
   * @return True if task is complete, false if coroutine should suspend
   */
  bool await_ready() const noexcept {
    return IsComplete();
  }

  /**
   * Suspend the coroutine and register for resumption (coroutine await_suspend)
   *
   * This is called when await_ready returns false. It stores the coroutine
   * handle in the RunContext so the worker can resume it when the task
   * completes. Also marks this Future as the owner of the task.
   *
   * @tparam PromiseT The promise type of the calling coroutine
   * @param handle The coroutine handle to resume when task completes
   * @return True to suspend, false to continue without suspending
   */
  template<typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> handle) noexcept {
    // Mark this Future as owner of the task (will be destroyed on Future destruction)
    is_owner_ = true;
    auto* run_ctx = handle.promise().get_run_context();
    if (!run_ctx) {
      // No RunContext available, don't suspend
      return false;
    }
    // Store parent context for resumption tracking
    SetParentTask(run_ctx);
    // Store coroutine handle in RunContext for worker to resume
    run_ctx->coro_handle_ = handle;
    run_ctx->is_yielded_ = true;
    run_ctx->yield_time_us_ = 0.0;
    return true;  // Suspend the coroutine
  }

  /**
   * Get the result after resumption (coroutine await_resume)
   *
   * Returns reference to this Future so caller can access the completed task.
   * Marks this Future as the owner if not already set (for await_ready=true case).
   * Calls PostWait() on the task for post-completion actions.
   * @return Reference to this Future
   */
  Future<TaskT, AllocT>& await_resume() noexcept {
    // If await_ready returned true, await_suspend wasn't called, so set ownership here
    is_owner_ = true;
    // Call PostWait() callback on the task for post-completion actions
    if (!task_ptr_.IsNull()) {
      task_ptr_->PostWait();
    }
    return *this;
  }
};

/**
 * YieldAwaiter - Awaitable for yielding control in coroutines
 *
 * This class implements the awaitable interface for cooperative yielding
 * within ChiMod runtime coroutines. It allows tasks to yield control
 * back to the worker with an optional delay before resumption.
 *
 * Usage:
 *   co_await chi::yield();       // Yield immediately
 *   co_await chi::yield(25.0);   // Yield with 25 microsecond delay
 */
class YieldAwaiter {
 private:
  /** Time in microseconds to delay before resumption */
  double yield_time_us_;

 public:
  /**
   * Construct a YieldAwaiter with optional delay
   * @param us Microseconds to delay before resumption (default: 0)
   */
  explicit YieldAwaiter(double us = 0.0) : yield_time_us_(us) {}

  /**
   * Yield is never immediately ready - always suspends
   * @return Always false (always suspend)
   */
  bool await_ready() const noexcept {
    return false;
  }

  /**
   * Suspend the coroutine and mark for yielded resumption
   *
   * @tparam PromiseT The promise type of the calling coroutine
   * @param handle The coroutine handle to resume after yield
   * @return True to suspend, false if no RunContext available
   */
  template<typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> handle) noexcept {
    auto* run_ctx = handle.promise().get_run_context();
    if (!run_ctx) {
      // No RunContext available, don't suspend
      return false;
    }
    // Store coroutine handle in RunContext for worker to resume
    run_ctx->coro_handle_ = handle;
    run_ctx->is_yielded_ = true;
    run_ctx->yield_time_us_ = yield_time_us_;
    return true;  // Suspend the coroutine
  }

  /**
   * Resume after yield - nothing to return
   */
  void await_resume() noexcept {}
};

/**
 * Create a YieldAwaiter for cooperative yielding in coroutines
 *
 * This function provides a clean syntax for yielding control within
 * ChiMod runtime coroutines.
 *
 * @param us Microseconds to delay before resumption (default: 0)
 * @return YieldAwaiter object that can be co_awaited
 *
 * Usage:
 *   co_await chi::yield();       // Yield immediately
 *   co_await chi::yield(25.0);   // Yield with 25 microsecond delay
 */
inline YieldAwaiter yield(double us = 0.0) {
  return YieldAwaiter(us);
}

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_
