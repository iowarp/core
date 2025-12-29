# C++20 Stackless Coroutines Migration Plan

This document outlines the comprehensive plan to migrate IOWarp Core from Boost Fibers to C++20 stackless coroutines.

## Executive Summary

Replace Boost Fibers (stackful coroutines using `boost::context::detail::fcontext_t` with 64KB per-task stacks) with C++20 stackless coroutines. This enables `co_await` semantics in runtime code while preserving blocking `Wait()` for client code. No backwards compatibility with Boost Fibers.

**Key Design Decisions:**
- **Coroutine return type**: `TaskResume` - new lightweight class for runtime method return type
- **Future**: Augmented to be directly awaitable (has `await_ready/suspend/resume` built-in)
- **Await pattern**: `co_await future` and `co_await chi::yield()` both work in coroutines
- **Blocking pattern**: `future.Wait()` for non-coroutine client code only
- **No new methods**: Augment existing `Run` method, don't create `RunCoro`
- **Code generation**: Update `chi_refresh_repo` to generate `TaskResume` return types

---

## 1. Current Architecture Summary

### 1.1 Boost Fiber Implementation

**Key Files:**
- `context-runtime/src/worker.cc` - Worker thread execution loop
- `context-runtime/src/task.cc` - Task Wait/Yield implementation
- `context-runtime/include/chimaera/worker.h` - Worker class definition
- `context-runtime/include/chimaera/task.h` - Task and RunContext definitions

**Current Mechanism:**
```cpp
// Fiber creation (worker.cc:952-956)
bctx::fcontext_t fiber_fctx = bctx::make_fcontext(
    run_ctx->stack_ptr, run_ctx->stack_size, fiber_fn);
bctx::transfer_t fiber_result = bctx::jump_fcontext(fiber_fctx, nullptr);

// Yield back to worker (task.cc:92-96)
bctx::fcontext_t yield_fctx = run_ctx->yield_context.fctx;
bctx::transfer_t yield_result = bctx::jump_fcontext(yield_fctx, yield_data);
```

**Stack Management:**
- 64KB stacks via `posix_memalign()` in `AllocateStackAndContext()`
- Stack cache (`std::queue<StackAndContext>`) for reuse
- Stack pointer adjustment based on growth direction

### 1.2 RunContext Fields to Remove

```cpp
// REMOVE these fiber-specific fields from RunContext (task.h:348-458)
void *stack_ptr;
void *stack_base_for_free;
size_t stack_size;
boost::context::detail::transfer_t yield_context;
boost::context::detail::transfer_t resume_context;
```

### 1.3 Current Task Wait/Yield (task.cc)

```cpp
void Task::Wait(std::atomic<u32>& is_complete, double yield_time_us) {
  do {
    worker->AddToBlockedQueue(run_ctx, true);
    YieldBase();  // jump_fcontext back to worker
  } while (is_complete.load() == 0);
}

void Task::Yield(double yield_time_us) {
  worker->AddToBlockedQueue(run_ctx, false);
  YieldBase();
}
```

---

## 2. New C++20 Coroutine Infrastructure

### 2.1 TaskResume - Coroutine Return Type for Runtime Methods

**Add to `context-runtime/include/chimaera/future.h`**

```cpp
/**
 * TaskResume - Return type for coroutine runtime methods
 * This is a lightweight class that holds the coroutine handle
 */
class TaskResume {
public:
  struct promise_type {
    RunContext* run_ctx_ = nullptr;

    TaskResume get_return_object() {
      return TaskResume(std::coroutine_handle<promise_type>::from_promise(*this));
    }

    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() { std::terminate(); }

    void set_run_context(RunContext* ctx) { run_ctx_ = ctx; }
    RunContext* get_run_context() const { return run_ctx_; }
  };

  using handle_type = std::coroutine_handle<promise_type>;

private:
  handle_type handle_;

public:
  explicit TaskResume(handle_type h) : handle_(h) {}
  TaskResume() : handle_(nullptr) {}

  handle_type get_handle() const { return handle_; }
  bool done() const { return handle_ && handle_.done(); }
  void resume() { if (handle_ && !handle_.done()) handle_.resume(); }
  void destroy() { if (handle_) { handle_.destroy(); handle_ = nullptr; } }
};
```

### 2.2 Future - Directly Awaitable (No Separate Awaiter Class)

Future remains unchanged for client code (`Wait()` still works). Add await methods directly to Future so `co_await future` works in coroutines.

**Add these methods to Future class in `context-runtime/include/chimaera/future.h`**

```cpp
template<typename TaskT, typename AllocT = CHI_MAIN_ALLOC_T>
class Future {
  // ... existing members ...

  /**
   * Check if task is already complete (skip suspension if true)
   */
  bool await_ready() const noexcept {
    return IsComplete();
  }

  /**
   * Called when coroutine suspends - set up wakeup mechanism
   */
  template<typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> handle) noexcept {
    auto* run_ctx = handle.promise().get_run_context();
    if (!run_ctx) return false;

    SetParentTask(run_ctx);
    run_ctx->coro_handle_ = handle;
    run_ctx->is_yielded_ = true;
    run_ctx->yield_time_us_ = 0.0;
    return true;  // Suspend coroutine
  }

  /**
   * Called when coroutine resumes - return this future
   */
  Future<TaskT, AllocT>& await_resume() {
    return *this;
  }
};
```

Now `co_await future` just works - no casting, no separate class.

### 2.3 Yield Awaiter (Replaces Task::Yield)

**Add to `context-runtime/include/chimaera/future.h`**

```cpp
/**
 * YieldAwaiter - Cooperative time-based yielding
 * Usage: co_await chi::yield(25);  // yield for 25 microseconds
 */
class YieldAwaiter {
private:
  double yield_time_us_;

public:
  explicit YieldAwaiter(double us = 0.0) : yield_time_us_(us) {}

  bool await_ready() const noexcept { return false; }

  template<typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> handle) noexcept {
    auto* run_ctx = handle.promise().get_run_context();
    if (!run_ctx) return false;

    run_ctx->coro_handle_ = handle;
    run_ctx->is_yielded_ = true;
    run_ctx->yield_time_us_ = yield_time_us_;
    return true;
  }

  void await_resume() {}
};

inline YieldAwaiter yield(double us = 0.0) { return YieldAwaiter(us); }
```

### 2.4 Updated RunContext

**Modify: `context-runtime/include/chimaera/task.h`**

```cpp
struct RunContext {
  // NEW: Generic coroutine handle for resumption
  std::coroutine_handle<> coro_handle_;

  // RETAINED: Execution state (unchanged)
  ThreadType thread_type;
  u32 worker_id;
  FullPtr<Task> task;
  bool is_yielded_;
  double yield_time_us_;
  hshm::Timepoint block_start;
  Container *container;
  TaskLane *lane;
  ExecMode exec_mode;
  void *event_queue_;
  std::vector<PoolQuery> pool_queries;
  std::vector<FullPtr<Task>> subtasks_;
  u32 completed_replicas_;
  u32 yield_count_;
  Future<Task> future_;
  bool destroy_in_end_task_;

  void Clear() {
    coro_handle_ = nullptr;
    pool_queries.clear();
    subtasks_.clear();
    completed_replicas_ = 0;
    yield_time_us_ = 0.0;
    yield_count_ = 0;
  }
};
```

---

## 3. Worker Changes

### 3.1 Methods to Remove

```cpp
// REMOVE from worker.h and worker.cc
void BeginFiber(const FullPtr<Task>&, RunContext*, void (*)(bctx::transfer_t));
void ResumeFiber(const FullPtr<Task>&, RunContext*);
static void FiberExecutionFunction(bctx::transfer_t);
RunContext* AllocateStackAndContext(size_t size);
void DeallocateStackAndContext(RunContext*);

// REMOVE member
std::queue<StackAndContext> stack_cache_;
```

### 3.2 New Methods to Add

```cpp
class Worker {
public:
  void StartCoroutine(const FullPtr<Task>& task_ptr, RunContext* run_ctx,
                      TaskResume task_resume);
  void ResumeCoroutine(RunContext* run_ctx);
  RunContext* AllocateRunContext();
  void DeallocateRunContext(RunContext* run_ctx);

private:
  std::queue<RunContext*> run_ctx_cache_;  // Replaces stack_cache_
};
```

### 3.3 Updated ExecTask Implementation

```cpp
void Worker::ExecTask(const FullPtr<Task>& task_ptr, RunContext* run_ctx,
                      bool is_started) {
  SetTaskDidWork(true);
  if (task_ptr.IsNull() || !run_ctx) return;

  SetCurrentRunContext(run_ctx);

  if (is_started) {
    ResumeCoroutine(run_ctx);
  } else {
    Container* container = run_ctx->container;
    if (container) {
      // Run returns TaskResume which is a coroutine
      TaskResume task_resume = container->Run(task_ptr->method_, task_ptr, *run_ctx);
      StartCoroutine(task_ptr, run_ctx, std::move(task_resume));
    }
    task_ptr->SetFlags(TASK_STARTED);
  }

  if (run_ctx->is_yielded_) {
    AddToBlockedQueue(run_ctx, run_ctx->yield_time_us_ == 0.0);
    return;
  }

  if (run_ctx->exec_mode == ExecMode::kDynamicSchedule) {
    RerouteDynamicTask(task_ptr, run_ctx);
    return;
  }

  EndTask(task_ptr, run_ctx, true);
}

void Worker::StartCoroutine(const FullPtr<Task>& task_ptr, RunContext* run_ctx,
                            TaskResume task_resume) {
  run_ctx->coro_handle_ = task_resume.get_handle();
  auto& promise = task_resume.get_handle().promise();
  promise.set_run_context(run_ctx);

  if (run_ctx->container && !task_ptr->IsPeriodic()) {
    run_ctx->container->UpdateWork(task_ptr, *run_ctx, 1);
  }

  run_ctx->coro_handle_.resume();  // Resume from initial_suspend

  if (run_ctx->coro_handle_.done()) {
    run_ctx->is_yielded_ = false;
  }
}

void Worker::ResumeCoroutine(RunContext* run_ctx) {
  if (!run_ctx || !run_ctx->coro_handle_) return;

  run_ctx->is_yielded_ = false;
  run_ctx->coro_handle_.resume();

  if (run_ctx->coro_handle_.done()) {
    run_ctx->is_yielded_ = false;
  }
}
```

---

## 4. Container and Runtime Method Changes

### 4.1 Augment Container::Run (No New Method)

**Modify: `context-runtime/include/chimaera/container.h`**

```cpp
class Container {
public:
  // CHANGED: Run now returns TaskResume instead of void
  virtual TaskResume Run(u32 method, hipc::FullPtr<Task> task_ptr,
                         RunContext& rctx) = 0;
};
```

### 4.2 Runtime Method Conversion Pattern

**Before (current):**
```cpp
void Runtime::Flush(hipc::FullPtr<FlushTask> task, chi::RunContext& rctx) {
  while (work_orchestrator->HasWorkRemaining(total_work_remaining)) {
    task->Yield(25);  // OLD: Task::Yield
  }
  task->return_code_ = 0;
}
```

**After (with coroutines):**
```cpp
TaskResume Runtime::Flush(hipc::FullPtr<FlushTask> task, chi::RunContext& rctx) {
  while (work_orchestrator->HasWorkRemaining(total_work_remaining)) {
    co_await chi::yield(25);  // NEW: co_await yield
  }
  task->return_code_ = 0;
  co_return;
}
```

### 4.3 Subtask Pattern with co_await

**Before:**
```cpp
void Runtime::ReorganizeBlob(hipc::FullPtr<ReorganizeBlobTask> task, chi::RunContext& rctx) {
  auto get_task = client_.AsyncGetBlob(...);
  get_task.Wait();  // OLD: blocking wait

  auto put_task = client_.AsyncPutBlob(...);
  put_task.Wait();  // OLD: blocking wait
}
```

**After:**
```cpp
TaskResume Runtime::ReorganizeBlob(hipc::FullPtr<ReorganizeBlobTask> task, chi::RunContext& rctx) {
  auto get_task = client_.AsyncGetBlob(...);
  co_await get_task;  // NEW: co_await future directly

  auto put_task = client_.AsyncPutBlob(...);
  co_await put_task;  // NEW: co_await future directly

  co_return;
}
```

### 4.4 Accessing Results After co_await

```cpp
TaskResume Runtime::SomeMethod(hipc::FullPtr<SomeTask> task, chi::RunContext& rctx) {
  auto subtask = client_.AsyncGetData(...);

  // co_await returns the Future reference for casting
  auto& completed = co_await subtask;

  // Cast to specific task type to access results
  auto typed = completed.Cast<GetDataTask>();
  task->output_ = typed->data_;

  co_return;
}
```

---

## 5. Future Dual-Mode Support

### 5.1 Client Mode (Blocking Wait) - For Non-Coroutines

```cpp
// In client code (non-coroutine)
auto task = client.AsyncCreate(...);
task.Wait();  // Blocking wait - spins until complete
auto result = task->GetReturnCode();
```

### 5.2 Runtime Mode (co_await) - For Coroutines

```cpp
// In runtime code (coroutine)
TaskResume Runtime::SomeMethod(...) {
  auto subtask = client_.AsyncDoWork(...);
  co_await subtask;  // Suspends coroutine until subtask completes

  // Results available after co_await
  auto typed = subtask.Cast<DoWorkTask>();

  co_return;
}
```

---

## 6. Code Generation Changes

### 6.1 Update chi_refresh_repo.cc

The autogenerated `Run` method in `*_lib_exec.cc` changes from `void` to `Future<Task>`:

**Before:**
```cpp
void Runtime::Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                  chi::RunContext& rctx) {
  switch (method) {
    case Method::kCreate: {
      auto typed_task = task_ptr.template Cast<CreateTask>();
      Create(typed_task, rctx);
      break;
    }
    // ...
  }
}
```

**After:**
```cpp
TaskResume Runtime::Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                        chi::RunContext& rctx) {
  switch (method) {
    case Method::kCreate: {
      auto typed_task = task_ptr.template Cast<CreateTask>();
      co_await Create(typed_task, rctx);
      break;
    }
    case Method::kFlush: {
      auto typed_task = task_ptr.template Cast<FlushTask>();
      co_await Flush(typed_task, rctx);
      break;
    }
    // ... all other methods
    default: break;
  }
  co_return;
}
```

### 6.2 Update Method Declarations

Individual method declarations change from `void` to `TaskResume`:

**Before:**
```cpp
void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx);
void Flush(hipc::FullPtr<FlushTask> task, chi::RunContext& rctx);
```

**After:**
```cpp
TaskResume Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx);
TaskResume Flush(hipc::FullPtr<FlushTask> task, chi::RunContext& rctx);
```

---

## 7. Files to Modify

### 7.1 No New Files Needed

The coroutine infrastructure is added to existing files.

### 7.2 Files to Modify

| File | Changes |
|------|---------|
| `context-runtime/include/chimaera/future.h` | Add await_ready/suspend/resume methods to Future, add YieldAwaiter, add TaskResume class |
| `context-runtime/include/chimaera/task.h` | Remove fiber fields from RunContext, add coro_handle_ |
| `context-runtime/include/chimaera/worker.h` | Remove fiber methods, add coroutine methods |
| `context-runtime/src/worker.cc` | Replace BeginFiber/ResumeFiber with StartCoroutine/ResumeCoroutine |
| `context-runtime/src/task.cc` | Remove Wait/Yield/YieldBase methods |
| `context-runtime/include/chimaera/container.h` | Change Run return type to Future<Task> |
| `context-runtime/util/chi_refresh_repo.cc` | Generate coroutine Run dispatch, update method signatures |
| `CMakeLists.txt` (various) | Add C++20 flags |

### 7.3 ChiMod Runtime Files to Convert

| File | Key Methods |
|------|-------------|
| `context-runtime/modules/admin/src/admin_runtime.cc` | Flush, Send, Recv, GetOrCreatePool |
| `context-runtime/modules/bdev/src/bdev_runtime.cc` | Read, Write, PerformAsyncIO |
| `context-transfer-engine/core/src/core_runtime.cc` | PutBlob, GetBlob, DelTag, ReorganizeBlob |
| `context-assimilation-engine/core/src/core_runtime.cc` | All methods with Wait/Yield |

---

## 8. CMake Changes

```cmake
# Add to root CMakeLists.txt or relevant component CMakeLists.txt
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fcoroutines)
endif()

# For Clang (if needed)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-fcoroutines-ts)  # or just works in C++20 mode
endif()
```

---

## 9. Migration Order

### Phase 1: Infrastructure
1. Update CMake for C++20 coroutines
2. Add TaskResume class to future.h
3. Add await_ready/suspend/resume methods directly to Future class
4. Add YieldAwaiter to future.h
5. Add `coro_handle_` to RunContext (keep fiber fields temporarily)

### Phase 2: Worker
6. Add new Worker methods (StartCoroutine, ResumeCoroutine, etc.)
7. Update ExecTask to call coroutine Run
8. Update blocked queue processing to call ResumeCoroutine

### Phase 3: Code Generation
9. Update chi_refresh_repo.cc to generate TaskResume return types
10. Regenerate all *_lib_exec.cc files

### Phase 4: ChiMod Migration (one at a time)
11. Convert admin ChiMod (Flush, Send, Recv first)
12. Convert bdev ChiMod
13. Convert CTE core ChiMod
14. Convert CAE core ChiMod

### Phase 5: Cleanup
15. Remove all Boost Fiber code from worker.cc
16. Remove Task::Wait, Task::Yield, Task::YieldBase from task.cc
17. Remove fiber-related RunContext fields
18. Update documentation (CLAUDE.md, MODULE_DEVELOPMENT_GUIDE.md)

---

## 10. Testing Strategy

### Unit Tests
```cpp
TEST_CASE("Coroutine basic execution") {
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto task = admin_client.AsyncFlush(chi::PoolQuery::Local());
  task.Wait();  // Client-side blocking wait
  REQUIRE(task->GetReturnCode() == 0);
}

TEST_CASE("co_await Future suspends and resumes") {
  // Test that parent task properly suspends when co_await is called
  // and resumes when subtask completes
}

TEST_CASE("co_await yield adds to periodic queue") {
  // Test that yield with timeout routes to periodic queue
}
```

### Integration Tests
1. Admin ChiMod: Flush, Send/Recv patterns
2. BDev ChiMod: File I/O with coroutines
3. Distributed tests: Remote task execution
4. Performance benchmarks: Memory and throughput comparison

---

## 11. Key Differences from Boost Fibers

| Aspect | Boost Fibers | C++20 Coroutines |
|--------|--------------|------------------|
| Stack | 64KB per task | Compiler-generated frame |
| Memory | Manual allocation | Heap-allocated frame |
| Context switch | jump_fcontext | co_await/resume |
| State storage | Stack variables | Coroutine frame |
| Yielding | YieldBase + blocked queue | co_await + blocked queue |
| Compilation | Runtime stack manipulation | Compiler transforms code |
| Return type | void | TaskResume |

---

## 12. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compiler support | Require GCC 10+ or Clang 14+, test both |
| Coroutine frame size | Profile hot paths, may need optimization |
| Debugging difficulty | Add logging, use debugger with coroutine support |
| Performance regression | Benchmark before/after, optimize if needed |
| Incremental migration | Keep dual-path temporarily during migration |
