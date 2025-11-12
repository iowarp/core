/**
 * Test that replicates the exact fiber context pattern used in worker.cc and task.cc
 * This test validates the task Wait/Resume cycle that is failing in the actual code
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <boost/context/detail/fcontext.hpp>

// Namespace alias for convenience
namespace bctx = boost::context::detail;

struct WorkerFiberTest {
    void* stack_ptr;
    void* stack_base_for_free;
    size_t stack_size;
    
    // Mimic RunContext fields from worker.cc
    bctx::fcontext_t fiber_context;    // Initial fiber entry point
    bctx::transfer_t fiber_transfer;   // Current yield/resume point
    bctx::transfer_t worker_context;   // Worker context for yielding back
    bool is_blocked;

    WorkerFiberTest() : stack_ptr(nullptr), stack_base_for_free(nullptr),
                        stack_size(65536), fiber_context{}, fiber_transfer{},
                        worker_context{}, is_blocked(false) {
        AllocateStack();
    }
    
    ~WorkerFiberTest() {
        DeallocateStack();
    }
    
    void AllocateStack() {
        const size_t page_size = 4096;
        stack_size = ((stack_size + page_size - 1) / page_size) * page_size;
        
        int ret = posix_memalign(&stack_base_for_free, page_size, stack_size);
        if (ret != 0) {
            std::cerr << "Failed to allocate aligned stack memory" << std::endl;
            std::abort();
        }
        
        std::memset(stack_base_for_free, 0, stack_size);
        
        // For x86-64, stack grows downward
        // Ensure 16-byte alignment (required for x86-64 ABI)
        char* stack_top = static_cast<char*>(stack_base_for_free) + stack_size;
        stack_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(stack_top) & ~static_cast<uintptr_t>(15));
        
        // Verify alignment and validity
        if (!stack_ptr || reinterpret_cast<uintptr_t>(stack_ptr) % 16 != 0) {
            std::cerr << "Stack pointer not properly aligned" << std::endl;
            std::abort();
        }
    }
    
    void DeallocateStack() {
        if (stack_base_for_free) {
            std::free(stack_base_for_free);
            stack_base_for_free = nullptr;
            stack_ptr = nullptr;
        }
    }
};

// Global test instance for fiber function access (like CHI_CUR_WORKER pattern)
static WorkerFiberTest* g_current_test = nullptr;
static volatile int g_task_step = 0;
static volatile bool g_task_complete = false;

// Simulate Task::YieldBase() behavior
void TaskYieldBase() {
    std::cout << "Task: YieldBase called, step " << g_task_step << std::endl;
    
    // Mark as blocked (from task.cc:89)
    g_current_test->is_blocked = true;
    
    // Jump back to worker using worker_context (from task.cc:94-95)
    g_current_test->fiber_transfer = bctx::jump_fcontext(
        g_current_test->worker_context.fctx,
        g_current_test->worker_context.data);
        
    std::cout << "Task: Resumed from yield, step " << g_task_step << std::endl;
}

// Simulate container->Run() calling Task::Wait() multiple times
void TaskExecutionFunction(bctx::transfer_t t) {
    std::cout << "=== Task Fiber Started ===" << std::endl;
    
    // Store worker context (from worker.cc:774)
    g_current_test->worker_context = t;
    
    // Simulate first subtask requiring Wait()
    g_task_step = 1;
    std::cout << "Task: Starting step 1, calling Wait()" << std::endl;
    TaskYieldBase();  // This simulates task->Wait() -> YieldBase()
    
    // Simulate second subtask requiring Wait()  
    g_task_step = 2;
    std::cout << "Task: Starting step 2, calling Wait()" << std::endl;
    TaskYieldBase();  // This simulates another task->Wait() -> YieldBase()
    
    // Task completion
    g_task_step = 3;
    g_task_complete = true;
    std::cout << "Task: Completed step 3" << std::endl;
    
    // Jump back to worker when done (from worker.cc:813)
    bctx::jump_fcontext(t.fctx, t.data);
}

bool TestWorkerTaskPattern() {
    std::cout << "\n=== Test: Worker-Task Fiber Pattern ===" << std::endl;
    
    WorkerFiberTest test;
    g_current_test = &test;
    g_task_step = 0;
    g_task_complete = false;
    
    // === NEW TASK EXECUTION (from worker.cc:689-697) ===
    std::cout << "Worker: Starting new task" << std::endl;
    
    // Create fiber context for task (from worker.cc:691-693)
    test.fiber_context = bctx::make_fcontext(test.stack_ptr, test.stack_size, TaskExecutionFunction);
    
    // Jump to fiber context to execute task (from worker.cc:696-697)
    test.fiber_transfer = bctx::jump_fcontext(test.fiber_context, nullptr);
    
    // Check that task yielded at step 1
    if (g_task_step != 1 || !test.is_blocked) {
        std::cerr << "ERROR: Expected step 1 and blocked, got step " 
                  << g_task_step << " blocked=" << test.is_blocked << std::endl;
        return false;
    }
    std::cout << "Worker: Task yielded at step 1" << std::endl;
    
    // === RESUME BLOCKED TASK (from worker.cc:679-680) ===
    std::cout << "Worker: Resuming blocked task" << std::endl;
    test.is_blocked = false;
    
    // Resume execution using fiber_transfer (from worker.cc:679-680)
    test.fiber_transfer = bctx::jump_fcontext(
        test.fiber_transfer.fctx, 
        test.fiber_transfer.data);
    
    // Check that task yielded at step 2
    if (g_task_step != 2 || !test.is_blocked) {
        std::cerr << "ERROR: Expected step 2 and blocked, got step " 
                  << g_task_step << " blocked=" << test.is_blocked << std::endl;
        return false;
    }
    std::cout << "Worker: Task yielded at step 2" << std::endl;
    
    // === SECOND RESUME ===
    std::cout << "Worker: Resuming blocked task second time" << std::endl;
    test.is_blocked = false;
    
    // Resume execution again
    test.fiber_transfer = bctx::jump_fcontext(
        test.fiber_transfer.fctx,
        test.fiber_transfer.data);
    
    // Check that task completed
    if (g_task_step != 3 || !g_task_complete || test.is_blocked) {
        std::cerr << "ERROR: Expected step 3 and complete, got step " 
                  << g_task_step << " complete=" << g_task_complete 
                  << " blocked=" << test.is_blocked << std::endl;
        return false;
    }
    
    std::cout << "Worker: Task completed successfully" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Worker-Task Fiber Pattern Test Suite ===" << std::endl;
    std::cout << "This test replicates the exact pattern used in worker.cc and task.cc" << std::endl;
    
    if (TestWorkerTaskPattern()) {
        std::cout << "\n✓ Worker-Task pattern works correctly!" << std::endl;
        return 0;
    } else {
        std::cerr << "\n✗ Worker-Task pattern failed!" << std::endl;
        return 1;
    }
}