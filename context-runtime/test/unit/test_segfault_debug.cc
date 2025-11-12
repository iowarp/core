/**
 * Test to debug the segfault in worker.cc ExecTask is_started branch
 * This test will help isolate the exact condition causing the fiber_transfer corruption
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <boost/context/detail/fcontext.hpp>

// Namespace alias for convenience
namespace bctx = boost::context::detail;

struct DebugRunContext {
    void* stack_ptr;
    void* stack_base_for_free;
    size_t stack_size;
    
    // Mimic updated RunContext fields from worker.cc
    bctx::fcontext_t fiber_context;   // Initial fiber entry point
    bctx::transfer_t yield_context;   // Context from FiberExecutionFunction parameter - used for yielding back
    bctx::transfer_t resume_context;  // Context for resuming into yield function
    bool is_blocked;

    DebugRunContext() : stack_ptr(nullptr), stack_base_for_free(nullptr),
                        stack_size(65536), fiber_context{}, yield_context{},
                        resume_context{}, is_blocked(false) {
        AllocateStack();
    }
    
    ~DebugRunContext() {
        DeallocateStack();
    }
    
    void AllocateStack() {
        // Use the SAME allocation pattern as the fixed worker.cc
        const size_t page_size = 4096;
        stack_size = ((stack_size + page_size - 1) / page_size) * page_size;
        
        int ret = posix_memalign(&stack_base_for_free, page_size, stack_size);
        if (ret != 0) {
            std::cerr << "Failed to allocate aligned stack memory" << std::endl;
            std::abort();
        }
        
        std::memset(stack_base_for_free, 0, stack_size);
        
        // Stack grows downward - mimic worker.cc logic
        char* stack_top = static_cast<char*>(stack_base_for_free) + stack_size;
        stack_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(stack_top) & ~static_cast<uintptr_t>(15));
        
        // Verify alignment
        if (!stack_ptr || reinterpret_cast<uintptr_t>(stack_ptr) % 16 != 0) {
            std::cerr << "Stack pointer not properly aligned" << std::endl;
            std::abort();
        }
        
        std::cout << "Stack allocated: base=" << stack_base_for_free 
                  << " ptr=" << stack_ptr << " size=" << stack_size << std::endl;
    }
    
    void DeallocateStack() {
        if (stack_base_for_free) {
            std::free(stack_base_for_free);
            stack_base_for_free = nullptr;
            stack_ptr = nullptr;
        }
    }
    
    void PrintContextInfo(const std::string& label) {
        std::cout << "[" << label << "] fiber_context=" << &fiber_context 
                  << " yield_context.fctx=" << yield_context.fctx
                  << " resume_context.fctx=" << resume_context.fctx << std::endl;
    }
};

// Global test instance
static DebugRunContext* g_debug_ctx = nullptr;
static volatile int g_yield_count = 0;
static volatile bool g_should_segfault = false;

void DebugYieldBase() {
    g_yield_count++;
    std::cout << "Task: YieldBase called, yield_count=" << g_yield_count << std::endl;
    
    g_debug_ctx->PrintContextInfo("Before yield");
    
    // Mark as blocked
    g_debug_ctx->is_blocked = true;
    
    // Jump back to worker using yield_context - store result in resume_context
    g_debug_ctx->resume_context = bctx::jump_fcontext(
        g_debug_ctx->yield_context.fctx,
        g_debug_ctx->yield_context.data);
        
    std::cout << "Task: Resumed from yield, yield_count=" << g_yield_count << std::endl;
    g_debug_ctx->PrintContextInfo("After resume");
}

void DebugTaskFunction(bctx::transfer_t t) {
    std::cout << "=== Debug Task Fiber Started ===" << std::endl;
    
    // Store worker context for yielding back
    g_debug_ctx->yield_context = t;
    g_debug_ctx->PrintContextInfo("Task started");
    
    // First yield
    std::cout << "Task: About to yield first time" << std::endl;
    DebugYieldBase();
    
    // Second yield  
    std::cout << "Task: About to yield second time" << std::endl;
    DebugYieldBase();
    
    // Third yield - this might be where corruption happens
    std::cout << "Task: About to yield third time" << std::endl;
    DebugYieldBase();
    
    std::cout << "Task: Completed all yields" << std::endl;
    
    // Jump back to worker when done
    bctx::jump_fcontext(t.fctx, t.data);
}

bool TestSegfaultScenario() {
    std::cout << "\n=== Debug Segfault Scenario ===" << std::endl;
    
    DebugRunContext debug_ctx;
    g_debug_ctx = &debug_ctx;
    g_yield_count = 0;
    
    // === NEW TASK EXECUTION ===
    std::cout << "Worker: Starting new task" << std::endl;
    debug_ctx.PrintContextInfo("Initial state");
    
    // Create fiber context
    debug_ctx.fiber_context = bctx::make_fcontext(
        debug_ctx.stack_ptr, debug_ctx.stack_size, DebugTaskFunction);
    debug_ctx.PrintContextInfo("After make_fcontext");
    
    // Initial jump to start task
    debug_ctx.resume_context = bctx::jump_fcontext(debug_ctx.fiber_context, nullptr);
    
    if (!debug_ctx.is_blocked || g_yield_count != 1) {
        std::cerr << "ERROR: Expected first yield, got yield_count=" 
                  << g_yield_count << std::endl;
        return false;
    }
    std::cout << "Worker: Task yielded first time" << std::endl;
    debug_ctx.PrintContextInfo("After first yield");
    
    // === FIRST RESUME (this should work) ===
    std::cout << "Worker: Resuming task first time" << std::endl;
    debug_ctx.is_blocked = false;
    
    // This is the line that segfaults in worker.cc
    std::cout << "Worker: About to call jump_fcontext with fctx=" 
              << debug_ctx.resume_context.fctx << std::endl;
    debug_ctx.resume_context = bctx::jump_fcontext(
        debug_ctx.resume_context.fctx, 
        debug_ctx.resume_context.data);
    
    if (!debug_ctx.is_blocked || g_yield_count != 2) {
        std::cerr << "ERROR: Expected second yield, got yield_count=" 
                  << g_yield_count << std::endl;
        return false;
    }
    std::cout << "Worker: Task yielded second time" << std::endl;
    debug_ctx.PrintContextInfo("After second yield");
    
    // === SECOND RESUME (this might segfault) ===
    std::cout << "Worker: Resuming task second time" << std::endl;
    debug_ctx.is_blocked = false;
    
    std::cout << "Worker: About to call jump_fcontext with fctx=" 
              << debug_ctx.resume_context.fctx << std::endl;
    debug_ctx.resume_context = bctx::jump_fcontext(
        debug_ctx.resume_context.fctx,
        debug_ctx.resume_context.data);
    
    if (!debug_ctx.is_blocked || g_yield_count != 3) {
        std::cerr << "ERROR: Expected third yield, got yield_count=" 
                  << g_yield_count << std::endl;
        return false;
    }
    std::cout << "Worker: Task yielded third time" << std::endl;
    
    // === THIRD RESUME (final) ===
    std::cout << "Worker: Resuming task final time" << std::endl;
    debug_ctx.is_blocked = false;
    
    std::cout << "Worker: About to call jump_fcontext with fctx=" 
              << debug_ctx.resume_context.fctx << std::endl;
    debug_ctx.resume_context = bctx::jump_fcontext(
        debug_ctx.resume_context.fctx,
        debug_ctx.resume_context.data);
    
    std::cout << "Worker: Task completed successfully" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Segfault Debug Test Suite ===" << std::endl;
    std::cout << "This test isolates the segfault condition in worker.cc" << std::endl;
    
    try {
        if (TestSegfaultScenario()) {
            std::cout << "\n✓ No segfault occurred - test passed!" << std::endl;
            return 0;
        } else {
            std::cerr << "\n✗ Test failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n✗ Unknown exception occurred!" << std::endl;
        return 1;
    }
}