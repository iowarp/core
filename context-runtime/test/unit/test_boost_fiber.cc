/**
 * Test for boost fiber context system
 * This test validates our understanding of boost::context::detail API
 * and demonstrates the correct usage patterns for fiber management.
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <boost/context/detail/fcontext.hpp>

// Namespace alias for convenience
namespace bctx = boost::context::detail;

struct FiberTest {
    void* stack_ptr;
    void* stack_base_for_free;
    size_t stack_size;
    bctx::fcontext_t fiber_context;
    
    FiberTest() : stack_ptr(nullptr), stack_base_for_free(nullptr), 
                  stack_size(65536), fiber_context{} {
        AllocateStack();
    }
    
    ~FiberTest() {
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

// Shared state variables
static volatile int g_fiber_step = 0;
static volatile bool g_test_complete = false;

// Main fiber function
void FiberMainFunction(bctx::transfer_t t) {
    std::cout << "=== Fiber Started ===" << std::endl;
    
    // First yield point
    g_fiber_step = 1;
    std::cout << "Step 1: Fiber initialized" << std::endl;
    bctx::jump_fcontext(t.fctx, t.data);
    
    // Second yield point
    g_fiber_step = 2;
    std::cout << "Step 2: Fiber resumed first time" << std::endl;
    bctx::jump_fcontext(t.fctx, t.data);
    
    // Final yield point
    g_fiber_step = 3;
    std::cout << "Step 3: Fiber completed" << std::endl;
    g_test_complete = true;
    bctx::jump_fcontext(t.fctx, t.data);
}

bool TestBasicFiberCreation() {
    std::cout << "\n=== Test: Basic Fiber Creation ===" << std::endl;
    
    FiberTest test;
    g_fiber_step = 0;
    g_test_complete = false;
    
    // Create fiber context
    test.fiber_context = bctx::make_fcontext(test.stack_ptr, test.stack_size, FiberMainFunction);
    
    // First jump into fiber
    bctx::transfer_t transfer = bctx::jump_fcontext(test.fiber_context, nullptr);
    (void)transfer;  // Suppress unused variable warning
    
    // Verify first fiber step
    if (g_fiber_step != 1) {
        std::cerr << "ERROR: First fiber jump failed. Step: " 
                  << g_fiber_step << std::endl;
        return false;
    }
    
    std::cout << "Pass: Basic fiber creation works" << std::endl;
    return true;
}

bool TestFiberYieldAndResume() {
    std::cout << "\n=== Test: Fiber Yield and Resume ===" << std::endl;
    
    FiberTest test;
    g_fiber_step = 0;
    g_test_complete = false;
    
    // Create fiber context
    test.fiber_context = bctx::make_fcontext(test.stack_ptr, test.stack_size, FiberMainFunction);
    
    // First jump into fiber
    bctx::transfer_t transfer = bctx::jump_fcontext(test.fiber_context, nullptr);
    
    // Resume fiber first time
    transfer = bctx::jump_fcontext(transfer.fctx, nullptr);
    
    // Verify second fiber step
    if (g_fiber_step != 2) {
        std::cerr << "ERROR: First resume failed. Step: " 
                  << g_fiber_step << std::endl;
        return false;
    }
    
    std::cout << "Pass: Fiber yield and first resume works" << std::endl;
    return true;
}

bool TestMultipleYieldCycles() {
    std::cout << "\n=== Test: Multiple Yield Cycles ===" << std::endl;
    
    FiberTest test;
    g_fiber_step = 0;
    g_test_complete = false;
    
    // Create fiber context
    test.fiber_context = bctx::make_fcontext(test.stack_ptr, test.stack_size, FiberMainFunction);
    
    // First jump into fiber
    bctx::transfer_t transfer = bctx::jump_fcontext(test.fiber_context, nullptr);
    
    // Resume fiber first time
    transfer = bctx::jump_fcontext(transfer.fctx, nullptr);
    
    // Resume fiber second time
    transfer = bctx::jump_fcontext(transfer.fctx, nullptr);
    
    // Verify final fiber step
    if (!g_test_complete || g_fiber_step != 3) {
        std::cerr << "ERROR: Final resume failed. Step: " 
                  << g_fiber_step << " Complete: " 
                  << g_test_complete << std::endl;
        return false;
    }
    
    std::cout << "Pass: Multiple yield cycles work correctly" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Boost Fiber Context Test Suite ===" << std::endl;
    std::cout << "This test validates our understanding of boost::context::detail API" << std::endl;
    
    int failed_tests = 0;
    
    // Run tests in sequence
    if (!TestBasicFiberCreation()) {
        failed_tests++;
    }
    
    if (!TestFiberYieldAndResume()) {
        failed_tests++;
    }
    
    if (!TestMultipleYieldCycles()) {
        failed_tests++;  
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    if (failed_tests == 0) {
        std::cout << "Pass: All tests passed Boost context understanding verified." << std::endl;
        return 0;
    } else {
        std::cerr << "Fail: " << failed_tests << " test(s) failed" << std::endl;
        return 1;
    }
}
