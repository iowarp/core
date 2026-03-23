/**
 * Minimal external build test for the wrp_cte_uvm library.
 *
 * This file verifies that an external project can:
 *   1. Include the public header <wrp_cte/uvm/gpu_vmm.h>
 *   2. Instantiate GpuVmmConfig and GpuVirtualMemoryManager
 *   3. Link against libwrp_cte_uvm.so + CUDA driver/runtime
 *
 * It does NOT require a GPU to build — only to run.
 */

#include <wrp_cte/uvm/gpu_vmm.h>

#include <cstdio>

int main() {
    // Verify we can instantiate the config with defaults
    wrp_cte::uvm::GpuVmmConfig config;
    config.va_size_bytes = 64ULL * 1024 * 1024;
    config.page_size = 2ULL * 1024 * 1024;
    config.prefetch_window = 2;

    // Verify we can instantiate the manager (do NOT call init — no GPU required)
    wrp_cte::uvm::GpuVirtualMemoryManager vmm;

    printf("External UVM link test: OK (compiled and linked successfully)\n");
    printf("  va_size_bytes   = %zu\n", config.va_size_bytes);
    printf("  page_size       = %zu\n", config.page_size);
    printf("  prefetch_window = %zu\n", config.prefetch_window);
    return 0;
}
