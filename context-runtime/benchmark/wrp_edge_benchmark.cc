#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <cmath>
#include <fstream>
#include <chrono>

// clio-core and hardware headers
#include "chimaera/chimaera.h"
#include "pi_timer.h" 

using namespace chi;

// --- Phase 1: Zero-Overhead Telemetry Structures ---
// Pre-allocated to avoid malloc() / cache-misses during the benchmark
struct TraceEvent {
    uint64_t start_tick;
    uint64_t end_tick;
};

// --- Phase 2: Mathematical DES Distributions ---
// Simulates heavy-tailed Zstandard compression spikes
double pareto_random(std::mt19937& gen, double scale, double shape) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(gen);
    if (u == 0.0) u = 0.0001; // Prevent division by zero
    return scale / std::pow(u, 1.0 / shape);
}

// --- Phase 3: Workload Generators ---
std::atomic<bool> keep_running{true};

void NoisyNeighborThread() {
    // DES Parameter: Pareto Distribution for heavy compute spikes
    std::mt19937 gen(42); 
    
    while (keep_running.load(std::memory_order_relaxed)) {
        // Generate a task that is heavily skewed (80% small, 20% massive spikes)
        double compute_cost = pareto_random(gen, 1000.0, 1.5); // min 1ms, heavy tail
        
        // Construct the dummy Task
        // (Note: Adjust to your specific module's task constructor if needed)
        TaskStat stat;
        stat.io_size_ = 1048576; // 1MB blocks
        stat.compute_ = static_cast<size_t>(compute_cost);

        // Submit to the runtime to stress the DWRR scheduler
        // CHI_CLIENT->SubmitTask(...); // INJECT YOUR MODULE SUBMISSION HERE
        
        // Small sleep to prevent total lockup during while loop
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void MicroIoBarrageThread(std::vector<TraceEvent>& trace_buffer, size_t num_events) {
    // DES Parameter: Exponential distribution models Poisson inter-arrival times
    std::mt19937 gen(1337);
    // Average arrival rate: 1 packet every 200 microseconds
    std::exponential_distribution<double> poisson_arrival(1.0 / 200.0); 

    for (size_t i = 0; i < num_events; ++i) {
        // 1. Simulate erratic sensor burst timing
        double delay_us = poisson_arrival(gen);
        
        // Fast yield if delay is tiny, otherwise spin briefly to mimic sensor wait
        auto start_wait = get_cntvct_el0();
        // Assuming 54MHz Pi 4 clock (~54 ticks per microsecond)
        uint64_t wait_ticks = static_cast<uint64_t>(delay_us * 54.0); 
        while ((get_cntvct_el0() - start_wait) < wait_ticks) {
            // Busy-wait to maintain microsecond accuracy on the arrival delay
        }

        // 2. Prepare Task
        TaskStat stat;
        stat.io_size_ = 64; // Tiny 64-byte sensor packet
        stat.compute_ = 2;  // 2us compute

        // 3. The Critical Measurement (Zero Observer Effect)
        trace_buffer[i].start_tick = get_cntvct_el0();
        
        // CHI_CLIENT->SubmitTaskAndWait(...); // INJECT YOUR MODULE SUBMISSION HERE
        
        trace_buffer[i].end_tick = get_cntvct_el0();
    }
    
    // Stop the Noisy Neighbor once the I/O barrage finishes
    keep_running.store(false, std::memory_order_relaxed);
}

int main(int argc, char** argv) {
    std::cout << "[IOWarp] Starting RCFS Edge Benchmark..." << std::endl;

    // 1. Validate the hardware timer
    uint64_t calib = chi::calibrate_timer_overhead();
    std::cout << "[IOWarp] CNTVCT_EL0 Read Overhead: " << calib << " ticks" << std::endl;

    // 2. Initialize Chimaera Runtime (Reads your wrp_runtime_config.yaml)
    // CHI_RUNTIME->Initialize(config_path);

    // 3. Pre-allocate 100,000 trace events (avoids mid-test malloc)
    size_t test_size = 100000;
    std::vector<TraceEvent> trace_buffer(test_size);

    // 4. Launch the Experimental Matrix
    std::thread noisy_neighbor(NoisyNeighborThread);
    std::thread micro_io(MicroIoBarrageThread, std::ref(trace_buffer), test_size);

    micro_io.join();
    noisy_neighbor.join();

    // 5. Post-Process Hardware Ticks to CSV (Offline, no observer effect)
    std::cout << "[IOWarp] Writing results to trace_results.csv..." << std::endl;
    std::ofstream outfile("trace_results.csv");
    outfile << "task_id,latency_ticks,latency_us\n";
    
    for (size_t i = 0; i < test_size; ++i) {
        uint64_t elapsed_ticks = trace_buffer[i].end_tick - trace_buffer[i].start_tick;
        // Convert to microseconds (Assuming Pi 4's 54MHz generic timer, adjust if your Pi uses 1MHz)
        double elapsed_us = static_cast<double>(elapsed_ticks) / 54.0; 
        outfile << i << "," << elapsed_ticks << "," << elapsed_us << "\n";
    }
    outfile.close();

    std::cout << "[IOWarp] Benchmark Complete." << std::endl;
    return 0;
}