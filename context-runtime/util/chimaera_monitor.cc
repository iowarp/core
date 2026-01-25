/**
 * Chimaera worker monitoring utility
 *
 * This utility connects to a running Chimaera runtime and displays
 * real-time statistics about worker threads, including:
 * - Number of queued, blocked, and periodic tasks
 * - Worker idle status and suspend periods
 * - Overall system load and utilization
 */

#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include <iomanip>

#include "chimaera/chimaera.h"
#include "chimaera/singletons.h"
#include "chimaera/types.h"
#include "chimaera/admin/admin_client.h"

namespace {

volatile bool g_keep_running = true;

/**
 * Print usage information
 */
void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]\n"
            << "\n"
            << "Options:\n"
            << "  -h, --help        Show this help message\n"
            << "  -i, --interval N  Set monitoring interval in seconds (default: 1)\n"
            << "  -o, --once        Run once and exit (default: continuous monitoring)\n"
            << "  -j, --json        Output raw JSON format\n"
            << "  -v, --verbose     Enable verbose output\n"
            << "\n"
            << "Examples:\n"
            << "  " << program_name << "              # Continuous monitoring at 1 second intervals\n"
            << "  " << program_name << " -i 5         # Update every 5 seconds\n"
            << "  " << program_name << " -o           # Run once and exit\n"
            << "  " << program_name << " -j           # Output raw JSON\n"
            << std::endl;
}

/**
 * Parse command line arguments
 */
struct MonitorOptions {
  int interval_sec = 1;
  bool once = false;
  bool json_output = false;
  bool verbose = false;
};

bool ParseArgs(int argc, char* argv[], MonitorOptions& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "-i" || arg == "--interval") {
      if (i + 1 < argc) {
        opts.interval_sec = std::atoi(argv[++i]);
        if (opts.interval_sec < 1) {
          std::cerr << "Error: Interval must be >= 1 second" << std::endl;
          return false;
        }
      } else {
        std::cerr << "Error: -i/--interval requires an argument" << std::endl;
        return false;
      }
    } else if (arg == "-o" || arg == "--once") {
      opts.once = true;
    } else if (arg == "-j" || arg == "--json") {
      opts.json_output = true;
    } else if (arg == "-v" || arg == "--verbose") {
      opts.verbose = true;
    } else {
      std::cerr << "Error: Unknown option: " << arg << std::endl;
      PrintUsage(argv[0]);
      return false;
    }
  }

  return true;
}

/**
 * Print worker statistics in human-readable format
 */
void PrintStats(const chimaera::admin::MonitorTask& task) {
  // Clear screen and move cursor to top
  std::cout << "\033[2J\033[H";

  // Print header
  auto now = std::chrono::system_clock::now();
  auto now_t = std::chrono::system_clock::to_time_t(now);
  std::cout << "==================================================" << std::endl;
  std::cout << "        Chimaera Worker Monitor" << std::endl;
  std::cout << "        " << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S") << std::endl;
  std::cout << "==================================================" << std::endl;
  std::cout << std::endl;

  // Calculate summary statistics
  chi::u32 total_queued = 0;
  chi::u32 total_blocked = 0;
  chi::u32 total_periodic = 0;

  for (const auto& stats : task.info_) {
    total_queued += stats.num_queued_tasks_;
    total_blocked += stats.num_blocked_tasks_;
    total_periodic += stats.num_periodic_tasks_;
  }

  // Print summary
  std::cout << "Summary:" << std::endl;
  std::cout << "  Total Workers:        " << task.info_.size() << std::endl;
  std::cout << "  Total Queued Tasks:   " << total_queued << std::endl;
  std::cout << "  Total Blocked Tasks:  " << total_blocked << std::endl;
  std::cout << "  Total Periodic Tasks: " << total_periodic << std::endl;
  std::cout << std::endl;

  // Print table header
  std::cout << "Worker Details:" << std::endl;
  std::cout << std::setw(6) << "ID"
            << std::setw(10) << "Running"
            << std::setw(10) << "Active"
            << std::setw(12) << "Idle Iters"
            << std::setw(10) << "Queued"
            << std::setw(10) << "Blocked"
            << std::setw(10) << "Periodic"
            << std::setw(15) << "Suspend (us)"
            << std::endl;
  std::cout << std::string(83, '-') << std::endl;

  // Print worker statistics
  for (const auto& stats : task.info_) {
    std::cout << std::setw(6) << stats.worker_id_
              << std::setw(10) << (stats.is_running_ ? "Yes" : "No")
              << std::setw(10) << (stats.is_active_ ? "Yes" : "No")
              << std::setw(12) << stats.idle_iterations_
              << std::setw(10) << stats.num_queued_tasks_
              << std::setw(10) << stats.num_blocked_tasks_
              << std::setw(10) << stats.num_periodic_tasks_
              << std::setw(15) << stats.suspend_period_us_
              << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Press Ctrl+C to exit" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
  // Parse command line arguments
  MonitorOptions opts;
  if (!ParseArgs(argc, argv, opts)) {
    return (argc > 1) ? 1 : 0;  // Return 0 if help was requested, 1 for errors
  }

  // Don't install signal handler - allow default Ctrl+C behavior to terminate immediately
  // std::signal(SIGINT, SignalHandler);
  // std::signal(SIGTERM, SignalHandler);

  if (opts.verbose) {
    HLOG(kInfo, "Initializing Chimaera client...");
  }

  // Initialize Chimaera in client mode
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "ERROR: Failed to initialize Chimaera client" << std::endl;
    std::cerr << "Make sure the Chimaera runtime is running" << std::endl;
    return 1;
  }

  if (opts.verbose) {
    HLOG(kInfo, "Chimaera client initialized successfully");
  }

  // Get admin client
  HLOG(kInfo, "Getting admin client...");
  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    std::cerr << "ERROR: Failed to get admin client" << std::endl;
    return 1;
  }

  HLOG(kInfo, "Admin client obtained successfully");
  if (opts.verbose) {
    HLOG(kInfo, "Connected to admin module");
  }

  // Main monitoring loop
  while (g_keep_running) {
    try {
      // Request worker statistics
      HLOG(kInfo, "Sending AsyncMonitor request...");
      auto future = admin_client->AsyncMonitor(chi::PoolQuery::Local());
      HLOG(kInfo, "AsyncMonitor returned future");

      HLOG(kInfo, "About to call future.Wait()...");
      // Wait for the result
      future.Wait();
      HLOG(kInfo, "future.Wait() returned - monitor task completed");

      // Get the task result (Future has operator->)
      if (future->GetReturnCode() != 0) {
        std::cerr << "ERROR: Monitor task failed with return code "
                  << future->GetReturnCode() << std::endl;
        break;
      }

      // Display the results
      if (opts.json_output) {
        // Output JSON format
        std::cout << "{\"workers\":[";
        bool first = true;
        for (const auto& stats : future->info_) {
          if (!first) std::cout << ",";
          first = false;
          std::cout << "{"
                    << "\"worker_id\":" << stats.worker_id_ << ","
                    << "\"is_running\":" << (stats.is_running_ ? "true" : "false") << ","
                    << "\"is_active\":" << (stats.is_active_ ? "true" : "false") << ","
                    << "\"idle_iterations\":" << stats.idle_iterations_ << ","
                    << "\"num_queued_tasks\":" << stats.num_queued_tasks_ << ","
                    << "\"num_blocked_tasks\":" << stats.num_blocked_tasks_ << ","
                    << "\"num_periodic_tasks\":" << stats.num_periodic_tasks_ << ","
                    << "\"suspend_period_us\":" << stats.suspend_period_us_ << "}";
        }
        std::cout << "]}" << std::endl;
      } else {
        // Print formatted output
        PrintStats(*future);
      }

      // Exit if running once
      if (opts.once) {
        break;
      }

      // Wait for the specified interval
      for (int i = 0; i < opts.interval_sec && g_keep_running; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

    } catch (const std::exception& e) {
      std::cerr << "ERROR: Exception during monitoring: " << e.what() << std::endl;
      break;
    }
  }

  if (opts.verbose) {
    HLOG(kInfo, "Shutting down Chimaera client...");
  }

  // Chimaera cleanup happens automatically
  return 0;
}
