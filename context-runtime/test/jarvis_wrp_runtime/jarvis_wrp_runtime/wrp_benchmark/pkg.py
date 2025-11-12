"""
IOWarp Throughput Benchmark Package

Benchmarks task throughput with three test cases:
1. bdev_io: Full BDev I/O path (Allocate -> Write -> Free operations)
2. bdev_allocation: BDev allocation only (AllocateBlocks -> FreeBlocks operations)
3. latency: Pure task round-trip latency using MOD_NAME Custom function

Each thread continuously performs operations for a specified duration
to measure sustained performance.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os


class WrpBenchmark(Application):
    """
    IOWarp Throughput Benchmark

    Measures sustained task throughput with three test cases:
    1. bdev_io: Full BDev I/O path (Allocate -> Write -> Free operations)
       Reports: IOPS, bandwidth (MB/s), and average latency
    2. bdev_allocation: BDev allocation only (AllocateBlocks -> FreeBlocks operations)
       Reports: allocation throughput (ops/sec) and average latency
    3. latency: Pure task round-trip latency using MOD_NAME Custom function
       Reports: task throughput (ops/sec) and average latency

    Each thread continuously performs operations for a specified duration.

    Parameters:
    - threads: Number of concurrent client threads
    - duration: Time to run benchmark in seconds
    - test_case: Which test to run (bdev_io, bdev_allocation, latency)
    - max_file_size: Maximum size of BDev file container (bdev_io and bdev_allocation only)
    - io_size: Size of each I/O operation (bdev_io only)
    - lane_policy: Task lane mapping strategy

    Assumes wrp_run_thrpt_benchmark is installed and available in PATH.
    Requires wrp_runtime to be running.
    """

    def _init(self):
        """Initialize benchmark variables"""
        self.output_file = None

    def _configure_menu(self):
        """Define configuration options for benchmark"""
        return [
            {
                'name': 'threads',
                'msg': 'Number of client threads',
                'type': int,
                'default': 4
            },
            {
                'name': 'duration',
                'msg': 'Duration to run benchmark (seconds)',
                'type': float,
                'default': 10.0
            },
            {
                'name': 'test_case',
                'msg': 'Test case to run (bdev_io, bdev_allocation, latency)',
                'type': str,
                'choices': ['bdev_io', 'bdev_allocation', 'latency'],
                'default': 'bdev_io'
            },
            {
                'name': 'max_file_size',
                'msg': 'Maximum file size (supports suffixes: k, m, g)',
                'type': 'size_type',
                'default': '1g'
            },
            {
                'name': 'io_size',
                'msg': 'I/O size per operation (supports suffixes: k, m, g)',
                'type': 'size_type',
                'default': '4k'
            },
            {
                'name': 'lane_policy',
                'msg': 'Lane mapping policy (map_by_pid_tid, round_robin, random)',
                'type': str,
                'choices': ['map_by_pid_tid', 'round_robin', 'random'],
                'default': 'round_robin'
            },
            {
                'name': 'verbose',
                'msg': 'Enable verbose per-thread output',
                'type': bool,
                'default': False
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for benchmark results',
                'type': str,
                'default': '/tmp/wrp_benchmark'
            }
        ]

    def _configure(self, **kwargs):
        """Configure the benchmark"""
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.output_file = os.path.join(self.config['output_dir'], 'benchmark_results.txt')

        # Set benchmark environment variables
        self.setenv('BENCHMARK_OUTPUT_DIR', self.config['output_dir'])

        self.log("IOWarp throughput benchmark configured")
        self.log(f"  Test case: {self.config['test_case']}")
        self.log(f"  Threads: {self.config['threads']}")
        self.log(f"  Duration: {self.config['duration']} seconds")
        if self.config['test_case'] != 'latency':
            self.log(f"  Max file size: {self.config['max_file_size']}")
        if self.config['test_case'] == 'bdev_io':
            self.log(f"  I/O size per operation: {self.config['io_size']}")
        self.log(f"  Lane policy: {self.config['lane_policy']}") 

    def start(self):
        """Run the benchmark"""
        # Verify benchmark executable is available
        Which('wrp_run_thrpt_benchmark', LocalExecInfo(env=self.mod_env)).run()

        self.log(f"Starting {self.config['test_case']} throughput benchmark")

        # Build benchmark command
        cmd_parts = [
            'wrp_run_thrpt_benchmark',
            f'--test-case {self.config["test_case"]}',
            f'--threads {self.config["threads"]}',
            f'--duration {self.config["duration"]}',
            f'--lane-policy {self.config["lane_policy"]}',
            f'--output-dir {self.config["output_dir"]}'
        ]

        # Add BDev-specific parameters for bdev_io and bdev_allocation
        if self.config['test_case'] != 'latency':
            cmd_parts.append(f'--max-file-size {self.config["max_file_size"]}')

        # Add io-size only for bdev_io test
        if self.config['test_case'] == 'bdev_io':
            cmd_parts.append(f'--io-size {self.config["io_size"]}')

        if self.config['verbose']:
            cmd_parts.append('--verbose')

        cmd = ' '.join(cmd_parts)

        # Redirect output to file and console
        cmd_with_redirect = f'{cmd} 2>&1 | tee {self.output_file}'

        # Execute benchmark
        Exec(cmd_with_redirect, LocalExecInfo(env=self.mod_env)).run()

        self.log(f"Benchmark completed - results saved to {self.output_file}")

    def stop(self):
        """Stop method - benchmark completes automatically"""
        pass

    def clean(self):
        """Clean benchmark output"""
        self.log("Cleaning benchmark data")

        # Remove output file
        if self.output_file and os.path.exists(self.output_file):
            os.remove(self.output_file)

        # Remove output directory if empty
        try:
            os.rmdir(self.config['output_dir'])
        except OSError:
            pass  # Directory not empty or doesn't exist

        self.log("Cleanup completed")
