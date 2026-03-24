"""
CTE GPU Benchmark Package

Benchmarks GPU-initiated PutBlob operations through the Content Transfer
Engine (CTE).  wrp_cte_gpu_bench is self-contained: it starts its own
Chimaera runtime internally, so no wrp_runtime package is needed.

Supported test cases:
  putblob      -- GPU client -> CTE via GPU->CPU path (ToLocalCpu)
  putblob_gpu  -- GPU client -> CTE via GPU-local path (Local)
  direct       -- GPU kernel writes directly to pinned host memory (baseline)
  cudamemcpy   -- cudaMemcpyAsync baseline (theoretical PCIe max)
  alloc_test   -- Multi-block ThreadAllocator stress test

Parameters:
- test_case:      Benchmark mode (putblob, putblob_gpu, direct, cudamemcpy, alloc_test)
- rt_blocks:      GPU runtime orchestrator block count
- rt_threads:     GPU runtime orchestrator threads per block
- client_blocks:  GPU client kernel blocks
- client_threads: GPU client kernel threads per block
- io_size:        Per-warp I/O size (supports k/m/g suffixes)
- iterations:     Number of iterations per warp
- output_dir:     Directory for benchmark result files

Assumes wrp_cte_gpu_bench is installed and available in PATH.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os
import re
import signal
import subprocess
import time


class WrpCteGpuBench(Application):
    """
    CTE GPU Bandwidth Benchmark

    Runs wrp_cte_gpu_bench to measure GPU-initiated CTE PutBlob throughput.
    The benchmark is self-contained and starts its own Chimaera runtime.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'test_case',
                'msg': 'Benchmark test case',
                'type': str,
                'choices': [
                    'putblob', 'putblob_gpu', 'putget_gpu',
                    'direct', 'cudamemcpy', 'alloc_test'
                ],
                'default': 'putblob_gpu',
            },
            {
                'name': 'rt_blocks',
                'msg': 'GPU runtime orchestrator block count',
                'type': int,
                'default': 1,
            },
            {
                'name': 'rt_threads',
                'msg': 'GPU runtime orchestrator threads per block',
                'type': int,
                'default': 32,
            },
            {
                'name': 'client_blocks',
                'msg': 'GPU client kernel blocks',
                'type': int,
                'default': 1,
            },
            {
                'name': 'client_threads',
                'msg': 'GPU client kernel threads per block',
                'type': int,
                'default': 256,
            },
            {
                'name': 'io_size',
                'msg': 'Per-warp I/O size (supports k/m/g suffixes)',
                'type': str,
                'default': '128k',
            },
            {
                'name': 'iterations',
                'msg': 'Iterations per warp',
                'type': int,
                'default': 16,
            },
            {
                'name': 'bdev_type',
                'msg': 'Storage backend type',
                'type': str,
                'choices': ['pinned', 'hbm', 'ram'],
                'default': 'pinned',
            },
            {
                'name': 'timeout',
                'msg': 'PollDone timeout in seconds',
                'type': int,
                'default': 60,
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for benchmark results',
                'type': str,
                'default': '/tmp/wrp_cte_gpu_bench',
            },
        ]

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)

        warps = (self.config['client_blocks'] *
                 self.config['client_threads']) // 32
        self.log("CTE GPU benchmark configured")
        self.log(f"  Test case:      {self.config['test_case']}")
        self.log(f"  RT config:      {self.config['rt_blocks']}b x "
                 f"{self.config['rt_threads']}t")
        self.log(f"  Client config:  {self.config['client_blocks']}b x "
                 f"{self.config['client_threads']}t ({warps} warps)")
        self.log(f"  IO/warp:        {self.config['io_size']}")
        self.log(f"  Iterations:     {self.config['iterations']}")
        self.log(f"  Bdev type:      {self.config['bdev_type']}")
        self.log(f"  Timeout:        {self.config['timeout']}s")

    def _kill_stale_processes(self):
        """Kill any leftover wrp_cte_gpu_bench or chimaera processes and
        free port 9413 so the next run can start cleanly."""
        for proc_name in ['wrp_cte_gpu_bench', 'chimaera']:
            try:
                subprocess.run(
                    ['pkill', '-9', '-f', proc_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except Exception:
                pass
        # Also kill anything holding port 9413
        try:
            result = subprocess.run(
                ['lsof', '-ti', ':9413'],
                capture_output=True, text=True)
            for pid in result.stdout.strip().split('\n'):
                if pid.strip():
                    try:
                        os.kill(int(pid.strip()), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass
        time.sleep(2)

    def start(self):
        # Kill stale processes from previous runs before starting
        self._kill_stale_processes()

        Which('wrp_cte_gpu_bench', LocalExecInfo(env=self.mod_env)).run()

        output_file = os.path.join(
            self.config['output_dir'],
            f"cte_gpu_{self.config['test_case']}.txt")

        cmd = ' '.join([
            'wrp_cte_gpu_bench',
            f'--test-case {self.config["test_case"]}',
            f'--rt-blocks {self.config["rt_blocks"]}',
            f'--rt-threads {self.config["rt_threads"]}',
            f'--client-blocks {self.config["client_blocks"]}',
            f'--client-threads {self.config["client_threads"]}',
            f'--io-size {self.config["io_size"]}',
            f'--iterations {self.config["iterations"]}',
            f'--bdev-type {self.config["bdev_type"]}',
            f'--timeout {self.config["timeout"]}',
        ])

        self.log(f"Running: {cmd}")
        self.exec = Exec(f'{cmd} 2>&1 | tee {output_file}',
             LocalExecInfo(env=self.mod_env,
                           collect_output=True)).run()
        self.log(f"Results saved to {output_file}")

    def _get_stat(self, stat_dict):
        output = self.exec.stdout['localhost']
        bandwidth = re.search(r'Bandwidth:\s+([0-9.]+)\s+GB/s', output)
        if bandwidth:
            stat_dict[f'{self.pkg_id}.bandwidth_gbps'] = float(bandwidth.group(1))
        stat_dict[f'{self.pkg_id}.test_case'] = self.config['test_case']
        stat_dict[f'{self.pkg_id}.warps'] = (
            self.config['client_blocks'] * self.config['client_threads']) // 32
        stat_dict[f'{self.pkg_id}.io_size'] = self.config['io_size']
        stat_dict[f'{self.pkg_id}.bdev_type'] = self.config['bdev_type']

    def _plot(self, results_csv, output_dir):
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.log("Skipping plots: pandas or matplotlib not installed")
            return

        df = pd.read_csv(results_csv)

        # Find bandwidth column for this package
        bw_col = None
        warps_col = None
        io_col = None
        for col in df.columns:
            if col.endswith('.bandwidth_gbps'):
                bw_col = col
            elif col.endswith('.warps'):
                warps_col = col
            elif col.endswith('.io_size'):
                io_col = col

        if not bw_col:
            return

        # Bandwidth vs warps
        if warps_col and len(df[warps_col].dropna().unique()) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped = df.groupby(warps_col)[bw_col].mean()
            grouped.plot(kind='bar', ax=ax)
            ax.set_xlabel('Warps')
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.set_title('CTE GPU Bandwidth vs Warp Count')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'bandwidth_vs_warps.png'),
                        dpi=150)
            plt.close(fig)

        # Bandwidth vs IO size
        if io_col and len(df[io_col].dropna().unique()) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped = df.groupby(io_col)[bw_col].mean()
            grouped.plot(kind='bar', ax=ax)
            ax.set_xlabel('I/O Size per Warp')
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.set_title('CTE GPU Bandwidth vs I/O Size')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'bandwidth_vs_iosize.png'),
                        dpi=150)
            plt.close(fig)

        self.log(f"Plots saved to {output_dir}")

    def stop(self):
        self._kill_stale_processes()

    def clean(self):
        output_dir = self.config['output_dir']
        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                path = os.path.join(output_dir, f)
                if os.path.isfile(path) and f.startswith('cte_gpu_'):
                    os.remove(path)
            try:
                os.rmdir(output_dir)
            except OSError:
                pass
