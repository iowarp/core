/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * ZeroMQ IPC Round-Trip Latency Benchmark
 *
 * Measures ZMQ round-trip latency over POSIX domain sockets (IPC transport).
 * Client sends a message -> server receives -> server sends back -> client
 * receives. Reports min, max, median, mean, and p99 latency.
 *
 * Usage:
 *   zmq_ipc_latency_benchmark [num_iterations] [message_size]
 *
 * Parameters:
 *   num_iterations: Number of round-trip iterations (default: 10000)
 *   message_size:   Message size in bytes (default: 256)
 *
 * Examples:
 *   zmq_ipc_latency_benchmark
 *   zmq_ipc_latency_benchmark 50000
 *   zmq_ipc_latency_benchmark 50000 1024
 */

#include <zmq.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

static const char* kEndpoint = "ipc:///tmp/zmq_ipc_latency_bench";
static const int kWarmupIterations = 100;

void ServerThread(int num_iterations) {
  void* ctx = zmq_ctx_new();
  void* sock = zmq_socket(ctx, ZMQ_REP);
  zmq_bind(sock, kEndpoint);

  int total = kWarmupIterations + num_iterations;
  std::vector<char> buf(65536);

  for (int i = 0; i < total; ++i) {
    int nbytes = zmq_recv(sock, buf.data(), buf.size(), 0);
    if (nbytes < 0) break;
    zmq_send(sock, buf.data(), nbytes, 0);
  }

  zmq_close(sock);
  zmq_ctx_destroy(ctx);
}

int main(int argc, char** argv) {
  int num_iterations = 10000;
  int message_size = 256;

  if (argc > 1) {
    num_iterations = std::atoi(argv[1]);
    if (num_iterations <= 0) {
      std::cerr << "Error: num_iterations must be positive\n";
      return 1;
    }
  }
  if (argc > 2) {
    message_size = std::atoi(argv[2]);
    if (message_size <= 0) {
      std::cerr << "Error: message_size must be positive\n";
      return 1;
    }
  }

  std::cout << "ZMQ IPC Round-Trip Latency Benchmark\n";
  std::cout << "  Iterations:   " << num_iterations << "\n";
  std::cout << "  Message size: " << message_size << " bytes\n";
  std::cout << "  Warmup:       " << kWarmupIterations << " iterations\n";
  std::cout << "  Endpoint:     " << kEndpoint << "\n\n";

  // Remove stale IPC endpoint file
  unlink("/tmp/zmq_ipc_latency_bench");

  // Start server thread
  std::thread server(ServerThread, num_iterations);

  // Client setup
  void* ctx = zmq_ctx_new();
  void* sock = zmq_socket(ctx, ZMQ_REQ);

  // Brief sleep to let server bind
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  zmq_connect(sock, kEndpoint);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::vector<char> send_buf(message_size, 'A');
  std::vector<char> recv_buf(message_size);

  // Warmup phase
  for (int i = 0; i < kWarmupIterations; ++i) {
    zmq_send(sock, send_buf.data(), message_size, 0);
    zmq_recv(sock, recv_buf.data(), recv_buf.size(), 0);
  }

  // Timed phase
  std::vector<double> latencies(num_iterations);

  for (int i = 0; i < num_iterations; ++i) {
    auto start = std::chrono::steady_clock::now();
    zmq_send(sock, send_buf.data(), message_size, 0);
    zmq_recv(sock, recv_buf.data(), recv_buf.size(), 0);
    auto end = std::chrono::steady_clock::now();

    latencies[i] = std::chrono::duration<double, std::milli>(end - start).count();
  }

  // Cleanup client
  zmq_close(sock);
  zmq_ctx_destroy(ctx);
  server.join();

  // Remove IPC endpoint file
  unlink("/tmp/zmq_ipc_latency_bench");

  // Compute statistics
  std::sort(latencies.begin(), latencies.end());

  double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
  double mean = sum / num_iterations;
  double min = latencies.front();
  double max = latencies.back();
  double median = latencies[num_iterations / 2];
  double p99 = latencies[static_cast<size_t>(num_iterations * 0.99)];

  std::cout << "=== Results ===\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  Min:    " << min << " ms\n";
  std::cout << "  Max:    " << max << " ms\n";
  std::cout << "  Median: " << median << " ms\n";
  std::cout << "  Mean:   " << mean << " ms\n";
  std::cout << "  p99:    " << p99 << " ms\n";
  std::cout << "===============\n";

  return 0;
}
