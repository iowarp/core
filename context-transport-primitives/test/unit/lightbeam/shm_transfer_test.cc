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
 * Unit tests for ShmTransport WriteTransfer/ReadTransfer (SPSC ring buffer)
 *
 * Tests the chunked data transfer mechanism using ShmTransferInfo and
 * copy_space directly, without requiring the full Chimaera runtime.
 */

#include <catch2/catch_all.hpp>

#include <hermes_shm/lightbeam/shm_transport.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

using namespace hshm::lbm;

// Helper context for tests
template <size_t N>
struct TestTransferContext {
  char copy_space[N];
  ShmTransferInfo shm_info;

  TestTransferContext() {
    std::memset(copy_space, 0, sizeof(copy_space));
    shm_info.copy_space_size_ = N;
  }

  LbmContext MakeCtx() {
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = &shm_info;
    return ctx;
  }
};

// Generate test data with a predictable pattern
static std::vector<char> GenerateTestData(size_t size) {
  std::vector<char> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  return data;
}

// Verify data matches the expected pattern
static bool VerifyTestData(const std::vector<char>& data, size_t expected_size) {
  if (data.size() != expected_size) {
    return false;
  }
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] != static_cast<char>(i % 256)) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// Single-Chunk Transfer Tests (data fits in copy_space)
// ============================================================================

TEST_CASE("ShmTransfer - Single Chunk Transfer", "[shm_transfer][single]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 1000;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;
  auto ctx = ctx_store.MakeCtx();

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  // Write all data (fits in single pass since DATA_SIZE < COPY_SPACE_SIZE)
  ShmTransport::WriteTransfer(send_data.data(), send_data.size(), ctx);

  // Read it back
  std::vector<char> recv_data(DATA_SIZE);
  ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);

  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}

// ============================================================================
// Multi-Chunk Threaded Transfer Tests (data larger than copy_space)
// ============================================================================

TEST_CASE("ShmTransfer - Multi Chunk Transfer (Threaded)", "[shm_transfer][multi]") {
  constexpr size_t COPY_SPACE_SIZE = 256;
  constexpr size_t DATA_SIZE = 1000;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  std::vector<char> recv_data(DATA_SIZE, 0);
  bool send_done = false;
  bool recv_done = false;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::WriteTransfer(send_data.data(), send_data.size(), ctx);
    send_done = true;
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);
    recv_done = true;
  });

  sender.join();
  receiver.join();

  REQUIRE(send_done);
  REQUIRE(recv_done);
  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}

// ============================================================================
// Large Data Transfer Tests
// ============================================================================

TEST_CASE("ShmTransfer - Large Data (64KB) Threaded", "[shm_transfer][large]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 64 * 1024;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  std::vector<char> recv_data(DATA_SIZE, 0);

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::WriteTransfer(send_data.data(), send_data.size(), ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}

TEST_CASE("ShmTransfer - Very Large Data (1MB) Threaded", "[shm_transfer][verylarge]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 1024 * 1024;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  std::vector<char> recv_data(DATA_SIZE, 0);

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::WriteTransfer(send_data.data(), send_data.size(), ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("ShmTransfer - Empty Data", "[shm_transfer][edge]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;
  auto ctx = ctx_store.MakeCtx();

  // Zero-length write and read should be no-ops
  ShmTransport::WriteTransfer(nullptr, 0, ctx);
  ShmTransport::ReadTransfer(nullptr, 0, ctx);

  // Verify ring buffer counters unchanged
  REQUIRE(ctx_store.shm_info.total_written_.load() == 0);
  REQUIRE(ctx_store.shm_info.total_read_.load() == 0);
}

TEST_CASE("ShmTransfer - Exact Copy Space Size", "[shm_transfer][edge]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = COPY_SPACE_SIZE;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;
  auto ctx = ctx_store.MakeCtx();

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  // Exact fit - fills entire ring buffer
  ShmTransport::WriteTransfer(send_data.data(), send_data.size(), ctx);

  std::vector<char> recv_data(DATA_SIZE, 0);
  ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);

  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}

TEST_CASE("ShmTransfer - One Byte Over Copy Space (Threaded)", "[shm_transfer][edge]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = COPY_SPACE_SIZE + 1;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  std::vector<char> recv_data(DATA_SIZE, 0);

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::WriteTransfer(send_data.data(), send_data.size(), ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}

// ============================================================================
// Concurrent Transfer with Delays
// ============================================================================

TEST_CASE("ShmTransfer - Concurrent with Delays", "[shm_transfer][threaded]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = 10000;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  std::vector<char> recv_data(DATA_SIZE, 0);

  // Sender pauses between chunks to simulate bursty writes
  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    size_t offset = 0;
    while (offset < DATA_SIZE) {
      size_t chunk = std::min(DATA_SIZE - offset, size_t(512));
      ShmTransport::WriteTransfer(send_data.data() + offset, chunk, ctx);
      offset += chunk;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::ReadTransfer(recv_data.data(), DATA_SIZE, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyTestData(recv_data, DATA_SIZE));
}
