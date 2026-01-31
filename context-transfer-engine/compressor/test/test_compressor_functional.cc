/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * Compressor ChiMod Functional Tests
 *
 * Tests the actual functionality of the compressor chimod tasks:
 * - CompressTask: Compression with various libraries
 * - DecompressTask: Decompression and data integrity
 * - DynamicScheduleTask: Intelligent compression selection
 * - Round-trip: Compress + Decompress data verification
 */

#include "simple_test.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>

#include <chimaera/chimaera.h>
#include <wrp_cte/compressor/compressor_client.h>
#include <wrp_cte/compressor/compressor_tasks.h>
#include <wrp_cte/compressor/compressor_runtime.h>

using namespace wrp_cte::compressor;

namespace {

// Compression library IDs
namespace CompLib {
  constexpr int NONE = 0;
  constexpr int BROTLI = 0;
  constexpr int BZIP2 = 1;
  constexpr int BLOSC2 = 2;
  constexpr int FPZIP = 3;
  constexpr int LZ4 = 4;
  constexpr int LZMA = 5;
  constexpr int SNAPPY = 6;
  constexpr int SZ3 = 7;
  constexpr int ZFP = 8;
  constexpr int ZLIB = 9;
  constexpr int ZSTD = 10;
}

/**
 * Generate test data with specified pattern
 */
std::vector<char> GenerateTestData(size_t size, const std::string& pattern) {
  std::vector<char> data(size);

  if (pattern == "zeros") {
    // All zeros - highly compressible
    std::fill(data.begin(), data.end(), 0);
  } else if (pattern == "ones") {
    // All ones - highly compressible
    std::fill(data.begin(), data.end(), 1);
  } else if (pattern == "repeating") {
    // Repeating pattern - moderately compressible
    const char pattern_bytes[] = {0x01, 0x02, 0x03, 0x04};
    for (size_t i = 0; i < size; ++i) {
      data[i] = pattern_bytes[i % 4];
    }
  } else if (pattern == "random") {
    // Random data - poorly compressible
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(dis(gen));
    }
  } else if (pattern == "text") {
    // Text-like data - moderately compressible
    const char* text = "The quick brown fox jumps over the lazy dog. ";
    size_t text_len = strlen(text);
    for (size_t i = 0; i < size; ++i) {
      data[i] = text[i % text_len];
    }
  }

  return data;
}

/**
 * Initialize Chimaera runtime for compressor tests
 */
void InitializeChimaera() {
  // Initialize Chimaera runtime in client mode with runtime
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!success) {
    throw std::runtime_error("Failed to initialize Chimaera runtime");
  }
}

/**
 * Global compressor pool for testing (initialized once)
 */
chi::PoolId g_compressor_pool_id;
bool g_compressor_initialized = false;

/**
 * Create compressor pool for testing
 */
chi::PoolId CreateCompressorPool() {
  if (!g_compressor_initialized) {
    InitializeChimaera();

    // Generate a unique pool ID for this test session
    int rand_id = 1000 + (std::rand() % 9000);  // Random ID 1000-9999
    g_compressor_pool_id = chi::PoolId(static_cast<chi::u32>(rand_id), 0);

    // Create the compressor pool using Client's AsyncCreate
    Client client;
    auto create_task = client.AsyncCreate(
        chi::PoolQuery::Dynamic(),
        "test_compressor_pool",
        g_compressor_pool_id);
    create_task.Wait();

    // Check result
    if (create_task->GetReturnCode() != 0) {
      throw std::runtime_error("Failed to create compressor pool");
    }

    g_compressor_initialized = true;
  }

  return g_compressor_pool_id;
}

} // anonymous namespace

/**
 * Test Case 1: Compress Task - LZ4 Compression
 */
TEST_CASE("CompressTask - LZ4 Compression", "[compressor][functional][compress]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  // Generate test data (highly compressible)
  auto test_data = GenerateTestData(4096, "repeating");

  // Create compression context
  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  // Execute compression task
  auto task = client.AsyncCompress(test_data.data(), test_data.size(), context);
  task.Wait();

  // Verify compression succeeded
  REQUIRE(task->return_code_ == 0);
  REQUIRE(task->output_data_ != nullptr);
  REQUIRE(task->output_size_ > 0);
  REQUIRE(task->output_size_ < test_data.size()); // Should compress
  REQUIRE(task->compress_time_ms_ > 0.0);

  INFO("Compressed " << test_data.size() << " bytes to " << task->output_size_
       << " bytes (ratio: " << (double)test_data.size() / task->output_size_ << ")");

  // Cleanup
  delete[] static_cast<char*>(task->output_data_);
}

/**
 * Test Case 2: Decompress Task - LZ4 Decompression
 */
TEST_CASE("DecompressTask - LZ4 Decompression", "[compressor][functional][decompress]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  // Generate and compress test data
  auto original_data = GenerateTestData(4096, "repeating");

  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  auto compress_task = client.AsyncCompress(original_data.data(), original_data.size(), context);
  compress_task.Wait();
  REQUIRE(compress_task->return_code_ == 0);

  // Now decompress
  auto decompress_task = client.AsyncDecompress(
      compress_task->output_data_, compress_task->output_size_,
      original_data.size(), CompLib::LZ4, 2);
  decompress_task.Wait();

  // Verify decompression succeeded
  REQUIRE(decompress_task->return_code_ == 0);
  REQUIRE(decompress_task->output_data_ != nullptr);
  REQUIRE(decompress_task->output_size_ == original_data.size());
  REQUIRE(decompress_task->decompress_time_ms_ > 0.0);

  INFO("Decompressed " << compress_task->output_size_ << " bytes to "
       << decompress_task->output_size_ << " bytes");

  // Cleanup
  delete[] static_cast<char*>(compress_task->output_data_);
  delete[] static_cast<char*>(decompress_task->output_data_);
}

/**
 * Test Case 3: Round-trip - Compress + Decompress Data Integrity
 */
TEST_CASE("Round-trip - Data Integrity", "[compressor][functional][roundtrip]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  SECTION("LZ4 - Zeros pattern") {
    auto original_data = GenerateTestData(8192, "zeros");

    Context context;
    context.compress_lib_ = CompLib::LZ4;
    context.compress_preset_ = 2;

    // Compress
    auto compress_task = client.AsyncCompress(original_data.data(), original_data.size(), context);
    compress_task.Wait();
    REQUIRE(compress_task->return_code_ == 0);

    // Decompress
    auto decompress_task = client.AsyncDecompress(
        compress_task->output_data_, compress_task->output_size_,
        original_data.size(), CompLib::LZ4, 2);
    decompress_task.Wait();
    REQUIRE(decompress_task->return_code_ == 0);

    // Verify data integrity
    REQUIRE(decompress_task->output_size_ == original_data.size());
    int cmp = std::memcmp(original_data.data(), decompress_task->output_data_, original_data.size());
    REQUIRE(cmp == 0);

    INFO("Data integrity verified for zeros pattern");

    delete[] static_cast<char*>(compress_task->output_data_);
    delete[] static_cast<char*>(decompress_task->output_data_);
  }

  SECTION("ZSTD - Text pattern") {
    auto original_data = GenerateTestData(16384, "text");

    Context context;
    context.compress_lib_ = CompLib::ZSTD;
    context.compress_preset_ = 3;

    // Compress
    auto compress_task = client.AsyncCompress(original_data.data(), original_data.size(), context);
    compress_task.Wait();
    REQUIRE(compress_task->return_code_ == 0);

    // Decompress
    auto decompress_task = client.AsyncDecompress(
        compress_task->output_data_, compress_task->output_size_,
        original_data.size(), CompLib::ZSTD, 3);
    decompress_task.Wait();
    REQUIRE(decompress_task->return_code_ == 0);

    // Verify data integrity
    REQUIRE(decompress_task->output_size_ == original_data.size());
    int cmp = std::memcmp(original_data.data(), decompress_task->output_data_, original_data.size());
    REQUIRE(cmp == 0);

    INFO("Data integrity verified for text pattern");

    delete[] static_cast<char*>(compress_task->output_data_);
    delete[] static_cast<char*>(decompress_task->output_data_);
  }
}

/**
 * Test Case 4: Compression Ratio - Different Data Patterns
 */
TEST_CASE("Compression Ratio - Data Patterns", "[compressor][functional][ratio]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  SECTION("Highly compressible - zeros") {
    auto data = GenerateTestData(65536, "zeros");

    auto task = client.AsyncCompress(data.data(), data.size(), context);
    task.Wait();
    REQUIRE(task->return_code_ == 0);

    double ratio = (double)data.size() / task->output_size_;
    REQUIRE(ratio > 10.0); // Should compress very well

    INFO("Zeros pattern: ratio = " << ratio);
    delete[] static_cast<char*>(task->output_data_);
  }

  SECTION("Moderately compressible - repeating") {
    auto data = GenerateTestData(65536, "repeating");

    auto task = client.AsyncCompress(data.data(), data.size(), context);
    task.Wait();
    REQUIRE(task->return_code_ == 0);

    double ratio = (double)data.size() / task->output_size_;
    REQUIRE(ratio > 2.0); // Should compress moderately

    INFO("Repeating pattern: ratio = " << ratio);
    delete[] static_cast<char*>(task->output_data_);
  }

  SECTION("Poorly compressible - random") {
    auto data = GenerateTestData(65536, "random");

    auto task = client.AsyncCompress(data.data(), data.size(), context);
    task.Wait();
    REQUIRE(task->return_code_ == 0);

    double ratio = (double)data.size() / task->output_size_;
    REQUIRE(ratio < 1.5); // Won't compress much

    INFO("Random pattern: ratio = " << ratio);
    delete[] static_cast<char*>(task->output_data_);
  }
}

/**
 * Test Case 5: DynamicSchedule - Compression Selection
 */
TEST_CASE("DynamicSchedule - Compression Selection", "[compressor][functional][schedule]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  // Generate test data
  auto test_data = GenerateTestData(32768, "text");

  // Create context with dynamic scheduling enabled
  Context context;
  context.dynamic_compress_ = 2; // Enable dynamic scheduling

  // Execute dynamic scheduling
  auto task = client.AsyncDynamicSchedule(test_data.size(), test_data.data(), context);
  task.Wait();

  // Verify scheduling succeeded
  REQUIRE(task->return_code_ == 0);
  REQUIRE(task->context_.compress_lib_ > 0); // Should select a library
  REQUIRE(task->context_.compress_preset_ > 0); // Should select a preset

  INFO("Selected library: " << task->context_.compress_lib_
       << ", preset: " << task->context_.compress_preset_);
}

/**
 * Test Case 6: Multiple Compression Libraries
 */
TEST_CASE("Multiple Libraries - Compression", "[compressor][functional][multilib]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  auto test_data = GenerateTestData(16384, "text");

  // Test LZ4
  SECTION("LZ4") {
    Context context;
    context.compress_lib_ = CompLib::LZ4;
    context.compress_preset_ = 2;

    auto task = client.AsyncCompress(test_data.data(), test_data.size(), context);
    task.Wait();

    REQUIRE(task->return_code_ == 0);
    REQUIRE(task->output_size_ < test_data.size());

    INFO("LZ4: " << test_data.size() << " -> " << task->output_size_);
    delete[] static_cast<char*>(task->output_data_);
  }

  // Test ZSTD
  SECTION("ZSTD") {
    Context context;
    context.compress_lib_ = CompLib::ZSTD;
    context.compress_preset_ = 3;

    auto task = client.AsyncCompress(test_data.data(), test_data.size(), context);
    task.Wait();

    REQUIRE(task->return_code_ == 0);
    REQUIRE(task->output_size_ < test_data.size());

    INFO("ZSTD: " << test_data.size() << " -> " << task->output_size_);
    delete[] static_cast<char*>(task->output_data_);
  }

  // Test ZLIB
  SECTION("ZLIB") {
    Context context;
    context.compress_lib_ = CompLib::ZLIB;
    context.compress_preset_ = 2;

    auto task = client.AsyncCompress(test_data.data(), test_data.size(), context);
    task.Wait();

    REQUIRE(task->return_code_ == 0);
    REQUIRE(task->output_size_ < test_data.size());

    INFO("ZLIB: " << test_data.size() << " -> " << task->output_size_);
    delete[] static_cast<char*>(task->output_data_);
  }
}

/**
 * Test Case 7: Error Handling - Invalid Parameters
 */
TEST_CASE("Error Handling - Invalid Parameters", "[compressor][functional][error]") {
  auto pool_id = CreateCompressorPool();
  Client client(pool_id);

  auto test_data = GenerateTestData(1024, "text");

  SECTION("Invalid compression library") {
    Context context;
    context.compress_lib_ = 99; // Invalid library ID
    context.compress_preset_ = 2;

    auto task = client.AsyncCompress(test_data.data(), test_data.size(), context);
    task.Wait();

    // Should fail gracefully
    REQUIRE(task->return_code_ != 0);
    INFO("Correctly handled invalid library ID");
  }

  SECTION("Null input data") {
    Context context;
    context.compress_lib_ = CompLib::LZ4;
    context.compress_preset_ = 2;

    auto task = client.AsyncCompress(nullptr, test_data.size(), context);
    task.Wait();

    // Should fail gracefully
    REQUIRE(task->return_code_ != 0);
    INFO("Correctly handled null input data");
  }
}

// Main function using simple_test.h framework
SIMPLE_TEST_MAIN()
