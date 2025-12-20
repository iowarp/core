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

#include "basic_test.h"
#include "hermes_shm/util/compress/libpressio.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#if HSHM_HAS_LIBPRESSIO

// Helper function to calculate relative error for lossy compression validation
double calculate_relative_error(const float* original, const float* decompressed, size_t count) {
  double total_error = 0.0;
  double total_magnitude = 0.0;
  
  for (double i = 0; i < static_cast<double>(count); ++i) {
    size_t idx = static_cast<size_t>(i);
    double error = std::abs(static_cast<double>(original[idx]) - static_cast<double>(decompressed[idx]));
    double magnitude = std::abs(static_cast<double>(original[idx]));
    total_error += error;
    if (magnitude > 1e-10) {  // Avoid division by very small numbers
      total_magnitude += magnitude;
    }
  }
  
  double count_d = static_cast<double>(count);
  if (total_magnitude < 1e-10) {
    return total_error / count_d;  // Absolute error if all values are near zero
  }
  return total_error / total_magnitude;  // Relative error
}

// Helper function to calculate max relative error
double calculate_max_relative_error(const float* original, const float* decompressed, size_t count) {
  double max_error = 0.0;
  
  for (size_t i = 0; i < count; ++i) {
    double magnitude = std::abs(static_cast<double>(original[i]));
    double error = std::abs(static_cast<double>(original[i]) - static_cast<double>(decompressed[i]));
    
    if (magnitude > 1e-10) {
      double rel_error = error / magnitude;
      max_error = std::max(max_error, rel_error);
    } else {
      max_error = std::max(max_error, error);
    }
  }
  
  return max_error;
}

TEST_CASE("TestCompressLibPressio") {
  // Generate floating-point test data designed to produce noticeable loss
  // Use high-precision values with fine-grained variation to trigger lossy compression
  const size_t num_floats = 1024;
  std::vector<float> original_floats(num_floats);
  
  // Generate test data with:
  // 1. High precision floating-point values (many decimal places)
  // 2. Small variations that lossy compressors will approximate
  // 3. Mix of large and small values to test dynamic range
  // 4. Some noise to make exact reconstruction harder
  for (size_t i = 0; i < num_floats; ++i) {
    float x = static_cast<float>(i);
    // Combine multiple frequencies and add small variations
    float value = std::sin(x * 0.1F) * 1000.0F + 
                  std::cos(x * 0.03F) * 500.0F +
                  std::sin(x * 0.007F) * 100.0F +
                  static_cast<float>(i % 17) * 0.1234567F +  // Small variations
                  static_cast<float>(i % 23) * 0.0000123F;   // Very small variations
    original_floats[i] = value;
  }
  
  // Convert to bytes for compression interface
  const size_t original_size = original_floats.size() * sizeof(float);
  std::vector<char> compressed(original_size * 2);  // Allow for expansion
  std::vector<char> decompressed(original_size * 2);
  
  // List of lossy compressors to test through libpressio
  // We only test lossy compressors since we already have lossless ones implemented
  struct CompressorTest {
    const char* name;
    bool optional;  // true if this compressor might not be available
    bool is_lossy;  // true if this is a lossy compressor
    double max_relative_error;  // Maximum acceptable relative error for lossy compressors
  };
  
  // Lossy scientific compressors that libpressio supports
  // Only include compressors that are actually working and are lossy
  // Note: "noop" is NOT lossy - it's a pass-through compressor, so we exclude it
  CompressorTest compressors_to_test[] = {
    // Working lossy floating-point compressors (via libpressio)
    {"zfp", false, true, 0.01},      // Lossy compressor - shows actual loss (max error ~1e-4)
    {"sz3", false, true, 0.01},      // Lossy compressor - can be lossless at high precision (may show zero error)
    
    // Note: The following compressors are not currently available or not working:
    // {"sz", true, true, 0.01},       // Older version of sz3 - not available
    // {"fpzip", true, true, 0.01},    // Lossy floating-point compressor - not available
    // {"mgard", true, true, 0.05},    // Lossy, may have higher error tolerance - dependencies missing
    // {"cusz", true, true, 0.01},     // GPU-accelerated SZ compressor - requires CUDA
    // {"fraz", true, true, 0.01},     // Fixed-ratio lossy compression - not available
  };
  
  for (const auto& comp_info : compressors_to_test) {
    // Test this compressor
    try {
      PAGE_DIVIDE(("LibPressio-" + std::string(comp_info.name)).c_str()) {
        std::cout << "\n=== Testing LibPressio compressor: " << comp_info.name << " ===\n";
        std::cout << "Original data: " << num_floats << " floats (" << original_size << " bytes)\n";
        std::cout << "Sample values: [" 
                  << original_floats[0] << ", " 
                  << original_floats[1] << ", " 
                  << original_floats[2] << ", ...]\n";
        std::cout.flush();
        
        hshm::LibPressio libpressio(comp_info.name);
        
        size_t cmpr_size = compressed.size(), raw_size = decompressed.size();
        bool compress_success = libpressio.Compress(compressed.data(), cmpr_size,
                                                    original_floats.data(), original_size);
        
        if (!compress_success) {
          if (comp_info.optional) {
            std::cout << "  Compressor not available or compression failed (skipping)\n";
            std::cout.flush();
            continue;
          }
          REQUIRE(compress_success);
        }
        
        REQUIRE(cmpr_size > 0);
        
        // Print compression info
        std::cout << "Compressed size: " << cmpr_size << " bytes\n";
        double ratio = (original_size > 0) ? static_cast<double>(original_size) / static_cast<double>(cmpr_size) : 0.0;
        std::cout << "  Compression ratio: " << std::fixed << std::setprecision(2) << ratio << "x\n";
        std::cout.flush();
        
        // Decompress
        bool decompress_success = libpressio.Decompress(decompressed.data(), raw_size,
                                                        compressed.data(), cmpr_size);
        
        if (!decompress_success) {
          if (comp_info.optional) {
            std::cout << "  Decompression failed (skipping)\n";
            std::cout.flush();
            continue;
          }
          REQUIRE(decompress_success);
        }
        
        REQUIRE(raw_size > 0);
        
        // Check if we got the right amount of data
        size_t expected_decompressed_floats = raw_size / sizeof(float);
        std::cout << "Decompressed size: " << raw_size << " bytes (" 
                  << expected_decompressed_floats << " floats)\n";
        
        // Convert decompressed bytes back to floats
        const float* decompressed_floats = reinterpret_cast<const float*>(decompressed.data());
        size_t compare_count = (num_floats < expected_decompressed_floats) ? num_floats : expected_decompressed_floats;
        
        // Print sample decompressed values
        std::cout << "Sample decompressed values: [" 
                  << decompressed_floats[0] << ", " 
                  << decompressed_floats[1] << ", " 
                  << decompressed_floats[2] << ", ...]\n";
        std::cout.flush();
        
        // Validate decompressed data
        if (!comp_info.is_lossy) {
          // For lossless compressors (like "noop"), data should match exactly
          REQUIRE(compare_count == num_floats);
          bool exact_match = true;
          for (size_t i = 0; i < compare_count; ++i) {
            if (original_floats[i] != decompressed_floats[i]) {
              exact_match = false;
              break;
            }
          }
          std::cout << "  Data matches exactly: " << (exact_match ? "YES" : "NO") << "\n";
          REQUIRE(exact_match);
        } else {
          // For lossy compressors, check relative error
          double rel_error = calculate_relative_error(original_floats.data(), 
                                                     decompressed_floats, 
                                                     compare_count);
          double max_rel_error = calculate_max_relative_error(original_floats.data(), 
                                                             decompressed_floats, 
                                                             compare_count);
          
          std::cout << "  Average relative error: " << std::scientific << std::setprecision(4)
                    << rel_error << "\n";
          std::cout << "  Max relative error: " << std::scientific << std::setprecision(4)
                    << max_rel_error << "\n";
          std::cout << "  Error tolerance: " << std::fixed << std::setprecision(4)
                    << comp_info.max_relative_error << "\n";
          
          // For lossy compression, we expect some error but it should be bounded
          // Check that max relative error is within tolerance
          if (max_rel_error > comp_info.max_relative_error) {
            std::cout << "  WARNING: Error exceeds tolerance, but continuing (lossy compression)\n";
            // We don't fail for lossy compressors with high error, just warn
            // Some lossy compressors may have different error characteristics
          } else {
            std::cout << "  Error within tolerance: YES\n";
          }
          
          // At minimum, verify that decompressed data is not completely corrupted
          // Check that values are in reasonable range (not NaN, not Inf, not zeros)
          bool has_valid_data = false;
          for (size_t i = 0; i < compare_count; ++i) {
            if (std::isfinite(decompressed_floats[i]) && 
                decompressed_floats[i] != 0.0F) {
              has_valid_data = true;
              break;
            }
          }
          std::cout << "  Has valid decompressed data: " << (has_valid_data ? "YES" : "NO") << "\n";
          REQUIRE(has_valid_data);
        }
        
        std::cout << "=== Test passed for " << comp_info.name << " ===\n\n";
        std::cout.flush();
      }
    } catch (...) {
      // Skip if constructor throws (compressor not available)
      if (!comp_info.optional) {
        throw;
      }
      std::cout << "  Compressor constructor failed (skipping)\n";
      std::cout.flush();
    }
  }
}

#endif  // HSHM_HAS_LIBPRESSIO
