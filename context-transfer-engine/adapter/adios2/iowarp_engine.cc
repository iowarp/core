/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of IOWarp Core. The full IOWarp Core copyright         *
 * notice, including terms governing use, modification, and redistribution,  *
 * is contained in the COPYING file, which can be found at the top directory.*
 * If you do not have access to the file, you may request a copy             *
 * from scslab@iit.edu.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "iowarp_engine.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if HSHM_ENABLE_COMPRESS
#include <lzo/lzo1x.h>  // Required for LZO initialization
#endif

namespace coeus {

/**
 * Main constructor for IowarpEngine
 * @param io ADIOS2 IO object
 * @param name Engine name
 * @param mode Open mode (Read, Write, Append, etc.)
 * @param comm MPI communicator
 */
IowarpEngine::IowarpEngine(adios2::core::IO &io, const std::string &name,
                           const adios2::Mode mode, adios2::helper::Comm comm)
    : adios2::plugin::PluginEngineInterface(io, name, mode, std::move(comm)),
      current_tag_(nullptr),
      current_step_(0),
      rank_(m_Comm.Rank()),
      open_(false),
      compressor_(nullptr),
      compress_type_("none") {
  // CTE client will be accessed via WRP_CTE_CLIENT singleton when needed
  wrp_cte::core::WRP_CTE_CLIENT_INIT("", chi::PoolQuery::Local());

  // Initialize compression from environment variable
  InitCompression();
}

/**
 * Destructor
 */
IowarpEngine::~IowarpEngine() {
  if (open_) {
    DoClose();
  }
}

/**
 * Initialize the engine
 */
void IowarpEngine::Init_() {
  if (open_) {
    throw std::runtime_error("IowarpEngine::Init_: Engine already initialized");
  }

  // Create or get tag for this ADIOS file/session
  // Use the engine name as the tag name
  try {
    current_tag_ = std::make_unique<wrp_cte::core::Tag>(m_Name);
    open_ = true;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::Init_: Failed to create/get tag: ") +
        e.what());
  }
}

/**
 * Begin a new step
 * @param mode Step mode (Read, Append, Update)
 * @param timeoutSeconds Timeout in seconds (-1 for no timeout)
 * @return Step status
 */
adios2::StepStatus IowarpEngine::BeginStep(adios2::StepMode mode,
                                           const float timeoutSeconds) {
  (void)mode;            // Suppress unused parameter warning
  (void)timeoutSeconds;  // Suppress unused parameter warning

  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  // Increment step counter
  IncrementCurrentStep();

  return adios2::StepStatus::OK;
}

/**
 * End the current step
 */
void IowarpEngine::EndStep() {
  if (!open_) {
    throw std::runtime_error("IowarpEngine::EndStep: Engine not initialized");
  }

  // Process all deferred put tasks from this step
  for (auto &deferred : deferred_tasks_) {
    // Set TASK_DATA_OWNER flag so task destructor will free the buffer
    auto *task_ptr = deferred.task.get();
    if (task_ptr != nullptr) {
      task_ptr->SetFlags(TASK_DATA_OWNER);
    }

    // Wait for task to complete
    deferred.task.Wait();
  }

  // Clear the deferred tasks vector for the next step
  deferred_tasks_.clear();
}

/**
 * Get current step number
 * @return Current step
 */
size_t IowarpEngine::CurrentStep() const { return current_step_; }

/**
 * Close the engine
 * @param transportIndex Transport index to close (-1 for all)
 */
void IowarpEngine::DoClose(const int transportIndex) {
  (void)transportIndex;  // Suppress unused parameter warning

  if (!open_) {
    return;
  }

  // Clean up resources
  current_tag_.reset();
  open_ = false;
}

/**
 * Initialize compression from COMPRESS_TYPE environment variable
 * Supported types:
 *   Lossless: BZIP2, LZO, ZSTD, LZ4, ZLIB, LZMA, BROTLI, SNAPPY, BLOSC2
 *   Lossy: ZFP, ZFP_<tol>, BITGROOMING, BITGROOMING_<nsd>, FPZIP, FPZIP_<prec>
 *   Examples: ZSTD, ZFP_1e-3, BITGROOMING_3, FPZIP_16
 */
void IowarpEngine::InitCompression() {
#if HSHM_ENABLE_COMPRESS
  const char *compress_type_env = std::getenv("COMPRESS_TYPE");
  if (compress_type_env == nullptr || std::strlen(compress_type_env) == 0) {
    compress_type_ = "none";
    compressor_ = nullptr;
    return;
  }

  std::string compress_type(compress_type_env);
  compress_type_ = compress_type;

  // Parse compression type and optional parameters
  std::string base_type = compress_type;
  std::string param_str;
  size_t underscore_pos = compress_type.find('_');
  if (underscore_pos != std::string::npos) {
    base_type = compress_type.substr(0, underscore_pos);
    param_str = compress_type.substr(underscore_pos + 1);
  }

  // Convert to uppercase for comparison
  for (auto &c : base_type) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }

  // Create the appropriate compressor
  if (base_type == "BZIP2") {
    compressor_ = std::make_unique<hshm::Bzip2>();
  } else if (base_type == "LZO") {
    lzo_init();  // LZO requires initialization
    compressor_ = std::make_unique<hshm::Lzo>();
  } else if (base_type == "ZSTD") {
    compressor_ = std::make_unique<hshm::Zstd>();
  } else if (base_type == "LZ4") {
    compressor_ = std::make_unique<hshm::Lz4>();
  } else if (base_type == "ZLIB") {
    compressor_ = std::make_unique<hshm::Zlib>();
  } else if (base_type == "LZMA") {
    compressor_ = std::make_unique<hshm::Lzma>();
  } else if (base_type == "BROTLI") {
    compressor_ = std::make_unique<hshm::Brotli>();
  } else if (base_type == "SNAPPY") {
    compressor_ = std::make_unique<hshm::Snappy>();
  } else if (base_type == "BLOSC2" || base_type == "BLOSC") {
    compressor_ = std::make_unique<hshm::Blosc>();
  } else if (base_type == "ZFP") {
    // ZFP lossy compressor - parse tolerance parameter
    double tolerance = 1e-3;  // Default tolerance
    if (!param_str.empty()) {
      try {
        tolerance = std::stod(param_str);
      } catch (...) {
        // Use default if parsing fails
      }
    }
    compressor_ = std::make_unique<hshm::Zfp>(tolerance);
  } else if (base_type == "BITGROOMING") {
    // BitGrooming lossy compressor - parse NSD parameter
    int nsd = 3;  // Default number of significant digits
    if (!param_str.empty()) {
      try {
        nsd = std::stoi(param_str);
      } catch (...) {
        // Use default if parsing fails
      }
    }
    compressor_ = std::make_unique<hshm::BitGrooming>(nsd);
  } else if (base_type == "FPZIP") {
    // FPZIP compressor - parse precision parameter (0 = lossless)
    int precision = 0;  // Default is lossless
    if (!param_str.empty()) {
      try {
        precision = std::stoi(param_str);
      } catch (...) {
        // Use default if parsing fails
      }
    }
    compressor_ = std::make_unique<hshm::Fpzip>(precision);
  } else {
    throw std::runtime_error(
        "IowarpEngine::InitCompression: Unknown compression type: " +
        compress_type);
  }
#else
  const char *compress_type_env = std::getenv("COMPRESS_TYPE");
  if (compress_type_env != nullptr && std::strlen(compress_type_env) > 0) {
    throw std::runtime_error(
        "IowarpEngine::InitCompression: Compression requested but "
        "HSHM_ENABLE_COMPRESS is not enabled at compile time");
  }
  compress_type_ = "none";
  compressor_ = nullptr;
#endif
}

/**
 * Compress data before Put operation
 * @param input Input data buffer
 * @param input_size Input data size in bytes
 * @param output Output buffer (allocated by caller, must be large enough)
 * @param output_size On input: max output buffer size; On output: actual
 *                    compressed size
 * @return true if compression successful or no compressor configured
 */
bool IowarpEngine::CompressData(const void *input, size_t input_size,
                                void *output, size_t &output_size) {
  if (!compressor_) {
    // No compression - just copy data
    if (output_size < input_size) {
      return false;
    }
    std::memcpy(output, input, input_size);
    output_size = input_size;
    return true;
  }

  return compressor_->Compress(output, output_size,
                               const_cast<void *>(input), input_size);
}

/**
 * Decompress data after Get operation
 * @param input Compressed data buffer
 * @param input_size Compressed data size in bytes
 * @param output Output buffer (allocated by caller)
 * @param output_size On input: expected decompressed size; On output: actual
 *                    size
 * @return true if decompression successful or no compressor configured
 */
bool IowarpEngine::DecompressData(const void *input, size_t input_size,
                                  void *output, size_t &output_size) {
  if (!compressor_) {
    // No compression - just copy data
    if (output_size < input_size) {
      return false;
    }
    std::memcpy(output, input, input_size);
    output_size = input_size;
    return true;
  }

  return compressor_->Decompress(output, output_size,
                                 const_cast<void *>(input), input_size);
}

/**
 * Put data synchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Data pointer
 */
template <typename T>
void IowarpEngine::DoPutSync_(const adios2::core::Variable<T> &variable,
                              const T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoPutSync_: No active tag");
  }

  // Calculate blob name from variable name and current step
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_);

  // Calculate data size
  size_t element_count = 1;
  for (size_t dim : variable.m_Shape) {
    element_count *= dim;
  }
  size_t data_size = element_count * sizeof(T);

  // Put blob to CTE synchronously
  try {
    if (compressor_) {
      // Compress data before storing
      // Format: [original_size (8 bytes)][compressed_data]
      size_t header_size = sizeof(size_t);
      size_t max_compressed_size = data_size * 2;  // Safety margin
      std::vector<char> compressed_buffer(header_size + max_compressed_size);

      // Store original size in header
      std::memcpy(compressed_buffer.data(), &data_size, sizeof(size_t));

      // Compress the data
      size_t compressed_size = max_compressed_size;
      bool compress_ok =
          CompressData(values, data_size, compressed_buffer.data() + header_size,
                       compressed_size);

      if (!compress_ok) {
        throw std::runtime_error("Compression failed");
      }

      // Store compressed data with header
      size_t total_size = header_size + compressed_size;
      current_tag_->PutBlob(blob_name, compressed_buffer.data(), total_size, 0);
    } else {
      // No compression - store data directly
      current_tag_->PutBlob(blob_name, reinterpret_cast<const char *>(values),
                            data_size, 0);
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::DoPutSync_: Failed to put blob: ") +
        e.what());
  }
}

/**
 * Put data asynchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Data pointer
 */
template <typename T>
void IowarpEngine::DoPutDeferred_(const adios2::core::Variable<T> &variable,
                                  const T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoPutDeferred_: No active tag");
  }

  // Calculate blob name from variable name and current step
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_);

  // Calculate data size
  size_t element_count = 1;
  for (size_t dim : variable.m_Shape) {
    element_count *= dim;
  }
  size_t data_size = element_count * sizeof(T);

  // Put blob asynchronously
  try {
    auto *ipc_manager = CHI_IPC;

    if (compressor_) {
      // Compress data before storing
      // Format: [original_size (8 bytes)][compressed_data]
      size_t header_size = sizeof(size_t);
      size_t max_compressed_size = data_size * 2;  // Safety margin
      std::vector<char> temp_buffer(max_compressed_size);

      // Compress the data into temp buffer
      size_t compressed_size = max_compressed_size;
      bool compress_ok =
          CompressData(values, data_size, temp_buffer.data(), compressed_size);

      if (!compress_ok) {
        throw std::runtime_error("Compression failed");
      }

      // Allocate shared memory for header + compressed data
      size_t total_size = header_size + compressed_size;
      auto buffer = ipc_manager->AllocateBuffer(total_size);

      // Store original size in header
      std::memcpy(buffer.ptr_, &data_size, sizeof(size_t));

      // Copy compressed data after header
      std::memcpy(static_cast<char *>(buffer.ptr_) + header_size,
                  temp_buffer.data(), compressed_size);

      auto task = current_tag_->AsyncPutBlob(
          blob_name, buffer.shm_.template Cast<void>(), total_size, 0, 1.0F);

      // Store task and buffer in deferred_tasks_ vector
      deferred_tasks_.push_back({std::move(task), std::move(buffer)});
    } else {
      // No compression - store data directly
      auto buffer = ipc_manager->AllocateBuffer(data_size);

      // Copy data to shared memory
      std::memcpy(buffer.ptr_, values, data_size);

      auto task = current_tag_->AsyncPutBlob(
          blob_name, buffer.shm_.template Cast<void>(), data_size, 0, 1.0F);

      // Store task and buffer in deferred_tasks_ vector
      // Buffer will be kept alive until EndStep processes the task
      deferred_tasks_.push_back({std::move(task), std::move(buffer)});
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::DoPutDeferred_: Failed to put blob: ") +
        e.what());
  }
}

/**
 * Get data synchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Output buffer
 */
template <typename T>
void IowarpEngine::DoGetSync_(const adios2::core::Variable<T> &variable,
                              T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoGetSync_: No active tag");
  }

  // Calculate blob name from variable name and current step
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_);

  // Calculate expected data size
  size_t element_count = 1;
  for (size_t dim : variable.m_Shape) {
    element_count *= dim;
  }
  size_t expected_size = element_count * sizeof(T);

  // Get blob from CTE synchronously
  try {
    if (compressor_) {
      // First, get the blob size to know how much to read
      // We need to read the header + compressed data
      // For simplicity, allocate a large buffer and read into it
      size_t header_size = sizeof(size_t);
      size_t max_compressed_size = expected_size * 2;  // Safety margin
      std::vector<char> compressed_buffer(header_size + max_compressed_size);

      // Read the compressed blob
      current_tag_->GetBlob(blob_name, compressed_buffer.data(),
                            compressed_buffer.size(), 0);

      // Extract original size from header
      size_t original_size;
      std::memcpy(&original_size, compressed_buffer.data(), sizeof(size_t));

      // Verify size matches expected
      if (original_size != expected_size) {
        throw std::runtime_error(
            "Decompressed size mismatch: expected " +
            std::to_string(expected_size) + " but got " +
            std::to_string(original_size));
      }

      // Decompress the data
      size_t decompressed_size = expected_size;
      bool decompress_ok =
          DecompressData(compressed_buffer.data() + header_size,
                         compressed_buffer.size() - header_size, values,
                         decompressed_size);

      if (!decompress_ok) {
        throw std::runtime_error("Decompression failed");
      }
    } else {
      // No compression - read data directly
      current_tag_->GetBlob(blob_name, reinterpret_cast<char *>(values),
                            expected_size, 0);
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("IowarpEngine::DoGetSync_: Failed to get blob: ") +
        e.what());
  }
}

/**
 * Get data asynchronously
 * @tparam T Data type
 * @param variable ADIOS2 variable
 * @param values Output buffer
 */
template <typename T>
void IowarpEngine::DoGetDeferred_(const adios2::core::Variable<T> &variable,
                                  T *values) {
  // Lazy initialization if not already initialized
  if (!open_) {
    Init_();
  }

  if (!current_tag_) {
    throw std::runtime_error("IowarpEngine::DoGetDeferred_: No active tag");
  }

  // For now, just call the sync version
  // In production, should use async API and track the Future
  DoGetSync_(variable, values);
}

}  // namespace coeus

/**
 * C wrapper to create engine
 * @param io ADIOS2 IO object
 * @param name Engine name
 * @param mode Open mode
 * @param comm MPI communicator
 * @return Engine pointer
 */
extern "C" {
coeus::IowarpEngine *EngineCreate(adios2::core::IO &io, const std::string &name,
                                  const adios2::Mode mode,
                                  adios2::helper::Comm comm) {
  return new coeus::IowarpEngine(io, name, mode, std::move(comm));
}

/**
 * C wrapper to destroy engine
 * @param obj Engine pointer to destroy
 */
void EngineDestroy(coeus::IowarpEngine *obj) { delete obj; }
}
