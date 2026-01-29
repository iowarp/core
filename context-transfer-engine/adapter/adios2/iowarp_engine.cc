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
      compress_mode_(0),
      compress_lib_(0),
      compress_trace_(false) {
  // CTE client will be accessed via WRP_CTE_CLIENT singleton when needed
  wrp_cte::core::WRP_CTE_CLIENT_INIT("", chi::PoolQuery::Local());

  // Read compression environment variables
  ReadCompressionEnvVars();
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
 * Read compression environment variables
 */
void IowarpEngine::ReadCompressionEnvVars() {
  // Read COMPRESS_MODE: none (0), static (1), dynamic (2)
  const char* mode_env = std::getenv("COMPRESS_MODE");
  if (mode_env != nullptr) {
    std::string mode_str(mode_env);
    if (mode_str == "none") {
      compress_mode_ = 0;
    } else if (mode_str == "static") {
      compress_mode_ = 1;
    } else if (mode_str == "dynamic") {
      compress_mode_ = 2;
    }
  }

  // Read COMPRESS_LIB: library ID for static compression
  const char* lib_env = std::getenv("COMPRESS_LIB");
  if (lib_env != nullptr) {
    compress_lib_ = std::atoi(lib_env);
  }

  // Read COMPRESS_TRACE: on (true) or off (false)
  const char* trace_env = std::getenv("COMPRESS_TRACE");
  if (trace_env != nullptr) {
    std::string trace_str(trace_env);
    compress_trace_ = (trace_str == "on" || trace_str == "1" || trace_str == "true");
  }
}

/**
 * Create Context object for Put operations based on environment settings
 * @return Context object configured from environment variables
 */
wrp_cte::core::Context IowarpEngine::CreateCompressionContext() {
  wrp_cte::core::Context context;

  context.dynamic_compress_ = compress_mode_;
  context.compress_lib_ = compress_lib_;
  context.trace_ = compress_trace_;

  // Set other context fields as needed
  context.target_psnr_ = 0;          // No PSNR requirement by default
  context.psnr_chance_ = 100;        // Always validate PSNR if required
  context.max_performance_ = false;  // Optimize for ratio by default
  context.consumer_node_ = -1;       // Unknown consumer node
  context.data_type_ = 0;            // Unknown data type

  return context;
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

  // Calculate blob name from variable name, current step, and rank
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_) +
      "_rank_" + std::to_string(rank_);

  // Calculate data size using m_Count (local selection size), not m_Shape (global)
  size_t element_count = 1;
  if (!variable.m_Count.empty()) {
    for (size_t dim : variable.m_Count) {
      element_count *= dim;
    }
  } else if (!variable.m_Shape.empty()) {
    for (size_t dim : variable.m_Shape) {
      element_count *= dim;
    }
  }
  size_t data_size = element_count * sizeof(T);

  // Create compression context from environment variables
  auto context = CreateCompressionContext();

  // Put blob to CTE synchronously
  try {
    current_tag_->PutBlob(blob_name, reinterpret_cast<const char *>(values),
                          data_size, 0, 1.0f, context);
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

  // Calculate blob name from variable name, current step, and rank
  // Each rank writes its own portion with a unique blob name
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_) +
      "_rank_" + std::to_string(rank_);

  // Calculate data size using m_Count (local selection size), not m_Shape (global)
  // In MPI applications, each rank only has a portion of the global array
  size_t element_count = 1;
  if (!variable.m_Count.empty()) {
    // Use local count if available (for selections/MPI decomposition)
    for (size_t dim : variable.m_Count) {
      element_count *= dim;
    }
  } else if (!variable.m_Shape.empty()) {
    // Fall back to global shape if no selection
    for (size_t dim : variable.m_Shape) {
      element_count *= dim;
    }
  }
  size_t data_size = element_count * sizeof(T);

  // Put blob asynchronously
  try {
    auto *ipc_manager = CHI_IPC;
    if (ipc_manager == nullptr) {
      throw std::runtime_error("IowarpEngine::DoPutDeferred_: CHI_IPC is null");
    }

    // Allocate shared memory buffer and copy data
    auto buffer = ipc_manager->AllocateBuffer(data_size);
    if (buffer.ptr_ == nullptr) {
      throw std::runtime_error(
          "IowarpEngine::DoPutDeferred_: Failed to allocate buffer");
    }

    // Check if values pointer is valid
    if (values == nullptr) {
      throw std::runtime_error(
          "IowarpEngine::DoPutDeferred_: values pointer is null");
    }

    std::memcpy(buffer.ptr_, values, data_size);

    // Create compression context from environment variables
    auto context = CreateCompressionContext();

    auto task = current_tag_->AsyncPutBlob(
        blob_name, buffer.shm_.template Cast<void>(), data_size, 0, 1.0f, context);

    // Store task and buffer in deferred_tasks_ vector
    // Buffer will be kept alive until EndStep processes the task
    deferred_tasks_.emplace_back(DeferredTask{std::move(task), std::move(buffer)});
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

  // Calculate blob name from variable name, current step, and rank
  std::string blob_name =
      variable.m_Name + "_step_" + std::to_string(current_step_) +
      "_rank_" + std::to_string(rank_);

  // Calculate expected data size using m_Count (local selection size)
  size_t element_count = 1;
  if (!variable.m_Count.empty()) {
    for (size_t dim : variable.m_Count) {
      element_count *= dim;
    }
  } else if (!variable.m_Shape.empty()) {
    for (size_t dim : variable.m_Shape) {
      element_count *= dim;
    }
  }
  size_t expected_size = element_count * sizeof(T);

  // Get blob from CTE synchronously
  try {
    current_tag_->GetBlob(blob_name, reinterpret_cast<char *>(values),
                          expected_size, 0);
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
