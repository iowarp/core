/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Coeus-adapter. The full Coeus-adapter copyright      *
 * notice, including terms governing use, modification, and redistribution,  *
 * is contained in the COPYING file, which can be found at the top directory.*
 * If you do not have access to the file, you may request a copy             *
 * from scslab@iit.edu.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

// Include ADIOS2 headers FIRST to avoid macro conflicts
#include <adios2.h>
#include <adios2/engine/plugin/PluginEngineInterface.h>

// Then include IOWarp headers
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

// Compression support
#include <hermes_shm/util/compress/compress.h>
#if HSHM_ENABLE_COMPRESS
#include <hermes_shm/util/compress/compress_factory.h>
#include <hermes_shm/util/compress/zfp.h>
#include <hermes_shm/util/compress/bitgrooming.h>
#include <hermes_shm/util/compress/fpzip.h>
#endif

namespace coeus {

class IowarpEngine : public adios2::plugin::PluginEngineInterface {
 public:
  /** Construct the IowarpEngine */
  IowarpEngine(adios2::core::IO &io,  // NOLINT
               const std::string &name, const adios2::Mode mode,
               adios2::helper::Comm comm);

  /** Destructor */
  ~IowarpEngine() override;

  /**
   * Define the beginning of a step. A step is typically the offset from
   * the beginning of a file. It is measured as a size_t.
   *
   * Logically, a "step" represents a snapshot of the data at a specific time,
   * and can be thought of as a frame in a video or a snapshot of a simulation.
   * */
  adios2::StepStatus BeginStep(adios2::StepMode mode,
                               const float timeoutSeconds = -1.0) override;

  /** Define the end of a step */
  void EndStep() override;

  /**
   * Returns the current step
   * */
  size_t CurrentStep() const override;

 protected:
  /** Initialize parameters */
  void InitParameters() override {}

  /** Initialize transports */
  void InitTransports() override {}

  /** Initialize (wrapper around Init_)*/
  void Init() override { Init_(); }

  /** Actual engine initialization */
  void Init_();

  /** Close a particular transport */
  void DoClose(const int transportIndex = -1) override;

  /** Place data in CTE */
  template <typename T>
  void DoPutSync_(const adios2::core::Variable<T> &variable, const T *values);

  /** Place data in CTE asynchronously */
  template <typename T>
  void DoPutDeferred_(const adios2::core::Variable<T> &variable,
                      const T *values);

  /** Get data from CTE (sync) */
  template <typename T>
  void DoGetSync_(const adios2::core::Variable<T> &variable, T *values);

  /** Get data from CTE (async) */
  template <typename T>
  void DoGetDeferred_(const adios2::core::Variable<T> &variable, T *values);

#define declare_type(T)                                                     \
  void DoPutSync(adios2::core::Variable<T> &variable, const T *values)      \
      override {                                                            \
    DoPutSync_(variable, values);                                           \
  }                                                                         \
  void DoPutDeferred(adios2::core::Variable<T> &variable, const T *values)  \
      override {                                                            \
    DoPutDeferred_(variable, values);                                       \
  }                                                                         \
  void DoGetSync(adios2::core::Variable<T> &variable, T *values) override { \
    DoGetSync_(variable, values);                                           \
  }                                                                         \
  void DoGetDeferred(adios2::core::Variable<T> &variable, T *values)        \
      override {                                                            \
    DoGetDeferred_(variable, values);                                       \
  }
  ADIOS2_FOREACH_STDTYPE_1ARG(declare_type)
#undef declare_type

 private:
  /** Structure to hold deferred task and its buffer */
  struct DeferredTask {
    chi::Future<wrp_cte::core::PutBlobTask> task;
    hipc::FullPtr<char> buffer;
  };

  /** CTE Tag for this ADIOS file/session */
  std::unique_ptr<wrp_cte::core::Tag> current_tag_;

  /** Current step counter */
  size_t current_step_;

  /** Process rank */
  int rank_;

  /** Engine open status */
  bool open_;

  /** Vector of deferred put tasks for current step */
  std::vector<DeferredTask> deferred_tasks_;

  /** Compressor instance (nullptr if compression disabled) */
  std::unique_ptr<hshm::Compressor> compressor_;

  /** Compression type name for logging */
  std::string compress_type_;

  /** Increment the current step */
  void IncrementCurrentStep() { current_step_++; }

  /** Initialize compression from COMPRESS_TYPE environment variable */
  void InitCompression();

  /**
   * Compress data before Put operation
   * @param input Input data buffer
   * @param input_size Input data size in bytes
   * @param output Output buffer (allocated by caller, must be large enough)
   * @param output_size On input: max output buffer size; On output: actual compressed size
   * @return true if compression successful or no compressor configured
   */
  bool CompressData(const void *input, size_t input_size, void *output,
                    size_t &output_size);

  /**
   * Decompress data after Get operation
   * @param input Compressed data buffer
   * @param input_size Compressed data size in bytes
   * @param output Output buffer (allocated by caller)
   * @param output_size On input: expected decompressed size; On output: actual size
   * @return true if decompression successful or no compressor configured
   */
  bool DecompressData(const void *input, size_t input_size, void *output,
                      size_t &output_size);
};

}  // namespace coeus

/**
 * This is how ADIOS figures out where to dynamically load the engine.
 * */
extern "C" {

/** C wrapper to create engine */
coeus::IowarpEngine *EngineCreate(adios2::core::IO &io,  // NOLINT
                                  const std::string &name,
                                  const adios2::Mode mode,
                                  adios2::helper::Comm comm);

/** C wrapper to destroy engine */
void EngineDestroy(coeus::IowarpEngine *obj);
}