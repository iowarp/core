#ifndef WRP_CAE_CORE_BINARY_FILE_ASSIMILATOR_H_
#define WRP_CAE_CORE_BINARY_FILE_ASSIMILATOR_H_

#include <wrp_cae/core/factory/base_assimilator.h>
#include <string>
#include <memory>

// Forward declaration
namespace wrp_cte::core {
class Client;
}  // namespace wrp_cte::core

namespace wrp_cae::core {

/**
 * BinaryFileAssimilator - Handles assimilation of binary files
 * Reads files in chunks and stores them in the CTE system
 */
class BinaryFileAssimilator : public BaseAssimilator {
 public:
  /**
   * Constructor with CTE client
   * @param cte_client Shared pointer to initialized CTE client
   */
  explicit BinaryFileAssimilator(std::shared_ptr<wrp_cte::core::Client> cte_client);

  /**
   * Schedule assimilation tasks for a binary file
   * This is a coroutine that uses co_await for async CTE operations.
   * @param ctx Assimilation context with source, destination, and metadata
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) override;

 private:
  /**
   * Extract protocol from URL (part before ::)
   * @param url URL in format protocol::path
   * @return Protocol string, or empty string if no protocol found
   */
  std::string GetUrlProtocol(const std::string& url);

  /**
   * Extract path from URL (part after ::)
   * @param url URL in format protocol::path
   * @return Path string, or empty string if no protocol found
   */
  std::string GetUrlPath(const std::string& url);

  /**
   * Get file size for the given file path
   * @param file_path Path to the file
   * @return File size in bytes, or 0 on error
   */
  size_t GetFileSize(const std::string& file_path);

  std::shared_ptr<wrp_cte::core::Client> cte_client_;
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_BINARY_FILE_ASSIMILATOR_H_
