#ifndef WRP_CAE_CORE_BASE_ASSIMILATOR_H_
#define WRP_CAE_CORE_BASE_ASSIMILATOR_H_

#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <chimaera/future.h>

namespace wrp_cae::core {

/**
 * BaseAssimilator - Abstract interface for data assimilators
 * Concrete implementations handle different data sources (file, URL, etc.)
 *
 * NOTE: Schedule is a coroutine that must be co_awaited from runtime code.
 * The error code is returned via output parameter since coroutines return TaskResume.
 */
class BaseAssimilator {
 public:
  virtual ~BaseAssimilator() = default;

  /**
   * Schedule assimilation tasks based on the provided context
   * This is a coroutine that uses co_await for async CTE operations.
   * @param ctx Assimilation context with source, destination, and metadata
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  virtual chi::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) = 0;
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_BASE_ASSIMILATOR_H_
