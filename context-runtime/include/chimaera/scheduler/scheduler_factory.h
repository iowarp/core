// Copyright 2024 IOWarp contributors
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_SCHEDULER_FACTORY_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_SCHEDULER_FACTORY_H_

#include <memory>
#include <string>

#include "chimaera/scheduler/scheduler.h"

namespace chi {

/**
 * Factory class for creating scheduler instances based on name.
 */
class SchedulerFactory {
 public:
  /**
   * Get a scheduler instance based on the scheduler name.
   *
   * @param sched_name Name of the scheduler (e.g., "default")
   * @return Unique pointer to the scheduler instance, or nullptr if not found
   */
  static std::unique_ptr<Scheduler> Get(const std::string &sched_name);
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_SCHEDULER_FACTORY_H_
