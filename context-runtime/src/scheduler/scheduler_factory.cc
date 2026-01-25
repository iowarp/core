// Copyright 2024 IOWarp contributors
#include "chimaera/scheduler/scheduler_factory.h"

#include "chimaera/scheduler/default_sched.h"

namespace chi {

std::unique_ptr<Scheduler> SchedulerFactory::Get(const std::string &sched_name) {
  if (sched_name == "default") {
    return std::make_unique<DefaultScheduler>();
  }

  // If scheduler name not recognized, return default scheduler
  HLOG(kWarning, "Unknown scheduler name '{}', using default scheduler",
       sched_name);
  return std::make_unique<DefaultScheduler>();
}

}  // namespace chi
