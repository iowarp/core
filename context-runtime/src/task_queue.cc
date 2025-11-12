/**
 * TaskQueue implementation - simple wrapper around hipc::multi_mpsc_queue
 */

#include "chimaera/task_queue.h"

namespace chi {

TaskQueue::TaskQueue(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T>& alloc,
                     u32 num_lanes, u32 num_prios, u32 depth_per_lane)
    : queue_(alloc, num_lanes, num_prios, depth_per_lane) {
  // Headers are now managed automatically by the multi_mpsc_queue per lane
}


/*static*/ bool TaskQueue::EmplaceTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::TypedPointer<Task> task_ptr) {
  if (lane_ptr.IsNull() || task_ptr.IsNull()) {
    return false;
  }
  
  // Push to the lane
  lane_ptr->push(task_ptr);

  return true;
}

/*static*/ bool TaskQueue::PopTask(hipc::FullPtr<TaskLane>& lane_ptr, hipc::TypedPointer<Task>& task_ptr) {
  if (lane_ptr.IsNull()) {
    return false;
  }
  
  auto token = lane_ptr->pop(task_ptr);
  return !token.IsNull();
}

}  // namespace chi