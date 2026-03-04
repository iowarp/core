/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

#ifndef CHIMAERA_MOD_NAME_GPU_RUNTIME_H_
#define CHIMAERA_MOD_NAME_GPU_RUNTIME_H_

#include "chimaera/gpu_container.h"
#include "chimaera/MOD_NAME/MOD_NAME_tasks.h"
#include "chimaera/MOD_NAME/autogen/MOD_NAME_methods.h"

namespace chimaera::MOD_NAME {

/**
 * GPU-side container for the MOD_NAME module.
 * Processes GpuSubmitTask on the GPU megakernel.
 */
class GpuRuntime : public chi::gpu::Container {
 public:
  HSHM_GPU_FUN GpuRuntime() = default;
  HSHM_GPU_FUN ~GpuRuntime() = default;

  /** Execute a task method on the GPU. */
  HSHM_GPU_FUN void RunImpl(chi::u32 method,
                              hipc::FullPtr<chi::Task> task_ptr,
                              chi::gpu::GpuRunContext &rctx) {
    (void)rctx;
    if (method == Method::kGpuSubmit) {
      auto task = task_ptr.template Cast<GpuSubmitTask>();
      task->result_value_ = (task->test_value_ * 2) + task->gpu_id_;
    }
  }

  /** Allocate and deserialize a task from a local archive. */
  HSHM_GPU_FUN hipc::FullPtr<chi::Task> AllocLoadImpl(
      chi::u32 method, chi::LocalLoadTaskArchive &archive) {
    if (method == Method::kGpuSubmit) {
      auto *alloc = gpu_alloc_;
      auto task = alloc->template AllocateObjs<GpuSubmitTask>(1);
      if (task.IsNull()) {
        return hipc::FullPtr<chi::Task>::GetNull();
      }
      new (task.ptr_) GpuSubmitTask();
      archive.SetMsgType(chi::LocalMsgType::kSerializeIn);
      task.ptr_->SerializeIn(archive);
      return task.template Cast<chi::Task>();
    }
    return hipc::FullPtr<chi::Task>::GetNull();
  }

  /** Serialize task output into a local archive. */
  HSHM_GPU_FUN void SaveImpl(
      chi::u32 method, chi::LocalSaveTaskArchive &archive,
      const hipc::FullPtr<chi::Task> &task) {
    if (method == Method::kGpuSubmit) {
      auto gpu_task = task.template Cast<GpuSubmitTask>();
      gpu_task->SerializeOut(archive);
    }
  }
};

}  // namespace chimaera::MOD_NAME

#endif  // CHIMAERA_MOD_NAME_GPU_RUNTIME_H_
