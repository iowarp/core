/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

#ifndef CHIMAERA_ADMIN_GPU_RUNTIME_H_
#define CHIMAERA_ADMIN_GPU_RUNTIME_H_

#include "chimaera/gpu_container.h"

namespace chimaera::admin {

/**
 * GPU-side container for Admin ChiMod.
 * All admin methods are no-ops on GPU.
 */
class GpuRuntime : public chi::gpu::Container {
 public:
  HSHM_GPU_FUN GpuRuntime() = default;
  HSHM_GPU_FUN ~GpuRuntime() = default;

  HSHM_GPU_FUN void RunImpl(chi::u32 method,
                              hipc::FullPtr<chi::Task> task_ptr,
                              chi::gpu::GpuRunContext &rctx) {
    (void)method;
    (void)task_ptr;
    (void)rctx;
  }

  HSHM_GPU_FUN hipc::FullPtr<chi::Task> AllocLoadImpl(
      chi::u32 method, chi::LocalLoadTaskArchive &archive) {
    (void)method;
    (void)archive;
    return hipc::FullPtr<chi::Task>::GetNull();
  }

  HSHM_GPU_FUN void SaveImpl(
      chi::u32 method, chi::LocalSaveTaskArchive &archive,
      const hipc::FullPtr<chi::Task> &task) {
    (void)method;
    (void)archive;
    (void)task;
  }
};

}  // namespace chimaera::admin

#endif  // CHIMAERA_ADMIN_GPU_RUNTIME_H_
