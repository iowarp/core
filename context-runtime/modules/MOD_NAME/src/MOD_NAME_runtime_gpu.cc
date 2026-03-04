/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * See COPYING file in the top-level directory.
 */

/**
 * GPU companion library for MOD_NAME ChiMod.
 * Provides alloc_chimod_gpu / new_chimod_gpu entry points
 * for GPU container allocation.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include "chimaera/MOD_NAME/MOD_NAME_gpu_runtime.h"
#include "chimaera/ipc_manager.h"

CHI_TASK_GPU_CC(chimaera::MOD_NAME::GpuRuntime)

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
