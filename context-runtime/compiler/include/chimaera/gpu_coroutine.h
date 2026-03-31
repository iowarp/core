/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_
#define CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_

/**
 * @file gpu_coroutine.h
 * @brief GPU-side RunContext for CDP-based task dispatch.
 *
 * Provides chi::gpu::RunContext, a lightweight execution context passed
 * to GPU container methods.  Task parallelism is achieved via CUDA
 * Dynamic Parallelism (CDP) child kernel launches rather than coroutines.
 */

#include <cstddef>
#include <cstdint>

#include "chimaera/task.h"

namespace chi {
namespace gpu {

/** Unsigned 32-bit integer (self-contained, no dependency on hshm types) */
using u32 = uint32_t;

// Forward declarations
class Container;

static constexpr u32 kWarpSize = 32;

// ============================================================================
// RunContext -- execution context for CDP-dispatched task methods (GPU-side)
// ============================================================================

/**
 * GPU-side execution context for task methods dispatched via CDP.
 *
 * One RunContext is created per child kernel launch, typically as a
 * stack-local variable.  Contains the dispatch metadata needed by
 * container methods: which container, which method, and the task pointer.
 *
 * Layout is GPU-friendly: no STL, no virtual functions, trivially copyable.
 */
struct RunContext {
  /** GPU container that owns the method being executed */
  Container *container_;

  /** Method ID to dispatch */
  u32 method_id_;

  /** Full pointer to the task being executed */
  hipc::FullPtr<chi::Task> task_ptr_;

  /** FutureShm associated with this task (for completion signaling) */
  chi::FutureShm *task_fshm_;

  /** Total thread parallelism for this task (gridDim.x * blockDim.x) */
  u32 parallelism_;

  /** Default constructor — intentionally does NOT zero fields.
   *  Callers set all fields explicitly after construction. */
  __host__ __device__ RunContext() = default;
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_COMPILER_INCLUDE_CHIMAERA_GPU_COROUTINE_H_
