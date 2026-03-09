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

/**
 * GPU implementation of MOD_NAME ChiMod methods.
 *
 * GpuSubmit computes a simple polynomial on the GPU to verify end-to-end
 * GPU task submission and execution.
 */

#include "chimaera/MOD_NAME/MOD_NAME_gpu_runtime.h"

namespace chimaera::MOD_NAME {

/**
 * Execute a GpuSubmit task on the GPU.
 * Computes result_value_ = (test_value_ * 3) + gpu_id_ to verify
 * that the correct GPU and input values were received.
 * @param task The GPU submit task.
 * @param rctx GPU run context (unused).
 */
HSHM_GPU_FUN void GpuRuntime::GpuSubmit(hipc::FullPtr<GpuSubmitTask> task,
                                          chi::gpu::GpuRunContext &rctx) {
  (void)rctx;
#if HSHM_IS_GPU
  printf("[GpuSubmit] blk=%d thr=%d test_value_=%u gpu_id_=%u\n",
         (int)blockIdx.x, (int)threadIdx.x,
         (unsigned)task->test_value_, (unsigned)task->gpu_id_);
#endif
  task->result_value_ = (task->test_value_ * 3) + task->gpu_id_;
}

}  // namespace chimaera::MOD_NAME
