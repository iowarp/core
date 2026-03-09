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
 * GPU implementation of CTE Core ChiMod methods.
 *
 * All methods are currently stubs on GPU. PutBlob and GetBlob demonstrate
 * the CHI_IPC->ToFullPtr pattern for converting ShmPtr blob data references
 * to GPU-accessible pointers. Full GPU implementations of the CTE data
 * placement logic will be added as GPU support matures.
 *
 * Note: core_tasks.h is included here (not in the header) to keep GPU
 * compilation isolated from CPU-only task constructors that use HSHM_MALLOC.
 * Task constructors are host-only and never called from device code.
 */

#include "wrp_cte/core/core_gpu_runtime.h"
#include "wrp_cte/core/core_tasks.h"

namespace wrp_cte::core {

HSHM_GPU_FUN void GpuRuntime::RegisterTarget(
    hipc::FullPtr<RegisterTargetTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::UnregisterTarget(
    hipc::FullPtr<UnregisterTargetTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::ListTargets(
    hipc::FullPtr<ListTargetsTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::StatTargets(
    hipc::FullPtr<StatTargetsTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParams>> task,
    chi::gpu::GpuRunContext& rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetTagSize(
    hipc::FullPtr<GetTagSizeTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::DelTag(
    hipc::FullPtr<DelTagTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetContainedBlobs(
    hipc::FullPtr<GetContainedBlobsTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

/**
 * GPU PutBlob stub: converts the blob_data_ ShmPtr to a GPU-accessible
 * FullPtr via CHI_IPC->ToFullPtr. This demonstrates the correct pattern
 * for accessing client-provided blob data in GPU runtime methods.
 */
HSHM_GPU_FUN void GpuRuntime::PutBlob(
    hipc::FullPtr<PutBlobTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)rctx;
  // Convert ShmPtr blob data reference to GPU-accessible pointer.
  // The blob_data_ ShmPtr is set by the CPU emplace constructor; ToFullPtr
  // resolves it to a direct UVA pointer accessible from this GPU thread.
  auto data_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  // Verify the pointer is valid by zeroing the buffer.
  memset(data_ptr.ptr_, 0, task->size_);
}

/**
 * GPU GetBlob stub: converts the blob_data_ ShmPtr output buffer to a
 * GPU-accessible FullPtr via CHI_IPC->ToFullPtr. This demonstrates the
 * correct pattern for writing blob data back to the client buffer.
 */
HSHM_GPU_FUN void GpuRuntime::GetBlob(
    hipc::FullPtr<GetBlobTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)rctx;
  // Convert ShmPtr output buffer reference to GPU-accessible pointer.
  // The blob_data_ ShmPtr is the client-provided output buffer; ToFullPtr
  // resolves it so GPU code can write blob data directly to the UVA region.
  auto data_ptr = CHI_IPC->ToFullPtr(task->blob_data_);
  (void)data_ptr;
}

HSHM_GPU_FUN void GpuRuntime::ReorganizeBlob(
    hipc::FullPtr<ReorganizeBlobTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::DelBlob(
    hipc::FullPtr<DelBlobTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetBlobScore(
    hipc::FullPtr<GetBlobScoreTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetBlobSize(
    hipc::FullPtr<GetBlobSizeTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetBlobInfo(
    hipc::FullPtr<GetBlobInfoTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::PollTelemetryLog(
    hipc::FullPtr<PollTelemetryLogTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::TagQuery(
    hipc::FullPtr<TagQueryTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::BlobQuery(
    hipc::FullPtr<BlobQueryTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::GetTargetInfo(
    hipc::FullPtr<GetTargetInfoTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::FlushMetadata(
    hipc::FullPtr<FlushMetadataTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

HSHM_GPU_FUN void GpuRuntime::FlushData(
    hipc::FullPtr<FlushDataTask> task, chi::gpu::GpuRunContext &rctx) {
  (void)task; (void)rctx;
}

}  // namespace wrp_cte::core
