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
 * GPU kernels for the CTE GPU benchmark.
 * Compiled via add_cuda_library (clang-cuda dual-pass).
 *
 * Contains __global__ kernels and extern "C" launcher wrappers that the
 * host-side benchmark driver (wrp_cte_gpu_bench.cc) calls.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <hermes_shm/util/gpu_api.h>
#include <chimaera/ipc_manager.h>

//==============================================================================
// GPU Kernels
//==============================================================================

/**
 * Kernel 1: Initialize a BuddyAllocator over device memory and allocate
 * a contiguous array of `total_bytes` bytes.  Returns the FullPtr via
 * pinned host memory so the CPU can read it.
 */
__global__ void gpu_putblob_alloc_kernel(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) {
    d_out_ptr->SetNull();
    return;
  }
  auto result = alloc->AllocateObjs<char>(total_bytes);
  *d_out_ptr = result;
}

/**
 * Kernel 2: Each warp memsets its slice of A to a constant, then calls
 * AsyncPutBlob to store that slice as a blob via the CTE runtime.
 * Only the warp scheduler (lane 0) submits the PutBlob task.
 */
__global__ void gpu_putblob_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> array_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 total_bytes,
    chi::u32 total_warps,
    bool to_cpu,
    int *d_done) {
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id < total_warps) {
    // Compute this warp's slice of the array
    chi::u64 slice_size = total_bytes / total_warps;
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * slice_size;
    char *my_data = array_ptr.ptr_ + my_offset;

    // All lanes participate in memset
    for (chi::u64 i = lane_id; i < slice_size; i += 32) {
      my_data[i] = static_cast<char>(warp_id & 0xFF);
    }
    __syncwarp();

    // Only lane 0 submits PutBlob
    if (chi::IpcManager::IsWarpScheduler()) {
      wrp_cte::core::Client cte_client(cte_pool_id);

      // Build ShmPtr referencing the data allocator backend
      hipc::ShmPtr<> blob_shm;
      blob_shm.alloc_id_ = data_alloc_id;
      size_t base_off = array_ptr.shm_.off_.load();
      blob_shm.off_.exchange(base_off + my_offset);

      // Build blob name: "w_<id>"
      char name_buf[32];
      int pos = 0;
      name_buf[pos++] = 'w';
      name_buf[pos++] = '_';
      chi::u32 wid = warp_id;
      char digits[10];
      int nd = 0;
      do { digits[nd++] = '0' + (wid % 10); wid /= 10; } while (wid > 0);
      for (int d = nd - 1; d >= 0; --d) name_buf[pos++] = digits[d];
      name_buf[pos] = '\0';

      auto *ipc = CHI_IPC;
      auto task = ipc->NewTask<wrp_cte::core::PutBlobTask>(
          chi::CreateTaskId(), cte_pool_id,
          to_cpu ? chi::PoolQuery::ToLocalCpu() : chi::PoolQuery::Local(),
          tag_id, name_buf,
          (chi::u64)0, slice_size,
          blob_shm, -1.0f,
          wrp_cte::core::Context(), (chi::u32)0);
      auto future = ipc->Send(task);
      future.Wait();
    }
  }

  __syncwarp();
  if (chi::IpcManager::IsWarpScheduler()) {
    __threadfence();
    int prev = atomicAdd(d_done, 1);
    if (prev == static_cast<int>(total_warps) - 1) {
      __threadfence_system();
    }
  }
}

/**
 * Kernel 3: Direct copy baseline — all threads cooperatively copy from
 * device memory to pinned host memory using 4-byte coalesced stores.
 * 32 threads x 4 bytes = 128 bytes per iteration = one PCIe cache line.
 */
__global__ void gpu_direct_memcpy_kernel(
    const char *d_src,
    char *h_dst,
    chi::u64 total_bytes,
    chi::u32 total_threads_used,
    int *d_done) {
  chi::u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  chi::u32 stride = blockDim.x * gridDim.x;

  // Coalesced 4-byte stores across all threads
  chi::u64 n_words = total_bytes / 4;
  const unsigned int *src4 = reinterpret_cast<const unsigned int *>(d_src);
  unsigned int *dst4 = reinterpret_cast<unsigned int *>(h_dst);
  for (chi::u64 i = tid; i < n_words; i += stride) {
    dst4[i] = src4[i];
  }

  // Tail bytes
  if (tid == 0) {
    chi::u64 tail = n_words * 4;
    for (chi::u64 i = tail; i < total_bytes; ++i) {
      h_dst[i] = d_src[i];
    }
  }

  __threadfence_system();
  atomicAdd(d_done, 1);
}

/**
 * Kernel 4: Write to managed memory — same warp structure as PutBlob.
 * All lanes cooperatively memset their slice. Pages reside in VRAM,
 * so writes happen at full VRAM bandwidth (~256 GB/s).
 */
__global__ void gpu_managed_write_kernel(
    char *managed_buf,
    chi::u64 total_bytes,
    chi::u32 total_warps,
    int *d_done) {
  chi::u32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  chi::u32 lane_id = threadIdx.x % 32;

  if (warp_id < total_warps) {
    chi::u64 slice_size = total_bytes / total_warps;
    chi::u64 my_offset = static_cast<chi::u64>(warp_id) * slice_size;
    char *my_data = managed_buf + my_offset;

    // All lanes participate in write (same as PutBlob kernel memset)
    for (chi::u64 i = lane_id; i < slice_size; i += 32) {
      my_data[i] = static_cast<char>(warp_id & 0xFF);
    }
    __syncwarp();

    if (lane_id == 0) {
      __threadfence();
      atomicAdd(d_done, 1);
    }
  }
}

//==============================================================================
// extern "C" launcher wrappers (called from wrp_cte_gpu_bench.cc)
//==============================================================================

extern "C" void launch_gpu_putblob_alloc(
    hipc::MemoryBackend data_backend,
    chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  gpu_putblob_alloc_kernel<<<1, 1>>>(data_backend, total_bytes, d_out_ptr);
}

extern "C" void launch_gpu_putblob(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> array_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 total_bytes,
    chi::u32 total_warps,
    bool to_cpu,
    int *d_done,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    void *stream) {
  gpu_putblob_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, num_blocks,
      array_ptr, data_alloc_id,
      total_bytes, total_warps, to_cpu, d_done);
}

extern "C" void launch_gpu_direct_memcpy(
    const char *d_src,
    char *h_dst,
    chi::u64 total_bytes,
    chi::u32 total_threads_used,
    int *d_done,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    void *stream) {
  gpu_direct_memcpy_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      d_src, h_dst, total_bytes, total_threads_used, d_done);
}

extern "C" void launch_gpu_managed_write(
    char *managed_buf,
    chi::u64 total_bytes,
    chi::u32 total_warps,
    int *d_done,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    void *stream) {
  gpu_managed_write_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      managed_buf, total_bytes, total_warps, d_done);
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
