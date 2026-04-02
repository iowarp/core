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

#ifndef HSHM_UTIL_GPU_API_H
#define HSHM_UTIL_GPU_API_H

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/util/logging.h"

namespace hshm {

struct GpuIpcMemHandle {
#if HSHM_ENABLE_CUDA
  cudaIpcMemHandle_t cuda_;
#endif
#if HSHM_ENABLE_ROCM
  hipIpcMemHandle_t rocm_;
#endif
#if HSHM_ENABLE_SYCL
  void *sycl_ptr_;  // SYCL USM pointers are directly shareable; store base ptr
#endif
};

class GpuApi {
 public:
  static void SetDevice(int gpu_id) {
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaSetDevice(gpu_id));
#elif HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipSetDevice(gpu_id));
#elif HSHM_ENABLE_SYCL
    // SYCL device selection is done via queue construction; no global set
    (void)gpu_id;
#endif
  }

  static int GetDeviceCount() {
    int ngpu = 0;
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipGetDeviceCount(&ngpu));
#elif HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaGetDeviceCount(&ngpu));
#elif HSHM_ENABLE_SYCL
    auto platforms = sycl::platform::get_platforms();
    for (auto &p : platforms) {
      auto devs = p.get_devices(sycl::info::device_type::gpu);
      ngpu += static_cast<int>(devs.size());
    }
#endif
    return ngpu;
  }

  static void Synchronize() {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipDeviceSynchronize());
#elif HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
#elif HSHM_ENABLE_SYCL
    SyclQueue().wait_and_throw();
#endif
  }

  static void GetIpcMemHandle(GpuIpcMemHandle &ipc, void *data) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipIpcGetMemHandle(&ipc.rocm_, (void *)data));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaIpcGetMemHandle(&ipc.cuda_, (void *)data));
#endif
  }

  template <typename T>
  static void OpenIpcMemHandle(GpuIpcMemHandle &ipc, T **data) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipIpcOpenMemHandle((void **)data, ipc.rocm_,
                                        hipIpcMemLazyEnablePeerAccess));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaIpcOpenMemHandle((void **)data, ipc.cuda_,
                                          cudaIpcMemLazyEnablePeerAccess));
#endif
  }

  template <typename T>
  static T *Malloc(size_t size) {
#if HSHM_ENABLE_ROCM
    T *ptr;
    HIP_ERROR_CHECK(hipMalloc(&ptr, size));
    return ptr;
#endif
#if HSHM_ENABLE_CUDA
    T *ptr;
    CUDA_ERROR_CHECK(cudaMalloc(&ptr, size));
    return ptr;
#endif
  }

  template <typename T>
  static T *MallocManaged(size_t size) {
#if HSHM_ENABLE_ROCM
    T *ptr;
    HIP_ERROR_CHECK(hipMallocManaged(&ptr, size));
    return ptr;
#elif HSHM_ENABLE_CUDA
    T *ptr;
    CUDA_ERROR_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
#elif HSHM_ENABLE_SYCL
    return static_cast<T *>(sycl::malloc_shared(size, SyclQueue()));
#endif
    return nullptr;
  }

  template <typename T>
  static void RegisterHostMemory(T *ptr, size_t size) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(
        hipHostRegister((void *)ptr, size, hipHostRegisterPortable));
#elif HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaHostRegister((void *)ptr, size, cudaHostRegisterPortable));
#elif HSHM_ENABLE_SYCL
    // SYCL USM host memory doesn't require explicit registration;
    // use sycl::malloc_host for GPU-accessible host allocations when needed.
    (void)ptr; (void)size;
#endif
  }

  template <typename T>
  static void UnregisterHostMemory(T *ptr) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipHostUnregister((void *)ptr));
#elif HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaHostUnregister((void *)ptr));
#elif HSHM_ENABLE_SYCL
    (void)ptr;
#endif
  }

  template <typename T>
  static void Memcpy(T *dst, T *src, size_t size) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipMemcpy(dst, src, size, hipMemcpyDefault));
#elif HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
#elif HSHM_ENABLE_SYCL
    SyclQueue().memcpy(dst, src, size).wait_and_throw();
#endif
  }

  template <typename T>
  static bool IsDevicePointer(T *ptr) {
#if HSHM_ENABLE_ROCM
    hipPointerAttribute_t attributes;
    HIP_ERROR_CHECK(hipPointerGetAttributes(&attributes, (void *)ptr));
    return attributes.type == hipMemoryTypeDevice;
#endif
#if HSHM_ENABLE_CUDA
    cudaPointerAttributes attributes;
    CUDA_ERROR_CHECK(cudaPointerGetAttributes(&attributes, (void *)ptr));
    return attributes.type == cudaMemoryTypeDevice;
#endif
    return false;
  }

  template <typename T>
  static void Memset(T *dst, int value, size_t size) {
    if (IsDevicePointer(dst)) {
#if HSHM_ENABLE_ROCM
      HIP_ERROR_CHECK(hipMemset(dst, value, size));
#endif
#if HSHM_ENABLE_CUDA
      CUDA_ERROR_CHECK(cudaMemset(dst, value, size));
#endif
    } else {
      memset(dst, value, size);
    }
  }

  template <typename T>
  static void Free(T *ptr) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipFree(ptr));
#elif HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaFree(ptr));
#elif HSHM_ENABLE_SYCL
    sycl::free(ptr, SyclQueue());
#endif
  }

#if HSHM_ENABLE_CUDA_OR_ROCM
  HSHM_GPU_FUN static size_t GetGlobalThreadId() {
    return threadIdx.x + blockIdx.x * blockDim.x +
           (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x) +
           (threadIdx.z + blockIdx.z * blockDim.z) *
               (blockDim.x * gridDim.x * blockDim.y * gridDim.y);
  }
#endif

#if HSHM_ENABLE_SYCL
  static sycl::queue &SyclQueue() {
    static sycl::queue q{sycl::gpu_selector_v};
    return q;
  }
#endif
};

}  // namespace hshm

#endif