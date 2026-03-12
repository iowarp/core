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
};

class GpuApi {
 public:
  static void SetDevice(int gpu_id) {
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaSetDevice(gpu_id));
#elif HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipSetDevice(gpu_id));
#endif
  }

  static int GetDeviceCount() {
    int ngpu = 0;
#if HSHM_ENABLE_ROCM
    hipError_t err = hipGetDeviceCount(&ngpu);
    if (err != hipSuccess) {
      HLOG(kDebug, "hipGetDeviceCount failed (err={})", static_cast<int>(err));
      return 0;
    }
#endif
#if HSHM_ENABLE_CUDA
    cudaError_t err = cudaGetDeviceCount(&ngpu);
    if (err != cudaSuccess) {
      HLOG(kDebug, "cudaGetDeviceCount failed (err={}): {}",
           static_cast<int>(err), cudaGetErrorString(err));
      return 0;
    }
#endif
    return ngpu;
  }

  static void Synchronize() {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipDeviceSynchronize());
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
#endif
  }

  /** Synchronize a specific GPU stream instead of the whole device */
  static void Synchronize(void *stream) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
#endif
  }

  /** Create a non-blocking GPU stream */
  static void *CreateStream() {
    void *stream = nullptr;
#if HSHM_ENABLE_ROCM
    hipStream_t s;
    HIP_ERROR_CHECK(
        hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    stream = s;
#endif
#if HSHM_ENABLE_CUDA
    cudaStream_t s;
    CUDA_ERROR_CHECK(
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    stream = s;
#endif
    return stream;
  }

  /** Destroy a GPU stream */
  static void DestroyStream(void *stream) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipStreamDestroy(static_cast<hipStream_t>(stream)));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
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

  static void CloseIpcMemHandle(void *data) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipIpcCloseMemHandle(data));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaIpcCloseMemHandle(data));
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
    void *vptr;
    CUDA_ERROR_CHECK(cudaMalloc(&vptr, size));
    return static_cast<T *>(vptr);
#endif
  }

  template <typename T>
  static T *MallocManaged(size_t size) {
#if HSHM_ENABLE_ROCM
    T *ptr;
    HIP_ERROR_CHECK(hipMallocManaged(&ptr, size));
    return ptr;
#endif
#if HSHM_ENABLE_CUDA
    void *vptr;
    CUDA_ERROR_CHECK(cudaMallocManaged(&vptr, size));
    return static_cast<T *>(vptr);
#endif
    return nullptr;
  }

  template <typename T>
  static void RegisterHostMemory(T *ptr, size_t size) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(
        hipHostRegister((void *)ptr, size, hipHostRegisterPortable | hipHostRegisterMapped));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(
        cudaHostRegister((void *)ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped));
#endif
  }

  template <typename T>
  static void UnregisterHostMemory(T *ptr) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipHostUnregister((void *)ptr));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaHostUnregister((void *)ptr));
#endif
  }

  template <typename T>
  static void Memcpy(T *dst, const T *src, size_t size) {
#if HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipMemcpy(dst, src, size, hipMemcpyDefault));
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
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
#endif
#if HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaFree(ptr));
#endif
  }

#if HSHM_IS_GPU_COMPILER
  HSHM_GPU_FUN static size_t GetGlobalThreadId() {
    return threadIdx.x + blockIdx.x * blockDim.x +
           (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x) +
           (threadIdx.z + blockIdx.z * blockDim.z) *
               (blockDim.x * gridDim.x * blockDim.y * gridDim.y);
  }
#endif
};

}  // namespace hshm

#endif