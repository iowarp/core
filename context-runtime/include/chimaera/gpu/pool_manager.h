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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_GPU_POOL_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_GPU_POOL_MANAGER_H_

#include "chimaera/types.h"
#include "chimaera/gpu/container.h"

namespace chi {
namespace gpu {

/**
 * Entry in the GPU pool table
 * Stores a pool-to-container mapping
 */
struct GpuPoolInfo {
  PoolId pool_id_;
  Container *container_ = nullptr;
  bool occupied_ = false;
};

/**
 * GPU-side pool manager using a fixed-size open-addressed hash table
 *
 * No STL, no dynamic allocation - suitable for GPU __device__ memory.
 * Uses linear probing with 64 slots (sufficient for typical deployments).
 */
class PoolManager {
 public:
  static constexpr u32 kMaxPools = 64;

  HSHM_CROSS_FUN PoolManager() {
    for (u32 i = 0; i < kMaxPools; ++i) {
      pools_[i].occupied_ = false;
      pools_[i].container_ = nullptr;
    }
  }

  /**
   * Register a GPU container for a pool
   * @param pool_id Pool identifier
   * @param container Device pointer to GPU container
   */
  HSHM_CROSS_FUN void RegisterContainer(const PoolId &pool_id,
                                       Container *container) {
    u32 idx = Hash(pool_id);
    for (u32 i = 0; i < kMaxPools; ++i) {
      u32 slot = (idx + i) % kMaxPools;
      if (!pools_[slot].occupied_) {
        pools_[slot].pool_id_ = pool_id;
        pools_[slot].container_ = container;
        pools_[slot].occupied_ = true;
        return;
      }
      if (pools_[slot].pool_id_ == pool_id) {
        pools_[slot].container_ = container;
        return;
      }
    }
  }

  /**
   * Look up a GPU container by pool ID
   * @param pool_id Pool identifier
   * @return Device pointer to container, or nullptr if not found
   */
  HSHM_GPU_FUN Container *GetContainer(const PoolId &pool_id) {
    u32 idx = Hash(pool_id);
    for (u32 i = 0; i < kMaxPools; ++i) {
      u32 slot = (idx + i) % kMaxPools;
      if (!pools_[slot].occupied_) {
        return nullptr;
      }
      if (pools_[slot].pool_id_ == pool_id) {
        return pools_[slot].container_;
      }
    }
    return nullptr;
  }

 private:
  /**
   * Hash function for PoolId
   * @param pid Pool identifier
   * @return Hash index into pools_ array
   */
  HSHM_CROSS_FUN static u32 Hash(const PoolId &pid) {
    return (pid.major_ * 31 + pid.minor_) % kMaxPools;
  }

  GpuPoolInfo pools_[kMaxPools];
};

}  // namespace gpu
}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_GPU_POOL_MANAGER_H_
