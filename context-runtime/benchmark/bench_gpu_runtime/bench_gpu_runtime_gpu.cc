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
 * GPU Runtime Benchmark — GPU kernel and CUDA wrapper
 *
 * Contains only the GPU kernel and a thin kernel-launch wrapper.
 * All IpcManager access is in bench_gpu_runtime.cc (g++-compiled) to avoid
 * ODR layout mismatches between nvcc and g++ views of IpcManager.
 *
 * Only thread 0 of each block submits tasks (matches the single-thread
 * pattern proven in unit tests). Parallelism comes from client_blocks:
 * each block independently submits total_tasks sequential tasks. Block 0
 * thread 0 writes d_done after its tasks complete to signal the CPU.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/ipc_manager.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/task.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/MOD_NAME/autogen/MOD_NAME_methods.h>
#include <chimaera/local_task_archives.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/gpu_api.h>
#include <hermes_shm/memory/backend/gpu_shm_mmap.h>
#include <hermes_shm/memory/backend/gpu_malloc.h>
#include <hermes_shm/data_structures/priv/string.h>
#include <hermes_shm/data_structures/priv/array.h>
#include <hermes_shm/memory/allocator/arena_allocator.h>
#include <hermes_shm/memory/allocator/buddy_allocator.h>
#include <hermes_shm/memory/allocator/slab_allocator.h>

#include <chrono>
#include <thread>

namespace chi_bench {

/**
 * GPU client benchmark kernel.
 *
 * Each block's thread 0 initializes its IpcManager (via
 * CHIMAERA_GPU_CLIENT_INIT for per-block backend partitioning), then
 * submits total_tasks tasks sequentially via AsyncGpuSubmit + Wait(). Other
 * threads in the block are idle. Block 0 thread 0 writes d_done after its
 * loop completes.
 *
 * @param gpu_info   IpcManagerGpuInfo with backend and GPU→GPU queue
 * @param pool_id    Pool ID of the MOD_NAME container
 * @param num_blocks Number of blocks in the grid (for per-block backend slice)
 * @param total_tasks Total tasks for thread 0 of each block to submit
 * @param d_done     Pinned host flag set to 1 by block 0, thread 0 on finish
 */
__global__ void gpu_bench_client_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done,
    chi::u32 total_warps) {
  // Partition backend per block; initialize block-local IpcManager
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  // Only lane 0 of each warp submits tasks (warp-level dispatch model).
  // The allocator is partitioned per-warp, so only one thread per warp
  // should allocate to avoid contention.
  if (chi::IpcManager::IsWarpScheduler()) {
    chimaera::MOD_NAME::Client client(pool_id);

    for (chi::u32 i = 0; i < total_tasks; ++i) {
      auto future = client.AsyncGpuSubmit(chi::PoolQuery::Local(), 0, i);
      future.Wait();
    }
  }

  // All warps signal completion via lane 0
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
 * GPU client benchmark kernel for coroutine subtask throughput.
 *
 * Same structure as gpu_bench_client_kernel but dispatches SubtaskTest
 * instead of GpuSubmit. SubtaskTest's GPU Run() co_awaits GpuSubmit
 * via SendGpuDirect (no serialization), testing the full coroutine
 * suspend/resume path inside the GPU worker.
 */
__global__ void gpu_bench_coroutine_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    chi::u32 subtasks,
    int *d_done,
    chi::u32 total_warps) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  if (chi::IpcManager::IsWarpScheduler()) {
    chimaera::MOD_NAME::Client client(pool_id);

    for (chi::u32 i = 0; i < total_tasks; ++i) {
      auto future = client.AsyncSubtaskTest(chi::PoolQuery::Local(), i, subtasks);
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
 * GPU alloc/free benchmark kernel.
 *
 * Each thread does total_tasks cycles of NewTask<PutBlobTask> + DelTask
 * through IpcManager, measuring pure PartitionedAllocator throughput with no
 * runtime dispatch. No orchestrator needed.
 */
__global__ void gpu_bench_alloc_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done,
    chi::u32 total_threads) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  auto *ipc = CHI_IPC;
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto task = ipc->NewTask<wrp_cte::core::PutBlobTask>();
    task->pool_id_ = pool_id;
    task->size_ = 4096;
    ipc->DelTask(task);
  }

  __threadfence();
  int prev = atomicAdd(d_done, 1);
  if (prev == static_cast<int>(total_threads) - 1) {
    __threadfence_system();
  }
}

/**
 * GPU alloc+free benchmark kernel for PutBlobTask + LocalSaveTaskArchive.
 *
 * Each warp scheduler does total_tasks cycles of:
 *   1. NewTask<PutBlobTask> — allocate task
 *   2. NewObj<LocalSaveTaskArchive> — allocate save archive
 *   3. DelObj(ar_save) + DelTask(task) — free both
 *
 * No serialization — purely measures allocation/deallocation cost of the
 * two objects that SendToGpu's copy path must create.
 *
 */
__global__ void gpu_bench_alloc_serde_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done,
    chi::u32 total_threads) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  auto *ipc = CHI_IPC;

  // Only warp scheduler runs — BuddyAllocator is not thread-safe
  if (chi::IpcManager::IsWarpScheduler()) {
    long long t_push_arena = 0, t_pop_arena = 0;
    long long t_alloc_task = 0, t_ctor_save = 0, t_ctor_load = 0, t_alloc_task2 = 0;
    long long t_free_task2 = 0, t_dtor_load = 0, t_dtor_save = 0, t_free_task = 0;

    // Push arenas once (same as serde kernel)
    long long tc = clock64();
    const size_t arena_bytes = static_cast<size_t>(4096) *
                               static_cast<size_t>(total_tasks + 8);
    auto heap_arena = ipc->PushPrivArena(arena_bytes);
    auto arena = ipc->PushArena(arena_bytes);
    t_push_arena += clock64() - tc;

    for (chi::u32 i = 0; i < total_tasks; ++i) {
      // --- Alloc task ---
      tc = clock64();
      auto task = ipc->NewTask<wrp_cte::core::PutBlobTask>();
      task->pool_id_ = pool_id;
      task->size_ = 0;
      t_alloc_task += clock64() - tc;

      // --- Construct SaveArchive on stack ---
      tc = clock64();
      chi::LocalSaveTaskArchive ar_save(chi::LocalMsgType::kSerializeIn);
      t_ctor_save += clock64() - tc;

      // --- (skip serialize) ---

      // --- Construct LoadArchive on stack ---
      tc = clock64();
      chi::LocalLoadTaskArchive ar_load(ar_save.GetData());
      ar_load.SetMsgType(chi::LocalMsgType::kSerializeIn);
      t_ctor_load += clock64() - tc;

      // --- Alloc task2 ---
      tc = clock64();
      auto task2 = ipc->NewObj<wrp_cte::core::PutBlobTask>();
      t_alloc_task2 += clock64() - tc;

      // --- (skip deserialize) ---

      // --- Free / Destroy ---
      tc = clock64(); ipc->DelObj(task2);
      t_free_task2 += clock64() - tc;
      tc = clock64(); ar_load.~LocalLoadTaskArchive();
      t_dtor_load += clock64() - tc;
      tc = clock64(); ar_save.~LocalSaveTaskArchive();
      t_dtor_save += clock64() - tc;
      tc = clock64(); ipc->DelTask(task);
      t_free_task += clock64() - tc;
    }

    // --- Arena pop (once) ---
    tc = clock64();
    arena.Release();
    heap_arena.Release();
    t_pop_arena += clock64() - tc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("=== Alloc+Free serde objects, no serialize/deserialize (%u tasks) ===\n",
             total_tasks);
      printf("  sizeof(PutBlobTask):    %llu bytes\n",
             (unsigned long long)sizeof(wrp_cte::core::PutBlobTask));
      printf("  PushArena:              %llu  (%llu/task)\n",
             (unsigned long long)t_push_arena,
             (unsigned long long)(t_push_arena / total_tasks));
      printf("  NewTask<PutBlob>:       %llu  (%llu/task)\n",
             (unsigned long long)t_alloc_task,
             (unsigned long long)(t_alloc_task / total_tasks));
      printf("  LocalSaveTaskArchive(): %llu  (%llu/task)\n",
             (unsigned long long)t_ctor_save,
             (unsigned long long)(t_ctor_save / total_tasks));
      printf("  LocalLoadTaskArchive(): %llu  (%llu/task)\n",
             (unsigned long long)t_ctor_load,
             (unsigned long long)(t_ctor_load / total_tasks));
      printf("  NewObj<PutBlob> task2:  %llu  (%llu/task)\n",
             (unsigned long long)t_alloc_task2,
             (unsigned long long)(t_alloc_task2 / total_tasks));
      printf("  Free:\n");
      printf("    DelObj(task2):        %llu  (%llu/task)\n",
             (unsigned long long)t_free_task2,
             (unsigned long long)(t_free_task2 / total_tasks));
      printf("    ~LoadArchive():       %llu  (%llu/task)\n",
             (unsigned long long)t_dtor_load,
             (unsigned long long)(t_dtor_load / total_tasks));
      printf("    ~SaveArchive():       %llu  (%llu/task)\n",
             (unsigned long long)t_dtor_save,
             (unsigned long long)(t_dtor_save / total_tasks));
      printf("    DelTask(task):        %llu  (%llu/task)\n",
             (unsigned long long)t_free_task,
             (unsigned long long)(t_free_task / total_tasks));
      long long t_free_total = t_free_task2 + t_dtor_load + t_dtor_save + t_free_task;
      printf("    SUBTOTAL:             %llu  (%llu/task)\n",
             (unsigned long long)t_free_total,
             (unsigned long long)(t_free_total / total_tasks));
      printf("  PopArena:               %llu  (%llu/task)\n",
             (unsigned long long)t_pop_arena,
             (unsigned long long)(t_pop_arena / total_tasks));
      long long total = t_push_arena + t_alloc_task + t_ctor_save +
                         t_ctor_load + t_alloc_task2 +
                         t_free_total + t_pop_arena;
      printf("  TOTAL:                  %llu  (%llu/task)\n",
             (unsigned long long)total,
             (unsigned long long)(total / total_tasks));
    }
  }

  __threadfence();
  int prev = atomicAdd(d_done, 1);
  if (prev == static_cast<int>(total_threads) - 1) {
    __threadfence_system();
  }
}

/**
 * GPU alloc+serialize+deserialize+free benchmark kernel.
 *
 * Each thread does total_tasks cycles of:
 *   1. NewTask<PutBlobTask> — allocate
 *   2. LocalSaveTaskArchive << task — serialize input
 *   3. LocalLoadTaskArchive >> task2 — deserialize into new task
 *   4. DelTask — free both
 *
 */
__global__ void gpu_bench_serde_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done,
    chi::u32 total_threads) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  auto *ipc = CHI_IPC;

  // BuddyAllocator is not thread-safe for concurrent access by multiple
  // threads within a warp.  Only the warp scheduler (lane 0) runs the
  // serde benchmarks to avoid corrupting the allocator free lists.
  if (chi::IpcManager::IsWarpScheduler()) {

  // === Baseline 1: 8x memcpy with offset tracking (stack→stack) ===
  long long t_memcpy_stack = 0;
  volatile char sink = 0;
  {
    constexpr size_t kBufSize = 512;
    char stack_buf[kBufSize];
    memset(stack_buf, 0, kBufSize);
    char src_buf[92];
    memset(src_buf, 0x42, sizeof(src_buf));
    constexpr size_t field_sizes[8] = {8, 8, 32, 4, 4, 8, 24, 4};
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tm0 = clock64();
      size_t off = 0;
      for (int f = 0; f < 8; ++f) {
        memcpy(stack_buf + off, src_buf, field_sizes[f]);
        off += field_sizes[f];
      }
      long long tm1 = clock64();
      t_memcpy_stack += (tm1 - tm0);
    }
    sink = stack_buf[0];
  }

// === Baseline 2: LocalSerialize 8 scalars (stack locals, no arena) ===
  long long t_ls_vec_alloc = 0, t_ls_serialize = 0;
  {
    chi::u64 f0 = 0x1111111111111111ULL, f1 = 0x2222222222222222ULL;
    chi::u32 f2 = 0x33333333, f3 = 0x44444444;
    chi::u64 f4 = 0x5555555555555555ULL, f5 = 0x6666666666666666ULL;
    chi::u32 f6 = 0x77777777, f7 = 0x88888888;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tl0 = clock64();
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      long long tl1 = clock64();
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      ser(f0, f1, f2, f3, f4, f5, f6, f7);
      long long tl2 = clock64();
      t_ls_vec_alloc += (tl1 - tl0);
      t_ls_serialize += (tl2 - tl1);
    }
  }
  long long t_ls_vec_free = 0;
  {
    chi::u64 f0 = 0x1111111111111111ULL, f1 = 0x2222222222222222ULL;
    chi::u32 f2 = 0x33333333, f3 = 0x44444444;
    chi::u64 f4 = 0x5555555555555555ULL, f5 = 0x6666666666666666ULL;
    chi::u32 f6 = 0x77777777, f7 = 0x88888888;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      ser(f0, f1, f2, f3, f4, f5, f6, f7);
      long long tf0 = clock64();
      buf.clear();
      buf.shrink_to_fit();
      long long tf1 = clock64();
      t_ls_vec_free += (tf1 - tf0);
    }
  }

// === Baseline 3: LocalSerialize 8 scalars WITH arena (should be faster) ===
  long long t_ls_arena_alloc = 0, t_ls_arena_ser = 0, t_ls_arena_free = 0;
  long long t_ls_arena_push = 0, t_ls_arena_pop = 0;
  {
    chi::u64 f0 = 0x1111111111111111ULL, f1 = 0x2222222222222222ULL;
    chi::u32 f2 = 0x33333333, f3 = 0x44444444;
    chi::u64 f4 = 0x5555555555555555ULL, f5 = 0x6666666666666666ULL;
    chi::u32 f6 = 0x77777777, f7 = 0x88888888;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc;
      tc = clock64();
      auto ha = ipc->PushPrivArena(4096);
      t_ls_arena_push += clock64() - tc;
      tc = clock64();
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      t_ls_arena_alloc += clock64() - tc;
      tc = clock64();
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      ser(f0, f1, f2, f3, f4, f5, f6, f7);
      t_ls_arena_ser += clock64() - tc;
      tc = clock64();
      buf.clear();
      buf.shrink_to_fit();
      t_ls_arena_free += clock64() - tc;
      tc = clock64();
      ha.Release();
      t_ls_arena_pop += clock64() - tc;
    }
  }

// === Baseline 4: LocalSerialize 8 scalars from HEAP object ===
  struct HeapFields {
    chi::u64 f0, f1; chi::u32 f2, f3;
    chi::u64 f4, f5; chi::u32 f6, f7;
    HSHM_CROSS_FUN HeapFields()
        : f0(0x11), f1(0x22), f2(0x33), f3(0x44),
          f4(0x55), f5(0x66), f6(0x77), f7(0x88) {}
  };
  long long t_ls_heap_alloc = 0, t_ls_heap_ser = 0, t_ls_heap_free = 0;
  {
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc;
      tc = clock64();
      auto hf = ipc->NewObj<HeapFields>();
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      t_ls_heap_alloc += clock64() - tc;
      tc = clock64();
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      ser(hf->f0, hf->f1, hf->f2, hf->f3, hf->f4, hf->f5, hf->f6, hf->f7);
      t_ls_heap_ser += clock64() - tc;
      tc = clock64();
      buf.clear();
      buf.shrink_to_fit();
      ipc->DelObj(hf);
      t_ls_heap_free += clock64() - tc;
    }
  }

// === Baseline 5: LocalSerialize 8 scalars via POINTER to stack struct ===
  long long t_ls_ptr_ser = 0;
  {
    HeapFields stack_obj;
    HeapFields *ptr = &stack_obj;
    // Prevent compiler from optimizing away the indirection
    asm volatile("" : "+l"(ptr));
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      long long tc = clock64();
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      ser(ptr->f0, ptr->f1, ptr->f2, ptr->f3, ptr->f4, ptr->f5, ptr->f6, ptr->f7);
      t_ls_ptr_ser += clock64() - tc;
    }
  }

// === Baseline 7: Copy struct fields to locals, THEN serialize locals ===
  // Tests register-promotion theory: if overhead is pointer indirection,
  // copying to locals first should match baseline 2 speed.
  long long t_ls_copy_locals_ser = 0;
  {
    HeapFields stack_obj;
    HeapFields *ptr = &stack_obj;
    asm volatile("" : "+l"(ptr));
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      // Copy fields out through pointer into locals (forces loads)
      chi::u64 l0 = ptr->f0, l1 = ptr->f1;
      chi::u32 l2 = ptr->f2, l3 = ptr->f3;
      chi::u64 l4 = ptr->f4, l5 = ptr->f5;
      chi::u32 l6 = ptr->f6, l7 = ptr->f7;
      // Now serialize from locals (should be register-promoted)
      long long tc = clock64();
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      ser(l0, l1, l2, l3, l4, l5, l6, l7);
      t_ls_copy_locals_ser += clock64() - tc;
    }
  }

// === Baseline 6: Serialize 8 scalars one-at-a-time (separate clock64 per field) ===
  // Tests if clock64() granularity / per-call overhead inflates measurements
  long long t_ls_1by1_total = 0;
  long long t_ls_1by1_fields[8] = {};
  {
    chi::u64 f0 = 0x11, f1 = 0x22; chi::u32 f2 = 0x33, f3 = 0x44;
    chi::u64 f4 = 0x55, f5 = 0x66; chi::u32 f6 = 0x77, f7 = 0x88;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      long long tc;
      tc = clock64(); ser(f0); t_ls_1by1_fields[0] += clock64() - tc;
      tc = clock64(); ser(f1); t_ls_1by1_fields[1] += clock64() - tc;
      tc = clock64(); ser(f2); t_ls_1by1_fields[2] += clock64() - tc;
      tc = clock64(); ser(f3); t_ls_1by1_fields[3] += clock64() - tc;
      tc = clock64(); ser(f4); t_ls_1by1_fields[4] += clock64() - tc;
      tc = clock64(); ser(f5); t_ls_1by1_fields[5] += clock64() - tc;
      tc = clock64(); ser(f6); t_ls_1by1_fields[6] += clock64() - tc;
      tc = clock64(); ser(f7); t_ls_1by1_fields[7] += clock64() - tc;
    }
    for (int j = 0; j < 8; ++j) t_ls_1by1_total += t_ls_1by1_fields[j];
  }

// === Baseline 8: String serialization cost ===
  long long t_str_ser = 0, t_str_deser = 0;
  long long t_str_alloc = 0, t_str_free = 0;
  {
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc;
      // Allocate string + buffer
      tc = clock64();
      chi::priv::string test_str(CHI_PRIV_ALLOC, "my_blob_name_test");
      chi::priv::vector<char> buf(CHI_PRIV_ALLOC);
      buf.reserve(256);
      t_str_alloc += clock64() - tc;

      // Serialize string
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ser(buf);
      tc = clock64();
      ser << test_str;
      t_str_ser += clock64() - tc;

      // Deserialize string
      hshm::ipc::LocalDeserialize<chi::priv::vector<char>> deser(buf);
      chi::priv::string out_str(CHI_PRIV_ALLOC);
      tc = clock64();
      deser >> out_str;
      t_str_deser += clock64() - tc;

      // Free
      tc = clock64();
      out_str.~basic_string();
      test_str.~basic_string();
      t_str_free += clock64() - tc;
    }
  }

  // === Baseline 8b: DISABLED — sharing backend between allocators corrupts memory ===
  long long t_str_arena_alloc = 0, t_str_arena_free = 0;
  long long t_str_buddy_alloc = 0, t_str_buddy_free = 0;
  long long t_str_thread_alloc = 0, t_str_thread_free = 0;

// === Baseline 8c: LocalSerialize 8 scalars into stack array (range, no allocator) ===
  long long t_ls_array_ser = 0;
  {
    struct Fields {
      chi::u64 f0, f1;
      chi::u32 f2, f3;
      chi::u64 f4, f5;
      chi::u32 f6, f7;
    };
    Fields fields = {0x1111111111111111ULL, 0x2222222222222222ULL,
                     0x33333333, 0x44444444,
                     0x5555555555555555ULL, 0x6666666666666666ULL,
                     0x77777777, 0x88888888};
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      hshm::ipc::array<char, 4096> buf;
      hshm::ipc::LocalSerialize<hshm::ipc::array<char, 4096>> ser(buf);
      long long tc = clock64();
      ser.range(fields.f0, fields.f1, fields.f2, fields.f3,
                fields.f4, fields.f5, fields.f6, fields.f7);
      t_ls_array_ser += clock64() - tc;
    }
  }

// === Baseline 8d: LocalSerialize 8 scalars + string into stack array (range) ===
  long long t_ls_array_str_ser = 0, t_ls_array_str_deser = 0;
  {
    struct Fields {
      chi::u64 f0, f1;
      chi::u32 f2, f3;
      chi::u64 f4, f5;
      chi::u32 f6, f7;
    };
    Fields fields = {0x1111111111111111ULL, 0x2222222222222222ULL,
                     0x33333333, 0x44444444,
                     0x5555555555555555ULL, 0x6666666666666666ULL,
                     0x77777777, 0x88888888};
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      // Serialize
      hshm::ipc::array<char, 4096> buf;
      hshm::ipc::LocalSerialize<hshm::ipc::array<char, 4096>> ser(buf);
      chi::priv::string test_str(CHI_PRIV_ALLOC, "my_blob_name_test");
      long long tc = clock64();
      ser.range(fields.f0, fields.f1, fields.f2, fields.f3,
                fields.f4, fields.f5, fields.f6, fields.f7);
      ser << test_str;
      ser.Finalize();
      t_ls_array_str_ser += clock64() - tc;

      // Deserialize
      hshm::ipc::LocalDeserialize<hshm::ipc::array<char, 4096>> deser(buf);
      Fields out_fields;
      chi::priv::string out_str(CHI_PRIV_ALLOC);
      tc = clock64();
      deser.range(out_fields.f0, out_fields.f1, out_fields.f2, out_fields.f3,
                  out_fields.f4, out_fields.f5, out_fields.f6, out_fields.f7);
      deser >> out_str;
      t_ls_array_str_deser += clock64() - tc;
    }
  }

// === Baseline 8e: Matrix A = B + C (global memory, varying sizes) ===
  // Tests raw global memory throughput for comparison
  long long t_mat_64 = 0, t_mat_256 = 0, t_mat_1024 = 0, t_mat_4096 = 0;
  {
    auto *alloc = CHI_PRIV_ALLOC;
    // 64 bytes (8 doubles)
    {
      auto pa = alloc->AllocateObjs<double>(8);
      auto pb = alloc->AllocateObjs<double>(8);
      auto pc = alloc->AllocateObjs<double>(8);
      for (int j = 0; j < 8; ++j) { pb.ptr_[j] = 1.0; pc.ptr_[j] = 2.0; }
      for (chi::u32 i = 0; i < total_tasks; ++i) {
        long long tc = clock64();
        for (int j = 0; j < 8; ++j) pa.ptr_[j] = pb.ptr_[j] + pc.ptr_[j];
        t_mat_64 += clock64() - tc;
      }
      alloc->Free(pc); alloc->Free(pb); alloc->Free(pa);
    }
    // 256 bytes (32 doubles)
    {
      auto pa = alloc->AllocateObjs<double>(32);
      auto pb = alloc->AllocateObjs<double>(32);
      auto pc = alloc->AllocateObjs<double>(32);
      for (int j = 0; j < 32; ++j) { pb.ptr_[j] = 1.0; pc.ptr_[j] = 2.0; }
      for (chi::u32 i = 0; i < total_tasks; ++i) {
        long long tc = clock64();
        for (int j = 0; j < 32; ++j) pa.ptr_[j] = pb.ptr_[j] + pc.ptr_[j];
        t_mat_256 += clock64() - tc;
      }
      alloc->Free(pc); alloc->Free(pb); alloc->Free(pa);
    }
    // 1024 bytes (128 doubles)
    {
      auto pa = alloc->AllocateObjs<double>(128);
      auto pb = alloc->AllocateObjs<double>(128);
      auto pc = alloc->AllocateObjs<double>(128);
      for (int j = 0; j < 128; ++j) { pb.ptr_[j] = 1.0; pc.ptr_[j] = 2.0; }
      for (chi::u32 i = 0; i < total_tasks; ++i) {
        long long tc = clock64();
        for (int j = 0; j < 128; ++j) pa.ptr_[j] = pb.ptr_[j] + pc.ptr_[j];
        t_mat_1024 += clock64() - tc;
      }
      alloc->Free(pc); alloc->Free(pb); alloc->Free(pa);
    }
    // 4096 bytes (512 doubles)
    {
      auto pa = alloc->AllocateObjs<double>(512);
      auto pb = alloc->AllocateObjs<double>(512);
      auto pc = alloc->AllocateObjs<double>(512);
      for (int j = 0; j < 512; ++j) { pb.ptr_[j] = 1.0; pc.ptr_[j] = 2.0; }
      for (chi::u32 i = 0; i < total_tasks; ++i) {
        long long tc = clock64();
        for (int j = 0; j < 512; ++j) pa.ptr_[j] = pb.ptr_[j] + pc.ptr_[j];
        t_mat_4096 += clock64() - tc;
      }
      alloc->Free(pc); alloc->Free(pb); alloc->Free(pa);
    }
  }

// === Baseline 9: CHI_IPC singleton access latency ===
  long long t_ipc_access = 0;
  {
    volatile void *sink_ptr = nullptr;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      auto *ipc_ptr = CHI_IPC;
      t_ipc_access += clock64() - tc;
      sink_ptr = ipc_ptr;  // prevent optimization
    }
    (void)sink_ptr;
  }

  // === Baseline 10: CHI_PRIV_ALLOC access latency ===
  long long t_priv_alloc_access = 0;
  {
    volatile void *sink_ptr = nullptr;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      auto *alloc_ptr = CHI_PRIV_ALLOC;
      t_priv_alloc_access += clock64() - tc;
      sink_ptr = alloc_ptr;  // prevent optimization
    }
    (void)sink_ptr;
  }

  // === Baseline 11: Raw BuddyAllocator Allocate+Free (256 bytes) ===
  long long t_buddy_alloc = 0, t_buddy_free = 0;
  {
    auto *alloc = CHI_PRIV_ALLOC;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc;
      tc = clock64();
      auto p = alloc->AllocateOffset(256);
      t_buddy_alloc += clock64() - tc;
      tc = clock64();
      alloc->FreeOffsetNoNullCheck(p);
      t_buddy_free += clock64() - tc;
    }
  }

  // === Baseline 12: Raw BuddyAllocator Allocate+Free (32 bytes, min size) ===
  long long t_buddy32_alloc = 0, t_buddy32_free = 0;
  {
    auto *alloc = CHI_PRIV_ALLOC;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc;
      tc = clock64();
      auto p = alloc->AllocateOffset(32);
      t_buddy32_alloc += clock64() - tc;
      tc = clock64();
      alloc->FreeOffsetNoNullCheck(p);
      t_buddy32_free += clock64() - tc;
    }
  }

  // === Baseline 13: priv::vector<char> construct+reserve+destroy (no serialize) ===
  long long t_vec_ctor = 0, t_vec_reserve = 0, t_vec_dtor = 0;
  {
    auto *alloc = CHI_PRIV_ALLOC;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc;
      tc = clock64();
      chi::priv::vector<char> buf(alloc);
      t_vec_ctor += clock64() - tc;
      tc = clock64();
      buf.reserve(256);
      t_vec_reserve += clock64() - tc;
      tc = clock64();
      buf.~vector();
      t_vec_dtor += clock64() - tc;
    }
  }

// Full serde timing accumulators
  long long t_push_arena = 0, t_pop_arena = 0;
  long long t_alloc_task = 0, t_ctor_save = 0, t_ctor_load = 0, t_alloc_task2 = 0;
  long long t_serialize = 0, t_deserialize = 0;
  long long t_free_task2 = 0, t_dtor_load = 0, t_dtor_save = 0, t_free_task = 0;

  // Push arenas once to avoid per-iteration push/pop overhead.
  // The benchmark is intended to measure task/archive allocation and serde,
  // not arena management costs.
  long long tc = clock64();
  const size_t arena_bytes = static_cast<size_t>(4096) *
                             static_cast<size_t>(total_tasks + 8);
  auto heap_arena = ipc->PushPrivArena(arena_bytes);
  auto arena = ipc->PushArena(arena_bytes);
  t_push_arena += clock64() - tc;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    // --- Alloc task ---
    tc = clock64();
    auto task = ipc->NewTask<wrp_cte::core::PutBlobTask>();
    task->pool_id_ = pool_id;
    // Avoid timing bulk payload copies here; this phase focuses on structure serde.
    task->size_ = 0;
    t_alloc_task += clock64() - tc;

    // --- Construct SaveArchive on stack ---
    tc = clock64();
    chi::LocalSaveTaskArchive ar_save(chi::LocalMsgType::kSerializeIn);
    t_ctor_save += clock64() - tc;

    // --- Serialize via SerializeIn (uses write_range batching) ---
    tc = clock64();
    ar_save << (*task.ptr_);
    t_serialize += clock64() - tc;

    // --- Construct LoadArchive on stack ---
    tc = clock64();
    chi::LocalLoadTaskArchive ar_load(ar_save.GetData());
    ar_load.SetMsgType(chi::LocalMsgType::kSerializeIn);
    t_ctor_load += clock64() - tc;

    // --- Alloc task2 ---
    tc = clock64();
    auto task2 = ipc->NewObj<wrp_cte::core::PutBlobTask>();
    t_alloc_task2 += clock64() - tc;

    // --- Deserialize via SerializeIn (uses read_range batching) ---
    tc = clock64();
    ar_load >> (*task2.ptr_);
    t_deserialize += clock64() - tc;

    // --- Free / Destroy ---
    tc = clock64(); ipc->DelObj(task2);
    t_free_task2 += clock64() - tc;
    tc = clock64(); ar_load.~LocalLoadTaskArchive();
    t_dtor_load += clock64() - tc;
    tc = clock64(); ar_save.~LocalSaveTaskArchive();
    t_dtor_save += clock64() - tc;
    tc = clock64(); ipc->DelTask(task);
    t_free_task += clock64() - tc;
  }

  // --- Arena pop (once) ---
  tc = clock64();
  arena.Release();
  heap_arena.Release();
  t_pop_arena += clock64() - tc;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("=== Serde per-phase clocks (total over %u tasks) ===\n", total_tasks);
    printf("  sizeof(PutBlobTask):    %llu bytes\n", (unsigned long long)sizeof(wrp_cte::core::PutBlobTask));
    printf("--- Baselines ---\n");
    printf("  1. 8x memcpy(stack):    %llu  (%llu/task)\n",
           (unsigned long long)t_memcpy_stack, (unsigned long long)(t_memcpy_stack / total_tasks));
    printf("  2. LocalSerialize 8 scalars (stack, no arena):\n");
    printf("    vec alloc+reserve:    %llu  (%llu/task)\n", (unsigned long long)t_ls_vec_alloc, (unsigned long long)(t_ls_vec_alloc / total_tasks));
    printf("    serialize:            %llu  (%llu/task)\n", (unsigned long long)t_ls_serialize, (unsigned long long)(t_ls_serialize / total_tasks));
    printf("    vec free:             %llu  (%llu/task)\n", (unsigned long long)t_ls_vec_free, (unsigned long long)(t_ls_vec_free / total_tasks));
    long long t_ls2 = t_ls_vec_alloc + t_ls_serialize + t_ls_vec_free;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_ls2, (unsigned long long)(t_ls2 / total_tasks));
    printf("  3. LocalSerialize 8 scalars (stack, WITH arena):\n");
    printf("    arena push:           %llu  (%llu/task)\n", (unsigned long long)t_ls_arena_push, (unsigned long long)(t_ls_arena_push / total_tasks));
    printf("    vec alloc+reserve:    %llu  (%llu/task)\n", (unsigned long long)t_ls_arena_alloc, (unsigned long long)(t_ls_arena_alloc / total_tasks));
    printf("    serialize:            %llu  (%llu/task)\n", (unsigned long long)t_ls_arena_ser, (unsigned long long)(t_ls_arena_ser / total_tasks));
    printf("    vec free:             %llu  (%llu/task)\n", (unsigned long long)t_ls_arena_free, (unsigned long long)(t_ls_arena_free / total_tasks));
    printf("    arena pop:            %llu  (%llu/task)\n", (unsigned long long)t_ls_arena_pop, (unsigned long long)(t_ls_arena_pop / total_tasks));
    long long t_ls3 = t_ls_arena_push + t_ls_arena_alloc + t_ls_arena_ser + t_ls_arena_free + t_ls_arena_pop;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_ls3, (unsigned long long)(t_ls3 / total_tasks));
    printf("  4. LocalSerialize 8 scalars (HEAP object, no arena):\n");
    printf("    obj+vec alloc:        %llu  (%llu/task)\n", (unsigned long long)t_ls_heap_alloc, (unsigned long long)(t_ls_heap_alloc / total_tasks));
    printf("    serialize:            %llu  (%llu/task)\n", (unsigned long long)t_ls_heap_ser, (unsigned long long)(t_ls_heap_ser / total_tasks));
    printf("    obj+vec free:         %llu  (%llu/task)\n", (unsigned long long)t_ls_heap_free, (unsigned long long)(t_ls_heap_free / total_tasks));
    long long t_ls4 = t_ls_heap_alloc + t_ls_heap_ser + t_ls_heap_free;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_ls4, (unsigned long long)(t_ls4 / total_tasks));
    printf("  5. LocalSerialize 8 scalars (POINTER to stack struct):\n");
    printf("    serialize:            %llu  (%llu/task)\n", (unsigned long long)t_ls_ptr_ser, (unsigned long long)(t_ls_ptr_ser / total_tasks));
    printf("  7. LocalSerialize 8 scalars (ptr->fields copied to locals first):\n");
    printf("    serialize:            %llu  (%llu/task)\n", (unsigned long long)t_ls_copy_locals_ser, (unsigned long long)(t_ls_copy_locals_ser / total_tasks));
    printf("  6. LocalSerialize 8 scalars (stack, 1-at-a-time timing):\n");
    for (int j = 0; j < 8; ++j) {
      printf("    field[%d]:             %llu  (%llu/task)\n", j,
             (unsigned long long)t_ls_1by1_fields[j], (unsigned long long)(t_ls_1by1_fields[j] / total_tasks));
    }
    printf("    SUBTOTAL:             %llu  (%llu/task)\n", (unsigned long long)t_ls_1by1_total, (unsigned long long)(t_ls_1by1_total / total_tasks));
    printf("  8. String serialization (\"my_blob_name_test\", 17 chars):\n");
    printf("    str+vec alloc:        %llu  (%llu/task)\n", (unsigned long long)t_str_alloc, (unsigned long long)(t_str_alloc / total_tasks));
    printf("    serialize:            %llu  (%llu/task)\n", (unsigned long long)t_str_ser, (unsigned long long)(t_str_ser / total_tasks));
    printf("    deserialize:          %llu  (%llu/task)\n", (unsigned long long)t_str_deser, (unsigned long long)(t_str_deser / total_tasks));
    printf("    free:                 %llu  (%llu/task)\n", (unsigned long long)t_str_free, (unsigned long long)(t_str_free / total_tasks));
    long long t_str_total = t_str_alloc + t_str_ser + t_str_deser + t_str_free;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_str_total, (unsigned long long)(t_str_total / total_tasks));
    printf("  8b. hshm::priv::string alloc+free by allocator (no serialize, long string):\n");
    printf("    Arena alloc:          %llu  (%llu/task)\n",
           (unsigned long long)t_str_arena_alloc, (unsigned long long)(t_str_arena_alloc / total_tasks));
    printf("    Arena free:           %llu  (%llu/task)\n",
           (unsigned long long)t_str_arena_free, (unsigned long long)(t_str_arena_free / total_tasks));
    long long t_str_arena_total = t_str_arena_alloc + t_str_arena_free;
    printf("    Arena total:          %llu  (%llu/task)\n",
           (unsigned long long)t_str_arena_total, (unsigned long long)(t_str_arena_total / total_tasks));
    printf("    Buddy alloc:          %llu  (%llu/task)\n",
           (unsigned long long)t_str_buddy_alloc, (unsigned long long)(t_str_buddy_alloc / total_tasks));
    printf("    Buddy free:           %llu  (%llu/task)\n",
           (unsigned long long)t_str_buddy_free, (unsigned long long)(t_str_buddy_free / total_tasks));
    long long t_str_buddy_total = t_str_buddy_alloc + t_str_buddy_free;
    printf("    Buddy total:          %llu  (%llu/task)\n",
           (unsigned long long)t_str_buddy_total, (unsigned long long)(t_str_buddy_total / total_tasks));
    printf("    Thread alloc:         %llu  (%llu/task)\n",
           (unsigned long long)t_str_thread_alloc, (unsigned long long)(t_str_thread_alloc / total_tasks));
    printf("    Thread free:          %llu  (%llu/task)\n",
           (unsigned long long)t_str_thread_free, (unsigned long long)(t_str_thread_free / total_tasks));
    long long t_str_thread_total = t_str_thread_alloc + t_str_thread_free;
    printf("    Thread total:         %llu  (%llu/task)\n",
           (unsigned long long)t_str_thread_total, (unsigned long long)(t_str_thread_total / total_tasks));
    printf("  8c. LocalSerialize 8 scalars into stack array (no allocator):\n");
    printf("    serialize:            %llu  (%llu/task)\n",
           (unsigned long long)t_ls_array_ser, (unsigned long long)(t_ls_array_ser / total_tasks));
    printf("  8d. LocalSerialize 8 scalars + string into stack array:\n");
    printf("    serialize:            %llu  (%llu/task)\n",
           (unsigned long long)t_ls_array_str_ser, (unsigned long long)(t_ls_array_str_ser / total_tasks));
    printf("    deserialize:          %llu  (%llu/task)\n",
           (unsigned long long)t_ls_array_str_deser, (unsigned long long)(t_ls_array_str_deser / total_tasks));
    long long t_ls_array_str_total = t_ls_array_str_ser + t_ls_array_str_deser;
    printf("    total:                %llu  (%llu/task)\n",
           (unsigned long long)t_ls_array_str_total, (unsigned long long)(t_ls_array_str_total / total_tasks));
    printf("  8e. Matrix A = B + C (global memory):\n");
    printf("    64B  (8 doubles):     %llu  (%llu/task)\n",
           (unsigned long long)t_mat_64, (unsigned long long)(t_mat_64 / total_tasks));
    printf("    256B (32 doubles):    %llu  (%llu/task)\n",
           (unsigned long long)t_mat_256, (unsigned long long)(t_mat_256 / total_tasks));
    printf("    1KB  (128 doubles):   %llu  (%llu/task)\n",
           (unsigned long long)t_mat_1024, (unsigned long long)(t_mat_1024 / total_tasks));
    printf("    4KB  (512 doubles):   %llu  (%llu/task)\n",
           (unsigned long long)t_mat_4096, (unsigned long long)(t_mat_4096 / total_tasks));
    printf("--- Allocator & Singleton Baselines ---\n");
    printf("  9. CHI_IPC access:      %llu  (%llu/task)\n",
           (unsigned long long)t_ipc_access, (unsigned long long)(t_ipc_access / total_tasks));
    printf(" 10. CHI_PRIV_ALLOC access: %llu  (%llu/task)\n",
           (unsigned long long)t_priv_alloc_access, (unsigned long long)(t_priv_alloc_access / total_tasks));
    printf(" 11. BuddyAllocator 256B alloc+free:\n");
    printf("    allocate:             %llu  (%llu/task)\n", (unsigned long long)t_buddy_alloc, (unsigned long long)(t_buddy_alloc / total_tasks));
    printf("    free:                 %llu  (%llu/task)\n", (unsigned long long)t_buddy_free, (unsigned long long)(t_buddy_free / total_tasks));
    long long t_buddy_total = t_buddy_alloc + t_buddy_free;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_buddy_total, (unsigned long long)(t_buddy_total / total_tasks));
    printf(" 12. BuddyAllocator 32B alloc+free:\n");
    printf("    allocate:             %llu  (%llu/task)\n", (unsigned long long)t_buddy32_alloc, (unsigned long long)(t_buddy32_alloc / total_tasks));
    printf("    free:                 %llu  (%llu/task)\n", (unsigned long long)t_buddy32_free, (unsigned long long)(t_buddy32_free / total_tasks));
    long long t_buddy32_total = t_buddy32_alloc + t_buddy32_free;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_buddy32_total, (unsigned long long)(t_buddy32_total / total_tasks));
    printf(" 13. priv::vector<char> lifecycle (no serialize):\n");
    printf("    construct:            %llu  (%llu/task)\n", (unsigned long long)t_vec_ctor, (unsigned long long)(t_vec_ctor / total_tasks));
    printf("    reserve(256):         %llu  (%llu/task)\n", (unsigned long long)t_vec_reserve, (unsigned long long)(t_vec_reserve / total_tasks));
    printf("    destroy:              %llu  (%llu/task)\n", (unsigned long long)t_vec_dtor, (unsigned long long)(t_vec_dtor / total_tasks));
    long long t_vec_total = t_vec_ctor + t_vec_reserve + t_vec_dtor;
    printf("    total:                %llu  (%llu/task)\n", (unsigned long long)t_vec_total, (unsigned long long)(t_vec_total / total_tasks));
    printf("--- Full PutBlobTask serde (with write_range batching) ---\n");
    printf("  PushArena:              %llu  (%llu/task)\n", (unsigned long long)t_push_arena, (unsigned long long)(t_push_arena/total_tasks));
    printf("  NewTask<PutBlob>:       %llu  (%llu/task)\n", (unsigned long long)t_alloc_task, (unsigned long long)(t_alloc_task/total_tasks));
    printf("  LocalSaveTaskArchive(): %llu  (%llu/task)\n", (unsigned long long)t_ctor_save, (unsigned long long)(t_ctor_save/total_tasks));
    printf("  Serialize (full):       %llu  (%llu/task)\n", (unsigned long long)t_serialize, (unsigned long long)(t_serialize/total_tasks));
    printf("  LocalLoadTaskArchive(): %llu  (%llu/task)\n", (unsigned long long)t_ctor_load, (unsigned long long)(t_ctor_load/total_tasks));
    printf("  NewObj<PutBlob> task2:  %llu  (%llu/task)\n", (unsigned long long)t_alloc_task2, (unsigned long long)(t_alloc_task2/total_tasks));
    printf("  Deserialize (full):     %llu  (%llu/task)\n", (unsigned long long)t_deserialize, (unsigned long long)(t_deserialize/total_tasks));
    printf("  Free:\n");
    printf("    DelObj(task2):        %llu  (%llu/task)\n", (unsigned long long)t_free_task2, (unsigned long long)(t_free_task2/total_tasks));
    printf("    ~LoadArchive():       %llu  (%llu/task)\n", (unsigned long long)t_dtor_load, (unsigned long long)(t_dtor_load/total_tasks));
    printf("    ~SaveArchive():       %llu  (%llu/task)\n", (unsigned long long)t_dtor_save, (unsigned long long)(t_dtor_save/total_tasks));
    printf("    DelTask(task):        %llu  (%llu/task)\n", (unsigned long long)t_free_task, (unsigned long long)(t_free_task/total_tasks));
    long long t_free_total = t_free_task2 + t_dtor_load + t_dtor_save + t_free_task;
    printf("    SUBTOTAL:             %llu  (%llu/task)\n", (unsigned long long)t_free_total, (unsigned long long)(t_free_total/total_tasks));
    printf("  PopArena:               %llu  (%llu/task)\n", (unsigned long long)t_pop_arena, (unsigned long long)(t_pop_arena/total_tasks));
    long long total = t_push_arena + t_alloc_task + t_ctor_save +
                      t_serialize + t_ctor_load + t_alloc_task2 +
                      t_deserialize + t_free_total + t_pop_arena;
    printf("  TOTAL:                  %llu  (%llu/task)\n", (unsigned long long)total, (unsigned long long)(total / total_tasks));
  }

  }  // IsWarpScheduler

  __threadfence();
  int prev = atomicAdd(d_done, 1);
  if (prev == static_cast<int>(total_threads) - 1) {
    __threadfence_system();
  }
}

}  // namespace chi_bench

/**
 * Poll the pinned done flag until set or timeout.
 */
static bool PollDone(volatile int *d_done, int total_threads, int timeout_us) {
  int elapsed_us = 0;
  while (*d_done < total_threads && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    elapsed_us += 100;
  }
  return *d_done >= total_threads;
}

/**
 * Run the GPU runtime latency benchmark.
 *
 * All IpcManager access (RegisterGpuAllocator, GetGpuToGpuQueue,
 * SetGpuOrchestratorBlocks) is done via non-inline methods defined in
 * ipc_manager.cc (g++-compiled) to avoid ODR layout mismatches.
 *
 * @param pool_id        Pool ID of the MOD_NAME container
 * @param method_id      (unused; kept for ABI compatibility)
 * @param rt_blocks      GPU work orchestrator block count
 * @param rt_threads     GPU work orchestrator threads per block
 * @param client_blocks  GPU client kernel block count
 * @param client_threads (unused; only thread 0 per block does work)
 * @param batch_size     (unused; kept for ABI compatibility)
 * @param total_tasks    Total sequential tasks per block's thread 0
 * @param out_elapsed_ms Output: elapsed wall-clock time in ms
 * @return 0 on success, negative error code on failure
 */
extern "C" int run_gpu_bench_latency(
    chi::PoolId pool_id,
    chi::u32 method_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 batch_size,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  // Use non-inline SetGpuOrchestratorBlocks to avoid ODR layout mismatch
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Allocate primary GPU backend (GpuMalloc, device memory): 10 MB per block.
  // Used by PartitionedAllocator for FutureShm allocation on GPU→GPU path.
  // Device memory avoids PCIe round-trips for gpu2gpu atomics.
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(100, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  // Allocate GPU heap backend (GpuMalloc, device memory): 4 MB per block.
  // Build IpcManagerGpuInfo from runtime's full GPU info, then override backends
  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  void *stream = hshm::GpuApi::CreateStream();

  // Pause orchestrator to free SMs, launch client kernel, then resume
  CHI_IPC->PauseGpuOrchestrator();
  cudaGetLastError();  // Clear any sticky CUDA errors

  chi_bench::gpu_bench_client_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, d_done, total_warps);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  // Resume orchestrator so it can process the GPU→GPU tasks
  CHI_IPC->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  // Poll for all warps to complete (60 s timeout)
  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start)
                          .count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  // Synchronize client kernel stream before pausing the orchestrator.
  // Block 0 signals d_done=1 but blocks 1-(client_blocks-1) may still be
  // processing their last GPU task.  The orchestrator must remain active so
  // it can deliver those final responses.  Without this synchronize those
  // blocks spin forever in future.Wait() and CUDA context cleanup at process
  // exit hangs waiting for the GPU kernel to terminate.
  hshm::GpuApi::Synchronize(stream);

  // All client blocks have now finished — safe to stop the orchestrator.
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;  // -4 = timeout
}

/**
 * Run the GPU runtime coroutine benchmark.
 *
 * Same structure as run_gpu_bench_latency but launches gpu_bench_coroutine_kernel
 * which uses SubtaskTest (coroutine with co_await) instead of leaf GpuSubmit.
 */
extern "C" int run_gpu_bench_coroutine(
    chi::PoolId pool_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    chi::u32 subtasks,
    float *out_elapsed_ms) {
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(102, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  CHI_IPC->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  void *stream = hshm::GpuApi::CreateStream();

  CHI_IPC->PauseGpuOrchestrator();
  cudaGetLastError();

  chi_bench::gpu_bench_coroutine_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, subtasks, d_done, total_warps);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_IPC->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start)
                          .count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * Run the GPU alloc/free benchmark.
 *
 * No orchestrator needed — just initializes a GpuMalloc backend with
 * PartitionedAllocator and runs NewTask/DelTask cycles on the GPU.
 */
extern "C" int run_gpu_bench_alloc(
    chi::PoolId pool_id,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(104, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  // Build minimal gpu_info — only the primary backend matters for alloc/free
  chi::IpcManagerGpu gpu_info{};
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_threads = client_blocks * client_threads;

  void *stream = hshm::GpuApi::CreateStream();

  cudaGetLastError();

  chi_bench::gpu_bench_alloc_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, d_done, total_threads);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_threads), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start)
                          .count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * Run the GPU serialize/deserialize benchmark.
 *
 * Allocates both a primary backend (for NewTask) and a heap backend
 * (for serialization scratch via CHI_PRIV_ALLOC / CHI_PRIV_ALLOC).
 */
extern "C" int run_gpu_bench_serde(
    chi::PoolId pool_id,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  // Primary backend: sized to hold arena allocations for all threads.
  // Each thread bump-allocates ~4KB * total_tasks in the serde loop.
  constexpr size_t kPerBlockBytes = 32 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(105, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  // Heap backend for serialization scratch (CHI_PRIV_ALLOC / CHI_PRIV_ALLOC).
  // Arena allocations are bump-only (frees are no-ops), so the heap must
  // hold all baseline + full-serde allocations across all threads.
  chi::IpcManagerGpu gpu_info{};
  gpu_info.backend = gpu_backend;

  // The serde kernel has many local variables and deep call stacks
  // (serialization, allocator, vector operations).  32KB stack is needed.
  cudaDeviceSetLimit(cudaLimitStackSize, 32768);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024);

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_threads = client_blocks * client_threads;

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  chi_bench::gpu_bench_serde_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, d_done, total_threads);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 10000000;  // 10s for debugging
  bool completed = PollDone(d_done, static_cast<int>(total_threads), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start)
                          .count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  cudaError_t sync_err = cudaStreamSynchronize(
      static_cast<cudaStream_t>(stream));
  if (sync_err != cudaSuccess) {
    printf("serde kernel error: %s\n", cudaGetErrorString(sync_err));
  }

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * Run the GPU alloc+free benchmark for PutBlobTask + LocalSaveTaskArchive.
 *
 * Same backend setup as run_gpu_bench_serde but launches
 * gpu_bench_alloc_serde_kernel (alloc/free only, no serialization).
 */
extern "C" int run_gpu_bench_alloc_serde(
    chi::PoolId pool_id,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  // Primary backend for NewTask (PartitionedAllocator)
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(107, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  chi::IpcManagerGpu gpu_info{};
  gpu_info.backend = gpu_backend;

  cudaDeviceSetLimit(cudaLimitStackSize, 32768);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 4 * 1024 * 1024);

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_threads = client_blocks * client_threads;

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  chi_bench::gpu_bench_alloc_serde_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, d_done, total_threads);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("alloc_serde kernel launch error: %s\n",
           cudaGetErrorString(launch_err));
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_threads), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * GPU string allocator comparison kernel.
 *
 * Compares ArenaAllocator, BuddyAllocator, SlabAllocator, and PartitionedAllocator
 * for allocating and freeing a heap-backed string (>SSO size) repeatedly.
 * Each allocator has its own dedicated GpuMalloc backend to avoid corruption.
 * Only thread 0 of block 0 runs the benchmark (single-thread comparison).
 */
__global__ void gpu_bench_string_alloc_kernel(
    hipc::MemoryBackend arena_backend,
    hipc::MemoryBackend buddy_backend,
    hipc::MemoryBackend slab_backend,
    hipc::MemoryBackend thread_backend,
    chi::u32 total_tasks,
    int *d_done) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    atomicAdd(d_done, 1);
    return;
  }

  // 16-char string fits in SSO (< 31 chars), no heap allocation
  const char *kTestStr =
      "16_char_string!!";

  using ArenaAllocT = hshm::ipc::ArenaAllocator<false>;
  using BuddyAllocT = hipc::PrivateBuddyAllocator;
  using SlabAllocT = hipc::BaseAllocator<hipc::PrivateSlabAllocator>;
  using ThreadAllocT = hipc::PartitionedAllocator;

  using ArenaString = hshm::priv::basic_string<char, ArenaAllocT>;
  using BuddyString = hshm::priv::basic_string<char, BuddyAllocT>;
  using SlabString = hshm::priv::basic_string<char, SlabAllocT>;
  using ThreadString = hshm::priv::basic_string<char, ThreadAllocT>;

  // Place each allocator at the start of its backend data region via placement new.
  // This matches how the runtime initializes allocators (header lives in the backend).

  // --- ArenaAllocator ---
  long long t_arena_alloc = 0, t_arena_free = 0;
  {
    auto *alloc = new (arena_backend.data_) ArenaAllocT();
    alloc->shm_init(arena_backend, 0);
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      // Re-init arena each iteration to reclaim memory (arena doesn't free)
      alloc->shm_init(arena_backend, 0);
      long long tc = clock64();
      auto p = alloc->template AllocateObjs<ArenaString>(1);
      new (p.ptr_) ArenaString(alloc, kTestStr);
      t_arena_alloc += clock64() - tc;
      tc = clock64();
      p.ptr_->~ArenaString();
      alloc->Free(p);
      t_arena_free += clock64() - tc;
    }
  }

  // --- BuddyAllocator (global memory) ---
  long long t_buddy_alloc = 0, t_buddy_free = 0;
  {
    auto *alloc = new (buddy_backend.data_) BuddyAllocT();
    alloc->shm_init(buddy_backend, 0);
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      auto p = alloc->template AllocateObjs<BuddyString>(1);
      new (p.ptr_) BuddyString(alloc, kTestStr);
      t_buddy_alloc += clock64() - tc;
      tc = clock64();
      p.ptr_->~BuddyString();
      alloc->Free(p);
      t_buddy_free += clock64() - tc;
    }
  }

  // --- BuddyAllocator + Arena (global memory) ---
  long long t_buddy_arena_alloc = 0, t_buddy_arena_free = 0;
  {
    auto *alloc = new (buddy_backend.data_) BuddyAllocT();
    alloc->shm_init(buddy_backend, 0);
    auto arena = alloc->PushArena(
        static_cast<size_t>(total_tasks + 8) * 256);
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      auto p = alloc->template AllocateObjs<BuddyString>(1);
      new (p.ptr_) BuddyString(alloc, kTestStr);
      t_buddy_arena_alloc += clock64() - tc;
      tc = clock64();
      p.ptr_->~BuddyString();
      alloc->Free(p);
      t_buddy_arena_free += clock64() - tc;
    }
    arena.Release();
  }

  // --- BuddyAllocator + Arena (shared memory) ---
  // DISABLED: Constructing allocator in __shared__ memory causes CUDA error 717
  // ("operation not supported on global/shared address space") because the
  // allocator's this_ offset is computed relative to the global-memory backend,
  // and atomics on shared-memory addresses are incompatible.
  long long t_buddy_smem_alloc = 0, t_buddy_smem_free = 0;

  // --- SlabAllocator ---
  long long t_slab_alloc = 0, t_slab_free = 0;
  {
    auto *alloc = new (slab_backend.data_) SlabAllocT();
    alloc->shm_init(slab_backend, 0);
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      auto p = alloc->template AllocateObjs<SlabString>(1);
      new (p.ptr_) SlabString(alloc, kTestStr);
      t_slab_alloc += clock64() - tc;
      tc = clock64();
      p.ptr_->~SlabString();
      alloc->Free(p);
      t_slab_free += clock64() - tc;
    }
  }

  // --- PartitionedAllocator ---
  long long t_thread_alloc = 0, t_thread_free = 0;
  {
    auto *alloc = new (thread_backend.data_) ThreadAllocT();
    alloc->shm_init(thread_backend, 0, 1, 1024 * 1024);
    alloc->MarkReady();
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      auto p = alloc->template AllocateObjs<ThreadString>(1);
      new (p.ptr_) ThreadString(alloc, kTestStr);
      t_thread_alloc += clock64() - tc;
      tc = clock64();
      p.ptr_->~ThreadString();
      alloc->Free(p);
      t_thread_free += clock64() - tc;
    }
  }

  int kTestStrLen = 0;
  while (kTestStr[kTestStrLen] != '\0') ++kTestStrLen;
  printf("=== String Alloc+Free Benchmark (%u tasks, string=\"%.40s\" [%d chars]) ===\n",
         total_tasks, kTestStr, kTestStrLen);
  printf("  Allocator         Alloc(clk/task)  Free(clk/task)  Total(clk/task)\n");
  printf("  Arena              %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_arena_alloc / total_tasks),
         (unsigned long long)(t_arena_free / total_tasks),
         (unsigned long long)((t_arena_alloc + t_arena_free) / total_tasks));
  printf("  Buddy              %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_buddy_alloc / total_tasks),
         (unsigned long long)(t_buddy_free / total_tasks),
         (unsigned long long)((t_buddy_alloc + t_buddy_free) / total_tasks));
  printf("  Buddy+Arena        %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_buddy_arena_alloc / total_tasks),
         (unsigned long long)(t_buddy_arena_free / total_tasks),
         (unsigned long long)((t_buddy_arena_alloc + t_buddy_arena_free) / total_tasks));
  printf("  Buddy+Arena+Smem   %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_buddy_smem_alloc / total_tasks),
         (unsigned long long)(t_buddy_smem_free / total_tasks),
         (unsigned long long)((t_buddy_smem_alloc + t_buddy_smem_free) / total_tasks));
  printf("  Slab               %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_slab_alloc / total_tasks),
         (unsigned long long)(t_slab_free / total_tasks),
         (unsigned long long)((t_slab_alloc + t_slab_free) / total_tasks));
  printf("  Thread(Slab)       %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_thread_alloc / total_tasks),
         (unsigned long long)(t_thread_free / total_tasks),
         (unsigned long long)((t_thread_alloc + t_thread_free) / total_tasks));

  __threadfence_system();
  atomicAdd(d_done, 1);
}

/**
 * Run the GPU string allocator comparison benchmark.
 *
 * Creates 4 separate GpuMalloc backends (one per allocator) and launches
 * a single-thread kernel that times string alloc+free with each allocator.
 */
extern "C" int run_gpu_bench_string_alloc(
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  constexpr size_t kBackendSize = 4 * 1024 * 1024;  // 4MB per allocator

  hipc::GpuMalloc arena_gpu, buddy_gpu, slab_gpu, thread_gpu;
  if (!arena_gpu.shm_init(hipc::MemoryBackendId(200, 0), kBackendSize, "", 0) ||
      !buddy_gpu.shm_init(hipc::MemoryBackendId(201, 0), kBackendSize, "", 0) ||
      !slab_gpu.shm_init(hipc::MemoryBackendId(202, 0), kBackendSize, "", 0) ||
      !thread_gpu.shm_init(hipc::MemoryBackendId(203, 0), kBackendSize, "", 0)) {
    return -1;
  }

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  gpu_bench_string_alloc_kernel<<<1, 1, 0,
      static_cast<cudaStream_t>(stream)>>>(
      static_cast<hipc::MemoryBackend>(arena_gpu),
      static_cast<hipc::MemoryBackend>(buddy_gpu),
      static_cast<hipc::MemoryBackend>(slab_gpu),
      static_cast<hipc::MemoryBackend>(thread_gpu),
      total_tasks, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("string_alloc kernel launch error: %s\n",
           cudaGetErrorString(launch_err));
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 30000000;  // 30s
  bool completed = PollDone(d_done, 1, kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  cudaError_t sync_err = cudaStreamSynchronize(
      static_cast<cudaStream_t>(stream));
  if (sync_err != cudaSuccess) {
    printf("string_alloc kernel error: %s\n", cudaGetErrorString(sync_err));
  }

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

// ==========================================================================
// PutBlob data placement benchmark
// ==========================================================================

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
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

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
      // offset = distance from backend base
      size_t base_off = array_ptr.shm_.off_.load();
      blob_shm.off_.exchange(base_off + my_offset);

      // Build blob name: "warp_<id>"
      char name_buf[32];
      int pos = 0;
      name_buf[pos++] = 'w';
      name_buf[pos++] = '_';
      chi::u32 wid = warp_id;
      // Simple itoa
      char digits[10];
      int nd = 0;
      do { digits[nd++] = '0' + (wid % 10); wid /= 10; } while (wid > 0);
      for (int d = nd - 1; d >= 0; --d) name_buf[pos++] = digits[d];
      name_buf[pos] = '\0';

      auto future = cte_client.AsyncPutBlob(
          tag_id, name_buf,
          /*offset=*/0, /*size=*/slice_size,
          blob_shm, /*score=*/-1.0f,
          wrp_cte::core::Context(), /*flags=*/0,
          to_cpu ? chi::PoolQuery::ToLocalCpu()
                 : chi::PoolQuery::Local());

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
 * CPU-side launcher for the PutBlob data placement benchmark.
 *
 * 1. Allocates a device memory backend + BuddyAllocator
 * 2. Runs alloc kernel to allocate array A
 * 3. Registers backend with runtime
 * 4. Creates CTE pool + tag
 * 5. Launches data placement kernel (memset + PutBlob per warp)
 */
extern "C" int run_gpu_bench_putblob(
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u64 total_bytes,
    bool to_cpu,
    float *out_elapsed_ms) {
  CHI_IPC->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Pause GPU orchestrator before any cudaDeviceSynchronize / GPU init.
  // The orchestrator is a persistent kernel; cudaDeviceSynchronize would block
  // forever waiting for it.
  CHI_IPC->PauseGpuOrchestrator();

  // --- 1. Data backend: device memory for array A ---
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  // Extra space for BuddyAllocator header
  size_t data_backend_size = total_bytes + 4 * 1024 * 1024;
  if (!data_backend.shm_init(data_backend_id, data_backend_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 2. Client scratch backend (for FutureShm, serialization) ---
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t scratch_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 3. GPU heap backend (for PartitionedAllocator) ---
  constexpr size_t kPerBlockHeapBytes = 4 * 1024 * 1024;
  size_t heap_size = static_cast<size_t>(client_blocks) * kPerBlockHeapBytes;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
    CHI_IPC->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 4. Run alloc kernel to initialize allocator + allocate A ---
  hipc::FullPtr<char> *d_array_ptr;
  cudaMallocHost(&d_array_ptr, sizeof(hipc::FullPtr<char>));
  d_array_ptr->SetNull();

  gpu_putblob_alloc_kernel<<<1, 1>>>(
      static_cast<hipc::MemoryBackend &>(data_backend),
      total_bytes, d_array_ptr);
  cudaDeviceSynchronize();

  if (d_array_ptr->IsNull()) {
    cudaFreeHost(d_array_ptr);
    return -2;
  }

  hipc::FullPtr<char> array_ptr = *d_array_ptr;
  cudaFreeHost(d_array_ptr);

  // --- 5. Register data backend with runtime for ShmPtr resolution ---
  CHI_IPC->RegisterGpuAllocator(data_backend_id, data_backend.data_,
                                 data_backend.data_capacity_);

  // --- 6. Build GPU info and launch data placement kernel ---
  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();

  // Orchestrator is already paused from function start
  cudaGetLastError();

  gpu_putblob_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, client_blocks,
      array_ptr,
      hipc::AllocatorId(data_backend_id.major_, data_backend_id.minor_),
      total_bytes, total_warps, to_cpu, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_IPC->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_IPC->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;  // 60s
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_IPC->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * CPU→GPU parallel dispatch benchmark.
 *
 * Submits GpuSubmit tasks from the CPU to the GPU orchestrator via
 * PoolQuery::ToLocalGpu with the given parallelism. Measures round-trip
 * latency for single-warp (parallelism=32) vs cross-warp (parallelism=2048)
 * dispatch.
 *
 * Must be compiled with HSHM_ENABLE_CUDA so that Send() routes ToLocalGpu
 * tasks to SendToGpu() instead of falling through to the CPU path.
 */
extern "C" int run_gpu_bench_parallel_dispatch(
    chi::PoolId pool_id,
    chi::u32 parallelism,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  chimaera::MOD_NAME::Client client(pool_id);
  chi::u32 gpu_id = 0;

  // Warmup: first task may take longer while orchestrator registers the pool
  for (chi::u32 i = 0; i < 10; ++i) {
    auto future = client.AsyncGpuSubmit(
        chi::PoolQuery::ToLocalGpu(gpu_id, parallelism),
        gpu_id, i);
    if (!future.Wait(30.0f)) {
      return -2;
    }
  }

  auto t_start = std::chrono::high_resolution_clock::now();
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto future = client.AsyncGpuSubmit(
        chi::PoolQuery::ToLocalGpu(gpu_id, parallelism),
        gpu_id, i);
    if (!future.Wait(30.0f)) {
      return -4;
    }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  *out_elapsed_ms = static_cast<float>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          t_end - t_start).count() / 1e6);
  return 0;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
