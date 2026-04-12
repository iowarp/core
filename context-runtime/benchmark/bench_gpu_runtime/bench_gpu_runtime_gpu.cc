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
 * All lanes in each warp call the Async* client methods; IpcManager's
 * NewTask and SendGpu guard internally so only the warp leader (lane 0)
 * allocates and dispatches. Parallelism comes from client_blocks: each
 * block's warps independently submit total_tasks sequential tasks.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/ipc_manager.h>
#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/task.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/MOD_NAME/MOD_NAME_gpu_runtime.h>
#include <chimaera/MOD_NAME/autogen/MOD_NAME_methods.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
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
 * Zero-copy direct dispatch benchmark.
 *
 * Two warps in one block share a pre-allocated task + FutureShm.
 * Warp 0 (client): writes task fields, signals ready via atomic.
 * Warp 1 (orchestrator): reads task in-place, writes result, signals done.
 * No serialization, no allocation, no queues.
 */
struct alignas(128) ZeroCopySlot {
  chimaera::MOD_NAME::GpuSubmitTask task;
  volatile unsigned int ready;   // client → orch: iteration number
  volatile unsigned int done;    // orch → client: iteration number
};

__global__ void gpu_bench_zerocopy_kernel(
    ZeroCopySlot *slot, chi::u32 total_tasks, int *d_done) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  // Init
  if (threadIdx.x == 0) {
    slot->ready = 0;
    slot->done = 0;
    slot->task.test_value_ = 7;
    slot->task.gpu_id_ = 0;
    slot->task.result_value_ = 0;
  }
  __syncthreads();

  chi::u32 errors = 0;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    if (warp_id == 0) {
      // === CLIENT WARP ===
      if (lane == 0) {
        // Write input fields (reuse task, just reset result)
        slot->task.result_value_ = 0;
        __threadfence();
        // Signal ready
        atomicExch(const_cast<unsigned int *>(&slot->ready), i + 1);
      }
      __syncwarp();

      // Spin-wait for done
      if (lane == 0) {
        while (atomicAdd(const_cast<unsigned int *>(&slot->done), 0) != (unsigned)(i + 1)) {}
        __threadfence();
        // Check result
        if (slot->task.result_value_ != 21) ++errors;
      }
      __syncwarp();

    } else if (warp_id == 1) {
      // === ORCHESTRATOR WARP ===
      // Spin-wait for ready
      if (lane == 0) {
        while (atomicAdd(const_cast<unsigned int *>(&slot->ready), 0) != (unsigned)(i + 1)) {}
        __threadfence();
      }
      __syncwarp();

      // Execute task in-place (read fields, compute, write result)
      if (lane == 0) {
        slot->task.result_value_ =
            (slot->task.test_value_ * 3) + slot->task.gpu_id_;
        __threadfence();
        // Signal done
        atomicExch(const_cast<unsigned int *>(&slot->done), i + 1);
      }
      __syncwarp();
    }
  }

  if (warp_id == 0 && lane == 0) {
    if (errors > 0) {
      printf("[ZEROCOPY FAIL] %u/%u errors\n", errors, total_tasks);
    } else {
      printf("[ZEROCOPY OK] %u/%u passed\n", total_tasks, total_tasks);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    *d_done = 1;
    __threadfence_system();
  }
}

/**
 * GPU client benchmark kernel.
 *
 * Each block initializes its IpcManager (via CHIMAERA_GPU_CLIENT_INIT
 * for per-block backend partitioning). All lanes call AsyncGpuSubmit +
 * Wait(); NewTask and SendGpu guard internally so only lane 0 allocates
 * and dispatches.
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

  __syncwarp();

  auto *ipc = CHI_IPC;
  // Allocate task once, reuse across iterations to avoid per-task alloc/free.
  // All lanes participate in NewTask (lane 0 allocates) and Send (warp-parallel).
  // test_value=7, gpu_id=0 → expected result = 7*3+0 = 21
  auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
      (chi::u32)0, (chi::u32)7);

  if (chi::gpu::IpcManager::IsWarpScheduler() && task.IsNull()) {
    printf("[BENCH FATAL] blk=%u NewTask returned null — priv alloc too small\n",
           blockIdx.x);
    atomicAdd(d_done, static_cast<int>(total_warps));
    return;
  }

  chi::u32 errors = 0;
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      task->result_value_ = 0;  // Reset before each send
    }
    auto future = ipc->Send(task);
    future.Wait(0, /*reuse_task=*/true);
    // Verify: GpuSubmit computes test_value*3 + gpu_id = 7*3+0 = 21
    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      if (task->result_value_ != 21) {
        if (errors == 0) {
          printf("[BENCH ERROR] blk=%u task %u: expected result=21, got %u (rc=%d)\n",
                 blockIdx.x, i, (unsigned)task->result_value_,
                 (int)task->return_code_);
        }
        ++errors;
      }
    }
  }

  if (chi::gpu::IpcManager::IsWarpScheduler() && errors > 0) {
    printf("[BENCH FAIL] blk=%u %u/%u tasks returned wrong result\n",
           blockIdx.x, errors, total_tasks);
  }

  // Free the reused task after the loop
  ipc->DelTask(task);

  // All warps signal completion via atomicAdd — no reset race
  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    __threadfence_system();
    atomicAdd(d_done, 1);
  }
}

/**
 * GPU client benchmark kernel for coroutine subtask throughput.
 *
 * Same structure as gpu_bench_client_kernel but dispatches SubtaskTest
 * instead of GpuSubmit. SubtaskTest's GPU Run() dispatches GpuSubmit as subtask
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

  auto *ipc = CHI_IPC;
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto task = ipc->NewTask<chimaera::MOD_NAME::SubtaskTestTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(), i, subtasks);
    auto future = ipc->Send(task);
    future.Wait();
  }

  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
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
  using GpuSubmitTask = chimaera::MOD_NAME::GpuSubmitTask;

  auto *ipc = CHI_IPC;
  auto *alloc = ipc->gpu_alloc_;
  if (!alloc) return;

  // Warm up
  {
    auto w = alloc->template AllocateObjs<GpuSubmitTask>(1);
    if (!w.IsNull()) alloc->Free(w);
  }

  long long t_raw_alloc = 0, t_raw_free = 0;
  long long t_ctor = 0, t_dtor = 0;
  long long t_ser_in = 0, t_ser_out = 0;
  long long t_coro_alloc = 0, t_coro_free = 0;
  long long tc;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    // 1. Raw alloc
    tc = clock64();
    auto fp = alloc->template AllocateObjs<GpuSubmitTask>(1);
    t_raw_alloc += clock64() - tc;
    if (fp.IsNull()) continue;

    // 2. Placement new (constructor)
    tc = clock64();
    new (fp.ptr_) GpuSubmitTask();
    t_ctor += clock64() - tc;

    // 3. SerializeIn (what the orchestrator does to populate the task)
    // Simulate with direct field writes matching SerializeIn
    tc = clock64();
    fp->pool_id_ = pool_id;
    fp->method_ = 25;
    fp->gpu_id_ = 0;
    fp->test_value_ = i;
    fp->result_value_ = 0;
    fp->counter_value_ = 0;
    t_ser_in += clock64() - tc;

    // 4. "Run" the task (trivial — what GpuSubmit does)
    fp->result_value_ = (fp->test_value_ * 3) + fp->gpu_id_;

    // 5. SerializeOut
    tc = clock64();
    chi::u32 tmp = fp->result_value_;
    (void)tmp;
    t_ser_out += clock64() - tc;

    // 6. Destructor
    tc = clock64();
    fp.ptr_->~GpuSubmitTask();
    t_dtor += clock64() - tc;

    // 7. Free
    tc = clock64();
    alloc->Free(fp);
    t_raw_free += clock64() - tc;
  }

  // 8. Coroutine frame alloc/free (simulates inner coroutine overhead)
  // A coroutine frame is ~promise_type + locals, typically 64-128 bytes
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    tc = clock64();
    auto frame = alloc->template AllocateObjs<char>(128);
    t_coro_alloc += clock64() - tc;

    tc = clock64();
    if (!frame.IsNull()) alloc->Free(frame);
    t_coro_free += clock64() - tc;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("=== Orchestrator Task Lifecycle (%u tasks) ===\n", total_tasks);
    printf("  sizeof(GpuSubmitTask):    %llu bytes\n",
           (unsigned long long)sizeof(GpuSubmitTask));
    printf("  1. Raw alloc:             %llu/task\n",
           (unsigned long long)(t_raw_alloc / total_tasks));
    printf("  2. Placement new (ctor):  %llu/task\n",
           (unsigned long long)(t_ctor / total_tasks));
    printf("  3. SerializeIn (fields):  %llu/task\n",
           (unsigned long long)(t_ser_in / total_tasks));
    printf("  4. SerializeOut (fields): %llu/task\n",
           (unsigned long long)(t_ser_out / total_tasks));
    printf("  5. Destructor:            %llu/task\n",
           (unsigned long long)(t_dtor / total_tasks));
    printf("  6. Free:                  %llu/task\n",
           (unsigned long long)(t_raw_free / total_tasks));
    long long t_task = t_raw_alloc + t_ctor + t_ser_in + t_ser_out + t_dtor + t_raw_free;
    printf("  Task subtotal:            %llu/task\n",
           (unsigned long long)(t_task / total_tasks));
    printf("  7. Coro frame alloc:      %llu/task\n",
           (unsigned long long)(t_coro_alloc / total_tasks));
    printf("  8. Coro frame free:       %llu/task\n",
           (unsigned long long)(t_coro_free / total_tasks));
    printf("  Full lifecycle:           %llu/task\n",
           (unsigned long long)((t_task + t_coro_alloc + t_coro_free) / total_tasks));
  }

  __threadfence();
  int prev = atomicAdd(d_done, 1);
  if (prev == static_cast<int>(total_threads) - 1) {
    __threadfence_system();
  }
}

/**
 * GPU alloc+free benchmark kernel for PutBlobTask + DefaultSaveArchive.
 *
 * Each warp scheduler does total_tasks cycles of:
 *   1. NewTask<PutBlobTask> — allocate task
 *   2. NewObj<DefaultSaveArchive> — allocate save archive
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
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    long long t_push_arena = 0, t_pop_arena = 0;
    long long t_alloc_task = 0, t_ctor_save = 0, t_ctor_load = 0, t_alloc_task2 = 0;
    long long t_free_task2 = 0, t_dtor_load = 0, t_dtor_save = 0, t_free_task = 0;

    // Push arenas once (same as serde kernel)
    long long tc = clock64();
    const size_t arena_bytes = static_cast<size_t>(4096) *
                               static_cast<size_t>(total_tasks + 8);
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
      chi::priv::vector<char> save_buf;
      save_buf.reserve(256);
      chi::DefaultSaveArchive ar_save(chi::LocalMsgType::kSerializeIn, save_buf);
      t_ctor_save += clock64() - tc;

      // --- (skip serialize) ---

      // --- Construct LoadArchive on stack ---
      tc = clock64();
      chi::DefaultLoadArchive ar_load(save_buf);
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
      printf("  DefaultSaveArchive(): %llu  (%llu/task)\n",
             (unsigned long long)t_ctor_save,
             (unsigned long long)(t_ctor_save / total_tasks));
      printf("  DefaultLoadArchive(): %llu  (%llu/task)\n",
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
 * GPU SendGpu+RecvGpu client-side breakdown kernel.
 *
 * Mirrors the exact latency test path (GpuSubmitTask) but without the
 * orchestrator round-trip. Each step is timed individually so we can
 * see where the 122us is spent:
 *
 *   SEND side (mirrors SendGpu):
 *   1. NewTask<GpuSubmitTask>          — allocate task
 *   2. GetWarpManager + reset save_ar  — prepare warp archives
 *   3. SerializeIn (warp-parallel)     — serialize task fields
 *   4. ShmTransport::SendDevice        — frame + copy to copy_space
 *   5. threadfence + queue push        — enqueue for orchestrator
 *
 *   RECV side (mirrors RecvGpu):
 *   6. Simulate FUTURE_COMPLETE        — mark done (no real orchestrator)
 *   7. ShmTransport::RecvDevice        — unframe from copy_space
 *   8. SerializeOut (warp-parallel)    — deserialize output fields
 *   9. DelTask + cleanup               — free task memory
 *
 * Standalone serde pipeline test using the actual GpuRuntime container.
 */
__global__ void gpu_bench_serde_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done,
    chi::u32 total_threads) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  using GpuSubmitTask = chimaera::MOD_NAME::GpuSubmitTask;
  auto *ipc = CHI_IPC;
  chi::u32 lane = chi::gpu::IpcManager::GetLaneId();
  constexpr size_t kCopySpaceSize = 1024;

  // Construct the GpuRuntime container (sets function pointer table)
  chimaera::MOD_NAME::GpuRuntime container;
  container.pool_id_ = pool_id;

  // Allocate client task (with co-located FutureShm + copy_space)
  hipc::FullPtr<GpuSubmitTask> client_task;
  chi::FutureShm *fshm = nullptr;
  if (lane == 0) {
    client_task = ipc->NewTask<GpuSubmitTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
        (chi::u32)0, (chi::u32)7);
    fshm = reinterpret_cast<chi::FutureShm *>(
        chi::gpu::IpcManager::GetTaskFutureShm(client_task.ptr_));
  }
  unsigned long long task_ull = reinterpret_cast<unsigned long long>(client_task.ptr_);
  task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);

  // Inline buffer + archive helpers (replacing WarpIpcManager)
  hshm::priv::wrap_vector buffer;
  chi::GpuSaveTaskArchive save_ar(chi::LocalMsgType::kSerializeIn, buffer);
  chi::GpuLoadTaskArchive load_ar(buffer);

  auto BindSave = [&](char *copy_space, chi::LocalMsgType msg_type) {
    hipc::FullPtr<char> fp;
    fp.ptr_ = copy_space;
    fp.shm_.alloc_id_.SetNull();
    fp.shm_.off_ = reinterpret_cast<size_t>(fp.ptr_);
    buffer.set(fp, kCopySpaceSize);
    save_ar.Reset(msg_type);
  };
  auto BindLoad = [&](char *copy_space, size_t data_size) {
    hipc::FullPtr<char> fp;
    fp.ptr_ = copy_space;
    fp.shm_.alloc_id_.SetNull();
    fp.shm_.off_ = reinterpret_cast<size_t>(fp.ptr_);
    buffer.set(fp, data_size);
    buffer.resize(data_size);
    load_ar.Reset(chi::LocalMsgType::kSerializeIn);
  };

  int errors = 0;
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto *t = reinterpret_cast<GpuSubmitTask *>(task_ull);

    // === CLIENT: Serialize input into copy_space (mirrors SendGpu) ===
    if (lane == 0) {
      fshm->Reset(client_task->pool_id_, client_task->method_);
      fshm->flags_.SetBits(chi::FutureShm::FUTURE_COPY_FROM_CLIENT |
                            chi::FutureShm::FUTURE_DEVICE_SCOPE);
      fshm->input_.copy_space_size_.store(kCopySpaceSize);
      fshm->output_.copy_space_size_.store(kCopySpaceSize);
      client_task->test_value_ = 7;
      client_task->gpu_id_ = 0;
      client_task->result_value_ = 0;
      BindSave(fshm->copy_space, chi::LocalMsgType::kSerializeIn);
    }
    __syncwarp();
    if (t) {
      save_ar.SetWarpConverged(true);
      t->SerializeIn(save_ar);
      __syncwarp();
      if (lane == 0) save_ar.SetWarpConverged(false);
    }
    __syncwarp();
    if (lane == 0) {
      hipc::threadfence();
      fshm->input_.total_written_.store(save_ar.GetSerializedSize());
    }
    __syncwarp();

    // === ORCHESTRATOR: AllocLoadDeser + Run + SaveTask (mirrors runtime) ===
    size_t data_size = 0;
    int valid = 0;
    if (lane == 0) {
      size_t tw = fshm->input_.total_written_.load_device();
      size_t cs = fshm->input_.copy_space_size_.load_device();
      if (tw > 0 && tw <= cs) { data_size = tw; valid = 1; }
    }
    valid = __shfl_sync(0xFFFFFFFF, valid, 0);
    hipc::threadfence();

    hipc::FullPtr<chi::Task> task_ptr = hipc::FullPtr<chi::Task>::GetNull();
    if (lane == 0 && valid) {
      BindLoad(fshm->copy_space, data_size);
      task_ptr = container.LocalAllocLoadDeser(
          fshm->method_id_, load_ar);
    }
    __syncwarp();

    // Run
    if (lane == 0 && !task_ptr.IsNull()) {
      auto *ot = task_ptr.template Cast<GpuSubmitTask>().ptr_;
      ot->result_value_ = (ot->test_value_ * 3) + ot->gpu_id_;
    }
    __syncwarp();

    // SaveTask + set total_written
    if (lane == 0 && !task_ptr.IsNull()) {
      chi::u32 method = fshm->method_id_;
      BindSave(fshm->copy_space, chi::LocalMsgType::kSerializeOut);
      container.LocalSaveTask(method, save_ar, task_ptr);
      hipc::threadfence();
      fshm->output_.total_written_.store(save_ar.GetSerializedSize());
      container.LocalDestroyTask(method, task_ptr);
      ipc->DelTask(task_ptr);
    }
    __syncwarp();

    // === CLIENT: Read output (mirrors RecvGpu) ===
    if (lane == 0) {
      size_t output_tw = fshm->output_.total_written_.load_device();
      hipc::threadfence();
      BindLoad(fshm->copy_space, output_tw);
      load_ar.SetMsgType(chi::LocalMsgType::kSerializeOut);
      client_task->SerializeOut(load_ar);
      chi::u32 expected = 7 * 3 + 0;
      if (client_task->result_value_ != expected) {
        printf("[SERDE FAIL] task %u: expected=%u got=%u rc=%d\n",
               i, expected, (unsigned)client_task->result_value_,
               (int)client_task->return_code_);
        ++errors;
      }
    }
    __syncwarp();
  }

  if (lane == 0) { ipc->DelTask(client_task); }
  __syncwarp();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (errors == 0)
      printf("[SERDE OK] %u/%u tasks passed\n", total_tasks, total_tasks);
    else
      printf("[SERDE FAIL] %d/%u tasks failed\n", errors, total_tasks);
  }

  __threadfence();
  int prev = atomicAdd(d_done, 1);
  if (prev == static_cast<int>(total_threads) - 1) {
    __threadfence_system();
  }
}

}  // namespace chi_bench

/**
 * Single Gray-Scott stencil iteration kernel.
 * 5-point Laplacian, feed/kill reaction on NxN grid.
 * Launched once per iteration from host for accurate timing.
 */
__global__ void gpu_bench_grayscott_kernel(
    float *U, float *V, float *U2, float *V2,
    int N) {
  const float Du = 0.16f, Dv = 0.08f;
  const float F = 0.06f, k = 0.062f;
  const float dt = 1.0f;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  int cells = N * N;

  for (int idx = tid; idx < cells; idx += total_threads) {
    int x = idx % N;
    int y = idx / N;

    int xp = (x + 1) % N, xm = (x - 1 + N) % N;
    int yp = (y + 1) % N, ym = (y - 1 + N) % N;

    float u = U[idx];
    float v = V[idx];
    float lap_u = U[ym * N + x] + U[yp * N + x] +
                  U[y * N + xm] + U[y * N + xp] - 4.0f * u;
    float lap_v = V[ym * N + x] + V[yp * N + x] +
                  V[y * N + xm] + V[y * N + xp] - 4.0f * v;

    float uvv = u * v * v;
    U2[idx] = u + dt * (Du * lap_u - uvv + F * (1.0f - u));
    V2[idx] = v + dt * (Dv * lap_v + uvv - (F + k) * v);
  }
}

/**
 * Warp-to-warp transfer benchmark.
 *
 * Two warps in one block. Warp 0 (sender) writes data to a shared
 * global buffer, does threadfence, sets a flag. Warp 1 (receiver)
 * spins on the flag, reads the data. Measures raw round-trip: send +
 * signal + poll + receive + signal back.
 *
 * Layout: [256B payload] [flag_ready] [flag_done] [results]
 */
__global__ void gpu_bench_warp_xfer_kernel(
    char *buffer, int total_iters, int payload_bytes) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  // Shared flag in global memory (after payload)
  volatile unsigned int *flag_ready =
      reinterpret_cast<volatile unsigned int *>(buffer + payload_bytes);
  volatile unsigned int *flag_done = flag_ready + 1;
  // Result storage
  long long *results = reinterpret_cast<long long *>(
      const_cast<unsigned int *>(flag_done) + 1);

  // Init flags
  if (threadIdx.x == 0) {
    *flag_ready = 0;
    *flag_done = 0;
  }
  __syncthreads();

  long long t_send = 0, t_recv = 0, t_roundtrip = 0;
  long long tc;

  for (int i = 0; i < total_iters; ++i) {
    if (warp_id == 0) {
      // === SENDER (warp 0) ===
      tc = clock64();

      // Write payload (warp-cooperative: each lane writes 4/8 bytes)
      int bytes_per_lane = payload_bytes / 32;
      if (bytes_per_lane >= 4 && lane < payload_bytes / bytes_per_lane) {
        for (int b = 0; b < bytes_per_lane; b += 4) {
          int off = lane * bytes_per_lane + b;
          *reinterpret_cast<unsigned int *>(buffer + off) = (unsigned int)(i + off);
        }
      }
      __threadfence();

      // Signal ready
      if (lane == 0) {
        atomicExch(const_cast<unsigned int *>(flag_ready), i + 1);
      }

      if (lane == 0) t_send += clock64() - tc;

      // Wait for receiver to ack
      if (lane == 0) {
        tc = clock64();
        while (atomicAdd(const_cast<unsigned int *>(flag_done), 0) != (unsigned)(i + 1)) {}
        t_roundtrip += clock64() - tc;
      }
      __syncwarp();

    } else if (warp_id == 1) {
      // === RECEIVER (warp 1) ===

      // Spin on flag
      if (lane == 0) {
        while (atomicAdd(const_cast<unsigned int *>(flag_ready), 0) != (unsigned)(i + 1)) {}
      }
      __syncwarp();
      __threadfence();

      tc = clock64();
      // Read payload
      int bytes_per_lane = payload_bytes / 32;
      unsigned int checksum = 0;
      if (bytes_per_lane >= 4 && lane < payload_bytes / bytes_per_lane) {
        for (int b = 0; b < bytes_per_lane; b += 4) {
          int off = lane * bytes_per_lane + b;
          checksum += *reinterpret_cast<volatile unsigned int *>(buffer + off);
        }
      }
      // Prevent dead-code elimination
      if (checksum == 0xDEADBEEF && lane == 0) printf("x");

      if (lane == 0) t_recv += clock64() - tc;

      // Ack done
      __threadfence();
      if (lane == 0) {
        atomicExch(const_cast<unsigned int *>(flag_done), i + 1);
      }
      __syncwarp();
    }
  }

  // Store results
  if (warp_id == 0 && lane == 0) {
    results[0] = t_send;
    results[1] = t_roundtrip;
  }
  if (warp_id == 1 && lane == 0) {
    results[2] = t_recv;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    printf("  %4d bytes:  send=%lld (%lld/iter)  wait_ack=%lld (%lld/iter)  "
           "recv=%lld (%lld/iter)  total=%lld (%lld/iter)\n",
           payload_bytes,
           results[0], results[0] / total_iters,
           results[1], results[1] / total_iters,
           results[2], results[2] / total_iters,
           results[0] + results[1] + results[2],
           (results[0] + results[1] + results[2]) / total_iters);
  }
}

/**
 * Run warp-to-warp transfer benchmark at various payload sizes.
 */
extern "C" int run_gpu_bench_warp_xfer(float *out_elapsed_ms) {
  const int payload_sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
  const int num_sizes = sizeof(payload_sizes) / sizeof(payload_sizes[0]);
  const int iters = 10000;

  // Allocate buffer large enough for biggest payload + flags + results
  size_t buf_size = 4096 + 64 + 3 * sizeof(long long);
  char *d_buf;
  cudaMalloc(&d_buf, buf_size);
  cudaMemset(d_buf, 0, buf_size);

  printf("\n=== Warp-to-Warp Transfer Latency (device-scope atomics) ===\n");
  printf("  2 warps in 1 block, %d iterations\n", iters);

  for (int s = 0; s < num_sizes; ++s) {
    cudaMemset(d_buf, 0, buf_size);
    // Launch 2 warps = 64 threads in 1 block
    gpu_bench_warp_xfer_kernel<<<1, 64>>>(d_buf, iters, payload_sizes[s]);
    cudaDeviceSynchronize();
  }

  *out_elapsed_ms = 0;
  cudaFree(d_buf);
  return 0;
}

/**
 * Global memory copy microbenchmark.
 * Compares single-thread vs 32-thread (warp) memcpy at various sizes.
 * src and dst are in global memory, separated to avoid aliasing.
 */
__global__ void gpu_bench_memcpy_kernel(
    char *src, char *dst, int total_iters, long long *results) {
  int lane = threadIdx.x % 32;
  const int sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
  constexpr int kNumSizes = 7;

  for (int s = 0; s < kNumSizes; ++s) {
    int sz = sizes[s];
    long long t_single = 0, t_warp = 0;
    long long tc;

    // --- Single-thread copy (lane 0 only, byte-by-byte) ---
    if (lane == 0) {
      for (int i = 0; i < total_iters; ++i) {
        tc = clock64();
        for (int b = 0; b < sz; b += 4) {
          *reinterpret_cast<unsigned int *>(dst + b) =
              *reinterpret_cast<unsigned int *>(src + b);
        }
        __threadfence();
        t_single += clock64() - tc;
      }
    }
    __syncwarp();

    // --- Warp-cooperative copy (32 lanes, 4B each) ---
    for (int i = 0; i < total_iters; ++i) {
      tc = clock64();
      int bytes_per_lane = sz / 32;
      if (bytes_per_lane >= 4) {
        for (int b = 0; b < bytes_per_lane; b += 4) {
          int off = lane * bytes_per_lane + b;
          *reinterpret_cast<unsigned int *>(dst + off) =
              *reinterpret_cast<unsigned int *>(src + off);
        }
      } else if (lane < sz / 4) {
        // Fewer than 1 byte per lane — only first sz/4 lanes write 4B each
        int off = lane * 4;
        *reinterpret_cast<unsigned int *>(dst + off) =
            *reinterpret_cast<unsigned int *>(src + off);
      }
      __threadfence();
      if (lane == 0) t_warp += clock64() - tc;
    }
    __syncwarp();

    if (lane == 0) {
      results[s * 2] = t_single;
      results[s * 2 + 1] = t_warp;
    }
  }

  if (lane == 0) {
    printf("\n=== Global Memory Copy Latency ===\n");
    printf("  %6s  %14s  %14s  %10s\n",
           "Bytes", "1-thread(cyc)", "32-thread(cyc)", "Speedup");
    for (int s = 0; s < kNumSizes; ++s) {
      long long ts = results[s * 2] / total_iters;
      long long tw = results[s * 2 + 1] / total_iters;
      printf("  %6d  %14lld  %14lld  %9.1fx\n",
             sizes[s], ts, tw, (double)ts / (double)(tw > 0 ? tw : 1));
    }
  }
}

extern "C" int run_gpu_bench_memcpy(float *out_elapsed_ms) {
  const int iters = 10000;
  size_t buf_size = 8192;  // src + dst side by side
  char *d_buf;
  cudaMalloc(&d_buf, buf_size * 2);

  // Init src with pattern
  std::vector<char> h_src(buf_size, 0x42);
  cudaMemcpy(d_buf, h_src.data(), buf_size, cudaMemcpyHostToDevice);
  cudaMemset(d_buf + buf_size, 0, buf_size);

  long long *d_results;
  cudaMalloc(&d_results, 7 * 2 * sizeof(long long));
  cudaMemset(d_results, 0, 7 * 2 * sizeof(long long));

  // 1 warp = 32 threads
  gpu_bench_memcpy_kernel<<<1, 32>>>(d_buf, d_buf + buf_size, iters, d_results);
  cudaDeviceSynchronize();

  *out_elapsed_ms = 0;
  cudaFree(d_buf);
  cudaFree(d_results);
  return 0;
}

/**
 * Run zero-copy direct dispatch benchmark.
 */
extern "C" int run_gpu_bench_zerocopy(chi::u32 total_tasks,
                                       float *out_elapsed_ms) {
  chi_bench::ZeroCopySlot *d_slot;
  cudaMalloc(&d_slot, sizeof(chi_bench::ZeroCopySlot));
  cudaMemset(d_slot, 0, sizeof(chi_bench::ZeroCopySlot));

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  // 2 warps = 64 threads, 1 block
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  chi_bench::gpu_bench_zerocopy_kernel<<<1, 64>>>(d_slot, 1, d_done);
  cudaDeviceSynchronize();
  *d_done = 0;
  cudaMemset(d_slot, 0, sizeof(chi_bench::ZeroCopySlot));

  cudaEventRecord(start);
  chi_bench::gpu_bench_zerocopy_kernel<<<1, 64>>>(d_slot, total_tasks, d_done);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  *out_elapsed_ms = ms;

  printf("  Throughput:          %.0f tasks/sec\n", total_tasks / (ms / 1000.0f));
  printf("  Avg latency:         %.3f us/task\n", (ms * 1000.0f) / total_tasks);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_slot);
  cudaFreeHost(d_done);
  return 0;
}

/**
 * Run Gray-Scott stencil benchmark for various grid sizes.
 */
extern "C" int run_gpu_bench_grayscott(float *out_elapsed_ms) {
  const int grid_sizes[] = {32, 64, 128, 256, 512, 1024};
  const int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
  const int iters = 100;

  printf("\n=== Gray-Scott Single Iteration Latency ===\n");
  printf("  %8s  %12s  %12s  %12s\n", "Grid", "Total(us)", "Per-iter(us)", "Cells");

  for (int s = 0; s < num_sizes; ++s) {
    int N = grid_sizes[s];
    int cells = N * N;
    size_t bytes = cells * sizeof(float);

    float *d_U, *d_V, *d_U2, *d_V2;
    cudaMalloc(&d_U, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_U2, bytes);
    cudaMalloc(&d_V2, bytes);

    // Init: U=1 everywhere, V=0 with small seed in center
    std::vector<float> h_U(cells, 1.0f), h_V(cells, 0.0f);
    int cx = N / 2, cy = N / 2, r = N / 10;
    for (int y = cy - r; y <= cy + r; ++y)
      for (int x = cx - r; x <= cx + r; ++x)
        if (x >= 0 && x < N && y >= 0 && y < N) {
          h_U[y * N + x] = 0.5f;
          h_V[y * N + x] = 0.25f;
        }
    cudaMemcpy(d_U, h_U.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_U2, 0, bytes);
    cudaMemset(d_V2, 0, bytes);

    int threads = 256;
    int blocks = (cells + threads - 1) / threads;
    if (blocks > 256) blocks = 256;

    // Warmup
    gpu_bench_grayscott_kernel<<<blocks, threads>>>(
        d_U, d_V, d_U2, d_V2, N);
    cudaDeviceSynchronize();

    // Timed: launch one kernel per iteration, swap buffers on host
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *pU = d_U, *pV = d_V, *pU2 = d_U2, *pV2 = d_V2;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      gpu_bench_grayscott_kernel<<<blocks, threads>>>(
          pU, pV, pU2, pV2, N);
      // Swap for next iteration
      float *tmp;
      tmp = pU; pU = pU2; pU2 = tmp;
      tmp = pV; pV = pV2; pV2 = tmp;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float per_iter_us = (ms * 1000.0f) / iters;

    printf("  %5dx%-3d  %12.3f  %12.3f  %12d\n",
           N, N, ms * 1000.0f, per_iter_us, cells);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_U); cudaFree(d_V); cudaFree(d_U2); cudaFree(d_V2);
  }

  *out_elapsed_ms = 0;
  return 0;
}

namespace chi_bench {

/**
 * Isolated BuddyAllocator microbenchmark.
 *
 * Measures alloc+free cycles for the exact same allocation size
 * that the orchestrator uses: sizeof(GpuSubmitTask) + kRunContextExtra.
 * Single thread (lane 0 only), no orchestrator, no queues.
 */
__global__ void gpu_bench_buddy_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    int *d_done,
    chi::u32 total_threads) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);
  using GpuSubmitTask = chimaera::MOD_NAME::GpuSubmitTask;

  if (threadIdx.x != 0 || blockIdx.x != 0) {
    __threadfence();
    atomicAdd(d_done, 1);
    return;
  }

  auto *alloc = CHI_IPC->gpu_alloc_;
  if (!alloc) {
    printf("[BUDDY] alloc null\n");
    atomicAdd(d_done, 1);
    return;
  }

  // Exact sizes used by the orchestrator: Task + FutureShm
  constexpr size_t kTaskExecSize =
      sizeof(GpuSubmitTask) + sizeof(chi::FutureShm);
  constexpr size_t kSmallSize = 64;   // small alloc for comparison

  // Warm up
  {
    auto w = alloc->template AllocateObjs<char>(kTaskExecSize);
    if (!w.IsNull()) alloc->Free(w);
  }

  long long t_alloc_large = 0, t_free_large = 0;
  long long t_alloc_small = 0, t_free_small = 0;
  long long t_alloc_arena = 0;
  long long tc;

  // --- Test 1: alloc+free kTaskExecSize ---
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    tc = clock64();
    auto fp = alloc->template AllocateObjs<char>(kTaskExecSize);
    t_alloc_large += clock64() - tc;

    tc = clock64();
    if (!fp.IsNull()) alloc->Free(fp);
    t_free_large += clock64() - tc;
  }

  // --- Test 2: alloc+free kSmallSize (64B) ---
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    tc = clock64();
    auto fp = alloc->template AllocateObjs<char>(kSmallSize);
    t_alloc_small += clock64() - tc;

    tc = clock64();
    if (!fp.IsNull()) alloc->Free(fp);
    t_free_small += clock64() - tc;
  }

  // --- Test 3: arena bump only (no free, fresh arena) ---
  {
    const size_t arena_bytes = static_cast<size_t>(kTaskExecSize + 256) *
                               static_cast<size_t>(total_tasks + 4);
    auto arena = CHI_IPC->PushArena(arena_bytes);
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      tc = clock64();
      auto fp = alloc->template AllocateObjs<char>(kTaskExecSize);
      t_alloc_arena += clock64() - tc;
      // No free — pure bump
    }
    arena.Release();
  }

  printf("=== BuddyAllocator Microbenchmark (%u iters) ===\n", total_tasks);
  printf("  Alloc size (large):  %llu bytes (kTaskExecSize)\n",
         (unsigned long long)kTaskExecSize);
  printf("  Alloc size (small):  %llu bytes\n",
         (unsigned long long)kSmallSize);
  printf("  Large alloc:         %lld  (%lld/iter)\n",
         t_alloc_large, t_alloc_large / total_tasks);
  printf("  Large free:          %lld  (%lld/iter)\n",
         t_free_large, t_free_large / total_tasks);
  printf("  Small alloc:         %lld  (%lld/iter)\n",
         t_alloc_small, t_alloc_small / total_tasks);
  printf("  Small free:          %lld  (%lld/iter)\n",
         t_free_small, t_free_small / total_tasks);
  printf("  Arena bump (large):  %lld  (%lld/iter)\n",
         t_alloc_arena, t_alloc_arena / total_tasks);

  __threadfence();
  atomicAdd(d_done, 1);
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
 * Query and print kernel resource usage (registers, shared memory, max threads).
 */
static void PrintKernelInfo(const char *name, const void *func,
                             chi::u32 blocks, chi::u32 threads) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, func) == cudaSuccess) {
    int max_threads = (attr.numRegs > 0) ? (65536 / attr.numRegs) : 1024;
    max_threads = (max_threads / 32) * 32;
    if (max_threads > 1024) max_threads = 1024;
    HIPRINT("Kernel {}:", name);
    HIPRINT("  Registers/thread:    {}", attr.numRegs);
    HIPRINT("  Shared memory:       {} bytes", attr.sharedSizeBytes);
    HIPRINT("  Max threads/block:   {} (register-limited)", max_threads);
    HIPRINT("  Launch config:       {}b x {}t", blocks, threads);
  }
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
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Use the orchestrator's shared allocator backend
  chi::IpcManagerGpu gpu_info =
      CHI_CPU_IPC->GetGpuIpcManager()->CreateGpuAllocator(0, 0);

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  // Pause orchestrator FIRST — cudaMallocHost and cudaStreamCreate
  // are device-synchronizing and deadlock with the persistent CDP kernel.
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;
  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();  // Clear any sticky CUDA errors

  // Re-fetch gpu_info after pause (queue may have been rebuilt)
  gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->CreateGpuAllocator(0, 0);

  PrintKernelInfo("gpu_bench_client_kernel",
                  (const void *)chi_bench::gpu_bench_client_kernel,
                  client_blocks, client_threads);
  chi_bench::gpu_bench_client_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, d_done, total_warps);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  // Resume orchestrator AFTER client launch — both run concurrently
  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
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
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;  // -4 = timeout
}

/**
 * Run the GPU runtime coroutine benchmark.
 *
 * Same structure as run_gpu_bench_latency but launches gpu_bench_coroutine_kernel
 * which uses SubtaskTest (dispatches subtask) instead of leaf GpuSubmit.
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
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(102, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  void *stream = hshm::GpuApi::CreateStream();

  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();
  cudaGetLastError();

  PrintKernelInfo("gpu_bench_coroutine_kernel",
                  (const void *)chi_bench::gpu_bench_coroutine_kernel,
                  client_blocks, client_threads);
  chi_bench::gpu_bench_coroutine_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, subtasks, d_done, total_warps);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start)
                          .count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

namespace chi_bench {

/**
 * Multi-warp parallelism benchmark kernel.
 *
 * Uses the Client API with PoolQuery::Local(parallelism) to submit
 * GpuSubmit tasks that span multiple warps (parallelism > 32).
 * Measures round-trip latency when the orchestrator must coordinate
 * across warps to execute a single task.
 */
__global__ void gpu_bench_parallel_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    chi::u32 parallelism,
    int *d_done,
    chi::u32 total_warps) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  auto *ipc = CHI_IPC;
  chi::u32 errors = 0;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        chi::CreateTaskId(), pool_id,
        chi::PoolQuery::Local(parallelism), (chi::u32)0, (chi::u32)7);
    auto future = ipc->Send(task);
    future.Wait();

    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      if (!future.IsNull() && task->result_value_ != 21) {
        if (errors == 0) {
          printf("[PARALLEL ERROR] blk=%u task %u: expected=21, got=%u\n",
                 blockIdx.x, i, (unsigned)task->result_value_);
        }
        ++errors;
      }
    }
  }

  if (chi::gpu::IpcManager::IsWarpScheduler() && errors > 0) {
    printf("[PARALLEL FAIL] blk=%u %u/%u wrong results\n",
           blockIdx.x, errors, total_tasks);
  }

  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    __threadfence_system();
    atomicAdd(d_done, 1);
  }
}

}  // namespace chi_bench (close temporarily to define extern "C")

/**
 * Run the multi-warp parallelism benchmark.
 *
 * Same setup as run_gpu_bench_latency but the client kernel uses
 * PoolQuery::Local(parallelism) with parallelism > 32.
 */
extern "C" int run_gpu_bench_parallel(
    chi::PoolId pool_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    chi::u32 parallelism,
    float *out_elapsed_ms) {
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(103, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  void *stream = hshm::GpuApi::CreateStream();

  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();
  cudaGetLastError();

  gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  PrintKernelInfo("gpu_bench_parallel_kernel",
                  (const void *)chi_bench::gpu_bench_parallel_kernel,
                  client_blocks, client_threads);
  chi_bench::gpu_bench_parallel_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, client_blocks, total_tasks, parallelism,
      d_done, total_warps);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

/**
 * GPU→bdev roundtrip benchmark kernel.
 *
 * Each warp's lane 0 calls bdev::Client::AsyncAllocateBlocks to request
 * a block allocation from the bdev container via the GPU orchestrator.
 * This exercises the full GPU client → IPC queue → orchestrator → bdev
 * runtime pipeline.
 */
namespace chi_bench {

__global__ void gpu_bench_bdev_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId bdev_pool_id,
    chi::u32 num_blocks,
    chi::u32 total_tasks,
    chi::u64 alloc_size,
    int *d_done,
    chi::u32 total_warps) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  auto *ipc = CHI_IPC;
  chi::u32 errors = 0;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto task = ipc->NewTask<chimaera::bdev::AllocateBlocksTask>(
        chi::CreateTaskId(), bdev_pool_id,
        chi::PoolQuery::Local(), alloc_size);
    auto future = ipc->Send(task);
    future.Wait();

    if (chi::gpu::IpcManager::IsWarpScheduler()) {
      if (future.IsNull() || task->return_code_ != 0) {
        if (errors == 0) {
          printf("[BDEV ERROR] blk=%u task %u: alloc failed (rc=%d)\n",
                 blockIdx.x, i,
                 future.IsNull() ? -999 : (int)task->return_code_);
        }
        ++errors;
      }
    }
  }

  if (chi::gpu::IpcManager::IsWarpScheduler() && errors > 0) {
    printf("[BDEV FAIL] blk=%u %u/%u allocations failed\n",
           blockIdx.x, errors, total_tasks);
  }

  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    __threadfence_system();
    atomicAdd(d_done, 1);
  }
}

}  // namespace chi_bench

extern "C" int run_gpu_bench_bdev(
    chi::PoolId bdev_pool_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    chi::u64 alloc_size,
    float *out_elapsed_ms) {
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(108, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  void *stream = hshm::GpuApi::CreateStream();

  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();
  cudaGetLastError();

  gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  PrintKernelInfo("gpu_bench_bdev_kernel",
                  (const void *)chi_bench::gpu_bench_bdev_kernel,
                  client_blocks, client_threads);
  chi_bench::gpu_bench_bdev_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, bdev_pool_id, client_blocks, total_tasks, alloc_size,
      d_done, total_warps);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  cudaFreeHost(d_done);
  hshm::GpuApi::DestroyStream(stream);

  return completed ? 0 : -4;
}

// ============================================================================
// Full bdev GPU→GPU test: AllocateBlocks → Write → Read → Verify
// ============================================================================

namespace chi_bench {

/**
 * Full bdev round-trip: allocate blocks, write data, read it back, verify.
 * All from GPU kernel → gpu2gpu queue → orchestrator → bdev GpuRuntime.
 */
__global__ void gpu_bench_bdev_full_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId bdev_pool_id,
    chi::u32 num_blocks,
    chi::u64 io_size,
    int *d_result) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  auto *ipc = CHI_IPC;
  int rc = 0;

  // Step 1: AllocateBlocks
  if (threadIdx.x == 0) {
    printf("[BDEV-FULL] Step 1: AllocateBlocks(%llu bytes)\n",
           (unsigned long long)io_size);
  }
  auto alloc_task = ipc->NewTask<chimaera::bdev::AllocateBlocksTask>(
      chi::CreateTaskId(), bdev_pool_id,
      chi::PoolQuery::Local(), io_size);
  auto alloc_future = ipc->Send(alloc_task);
  alloc_future.Wait();

  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    if (alloc_future.IsNull() || alloc_task->return_code_ != 0) {
      printf("[BDEV-FULL] FAIL: AllocateBlocks rc=%d null=%d\n",
             alloc_future.IsNull() ? -999 : (int)alloc_task->return_code_,
             (int)alloc_future.IsNull());
      rc = -1;
    } else if (alloc_task->blocks_.size() == 0) {
      printf("[BDEV-FULL] FAIL: AllocateBlocks returned 0 blocks (rc=%d)\n",
             (int)alloc_task->return_code_);
      rc = -1;
    } else {
      printf("[BDEV-FULL] AllocateBlocks OK, %u blocks, offset=%llu size=%llu\n",
             (unsigned)alloc_task->blocks_.size(),
             (unsigned long long)alloc_task->blocks_[0].offset_,
             (unsigned long long)alloc_task->blocks_[0].size_);
    }
  }
  // Broadcast rc to all lanes
  rc = __shfl_sync(0xFFFFFFFF, rc, 0);
  if (rc != 0) { if (threadIdx.x == 0) *d_result = rc; return; }

  // Get the allocated blocks (lane 0 only has the valid future)
  // We need to copy blocks to a local variable accessible by all lanes
  chi::priv::vector<chimaera::bdev::Block> blocks;
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    blocks = alloc_task->blocks_;
  }

  // Step 2: Write — fill a buffer with known pattern and write it
  if (threadIdx.x == 0) {
    printf("[BDEV-FULL] Step 2: Write %llu bytes\n",
           (unsigned long long)io_size);
  }

  // Allocate write buffer from GPU allocator
  auto write_buf = CHI_IPC->AllocateBuffer(io_size);
  if (write_buf.IsNull()) {
    if (threadIdx.x == 0) {
      printf("[BDEV-FULL] FAIL: Could not allocate write buffer\n");
      *d_result = -2;
    }
    return;
  }

  // Fill with pattern: byte[i] = (i & 0xFF)
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    char *wbuf = write_buf.ptr_;
    for (chi::u64 i = 0; i < io_size; ++i) {
      wbuf[i] = static_cast<char>(i & 0xFF);
    }
  }
  __syncwarp();

  hipc::ShmPtr<> write_data_shm;
  write_data_shm.off_ = write_buf.shm_.off_.load();
  write_data_shm.alloc_id_ = write_buf.shm_.alloc_id_;
  auto write_task = ipc->NewTask<chimaera::bdev::WriteTask>(
      chi::CreateTaskId(), bdev_pool_id,
      chi::PoolQuery::Local(), blocks, write_data_shm, io_size);
  auto write_future = ipc->Send(write_task);
  write_future.Wait();

  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    if (write_future.IsNull() || write_task->return_code_ != 0) {
      printf("[BDEV-FULL] FAIL: Write rc=%d\n",
             write_future.IsNull() ? -999 : (int)write_task->return_code_);
      rc = -3;
    } else {
      printf("[BDEV-FULL] Write OK\n");
    }
  }
  rc = __shfl_sync(0xFFFFFFFF, rc, 0);
  if (rc != 0) { if (threadIdx.x == 0) *d_result = rc; return; }

  // Step 3: Read into a fresh buffer
  if (threadIdx.x == 0) {
    printf("[BDEV-FULL] Step 3: Read %llu bytes\n",
           (unsigned long long)io_size);
  }

  auto read_buf = CHI_IPC->AllocateBuffer(io_size);
  if (read_buf.IsNull()) {
    if (threadIdx.x == 0) {
      printf("[BDEV-FULL] FAIL: Could not allocate read buffer\n");
      *d_result = -4;
    }
    return;
  }

  // Zero the read buffer
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    memset(read_buf.ptr_, 0, io_size);
  }
  __syncwarp();

  hipc::ShmPtr<> read_data_shm;
  read_data_shm.off_ = read_buf.shm_.off_.load();
  read_data_shm.alloc_id_ = read_buf.shm_.alloc_id_;
  auto read_task = ipc->NewTask<chimaera::bdev::ReadTask>(
      chi::CreateTaskId(), bdev_pool_id,
      chi::PoolQuery::Local(), blocks, read_data_shm, io_size);
  auto read_future = ipc->Send(read_task);
  read_future.Wait();

  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    if (read_future.IsNull() || read_task->return_code_ != 0) {
      printf("[BDEV-FULL] FAIL: Read rc=%d\n",
             read_future.IsNull() ? -999 : (int)read_task->return_code_);
      rc = -5;
    } else {
      printf("[BDEV-FULL] Read OK\n");
    }
  }
  rc = __shfl_sync(0xFFFFFFFF, rc, 0);
  if (rc != 0) { if (threadIdx.x == 0) *d_result = rc; return; }

  // Step 4: Verify data
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    char *rbuf = read_buf.ptr_;
    char *wbuf = write_buf.ptr_;
    chi::u32 mismatches = 0;
    for (chi::u64 i = 0; i < io_size; ++i) {
      if (rbuf[i] != wbuf[i]) {
        if (mismatches == 0) {
          printf("[BDEV-FULL] MISMATCH at byte %llu: wrote 0x%02x read 0x%02x\n",
                 (unsigned long long)i, (unsigned char)wbuf[i],
                 (unsigned char)rbuf[i]);
        }
        ++mismatches;
      }
    }
    if (mismatches > 0) {
      printf("[BDEV-FULL] FAIL: %u / %llu bytes mismatched\n",
             mismatches, (unsigned long long)io_size);
      rc = -6;
    } else {
      printf("[BDEV-FULL] PASS: All %llu bytes verified\n",
             (unsigned long long)io_size);
      rc = 1;  // Success
    }
  }

  // Cleanup
  CHI_IPC->FreeBuffer(write_buf.shm_);
  CHI_IPC->FreeBuffer(read_buf.shm_);

  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
    __threadfence_system();
    *d_result = rc;
  }
}

}  // namespace chi_bench

extern "C" int run_gpu_bench_bdev_full(
    chi::PoolId bdev_pool_id,
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u64 io_size,
    float *out_elapsed_ms) {
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = kPerBlockBytes;

  hipc::MemoryBackendId backend_id(109, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(backend_id, gpu_backend.data_,
                                 gpu_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  int *d_result;
  cudaMallocHost(&d_result, sizeof(int));
  *d_result = 0;

  void *stream = hshm::GpuApi::CreateStream();

  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();
  cudaGetLastError();

  gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  chi_bench::gpu_bench_bdev_full_kernel<<<1, 32, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, bdev_pool_id, 1, io_size, d_result);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("bdev_full kernel launch failed: %s\n",
           cudaGetErrorString(launch_err));
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_result);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  // Poll for completion (result != 0 means done)
  constexpr int kTimeoutUs = 30000000;  // 30s
  int elapsed_us = 0;
  while (*d_result == 0 && elapsed_us < kTimeoutUs) {
    usleep(1000);
    elapsed_us += 1000;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  int result = *d_result;

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  cudaFreeHost(d_result);
  hshm::GpuApi::DestroyStream(stream);

  return result;
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

  PrintKernelInfo("gpu_bench_alloc_kernel",
                  (const void *)chi_bench::gpu_bench_alloc_kernel,
                  client_blocks, client_threads);
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
 * Run the isolated BuddyAllocator microbenchmark.
 */
extern "C" int run_gpu_bench_buddy(
    chi::PoolId pool_id,
    chi::u32 client_blocks,
    chi::u32 client_threads,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t backend_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;

  hipc::MemoryBackendId backend_id(107, 0);
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, backend_size, "", 0)) {
    return -1;
  }

  chi::IpcManagerGpu gpu_info{};
  gpu_info.backend = gpu_backend;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  chi::u32 total_threads = client_blocks * client_threads;
  void *stream = hshm::GpuApi::CreateStream();
  cudaGetLastError();

  chi_bench::gpu_bench_buddy_kernel<<<
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
                          t_end - t_start).count();
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

  PrintKernelInfo("gpu_bench_serde_kernel",
                  (const void *)chi_bench::gpu_bench_serde_kernel,
                  client_blocks, client_threads);
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

  constexpr int kTimeoutUs = 60000000;
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
 * Run the GPU alloc+free benchmark for PutBlobTask + DefaultSaveArchive.
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

  PrintKernelInfo("gpu_bench_alloc_serde_kernel",
                  (const void *)chi_bench::gpu_bench_alloc_serde_kernel,
                  client_blocks, client_threads);
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

  // --- Stack SSO string (no allocator, no heap) ---
  long long t_stack_create = 0, t_stack_destroy = 0;
  {
    volatile char sink = 0;
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      long long tc = clock64();
      BuddyString s(nullptr, kTestStr);
      t_stack_create += clock64() - tc;
      sink = s.data()[0];
      tc = clock64();
      // Destructor is a no-op for SSO strings (no heap to free)
      s.~BuddyString();
      t_stack_destroy += clock64() - tc;
    }
    (void)sink;
  }

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
  printf("  Stack(SSO)         %7llu          %7llu          %7llu\n",
         (unsigned long long)(t_stack_create / total_tasks),
         (unsigned long long)(t_stack_destroy / total_tasks),
         (unsigned long long)((t_stack_create + t_stack_destroy) / total_tasks));
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
 * All lanes call AsyncPutBlob; NewTask and SendGpu guard internally.
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

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::gpu::IpcManager::GetLaneId();

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

    // All lanes call via NewTask/Send — internally guarded by warp leader
    auto *ipc = CHI_IPC;

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

    auto task = ipc->NewTask<wrp_cte::core::PutBlobTask>(
        chi::CreateTaskId(), cte_pool_id,
        to_cpu ? chi::PoolQuery::ToLocalCpu()
               : chi::PoolQuery::Local(),
        tag_id, name_buf,
        /*offset=*/(chi::u64)0, /*size=*/slice_size,
        blob_shm, /*score=*/-1.0f,
        wrp_cte::core::Context(), /*flags=*/(chi::u32)0);
    auto future = ipc->Send(task);
    future.Wait();
  }

  __syncwarp();
  if (chi::gpu::IpcManager::IsWarpScheduler()) {
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
  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(rt_blocks, rt_threads);

  // Pause GPU orchestrator before any cudaDeviceSynchronize / GPU init.
  // The orchestrator is a persistent kernel; cudaDeviceSynchronize would block
  // forever waiting for it.
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  // --- 1. Data backend: device memory for array A ---
  hipc::MemoryBackendId data_backend_id(200, 0);
  hipc::GpuMalloc data_backend;
  // Extra space for BuddyAllocator header
  size_t data_backend_size = total_bytes + 4 * 1024 * 1024;
  if (!data_backend.shm_init(data_backend_id, data_backend_size, "", 0)) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 2. Client scratch backend (for FutureShm, serialization) ---
  constexpr size_t kPerBlockBytes = 10 * 1024 * 1024;
  size_t scratch_size = static_cast<size_t>(client_blocks) * kPerBlockBytes;
  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  if (!scratch_backend.shm_init(scratch_id, scratch_size, "", 0)) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    return -1;
  }

  // --- 3. GPU heap backend (for PartitionedAllocator) ---
  constexpr size_t kPerBlockHeapBytes = 4 * 1024 * 1024;
  size_t heap_size = static_cast<size_t>(client_blocks) * kPerBlockHeapBytes;
  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  if (!heap_backend.shm_init(heap_id, heap_size, "", 0)) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
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
  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(data_backend_id, data_backend.data_,
                                 data_backend.data_capacity_);

  // --- 6. Build GPU info and launch data placement kernel ---
  chi::IpcManagerGpu gpu_info = CHI_CPU_IPC->GetGpuIpcManager()->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;

  chi::u32 total_warps = (client_blocks * client_threads) / 32;
  if (total_warps == 0) total_warps = 1;

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  void *stream = hshm::GpuApi::CreateStream();

  // Orchestrator is already paused from function start
  cudaGetLastError();

  PrintKernelInfo("gpu_putblob_kernel",
                  (const void *)gpu_putblob_kernel,
                  client_blocks, client_threads);
  gpu_putblob_kernel<<<
      client_blocks, client_threads, 0,
      static_cast<cudaStream_t>(stream)>>>(
      gpu_info, cte_pool_id, tag_id, client_blocks,
      array_ptr,
      hipc::AllocatorId(data_backend_id.major_, data_backend_id.minor_),
      total_bytes, total_warps, to_cpu, d_done);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    cudaFreeHost(d_done);
    hshm::GpuApi::DestroyStream(stream);
    return -3;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int kTimeoutUs = 60000000;  // 60s
  bool completed = PollDone(d_done, static_cast<int>(total_warps), kTimeoutUs);

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t_end - t_start).count();
  *out_elapsed_ms = static_cast<float>(elapsed_ns / 1e6);

  hshm::GpuApi::Synchronize(stream);
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

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
/**
 * CUDA kernel launch latency benchmark.
 *
 * Measures the host-side cost of launching a trivial kernel with
 * rt_blocks x rt_threads and waiting for it to complete. Each iteration
 * does one launch + cudaEventSynchronize, timed with CUDA events.
 * This tells you the real overhead of a kernel launch round-trip.
 */
__global__ void gpu_bench_cdp_kernel() {
  // Trivial kernel — just exists to be launched and completed
}

extern "C" int run_gpu_bench_cdp(
    chi::u32 rt_blocks,
    chi::u32 rt_threads,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  // Warmup
  for (int i = 0; i < 10; ++i) {
    gpu_bench_cdp_kernel<<<rt_blocks, rt_threads>>>();
    cudaDeviceSynchronize();
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Measure total wall time for all launches
  cudaEventRecord(start);
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    gpu_bench_cdp_kernel<<<rt_blocks, rt_threads>>>();
    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start, stop);

  // Also measure per-launch with individual events
  float min_us = 1e9f, max_us = 0, sum_us = 0;
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    gpu_bench_cdp_kernel<<<rt_blocks, rt_threads>>>();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    float us = ms * 1000.0f;
    sum_us += us;
    if (us < min_us) min_us = us;
    if (us > max_us) max_us = us;
    cudaEventDestroy(s);
    cudaEventDestroy(e);
  }

  *out_elapsed_ms = total_ms;

  printf("\n=== Kernel Launch Latency (%u launches, %u blocks x %u threads) ===\n",
         total_tasks, rt_blocks, rt_threads);
  printf("  Batch wall:   %.3f ms (%.3f us/launch)\n",
         total_ms, (total_ms * 1000.0f) / total_tasks);
  printf("  Per-launch:   avg=%.3f us  min=%.3f us  max=%.3f us\n",
         sum_us / total_tasks, min_us, max_us);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

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

/**
 * Atomic queue contention benchmark using chi::ipc::mpsc_ring_buffer.
 *
 * Exercises the real MPSC ring buffer Push path used by SendGpu:
 *   1. atomic fetch_add on shared tail (claim slot)
 *   2. write entry data
 *   3. threadfence + atomic set ready flag
 *
 * A setup kernel constructs a BuddyAllocator + mpsc_ring_buffer in device
 * memory. Then the contention kernel launches N client warps that all Push
 * to the same queue. Queue depth = total_tasks * num_clients so it never
 * fills. Each warp times its pushes with clock64().
 */

__global__ void gpu_bench_queue_setup_kernel(
    hipc::MemoryBackend backend,
    chi::u32 queue_depth,
    hipc::BuddyAllocator **d_alloc_out,
    chi::ipc::mpsc_ring_buffer<chi::u32> **d_queue_out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  auto *alloc = backend.MakeAlloc<hipc::BuddyAllocator>(0);
  if (!alloc) {
    printf("[QUEUE SETUP] BuddyAllocator init failed\n");
    *d_alloc_out = nullptr;
    *d_queue_out = nullptr;
    return;
  }

  auto fp = alloc->template AllocateObjs<
      chi::ipc::mpsc_ring_buffer<chi::u32>>(1);
  if (fp.IsNull()) {
    printf("[QUEUE SETUP] ring_buffer alloc failed\n");
    *d_alloc_out = alloc;
    *d_queue_out = nullptr;
    return;
  }
  new (fp.ptr_) chi::ipc::mpsc_ring_buffer<chi::u32>(alloc, queue_depth);

  *d_alloc_out = alloc;
  *d_queue_out = fp.ptr_;
}

__global__ void gpu_bench_queue_contention_kernel(
    chi::ipc::mpsc_ring_buffer<chi::u32> *queue,
    chi::u32 total_tasks,
    chi::u32 num_clients,
    long long *d_results,
    int *d_done) {
  chi::u32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  chi::u32 lane = threadIdx.x % 32;

  // Only warp schedulers (lane 0) are producers
  if (warp_id >= num_clients) {
    __syncwarp();
    return;
  }

  long long t_push = 0;
  long long tc;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    if (lane == 0) {
      chi::u32 val = warp_id * 1000000 + i;
      tc = clock64();
      queue->Push(val);
      t_push += clock64() - tc;
    }
    __syncwarp();
  }

  // Store per-warp result
  if (lane == 0) {
    d_results[warp_id] = t_push;
  }

  __syncwarp();
  if (lane == 0) {
    __threadfence_system();
    atomicAdd(d_done, 1);
  }
}

extern "C" int run_gpu_bench_queue_contention(
    chi::u32 num_clients,
    chi::u32 total_tasks,
    float *out_elapsed_ms) {
  // Queue depth must hold warmup (10/client) + main run (total_tasks/client).
  // Nobody pops, so every Push occupies a slot permanently.
  constexpr chi::u32 kWarmupTasks = 10;
  chi::u32 queue_depth =
      static_cast<chi::u32>(
          (static_cast<chi::u64>(total_tasks) + kWarmupTasks) * num_clients);

  // Backend sizing: BuddyAllocator rounds to power-of-2 internally.
  // Each RingBufferEntry<u32> is 8 bytes; vector needs (depth+1) entries.
  // Budget 3x raw entry size for buddy fragmentation + allocator headers.
  size_t entry_bytes = static_cast<size_t>(queue_depth + 2) * sizeof(chi::u32) * 4;
  size_t backend_size = entry_bytes * 3 + 4 * 1024 * 1024;
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(hipc::MemoryBackendId(210, 0),
                             backend_size, "", 0)) {
    printf("Backend alloc failed (requested %zu bytes)\n", backend_size);
    return -1;
  }

  // Setup kernel: construct allocator + ring_buffer on device
  hipc::BuddyAllocator **d_alloc_ptr;
  chi::ipc::mpsc_ring_buffer<chi::u32> **d_queue_ptr;
  cudaMallocHost(&d_alloc_ptr, sizeof(void *));
  cudaMallocHost(&d_queue_ptr, sizeof(void *));
  *d_alloc_ptr = nullptr;
  *d_queue_ptr = nullptr;

  gpu_bench_queue_setup_kernel<<<1, 1>>>(
      static_cast<hipc::MemoryBackend &>(gpu_backend),
      queue_depth, d_alloc_ptr, d_queue_ptr);
  cudaDeviceSynchronize();

  cudaError_t setup_err = cudaGetLastError();
  if (setup_err != cudaSuccess) {
    printf("Setup kernel error: %s\n", cudaGetErrorString(setup_err));
    cudaFreeHost(d_alloc_ptr);
    cudaFreeHost(d_queue_ptr);
    return -2;
  }

  auto *d_queue = *d_queue_ptr;
  if (!d_queue) {
    printf("Queue construction failed on device (depth=%u, backend=%zu bytes)\n",
           queue_depth, backend_size);
    cudaFreeHost(d_alloc_ptr);
    cudaFreeHost(d_queue_ptr);
    return -2;
  }

  // Per-warp results
  long long *d_results;
  cudaMalloc(&d_results, static_cast<size_t>(num_clients) * sizeof(long long));
  cudaMemset(d_results, 0, static_cast<size_t>(num_clients) * sizeof(long long));

  int *d_done;
  cudaMallocHost(&d_done, sizeof(int));
  *d_done = 0;

  // Launch config: 1 warp = 32 threads per client
  chi::u32 total_threads = num_clients * 32;
  chi::u32 threads_per_block = 256;
  if (total_threads < threads_per_block) threads_per_block = total_threads;
  chi::u32 blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  // Warmup — pushes kWarmupTasks per client into the same queue
  gpu_bench_queue_contention_kernel<<<blocks, threads_per_block>>>(
      d_queue, kWarmupTasks, num_clients, d_results, d_done);
  cudaDeviceSynchronize();
  *d_done = 0;
  cudaMemset(d_results, 0, static_cast<size_t>(num_clients) * sizeof(long long));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  gpu_bench_queue_contention_kernel<<<blocks, threads_per_block>>>(
      d_queue, total_tasks, num_clients, d_results, d_done);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  *out_elapsed_ms = ms;

  // Read results back
  std::vector<long long> results(num_clients);
  cudaMemcpy(results.data(), d_results,
             num_clients * sizeof(long long),
             cudaMemcpyDeviceToHost);

  // Aggregate stats
  long long sum = 0, min_t = LLONG_MAX, max_t = 0;
  for (chi::u32 w = 0; w < num_clients; ++w) {
    sum += results[w];
    if (results[w] < min_t) min_t = results[w];
    if (results[w] > max_t) max_t = results[w];
  }

  chi::u64 total_pushes = static_cast<chi::u64>(num_clients) * total_tasks;
  printf("\n=== Queue Contention: mpsc_ring_buffer::Push ===\n");
  printf("  %u clients x %u tasks (depth=%u)\n",
         num_clients, total_tasks, queue_depth);
  printf("  Launch:        %u blocks x %u threads\n", blocks, threads_per_block);
  printf("  Wall time:     %.3f ms\n", ms);
  printf("\n  Push latency (clk/push, averaged across all clients): %lld\n",
         (long long)(sum / total_pushes));
  printf("  Push across warps:  min=%lld/push  max=%lld/push\n",
         (long long)(min_t / total_tasks),
         (long long)(max_t / total_tasks));

  // Per-warp detail if small enough
  if (num_clients <= 32) {
    printf("\n  Per-warp breakdown (clk/push):\n");
    printf("  %6s  %10s\n", "Warp", "push");
    for (chi::u32 w = 0; w < num_clients; ++w) {
      printf("  %6u  %10lld\n", w,
             (long long)(results[w] / total_tasks));
    }
  }

  printf("\n  Throughput:    %.0f pushes/sec\n",
         total_pushes / (ms / 1000.0));
  printf("  Avg latency:   %.3f us/push\n",
         (ms * 1000.0) / total_pushes);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_results);
  cudaFreeHost(d_done);
  cudaFreeHost(d_alloc_ptr);
  cudaFreeHost(d_queue_ptr);
  return 0;
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
