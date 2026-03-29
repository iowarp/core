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
  auto task = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
      (chi::u32)0, (chi::u32)0);

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto future = ipc->Send(task);
    future.Wait(0, /*reuse_task=*/true);
  }

  // Free the reused task after the loop
  ipc->DelTask(task);

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

  // All lanes call AsyncSubtaskTest — internally guarded by warp leader.
  chimaera::MOD_NAME::Client client(pool_id);
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    auto future = client.AsyncSubtaskTest(chi::PoolQuery::Local(), i, subtasks);
    future.Wait();
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
  using GpuSubmitTask = chimaera::MOD_NAME::GpuSubmitTask;

  auto *ipc = CHI_IPC;
  auto *priv_alloc = ipc->GetPrivAlloc();
  if (!priv_alloc) return;

  // Warm up
  {
    auto w = priv_alloc->template AllocateObjs<GpuSubmitTask>(1);
    if (!w.IsNull()) priv_alloc->Free(w);
  }

  long long t_raw_alloc = 0, t_raw_free = 0;
  long long t_ctor = 0, t_dtor = 0;
  long long t_ser_in = 0, t_ser_out = 0;
  long long t_coro_alloc = 0, t_coro_free = 0;
  long long tc;

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    // 1. Raw alloc
    tc = clock64();
    auto fp = priv_alloc->template AllocateObjs<GpuSubmitTask>(1);
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
    fp->counter_addr_ = 0;
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
    priv_alloc->Free(fp);
    t_raw_free += clock64() - tc;
  }

  // 8. Coroutine frame alloc/free (simulates inner coroutine overhead)
  // A coroutine frame is ~promise_type + locals, typically 64-128 bytes
  for (chi::u32 i = 0; i < total_tasks; ++i) {
    tc = clock64();
    auto frame = priv_alloc->template AllocateObjs<char>(128);
    t_coro_alloc += clock64() - tc;

    tc = clock64();
    if (!frame.IsNull()) priv_alloc->Free(frame);
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
  chi::u32 lane = chi::IpcManager::GetLaneId();

  // Accumulators (lane 0 only records times)
  long long t_new_task = 0, t_warp_setup = 0, t_serialize_in = 0;
  long long t_shm_send = 0, t_fence_push = 0;
  long long t_shm_recv = 0, t_serialize_out = 0;
  long long t_del_task = 0;
  // SendDevice sub-breakdown
  long long t_send_alloc = 0, t_send_frame = 0, t_send_write = 0;
  // RecvDevice sub-breakdown
  long long t_recv_alloc = 0, t_recv_read = 0, t_recv_deser = 0;
  long long tc;

  // Allocate task + WarpIpcManager once before the loop (reused every iteration)
  unsigned long long mgr_ull = 0, task_ull = 0, fshm_ull = 0;
  chi::FutureShm *fshm = nullptr;
  hipc::FullPtr<chimaera::MOD_NAME::GpuSubmitTask> task_fp;
  if (lane == 0) {
    task_fp = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
        (chi::u32)0, (chi::u32)0);
    auto *mgr = ipc->GetWarpManager();
    auto &buffer = ipc->GetCachedFutureShm();
    fshm = reinterpret_cast<chi::FutureShm *>(buffer.ptr_);
    mgr_ull = reinterpret_cast<unsigned long long>(mgr);
    task_ull = reinterpret_cast<unsigned long long>(task_fp.ptr_);
    fshm_ull = reinterpret_cast<unsigned long long>(fshm);
  }
  mgr_ull = hipc::shfl_sync_u64(0xFFFFFFFF, mgr_ull, 0);
  task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);
  fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);

  for (chi::u32 i = 0; i < total_tasks; ++i) {
    // ============================================================
    // SEND side (mirrors SendGpu with task reuse)
    // ============================================================

    // Step 2: WarpMgr + FutureShm reset (no NewTask — task is reused)
    if (lane == 0) {
      tc = clock64();
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull);
      mgr->save_ar_.Reset(chi::LocalMsgType::kSerializeIn);
      mgr->save_ar_.SetWarpConverged(true);

      fshm->Reset(task_fp->pool_id_, task_fp->method_);
      fshm->flags_.SetBits(chi::FutureShm::FUTURE_COPY_FROM_CLIENT);
      fshm->flags_.SetBits(chi::FutureShm::FUTURE_DEVICE_SCOPE);
      fshm->input_.copy_space_size_.store(
          chi::IpcManager::WarpIpcManager::kCopySpaceSize);
      fshm->output_.copy_space_size_.store(
          chi::IpcManager::WarpIpcManager::kCopySpaceSize);
      t_warp_setup += clock64() - tc;
    }

    // Step 3: Warp-parallel SerializeIn
    mgr_ull = hipc::shfl_sync_u64(0xFFFFFFFF, mgr_ull, 0);
    task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);
    fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);

    tc = clock64();
    if (mgr_ull && task_ull) {
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull);
      auto *task = reinterpret_cast<chimaera::MOD_NAME::GpuSubmitTask *>(task_ull);
      task->SerializeIn(mgr->save_ar_);
    }
    __syncwarp();
    if (lane == 0) t_serialize_in += clock64() - tc;

    // Step 4: ShmTransport::SendDevice (warp-parallel via prealloc path)
    if (lane == 0 && mgr_ull && fshm_ull) {
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull);
      fshm = reinterpret_cast<chi::FutureShm *>(fshm_ull);
      mgr->save_ar_.SetWarpConverged(false);
      fshm->input_.copy_space_size_.store(
          chi::IpcManager::WarpIpcManager::kCopySpaceSize);
    }
    // All lanes participate in SendDevice (warp-parallel copy)
    tc = clock64();
    if (mgr_ull && fshm_ull) {
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull);
      fshm = reinterpret_cast<chi::FutureShm *>(fshm_ull);
      hshm::lbm::LbmContext ctx;
      ctx.copy_space = fshm->copy_space;
      ctx.shm_info_ = &fshm->input_;
      { static __shared__ char s_meta[256]; ctx.meta_buf_ = s_meta; }
      ctx.meta_buf_size_ = 256;
      ctx.warp_parallel_ = true;
      hshm::lbm::ShmTransport::SendDevice(mgr->save_ar_, ctx);
    }
    __syncwarp();
    if (lane == 0) {
      t_shm_send += clock64() - tc;
      // Step 5: threadfence
      tc = clock64();
      hipc::threadfence();
      t_fence_push += clock64() - tc;
    }
    __syncwarp();

    // ============================================================
    // RECV side (mirrors RecvGpu)
    // ============================================================

    // Step 6: Simulate FUTURE_COMPLETE + write mock output to copy_space
    // In the real path the orchestrator does this. Here we just mark complete
    // and write a small output (result_value_) so RecvDevice has data.
    if (lane == 0) {
      // Write mock output: SerializeOut produces return_code_ + completer_ +
      // result_value_ Simulate by doing a SendDevice with output data
      chi::priv::vector<char> out_buf;
      out_buf.reserve(256);
      chi::DefaultSaveArchive out_save(chi::LocalMsgType::kSerializeOut, out_buf);
      task_fp->SerializeOut(out_save);
      hshm::lbm::LbmContext out_ctx;
      out_ctx.copy_space = fshm->copy_space;
      out_ctx.shm_info_ = &fshm->output_;
      { static __shared__ char s_meta_out[256]; out_ctx.meta_buf_ = s_meta_out; }
      out_ctx.meta_buf_size_ = 256;
      hshm::lbm::ShmTransport::SendDevice(out_save, out_ctx);
      hipc::threadfence();
      fshm->flags_.SetBits(chi::FutureShm::FUTURE_COMPLETE);
    }
    __syncwarp();

    // Step 7: ShmTransport::RecvDevice (warp-parallel via prealloc path)
    unsigned long long mgr_ull2 = 0;
    int has_output = 0;
    if (lane == 0) {
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull);
      size_t output_written = fshm->output_.total_written_.load_device();
      if (output_written > 0) {
        mgr_ull2 = mgr_ull;
        has_output = 1;
      }
    }
    // All lanes participate in RecvDevice (warp-parallel copy)
    mgr_ull2 = hipc::shfl_sync_u64(0xFFFFFFFF, mgr_ull2, 0);
    has_output = __shfl_sync(0xFFFFFFFF, has_output, 0);
    fshm_ull = hipc::shfl_sync_u64(0xFFFFFFFF, fshm_ull, 0);
    tc = clock64();
    if (has_output && mgr_ull2) {
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull2);
      fshm = reinterpret_cast<chi::FutureShm *>(fshm_ull);
      hshm::lbm::LbmContext ctx;
      ctx.copy_space = fshm->copy_space;
      ctx.shm_info_ = &fshm->output_;
      { static __shared__ char s_meta[256]; ctx.meta_buf_ = s_meta; }
      ctx.meta_buf_size_ = 256;
      ctx.warp_parallel_ = true;
      hshm::lbm::ShmTransport::RecvDevice(mgr->load_ar_, ctx);
    }
    __syncwarp();
    if (lane == 0) {
      t_shm_recv += clock64() - tc;
      if (has_output) {
        auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull2);
        mgr->load_ar_.SetMsgType(chi::LocalMsgType::kSerializeOut);
        mgr->load_ar_.SetWarpConverged(true);
      }
    }

    // Step 8: Warp-parallel SerializeOut
    mgr_ull2 = hipc::shfl_sync_u64(0xFFFFFFFF, mgr_ull2, 0);
    task_ull = hipc::shfl_sync_u64(0xFFFFFFFF, task_ull, 0);
    has_output = __shfl_sync(0xFFFFFFFF, has_output, 0);

    tc = clock64();
    if (has_output) {
      auto *mgr = reinterpret_cast<chi::IpcManager::WarpIpcManager *>(mgr_ull2);
      auto *task = reinterpret_cast<chimaera::MOD_NAME::GpuSubmitTask *>(task_ull);
      task->SerializeOut(mgr->load_ar_);
    }
    __syncwarp();
    if (lane == 0) t_serialize_out += clock64() - tc;

    __syncwarp();
  }

  // Free the reused task after the loop
  if (lane == 0) {
    ipc->DelTask(task_fp);
  }
  __syncwarp();

  // === Sub-benchmarks: isolate SendDevice/RecvDevice internals ===
  // Run only on lane 0 after main loop to avoid interfering with main timings.
  if (lane == 0) {
    auto *mgr = ipc->GetWarpManager();
    auto &buffer = ipc->GetCachedFutureShm();
    auto *fshm_sub = reinterpret_cast<chi::FutureShm *>(buffer.ptr_);

    // Sub-benchmark: meta_buf alloc+reserve (the vector inside SendDevice)
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      tc = clock64();
      chi::priv::vector<char> meta_buf(mgr->save_ar_.alloc_);
      meta_buf.reserve(chi::IpcManager::WarpIpcManager::kCopySpaceSize);
      t_send_alloc += clock64() - tc;
    }

    // Sub-benchmark: LocalSerialize framing (ar(save_ar) + Finalize)
    // Re-serialize the last save_ar state (still valid from main loop)
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      chi::priv::vector<char> meta_buf(mgr->save_ar_.alloc_);
      meta_buf.reserve(chi::IpcManager::WarpIpcManager::kCopySpaceSize);
      tc = clock64();
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ar(meta_buf);
      ar(mgr->save_ar_);
      ar.Finalize();
      t_send_frame += clock64() - tc;
    }

    // Sub-benchmark: WriteTransferDevice (memcpy to copy_space + atomics)
    // Prepare a fixed-size source buffer and fresh copy_space
    {
      chi::priv::vector<char> meta_buf(mgr->save_ar_.alloc_);
      meta_buf.reserve(chi::IpcManager::WarpIpcManager::kCopySpaceSize);
      hshm::ipc::LocalSerialize<chi::priv::vector<char>> ar(meta_buf);
      ar(mgr->save_ar_);
      ar.Finalize();
      uint32_t meta_len = static_cast<uint32_t>(meta_buf.size());

      for (chi::u32 i = 0; i < total_tasks; ++i) {
        // Reset copy_space ring buffer state for each iteration
        new (fshm_sub) chi::FutureShm();
        fshm_sub->input_.copy_space_size_.store(
            chi::IpcManager::WarpIpcManager::kCopySpaceSize);
        hshm::lbm::LbmContext ctx;
        ctx.copy_space = fshm_sub->copy_space;
        ctx.shm_info_ = &fshm_sub->input_;

        tc = clock64();
        // This is what WriteTransferDevice does: memcpy + threadfence + atomic store
        size_t ring_size = ctx.shm_info_->copy_space_size_.load();
        memcpy(ctx.copy_space, &meta_len, sizeof(meta_len));
        memcpy(ctx.copy_space + sizeof(meta_len), meta_buf.data(), meta_len);
        hipc::threadfence();
        ctx.shm_info_->total_written_.store(sizeof(meta_len) + meta_len);
        t_send_write += clock64() - tc;
      }
    }

    // Sub-benchmark: ReadTransferDevice (memcpy from copy_space + atomics)
    // similar to above but reading
    {
      // Prepare copy_space with valid data once
      mgr->save_ar_.Reset(chi::LocalMsgType::kSerializeIn);
      auto task_sub = ipc->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
          chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
          (chi::u32)0, (chi::u32)999);
      task_sub->SerializeIn(mgr->save_ar_);
      new (fshm_sub) chi::FutureShm();
      fshm_sub->input_.copy_space_size_.store(
          chi::IpcManager::WarpIpcManager::kCopySpaceSize);
      hshm::lbm::LbmContext prep_ctx;
      prep_ctx.copy_space = fshm_sub->copy_space;
      prep_ctx.shm_info_ = &fshm_sub->input_;
      { static __shared__ char s_meta_prep[256]; prep_ctx.meta_buf_ = s_meta_prep; }
      prep_ctx.meta_buf_size_ = 256;
      hshm::lbm::ShmTransport::SendDevice(mgr->save_ar_, prep_ctx);
      size_t written = fshm_sub->input_.total_written_.load();

      for (chi::u32 i = 0; i < total_tasks; ++i) {
        // Reset read position but keep written data
        fshm_sub->input_.total_read_.store(0);
        hshm::lbm::LbmContext ctx;
        ctx.copy_space = fshm_sub->copy_space;
        ctx.shm_info_ = &fshm_sub->input_;

        tc = clock64();
        // Simulate ReadTransferDevice: memcpy from copy_space
        size_t ring_size = ctx.shm_info_->copy_space_size_.load();
        char local_buf[512];
        memcpy(local_buf, ctx.copy_space, written);
        ctx.shm_info_->total_read_.store(written);
        t_recv_read += clock64() - tc;
      }

      // Sub-benchmark: LocalDeserialize framing (from meta_buf back to load_ar)
      for (chi::u32 i = 0; i < total_tasks; ++i) {
        fshm_sub->input_.total_read_.store(0);
        hshm::lbm::LbmContext ctx;
        ctx.copy_space = fshm_sub->copy_space;
        ctx.shm_info_ = &fshm_sub->input_;
        { static __shared__ char s_meta[256]; ctx.meta_buf_ = s_meta; }
        ctx.meta_buf_size_ = 256;

        tc = clock64();
        hshm::lbm::ShmTransport::RecvDevice(mgr->load_ar_, ctx);
        t_recv_deser += clock64() - tc;
      }

      ipc->DelTask(task_sub);
    }
  }
  __syncwarp();

  // === RunContext sub-benchmarks (mirrors orchestrator AllocContext/FreeContext) ===
  // RunContext is ~612 bytes. We measure alloc+memset+free at that size.
  static constexpr size_t kRunContextSize = 640;  // rounded up
  long long t_rctx_alloc = 0, t_rctx_memset = 0;
  long long t_rctx_free = 0, t_rctx_total = 0;
  if (lane == 0) {
    for (chi::u32 i = 0; i < total_tasks; ++i) {
      // Allocate
      tc = clock64();
      auto alloc_result = ipc->gpu_alloc_->AllocateObjs<char>(kRunContextSize);
      t_rctx_alloc += clock64() - tc;

      char *ctx = alloc_result.ptr_;
      if (!ctx) continue;

      // Memset (simulates struct copy + zero of 64 coro handles)
      tc = clock64();
      memset(ctx, 0, kRunContextSize);
      t_rctx_memset += clock64() - tc;

      // Free
      tc = clock64();
      ipc->gpu_alloc_->Free(alloc_result);
      t_rctx_free += clock64() - tc;
    }
    t_rctx_total = t_rctx_alloc + t_rctx_memset + t_rctx_free;
  }
  __syncwarp();

  // Print breakdown (block 0, lane 0 only)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("--- SendGpu + RecvGpu Client-Side Breakdown (GpuSubmitTask) ---\n");
    printf("  SEND:\n");
    printf("    1. NewTask<GpuSubmit>:    %llu  (%llu/task)\n",
           (unsigned long long)t_new_task,
           (unsigned long long)(t_new_task / total_tasks));
    printf("    2. WarpMgr+FutureShm:    %llu  (%llu/task)\n",
           (unsigned long long)t_warp_setup,
           (unsigned long long)(t_warp_setup / total_tasks));
    printf("    3. SerializeIn (warp):   %llu  (%llu/task)\n",
           (unsigned long long)t_serialize_in,
           (unsigned long long)(t_serialize_in / total_tasks));
    printf("    4. ShmTransport::Send:   %llu  (%llu/task)\n",
           (unsigned long long)t_shm_send,
           (unsigned long long)(t_shm_send / total_tasks));
    printf("    5. threadfence:          %llu  (%llu/task)\n",
           (unsigned long long)t_fence_push,
           (unsigned long long)(t_fence_push / total_tasks));
    long long t_send = t_new_task + t_warp_setup + t_serialize_in +
                       t_shm_send + t_fence_push;
    printf("    SEND TOTAL:              %llu  (%llu/task)\n",
           (unsigned long long)t_send,
           (unsigned long long)(t_send / total_tasks));
    printf("  RECV:\n");
    printf("    7. ShmTransport::Recv:   %llu  (%llu/task)\n",
           (unsigned long long)t_shm_recv,
           (unsigned long long)(t_shm_recv / total_tasks));
    printf("    8. SerializeOut (warp):  %llu  (%llu/task)\n",
           (unsigned long long)t_serialize_out,
           (unsigned long long)(t_serialize_out / total_tasks));
    long long t_recv = t_shm_recv + t_serialize_out;
    printf("    RECV TOTAL:              %llu  (%llu/task)\n",
           (unsigned long long)t_recv,
           (unsigned long long)(t_recv / total_tasks));
    long long total = t_send + t_recv;
    printf("  CLIENT TOTAL:              %llu  (%llu/task)\n",
           (unsigned long long)total,
           (unsigned long long)(total / total_tasks));
    printf("\n--- ShmTransport Sub-Breakdown (isolated micro-benchmarks) ---\n");
    printf("  SendDevice internals:\n");
    printf("    meta_buf alloc+reserve:  %llu  (%llu/task)\n",
           (unsigned long long)t_send_alloc,
           (unsigned long long)(t_send_alloc / total_tasks));
    printf("    LocalSerialize framing:  %llu  (%llu/task)\n",
           (unsigned long long)t_send_frame,
           (unsigned long long)(t_send_frame / total_tasks));
    printf("    memcpy to copy_space:    %llu  (%llu/task)\n",
           (unsigned long long)t_send_write,
           (unsigned long long)(t_send_write / total_tasks));
    printf("  RecvDevice internals:\n");
    printf("    memcpy from copy_space:  %llu  (%llu/task)\n",
           (unsigned long long)t_recv_read,
           (unsigned long long)(t_recv_read / total_tasks));
    printf("    RecvDevice full:         %llu  (%llu/task)\n",
           (unsigned long long)t_recv_deser,
           (unsigned long long)(t_recv_deser / total_tasks));
    printf("\n--- RunContext Sub-Breakdown (640B alloc, mirrors orchestrator) ---\n");
    printf("  Allocate 640B:             %llu  (%llu/task)\n",
           (unsigned long long)t_rctx_alloc,
           (unsigned long long)(t_rctx_alloc / total_tasks));
    printf("  Memset 640B:               %llu  (%llu/task)\n",
           (unsigned long long)t_rctx_memset,
           (unsigned long long)(t_rctx_memset / total_tasks));
    printf("  Free 640B:                 %llu  (%llu/task)\n",
           (unsigned long long)t_rctx_free,
           (unsigned long long)(t_rctx_free / total_tasks));
    printf("  TOTAL:                     %llu  (%llu/task)\n",
           (unsigned long long)t_rctx_total,
           (unsigned long long)(t_rctx_total / total_tasks));
  }

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

  // Re-fetch gpu_info AFTER queue rebuild (SetGpuOrchestratorBlocks rebuilt it)
  gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = gpu_backend;

  PrintKernelInfo("gpu_bench_client_kernel",
                  (const void *)chi_bench::gpu_bench_client_kernel,
                  client_blocks, client_threads);
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

  PrintKernelInfo("gpu_bench_coroutine_kernel",
                  (const void *)chi_bench::gpu_bench_coroutine_kernel,
                  client_blocks, client_threads);
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

    // All lanes call AsyncPutBlob — internally guarded by warp leader
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
