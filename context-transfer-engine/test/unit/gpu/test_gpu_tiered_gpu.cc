/**
 * test_gpu_tiered_gpu.cc — GPU kernel for tiered storage CTE unit test
 *
 * Uses CHIMAERA_GPU_CLIENT_INIT with 1 warp (32 threads).
 * Tests PutBlob + GetBlob with 1MB blobs to exercise tiered placement.
 * Task objects are cached (allocated once, reused per-iteration).
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu/work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

/**
 * GPU kernel: PutBlob then GetBlob many 1MB blobs, verify correctness.
 * Task caching: allocates one PutBlobTask and one GetBlobTask up front,
 * then reuses them by reinitializing fields per-iteration.
 */
__global__ void gpu_tiered_test_kernel(
    chi::IpcManagerGpu gpu_info,
    chi::PoolId cte_pool_id,
    wrp_cte::core::TagId tag_id,
    chi::u32 num_blocks,
    hipc::FullPtr<char> write_ptr,
    hipc::FullPtr<char> read_ptr,
    hipc::AllocatorId data_alloc_id,
    chi::u64 blob_size,
    chi::u32 num_blobs,
    int *d_result,
    volatile int *d_progress) {
  CHIMAERA_GPU_CLIENT_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::gpu::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::gpu::IpcManager::GetLaneId();

  if (warp_id != 0) return;

  auto *ipc = CHI_IPC;
  using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;

  // Step 1: Fill write buffer with pattern (all lanes cooperate)
  if (lane_id == 0) { d_progress[0] = 1; __threadfence_system(); }
  __syncwarp();

  chi::u64 total_write = blob_size * num_blobs;
  for (chi::u64 i = lane_id; i < total_write; i += 32) {
    write_ptr.ptr_[i] = static_cast<char>(i % 251);
  }
  __syncwarp();
  __threadfence();

  // Step 2: PutBlob each chunk (thread 0 only, cached task)
  if (lane_id == 0) {
    d_progress[0] = 2; __threadfence_system();

    // Use null alloc_id with absolute device address — ToFullPtr on GPU
    // interprets null alloc_id + off_ as raw device pointer
    hipc::ShmPtr<> shm;
    shm.alloc_id_.SetNull();
    shm.off_.exchange(reinterpret_cast<size_t>(write_ptr.ptr_));

    auto put_task = ipc->NewTask<wrp_cte::core::PutBlobTask>(
        chi::CreateTaskId(), cte_pool_id, chi::PoolQuery::Local(),
        tag_id, "b_0", chi::u64(0), blob_size, shm, -1.0f,
        wrp_cte::core::Context(), chi::u32(0));
    if (put_task.IsNull()) {
      *d_result = -100; __threadfence_system(); return;
    }

    for (chi::u32 b = 0; b < num_blobs; b++) {
      char name[32]; int pos = 0;
      name[pos++] = 'b'; name[pos++] = '_';
      pos += StrT::NumberToStr(name + pos, 32 - pos, b);
      name[pos] = '\0';

      put_task->blob_name_ = chi::priv::string(CHI_PRIV_ALLOC, name);
      // Use absolute device address (ToFullPtr on GPU interprets off_ as raw ptr)
      put_task->blob_data_.off_.exchange(
          reinterpret_cast<size_t>(write_ptr.ptr_ + b * blob_size));
      put_task->task_id_ = chi::CreateTaskId();
      put_task->return_code_ = -1;

      auto f = ipc->Send(put_task);
      f.WaitGpu();

      if (put_task->GetReturnCode() != 0) {
        *d_result = -(200 + b); __threadfence_system(); return;
      }
    }

    ipc->DelTask(put_task);
    d_progress[0] = 3; __threadfence_system();
  }
  __syncwarp();

  chi::u64 total_read = blob_size * num_blobs;

  // Step 3: GetBlob each chunk back (thread 0 only, cached task)
  if (lane_id == 0) {
    d_progress[0] = 4; __threadfence_system();

    hipc::ShmPtr<> shm;
    shm.alloc_id_.SetNull();
    shm.off_.exchange(reinterpret_cast<size_t>(read_ptr.ptr_));

    auto get_task = ipc->NewTask<wrp_cte::core::GetBlobTask>(
        chi::CreateTaskId(), cte_pool_id, chi::PoolQuery::Local(),
        tag_id, "b_0", chi::u64(0), blob_size, chi::u32(0), shm);
    if (get_task.IsNull()) {
      *d_result = -300; __threadfence_system(); return;
    }

    for (chi::u32 b = 0; b < num_blobs; b++) {
      char name[32]; int pos = 0;
      name[pos++] = 'b'; name[pos++] = '_';
      pos += StrT::NumberToStr(name + pos, 32 - pos, b);
      name[pos] = '\0';

      get_task->blob_name_ = chi::priv::string(CHI_PRIV_ALLOC, name);
      // Use absolute device address (ToFullPtr on GPU interprets off_ as raw ptr)
      get_task->blob_data_.off_.exchange(
          reinterpret_cast<size_t>(read_ptr.ptr_ + b * blob_size));
      get_task->task_id_ = chi::CreateTaskId();
      get_task->return_code_ = -1;

      auto f = ipc->Send(get_task);
      f.WaitGpu();

      if (get_task->GetReturnCode() != 0) {
        *d_result = -(400 + b); __threadfence_system(); return;
      }
    }

    ipc->DelTask(get_task);
  }
  __syncwarp();

  if (lane_id == 0) { d_progress[0] = 5; __threadfence_system(); }

  __threadfence_system();

  // Step 4: Verify (all lanes cooperate for speed)
  __shared__ int s_mismatches;
  if (lane_id == 0) s_mismatches = 0;
  __syncwarp();

  for (chi::u64 i = lane_id; i < total_read; i += 32) {
    char actual = read_ptr.ptr_[i];
    char expected = static_cast<char>(i % 251);
    if (actual != expected) {
      atomicAdd(&s_mismatches, 1);
    }
  }
  __syncwarp();

  if (lane_id == 0) {
    if (s_mismatches > 0) {
      printf("GPU: %d total mismatches out of %llu\n",
             s_mismatches, (unsigned long long)total_read);
      *d_result = -500 - s_mismatches;
    } else {
      *d_result = 1;
    }
    d_progress[0] = 6;
    __threadfence_system();
  }
}

// Alloc kernel
__global__ void gpu_tiered_alloc_kernel(
    hipc::MemoryBackend data_backend, chi::u64 total_bytes,
    hipc::FullPtr<char> *d_out_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using AllocT = hipc::PrivateBuddyAllocator;
  auto *alloc = data_backend.MakeAlloc<AllocT>(data_backend.data_capacity_);
  if (!alloc) { d_out_ptr->SetNull(); return; }
  *d_out_ptr = alloc->AllocateObjs<char>(total_bytes);
}

#if HSHM_IS_HOST
#include <hermes_shm/lightbeam/transport_factory_impl.h>

extern "C" int run_gpu_tiered_test(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    int timeout_sec) {

  // 1MB blobs x 50 = 50MB total per direction (put + get)
  chi::u64 blob_size = 1ULL * 1024 * 1024;   // 1MB per blob
  chi::u32 num_blobs = 50;                     // 50MB total
  chi::u64 total_data = blob_size * num_blobs * 2;  // write + read buffers

  fprintf(stderr, "TEST: blob_size=%llu KB, num_blobs=%u, total=%llu MB\n",
          (unsigned long long)(blob_size/1024), num_blobs,
          (unsigned long long)(total_data/(1024*1024)));

  CHI_CPU_IPC->GetGpuIpcManager()->SetGpuOrchestratorBlocks(1, 32);
  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();

  // Data backend
  hipc::MemoryBackendId data_id(200, 0);
  hipc::GpuMalloc data_backend;
  if (!data_backend.shm_init(data_id, total_data + 16*1024*1024, "", 0)) {
    fprintf(stderr, "TEST: data backend init failed\n");
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator(); return -1;
  }

  // Alloc one contiguous buffer for both write and read
  chi::u64 write_bytes = blob_size * num_blobs;
  chi::u64 read_bytes = blob_size * num_blobs;
  hipc::FullPtr<char> *d_aptr;
  cudaMallocHost(&d_aptr, sizeof(hipc::FullPtr<char>));
  d_aptr->SetNull();
  gpu_tiered_alloc_kernel<<<1,1>>>(
      static_cast<hipc::MemoryBackend&>(data_backend),
      write_bytes + read_bytes, d_aptr);
  cudaDeviceSynchronize();
  if (d_aptr->IsNull()) {
    fprintf(stderr, "TEST: alloc failed\n");
    cudaFreeHost(d_aptr); CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator(); return -2;
  }
  hipc::FullPtr<char> all_ptr = *d_aptr;
  cudaFreeHost(d_aptr);

  // Split: write buffer = first half, read buffer = second half
  hipc::FullPtr<char> write_ptr = all_ptr;
  hipc::FullPtr<char> read_ptr;
  read_ptr.ptr_ = all_ptr.ptr_ + write_bytes;
  read_ptr.shm_ = all_ptr.shm_;
  read_ptr.shm_.off_.exchange(all_ptr.shm_.off_.load() + write_bytes);

  hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
  CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(data_id, data_backend.data_,
                                     data_backend.data_capacity_);

  // Use the orchestrator's shared allocator backend
  chi::IpcManagerGpuInfo gpu_info =
      CHI_CPU_IPC->GetGpuIpcManager()->CreateGpuAllocator(0, 0);

  // Result + progress in pinned host memory
  int *d_result;
  cudaMallocHost(&d_result, sizeof(int) * 2);
  d_result[0] = 0; d_result[1] = 0;
  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int));
  *d_progress = 0;

  void *stream = hshm::GpuApi::CreateStream();
  gpu_tiered_test_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, tag_id, 1,
      write_ptr, read_ptr, data_alloc_id,
      blob_size, num_blobs,
      d_result, d_progress);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "TEST: launch failed: %s\n", cudaGetErrorString(err));
    CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream); return -4;
  }

  CHI_CPU_IPC->GetGpuIpcManager()->ResumeGpuOrchestrator();

  int64_t timeout_us = (int64_t)timeout_sec * 1000000;
  int64_t elapsed_us = 0;
  int last_p = -1;
  while (d_result[0] == 0 && elapsed_us < timeout_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    elapsed_us += 500;
    int p = *d_progress;
    if (p != last_p) {
      const char *s[] = {"init","fill","put","put done","get","get done","verify"};
      fprintf(stderr, "TEST: step=%d (%s) t=%.1fs\n", p, p<7?s[p]:"?", elapsed_us/1e6);
      fflush(stderr);
      last_p = p;
    }
  }

  int result = d_result[0];

  fprintf(stderr, "TEST: result=%d\n", result);
  if (result == 1) fprintf(stderr, "TEST: PASSED\n");
  else if (result == 0) fprintf(stderr, "TEST: TIMEOUT (step=%d)\n", (int)*d_progress);
  else fprintf(stderr, "TEST: FAILED\n");
  fflush(stderr);

  CHI_CPU_IPC->GetGpuIpcManager()->PauseGpuOrchestrator();
  hshm::GpuApi::Synchronize(stream);
  hshm::GpuApi::DestroyStream(stream);
  cudaFreeHost(d_result);
  cudaFreeHost((void*)d_progress);

  return result;
}

#endif  // HSHM_IS_HOST
#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
