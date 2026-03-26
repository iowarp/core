/**
 * test_gpu_tiered_gpu.cc — GPU kernel for tiered storage CTE unit test
 *
 * Uses CHIMAERA_GPU_ORCHESTRATOR_INIT with 1 warp (32 threads).
 * Tests PutBlob + GetBlob at increasing sizes to find where data
 * correctness breaks.
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/gpu_work_orchestrator.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

/**
 * GPU kernel: PutBlob then GetBlob a single blob, verify correctness.
 * Uses orchestrator init (1 warp, 32 threads).
 * Blob data is in the GpuMalloc data backend.
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
  CHIMAERA_GPU_ORCHESTRATOR_INIT(gpu_info, num_blocks);

  chi::u32 warp_id = chi::IpcManager::GetWarpId();
  chi::u32 lane_id = chi::IpcManager::GetLaneId();

  if (warp_id != 0) return;

  // Step 1: Fill write buffer with pattern (all lanes cooperate)
  if (lane_id == 0) { d_progress[0] = 1; __threadfence_system(); }
  __syncwarp();

  chi::u64 total_write = blob_size * num_blobs;
  for (chi::u64 i = lane_id; i < total_write; i += 32) {
    write_ptr.ptr_[i] = static_cast<char>(i % 251);
  }
  __syncwarp();
  __threadfence();

  // Step 2: PutBlob each chunk
  if (lane_id == 0) { d_progress[0] = 2; __threadfence_system(); }
  __syncwarp();

  if (chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client client(cte_pool_id);
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;

    for (chi::u32 b = 0; b < num_blobs; b++) {
      char name[32]; int pos = 0;
      name[pos++] = 'b'; name[pos++] = '_';
      pos += StrT::NumberToStr(name + pos, 32 - pos, b);
      name[pos] = '\0';

      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(write_ptr.shm_.off_.load() + b * blob_size);

      auto f = client.AsyncPutBlob(tag_id, name,
          chi::u64(0), blob_size, shm, -1.0f,
          wrp_cte::core::Context(), chi::u32(0),
          chi::PoolQuery::Local());
      if (f.GetFutureShmPtr().IsNull()) {
        *d_result = -(100 + b); __threadfence_system(); return;
      }
      f.Wait();
      if (f->GetReturnCode() != 0) {
        *d_result = -(200 + b); __threadfence_system(); return;
      }
    }
  }
  __syncwarp();

  if (lane_id == 0) { d_progress[0] = 3; __threadfence_system(); }

  // Step 3: Skip zeroing — just let GetBlob overwrite whatever's in the buffer
  // This tests whether the L1 cache stale-data issue affects verification
  chi::u64 total_read = blob_size * num_blobs;

  // Step 4: GetBlob each chunk back
  if (lane_id == 0) { d_progress[0] = 4; __threadfence_system(); }
  __syncwarp();

  if (chi::IpcManager::IsWarpScheduler()) {
    wrp_cte::core::Client client(cte_pool_id);
    using StrT = hshm::priv::basic_string<char, CHI_PRIV_ALLOC_T>;

    for (chi::u32 b = 0; b < num_blobs; b++) {
      char name[32]; int pos = 0;
      name[pos++] = 'b'; name[pos++] = '_';
      pos += StrT::NumberToStr(name + pos, 32 - pos, b);
      name[pos] = '\0';

      hipc::ShmPtr<> shm;
      shm.alloc_id_ = data_alloc_id;
      shm.off_.exchange(read_ptr.shm_.off_.load() + b * blob_size);

      auto f = client.AsyncGetBlob(tag_id, name,
          chi::u64(0), blob_size, chi::u32(0), shm,
          chi::PoolQuery::Local());
      if (f.GetFutureShmPtr().IsNull()) {
        *d_result = -(300 + b); __threadfence_system(); return;
      }
      f.Wait();
      if (f->GetReturnCode() != 0) {
        *d_result = -(400 + b); __threadfence_system(); return;
      }
    }
  }
  __syncwarp();

  if (lane_id == 0) { d_progress[0] = 5; __threadfence_system(); }

  __threadfence_system();

  // Step 5: Verify — print first byte of each stripe to see per-lane writes
  if (lane_id == 0) {
    chi::u64 stripe = blob_size / 32;
    printf("GPU STRIPES: ");
    for (int s = 0; s < 32; s++) {
      printf("%02x ", (unsigned char)read_ptr.ptr_[s * stripe]);
    }
    printf("\n");
    // Still do normal verification
    int mismatches = 0;
    int first_bad = -1;
    for (chi::u64 i = 0; i < total_read; i++) {
      char actual = read_ptr.ptr_[i];
      char expected = static_cast<char>(i % 251);
      if (actual != expected) {
        mismatches++;
        if (first_bad < 0) {
          first_bad = (int)i;
          printf("GPU: first mismatch at byte %d: got 0x%02x expected 0x%02x\n",
                 first_bad, (unsigned char)actual, (unsigned char)expected);
        }
      }
    }
    if (mismatches > 0) {
      printf("GPU: %d total mismatches out of %llu\n",
             mismatches, (unsigned long long)total_read);
      *d_result = -500 - mismatches;
    } else {
      *d_result = 1;
    }
    d_progress[0] = 6;
    __threadfence_system();
  }

  // d_result[0] already set to 1 (success) or negative (failure) above
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

  // Start small to verify correctness
  chi::u64 blob_size = 50ULL * 1024 * 1024;  // 50MB per blob
  chi::u32 num_blobs = 4;                     // 200MB total
  chi::u64 total_data = blob_size * num_blobs * 2;  // write + read buffers

  fprintf(stderr, "TEST: blob_size=%llu MB, num_blobs=%u, total=%llu MB\n",
          (unsigned long long)(blob_size/(1024*1024)), num_blobs,
          (unsigned long long)(total_data/(1024*1024)));

  CHI_IPC->SetGpuOrchestratorBlocks(1, 32);
  CHI_IPC->PauseGpuOrchestrator();

  // Data backend
  hipc::MemoryBackendId data_id(200, 0);
  hipc::GpuMalloc data_backend;
  if (!data_backend.shm_init(data_id, total_data + 16*1024*1024, "", 0)) {
    fprintf(stderr, "TEST: data backend init failed\n");
    CHI_IPC->ResumeGpuOrchestrator(); return -1;
  }

  hipc::MemoryBackendId scratch_id(201, 0);
  hipc::GpuMalloc scratch_backend;
  scratch_backend.shm_init(scratch_id, 4*1024*1024, "", 0);

  hipc::MemoryBackendId heap_id(202, 0);
  hipc::GpuMalloc heap_backend;
  heap_backend.shm_init(heap_id, 4*1024*1024, "", 0);

  // Alloc one contiguous buffer for both write and read
  // (single MakeAlloc call — second call would reinitialize the allocator)
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
    cudaFreeHost(d_aptr); CHI_IPC->ResumeGpuOrchestrator(); return -2;
  }
  hipc::FullPtr<char> all_ptr = *d_aptr;
  cudaFreeHost(d_aptr);

  // Split: write buffer = first half, read buffer = second half
  hipc::FullPtr<char> write_ptr = all_ptr;
  hipc::FullPtr<char> read_ptr;
  read_ptr.ptr_ = all_ptr.ptr_ + write_bytes;
  read_ptr.shm_ = all_ptr.shm_;
  read_ptr.shm_.off_.exchange(all_ptr.shm_.off_.load() + write_bytes);

  fprintf(stderr, "TEST: write_ptr=%p (off=%llu) read_ptr=%p (off=%llu)\n",
          write_ptr.ptr_, (unsigned long long)write_ptr.shm_.off_.load(),
          read_ptr.ptr_, (unsigned long long)read_ptr.shm_.off_.load());

  hipc::AllocatorId data_alloc_id(data_id.major_, data_id.minor_);
  CHI_IPC->RegisterGpuAllocator(data_id, data_backend.data_,
                                 data_backend.data_capacity_);

  chi::IpcManagerGpu gpu_info = CHI_IPC->GetClientGpuInfo(0);
  gpu_info.backend = scratch_backend;

  // Result: [0]=test result, [1]=done counter
  int *d_result;
  cudaMallocHost(&d_result, sizeof(int) * 2);
  d_result[0] = 0; d_result[1] = 0;
  volatile int *d_progress;
  cudaMallocHost((void**)&d_progress, sizeof(int));
  *d_progress = 0;

  if (scratch_backend.data_)
    cudaMemset(scratch_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  if (heap_backend.data_)
    cudaMemset(heap_backend.data_, 0, sizeof(hipc::PartitionedAllocator));
  cudaDeviceSynchronize();

  void *stream = hshm::GpuApi::CreateStream();
  gpu_tiered_test_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      gpu_info, pool_id, tag_id, 1,
      write_ptr, read_ptr, data_alloc_id,
      blob_size, num_blobs,
      d_result, d_progress);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "TEST: launch failed: %s\n", cudaGetErrorString(err));
    CHI_IPC->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream); return -4;
  }

  CHI_IPC->ResumeGpuOrchestrator();
  auto *orch = static_cast<chi::gpu::WorkOrchestrator*>(CHI_IPC->gpu_orchestrator_);
  auto *ctrl = orch ? orch->control_ : nullptr;
  if (ctrl) {
    int w = 0;
    while (ctrl->running_flag == 0 && w < 5000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); ++w;
    }
  }

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

  CHI_IPC->PauseGpuOrchestrator();
  hshm::GpuApi::Synchronize(stream);
  hshm::GpuApi::DestroyStream(stream);
  cudaFreeHost(d_result);
  cudaFreeHost((void*)d_progress);

  return result;
}

#endif  // HSHM_IS_HOST
#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
