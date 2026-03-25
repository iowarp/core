/**
 * page_cache.cu -- GPU-side page cache: init kernel only.
 *
 * All runtime cache operations (acquire, release, I/O) are inline
 * in page_cache.cuh to avoid warp-level deadlocks and enable
 * the compiler to optimize across call boundaries.
 */
#include <bam/page_cache.cuh>
#include <cuda_runtime.h>

namespace bam {

__global__ void page_cache_init_kernel(PageCacheDeviceState state) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < state.num_pages) {
    state.page_states[tid] = static_cast<uint32_t>(PageState::kInvalid);
    state.page_tags[tid] = ~0ULL;
    state.page_locks[tid] = 0;
  }
}

}  // namespace bam
