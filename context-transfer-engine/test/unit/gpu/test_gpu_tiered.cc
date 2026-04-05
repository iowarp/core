/**
 * test_gpu_tiered.cc — CTE GPU unit test with two-tier storage (host side)
 *
 * Tests GPU-initiated PutBlob (200MB) and GetBlob (100MB) with:
 *   - HBM tier: 50MB (fills first, then spills to pinned)
 *   - Pinned host DRAM tier: 400MB (overflow)
 *
 * This file is compiled by g++ (not nvcc) so CHI_IPC resolves correctly.
 * The GPU kernel is in test_gpu_tiered_gpu.cc, compiled by nvcc.
 */

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/singletons.h>
#include <chimaera/ipc_manager.h>
#include <hermes_shm/util/gpu_api.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>

using namespace std::chrono_literals;

extern "C" int run_gpu_tiered_test(
    chi::PoolId pool_id,
    wrp_cte::core::TagId tag_id,
    int timeout_sec);

int main(int argc, char **argv) {
  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    printf("SKIP: No GPU available\n");
    return 0;
  }

  // Load the two-tier config
  std::string config_path = std::string(__FILE__);
  config_path = config_path.substr(0, config_path.rfind('/'));
  config_path += "/test_gpu_tiered_config.yaml";

  if (!std::filesystem::exists(config_path)) {
    fprintf(stderr, "ERROR: Config not found: %s\n", config_path.c_str());
    return 1;
  }

  setenv("CHI_SERVER_CONF", config_path.c_str(), 1);
  setenv("CHI_GPU_BLOCKS", "1", 1);
  setenv("CHI_GPU_THREADS", "32", 1);

  printf("============================================================\n");
  printf("  CTE GPU Tiered Storage Unit Test\n");
  printf("============================================================\n");
  printf("Config:    %s\n", config_path.c_str());
  printf("HBM tier:  50 MB\n");
  printf("Pinned:    400 MB\n");
  printf("Put:       50 MB (50 x 1MB blobs)\n");
  printf("Get:       50 MB (50 x 1MB blobs)\n");
  printf("Client:    1 warp (32 threads)\n");
  printf("Runtime:   1 warp\n");
  printf("------------------------------------------------------------\n\n");

  printf("Initializing Chimaera runtime...\n");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    fprintf(stderr, "ERROR: CHIMAERA_INIT failed\n");
    return 1;
  }
  std::this_thread::sleep_for(500ms);
  printf("Runtime initialized.\n\n");

  // Create CTE pool
  chi::PoolId pool_id(wrp_cte::core::kCtePoolId.major_ + 1,
                       wrp_cte::core::kCtePoolId.minor_);
  wrp_cte::core::Client cte_client(pool_id);
  wrp_cte::core::CreateParams params;
  auto create_task = cte_client.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      "cte_tiered_test_pool", pool_id, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    fprintf(stderr, "ERROR: CTE pool create failed: %d\n",
            create_task->GetReturnCode());
    return 1;
  }
  std::this_thread::sleep_for(200ms);
  printf("CTE pool created: %u.%u\n", pool_id.major_, pool_id.minor_);

  // Register HBM target (CPU + GPU)
  chi::PoolId hbm_bdev_pool_id(800, 0);
  {
    auto reg = cte_client.AsyncRegisterTarget(
        "hbm::tiered_test_hbm",
        chimaera::bdev::BdevType::kHbm,
        50ULL * 1024 * 1024,
        chi::PoolQuery::Local(), hbm_bdev_pool_id);
    reg.Wait();
    if (reg->GetReturnCode() != 0) {
      fprintf(stderr, "ERROR: RegisterTarget HBM (CPU) failed: %d\n",
              reg->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    auto gpu_reg = cte_client.AsyncRegisterTarget(
        "hbm::tiered_test_hbm",
        chimaera::bdev::BdevType::kHbm,
        50ULL * 1024 * 1024,
        chi::PoolQuery::Local(), hbm_bdev_pool_id,
        chi::PoolQuery::LocalGpuBcast());
    gpu_reg.Wait();
    if (gpu_reg->GetReturnCode() != 0) {
      fprintf(stderr, "ERROR: RegisterTarget HBM (GPU) failed: %d\n",
              gpu_reg->GetReturnCode());
      return 1;
    }
  }
  printf("HBM target registered.\n");

  // Register Pinned target (CPU + GPU)
  chi::PoolId pinned_bdev_pool_id(801, 0);
  {
    auto reg = cte_client.AsyncRegisterTarget(
        "pinned::tiered_test_pinned",
        chimaera::bdev::BdevType::kPinned,
        400ULL * 1024 * 1024,
        chi::PoolQuery::Local(), pinned_bdev_pool_id);
    reg.Wait();
    if (reg->GetReturnCode() != 0) {
      fprintf(stderr, "ERROR: RegisterTarget Pinned (CPU) failed: %d\n",
              reg->GetReturnCode());
      return 1;
    }
    std::this_thread::sleep_for(200ms);

    auto gpu_reg = cte_client.AsyncRegisterTarget(
        "pinned::tiered_test_pinned",
        chimaera::bdev::BdevType::kPinned,
        400ULL * 1024 * 1024,
        chi::PoolQuery::Local(), pinned_bdev_pool_id,
        chi::PoolQuery::LocalGpuBcast());
    gpu_reg.Wait();
    if (gpu_reg->GetReturnCode() != 0) {
      fprintf(stderr, "ERROR: RegisterTarget Pinned (GPU) failed: %d\n",
              gpu_reg->GetReturnCode());
      return 1;
    }
  }
  printf("Pinned target registered.\n");
  std::this_thread::sleep_for(200ms);
  printf("Target registered.\n");

  // Create tag
  auto tag_task = cte_client.AsyncGetOrCreateTag(
      "tiered_test_tag", wrp_cte::core::TagId::GetNull(),
      chi::PoolQuery::Local());
  tag_task.Wait();
  if (tag_task->GetReturnCode() != 0) {
    fprintf(stderr, "ERROR: GetOrCreateTag failed: %d\n",
            tag_task->GetReturnCode());
    return 1;
  }
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;

  auto gpu_tag = cte_client.AsyncGetOrCreateTag(
      "tiered_test_tag", tag_id, chi::PoolQuery::LocalGpuBcast());
  gpu_tag.Wait();
  std::this_thread::sleep_for(200ms);
  printf("Tag created: %u.%u\n\n", tag_id.major_, tag_id.minor_);

  // Run the GPU test
  printf("Launching GPU kernel...\n");
  int result = run_gpu_tiered_test(pool_id, tag_id, 120);

  printf("\n============================================================\n");
  if (result == 1) {
    printf("  PASSED\n");
  } else if (result == 0) {
    printf("  TIMEOUT\n");
  } else {
    printf("  FAILED (result=%d)\n", result);
    if (result <= -100 && result > -200) printf("  PutBlob alloc failed for blob %d\n", -(result+100));
    else if (result <= -200 && result > -300) printf("  PutBlob error\n");
    else if (result <= -300 && result > -400) printf("  GetBlob alloc failed for blob %d\n", -(result+300));
    else if (result <= -400 && result > -500) printf("  GetBlob error\n");
    else if (result <= -500) printf("  Verification: %d mismatches\n", -(result+500));
  }
  printf("============================================================\n");

  return (result == 1) ? 0 : 1;
}
