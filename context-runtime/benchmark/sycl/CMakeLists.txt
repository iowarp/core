# SYCL workload benchmark suite — mirrors context-transfer-engine/benchmark/gpu
# but uses sycl::queue and the HSHM_DEVICE_* abstraction layer instead of CUDA
# runtime APIs and __global__ kernel launches.
#
# Built only when WRP_CORE_ENABLE_SYCL is on. The chimod _gpu.cc files are
# embedded in the executable directly because DPC++ device-side linkage does
# not propagate SYCL_EXTERNAL symbols across .so boundaries on the nvptx64
# backend (same constraint that test_sycl_chimods works around).

if(NOT WRP_CORE_ENABLE_SYCL)
  return()
endif()

# Sources for the driver TU and the four workload TUs.
set(WRP_SYCL_BENCH_SOURCES
  wrp_sycl_bench.cc
  workload_usm_bandwidth.cc
  workload_atomic_throughput.cc
  workload_orchestrator_lifecycle.cc
  workload_container_alloc.cc
  workload_cte_client_overhead_sycl.cc
  workload_bdev_sycl.cc
)

# Pull in the chimod _gpu.cc + work_orchestrator_sycl.cc sources so every
# HSHM_DEVICE_EXTERN definition sits in the same SYCL device-link unit as
# the autogen kernel that takes its address (see test_sycl_chimods comment).
set(WRP_SYCL_BENCH_DEVICE_SOURCES
  ${CMAKE_SOURCE_DIR}/context-runtime/src/gpu/work_orchestrator_sycl.cc
  ${CMAKE_SOURCE_DIR}/context-runtime/modules/MOD_NAME/src/MOD_NAME_runtime_gpu.cc
  ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/src/bdev_runtime_gpu.cc
  ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/src/core_runtime_gpu.cc
  ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/src/compressor_runtime_gpu.cc
)

add_sycl_executable(wrp_sycl_bench FALSE
  ${WRP_SYCL_BENCH_SOURCES}
  ${WRP_SYCL_BENCH_DEVICE_SOURCES}
  INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/src             # autogen/gpu_work_orchestrator_modules.h
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/MOD_NAME/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
  LINK_LIBS chimaera_cxx hshm::cxx ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(wrp_sycl_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS wrp_sycl_bench RUNTIME DESTINATION bin)

# Quick CTest entries — small iteration counts so the regression suite
# stays under 30s per case. Real performance runs use larger numbers.
#
# Tests pull in libsycl.so from the DPC++ install, so the environment
# must include both the DPC++ runtime lib dir (typically /opt/intel/dpcpp/lib
# in the dev container; override via DPCPP_LIB_DIR) and the build's bin dir
# (for libchimaera_cxx*.so).
if(NOT DEFINED DPCPP_LIB_DIR)
  set(DPCPP_LIB_DIR "/opt/intel/dpcpp/lib")
endif()
set(WRP_SYCL_BENCH_TEST_ENV
  "LD_LIBRARY_PATH=${DPCPP_LIB_DIR}:${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")

add_test(NAME sycl_bench_usm_bandwidth
  COMMAND wrp_sycl_bench --test-case usm_bandwidth --iterations 5)
set_tests_properties(sycl_bench_usm_bandwidth PROPERTIES
  TIMEOUT 60 LABELS "gpu;sycl;bench"
  ENVIRONMENT "${WRP_SYCL_BENCH_TEST_ENV}")

add_test(NAME sycl_bench_atomic_throughput
  COMMAND wrp_sycl_bench --test-case atomic_throughput
          --threads 1024 --iterations 100)
set_tests_properties(sycl_bench_atomic_throughput PROPERTIES
  TIMEOUT 60 LABELS "gpu;sycl;bench"
  ENVIRONMENT "${WRP_SYCL_BENCH_TEST_ENV}")

add_test(NAME sycl_bench_orchestrator_lifecycle
  COMMAND wrp_sycl_bench --test-case orchestrator_lifecycle --iterations 10)
set_tests_properties(sycl_bench_orchestrator_lifecycle PROPERTIES
  TIMEOUT 60 LABELS "gpu;sycl;bench"
  ENVIRONMENT "${WRP_SYCL_BENCH_TEST_ENV}")

add_test(NAME sycl_bench_container_alloc
  COMMAND wrp_sycl_bench --test-case container_alloc --iterations 50)
set_tests_properties(sycl_bench_container_alloc PROPERTIES
  TIMEOUT 60 LABELS "gpu;sycl;bench"
  ENVIRONMENT "${WRP_SYCL_BENCH_TEST_ENV}")

add_test(NAME sycl_bench_cte_client_overhead
  COMMAND wrp_sycl_bench --test-case cte_client_overhead
          --threads 1024 --iterations 100)
set_tests_properties(sycl_bench_cte_client_overhead PROPERTIES
  TIMEOUT 60 LABELS "gpu;sycl;bench;cte"
  ENVIRONMENT "${WRP_SYCL_BENCH_TEST_ENV}")

add_test(NAME sycl_bench_bdev_client
  COMMAND wrp_sycl_bench --test-case bdev_client
          --threads 1024 --iterations 100)
set_tests_properties(sycl_bench_bdev_client PROPERTIES
  TIMEOUT 60 LABELS "gpu;sycl;bench;bdev"
  ENVIRONMENT "${WRP_SYCL_BENCH_TEST_ENV}")
