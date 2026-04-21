#------------------------------------------------------------------------------
# GPU Unit Tests
#------------------------------------------------------------------------------

if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
  message(STATUS "Building GPU unit tests")

  set(HSHM_GPU_INCLUDE_DIRS
    ${HSHM_ROOT}/include
    ${PROJECT_BINARY_DIR}/include
  )

  # GpuShmMmap test
  add_cuda_executable(test_gpu_shm_mmap TRUE test_gpu_shm_mmap.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS})
  target_link_libraries(test_gpu_shm_mmap
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_shm_mmap COMMAND test_gpu_shm_mmap)
  set_tests_properties(test_gpu_shm_mmap PROPERTIES TIMEOUT 60)

  # GpuMalloc test (commented out — not well-written)
  # add_cuda_executable(test_gpu_malloc TRUE test_gpu_malloc.cc)
  # target_link_libraries(test_gpu_malloc
  #   hshm::cuda_cxx
  #   Catch2::Catch2WithMain
  # )
  # add_test(NAME test_gpu_malloc COMMAND test_gpu_malloc)

  # LocalSerialize GPU test
  add_cuda_executable(test_local_serialize_gpu TRUE test_local_serialize_gpu.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS})
  target_link_libraries(test_local_serialize_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_local_serialize_gpu COMMAND test_local_serialize_gpu)
  set_tests_properties(test_local_serialize_gpu PROPERTIES TIMEOUT 60)

  # LocalTransfer GPU test
  add_cuda_executable(test_local_transfer_gpu TRUE test_local_transfer_gpu.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS})
  target_link_libraries(test_local_transfer_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_local_transfer_gpu COMMAND test_local_transfer_gpu)
  set_tests_properties(test_local_transfer_gpu PROPERTIES TIMEOUT 60)

  # BuddyAllocator GPU test
  add_cuda_executable(test_buddy_alloc_gpu TRUE test_buddy_alloc_gpu.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS})
  target_link_libraries(test_buddy_alloc_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_buddy_alloc_gpu COMMAND test_buddy_alloc_gpu)

  # PrivateBuddyAllocator shifted (stack) GPU test
  add_cuda_executable(test_buddy_alloc_stack_gpu TRUE test_buddy_alloc_stack_gpu.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS})
  target_link_libraries(test_buddy_alloc_stack_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_buddy_alloc_stack_gpu COMMAND test_buddy_alloc_stack_gpu)

  # ThreadAllocator GPU test
  add_cuda_executable(test_thread_alloc_gpu TRUE test_thread_alloc_gpu.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS})
  target_link_libraries(test_thread_alloc_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_thread_alloc_gpu COMMAND test_thread_alloc_gpu)

  # RoundRobinAllocator CDP test
  find_package(CUDAToolkit QUIET)
  add_cuda_executable(test_round_robin_alloc_gpu TRUE test_round_robin_alloc_gpu.cc
    INCLUDE_DIRS ${HSHM_GPU_INCLUDE_DIRS}
    DEFS CUDA_FORCE_CDP1_IF_SUPPORTED __CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING)
  target_link_libraries(test_round_robin_alloc_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
    cudadevrt
  )
  target_compile_options(test_round_robin_alloc_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -rdc=true>)
  add_test(NAME test_round_robin_alloc_gpu COMMAND test_round_robin_alloc_gpu)
  set_tests_properties(test_round_robin_alloc_gpu PROPERTIES
    TIMEOUT 60 LABELS "gpu;allocator")

  add_subdirectory(runtime)

endif()

if(WRP_CORE_ENABLE_SYCL)
  message(STATUS "Building SYCL GPU unit tests (Intel oneAPI)")
  add_executable(test_gpu_sycl test_gpu_sycl.cc)
  target_compile_definitions(test_gpu_sycl PRIVATE HSHM_ENABLE_SYCL=1)
  wrp_core_apply_sycl_flags(test_gpu_sycl)
  target_link_libraries(test_gpu_sycl
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_sycl COMMAND test_gpu_sycl)
  set_tests_properties(test_gpu_sycl PROPERTIES LABELS "gpu;sycl")
endif()

if(NOT WRP_CORE_ENABLE_CUDA AND NOT WRP_CORE_ENABLE_ROCM AND NOT WRP_CORE_ENABLE_SYCL)
  message(STATUS "GPU tests disabled (WRP_CORE_ENABLE_CUDA, WRP_CORE_ENABLE_ROCM, and WRP_CORE_ENABLE_SYCL are all OFF)")
endif()
