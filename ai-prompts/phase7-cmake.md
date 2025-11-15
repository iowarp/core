Let's make a single cmake directory at the root of the project. I want to unify each subdirectory cmake folders into one cohesive IowarpCore.

  It should have:
  IowarpCoreCommon.cmake, which will have functions we want new repos to inherit.
  IowarpCoreConfig.cmake.in, which will have a version number and include to the Common.cmake.

  We should consolidate the parameter lists. Most HSHM parameters should disappear. Most parameters were for turning on and off certain libraries. We should make these global settings. For
  example HSHM_ENABLE_MPI should become WRP_CORE_ENABLE_MPI. It will disable all MPI stuff in the project if disabled.

  We should migrate all find_package commands to the root cmake. Delete all context-* subdirectory cmake directories afterwards.

  Update CMakePresets.json afterwards as well. Ensure everything builds afterwards


  Let's make RPATH a configuration option, not a requirement. WRP_CORE_ENABLE_RPATH OFF.


Again, I only want two files in cmake. No individual component files. Just two files both for the core. I want all find_package, pkg_check_modules, and whatever out of the common 
  configuration and placed in the root cmake. If there is any code that does find_package(HermesShm, Chimaera, etc) pr any other package defined in this repo as either a submodule or actual code, it should be removed. 


Let's change the way hshm get's compiled. We should have the following targets:
hshm::cxx, cuda_cxx, rocm_cxx. these can stay. However we should have individual targets for the dependencies.

hshm::lightbeam, hshm::thread_all, hshm::mpi, hshm::compress, hshm::encrypt

lightbeam will include zeromq, thallium if enabled.
thread_all will include thallium if enabled.
mpi will include mpi if enabled

hshm components should not link to boost at all. Unit tests depending on it
should link to it. Chimaera runtime should link to boost directly. 
chimaera clients should link to only hshm::cxx.