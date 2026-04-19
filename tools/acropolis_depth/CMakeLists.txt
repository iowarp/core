# Acropolis CLI tool — set/get/clear indexing depth xattrs on files.
# Linux-only (uses <sys/xattr.h>). Build skipped on non-Linux platforms.

if(UNIX AND NOT APPLE)
  add_executable(acropolis-depth set_depth.cc)
  set_target_properties(acropolis-depth PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  install(TARGETS acropolis-depth DESTINATION bin)
endif()
