#ifndef CHIMAERA_MOD_NAME_AUTOGEN_METHODS_H_
#define CHIMAERA_MOD_NAME_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for MOD_NAME
 */

namespace chimaera::MOD_NAME {

namespace Method {

enum : chi::u32 {
  // Inherited methods
  kCreate = 0,
  kDestroy = 1,
  kMonitor = 9,

  // MOD_NAME-specific methods
  kCustom = 10,
  kCoMutexTest = 20,
  kCoRwLockTest = 21,
  kWaitTest = 23,
  kTestLargeOutput = 24,
  kGpuSubmit = 25,

  kMaxMethodId = 26,
};

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "Custom";
    v[20] = "CoMutexTest";
    v[21] = "CoRwLockTest";
    v[23] = "WaitTest";
    v[24] = "TestLargeOutput";
    v[25] = "GpuSubmit";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace chimaera::MOD_NAME

#endif  // MOD_NAME_AUTOGEN_METHODS_H_
