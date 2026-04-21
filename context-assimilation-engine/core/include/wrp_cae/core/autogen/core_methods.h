#ifndef WRP_CAE_CORE_AUTOGEN_METHODS_H_
#define WRP_CAE_CORE_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for core
 */

namespace wrp_cae::core {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST chi::u32 kCreate = 0;
GLOBAL_CROSS_CONST chi::u32 kDestroy = 1;
GLOBAL_CROSS_CONST chi::u32 kMonitor = 9;

// core-specific methods
GLOBAL_CROSS_CONST chi::u32 kParseOmni = 10;
GLOBAL_CROSS_CONST chi::u32 kProcessHdf5Dataset = 11;
GLOBAL_CROSS_CONST chi::u32 kExportData = 12;

GLOBAL_CROSS_CONST chi::u32 kMaxMethodId = 13;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "ParseOmni";
    v[11] = "ProcessHdf5Dataset";
    v[12] = "ExportData";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace wrp_cae::core

#endif  // CORE_AUTOGEN_METHODS_H_
