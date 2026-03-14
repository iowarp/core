#ifndef WRP_CTE_COMPRESSOR_AUTOGEN_METHODS_H_
#define WRP_CTE_COMPRESSOR_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for compressor
 */

namespace wrp_cte::compressor {

namespace Method {

enum : chi::u32 {
  // Inherited methods
  kCreate = 0,
  kDestroy = 1,
  kMonitor = 9,

  // compressor-specific methods
  kDynamicSchedule = 10,
  kCompress = 11,
  kDecompress = 12,

  kMaxMethodId = 13,
};

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "DynamicSchedule";
    v[11] = "Compress";
    v[12] = "Decompress";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace wrp_cte::compressor

#endif  // COMPRESSOR_AUTOGEN_METHODS_H_
