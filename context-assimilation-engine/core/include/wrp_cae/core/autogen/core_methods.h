#ifndef WRP_CAE_CORE_AUTOGEN_METHODS_H_
#define WRP_CAE_CORE_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

/**
 * Auto-generated method definitions for core
 */

namespace wrp_cae::core {

namespace Method {

enum : chi::u32 {
  // Inherited methods
  kCreate = 0,
  kDestroy = 1,
  kMonitor = 9,

  // core-specific methods
  kParseOmni = 10,
  kProcessHdf5Dataset = 11,
  kExportData = 12,

  kMaxMethodId = 13,
};
}  // namespace Method

}  // namespace wrp_cae::core

#endif  // CORE_AUTOGEN_METHODS_H_
