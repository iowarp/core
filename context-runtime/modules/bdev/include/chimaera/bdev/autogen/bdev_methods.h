#ifndef CHIMAERA_BDEV_AUTOGEN_METHODS_H_
#define CHIMAERA_BDEV_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

/**
 * Auto-generated method definitions for bdev
 */

namespace chimaera::bdev {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;

// bdev-specific methods
GLOBAL_CONST chi::u32 kAllocateBlocks = 10;
GLOBAL_CONST chi::u32 kFreeBlocks = 11;
GLOBAL_CONST chi::u32 kWrite = 12;
GLOBAL_CONST chi::u32 kRead = 13;
GLOBAL_CONST chi::u32 kGetStats = 14;
}  // namespace Method

}  // namespace chimaera::bdev

#endif  // BDEV_AUTOGEN_METHODS_H_
