#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

/**
 * Client implementation for bdev ChiMod
 * 
 * Provides the client-side implementation for block device operations
 */

namespace chimaera::bdev {

// Define static constexpr member for proper linkage when address is taken
constexpr const char* CreateParams::chimod_lib_name;

// Client implementation is header-only in this case
// All methods are implemented in bdev_client.h

} // namespace chimaera::bdev