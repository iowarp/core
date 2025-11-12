/**
 * Client implementation for MOD_NAME
 * 
 * Contains global variables and singletons for client-side code.
 */

#include "chimaera/MOD_NAME/MOD_NAME_client.h"
#include "chimaera/MOD_NAME/MOD_NAME_tasks.h"

namespace chimaera::MOD_NAME {

// Define static constexpr member for proper linkage when address is taken
constexpr const char* CreateParams::chimod_lib_name;

// Client implementation is mostly header-only
// This file exists for any global client-side state or initialization

} // namespace chimaera::MOD_NAME