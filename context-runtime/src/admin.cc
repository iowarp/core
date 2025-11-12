/**
 * Admin ChiMod singleton implementation
 */

#include "chimaera/admin.h"
#include "chimaera/admin/admin_client.h"

// Global pointer variable definition for Admin singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chimaera::admin::Client, g_admin);
