/**
 * Admin ChiMod singleton
 *
 * Provides global access to the admin ChiMod client via CHI_ADMIN macro.
 * The admin container is automatically created by the runtime and accessed
 * by clients without requiring explicit Create() calls.
 */

#ifndef CHIMAERA_INCLUDE_CHIMAERA_ADMIN_H_
#define CHIMAERA_INCLUDE_CHIMAERA_ADMIN_H_

#include "chimaera/types.h"

// Forward declaration to avoid circular dependency
namespace chimaera::admin {
class Client;
}

// Global pointer variable declaration for Admin singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chimaera::admin::Client, g_admin);

// Macro for accessing the Admin singleton using global pointer variable
#define CHI_ADMIN HSHM_GET_GLOBAL_PTR_VAR(::chimaera::admin::Client, g_admin)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_ADMIN_H_
