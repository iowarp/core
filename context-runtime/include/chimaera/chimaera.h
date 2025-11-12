#ifndef CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_

/**
 * Main header file for Chimaera distributed task execution framework
 *
 * This header provides the primary interface for both runtime and client
 * applications using the Chimaera framework.
 */

#include "chimaera/pool_query.h"
#include "chimaera/singletons.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"

namespace chi {

/**
 * Global initialization functions
 */

/**
 * Initialize Chimaera client components
 * @return true if initialization successful, false otherwise
 */
bool CHIMAERA_CLIENT_INIT();

/**
 * Initialize Chimaera runtime components
 * @return true if initialization successful, false otherwise
 */
bool CHIMAERA_RUNTIME_INIT();



}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_