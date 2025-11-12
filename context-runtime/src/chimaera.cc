/**
 * Main Chimaera initialization and global functions
 */

#include "chimaera/chimaera.h"
#include "chimaera/container.h"
#include "chimaera/work_orchestrator.h"

namespace chi {

bool CHIMAERA_CLIENT_INIT() {
  auto* chimaera_manager = CHI_CHIMAERA_MANAGER;
  return chimaera_manager->ClientInit();
}

bool CHIMAERA_RUNTIME_INIT() {
  auto* chimaera_manager = CHI_CHIMAERA_MANAGER;
  return chimaera_manager->ServerInit();
}

}  // namespace chi