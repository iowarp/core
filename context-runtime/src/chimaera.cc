/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Main Chimaera initialization and global functions
 */

#include "chimaera/chimaera.h"
#include "chimaera/container.h"
#include "chimaera/work_orchestrator.h"
#include <cstdlib>
#include <cstring>

namespace chi {

bool CHIMAERA_INIT(ChimaeraMode mode, bool default_with_runtime) {
  // Static guard to prevent double initialization
  static bool s_initialized = false;
  if (s_initialized) {
    return true;  // Already initialized, return success
  }

  // Suppress OS error dialogs so tests don't block on popups
  hshm::SystemInfo::SuppressErrorDialogs();

  auto* chimaera_manager = CHI_CHIMAERA_MANAGER;

  // Check environment variable CHI_WITH_RUNTIME
  // Use hshm::SystemInfo::Getenv for Windows compatibility (std::getenv
  // doesn't reflect changes made via SetEnvironmentVariable/Setenv)
  bool with_runtime = default_with_runtime;
  std::string env_val = hshm::SystemInfo::Getenv("CHI_WITH_RUNTIME");
  if (!env_val.empty()) {
    with_runtime = (env_val == "1" || env_val == "true" || env_val == "TRUE");
  }

  // Determine what to initialize based on mode and with_runtime flag
  bool init_runtime = false;
  bool init_client = false;

  if (mode == ChimaeraMode::kServer || mode == ChimaeraMode::kRuntime) {
    // Server/Runtime mode: always start runtime
    init_runtime = true;
    init_client = true;  // Runtime also needs client components
  } else {
    // Client mode
    init_client = true;
    init_runtime = with_runtime;
  }

  // Initialize runtime first if needed
  if (init_runtime) {
    if (!chimaera_manager->ServerInit()) {
      return false;
    }
  }

  // Initialize client components
  if (init_client) {
    if (!chimaera_manager->ClientInit()) {
      return false;
    }
  }

  // Mark as initialized on success
  s_initialized = true;
  return true;
}

void CHIMAERA_FINALIZE() {
  auto* chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsInitialized()) {
    chimaera_manager->ServerFinalize();
    chimaera_manager->ClientFinalize();
  }
}

}  // namespace chi
