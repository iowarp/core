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
 * @file transport_explicit_instantiations.cc
 * @brief Explicit template instantiations for Transport::Send to resolve LTO
 * linking errors
 *
 * This file forces template instantiation at library build time to prevent
 * undefined symbol errors when using Link-Time Optimization (LTO).
 *
 * The Transport::Send template is defined in hshm::lbm but instantiated with
 * chimaera task types. This file ensures these instantiations are exported
 * with proper visibility even when LTO is enabled.
 */

#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "hermes_shm/lightbeam/transport_factory_impl.h"

namespace hshm::lbm {

// Explicit instantiation for Transport::Send with chi task types
// These ensure symbols are exported even with LTO enabled
template HSHM_API int Transport::Send<chi::SaveTaskArchive>(
    chi::SaveTaskArchive& meta, const LbmContext& ctx);

// Add other chi:: task archive types as needed
// template HSHM_API int Transport::Send<chi::OtherTaskArchive>(...);

}  // namespace hshm::lbm
