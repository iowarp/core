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

#ifndef WRP_CTE_ABSTRACT_MAPPER_H
#define WRP_CTE_ABSTRACT_MAPPER_H

#include "chimaera/chimaera.h"

namespace wrp::cae {

/**
 * Define different types of mappers supported by POSIX Adapter.
 * Also define its construction in the MapperFactory.
 */
enum class MapperType { kBalancedMapper };

/**
 A structure to represent BLOB placement
*/
struct BlobPlacement {
  size_t page_;       /**< The index in the array placements */
  size_t bucket_off_; /**< Offset from file start (for FS) */
  size_t blob_off_;   /**< Offset from BLOB start */
  size_t blob_size_;  /**< Size after offset to read */

  /** create a BLOB name from index. */
  static chi::priv::string CreateBlobName(size_t page) {
    chi::priv::string buf(HSHM_MALLOC, std::string(sizeof(page), '\0'));
    hipc::LocalSerialize srl(buf);
    srl << page;
    return buf;
  }

  /** create a BLOB name from index. */
  chi::priv::string CreateBlobName() const {
    chi::priv::string buf(HSHM_MALLOC, std::string(sizeof(page_), '\0'));
    hipc::LocalSerialize srl(buf);
    srl << page_;
    return buf;
  }

  /** decode \a blob_name BLOB name to index.  */
  template <typename StringT>
  void DecodeBlobName(const StringT& blob_name, size_t page_size) {
    hipc::LocalDeserialize srl(blob_name);
    srl >> page_;
    bucket_off_ = page_ * page_size;
    blob_off_ = 0;
  }
};

typedef std::vector<BlobPlacement> BlobPlacements;

/**
   A class to represent abstract mapper
*/
class AbstractMapper {
 public:
  /** Virtual destructor */
  virtual ~AbstractMapper() = default;

  /**
   * This method maps the current operation to Hermes data structures.
   *
   * @param off offset
   * @param size size
   * @param page_size the page division factor
   * @param ps BLOB placement
   *
   */
  virtual void map(size_t off, size_t size, size_t page_size,
                   BlobPlacements& ps) = 0;
};
}  // namespace wrp::cae

#endif  // WRP_CTE_ABSTRACT_MAPPER_H
