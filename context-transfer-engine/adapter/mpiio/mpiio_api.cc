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

bool mpiio_intercepted = true;

#include "mpiio_api.h"

#include "hermes_shm/util/singleton.h"
#include "mpiio_fs_api.h"
#include "wrp_cte/core/core_client.h"

// #define WRP_CTE_DISABLE_MPIIO

/**
 * Namespace declarations
 */
using wrp::cae::AdapterStat;
using wrp::cae::File;
using wrp::cae::FsIoOptions;
using wrp::cae::IoStatus;
using wrp::cae::MetadataManager;
using wrp::cae::MpiioApi;
using wrp::cae::MpiioFs;
using wrp::cae::MpiioSeekModeConv;
using wrp::cae::SeekMode;

extern "C" {

/**
 * MPI
 */
int WRP_CTE_DECL(MPI_Init)(int* argc, char*** argv) {
  HLOG(kDebug, "MPI Init intercepted.");
  wrp_cte::core::WRP_CTE_CLIENT_INIT();
  auto real_api = WRP_CTE_MPIIO_API;
  return real_api->MPI_Init(argc, argv);
}

int WRP_CTE_DECL(MPI_Finalize)(void) {
  HLOG(kDebug, "MPI Finalize intercepted.");
  auto real_api = WRP_CTE_MPIIO_API;
  return real_api->MPI_Finalize();
}

int WRP_CTE_DECL(MPI_Wait)(MPI_Request* req, MPI_Status* status) {
  HLOG(kDebug, "In MPI_Wait.");
  auto fs_api = WRP_CTE_MPIIO_FS;
  return fs_api->Wait(req, status);
}

int WRP_CTE_DECL(MPI_Waitall)(int count, MPI_Request* req, MPI_Status* status) {
  HLOG(kDebug, "In MPI_Waitall.");
  auto fs_api = WRP_CTE_MPIIO_FS;
  return fs_api->WaitAll(count, req, status);
}

/**
 * Metadata functions
 */
int WRP_CTE_DECL(MPI_File_open)(MPI_Comm comm, const char* filename, int amode,
                                MPI_Info info, MPI_File* fh) {
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsPathTracked(filename)) {
    HLOG(kDebug, "Intercept MPI_File_open ({}) for filename: {} and mode {}",
         (void*)MPI_File_open, filename, amode);
    AdapterStat stat;
    stat.comm_ = comm;
    stat.amode_ = amode;
    stat.info_ = info;
    File f = fs_api->Open(stat, filename);
    (*fh) = stat.mpi_fh_;
    return f.mpi_status_;
  }
#endif
  HLOG(kDebug, "NOT intercept MPI_File_open ({}) for filename: {} and mode {}",
       (void*)MPI_File_open, filename, amode);
  return real_api->MPI_File_open(comm, filename, amode, info, fh);
}

int WRP_CTE_DECL(MPI_File_close)(MPI_File* fh) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(fh)) {
    HLOG(kDebug, "Intercept MPI_File_close");
    File f;
    f.hermes_mpi_fh_ = *fh;
    return fs_api->Close(f, stat_exists);
  }
#endif
  return real_api->MPI_File_close(fh);
}

int WRP_CTE_DECL(MPI_File_seek)(MPI_File fh, MPI_Offset offset, int whence) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_seek");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->Seek(f, stat_exists, offset, whence);
  }
#endif
  return real_api->MPI_File_seek(fh, offset, whence);
}

int WRP_CTE_DECL(MPI_File_seek_shared)(MPI_File fh, MPI_Offset offset,
                                       int whence) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_seek_shared offset: {} whence: {}", offset,
         whence);
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->SeekShared(f, stat_exists, offset, whence);
  }
#endif
  return real_api->MPI_File_seek_shared(fh, offset, whence);
}

int WRP_CTE_DECL(MPI_File_get_position)(MPI_File fh, MPI_Offset* offset) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_get_position");
    File f;
    f.hermes_mpi_fh_ = fh;
    (*offset) = static_cast<MPI_Offset>(fs_api->Tell(f, stat_exists));
    return MPI_SUCCESS;
  }
#endif
  return real_api->MPI_File_get_position(fh, offset);
}

int WRP_CTE_DECL(MPI_File_read_all)(MPI_File fh, void* buf, int count,
                                    MPI_Datatype datatype, MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_read_all");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->ReadAll(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_read_all(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_read_at_all)(MPI_File fh, MPI_Offset offset,
                                       void* buf, int count,
                                       MPI_Datatype datatype,
                                       MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_read_at_all");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->ReadAll(f, stat_exists, buf, offset, count, datatype,
                           status);
  }
#endif
  return real_api->MPI_File_read_at_all(fh, offset, buf, count, datatype,
                                        status);
}
int WRP_CTE_DECL(MPI_File_read_at)(MPI_File fh, MPI_Offset offset, void* buf,
                                   int count, MPI_Datatype datatype,
                                   MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_read_at");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->Read(f, stat_exists, buf, offset, count, datatype, status);
  }
#endif
  return real_api->MPI_File_read_at(fh, offset, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_read)(MPI_File fh, void* buf, int count,
                                MPI_Datatype datatype, MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_read");
    File f;
    f.hermes_mpi_fh_ = fh;
    int ret = fs_api->Read(f, stat_exists, buf, count, datatype, status);
    if (stat_exists) return ret;
  }
#endif
  return real_api->MPI_File_read(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_read_ordered)(MPI_File fh, void* buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_read_ordered");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->ReadOrdered(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_read_ordered(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_read_shared)(MPI_File fh, void* buf, int count,
                                       MPI_Datatype datatype,
                                       MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_read_shared");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->Read(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_read_shared(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_write_all)(MPI_File fh, const void* buf, int count,
                                     MPI_Datatype datatype,
                                     MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_write_all");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->WriteAll(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_write_all(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_write_at_all)(MPI_File fh, MPI_Offset offset,
                                        const void* buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_write_at_all");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->WriteAll(f, stat_exists, buf, offset, count, datatype,
                            status);
  }
#endif
  return real_api->MPI_File_write_at_all(fh, offset, buf, count, datatype,
                                         status);
}
int WRP_CTE_DECL(MPI_File_write_at)(MPI_File fh, MPI_Offset offset,
                                    const void* buf, int count,
                                    MPI_Datatype datatype, MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_write_at");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->Write(f, stat_exists, buf, offset, count, datatype, status);
  }
#endif
  return real_api->MPI_File_write_at(fh, offset, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_write)(MPI_File fh, const void* buf, int count,
                                 MPI_Datatype datatype, MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_write");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->Write(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_write(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_write_ordered)(MPI_File fh, const void* buf,
                                         int count, MPI_Datatype datatype,
                                         MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_write_ordered");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->WriteOrdered(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_write_ordered(fh, buf, count, datatype, status);
}
int WRP_CTE_DECL(MPI_File_write_shared)(MPI_File fh, const void* buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Status* status) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    // NOTE(llogan): originally WriteOrdered
    HLOG(kDebug, "Intercept MPI_File_write_shared");
    File f;
    f.hermes_mpi_fh_ = fh;
    return fs_api->WriteOrdered(f, stat_exists, buf, count, datatype, status);
  }
#endif
  return real_api->MPI_File_write_shared(fh, buf, count, datatype, status);
}

/**
 * Async Read/Write
 */
int WRP_CTE_DECL(MPI_File_iread_at)(MPI_File fh, MPI_Offset offset, void* buf,
                                    int count, MPI_Datatype datatype,
                                    MPI_Request* request) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_iread_at");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->ARead(f, stat_exists, buf, offset, count, datatype, request);
    return MPI_SUCCESS;
  }
#endif
  return real_api->MPI_File_iread_at(fh, offset, buf, count, datatype, request);
}
int WRP_CTE_DECL(MPI_File_iread)(MPI_File fh, void* buf, int count,
                                 MPI_Datatype datatype, MPI_Request* request) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_iread");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->ARead(f, stat_exists, buf, count, datatype, request);
  }
#endif
  return real_api->MPI_File_iread(fh, buf, count, datatype, request);
}
int WRP_CTE_DECL(MPI_File_iread_shared)(MPI_File fh, void* buf, int count,
                                        MPI_Datatype datatype,
                                        MPI_Request* request) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_iread_shared");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->ARead(f, stat_exists, buf, count, datatype, request);
    return MPI_SUCCESS;
  }
#endif
  return real_api->MPI_File_iread_shared(fh, buf, count, datatype, request);
}
int WRP_CTE_DECL(MPI_File_iwrite_at)(MPI_File fh, MPI_Offset offset,
                                     const void* buf, int count,
                                     MPI_Datatype datatype,
                                     MPI_Request* request) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_iwrite_at");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->AWrite(f, stat_exists, buf, offset, count, datatype, request);
    return MPI_SUCCESS;
  }
#endif
  return real_api->MPI_File_iwrite_at(fh, offset, buf, count, datatype,
                                      request);
}

int WRP_CTE_DECL(MPI_File_iwrite)(MPI_File fh, const void* buf, int count,
                                  MPI_Datatype datatype, MPI_Request* request) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_iwrite");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->AWrite(f, stat_exists, buf, count, datatype, request);
    return MPI_SUCCESS;
  }
#endif
  return real_api->MPI_File_iwrite(fh, buf, count, datatype, request);
}
int WRP_CTE_DECL(MPI_File_iwrite_shared)(MPI_File fh, const void* buf,
                                         int count, MPI_Datatype datatype,
                                         MPI_Request* request) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_iwrite_shared");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->AWriteOrdered(f, stat_exists, buf, count, datatype, request);
    return MPI_SUCCESS;
  }
#endif
  return real_api->MPI_File_iwrite_shared(fh, buf, count, datatype, request);
}

/**
 * Other functions
 */
int WRP_CTE_DECL(MPI_File_sync)(MPI_File fh) {
  bool stat_exists;
  auto real_api = WRP_CTE_MPIIO_API;
  auto fs_api = WRP_CTE_MPIIO_FS;
#ifndef WRP_CTE_DISABLE_MPIIO
  if (fs_api->IsMpiFpTracked(&fh)) {
    HLOG(kDebug, "Intercept MPI_File_sync");
    File f;
    f.hermes_mpi_fh_ = fh;
    fs_api->Sync(f, stat_exists);
    return 0;
  }
#endif
  return real_api->MPI_File_sync(fh);
}

}  // extern C
