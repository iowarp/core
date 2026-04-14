/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * IOWarp HDF5 VOL Connector
 *
 * Intercepts H5Dwrite/H5Dread and maps them to CTE AsyncPutBlob/AsyncGetBlob.
 * Large writes are split into configurable-size chunks and submitted
 * asynchronously. All other HDF5 operations pass through to the native VOL.
 */

#include "iowarp_vol.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/core/content_transfer_engine.h>

/* ========================================================================
 * Internal state structures
 * ======================================================================== */

struct iowarp_obj_t {
  void  *under_object;
  hid_t  under_vol_id;
};

struct iowarp_file_t {
  iowarp_obj_t obj;
  wrp_cte::core::TagId tag_id;
  std::string file_name;
  size_t chunk_size;
};

struct iowarp_dataset_t {
  iowarp_obj_t obj;
  iowarp_file_t *file;
  std::string dataset_path;
  /* Pending async writes flushed on close */
  std::vector<chi::Future<wrp_cte::core::PutBlobTask>> pending_puts;
  std::vector<hipc::FullPtr<char>> pending_buffers;
};

/* ========================================================================
 * Helper: Get CTE client
 * ======================================================================== */

static wrp_cte::core::Client *get_cte_client() {
  return WRP_CTE_CLIENT;
}

/* ========================================================================
 * Info callbacks
 * ======================================================================== */

static void *iowarp_info_copy(const void *_info) {
  const auto *info = static_cast<const iowarp_vol_info_t *>(_info);
  auto *new_info = new iowarp_vol_info_t(*info);
  if (info->under_vol_info) {
    H5VLcopy_connector_info(info->under_vol_id, &new_info->under_vol_info,
                            info->under_vol_info);
  }
  return new_info;
}

static herr_t iowarp_info_free(void *_info) {
  auto *info = static_cast<iowarp_vol_info_t *>(_info);
  if (info->under_vol_info) {
    H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
  }
  delete info;
  return 0;
}

/* ========================================================================
 * Wrap / unwrap callbacks
 * ======================================================================== */

static void *iowarp_wrap_get_object(const void *obj) {
  auto *o = static_cast<const iowarp_obj_t *>(obj);
  return H5VLget_object(o->under_object, o->under_vol_id);
}

static herr_t iowarp_get_wrap_ctx(const void *obj, void **wrap_ctx) {
  auto *o = static_cast<const iowarp_obj_t *>(obj);
  return H5VLget_wrap_ctx(o->under_object, o->under_vol_id, wrap_ctx);
}

static void *iowarp_wrap_object(void *under_obj, H5I_type_t obj_type,
                                void *wrap_ctx) {
  (void)obj_type; (void)wrap_ctx;
  /* For passthrough objects (groups, attributes, etc.) */
  auto *o = new iowarp_obj_t;
  o->under_object = under_obj;
  o->under_vol_id = H5VL_NATIVE;
  return o;
}

static void *iowarp_unwrap_object(void *obj) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  void *under = o->under_object;
  delete o;
  return under;
}

static herr_t iowarp_free_wrap_ctx(void *wrap_ctx) {
  (void)wrap_ctx;
  return 0;
}

/* ========================================================================
 * File callbacks
 * ======================================================================== */

static void *iowarp_file_create(const char *name, unsigned flags,
                                hid_t fcpl_id, hid_t fapl_id,
                                hid_t dxpl_id, void **req) {
  /* Get underlying VOL info */
  hid_t under_vol_id = H5VL_NATIVE;
  void *under_vol_info = nullptr;

  /* Check if user provided IOWarp VOL info */
  iowarp_vol_info_t *vol_info = nullptr;
  H5Pget_vol(fapl_id, nullptr, reinterpret_cast<void **>(&vol_info));
  size_t chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
  if (vol_info) {
    under_vol_id = vol_info->under_vol_id;
    under_vol_info = vol_info->under_vol_info;
    if (vol_info->chunk_size > 0) {
      chunk_size = vol_info->chunk_size;
    }
  }

  /* Check environment variable for chunk size */
  const char *env_chunk = std::getenv("IOWARP_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
  }

  /* Create file via native VOL */
  hid_t native_fapl = H5Pcopy(fapl_id);
  H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
  void *under_file = H5VLfile_create(name, flags, fcpl_id, native_fapl,
                                      dxpl_id, req);
  H5Pclose(native_fapl);
  if (!under_file) return nullptr;

  /* Create CTE tag for this file */
  auto *cte_client = get_cte_client();
  auto tag_task = cte_client->AsyncGetOrCreateTag(std::string("hdf5:") + name);
  tag_task.Wait();

  /* Create file state */
  auto *file = new iowarp_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->tag_id = tag_task->tag_id_;
  file->file_name = name;
  file->chunk_size = chunk_size;

  return file;
}

static void *iowarp_file_open(const char *name, unsigned flags,
                              hid_t fapl_id, hid_t dxpl_id, void **req) {
  size_t chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
  const char *env_chunk = std::getenv("IOWARP_VOL_CHUNK_SIZE");
  if (env_chunk) {
    chunk_size = std::strtoul(env_chunk, nullptr, 10);
    if (chunk_size == 0) chunk_size = IOWARP_VOL_DEFAULT_CHUNK_SIZE;
  }

  /* Open file via native VOL */
  hid_t native_fapl = H5Pcopy(fapl_id);
  H5Pset_vol(native_fapl, H5VL_NATIVE, nullptr);
  void *under_file = H5VLfile_open(name, flags, native_fapl, dxpl_id, req);
  H5Pclose(native_fapl);
  if (!under_file) return nullptr;

  /* Get or create CTE tag */
  auto *cte_client = get_cte_client();
  auto tag_task = cte_client->AsyncGetOrCreateTag(std::string("hdf5:") + name);
  tag_task.Wait();

  auto *file = new iowarp_file_t;
  file->obj.under_object = under_file;
  file->obj.under_vol_id = H5VL_NATIVE;
  file->tag_id = tag_task->tag_id_;
  file->file_name = name;
  file->chunk_size = chunk_size;

  return file;
}

static herr_t iowarp_file_get(void *obj, H5VL_file_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  return H5VLfile_get(file->obj.under_object, file->obj.under_vol_id,
                      args, dxpl_id, req);
}

static herr_t iowarp_file_specific(void *obj,
                                   H5VL_file_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  return H5VLfile_specific(file->obj.under_object, file->obj.under_vol_id,
                           args, dxpl_id, req);
}

static herr_t iowarp_file_close(void *obj, hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  herr_t ret = H5VLfile_close(file->obj.under_object, file->obj.under_vol_id,
                               dxpl_id, req);
  delete file;
  return ret;
}

/* ========================================================================
 * Dataset callbacks
 * ======================================================================== */

static void *iowarp_dataset_create(void *obj,
                                   const H5VL_loc_params_t *loc_params,
                                   const char *name, hid_t lcpl_id,
                                   hid_t type_id, hid_t space_id,
                                   hid_t dcpl_id, hid_t dapl_id,
                                   hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);

  /* Create dataset via native VOL */
  void *under_dset = H5VLdataset_create(
      file->obj.under_object, loc_params, file->obj.under_vol_id, name,
      lcpl_id, type_id, space_id, dcpl_id, dapl_id, dxpl_id, req);
  if (!under_dset) return nullptr;

  auto *dset = new iowarp_dataset_t;
  dset->obj.under_object = under_dset;
  dset->obj.under_vol_id = file->obj.under_vol_id;
  dset->file = file;
  dset->dataset_path = name ? name : "";

  return dset;
}

static void *iowarp_dataset_open(void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 const char *name, hid_t dapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);

  void *under_dset = H5VLdataset_open(
      file->obj.under_object, loc_params, file->obj.under_vol_id, name,
      dapl_id, dxpl_id, req);
  if (!under_dset) return nullptr;

  auto *dset = new iowarp_dataset_t;
  dset->obj.under_object = under_dset;
  dset->obj.under_vol_id = file->obj.under_vol_id;
  dset->file = file;
  dset->dataset_path = name ? name : "";

  return dset;
}

/**
 * Dataset write: chunk data into async PutBlob calls.
 *
 * Each chunk is copied to shared memory and submitted via AsyncPutBlob.
 * Futures are collected in the dataset state and flushed on close.
 */
static herr_t iowarp_dataset_write(size_t count, void *dset[],
                                   hid_t mem_type_id[],
                                   hid_t mem_space_id[],
                                   hid_t file_space_id[],
                                   hid_t dxpl_id, const void *buf[],
                                   void **req) {
  auto *cte_client = get_cte_client();

  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<iowarp_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;

    /* Compute total data size from memory dataspace */
    hid_t space = mem_space_id[d];
    if (space == H5S_ALL) {
      /* Get dataspace from the native dataset */
      H5VL_dataset_get_args_t get_args;
      get_args.op_type = H5VL_DATASET_GET_SPACE;
      get_args.args.get_space.space_id = H5I_INVALID_HID;
      H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id,
                       &get_args, dxpl_id, nullptr);
      space = get_args.args.get_space.space_id;
    }
    hssize_t nelem = H5Sget_simple_extent_npoints(space);
    if (nelem <= 0) continue;

    size_t type_size = H5Tget_size(mem_type_id[d]);
    size_t total_size = static_cast<size_t>(nelem) * type_size;
    size_t chunk_size = dataset->file->chunk_size;
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    const char *src = static_cast<const char *>(buf[d]);

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t offset = i * chunk_size;
      size_t this_size = std::min(chunk_size, total_size - offset);

      /* Allocate SHM buffer and copy data */
      auto buffer = CHI_IPC->AllocateBuffer(this_size);
      if (buffer.IsNull()) return -1;
      std::memcpy(buffer.ptr_, src + offset, this_size);

      hipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
      std::string blob_name = dataset->dataset_path + "/chunk_" +
                              std::to_string(i);

      auto future = cte_client->AsyncPutBlob(
          dataset->file->tag_id, blob_name, offset, this_size,
          blob_data, -1.0f, wrp_cte::core::Context(), 0);

      dataset->pending_puts.push_back(std::move(future));
      dataset->pending_buffers.push_back(std::move(buffer));
    }

    /* Also write to native VOL for metadata consistency */
    H5VLdataset_write(1, &dataset->obj.under_object, &mem_type_id[d],
                       &mem_space_id[d], &file_space_id[d],
                       dataset->obj.under_vol_id, dxpl_id, &buf[d], req);
  }

  return 0;
}

/**
 * Dataset read: chunk requests into async GetBlob calls,
 * wait for all, then assemble into output buffer.
 */
static herr_t iowarp_dataset_read(size_t count, void *dset[],
                                  hid_t mem_type_id[],
                                  hid_t mem_space_id[],
                                  hid_t file_space_id[],
                                  hid_t dxpl_id, void *buf[],
                                  void **req) {
  auto *cte_client = get_cte_client();

  for (size_t d = 0; d < count; ++d) {
    auto *dataset = static_cast<iowarp_dataset_t *>(dset[d]);
    if (!dataset || !buf[d]) continue;

    /* Compute total data size */
    hid_t space = mem_space_id[d];
    if (space == H5S_ALL) {
      H5VL_dataset_get_args_t get_args;
      get_args.op_type = H5VL_DATASET_GET_SPACE;
      get_args.args.get_space.space_id = H5I_INVALID_HID;
      H5VLdataset_get(dataset->obj.under_object, dataset->obj.under_vol_id,
                       &get_args, dxpl_id, nullptr);
      space = get_args.args.get_space.space_id;
    }
    hssize_t nelem = H5Sget_simple_extent_npoints(space);
    if (nelem <= 0) continue;

    size_t type_size = H5Tget_size(mem_type_id[d]);
    size_t total_size = static_cast<size_t>(nelem) * type_size;
    size_t chunk_size = dataset->file->chunk_size;
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    char *dst = static_cast<char *>(buf[d]);

    /* Submit async GetBlob for each chunk */
    std::vector<chi::Future<wrp_cte::core::GetBlobTask>> futures;
    std::vector<hipc::FullPtr<char>> buffers;

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t offset = i * chunk_size;
      size_t this_size = std::min(chunk_size, total_size - offset);

      auto buffer = CHI_IPC->AllocateBuffer(this_size);
      if (buffer.IsNull()) return -1;

      hipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
      std::string blob_name = dataset->dataset_path + "/chunk_" +
                              std::to_string(i);

      auto future = cte_client->AsyncGetBlob(
          dataset->file->tag_id, blob_name, offset, this_size,
          0, blob_data);

      futures.push_back(std::move(future));
      buffers.push_back(std::move(buffer));
    }

    /* Wait for all chunks and copy to output */
    for (size_t i = 0; i < futures.size(); ++i) {
      futures[i].Wait();
      size_t offset = i * chunk_size;
      size_t this_size = std::min(chunk_size, total_size - offset);
      std::memcpy(dst + offset, buffers[i].ptr_, this_size);
    }
  }

  return 0;
}

static herr_t iowarp_dataset_get(void *obj, H5VL_dataset_get_args_t *args,
                                 hid_t dxpl_id, void **req) {
  auto *dset = static_cast<iowarp_dataset_t *>(obj);
  return H5VLdataset_get(dset->obj.under_object, dset->obj.under_vol_id,
                          args, dxpl_id, req);
}

static herr_t iowarp_dataset_specific(void *obj,
                                      H5VL_dataset_specific_args_t *args,
                                      hid_t dxpl_id, void **req) {
  auto *dset = static_cast<iowarp_dataset_t *>(obj);
  return H5VLdataset_specific(dset->obj.under_object, dset->obj.under_vol_id,
                               args, dxpl_id, req);
}

static herr_t iowarp_dataset_close(void *obj, hid_t dxpl_id, void **req) {
  auto *dset = static_cast<iowarp_dataset_t *>(obj);

  /* Flush all pending async writes */
  for (auto &future : dset->pending_puts) {
    future.Wait();
  }
  dset->pending_puts.clear();
  dset->pending_buffers.clear();

  herr_t ret = H5VLdataset_close(dset->obj.under_object,
                                  dset->obj.under_vol_id, dxpl_id, req);
  delete dset;
  return ret;
}

/* ========================================================================
 * Passthrough: group, attribute, datatype, link, object, introspect
 * ======================================================================== */

/* Group */
static void *iowarp_group_create(void *obj,
                                 const H5VL_loc_params_t *loc_params,
                                 const char *name, hid_t lcpl_id,
                                 hid_t gcpl_id, hid_t gapl_id,
                                 hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  void *under = H5VLgroup_create(file->obj.under_object, loc_params,
                                  file->obj.under_vol_id, name, lcpl_id,
                                  gcpl_id, gapl_id, dxpl_id, req);
  if (!under) return nullptr;
  auto *o = new iowarp_obj_t;
  o->under_object = under;
  o->under_vol_id = file->obj.under_vol_id;
  return o;
}

static void *iowarp_group_open(void *obj,
                               const H5VL_loc_params_t *loc_params,
                               const char *name, hid_t gapl_id,
                               hid_t dxpl_id, void **req) {
  auto *file = static_cast<iowarp_file_t *>(obj);
  void *under = H5VLgroup_open(file->obj.under_object, loc_params,
                                file->obj.under_vol_id, name, gapl_id,
                                dxpl_id, req);
  if (!under) return nullptr;
  auto *o = new iowarp_obj_t;
  o->under_object = under;
  o->under_vol_id = file->obj.under_vol_id;
  return o;
}

static herr_t iowarp_group_get(void *obj, H5VL_group_get_args_t *args,
                               hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLgroup_get(o->under_object, o->under_vol_id, args, dxpl_id, req);
}

static herr_t iowarp_group_specific(void *obj,
                                    H5VL_group_specific_args_t *args,
                                    hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLgroup_specific(o->under_object, o->under_vol_id, args,
                             dxpl_id, req);
}

static herr_t iowarp_group_close(void *obj, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  herr_t ret = H5VLgroup_close(o->under_object, o->under_vol_id,
                                dxpl_id, req);
  delete o;
  return ret;
}

/* Attribute — full passthrough via native VOL */
static void *iowarp_attr_create(void *obj,
                                const H5VL_loc_params_t *loc_params,
                                const char *name, hid_t type_id,
                                hid_t space_id, hid_t acpl_id,
                                hid_t aapl_id, hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLattr_create(o->under_object, loc_params, o->under_vol_id, name,
                          type_id, space_id, acpl_id, aapl_id, dxpl_id, req);
}

static void *iowarp_attr_open(void *obj,
                              const H5VL_loc_params_t *loc_params,
                              const char *name, hid_t aapl_id,
                              hid_t dxpl_id, void **req) {
  auto *o = static_cast<iowarp_obj_t *>(obj);
  return H5VLattr_open(o->under_object, loc_params, o->under_vol_id, name,
                        aapl_id, dxpl_id, req);
}

static herr_t iowarp_attr_read(void *attr, hid_t dtype_id, void *buf,
                               hid_t dxpl_id, void **req) {
  return H5VLattr_read(attr, H5VL_NATIVE, dtype_id, buf, dxpl_id, req);
}

static herr_t iowarp_attr_write(void *attr, hid_t dtype_id, const void *buf,
                                hid_t dxpl_id, void **req) {
  return H5VLattr_write(attr, H5VL_NATIVE, dtype_id, buf, dxpl_id, req);
}

static herr_t iowarp_attr_get(void *obj, H5VL_attr_get_args_t *args,
                              hid_t dxpl_id, void **req) {
  return H5VLattr_get(obj, H5VL_NATIVE, args, dxpl_id, req);
}

static herr_t iowarp_attr_specific(void *obj, const H5VL_loc_params_t *lp,
                                   H5VL_attr_specific_args_t *args,
                                   hid_t dxpl_id, void **req) {
  return H5VLattr_specific(obj, lp, H5VL_NATIVE, args, dxpl_id, req);
}

static herr_t iowarp_attr_close(void *attr, hid_t dxpl_id, void **req) {
  return H5VLattr_close(attr, H5VL_NATIVE, dxpl_id, req);
}

/* Introspect */
static herr_t iowarp_introspect_get_conn_cls(void *obj,
                                             H5VL_get_conn_lvl_t lvl,
                                             const H5VL_class_t **conn_cls) {
  (void)obj; (void)lvl;
  *conn_cls = &H5VL_iowarp_cls;
  return 0;
}

static herr_t iowarp_introspect_get_cap_flags(const void *info,
                                              uint64_t *cap_flags) {
  (void)info;
  *cap_flags = H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |
               H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC;
  return 0;
}

static herr_t iowarp_introspect_opt_query(void *obj, H5VL_subclass_t cls,
                                          int opt_type, uint64_t *flags) {
  (void)obj; (void)cls; (void)opt_type;
  *flags = 0;
  return 0;
}

/* ========================================================================
 * VOL connector class definition
 * ======================================================================== */

const H5VL_class_t H5VL_iowarp_cls = {
    /* version      */ H5VL_VERSION,
    /* value        */ IOWARP_VOL_CONNECTOR_VALUE,
    /* name         */ IOWARP_VOL_CONNECTOR_NAME,
    /* conn_version */ IOWARP_VOL_CONNECTOR_VERSION,
    /* cap_flags    */ H5VL_CAP_FLAG_FILE_BASIC | H5VL_CAP_FLAG_DATASET_BASIC |
                       H5VL_CAP_FLAG_GROUP_BASIC | H5VL_CAP_FLAG_ATTR_BASIC,
    /* initialize   */ nullptr,
    /* terminate    */ nullptr,

    /* info_cls */ {
        /* size    */ sizeof(iowarp_vol_info_t),
        /* copy    */ iowarp_info_copy,
        /* cmp     */ nullptr,
        /* free    */ iowarp_info_free,
        /* to_str  */ nullptr,
        /* from_str*/ nullptr,
    },

    /* wrap_cls */ {
        /* get_object  */ iowarp_wrap_get_object,
        /* get_wrap_ctx*/ iowarp_get_wrap_ctx,
        /* wrap_object */ iowarp_wrap_object,
        /* unwrap_object*/ iowarp_unwrap_object,
        /* free_wrap_ctx*/ iowarp_free_wrap_ctx,
    },

    /* attr_cls */ {
        /* create   */ iowarp_attr_create,
        /* open     */ iowarp_attr_open,
        /* read     */ iowarp_attr_read,
        /* write    */ iowarp_attr_write,
        /* get      */ iowarp_attr_get,
        /* specific */ iowarp_attr_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_attr_close,
    },

    /* dataset_cls */ {
        /* create   */ iowarp_dataset_create,
        /* open     */ iowarp_dataset_open,
        /* read     */ iowarp_dataset_read,
        /* write    */ iowarp_dataset_write,
        /* get      */ iowarp_dataset_get,
        /* specific */ iowarp_dataset_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_dataset_close,
    },

    /* datatype_cls */ {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    },

    /* file_cls */ {
        /* create   */ iowarp_file_create,
        /* open     */ iowarp_file_open,
        /* get      */ iowarp_file_get,
        /* specific */ iowarp_file_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_file_close,
    },

    /* group_cls */ {
        /* create   */ iowarp_group_create,
        /* open     */ iowarp_group_open,
        /* get      */ iowarp_group_get,
        /* specific */ iowarp_group_specific,
        /* optional */ nullptr,
        /* close    */ iowarp_group_close,
    },

    /* link_cls */ {
        nullptr, nullptr, nullptr, nullptr, nullptr,
    },

    /* object_cls */ {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    },

    /* introspect_cls */ {
        /* get_conn_cls   */ iowarp_introspect_get_conn_cls,
        /* get_cap_flags  */ iowarp_introspect_get_cap_flags,
        /* opt_query      */ iowarp_introspect_opt_query,
    },

    /* request_cls */ {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    },

    /* blob_cls */ {
        nullptr, nullptr, nullptr, nullptr,
    },

    /* token_cls */ {
        nullptr, nullptr,
    },

    /* optional */ nullptr,
};

hid_t H5VL_iowarp_register(void) {
  return H5VLregister_connector(&H5VL_iowarp_cls, H5P_DEFAULT);
}
