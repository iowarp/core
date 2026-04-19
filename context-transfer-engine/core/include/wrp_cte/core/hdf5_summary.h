#ifndef WRPCTE_HDF5_SUMMARY_H_
#define WRPCTE_HDF5_SUMMARY_H_

/**
 * HDF5 L2 summary extractor for the DepthController.
 *
 * Opens a file with the HDF5 library, walks its group hierarchy, and
 * produces a text summary enumerating every dataset's name, shape, and
 * dtype plus every attribute. The output is intended to be appended to
 * the DepthController's L2 payload and indexed by the search backend,
 * so it's kept flat / word-tokenized rather than structured.
 *
 * This extractor does NOT depend on CAE. It links only against the HDF5
 * C library and is compiled into wrp_cte_core_runtime when HDF5 is
 * available (guarded by WRP_CORE_ENABLE_HDF5).
 *
 * Usage:
 *   depth_controller.RegisterL2Extractor("h5",   &Hdf5Summary::Extract);
 *   depth_controller.RegisterL2Extractor("hdf5", &Hdf5Summary::Extract);
 */

#ifdef WRP_CORE_ENABLE_HDF5

#include <hdf5.h>

#include <sstream>
#include <string>
#include <vector>

namespace wrp_cte::core {

class Hdf5Summary {
 public:
  /** Entry point — open `path`, emit a summary string. Empty on failure. */
  static std::string Extract(const std::string &path) {
    // Suppress HDF5's noisy error stack printing; we handle errors ourselves.
    H5E_auto2_t saved_func = nullptr;
    void *saved_client = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &saved_func, &saved_client);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    H5Eset_auto2(H5E_DEFAULT, saved_func, saved_client);

    if (file < 0) return {};

    std::ostringstream oss;
    oss << "content_kind=hdf5_scientific";

    // Recursively visit all objects using the v1.10 H5Ovisit2 API.
    Ctx ctx{&oss};
    H5Ovisit2(file, H5_INDEX_NAME, H5_ITER_NATIVE, &Hdf5Summary::OnObject,
              &ctx, H5O_INFO_BASIC);

    H5Fclose(file);
    return oss.str();
  }

 private:
  struct Ctx {
    std::ostringstream *oss;
  };

  // H5O visit callback — called once per object (group, dataset, datatype)
  static herr_t OnObject(hid_t obj_id, const char *name,
                         const H5O_info_t *info, void *op_data) {
    auto *ctx = static_cast<Ctx *>(op_data);
    auto &oss = *ctx->oss;
    std::string clean_name = (name && *name) ? name : "/";

    if (info->type == H5O_TYPE_GROUP) {
      oss << " group=" << clean_name;
      // Open the group explicitly so H5Aiterate iterates on this group's
      // attributes (otherwise we'd iterate on the file handle's root group).
      hid_t g = H5Gopen2(obj_id, name, H5P_DEFAULT);
      if (g >= 0) {
        EmitAttributes(g, clean_name, oss);
        H5Gclose(g);
      }
    } else if (info->type == H5O_TYPE_DATASET) {
      hid_t ds = H5Dopen2(obj_id, name, H5P_DEFAULT);
      if (ds < 0) return 0;
      EmitDatasetSummary(ds, clean_name, oss);
      EmitAttributes(ds, clean_name, oss);
      H5Dclose(ds);
    }
    return 0;
  }

  static void EmitDatasetSummary(hid_t ds, const std::string &name,
                                 std::ostringstream &oss) {
    oss << " dataset=" << name;

    hid_t space = H5Dget_space(ds);
    if (space >= 0) {
      int ndims = H5Sget_simple_extent_ndims(space);
      if (ndims > 0 && ndims < 16) {
        std::vector<hsize_t> dims(ndims);
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        oss << " shape=";
        for (int i = 0; i < ndims; ++i) {
          if (i) oss << "x";
          oss << dims[i];
        }
      }
      H5Sclose(space);
    }

    hid_t dtype = H5Dget_type(ds);
    if (dtype >= 0) {
      oss << " dtype=" << DTypeName(dtype);
      H5Tclose(dtype);
    }
  }

  static void EmitAttributes(hid_t obj, const std::string &parent_name,
                             std::ostringstream &oss) {
    auto callback = +[](hid_t loc, const char *attr_name,
                        const H5A_info_t * /*ainfo*/,
                        void *op_data) -> herr_t {
      auto &oss = *static_cast<std::ostringstream *>(op_data);
      oss << " attr=" << (attr_name ? attr_name : "");

      hid_t a = H5Aopen(loc, attr_name, H5P_DEFAULT);
      if (a < 0) return 0;
      hid_t atype = H5Aget_type(a);
      if (atype >= 0) {
        if (H5Tget_class(atype) == H5T_STRING) {
          // Read the string value (short strings only)
          size_t sz = H5Tget_size(atype);
          if (sz > 0 && sz < 512) {
            std::vector<char> buf(sz + 1, 0);
            if (H5Aread(a, atype, buf.data()) >= 0) {
              oss << "=" << buf.data();
            }
          }
        } else if (H5Tget_class(atype) == H5T_INTEGER) {
          long long v = 0;
          if (H5Aread(a, H5T_NATIVE_LLONG, &v) >= 0) oss << "=" << v;
        } else if (H5Tget_class(atype) == H5T_FLOAT) {
          double v = 0.0;
          if (H5Aread(a, H5T_NATIVE_DOUBLE, &v) >= 0) oss << "=" << v;
        }
        H5Tclose(atype);
      }
      H5Aclose(a);
      return 0;
    };
    (void)parent_name;
    hsize_t idx = 0;
    H5Aiterate2(obj, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, callback, &oss);
  }

  static const char *DTypeName(hid_t dtype) {
    H5T_class_t cls = H5Tget_class(dtype);
    switch (cls) {
      case H5T_INTEGER: return "int";
      case H5T_FLOAT:   return "float";
      case H5T_STRING:  return "string";
      case H5T_BITFIELD:return "bitfield";
      case H5T_OPAQUE:  return "opaque";
      case H5T_COMPOUND:return "compound";
      case H5T_REFERENCE:return "reference";
      case H5T_ENUM:    return "enum";
      case H5T_VLEN:    return "vlen";
      case H5T_ARRAY:   return "array";
      default:          return "unknown";
    }
  }
};

}  // namespace wrp_cte::core

#endif  // WRP_CORE_ENABLE_HDF5
#endif  // WRPCTE_HDF5_SUMMARY_H_
