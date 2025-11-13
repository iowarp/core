#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(iowarp_cee_api, m) {
  m.doc() = "IOWarp Context Exploration Engine API - Python Bindings";

  // Bind AssimilationCtx struct
  nb::class_<wrp_cae::core::AssimilationCtx>(m, "AssimilationCtx",
      "Context for data assimilation operations")
    .def(nb::init<>(),
         "Default constructor")
    .def(nb::init<const std::string&, const std::string&, const std::string&,
                  const std::string&, size_t, size_t, const std::string&, const std::string&>(),
         "src"_a, "dst"_a, "format"_a,
         "depends_on"_a = "", "range_off"_a = 0, "range_size"_a = 0,
         "src_token"_a = "", "dst_token"_a = "",
         "Full constructor")
    .def_rw("src", &wrp_cae::core::AssimilationCtx::src,
            "Source URL (e.g., file::/path/to/file)")
    .def_rw("dst", &wrp_cae::core::AssimilationCtx::dst,
            "Destination URL (e.g., iowarp::tag_name)")
    .def_rw("format", &wrp_cae::core::AssimilationCtx::format,
            "Data format (e.g., binary, hdf5)")
    .def_rw("depends_on", &wrp_cae::core::AssimilationCtx::depends_on,
            "Dependency identifier (empty if none)")
    .def_rw("range_off", &wrp_cae::core::AssimilationCtx::range_off,
            "Byte offset in source file")
    .def_rw("range_size", &wrp_cae::core::AssimilationCtx::range_size,
            "Number of bytes to read")
    .def_rw("src_token", &wrp_cae::core::AssimilationCtx::src_token,
            "Authentication token for source")
    .def_rw("dst_token", &wrp_cae::core::AssimilationCtx::dst_token,
            "Authentication token for destination")
    .def("__repr__", [](const wrp_cae::core::AssimilationCtx& ctx) {
      return "<AssimilationCtx src='" + ctx.src + "' dst='" + ctx.dst +
             "' format='" + ctx.format + "'>";
    });

  // Bind ContextInterface class
  // C++ uses PascalCase (Google style), Python exposes snake_case
  nb::class_<iowarp::ContextInterface>(m, "ContextInterface",
      "High-level API for context exploration and management")
    .def(nb::init<>(),
         "Default constructor - initializes the interface")
    .def("context_bundle", &iowarp::ContextInterface::ContextBundle,
         "bundle"_a,
         "Bundle a group of related objects together and assimilate them\n\n"
         "Parameters:\n"
         "  bundle: List of AssimilationCtx objects to assimilate\n\n"
         "Returns:\n"
         "  0 on success, non-zero error code on failure")
    .def("context_query", &iowarp::ContextInterface::ContextQuery,
         "tag_re"_a, "blob_re"_a,
         "Retrieve the identities of objects matching tag and blob patterns\n\n"
         "Parameters:\n"
         "  tag_re: Tag regex pattern to match\n"
         "  blob_re: Blob regex pattern to match\n\n"
         "Returns:\n"
         "  List of matching blob names")
    .def("context_retrieve", &iowarp::ContextInterface::ContextRetrieve,
         "tag_re"_a, "blob_re"_a,
         "Retrieve the identities and data of objects (NOT YET IMPLEMENTED)\n\n"
         "Parameters:\n"
         "  tag_re: Tag regex pattern to match\n"
         "  blob_re: Blob regex pattern to match\n\n"
         "Returns:\n"
         "  List of object identities (currently returns empty list)")
    .def("context_splice", &iowarp::ContextInterface::ContextSplice,
         "new_ctx"_a, "tag_re"_a, "blob_re"_a,
         "Split/splice objects into a new context (NOT YET IMPLEMENTED)\n\n"
         "Parameters:\n"
         "  new_ctx: Name of the new context to create\n"
         "  tag_re: Tag regex pattern to match for source objects\n"
         "  blob_re: Blob regex pattern to match for source objects\n\n"
         "Returns:\n"
         "  0 on success, non-zero error code on failure")
    .def("context_destroy", &iowarp::ContextInterface::ContextDestroy,
         "context_names"_a,
         "Destroy contexts by name\n\n"
         "Parameters:\n"
         "  context_names: List of context names to destroy\n\n"
         "Returns:\n"
         "  0 on success, non-zero error code on failure");
}
