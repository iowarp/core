#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>
#include <chimaera/chimaera.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(chimaera_runtime_ext, m) {
  m.doc() = "Python bindings for Chimaera runtime monitoring";

  // Bind ChimaeraMode enum
  nb::enum_<chi::ChimaeraMode>(m, "ChimaeraMode")
      .value("kClient", chi::ChimaeraMode::kClient)
      .value("kServer", chi::ChimaeraMode::kServer)
      .value("kRuntime", chi::ChimaeraMode::kRuntime);

  // Bind PoolQuery for routing queries
  nb::class_<chi::PoolQuery>(m, "PoolQuery")
      .def(nb::init<>())
      .def_static("Broadcast", &chi::PoolQuery::Broadcast)
      .def_static("Dynamic", &chi::PoolQuery::Dynamic)
      .def_static("Local", &chi::PoolQuery::Local)
      .def_static("DirectId", &chi::PoolQuery::DirectId, "container_id"_a)
      .def_static("DirectHash", &chi::PoolQuery::DirectHash, "hash"_a)
      .def_static("Range", &chi::PoolQuery::Range, "offset"_a, "count"_a)
      .def_static("Physical", &chi::PoolQuery::Physical, "node_id"_a)
      .def_static("FromString", &chi::PoolQuery::FromString, "str"_a)
      .def("ToString", &chi::PoolQuery::ToString);

  // Bind MonitorTask (read-only access to results)
  nb::class_<chimaera::admin::MonitorTask>(m, "MonitorTask")
      .def_ro("query_", &chimaera::admin::MonitorTask::query_)
      .def_ro("results_", &chimaera::admin::MonitorTask::results_);

  // Bind Future<MonitorTask>
  using MonitorFuture = chi::Future<chimaera::admin::MonitorTask>;
  nb::class_<MonitorFuture>(m, "MonitorFuture")
      .def("wait", &MonitorFuture::Wait, "Block until the task completes")
      .def(
          "get",
          [](MonitorFuture &self) -> chimaera::admin::MonitorTask & {
            return *self;
          },
          nb::rv_policy::reference,
          "Get the underlying MonitorTask (call after wait)")
      .def("del_task", &MonitorFuture::DelTask,
           "Explicitly free the task memory");

  // Bind admin::Client with monitor method
  nb::class_<chimaera::admin::Client>(m, "AdminClient")
      .def(nb::init<>())
      .def(nb::init<const chi::PoolId &>(), "pool_id"_a)
      .def(
          "async_monitor",
          [](chimaera::admin::Client &self, const chi::PoolQuery &pool_query,
             const std::string &query) {
            return self.AsyncMonitor(pool_query, query);
          },
          "pool_query"_a, "query"_a,
          "Submit a unified monitor query. Returns MonitorFuture.")
      .def(
          "monitor",
          [](chimaera::admin::Client &self, const chi::PoolQuery &pool_query,
             const std::string &query)
              -> std::unordered_map<chi::ContainerId, std::string> {
            auto future = self.AsyncMonitor(pool_query, query);
            future.Wait();
            // Copy results out before future goes out of scope
            std::unordered_map<chi::ContainerId, std::string> results =
                future->results_;
            future.DelTask();
            return results;
          },
          "pool_query"_a, "query"_a,
          "Synchronous monitor query. Returns dict[ContainerId, bytes].");

  // Bind PoolId (UniqueId)
  nb::class_<chi::PoolId>(m, "PoolId")
      .def(nb::init<>())
      .def(nb::init<chi::u32, chi::u32>(), "major"_a, "minor"_a)
      .def_static("GetNull", &chi::PoolId::GetNull)
      .def_static("FromString", &chi::PoolId::FromString, "str"_a)
      .def("ToString", &chi::PoolId::ToString)
      .def("IsNull", &chi::PoolId::IsNull)
      .def_rw("major_", &chi::PoolId::major_)
      .def_rw("minor_", &chi::PoolId::minor_);

  // Chimaera init/finalize
  m.def("chimaera_init", &chi::CHIMAERA_INIT, "mode"_a,
        "default_with_runtime"_a = false, "is_restart"_a = false,
        "Initialize Chimaera with specified mode");

  m.def("chimaera_finalize", &chi::CHIMAERA_FINALIZE,
        "Finalize Chimaera and release all resources");
}
