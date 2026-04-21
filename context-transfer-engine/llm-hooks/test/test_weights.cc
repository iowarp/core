#include <catch2/catch_test_macros.hpp>
#include "wrp_llm/weights/weight_manager.h"
#include "wrp_llm/weights/ggml_iowarp_backend.h"

using wrp_llm::weights::WeightManager;

TEST_CASE("WeightManager initializes GpuVmm without CTE", "[weights][integration]") {
    WeightManager::Config cfg;
    cfg.va_size_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB VA (small for test)
    cfg.page_size     = 2 * 1024 * 1024;             // 2 MB pages
    cfg.use_cte       = false;                        // host RAM backing only
    cfg.device        = 0;

    WeightManager mgr(cfg);
    REQUIRE(mgr.Init());
    REQUIRE(mgr.BaseAddress() != nullptr);
    REQUIRE(mgr.PageSize() == cfg.page_size);

    mgr.Shutdown();
}

TEST_CASE("WeightManager RegisterLayer + PrepareLayer + ReleaseLayer", "[weights][integration]") {
    WeightManager::Config cfg;
    cfg.va_size_bytes  = 4ULL * 1024 * 1024 * 1024;
    cfg.page_size      = 2 * 1024 * 1024;
    cfg.prefetch_window = 1;
    cfg.use_cte        = false;
    cfg.device         = 0;

    WeightManager mgr(cfg);
    REQUIRE(mgr.Init());

    // Register two dummy layers occupying 1 page each.
    mgr.RegisterLayer(0, /*page_start=*/0,  /*page_count=*/1);
    mgr.RegisterLayer(1, /*page_start=*/1,  /*page_count=*/1);

    // Should not throw / crash.
    mgr.PrepareLayer(0);
    mgr.ReleaseLayer(0);
    mgr.PrepareLayer(1);
    mgr.ReleaseLayer(1);

    mgr.Shutdown();
}

TEST_CASE("ggml_backend_iowarp_buffer_type returns non-null", "[weights][unit]") {
    WeightManager::Config cfg;
    cfg.va_size_bytes = 4ULL * 1024 * 1024 * 1024;
    cfg.page_size     = 2 * 1024 * 1024;
    cfg.use_cte       = false;
    cfg.device        = 0;

    WeightManager mgr(cfg);
    REQUIRE(mgr.Init());

    ggml_backend_buffer_type_t buft = ggml_backend_iowarp_buffer_type(&mgr);
    REQUIRE(buft != nullptr);
    REQUIRE(std::string(buft->iface.get_name(buft)) == GGML_BACKEND_IOWARP_NAME);

    mgr.Shutdown();
}
