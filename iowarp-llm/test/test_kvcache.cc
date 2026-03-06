#include <catch2/catch_test_macros.hpp>
#include "wrp_llm/kvcache/kvcache_manager.h"

using wrp_llm::kvcache::KVCacheManager;

// ---------------------------------------------------------------------------
// Hash utility tests (no CTE required)
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager::HashTokens is deterministic", "[kvcache][unit]") {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    std::string h1 = KVCacheManager::HashTokens(tokens);
    std::string h2 = KVCacheManager::HashTokens(tokens);
    REQUIRE(h1 == h2);
    REQUIRE(h1.size() == 16);  // 16 hex chars = 64-bit hash
}

TEST_CASE("KVCacheManager::HashTokens differs for different sequences", "[kvcache][unit]") {
    std::vector<int32_t> a = {1, 2, 3};
    std::vector<int32_t> b = {1, 2, 4};
    REQUIRE(KVCacheManager::HashTokens(a) != KVCacheManager::HashTokens(b));
}

TEST_CASE("KVCacheManager::HashTokens prefix of a longer sequence differs", "[kvcache][unit]") {
    std::vector<int32_t> full   = {10, 20, 30, 40};
    std::vector<int32_t> prefix = {10, 20, 30};
    REQUIRE(KVCacheManager::HashTokens(full) != KVCacheManager::HashTokens(prefix));
}

// ---------------------------------------------------------------------------
// Integration tests (require CTE runtime — run inside the container)
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager StoreBlock + LookupPrefix round-trip", "[kvcache][integration]") {
    KVCacheManager::Config cfg;
    cfg.tag_name = "test_kvcache_roundtrip";
    KVCacheManager mgr(cfg);

    // Skip if CTE is not available in this environment.
    if (!mgr.Init()) {
        SKIP("CTE runtime not available — skipping integration test");
    }

    std::vector<int32_t> tokens = {100, 200, 300, 400};
    std::vector<uint8_t> kv_data(1024, 0xAB);  // 1 KB of dummy KV data

    REQUIRE(mgr.StoreBlock(tokens, kv_data.data(), kv_data.size(), /*on_gpu=*/false));

    std::vector<uint8_t> restored(1024, 0x00);
    size_t matched = 0;
    bool found = mgr.LookupPrefix(tokens, restored.data(), restored.size(),
                                   /*on_gpu=*/false, matched);

    REQUIRE(found);
    REQUIRE(matched == tokens.size());
    REQUIRE(restored == kv_data);

    mgr.Shutdown();
}

TEST_CASE("KVCacheManager LookupPrefix returns longest match", "[kvcache][integration]") {
    KVCacheManager::Config cfg;
    cfg.tag_name = "test_kvcache_prefix";
    KVCacheManager mgr(cfg);

    if (!mgr.Init()) {
        SKIP("CTE runtime not available — skipping integration test");
    }

    std::vector<int32_t> prefix3 = {1, 2, 3};
    std::vector<uint8_t> kv3(512, 0x11);
    REQUIRE(mgr.StoreBlock(prefix3, kv3.data(), kv3.size(), /*on_gpu=*/false));

    // Query with a longer sequence — should match the stored 3-token prefix.
    std::vector<int32_t> query = {1, 2, 3, 4, 5};
    std::vector<uint8_t> out(512);
    size_t matched = 0;
    bool found = mgr.LookupPrefix(query, out.data(), out.size(),
                                   /*on_gpu=*/false, matched);

    REQUIRE(found);
    REQUIRE(matched == 3);
    REQUIRE(out == kv3);

    mgr.Shutdown();
}
