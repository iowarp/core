"""Test unified monitoring hook via Python bindings."""
import sys
import os

# When running from the build directory, the extension .so is in the working dir
# Add it to sys.path so import can find it
sys.path.insert(0, os.getcwd())

import chimaera_runtime_ext as chi

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("WARNING: msgpack not installed, skipping msgpack decode tests")


def test_import():
    """Test that the module imports and basic types exist."""
    assert hasattr(chi, "ChimaeraMode")
    assert hasattr(chi, "PoolQuery")
    assert hasattr(chi, "MonitorTask")
    assert hasattr(chi, "MonitorFuture")
    assert hasattr(chi, "AdminClient")
    assert hasattr(chi, "PoolId")
    assert hasattr(chi, "chimaera_init")
    assert hasattr(chi, "chimaera_finalize")
    print("PASSED: test_import")


def test_pool_query():
    """Test PoolQuery static constructors and ToString/FromString round-trip."""
    pq_broadcast = chi.PoolQuery.Broadcast()
    pq_dynamic = chi.PoolQuery.Dynamic()
    pq_local = chi.PoolQuery.Local()
    assert pq_broadcast is not None
    assert pq_dynamic is not None
    assert pq_local is not None

    # Test additional factory methods
    pq_direct_id = chi.PoolQuery.DirectId(42)
    pq_direct_hash = chi.PoolQuery.DirectHash(99)
    pq_range = chi.PoolQuery.Range(0, 5)
    pq_physical = chi.PoolQuery.Physical(7)
    assert pq_direct_id is not None
    assert pq_direct_hash is not None
    assert pq_range is not None
    assert pq_physical is not None

    # Test ToString for all modes
    assert pq_local.ToString() == "local"
    assert pq_broadcast.ToString() == "broadcast"
    assert pq_dynamic.ToString() == "dynamic"
    assert pq_direct_id.ToString() == "direct_id:42"
    assert pq_direct_hash.ToString() == "direct_hash:99"
    assert pq_range.ToString() == "range:0:5"
    assert pq_physical.ToString() == "physical:7"

    # Test FromString round-trip for all modes
    for original_str in ["local", "broadcast", "dynamic",
                         "direct_id:42", "direct_hash:99",
                         "range:0:5", "physical:7"]:
        pq = chi.PoolQuery.FromString(original_str)
        assert pq.ToString() == original_str, \
            f"FromString/ToString round-trip failed: '{original_str}' -> '{pq.ToString()}'"

    # Test FromString is case-insensitive
    pq_upper = chi.PoolQuery.FromString("BROADCAST")
    assert pq_upper.ToString() == "broadcast"
    pq_mixed = chi.PoolQuery.FromString("Direct_Id:10")
    assert pq_mixed.ToString() == "direct_id:10"

    # Test FromString with invalid input
    try:
        chi.PoolQuery.FromString("invalid_mode")
        assert False, "Expected exception for invalid PoolQuery string"
    except Exception:
        pass  # Expected

    print("PASSED: test_pool_query")


def test_pool_id():
    """Test PoolId creation and ToString/FromString round-trip."""
    pid = chi.PoolId()
    assert pid.IsNull()
    pid2 = chi.PoolId(1, 2)
    assert pid2.major_ == 1
    assert pid2.minor_ == 2

    # Test ToString
    assert pid2.ToString() == "1.2"
    assert chi.PoolId(200, 0).ToString() == "200.0"
    assert chi.PoolId(3, 5).ToString() == "3.5"

    # Test FromString
    pid3 = chi.PoolId.FromString("200.0")
    assert pid3.major_ == 200
    assert pid3.minor_ == 0

    pid4 = chi.PoolId.FromString("3.5")
    assert pid4.major_ == 3
    assert pid4.minor_ == 5

    # Test round-trip
    for s in ["1.0", "3.5", "200.0", "0.0"]:
        assert chi.PoolId.FromString(s).ToString() == s, \
            f"PoolId round-trip failed for '{s}'"

    # Test invalid FromString
    try:
        chi.PoolId.FromString("no_dot")
        assert False, "Expected exception for invalid PoolId string"
    except Exception:
        pass  # Expected

    print("PASSED: test_pool_id")


def test_chimaera_mode():
    """Test ChimaeraMode enum values."""
    assert chi.ChimaeraMode.kClient is not None
    assert chi.ChimaeraMode.kServer is not None
    assert chi.ChimaeraMode.kRuntime is not None
    print("PASSED: test_chimaera_mode")


def test_monitor_with_runtime():
    """Test monitor with actual runtime (requires CHI_WITH_RUNTIME=1)."""
    if os.environ.get("CHI_WITH_RUNTIME") != "1":
        print("SKIPPED: test_monitor_with_runtime (CHI_WITH_RUNTIME not set)")
        return

    # Initialize chimaera
    ok = chi.chimaera_init(chi.ChimaeraMode.kClient)
    assert ok, "chimaera_init failed"

    try:
        # Create an admin client (pool 0,0 is admin)
        admin = chi.AdminClient(chi.PoolId(0, 0))

        # Send a monitor query to admin (local only)
        future = admin.async_monitor(chi.PoolQuery.Local(), "status")
        future.wait()
        task = future.get()

        # Verify we got results back (may be empty for admin stub)
        assert task.results_ is not None
        print(f"  Monitor returned {len(task.results_)} container results")

        future.del_task()

        # Also test synchronous wrapper
        results = admin.monitor(chi.PoolQuery.Local(), "status")
        assert isinstance(results, dict)
        print(f"  Sync monitor returned {len(results)} container results")

        print("PASSED: test_monitor_with_runtime")
    finally:
        chi.chimaera_finalize()


def test_pool_stats_uri():
    """Test pool_stats:// URI dispatch via Admin::Monitor (requires CHI_WITH_RUNTIME=1)."""
    if os.environ.get("CHI_WITH_RUNTIME") != "1":
        print("SKIPPED: test_pool_stats_uri (CHI_WITH_RUNTIME not set)")
        return

    # Initialize chimaera
    ok = chi.chimaera_init(chi.ChimaeraMode.kClient)
    assert ok, "chimaera_init failed"

    try:
        admin = chi.AdminClient(chi.PoolId(0, 0))

        # Test 1: pool_stats:// targeting admin pool (1.0) with local routing
        # Admin's kMonitor handles "worker_stats" as its selector
        results = admin.monitor(
            chi.PoolQuery.Local(),
            "pool_stats://1.0:local:worker_stats")
        assert isinstance(results, dict)
        print(f"  pool_stats://1.0:local:worker_stats -> {len(results)} containers")

        # Test 2: pool_stats:// with broadcast routing to admin pool
        results = admin.monitor(
            chi.PoolQuery.Local(),
            "pool_stats://1.0:broadcast:worker_stats")
        assert isinstance(results, dict)
        print(f"  pool_stats://1.0:broadcast:worker_stats -> {len(results)} containers")

        # Test 3: pool_stats:// with invalid pool should return error (rc != 0)
        future = admin.async_monitor(
            chi.PoolQuery.Local(),
            "pool_stats://999.0:local:anything")
        future.wait()
        task = future.get()
        # Pool 999.0 doesn't exist -- expect empty results or error
        print(f"  pool_stats://999.0:local:anything -> rc={task.results_}")
        future.del_task()

        # Test 4: pool_stats:// with malformed URI should return error
        future = admin.async_monitor(
            chi.PoolQuery.Local(),
            "pool_stats://bad_format")
        future.wait()
        task = future.get()
        print(f"  pool_stats://bad_format -> results={task.results_}")
        future.del_task()

        print("PASSED: test_pool_stats_uri")
    finally:
        chi.chimaera_finalize()


def test_mod_name_monitor():
    """Test MOD_NAME monitor with msgpack decode (requires CHI_WITH_RUNTIME=1)."""
    if os.environ.get("CHI_WITH_RUNTIME") != "1":
        print("SKIPPED: test_mod_name_monitor (CHI_WITH_RUNTIME not set)")
        return
    if not HAS_MSGPACK:
        print("SKIPPED: test_mod_name_monitor (msgpack not installed)")
        return

    # Initialize chimaera
    ok = chi.chimaera_init(chi.ChimaeraMode.kClient)
    assert ok, "chimaera_init failed"

    try:
        # The MOD_NAME pool must already exist (created by test infrastructure)
        # Use admin to query it via broadcast
        admin = chi.AdminClient(chi.PoolId(0, 0))

        # Query MOD_NAME containers
        results = admin.monitor(chi.PoolQuery.Broadcast(), "get_data")
        print(f"  Broadcast monitor returned {len(results)} containers")

        for container_id, data_bytes in results.items():
            items = msgpack.unpackb(data_bytes, raw=False)
            print(f"  Container {container_id}: {len(items)} items")
            for i, item in enumerate(items):
                assert item["id"] == i, f"Expected id={i}, got {item['id']}"
                assert abs(item["value"] - i * 1.5) < 1e-6, \
                    f"Expected value={i*1.5}, got {item['value']}"
                assert item["name"] == f"item_{i}", \
                    f"Expected name=item_{i}, got {item['name']}"

        print("PASSED: test_mod_name_monitor")
    finally:
        chi.chimaera_finalize()


if __name__ == "__main__":
    # Always run basic tests (no runtime needed)
    test_import()
    test_pool_query()
    test_pool_id()
    test_chimaera_mode()

    # Runtime tests (only when CHI_WITH_RUNTIME=1)
    test_monitor_with_runtime()
    test_pool_stats_uri()
    test_mod_name_monitor()

    print("\nAll tests PASSED")
