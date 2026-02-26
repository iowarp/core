"""Wrapper around chimaera_runtime_ext for the visualizer."""

import threading

try:
    import msgpack
except ImportError:
    msgpack = None

_lock = threading.Lock()
_chi = None
_init_done = False


def _ensure_init():
    """Lazy-initialize the Chimaera client connection."""
    global _chi, _init_done
    if _init_done:
        return
    with _lock:
        if _init_done:
            return
        import chimaera_runtime_ext as chi
        _chi = chi
        # 0 = kClient — pass a plain int to avoid any nanobind enum objects
        ok = chi.chimaera_init(0)
        if not ok:
            raise RuntimeError("chimaera_init(kClient) failed -- is the runtime running?")
        _init_done = True


def _decode_results(results):
    """Decode a {container_id: bytes} dict into {container_id: decoded_data}."""
    decoded = {}
    for cid, blob in results.items():
        if msgpack is not None and isinstance(blob, (bytes, bytearray)):
            try:
                decoded[str(cid)] = msgpack.unpackb(blob, raw=False)
            except Exception:
                decoded[str(cid)] = blob
        else:
            decoded[str(cid)] = blob
    return decoded


def is_connected():
    """Return True if the client has been initialized."""
    return _init_done


def get_worker_stats():
    """Query worker_stats from the admin pool (local node)."""
    _ensure_init()
    results = _chi.async_monitor("local", "worker_stats").wait()
    return _decode_results(results)


def get_pool_worker_stats(pool_id_str, routing="local"):
    """Query worker_stats for a specific pool via pool_stats:// URI."""
    _ensure_init()
    uri = f"pool_stats://{pool_id_str}:{routing}:worker_stats"
    results = _chi.async_monitor("local", uri).wait()
    return _decode_results(results)


def get_status():
    """Query general status from the admin pool."""
    _ensure_init()
    results = _chi.async_monitor("local", "status").wait()
    return _decode_results(results)


def get_system_stats(pool_query="local", min_event_id=0):
    """Query system_stats from the admin pool."""
    _ensure_init()
    results = _chi.async_monitor(pool_query, f"system_stats:{min_event_id}").wait()
    return _decode_results(results)


def get_system_stats_all(min_event_id=0):
    """Broadcast system_stats to all nodes (each self-identifies via hostname/ip/node_id)."""
    _ensure_init()
    results = _chi.async_monitor("broadcast", f"system_stats:{min_event_id}").wait()
    return _decode_results(results)


def get_bdev_stats(pool_query="local"):
    """Query bdev_stats from the admin pool."""
    _ensure_init()
    results = _chi.async_monitor(pool_query, "bdev_stats").wait()
    return _decode_results(results)


def get_worker_stats_for_node(node_id):
    """Query worker_stats for a specific node."""
    _ensure_init()
    results = _chi.async_monitor(f"physical:{node_id}", "worker_stats").wait()
    return _decode_results(results)


def get_system_stats_for_node(node_id, min_event_id=0):
    """Query system_stats for a specific node."""
    _ensure_init()
    results = _chi.async_monitor(f"physical:{node_id}", f"system_stats:{min_event_id}").wait()
    return _decode_results(results)


def get_bdev_stats_for_node(node_id):
    """Query bdev_stats for a specific node."""
    _ensure_init()
    results = _chi.async_monitor(f"physical:{node_id}", "bdev_stats").wait()
    return _decode_results(results)


def shutdown_node(ip_address, grace_period_ms=5000):
    """Shutdown a remote node via SSH running 'chimaera runtime stop'.

    Exit codes 0 and 134 (SIGABRT from std::abort in InitiateShutdown)
    are both treated as success.
    """
    import subprocess
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        ip_address,
        f"chimaera runtime stop --grace-period {grace_period_ms}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    success = result.returncode in (0, 134)
    return {
        "success": success,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def restart_node(ip_address):
    """Restart a remote node via SSH.

    Uses nohup so the SSH session returns immediately while the restart
    proceeds in the background.
    """
    import subprocess
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        ip_address,
        "nohup chimaera runtime restart </dev/null >/dev/null 2>&1 &",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    success = result.returncode == 0
    return {
        "success": success,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def finalize():
    """Clean shutdown of the Chimaera client."""
    global _init_done
    if _init_done and _chi is not None:
        _chi.chimaera_finalize()
        _init_done = False
