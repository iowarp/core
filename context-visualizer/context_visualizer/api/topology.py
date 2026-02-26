"""GET /api/topology -- cluster topology overview + node management."""

import socket

from flask import Blueprint, jsonify

from .. import chimaera_client

bp = Blueprint("topology", __name__)


def _find_node_ip(raw, target_node_id):
    """Extract the IP address for a given node_id from system_stats broadcast results."""
    target_node_id = int(target_node_id)
    for cid, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            nid = entry.get("node_id")
            if nid is not None and int(nid) == target_node_id:
                ip = entry.get("ip_address", "")
                if ip:
                    return ip
    # Fallback: node_id 0 with no IP means we're on the local node
    if target_node_id == 0:
        return "127.0.0.1"
    return None


@bp.route("/topology")
def get_topology():
    try:
        raw = chimaera_client.get_system_stats_all()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    # Each container's response is an array of system_stats entries.
    # Extract the latest entry per node (identified by node_id).
    # Fallback: if node_id is missing (old runtime), use container_id as key.
    nodes = {}
    for cid, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Prefer node_id from response; fall back to container id
            nid = entry.get("node_id")
            if nid is None:
                nid = cid
            nid_str = str(nid)
            # Keep the latest entry per node (highest event_id)
            prev = nodes.get(nid_str)
            if prev is None or entry.get("event_id", 0) > prev.get("event_id", 0):
                nodes[nid_str] = entry

    # Fallback hostname when the runtime doesn't supply one
    local_hostname = socket.gethostname()

    result = []
    for nid_str, entry in nodes.items():
        # Derive a numeric node_id for the frontend URL
        raw_nid = entry.get("node_id")
        if raw_nid is not None:
            node_id = int(raw_nid)
        else:
            # Best-effort: try parsing the container id string, default to 0
            try:
                node_id = int(nid_str)
            except (ValueError, TypeError):
                node_id = 0

        result.append({
            "node_id": node_id,
            "hostname": entry.get("hostname") or local_hostname,
            "ip_address": entry.get("ip_address", ""),
            "cpu_usage_pct": entry.get("cpu_usage_pct", 0),
            "ram_usage_pct": entry.get("ram_usage_pct", 0),
            "gpu_count": entry.get("gpu_count", 0),
            "gpu_usage_pct": entry.get("gpu_usage_pct", 0),
            "hbm_usage_pct": entry.get("hbm_usage_pct", 0),
        })

    return jsonify({"nodes": result})


@bp.route("/topology/node/<node_id>/shutdown", methods=["POST"])
def shutdown_node(node_id):
    try:
        raw = chimaera_client.get_system_stats_all()
    except Exception as exc:
        return jsonify({"error": f"Cannot query cluster: {exc}"}), 503

    ip = _find_node_ip(raw, node_id)
    if ip is None:
        return jsonify({"error": f"Node {node_id} not found or has no IP"}), 404

    try:
        result = chimaera_client.shutdown_node(ip)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code


@bp.route("/topology/node/<node_id>/restart", methods=["POST"])
def restart_node(node_id):
    try:
        raw = chimaera_client.get_system_stats_all()
    except Exception as exc:
        return jsonify({"error": f"Cannot query cluster: {exc}"}), 503

    ip = _find_node_ip(raw, node_id)
    if ip is None:
        return jsonify({"error": f"Node {node_id} not found or has no IP"}), 404

    try:
        result = chimaera_client.restart_node(ip)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code
