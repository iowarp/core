#!/usr/bin/env python3
"""
Generate multi-node scaling figures for CTE Knowledge Graph benchmark.
Usage: python3 plot_multinode_comparison.py [output_dir]
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Collected results from benchmark runs
# 1-node: head node run (no network)
# 4-node: srun 4-node cluster (CHI_WITH_RUNTIME=0 client)
# 5-node: srun 5-node cluster (CHI_WITH_RUNTIME=0 client)
data = {
    "nodes":           [1,      4,      5],
    "top1":            [20,     20,     20],
    "top1_pct":        [100.0,  100.0,  100.0],
    "top3_pct":        [100.0,  100.0,  100.0],
    "top5_pct":        [100.0,  100.0,  100.0],
    "mrr":             [1.0,    1.0,    1.0],
    "ingest_ms":       [6.62548,  85.5972,  99.3366],
    "index_ms":        [1.63082,  106.134,  119.118],
    "query_ms":        [0.0143973, 1.59788,  1.68351],
    "global_n":        [20,     20,     20],
    "datasets":        [20,     20,     20],
    "queries":         [20,     20,     20],
}

nodes = data["nodes"]
colors_nodes = ['#27ae60', '#2980b9', '#8e44ad']

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)


def save(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


def format_latency(v):
    if v < 1:
        return f"{v*1000:.0f} \u00b5s"
    elif v < 1000:
        return f"{v:.1f} ms"
    else:
        return f"{v/1000:.1f} s"


node_labels = [f"{n} Node{'s' if n > 1 else ''}" for n in nodes]


# ── Figure 1: Accuracy invariance ──
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(nodes))
width = 0.22
gap = 0.04

metrics = [("Top-1", data["top1_pct"]),
           ("Top-3", data["top3_pct"]),
           ("Top-5", data["top5_pct"])]
bar_colors = ['#27ae60', '#2ecc71', '#82e0aa']

for i, (label, vals) in enumerate(metrics):
    offset = (i - 1) * (width + gap)
    bars = ax.bar(x + offset, vals, width, label=label, color=bar_colors[i],
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold',
                color=bar_colors[i])

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Retrieval Accuracy Across Node Counts\n(Accuracy is invariant to cluster size)',
             fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(node_labels, fontsize=12)
ax.set_ylim(0, 120)
ax.legend(fontsize=11, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)
save(fig, 'multinode_accuracy.png')


# ── Figure 2: Search latency ──
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(node_labels, data["query_ms"], color=colors_nodes,
              edgecolor='white', linewidth=0.8, width=0.5, zorder=3)
for bar, v, c in zip(bars, data["query_ms"], colors_nodes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
            format_latency(v), ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=c)

ax.set_ylabel('Avg Query Latency', fontsize=13)
ax.set_title('Search Latency vs. Node Count\n(20 Queries, BM25 Broadcast + Aggregate)',
             fontsize=14, pad=15)
ax.set_ylim(0, max(data["query_ms"]) * 1.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Annotation
ax.annotate(f'Network overhead: {data["query_ms"][1] - data["query_ms"][0]:.1f} ms\n'
            f'(broadcast to {nodes[1]} nodes)',
            xy=(1, data["query_ms"][1]), xytext=(1.5, data["query_ms"][1] * 0.6),
            fontsize=10, color='#7f8c8d',
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2))
save(fig, 'multinode_search_latency.png')


# ── Figure 3: Ingest throughput ──
# Throughput = datasets / ingest_time
ingest_throughput = [data["datasets"][i] / (data["ingest_ms"][i] / 1000.0)
                     for i in range(len(nodes))]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(node_labels, ingest_throughput, color=colors_nodes,
              edgecolor='white', linewidth=0.8, width=0.5, zorder=3)
for bar, v, c in zip(bars, ingest_throughput, colors_nodes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
            f'{v:.0f} ds/s', ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=c)

ax.set_ylabel('Datasets Ingested per Second', fontsize=13)
ax.set_title('Ingest Throughput vs. Node Count\n(20 Datasets with Tag Creation + Blob Storage)',
             fontsize=14, pad=15)
ax.set_ylim(0, max(ingest_throughput) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)
save(fig, 'multinode_ingest_throughput.png')


# ── Figure 4: Queries per second ──
qps = [1000.0 / data["query_ms"][i] for i in range(len(nodes))]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(node_labels, qps, color=colors_nodes,
              edgecolor='white', linewidth=0.8, width=0.5, zorder=3)
for bar, v, c in zip(bars, qps, colors_nodes):
    label = f'{v:.0f} q/s' if v < 10000 else f'{v/1000:.1f}K q/s'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
            label, ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=c)

ax.set_ylabel('Queries per Second', fontsize=13)
ax.set_title('Query Throughput vs. Node Count\n(Sequential Queries, BM25 Broadcast + Aggregate)',
             fontsize=14, pad=15)
ax.set_ylim(0, max(qps) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

ax.annotate(f'Single-node: no network\noverhead \u2192 {qps[0]/qps[1]:.0f}\u00d7 higher QPS',
            xy=(0, qps[0]), xytext=(0.8, qps[0] * 0.6),
            fontsize=10, color='#7f8c8d',
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2))
save(fig, 'multinode_qps.png')


# ── Figure 5: Combined 2x2 summary ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Accuracy
ax = axes[0, 0]
for i, (label, vals) in enumerate(metrics):
    offset = (i - 1) * (width + gap)
    ax.bar(x + offset, vals, width, label=label, color=bar_colors[i],
           edgecolor='white', linewidth=0.5, zorder=3)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Retrieval Accuracy (invariant)', fontsize=12, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(node_labels, fontsize=10)
ax.set_ylim(0, 120)
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Panel 2: Search Latency
ax = axes[0, 1]
bars = ax.bar(node_labels, data["query_ms"], color=colors_nodes,
              edgecolor='white', width=0.5, zorder=3)
for bar, v, c in zip(bars, data["query_ms"], colors_nodes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
            format_latency(v), ha='center', fontsize=10, fontweight='bold', color=c)
ax.set_ylabel('Latency', fontsize=11)
ax.set_title('Avg Search Latency', fontsize=12, pad=10)
ax.set_ylim(0, max(data["query_ms"]) * 1.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Panel 3: Ingest Throughput
ax = axes[1, 0]
bars = ax.bar(node_labels, ingest_throughput, color=colors_nodes,
              edgecolor='white', width=0.5, zorder=3)
for bar, v, c in zip(bars, ingest_throughput, colors_nodes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
            f'{v:.0f} ds/s', ha='center', fontsize=10, fontweight='bold', color=c)
ax.set_ylabel('Datasets / Second', fontsize=11)
ax.set_title('Ingest Throughput', fontsize=12, pad=10)
ax.set_ylim(0, max(ingest_throughput) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Panel 4: QPS
ax = axes[1, 1]
bars = ax.bar(node_labels, qps, color=colors_nodes,
              edgecolor='white', width=0.5, zorder=3)
for bar, v, c in zip(bars, qps, colors_nodes):
    label = f'{v:.0f} q/s' if v < 10000 else f'{v/1000:.1f}K q/s'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
            label, ha='center', fontsize=10, fontweight='bold', color=c)
ax.set_ylabel('Queries / Second', fontsize=11)
ax.set_title('Query Throughput', fontsize=12, pad=10)
ax.set_ylim(0, max(qps) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

fig.suptitle('CTE Knowledge Graph \u2014 Multi-Node Scaling\n20 Datasets \u2022 Partitioned KG \u2022 Global IDF Sync',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'multinode_summary.png')


# Print table
print("\n=== Multi-Node Scaling Summary ===")
print(f"{'Nodes':<8} {'Top-1':>8} {'MRR':>8} {'Query Lat':>12} {'QPS':>10} {'Ingest':>12} {'Ingest Tput':>14}")
print("-" * 72)
for i in range(len(nodes)):
    print(f"{nodes[i]:<8} {data['top1_pct'][i]:>7.0f}% {data['mrr'][i]:>8.3f} "
          f"{format_latency(data['query_ms'][i]):>12} {qps[i]:>9.0f} "
          f"{format_latency(data['ingest_ms'][i]):>12} {ingest_throughput[i]:>11.0f} ds/s")
