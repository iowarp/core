#!/usr/bin/env python3
"""Plot KG backend comparison results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

backends = ["CTE\nBM25", "Elastic-\nsearch", "Neo4j", "Redis", "Qdrant"]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

top1 = [100, 95, 100, 75, 95]
top3 = [100, 100, 100, 90, 100]
top5 = [100, 100, 100, 95, 100]
mrr = [1.0, 0.975, 1.0, 0.838, 0.975]
query_ms = [0.025, 18.2, 14.9, 0.29, 17.2]
index_ms = [0.52, 199, 588, 6.8, 399]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("CTE KG Backend Comparison\n(20 Scientific Datasets, Qwen 2.5 3B Keywords)",
             fontsize=15, fontweight='bold', y=0.98)

# 1. Top-1/3/5 Accuracy
ax = axes[0, 0]
x = np.arange(len(backends))
w = 0.25
bars1 = ax.bar(x - w, top1, w, label='Top-1', color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, top3, w, label='Top-3', color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + w, top5, w, label='Top-5', color=colors, alpha=0.3, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Retrieval Accuracy', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(backends, fontsize=9)
ax.set_ylim(0, 118)
ax.legend(fontsize=9, loc='upper right')
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 1,
            f'{h:.0f}', ha='center', va='bottom', fontsize=8)

# 2. Query Latency (log scale)
ax = axes[0, 1]
bars = ax.bar(x, query_ms, 0.5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Query Latency (ms)', fontsize=11)
ax.set_title('Average Query Latency', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(backends, fontsize=9)
ax.set_ylim(0.005, 200)
for bar, val in zip(bars, query_ms):
    y = bar.get_height()
    label = f'{val:.3f}' if val < 1 else f'{val:.1f}'
    ax.text(bar.get_x() + bar.get_width()/2., y * 2.0,
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Index Time (log scale)
ax = axes[1, 0]
bars = ax.bar(x, index_ms, 0.5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Index Time (ms)', fontsize=11)
ax.set_title('Index Time (20 Entries)', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(backends, fontsize=9)
ax.set_ylim(0.1, 5000)
for bar, val in zip(bars, index_ms):
    y = bar.get_height()
    label = f'{val:.1f}' if val < 10 else f'{val:.0f}'
    ax.text(bar.get_x() + bar.get_width()/2., y * 2.0,
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Speedup vs CTE
ax = axes[1, 1]
speedups = [query_ms[i] / query_ms[0] for i in range(len(backends))]
bars = ax.bar(x, speedups, 0.5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Slowdown vs CTE BM25', fontsize=11)
ax.set_title('Query Latency Relative to CTE', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(backends, fontsize=9)
ax.set_ylim(0.3, 3000)
for bar, val in zip(bars, speedups):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., y * 2.0,
            f'{val:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
outpath = '/mnt/common/rpawar4/clio-core/context-assimilation-engine/test/unit/hdf5_assim/backend_comparison.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
