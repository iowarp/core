#!/usr/bin/env python3
"""Plot GADGET-2 benchmark results: single-node + multi-node across all backends."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Single-node results (identical pre-generated summaries)
single = {
    "CTE+BM25":   {"top1": 14, "top3": 17, "top5": 17, "mrr": 0.781, "lat": 0.047},
    "CTE+ES":     {"top1": 15, "top3": 17, "top5": 17, "mrr": 0.806, "lat": 16.4},
    "CTE+Neo4j":  {"top1": 15, "top3": 17, "top5": 17, "mrr": 0.806, "lat": 13.5},
    "CTE+Redis":  {"top1": 10, "top3": 11, "top5": 11, "mrr": 0.525, "lat": 0.42},
    "CTE+Qdrant": {"top1": 19, "top3": 20, "top5": 20, "mrr": 0.967, "lat": 18.5},
}

# Multi-node (4 nodes) results
multi = {
    "CTE+BM25":   {"top1": 13, "top3": 17, "top5": 17, "mrr": 0.756, "lat": 1.28},
    "CTE+ES":     {"top1": 15, "top3": 17, "top5": 17, "mrr": 0.770, "lat": 16.0},
    "CTE+Neo4j":  {"top1": 15, "top3": 15, "top5": 17, "mrr": 0.770, "lat": 14.7},
    "CTE+Redis":  {"top1": 0,  "top3": 0,  "top5": 0,  "mrr": 0.0,   "lat": 1.8},
    "CTE+Qdrant": {"top1": 20, "top3": 20, "top5": 20, "mrr": 1.000, "lat": 61.7},
}

backends = list(single.keys())
n_queries = 20
colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']

# ===== Figure 1: Single-node accuracy =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, metric in enumerate(["top1", "top3", "top5"]):
    ax = axes[idx]
    vals = [single[b][metric] / n_queries * 100 for b in backends]
    bars = ax.bar(range(len(backends)), vals, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(backends)))
    ax.set_xticklabels([b.replace("CTE+", "") for b in backends], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(f"Top-{metric[-1]} Accuracy", fontsize=12, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("GADGET-2 Discovery — Single Node (100 snapshots, 20 queries, LLM summaries)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("gadget2_single_accuracy.png", dpi=150, bbox_inches='tight')
plt.close()

# ===== Figure 2: Query latency comparison =====
fig, ax = plt.subplots(figsize=(10, 6))
lats = [single[b]["lat"] for b in backends]
bars = ax.bar(range(len(backends)), lats, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(backends)))
ax.set_xticklabels([b.replace("CTE+", "") for b in backends], fontsize=10)
ax.set_ylabel("Query Latency (ms)", fontsize=11)
ax.set_title("GADGET-2 Discovery — Query Latency (Single Node)", fontsize=13, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(0.01, 100)
for bar, val in zip(bars, lats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
            f"{val:.2f}ms" if val < 1 else f"{val:.1f}ms",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("gadget2_query_latency.png", dpi=150, bbox_inches='tight')
plt.close()

# ===== Figure 3: Single vs Multi-node comparison =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top-1 accuracy
ax = axes[0]
x = np.arange(len(backends))
w = 0.35
s_vals = [single[b]["top1"] / n_queries * 100 for b in backends]
m_vals = [multi[b]["top1"] / n_queries * 100 for b in backends]
bars1 = ax.bar(x - w/2, s_vals, w, label='Single Node', color='#42A5F5', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, m_vals, w, label='4-Node', color='#EF5350', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([b.replace("CTE+", "") for b in backends], fontsize=9)
ax.set_ylim(0, 115)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax.set_title("Accuracy: Single vs 4-Node", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, s_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}%", ha='center', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, m_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}%", ha='center', fontsize=8, fontweight='bold')

# Query latency
ax = axes[1]
s_lats = [single[b]["lat"] for b in backends]
m_lats = [multi[b]["lat"] for b in backends]
bars1 = ax.bar(x - w/2, s_lats, w, label='Single Node', color='#42A5F5', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, m_lats, w, label='4-Node', color='#EF5350', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([b.replace("CTE+", "") for b in backends], fontsize=9)
ax.set_ylabel("Query Latency (ms)", fontsize=11)
ax.set_title("Latency: Single vs 4-Node", fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(0.01, 200)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.suptitle("GADGET-2 Discovery — Single Node vs 4-Node Distributed",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("gadget2_single_vs_multi.png", dpi=150, bbox_inches='tight')
plt.close()

# ===== Figure 4: MRR comparison =====
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(backends))
w = 0.35
s_mrr = [single[b]["mrr"] for b in backends]
m_mrr = [multi[b]["mrr"] for b in backends]
bars1 = ax.bar(x - w/2, s_mrr, w, label='Single Node', color='#42A5F5', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, m_mrr, w, label='4-Node', color='#EF5350', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([b.replace("CTE+", "") for b in backends], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Mean Reciprocal Rank", fontsize=11)
ax.set_title("GADGET-2 Discovery — MRR (Single vs 4-Node)", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, s_mrr):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, m_mrr):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("gadget2_mrr.png", dpi=150, bbox_inches='tight')
plt.close()

print("Generated 4 figures:")
print("  gadget2_single_accuracy.png")
print("  gadget2_query_latency.png")
print("  gadget2_single_vs_multi.png")
print("  gadget2_mrr.png")
