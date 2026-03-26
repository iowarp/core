#!/usr/bin/env python3
"""
Generate comparison figures for CTE vs Graphiti vs Hindsight benchmarks.
Usage: python3 plot_benchmark_comparison.py [output_dir]
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Results
results = {
    "CTE\n(BM25 + Qwen 2.5 3B)": {
        "top1_accuracy": 1.0,
        "top3_accuracy": 1.0,
        "top5_accuracy": 1.0,
        "mrr": 1.0,
        "ingest_latency_ms": 6.62548,
        "index_latency_ms": 1.63082,
        "avg_query_latency_ms": 0.0143973,
    },
    "Hindsight\n(LLM)": {
        "top1_accuracy": 1.0,
        "top3_accuracy": 1.0,
        "top5_accuracy": 1.0,
        "mrr": 1.0,
        "ingest_latency_ms": 126.3430118560791,
        "index_latency_ms": 106248.29578399658,
        "avg_query_latency_ms": 304.75289821624756,
    },
    "Graphiti\n(LLM)": {
        "top1_accuracy": 0.4,
        "top3_accuracy": 0.5,
        "top5_accuracy": 0.65,
        "mrr": 0.47666666666666657,
        "ingest_latency_ms": 221.77457809448242,
        "index_latency_ms": 486250.0,
        "avg_query_latency_ms": 146.70528173446655,
    },
}

providers = list(results.keys())
colors = ['#27ae60', '#2980b9', '#c0392b']

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)


def save(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white',
                pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


def format_latency(v):
    if v < 1:
        return f"{v*1000:.0f} \u00b5s"
    elif v < 1000:
        return f"{v:.1f} ms"
    elif v < 60000:
        return f"{v/1000:.1f} s"
    else:
        return f"{v/60000:.1f} min"


# ── Figure 1: Accuracy (Top-1 / Top-3 / Top-5) ──
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(3)
width = 0.22
gap = 0.04
metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
labels = ['Top-1', 'Top-3', 'Top-5']

for i, prov in enumerate(providers):
    vals = [results[prov][m] * 100 for m in metrics]
    offset = (i - 1) * (width + gap)
    bars = ax.bar(x + offset, vals, width, label=prov, color=colors[i],
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=colors[i])

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Dataset Discovery Accuracy\n20 Scientific Datasets \u2022 20 Natural-Language Queries',
             fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 120)
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)
save(fig, 'accuracy_comparison.png')


# ── Figure 2: MRR ──
fig, ax = plt.subplots(figsize=(7, 5))
mrr_vals = [results[p]['mrr'] for p in providers]
bars = ax.bar(providers, mrr_vals, color=colors, edgecolor='white',
              linewidth=0.8, width=0.45, zorder=3)
for bar, v, c in zip(bars, mrr_vals, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f'{v:.3f}', ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=c)

ax.set_ylabel('Mean Reciprocal Rank', fontsize=13)
ax.set_title('Mean Reciprocal Rank (MRR)', fontsize=14, pad=15)
ax.set_ylim(0, 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)
save(fig, 'mrr_comparison.png')


# ── Figure 3: Query Latency (log scale) ──
fig, ax = plt.subplots(figsize=(7, 5.5))
query_lat = [results[p]['avg_query_latency_ms'] for p in providers]
bars = ax.bar(providers, query_lat, color=colors, edgecolor='white',
              linewidth=0.8, width=0.45, zorder=3)
ax.set_yscale('log')
for bar, v, c in zip(bars, query_lat, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 3,
            format_latency(v), ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=c)

ax.set_ylabel('Avg Query Latency (log scale)', fontsize=13)
ax.set_title('Query Latency Comparison', fontsize=14, pad=15)
ax.set_ylim(0.005, 5000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Add speedup annotation
ax.annotate(f'CTE is {query_lat[1]/query_lat[0]:.0f}\u00d7 faster\nthan Hindsight',
            xy=(0, query_lat[0]), xytext=(0.8, 0.02),
            fontsize=10, color='#27ae60', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60'))
save(fig, 'query_latency_comparison.png')


# ── Figure 4: Index Latency (log scale) ──
fig, ax = plt.subplots(figsize=(7, 5.5))
index_lat = [results[p]['index_latency_ms'] for p in providers]
bars = ax.bar(providers, index_lat, color=colors, edgecolor='white',
              linewidth=0.8, width=0.45, zorder=3)
ax.set_yscale('log')
for bar, v, c in zip(bars, index_lat, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 3,
            format_latency(v), ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=c)

ax.set_ylabel('KG Index Latency (log scale)', fontsize=13)
ax.set_title('Knowledge Graph Indexing Latency\n20 Dataset Descriptions', fontsize=14, pad=15)
ax.set_ylim(0.1, 5e7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

ax.annotate(f'CTE is {index_lat[2]/index_lat[0]:.0f}\u00d7 faster\nthan Graphiti',
            xy=(0, index_lat[0]), xytext=(0.8, 0.3),
            fontsize=10, color='#27ae60', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60'))
save(fig, 'index_latency_comparison.png')


# ── Figure 5: Combined summary (2x2) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Accuracy
ax = axes[0, 0]
for i, prov in enumerate(providers):
    vals = [results[prov][m] * 100 for m in metrics]
    offset = (i - 1) * (width + gap)
    bars = ax.bar(x + offset, vals, width, label=prov, color=colors[i],
                  edgecolor='white', linewidth=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold', color=colors[i])
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Retrieval Accuracy', fontsize=12, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 120)
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Panel 2: MRR
ax = axes[0, 1]
bars = ax.bar(providers, mrr_vals, color=colors, edgecolor='white', width=0.45, zorder=3)
for bar, v, c in zip(bars, mrr_vals, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f'{v:.3f}', ha='center', fontsize=10, fontweight='bold', color=c)
ax.set_ylabel('MRR', fontsize=11)
ax.set_title('Mean Reciprocal Rank', fontsize=12, pad=10)
ax.set_ylim(0, 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Panel 3: Query Latency
ax = axes[1, 0]
bars = ax.bar(providers, query_lat, color=colors, edgecolor='white', width=0.45, zorder=3)
ax.set_yscale('log')
for bar, v, c in zip(bars, query_lat, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 4,
            format_latency(v), ha='center', fontsize=10, fontweight='bold', color=c)
ax.set_ylabel('Latency (log scale)', fontsize=11)
ax.set_title('Avg Query Latency', fontsize=12, pad=10)
ax.set_ylim(0.005, 5000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Panel 4: Index Latency
ax = axes[1, 1]
bars = ax.bar(providers, index_lat, color=colors, edgecolor='white', width=0.45, zorder=3)
ax.set_yscale('log')
for bar, v, c in zip(bars, index_lat, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 4,
            format_latency(v), ha='center', fontsize=10, fontweight='bold', color=c)
ax.set_ylabel('Latency (log scale)', fontsize=11)
ax.set_title('KG Index Latency', fontsize=12, pad=10)
ax.set_ylim(0.1, 5e7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, zorder=0)

fig.suptitle('CTE Knowledge Graph Benchmark\nSingle Node \u2022 20 Datasets \u2022 20 Queries',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'benchmark_summary.png')

# ── Speedup table ──
print("\n=== Speedup Summary ===")
prov_keys = ["CTE\n(BM25 + Qwen 2.5 3B)", "Hindsight\n(LLM)", "Graphiti\n(LLM)"]
print(f"{'Metric':<25} {'CTE':>12} {'Hindsight':>12} {'Graphiti':>12} {'vs Hindsight':>14} {'vs Graphiti':>14}")
print("-" * 89)
for label, key in [("Query Latency", "avg_query_latency_ms"),
                    ("Index Latency", "index_latency_ms"),
                    ("Ingest Latency", "ingest_latency_ms")]:
    c = results[prov_keys[0]][key]
    h = results[prov_keys[1]][key]
    g = results[prov_keys[2]][key]
    print(f"{label:<25} {format_latency(c):>12} {format_latency(h):>12} {format_latency(g):>12} {h/c:>13.0f}x {g/c:>13.0f}x")

print(f"\n{'Top-1 Accuracy':<25} {'100%':>12} {'100%':>12} {'40%':>12}")
print(f"{'MRR':<25} {'1.000':>12} {'1.000':>12} {'0.477':>12}")
