#!/usr/bin/env python3
"""
Generate motivation section figures from contention results.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Read contention results
df = pd.read_csv('compression_contention_results.csv')

# Filter out NONE and focus on lossless compressors for now
df = df[df['library_name'] != 'NONE']

# Exclude lossy compressors (ZFP, SZ3, FPZIP) - keep lossless only
lossless_libs = ['ZSTD', 'LZ4', 'ZLIB', 'SNAPPY', 'BROTLI', 'LZMA', 'BZIP2', 'Blosc2']
df = df[df['library_name'].isin(lossless_libs)]

# Sort by compression time for consistent ordering
df = df.sort_values('compress_wall_sec')

libraries = df['library_name'].values
compress_with_contention = df['compress_wall_sec'].values
compress_baseline = df['compress_baseline_sec'].values
contention_slowdown = compress_with_contention / compress_baseline
sim_slowdown_pct = df['sim_slowdown_pct'].values
compression_ratio = df['compression_ratio'].values

# Create color scheme: archival (slow) in red, performance (fast) in blue
colors = []
for lib in libraries:
    if lib in ['BROTLI', 'BZIP2', 'LZMA']:
        colors.append('#d62728')  # Red for archival
    elif lib in ['LZ4', 'SNAPPY']:
        colors.append('#1f77b4')  # Blue for performance
    else:
        colors.append('#ff7f0e')  # Orange for balanced

# Create three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: Compression time comparison (producer vs consumer)
ax = axes[0]
x = np.arange(len(libraries))
width = 0.35

bars1 = ax.bar(x - width/2, compress_baseline, width, label='Consumer (idle)',
               color='lightgray', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, compress_with_contention, width, label='Producer (contention)',
               color=colors, edgecolor='black', linewidth=1)

# Add slowdown text on top of producer bars
for i, (baseline, contention) in enumerate(zip(compress_baseline, compress_with_contention)):
    if baseline > 0:
        slowdown = contention / baseline
        if slowdown > 1.15:  # Only show significant slowdowns
            ax.text(i + width/2, contention, f'{slowdown:.2f}×',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Compression Time (seconds)', fontsize=11, fontweight='bold')
ax.set_xlabel('Compression Library', fontsize=11, fontweight='bold')
ax.set_title('(a) Compression Time Under Load', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(libraries, rotation=45, ha='right')
ax.legend(loc='upper left', fontsize=9)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0.001)

# Panel 2: Simulation slowdown
ax = axes[1]
bars = ax.bar(x, sim_slowdown_pct, color=colors, edgecolor='black', linewidth=1)

# Highlight problematic slowdowns
for i, slowdown in enumerate(sim_slowdown_pct):
    if slowdown > 10:
        ax.text(i, slowdown, f'{slowdown:.0f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')

ax.set_ylabel('Simulation Slowdown (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Compression Library', fontsize=11, fontweight='bold')
ax.set_title('(b) Workflow Slowdown at Producer', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(libraries, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Panel 3: Compression ratio
ax = axes[2]
bars = ax.bar(x, compression_ratio, color=colors, edgecolor='black', linewidth=1)

# Annotate best ratios
for i, ratio in enumerate(compression_ratio):
    if ratio > 1.4:
        ax.text(i, ratio, f'{ratio:.2f}×',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
ax.set_xlabel('Compression Library', fontsize=11, fontweight='bold')
ax.set_title('(c) Achieved Compression Ratio', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(libraries, rotation=45, ha='right')
ax.axhline(y=1, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0.9)

plt.tight_layout()
plt.savefig('motivation/contention_results.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('motivation/contention_results.svg', format='svg', bbox_inches='tight')
print("✅ Saved: motivation/contention_results.pdf")
print("✅ Saved: motivation/contention_results.svg")

# Create summary statistics
print("\n" + "="*80)
print("CONTENTION ANALYSIS SUMMARY")
print("="*80)
print("\nCompression Time Slowdown (Producer vs Consumer):")
for lib, baseline, contended in zip(libraries, compress_baseline, compress_with_contention):
    if baseline > 0:
        slowdown = contended / baseline
        category = "ARCHIVAL" if lib in ['BROTLI', 'BZIP2', 'LZMA'] else \
                  "FAST" if lib in ['LZ4', 'SNAPPY'] else "BALANCED"
        print(f"  {lib:10s} ({category:8s}): {slowdown:.2f}× slower under contention")

print("\nSimulation Impact:")
for lib, slowdown in zip(libraries, sim_slowdown_pct):
    if abs(slowdown) > 1:
        impact = "SEVERE" if slowdown > 100 else "HIGH" if slowdown > 10 else "MODERATE" if slowdown > 1 else "LOW"
        print(f"  {lib:10s}: {slowdown:6.1f}% slowdown ({impact})")

print("\nCompression Ratios:")
for lib, ratio in zip(libraries, compression_ratio):
    quality = "EXCELLENT" if ratio > 1.5 else "GOOD" if ratio > 1.2 else "MODERATE"
    print(f"  {lib:10s}: {ratio:.2f}× compression ({quality})")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("\n1. Archival compressors show highest contention slowdown:")
max_slowdown_idx = np.argmax(contention_slowdown)
print(f"   - {libraries[max_slowdown_idx]}: {contention_slowdown[max_slowdown_idx]:.2f}× slower under load")

print("\n2. Some compressors cause severe workflow slowdown:")
severe_slowdown_libs = [lib for lib, slow in zip(libraries, sim_slowdown_pct) if slow > 100]
if severe_slowdown_libs:
    print(f"   - {', '.join(severe_slowdown_libs)}: >100% simulation slowdown")

print("\n3. Trade-off between ratio and contention sensitivity:")
print("   - High ratio, high sensitivity: BROTLI, BZIP2, LZMA (archival)")
print("   - Low ratio, low sensitivity: LZ4, SNAPPY (performance)")
print("   - Balanced: ZSTD, ZLIB, Blosc2")

print("\n" + "="*80)
