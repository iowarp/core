#!/usr/bin/env python3
"""
Compression Benchmark Results - Comprehensive Analysis and Visualization

This script performs meaningful analysis of compression benchmark results:
- Compression ratio vs speed trade-offs
- Best compressor recommendations by use case
- Performance efficiency metrics
- Scalability analysis
- Data type/distribution effectiveness
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Color palette for libraries
LIBRARY_COLORS = {
    'BZIP2': '#1f77b4', 'Blosc2': '#ff7f0e', 'Brotli': '#2ca02c',
    'LZ4': '#d62728', 'LZO': '#9467bd', 'Lzma': '#8c564b',
    'Snappy': '#e377c2', 'Zlib': '#7f7f7f', 'Zstd': '#bcbd22'
}

def load_data(csv_path):
    """Load and preprocess benchmark data."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} benchmark records")
        
        # Add derived metrics
        df['Total Time (ms)'] = df['Compress Time (ms)'] + df['Decompress Time (ms)']
        df['Compression Throughput (MB/s)'] = (df['Chunk Size (bytes)'] / (1024**2)) / (df['Compress Time (ms)'] / 1000.0)
        df['Decompression Throughput (MB/s)'] = (df['Chunk Size (bytes)'] / (1024**2)) / (df['Decompress Time (ms)'] / 1000.0)
        df['Efficiency Score'] = df['Compression Ratio'] / (df['Total Time (ms)'] + 1)  # Ratio per ms
        df['Chunk Size (MB)'] = df['Chunk Size (bytes)'] / (1024**2)
        
        # Filter out invalid data
        df = df[df['Compress Time (ms)'] > 0]
        df = df[df['Decompress Time (ms)'] > 0]
        df = df[df['Compression Ratio'] > 0]
        
        print(f"✓ Processed {len(df)} valid records")
        return df
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        sys.exit(1)

def plot_compression_trade_off(df, output_dir):
    """Plot compression ratio vs speed trade-off - THE MOST IMPORTANT PLOT."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Average metrics per library
    library_stats = df.groupby('Library').agg({
        'Compression Ratio': 'mean',
        'Compress Time (ms)': 'mean',
        'Total Time (ms)': 'mean',
        'Compression Throughput (MB/s)': 'mean'
    }).reset_index()
    
    # Left: Compression Ratio vs Compression Time (log scale for time)
    for lib in library_stats['Library'].unique():
        lib_data = library_stats[library_stats['Library'] == lib]
        color = LIBRARY_COLORS.get(lib, 'gray')
        axes[0].scatter(lib_data['Compress Time (ms)'], lib_data['Compression Ratio'],
                       s=300, alpha=0.7, color=color, label=lib, edgecolors='black', linewidth=1.5)
        axes[0].annotate(lib, (lib_data['Compress Time (ms)'].values[0], 
                               lib_data['Compression Ratio'].values[0]),
                        fontsize=9, ha='center', va='bottom', fontweight='bold')
    
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Compression Time (ms, log scale)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Compression Ratio', fontsize=12, fontweight='bold')
    axes[0].set_title('Compression Ratio vs Speed Trade-off\n(Higher Ratio + Lower Time = Better)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='1ms threshold')
    axes[0].axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='No compression')
    
    # Right: Compression Ratio vs Total Time (compress + decompress)
    for lib in library_stats['Library'].unique():
        lib_data = library_stats[library_stats['Library'] == lib]
        color = LIBRARY_COLORS.get(lib, 'gray')
        axes[1].scatter(lib_data['Total Time (ms)'], lib_data['Compression Ratio'],
                       s=300, alpha=0.7, color=color, label=lib, edgecolors='black', linewidth=1.5)
        axes[1].annotate(lib, (lib_data['Total Time (ms)'].values[0], 
                               lib_data['Compression Ratio'].values[0]),
                        fontsize=9, ha='center', va='bottom', fontweight='bold')
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Total Time: Compress + Decompress (ms, log scale)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Compression Ratio', fontsize=12, fontweight='bold')
    axes[1].set_title('End-to-End Performance\n(Compression + Decompression)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_tradeoff.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: compression_tradeoff.png")
    plt.close()

def plot_efficiency_analysis(df, output_dir):
    """Plot efficiency metrics - compression ratio per unit time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    library_stats = df.groupby('Library').agg({
        'Compression Ratio': 'mean',
        'Compress Time (ms)': 'mean',
        'Decompress Time (ms)': 'mean',
        'Efficiency Score': 'mean',
        'Compression Throughput (MB/s)': 'mean'
    }).sort_values('Efficiency Score', ascending=False)
    
    # Top-left: Efficiency Score (Ratio per ms)
    colors = [LIBRARY_COLORS.get(lib, 'gray') for lib in library_stats.index]
    bars = axes[0, 0].barh(range(len(library_stats)), library_stats['Efficiency Score'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_yticks(range(len(library_stats)))
    axes[0, 0].set_yticklabels(library_stats.index)
    axes[0, 0].set_xlabel('Efficiency Score (Ratio / Total Time)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Compression Efficiency Ranking\n(Higher = Better Ratio per Time)', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, (lib, val) in enumerate(zip(library_stats.index, library_stats['Efficiency Score'])):
        axes[0, 0].text(val, i, f'{val:.1f}', va='center', ha='left', fontweight='bold')
    
    # Top-right: Throughput comparison
    library_stats_thru = library_stats.sort_values('Compression Throughput (MB/s)', ascending=True)
    colors = [LIBRARY_COLORS.get(lib, 'gray') for lib in library_stats_thru.index]
    bars = axes[0, 1].barh(range(len(library_stats_thru)), 
                          library_stats_thru['Compression Throughput (MB/s)'],
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_yticks(range(len(library_stats_thru)))
    axes[0, 1].set_yticklabels(library_stats_thru.index)
    axes[0, 1].set_xlabel('Compression Throughput (MB/s)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Compression Speed (Throughput)\n(Higher = Faster)', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    for i, (lib, val) in enumerate(zip(library_stats_thru.index, 
                                       library_stats_thru['Compression Throughput (MB/s)'])):
        axes[0, 1].text(val, i, f'{val:.1f}', va='center', ha='left', fontweight='bold')
    
    # Bottom-left: Compression ratio by data distribution
    pivot_ratio = df.pivot_table(values='Compression Ratio', index='Library', 
                                 columns='Distribution', aggfunc='mean')
    pivot_ratio = pivot_ratio.reindex(library_stats.index)
    pivot_ratio.plot(kind='bar', ax=axes[1, 0], width=0.8, alpha=0.8, edgecolor='black')
    axes[1, 0].set_xlabel('Library', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Compression Ratio by Data Distribution\n(Shows which compressors work best for which data)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].legend(title='Distribution', fontsize=9, title_fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Bottom-right: Speed comparison (compress vs decompress)
    library_speed = df.groupby('Library').agg({
        'Compress Time (ms)': 'mean',
        'Decompress Time (ms)': 'mean'
    }).sort_values('Compress Time (ms)')
    x = np.arange(len(library_speed))
    width = 0.35
    colors_compress = [LIBRARY_COLORS.get(lib, 'gray') for lib in library_speed.index]
    colors_decompress = [LIBRARY_COLORS.get(lib, 'lightblue') for lib in library_speed.index]
    axes[1, 1].bar(x - width/2, library_speed['Compress Time (ms)'], width,
                   label='Compress', alpha=0.8, color=colors_compress, edgecolor='black')
    axes[1, 1].bar(x + width/2, library_speed['Decompress Time (ms)'], width,
                   label='Decompress', alpha=0.8, color=colors_decompress, edgecolor='black')
    axes[1, 1].set_xlabel('Library', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Compression vs Decompression Speed\n(Compare compress vs decompress times)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(library_speed.index, rotation=45, ha='right')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: efficiency_analysis.png")
    plt.close()

def plot_scalability_analysis(df, output_dir):
    """Analyze how performance scales with chunk size."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    chunk_sizes = sorted(df['Chunk Size (bytes)'].unique())
    chunk_labels = [f"{cs/(1024**2):.1f}MB" if cs >= 1024**2 else f"{cs/1024:.0f}KB" 
                    for cs in chunk_sizes]
    
    # Top-left: Compression ratio vs chunk size
    for lib in df['Library'].unique():
        lib_data = df[df['Library'] == lib]
        ratios = [lib_data[lib_data['Chunk Size (bytes)'] == cs]['Compression Ratio'].mean()
                 for cs in chunk_sizes]
        axes[0, 0].plot(chunk_labels, ratios, marker='o', label=lib, linewidth=2.5, 
                       markersize=8, color=LIBRARY_COLORS.get(lib, 'gray'), alpha=0.8)
    axes[0, 0].set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Compression Ratio vs Chunk Size\n(Shows scalability of compression)', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Top-right: Compression time vs chunk size (log scale)
    for lib in df['Library'].unique():
        lib_data = df[df['Library'] == lib]
        times = [lib_data[lib_data['Chunk Size (bytes)'] == cs]['Compress Time (ms)'].mean()
                for cs in chunk_sizes]
        axes[0, 1].plot(chunk_labels, times, marker='s', label=lib, linewidth=2.5,
                       markersize=8, color=LIBRARY_COLORS.get(lib, 'gray'), alpha=0.8)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Compression Time (ms, log scale)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Compression Time vs Chunk Size\n(Linear = good scalability)', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, which='both')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Bottom-left: Throughput vs chunk size
    for lib in df['Library'].unique():
        lib_data = df[df['Library'] == lib]
        throughputs = []
        for cs in chunk_sizes:
            cs_data = lib_data[lib_data['Chunk Size (bytes)'] == cs]
            if len(cs_data) > 0:
                size_mb = cs / (1024.0 * 1024.0)
                avg_time = cs_data['Compress Time (ms)'].mean() / 1000.0
                if avg_time > 0:
                    throughputs.append(size_mb / avg_time)
                else:
                    throughputs.append(np.nan)
            else:
                throughputs.append(np.nan)
        axes[1, 0].plot(chunk_labels, throughputs, marker='^', label=lib, linewidth=2.5,
                       markersize=8, color=LIBRARY_COLORS.get(lib, 'gray'), alpha=0.8)
    axes[1, 0].set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Throughput (MB/s)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Compression Throughput vs Chunk Size\n(Higher = better for large files)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Bottom-right: Compression ratio improvement with size (relative to smallest chunk)
    for lib in df['Library'].unique():
        lib_data = df[df['Library'] == lib]
        base_ratio = lib_data[lib_data['Chunk Size (bytes)'] == chunk_sizes[0]]['Compression Ratio'].mean()
        ratios = [lib_data[lib_data['Chunk Size (bytes)'] == cs]['Compression Ratio'].mean() / base_ratio
                 for cs in chunk_sizes]
        axes[1, 1].plot(chunk_labels, ratios, marker='d', label=lib, linewidth=2.5,
                       markersize=8, color=LIBRARY_COLORS.get(lib, 'gray'), alpha=0.8)
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No improvement')
    axes[1, 1].set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Ratio Improvement (relative to smallest)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Compression Ratio Improvement with Size\n(>1 = better at larger sizes)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: scalability_analysis.png")
    plt.close()

def plot_data_type_effectiveness(df, output_dir):
    """Analyze which compressors work best for which data types/distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Heatmap - Compression ratio by library and distribution
    pivot_ratio = df.pivot_table(values='Compression Ratio', index='Library', 
                                 columns='Distribution', aggfunc='mean')
    # Reorder by average ratio
    pivot_ratio['avg'] = pivot_ratio.mean(axis=1)
    pivot_ratio = pivot_ratio.sort_values('avg', ascending=False).drop('avg', axis=1)
    
    sns.heatmap(pivot_ratio, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0, 0],
                cbar_kws={'label': 'Compression Ratio'}, linewidths=0.5, linecolor='gray')
    axes[0, 0].set_title('Compression Ratio Heatmap\n(Find best compressor for each data type)', 
                         fontsize=12, fontweight='bold', pad=15)
    axes[0, 0].set_xlabel('Data Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Library', fontsize=11, fontweight='bold')
    
    # Top-right: Heatmap - Compression time by library and distribution
    pivot_time = df.pivot_table(values='Compress Time (ms)', index='Library',
                                columns='Distribution', aggfunc='mean')
    pivot_time = pivot_time.reindex(pivot_ratio.index)
    
    sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='YlGnBu_r', ax=axes[0, 1],
                cbar_kws={'label': 'Time (ms)'}, linewidths=0.5, linecolor='gray')
    axes[0, 1].set_title('Compression Time Heatmap (ms)\n(Lower = faster)', 
                         fontsize=12, fontweight='bold', pad=15)
    axes[0, 1].set_xlabel('Data Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Library', fontsize=11, fontweight='bold')
    
    # Bottom-left: Best compressor for each distribution (compression ratio)
    best_by_dist = {}
    for dist in df['Distribution'].unique():
        dist_data = df[df['Distribution'] == dist]
        best_lib = dist_data.groupby('Library')['Compression Ratio'].mean().idxmax()
        best_ratio = dist_data.groupby('Library')['Compression Ratio'].mean().max()
        best_by_dist[dist] = (best_lib, best_ratio)
    
    dists = list(best_by_dist.keys())
    libs = [best_by_dist[d][0] for d in dists]
    ratios = [best_by_dist[d][1] for d in dists]
    colors = [LIBRARY_COLORS.get(lib, 'gray') for lib in libs]
    
    bars = axes[1, 0].bar(range(len(dists)), ratios, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
    axes[1, 0].set_xticks(range(len(dists)))
    axes[1, 0].set_xticklabels([f"{d}\n({libs[i]})" for i, d in enumerate(dists)], 
                               rotation=45, ha='right')
    axes[1, 0].set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Best Compressor for Each Distribution\n(Optimal choice by data type)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, ratio, f'{ratio:.2f}x',
                       ha='center', va='bottom', fontweight='bold')
    
    # Bottom-right: Compression ratio distribution (box plot)
    df_melted = df.melt(id_vars=['Library', 'Distribution'], 
                       value_vars=['Compression Ratio'],
                       var_name='Metric', value_name='Value')
    sns.boxplot(data=df_melted, x='Library', y='Value', hue='Distribution', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Library', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Compression Ratio Distribution\n(Shows variability and consistency)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend(title='Distribution', fontsize=9, title_fontsize=10, 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_type_effectiveness.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: data_type_effectiveness.png")
    plt.close()

def plot_performance_summary_dashboard(df, output_dir):
    """Create a comprehensive performance summary dashboard."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    
    # Calculate summary statistics
    library_stats = df.groupby('Library').agg({
        'Compression Ratio': ['mean', 'max'],
        'Compress Time (ms)': 'mean',
        'Decompress Time (ms)': 'mean',
        'Efficiency Score': 'mean',
        'Compression Throughput (MB/s)': 'mean'
    })
    library_stats.columns = ['Avg Ratio', 'Max Ratio', 'Avg Compress Time', 
                            'Avg Decompress Time', 'Efficiency', 'Throughput']
    library_stats = library_stats.sort_values('Efficiency', ascending=False)
    
    # 1. Overall ranking by efficiency (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = [LIBRARY_COLORS.get(lib, 'gray') for lib in library_stats.index]
    bars = ax1.barh(range(len(library_stats)), library_stats['Efficiency'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(library_stats)))
    ax1.set_yticklabels(library_stats.index, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Efficiency Score (Compression Ratio / Total Time)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Performance Ranking - Compression Efficiency', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    for i, (lib, eff) in enumerate(zip(library_stats.index, library_stats['Efficiency'])):
        ax1.text(eff, i, f'{eff:.1f}', va='center', ha='left', fontweight='bold', fontsize=10)
    
    # 2. Key metrics comparison (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    top5 = library_stats.head(5)
    metrics = ['Avg Ratio', 'Throughput']
    x = np.arange(len(metrics))
    width = 0.15
    for i, lib in enumerate(top5.index):
        values = [top5.loc[lib, 'Avg Ratio'] / top5['Avg Ratio'].max() * 100,
                 top5.loc[lib, 'Throughput'] / top5['Throughput'].max() * 100]
        ax2.bar(x + i*width, values, width, label=lib[:6], alpha=0.8, 
               color=LIBRARY_COLORS.get(lib, 'gray'), edgecolor='black')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(['Ratio\n(normalized)', 'Speed\n(normalized)'], fontsize=10)
    ax2.set_ylabel('Normalized Score (%)', fontsize=10)
    ax2.set_title('Top 5: Key Metrics', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Compression ratio comparison
    ax3 = fig.add_subplot(gs[1, 0])
    library_stats_ratio = library_stats.sort_values('Avg Ratio', ascending=True)
    colors = [LIBRARY_COLORS.get(lib, 'gray') for lib in library_stats_ratio.index]
    bars = ax3.barh(range(len(library_stats_ratio)), library_stats_ratio['Avg Ratio'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_yticks(range(len(library_stats_ratio)))
    ax3.set_yticklabels(library_stats_ratio.index, fontsize=10)
    ax3.set_xlabel('Avg Compression Ratio', fontsize=10, fontweight='bold')
    ax3.set_title('Compression Ratio\n(Higher = Better)', fontsize=11, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Speed comparison
    ax4 = fig.add_subplot(gs[1, 1])
    library_stats_speed = library_stats.sort_values('Throughput', ascending=True)
    colors = [LIBRARY_COLORS.get(lib, 'gray') for lib in library_stats_speed.index]
    bars = ax4.barh(range(len(library_stats_speed)), library_stats_speed['Throughput'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(library_stats_speed)))
    ax4.set_yticklabels(library_stats_speed.index, fontsize=10)
    ax4.set_xlabel('Throughput (MB/s)', fontsize=10, fontweight='bold')
    ax4.set_title('Compression Speed\n(Higher = Faster)', fontsize=11, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Best use cases
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Find best compressor for different scenarios
    best_for_speed = library_stats['Throughput'].idxmax()
    best_for_ratio = library_stats['Avg Ratio'].idxmax()
    best_for_efficiency = library_stats['Efficiency'].idxmax()
    
    use_cases = [
        f"⚡ Fastest: {best_for_speed}",
        f"📦 Best Ratio: {best_for_ratio}",
        f"⚖️ Best Balance: {best_for_efficiency}",
        f"\n📊 Top 3 Overall:",
        f"1. {library_stats.index[0]}",
        f"2. {library_stats.index[1]}",
        f"3. {library_stats.index[2]}"
    ]
    
    textstr = '\n'.join(use_cases)
    ax5.text(0.1, 0.5, textstr, transform=ax5.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontweight='bold')
    ax5.set_title('Recommendations', fontsize=12, fontweight='bold', pad=10)
    
    # 6. Ratio vs Speed scatter (bottom-left, spans 2 columns)
    ax6 = fig.add_subplot(gs[2, :2])
    for lib in df['Library'].unique():
        lib_data = df[df['Library'] == lib]
        ax6.scatter(lib_data['Compress Time (ms)'], lib_data['Compression Ratio'],
                   label=lib, alpha=0.5, s=30, color=LIBRARY_COLORS.get(lib, 'gray'),
                   edgecolors='black', linewidth=0.5)
    ax6.set_xscale('log')
    ax6.set_xlabel('Compression Time (ms, log scale)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    ax6.set_title('All Data Points: Ratio vs Speed\n(Each point = one benchmark run)', 
                 fontsize=12, fontweight='bold')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3, which='both')
    
    # 7. Performance table (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = library_stats.round(2)[['Avg Ratio', 'Throughput', 'Efficiency']].head(8)
    table = ax7.table(cellText=table_data.values,
                     colLabels=['Ratio', 'Speed\n(MB/s)', 'Efficiency'],
                     rowLabels=table_data.index,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for i in range(len(table_data.index)):
        table[(i+1, 0)].set_facecolor(list(LIBRARY_COLORS.values())[i % len(LIBRARY_COLORS)])
        table[(i+1, 0)].set_alpha(0.3)
    ax7.set_title('Performance Metrics', fontsize=11, fontweight='bold', pad=10)
    
    # 8. CPU utilization (bottom row, all 3 columns)
    ax8 = fig.add_subplot(gs[3, :])
    library_cpu = df.groupby('Library').agg({
        'Compress CPU %': 'mean',
        'Decompress CPU %': 'mean'
    }).reindex(library_stats.index)
    
    x = np.arange(len(library_cpu))
    width = 0.35
    ax8.bar(x - width/2, library_cpu['Compress CPU %'], width,
           label='Compress', alpha=0.8, color='steelblue', edgecolor='black')
    ax8.bar(x + width/2, library_cpu['Decompress CPU %'], width,
           label='Decompress', alpha=0.8, color='coral', edgecolor='black')
    ax8.set_xlabel('Library', fontsize=11, fontweight='bold')
    ax8.set_ylabel('CPU Utilization (%)', fontsize=11, fontweight='bold')
    ax8.set_title('CPU Utilization by Library\n(Shows computational cost)', 
                 fontsize=12, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(library_cpu.index, rotation=45, ha='right', fontsize=10)
    ax8.legend(fontsize=10)
    ax8.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Compression Benchmark - Comprehensive Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_dashboard.png")
    plt.close()

def generate_insights_report(df, output_dir):
    """Generate actionable insights and recommendations."""
    report_path = os.path.join(output_dir, 'insights_and_recommendations.txt')
    
    # Calculate key metrics
    library_stats = df.groupby('Library').agg({
        'Compression Ratio': ['mean', 'max', 'std'],
        'Compress Time (ms)': 'mean',
        'Decompress Time (ms)': 'mean',
        'Efficiency Score': 'mean',
        'Compression Throughput (MB/s)': 'mean',
        'Compress CPU %': 'mean'
    })
    
    library_stats.columns = ['Avg Ratio', 'Max Ratio', 'Ratio Std', 'Avg Compress Time',
                            'Avg Decompress Time', 'Efficiency', 'Throughput', 'CPU %']
    library_stats = library_stats.sort_values('Efficiency', ascending=False)
    
    # Find best for different use cases
    best_fastest = library_stats['Throughput'].idxmax()
    best_ratio = library_stats['Avg Ratio'].idxmax()
    best_efficiency = library_stats['Efficiency'].idxmax()
    best_consistent = (library_stats['Ratio Std'] / library_stats['Avg Ratio']).idxmin()
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPRESSION BENCHMARK - INSIGHTS & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total benchmarks: {len(df)}\n")
        f.write(f"Libraries tested: {df['Library'].nunique()}\n")
        f.write(f"Data types: {df['Distribution'].nunique()} distributions\n")
        f.write(f"Chunk sizes: {df['Chunk Size (bytes)'].nunique()} sizes\n\n")
        
        f.write("BEST COMPRESSOR BY USE CASE\n")
        f.write("-" * 80 + "\n")
        f.write(f"⚡ For Speed (Highest Throughput):\n")
        f.write(f"   → {best_fastest}\n")
        f.write(f"   Throughput: {library_stats.loc[best_fastest, 'Throughput']:.2f} MB/s\n")
        f.write(f"   Avg Compression Time: {library_stats.loc[best_fastest, 'Avg Compress Time']:.2f} ms\n\n")
        
        f.write(f"📦 For Maximum Compression:\n")
        f.write(f"   → {best_ratio}\n")
        f.write(f"   Avg Ratio: {library_stats.loc[best_ratio, 'Avg Ratio']:.2f}x\n")
        f.write(f"   Max Ratio: {library_stats.loc[best_ratio, 'Max Ratio']:.2f}x\n\n")
        
        f.write(f"⚖️ For Best Balance (Efficiency):\n")
        f.write(f"   → {best_efficiency}\n")
        f.write(f"   Efficiency Score: {library_stats.loc[best_efficiency, 'Efficiency']:.2f}\n")
        f.write(f"   Avg Ratio: {library_stats.loc[best_efficiency, 'Avg Ratio']:.2f}x\n")
        f.write(f"   Throughput: {library_stats.loc[best_efficiency, 'Throughput']:.2f} MB/s\n\n")
        
        f.write(f"🎯 For Consistency (Low Variability):\n")
        f.write(f"   → {best_consistent}\n")
        f.write(f"   Coefficient of Variation: {library_stats.loc[best_consistent, 'Ratio Std'] / library_stats.loc[best_consistent, 'Avg Ratio']:.3f}\n\n")
        
        f.write("PERFORMANCE RANKINGS\n")
        f.write("-" * 80 + "\n")
        f.write("\nBy Overall Efficiency (Ratio / Time):\n")
        for i, (lib, row) in enumerate(library_stats.iterrows(), 1):
            f.write(f"  {i}. {lib:8s} - Efficiency: {row['Efficiency']:7.2f} | "
                   f"Ratio: {row['Avg Ratio']:6.2f}x | Speed: {row['Throughput']:6.1f} MB/s\n")
        
        f.write("\nBy Compression Ratio:\n")
        ratio_ranked = library_stats.sort_values('Avg Ratio', ascending=False)
        for i, (lib, row) in enumerate(ratio_ranked.iterrows(), 1):
            f.write(f"  {i}. {lib:8s} - {row['Avg Ratio']:6.2f}x (max: {row['Max Ratio']:.2f}x)\n")
        
        f.write("\nBy Compression Speed (Throughput):\n")
        speed_ranked = library_stats.sort_values('Throughput', ascending=False)
        for i, (lib, row) in enumerate(speed_ranked.iterrows(), 1):
            f.write(f"  {i}. {lib:8s} - {row['Throughput']:6.1f} MB/s\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Best for each data distribution
        f.write("BEST COMPRESSOR BY DATA DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for dist in sorted(df['Distribution'].unique()):
            dist_data = df[df['Distribution'] == dist]
            best_lib = dist_data.groupby('Library')['Compression Ratio'].mean().idxmax()
            best_ratio = dist_data.groupby('Library')['Compression Ratio'].mean().max()
            best_speed_lib = dist_data.groupby('Library')['Compression Throughput (MB/s)'].mean().idxmax()
            best_speed = dist_data.groupby('Library')['Compression Throughput (MB/s)'].mean().max()
            
            f.write(f"{dist.upper()}:\n")
            f.write(f"  Best Ratio: {best_lib} ({best_ratio:.2f}x)\n")
            f.write(f"  Best Speed: {best_speed_lib} ({best_speed:.1f} MB/s)\n\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Speed vs Ratio Trade-off:\n")
        f.write(f"   - Fastest: {speed_ranked.index[0]} ({speed_ranked.iloc[0]['Throughput']:.1f} MB/s)\n")
        f.write(f"   - Best Ratio: {ratio_ranked.index[0]} ({ratio_ranked.iloc[0]['Avg Ratio']:.2f}x)\n")
        f.write(f"   - There is a clear trade-off between speed and compression ratio\n\n")
        
        f.write("2. CPU Utilization:\n")
        cpu_ranked = library_stats.sort_values('CPU %', ascending=False)
        f.write(f"   - Highest CPU: {cpu_ranked.index[0]} ({cpu_ranked.iloc[0]['CPU %']:.1f}%)\n")
        f.write(f"   - Lowest CPU: {cpu_ranked.index[-1]} ({cpu_ranked.iloc[-1]['CPU %']:.1f}%)\n\n")
        
        f.write("3. Scalability:\n")
        # Check which compressors scale best with chunk size
        scalability = {}
        for lib in df['Library'].unique():
            lib_data = df[df['Library'] == lib]
            small = lib_data[lib_data['Chunk Size (bytes)'] == lib_data['Chunk Size (bytes)'].min()]
            large = lib_data[lib_data['Chunk Size (bytes)'] == lib_data['Chunk Size (bytes)'].max()]
            if len(small) > 0 and len(large) > 0:
                ratio_improvement = large['Compression Ratio'].mean() / small['Compression Ratio'].mean()
                scalability[lib] = ratio_improvement
        
        best_scalable = max(scalability.items(), key=lambda x: x[1])
        f.write(f"   - Best scalability: {best_scalable[0]} ({best_scalable[1]:.2f}x improvement at larger sizes)\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"• For real-time applications: Use {speed_ranked.index[0]} (fastest)\n")
        f.write(f"• For storage optimization: Use {ratio_ranked.index[0]} (best compression)\n")
        f.write(f"• For general purpose: Use {best_efficiency} (best balance)\n")
        f.write(f"• For consistent performance: Use {best_consistent}\n")
        f.write(f"• For large files: Use {best_scalable[0]} (best scalability)\n")
        
    print(f"✓ Saved: insights_and_recommendations.txt")

def main():
    """Main function to generate all analysis and plots."""
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'compression_benchmark_results.csv'
    output_dir = script_dir / 'benchmark_plots'
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPRESSION BENCHMARK ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    print()
    
    # Generate plots
    print("Generating analysis plots...")
    plot_compression_trade_off(df, output_dir)
    plot_efficiency_analysis(df, output_dir)
    plot_scalability_analysis(df, output_dir)
    plot_data_type_effectiveness(df, output_dir)
    plot_performance_summary_dashboard(df, output_dir)
    print()
    
    # Generate insights
    print("Generating insights report...")
    generate_insights_report(df, output_dir)
    print()
    
    print("=" * 80)
    print(f"✓ Analysis complete! Outputs saved to: {output_dir}")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • compression_tradeoff.png - Ratio vs speed trade-offs")
    print("  • efficiency_analysis.png - Efficiency metrics and rankings")
    print("  • scalability_analysis.png - Performance scaling with chunk size")
    print("  • data_type_effectiveness.png - Best compressors by data type")
    print("  • performance_dashboard.png - Comprehensive summary dashboard")
    print("  • insights_and_recommendations.txt - Actionable insights and recommendations")

if __name__ == '__main__':
    main()
