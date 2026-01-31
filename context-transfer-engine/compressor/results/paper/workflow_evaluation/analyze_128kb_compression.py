#!/usr/bin/env python3
"""
Analyze compression speeds from benchmark results.
Filter for 128KB chunks with exponential distribution.
"""

import pandas as pd
import numpy as np

# Read the CSV file
csv_path = '/workspace/context-transport-primitives/test/unit/compress/results/compression_benchmark_results_no_exp.csv'
df = pd.read_csv(csv_path)

print(f"Total rows: {len(df)}")

# 128KB = 131,072 bytes
TARGET_SIZE_BYTES = 128 * 1024
TOLERANCE = 0.05  # 5% tolerance

# Filter for 128KB chunks (within tolerance)
df_128kb = df[(df['data_size'] >= TARGET_SIZE_BYTES * (1 - TOLERANCE)) &
              (df['data_size'] <= TARGET_SIZE_BYTES * (1 + TOLERANCE))]

print(f"Rows with ~128KB data size: {len(df_128kb)}")

# Filter for gamma distribution
df_gamma = df_128kb[df_128kb['distribution_name'].str.contains('gamma', case=False, na=False)]

print(f"Rows with gamma distribution: {len(df_gamma)}")

if len(df_gamma) == 0:
    print("\nNo gamma distribution found. Checking available distributions...")
    unique_dists = df_128kb['distribution_name'].unique()
    print(f"Found {len(unique_dists)} unique distributions")
    print("\nSample distributions:", unique_dists[:10])

    # Try using all data at 128KB instead
    print("\n" + "="*80)
    print("Using ALL distributions at 128KB (no gamma filter)")
    print("="*80)
    df_gamma = df_128kb
else:
    print("\n" + "="*80)
    print("GAMMA DISTRIBUTION DATA FOUND")
    print("="*80)

# Calculate compression speed (MB/s)
df_gamma['compress_speed_mbs'] = (df_gamma['data_size'] / (1024 * 1024)) / (df_gamma['compress_time_ms'] / 1000)

# Group by library and calculate averages
results = df_gamma.groupby('library_name').agg({
    'compression_ratio': ['mean', 'std'],
    'compress_speed_mbs': ['mean', 'std'],
    'compress_time_ms': 'mean',
    'data_size': 'count'  # Number of samples
}).round(2)

# Flatten column names
results.columns = ['_'.join(col).strip() for col in results.columns.values]
results.columns = ['Ratio Mean', 'Ratio Std', 'Speed (MB/s)', 'Speed Std', 'Time (ms)', 'Samples']

print("\n" + "="*80)
print("COMPRESSION PERFORMANCE FOR 128KB CHUNKS")
print("="*80)
print()

# Sort by speed and print table
results_sorted = results.sort_values('Speed (MB/s)', ascending=False)

print(f"{'Library':<15} {'Speed (MB/s)':<15} {'Ratio':<15} {'Time (ms)':<12} {'Samples':<10}")
print("-" * 80)

for library, row in results_sorted.iterrows():
    print(f"{library:<15} {row['Speed (MB/s)']:>8.2f} ± {row['Speed Std']:<4.2f} "
          f"{row['Ratio Mean']:>6.2f} ± {row['Ratio Std']:<4.2f} "
          f"{row['Time (ms)']:>8.2f}    {int(row['Samples']):>6}")

print("\n" + "="*80)
print("\nNotes:")
print("- Speed = throughput in MB/s (higher is better)")
print("- Ratio = compression ratio (lower is better for compression)")
print("- Time = average compression time per 128KB chunk")
print("="*80)
