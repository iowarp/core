#!/usr/bin/env python3
"""
Analyze compression speeds from benchmark results.
Filter for 2MB chunks with exponential distribution.
"""

import pandas as pd
import numpy as np

# Read the CSV file - try the no_exp version which may have more libraries
csv_path = '/workspace/context-transport-primitives/test/unit/compress/results/compression_benchmark_results_no_exp.csv'
df = pd.read_csv(csv_path)

print(f"Total rows: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")

# Check available libraries and their data sizes
print("\nLibraries and their data size ranges:")
for lib in df['library_name'].unique():
    lib_data = df[df['library_name'] == lib]
    min_size = lib_data['data_size'].min()
    max_size = lib_data['data_size'].max()
    print(f"  {lib}: {min_size:,} - {max_size:,} bytes ({min_size/1024/1024:.2f} - {max_size/1024/1024:.2f} MB)")

# 2MB = 2,097,152 bytes
TARGET_SIZE_BYTES = 2 * 1024 * 1024
TOLERANCE = 0.05  # 5% tolerance

# Filter for 2MB chunks (within tolerance)
df_2mb = df[(df['data_size'] >= TARGET_SIZE_BYTES * (1 - TOLERANCE)) &
            (df['data_size'] <= TARGET_SIZE_BYTES * (1 + TOLERANCE))]

print(f"\nRows with ~2MB data size: {len(df_2mb)}")

print("\nAvailable libraries in 2MB data:")
print(df_2mb['library_name'].unique())

print("\nAvailable distributions in 2MB data:")
print(df_2mb['distribution_name'].unique()[:20])  # Show first 20

# Filter for exponential distribution
df_exp = df_2mb[df_2mb['distribution_name'].str.contains('exponential', case=False, na=False)]

print(f"\nRows with exponential distribution: {len(df_exp)}")

if len(df_exp) == 0:
    print("\nNo exponential distribution found. Checking available distributions...")
    print(df_2mb['distribution_name'].unique())
else:
    # Calculate compression speed (MB/s)
    df_exp['compress_speed_mbs'] = (df_exp['data_size'] / (1024 * 1024)) / (df_exp['compress_time_ms'] / 1000)

    # Group by library and calculate averages
    results = df_exp.groupby('library_name').agg({
        'compression_ratio': 'mean',
        'compress_speed_mbs': 'mean',
        'compress_time_ms': 'mean',
        'data_size': 'count'  # Number of samples
    }).round(2)

    results.columns = ['Avg Compression Ratio', 'Avg Speed (MB/s)', 'Avg Time (ms)', 'Num Samples']

    print("\n" + "="*80)
    print("COMPRESSION PERFORMANCE FOR 2MB CHUNKS WITH EXPONENTIAL DISTRIBUTION")
    print("="*80)

    # Sort by speed and print each row
    results_sorted = results.sort_values('Avg Speed (MB/s)', ascending=False)

    for library, row in results_sorted.iterrows():
        print(f"\n{library}:")
        print(f"  Avg Compression Ratio: {row['Avg Compression Ratio']:.2f}")
        print(f"  Avg Speed: {row['Avg Speed (MB/s)']:.2f} MB/s")
        print(f"  Avg Time: {row['Avg Time (ms)']:.2f} ms")
        print(f"  Samples: {int(row['Num Samples'])}")

    print("\n" + "="*80)
