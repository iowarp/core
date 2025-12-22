#!/usr/bin/env python3
"""
Best Compressor Selection Table Generator

Generates a comprehensive table showing the best compression library for each
(distribution, data_type, chunk_size) combination across different metrics:
- Best Compression Ratio
- Best Compress Speed (lowest time)
- Best Decompress Speed (lowest time)
- Best Compress CPU Utilization (lowest %)
- Best Decompress CPU Utilization (lowest %)
"""

import pandas as pd
import sys
from pathlib import Path

def load_data(csv_path):
    """Load and preprocess benchmark data."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} benchmark records")

        # Filter out invalid data
        df = df[df['Compress Time (ms)'] > 0]
        df = df[df['Decompress Time (ms)'] > 0]
        df = df[df['Compression Ratio'] > 0]

        print(f"✓ Processed {len(df)} valid records")
        return df
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        sys.exit(1)

def format_chunk_size(size_bytes):
    """Format chunk size in human-readable format."""
    if size_bytes < 1024**2:
        return f"{size_bytes/1024:.0f} KB"
    else:
        return f"{size_bytes/(1024**2):.1f} MB"

def find_best_compressors(df):
    """
    Find the best compression library for each (distribution, data_type, chunk_size)
    combination across different metrics.

    Returns:
        DataFrame with best compressor for each combination and metric
    """
    # Group by distribution, data type, and chunk size
    results = []

    for (dist, dtype, chunk_size), group in df.groupby(['Distribution', 'Data Type', 'Chunk Size (bytes)']):
        # Find best library for each metric
        best_ratio = group.loc[group['Compression Ratio'].idxmax(), 'Library']
        best_compress_speed = group.loc[group['Compress Time (ms)'].idxmin(), 'Library']
        best_decompress_speed = group.loc[group['Decompress Time (ms)'].idxmin(), 'Library']
        best_compress_cpu = group.loc[group['Compress CPU %'].idxmin(), 'Library']
        best_decompress_cpu = group.loc[group['Decompress CPU %'].idxmin(), 'Library']

        # Get actual values for reference
        max_ratio = group['Compression Ratio'].max()
        min_compress_time = group['Compress Time (ms)'].min()
        min_decompress_time = group['Decompress Time (ms)'].min()
        min_compress_cpu = group['Compress CPU %'].min()
        min_decompress_cpu = group['Decompress CPU %'].min()

        results.append({
            'Distribution': dist,
            'Data Type': dtype,
            'Chunk Size': format_chunk_size(chunk_size),
            'Chunk Size (bytes)': chunk_size,
            'Best Ratio': best_ratio,
            'Best Ratio Value': f"{max_ratio:.2f}x",
            'Best Compress Speed': best_compress_speed,
            'Best Compress Speed Value': f"{min_compress_time:.2f} ms",
            'Best Decompress Speed': best_decompress_speed,
            'Best Decompress Speed Value': f"{min_decompress_time:.2f} ms",
            'Best Compress CPU': best_compress_cpu,
            'Best Compress CPU Value': f"{min_compress_cpu:.1f}%",
            'Best Decompress CPU': best_decompress_cpu,
            'Best Decompress CPU Value': f"{min_decompress_cpu:.1f}%"
        })

    return pd.DataFrame(results)

def generate_summary_table(best_df, output_dir):
    """Generate a clean summary table (library names only)."""
    summary_path = output_dir / 'best_compressor_by_category.txt'

    # Create clean table with just library names (keep Chunk Size (bytes) for sorting)
    clean_df = best_df[['Distribution', 'Data Type', 'Chunk Size', 'Chunk Size (bytes)',
                         'Best Ratio', 'Best Compress Speed', 'Best Decompress Speed',
                         'Best Compress CPU', 'Best Decompress CPU']].copy()

    with open(summary_path, 'w') as f:
        f.write("=" * 160 + "\n")
        f.write("BEST COMPRESSION LIBRARY BY CATEGORY\n")
        f.write("=" * 160 + "\n\n")
        f.write("This table shows which compression library performs best for each (Distribution, Data Type, Chunk Size)\n")
        f.write("combination across different performance metrics.\n\n")
        f.write("Metrics:\n")
        f.write("  • Best Ratio: Highest compression ratio (smaller output size)\n")
        f.write("  • Best Compress Speed: Lowest compression time (fastest)\n")
        f.write("  • Best Decompress Speed: Lowest decompression time (fastest)\n")
        f.write("  • Best Compress CPU: Lowest CPU utilization during compression\n")
        f.write("  • Best Decompress CPU: Lowest CPU utilization during decompression\n")
        f.write("\n" + "=" * 160 + "\n\n")

        # Sort by distribution, data type, chunk size
        clean_df = clean_df.sort_values(['Distribution', 'Data Type', 'Chunk Size (bytes)'])

        # Write table header
        f.write(f"{'Distribution':<15} {'Data Type':<12} {'Chunk Size':<12} {'Best Ratio':<12} "
                f"{'Best Compress':<15} {'Best Decompress':<15} {'Best Compress':<15} {'Best Decompress':<15}\n")
        f.write(f"{'':15} {'':12} {'':12} {'':12} "
                f"{'Speed':<15} {'Speed':<15} {'CPU Util':<15} {'CPU Util':<15}\n")
        f.write("-" * 160 + "\n")

        # Write data rows
        for _, row in clean_df.iterrows():
            f.write(f"{row['Distribution']:<15} {row['Data Type']:<12} {row['Chunk Size']:<12} "
                    f"{row['Best Ratio']:<12} {row['Best Compress Speed']:<15} "
                    f"{row['Best Decompress Speed']:<15} {row['Best Compress CPU']:<15} "
                    f"{row['Best Decompress CPU']:<15}\n")

        f.write("\n" + "=" * 160 + "\n")

    print(f"✓ Saved: {summary_path.name}")

def generate_detailed_table(best_df, output_dir):
    """Generate a detailed table with actual values."""
    detailed_path = output_dir / 'best_compressor_with_values.txt'

    with open(detailed_path, 'w') as f:
        f.write("=" * 180 + "\n")
        f.write("BEST COMPRESSION LIBRARY BY CATEGORY (WITH VALUES)\n")
        f.write("=" * 180 + "\n\n")

        # Sort by distribution, data type, chunk size
        sorted_df = best_df.sort_values(['Distribution', 'Data Type', 'Chunk Size (bytes)'])

        current_dist = None
        for _, row in sorted_df.iterrows():
            # Print distribution header
            if row['Distribution'] != current_dist:
                if current_dist is not None:
                    f.write("\n")
                f.write("=" * 180 + "\n")
                f.write(f"DISTRIBUTION: {row['Distribution'].upper()}\n")
                f.write("=" * 180 + "\n\n")
                current_dist = row['Distribution']

            f.write(f"Data Type: {row['Data Type']:<8}  |  Chunk Size: {row['Chunk Size']:<10}\n")
            f.write("-" * 180 + "\n")
            f.write(f"  Best Compression Ratio:       {row['Best Ratio']:<12} ({row['Best Ratio Value']})\n")
            f.write(f"  Best Compress Speed:          {row['Best Compress Speed']:<12} ({row['Best Compress Speed Value']})\n")
            f.write(f"  Best Decompress Speed:        {row['Best Decompress Speed']:<12} ({row['Best Decompress Speed Value']})\n")
            f.write(f"  Best Compress CPU Util:       {row['Best Compress CPU']:<12} ({row['Best Compress CPU Value']})\n")
            f.write(f"  Best Decompress CPU Util:     {row['Best Decompress CPU']:<12} ({row['Best Decompress CPU Value']})\n")
            f.write("\n")

    print(f"✓ Saved: {detailed_path.name}")

def generate_csv_table(best_df, output_dir):
    """Generate CSV version of the table for easy import into spreadsheets."""
    csv_path = output_dir / 'best_compressor_table.csv'

    # Create CSV with both library names and values
    csv_df = best_df[['Distribution', 'Data Type', 'Chunk Size', 'Chunk Size (bytes)',
                      'Best Ratio', 'Best Ratio Value',
                      'Best Compress Speed', 'Best Compress Speed Value',
                      'Best Decompress Speed', 'Best Decompress Speed Value',
                      'Best Compress CPU', 'Best Compress CPU Value',
                      'Best Decompress CPU', 'Best Decompress CPU Value']].copy()

    csv_df = csv_df.sort_values(['Distribution', 'Data Type', 'Chunk Size (bytes)'])
    csv_df = csv_df.drop(columns=['Chunk Size (bytes)'])

    csv_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path.name}")

def generate_library_frequency_analysis(best_df, output_dir):
    """Analyze which libraries appear most frequently as 'best' across categories."""
    freq_path = output_dir / 'best_compressor_frequency.txt'

    with open(freq_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPRESSION LIBRARY FREQUENCY ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        f.write("This analysis shows how often each library is the 'best' choice across all\n")
        f.write("combinations of distribution, data type, and chunk size.\n\n")

        # Count frequency for each metric
        metrics = {
            'Best Ratio': 'Best for Compression Ratio',
            'Best Compress Speed': 'Best for Compress Speed',
            'Best Decompress Speed': 'Best for Decompress Speed',
            'Best Compress CPU': 'Best for Compress CPU Utilization',
            'Best Decompress CPU': 'Best for Decompress CPU Utilization'
        }

        total_combinations = len(best_df)

        for metric, title in metrics.items():
            f.write("=" * 100 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 100 + "\n")

            counts = best_df[metric].value_counts().sort_values(ascending=False)

            for lib, count in counts.items():
                percentage = (count / total_combinations) * 100
                f.write(f"  {lib:<12} : {count:3d} times ({percentage:5.1f}%)\n")

            f.write("\n")

        # Overall "best" frequency (sum across all metrics)
        f.write("=" * 100 + "\n")
        f.write("OVERALL FREQUENCY (All Metrics Combined)\n")
        f.write("=" * 100 + "\n")

        all_best = pd.concat([
            best_df['Best Ratio'],
            best_df['Best Compress Speed'],
            best_df['Best Decompress Speed'],
            best_df['Best Compress CPU'],
            best_df['Best Decompress CPU']
        ])

        overall_counts = all_best.value_counts().sort_values(ascending=False)
        total_selections = len(all_best)

        for lib, count in overall_counts.items():
            percentage = (count / total_selections) * 100
            f.write(f"  {lib:<12} : {count:4d} times ({percentage:5.1f}%)\n")

        f.write("\n")

    print(f"✓ Saved: {freq_path.name}")

def main():
    """Main function to generate all tables and analyses."""
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'compression_benchmark_results.csv'
    output_dir = script_dir / 'benchmark_plots'

    output_dir.mkdir(exist_ok=True)

    print("=" * 100)
    print("BEST COMPRESSOR TABLE GENERATOR")
    print("=" * 100)
    print()

    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    print()

    # Find best compressors
    print("Analyzing best compressors for each category...")
    best_df = find_best_compressors(df)
    print(f"✓ Analyzed {len(best_df)} unique (distribution, data_type, chunk_size) combinations")
    print()

    # Generate outputs
    print("Generating tables and analyses...")
    generate_summary_table(best_df, output_dir)
    generate_detailed_table(best_df, output_dir)
    generate_csv_table(best_df, output_dir)
    generate_library_frequency_analysis(best_df, output_dir)
    print()

    print("=" * 100)
    print("✓ Analysis complete!")
    print("=" * 100)
    print("\nGenerated files:")
    print("  • best_compressor_by_category.txt")
    print("    → Clean table showing best library for each combination")
    print("  • best_compressor_with_values.txt")
    print("    → Detailed table with actual performance values")
    print("  • best_compressor_table.csv")
    print("    → CSV format for spreadsheet import")
    print("  • best_compressor_frequency.txt")
    print("    → Frequency analysis showing which libraries win most often")

if __name__ == '__main__':
    main()
