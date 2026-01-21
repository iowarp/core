#!/usr/bin/env python3
"""
Batch inference scaling study for compression prediction models.

Measures inference performance (throughput and latency) for different batch sizes
to understand how models perform when processing multiple predictions simultaneously.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def print_header(text, width=80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def load_models():
    """Load both XGBoost and neural network models."""
    print_header("LOADING MODELS")

    models = {}

    # Load XGBoost models
    try:
        models['xgb_ratio'] = joblib.load('model_output/compression_ratio_model.pkl')
        models['xgb_psnr'] = joblib.load('model_output/psnr_model.pkl')
        print("✓ Loaded XGBoost models")
    except FileNotFoundError as e:
        print(f"✗ XGBoost models not found: {e}")
        return None

    # Load neural network models
    try:
        models['nn_ratio'] = tf.keras.models.load_model('cnn_model_output/cnn_compression_ratio_model.keras')
        models['nn_psnr'] = tf.keras.models.load_model('cnn_model_output/cnn_psnr_model.keras')
        models['nn_ratio_scaler'] = joblib.load('cnn_model_output/cnn_compression_ratio_scaler.pkl')
        models['nn_psnr_scaler'] = joblib.load('cnn_model_output/cnn_psnr_scaler.pkl')
        print("✓ Loaded neural network models")
    except FileNotFoundError as e:
        print(f"✗ Neural network models not found: {e}")
        return None

    return models


def prepare_test_data(csv_path, n_samples=1000):
    """Load and prepare test data for inference."""
    print_header("PREPARING TEST DATA")

    # Load data
    df = pd.read_csv(csv_path)
    df = df[df['Success'] == 'YES'].copy()

    # Select features
    feature_cols = [
        'Library',
        'Chunk Size (bytes)',
        'Target CPU Util (%)',
        'Shannon Entropy (bits/byte)',
        'MAD',
        'Second Derivative Mean'
    ]

    # Add Data Type if present (lossy compression data)
    if 'Data Type' in df.columns:
        feature_cols.insert(1, 'Data Type')
        X = df[feature_cols].copy()
        X = pd.get_dummies(X, columns=['Library', 'Data Type'], drop_first=False)
    else:
        # Lossless data - infer data type from distribution name
        X = df[feature_cols].copy()
        # Add synthetic data type based on distribution
        if 'Distribution' in df.columns:
            X['Data Type'] = df['Distribution'].apply(lambda x: 'float' if 'uniform' in x or 'normal' in x else 'char')
        else:
            X['Data Type'] = 'float'  # Default
        X = pd.get_dummies(X, columns=['Library', 'Data Type'], drop_first=False)

    # Ensure all expected columns are present (for model compatibility)
    expected_cols = [
        'Chunk Size (bytes)',
        'Target CPU Util (%)',
        'Shannon Entropy (bits/byte)',
        'MAD',
        'Second Derivative Mean',
        'Library_BZIP2',
        'Library_ZFP_tol_0.010000',
        'Library_ZFP_tol_0.100000',
        'Data Type_char',
        'Data Type_float'
    ]

    # Add missing columns with zeros
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0

    # Reorder columns to match expected order
    X = X[expected_cols]

    # Sample or repeat to get desired number of samples
    if len(X) < n_samples:
        # Repeat data to reach n_samples
        repeats = (n_samples // len(X)) + 1
        X = pd.concat([X] * repeats, ignore_index=True)

    X = X.iloc[:n_samples].copy()

    print(f"✓ Prepared {len(X)} test samples")
    print(f"✓ Feature shape: {X.shape}")

    return X


def benchmark_xgboost(model, X, batch_sizes, n_warmup=10, n_iterations=100):
    """Benchmark XGBoost model with different batch sizes."""
    print("\nBenchmarking XGBoost...")
    results = []

    for batch_size in batch_sizes:
        # Warmup
        for _ in range(n_warmup):
            _ = model.predict(X.iloc[:batch_size])

        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = model.predict(X.iloc[:batch_size])
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        throughput = batch_size / (avg_time / 1000)  # predictions per second
        latency_per_sample = avg_time / batch_size  # ms per sample

        results.append({
            'batch_size': batch_size,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'throughput_per_sec': throughput,
            'latency_per_sample_ms': latency_per_sample
        })

        print(f"  Batch {batch_size:>4}: {avg_time:>7.3f} ± {std_time:>6.3f} ms | "
              f"{throughput:>8.1f} pred/s | {latency_per_sample:>6.3f} ms/sample")

    return pd.DataFrame(results)


def benchmark_neural_network(model, scaler, X, batch_sizes, n_warmup=10, n_iterations=100):
    """Benchmark neural network model with different batch sizes."""
    print("\nBenchmarking Neural Network...")
    results = []

    # Scale all data once
    X_scaled = scaler.transform(X)

    for batch_size in batch_sizes:
        # Warmup
        for _ in range(n_warmup):
            _ = model.predict(X_scaled[:batch_size], verbose=0)

        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = model.predict(X_scaled[:batch_size], verbose=0)
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        throughput = batch_size / (avg_time / 1000)  # predictions per second
        latency_per_sample = avg_time / batch_size  # ms per sample

        results.append({
            'batch_size': batch_size,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'throughput_per_sec': throughput,
            'latency_per_sample_ms': latency_per_sample
        })

        print(f"  Batch {batch_size:>4}: {avg_time:>7.3f} ± {std_time:>6.3f} ms | "
              f"{throughput:>8.1f} pred/s | {latency_per_sample:>6.3f} ms/sample")

    return pd.DataFrame(results)


def create_visualizations(xgb_results, nn_results, output_dir):
    """Create visualization plots for batch inference results."""
    print_header("GENERATING VISUALIZATIONS")

    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Total inference time vs batch size
    ax = axes[0, 0]
    ax.plot(xgb_results['batch_size'], xgb_results['avg_time_ms'],
            marker='o', label='XGBoost', linewidth=2)
    ax.plot(nn_results['batch_size'], nn_results['avg_time_ms'],
            marker='s', label='Neural Network', linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Total Inference Time (ms)', fontsize=12)
    ax.set_title('Total Inference Time vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # 2. Throughput vs batch size
    ax = axes[0, 1]
    ax.plot(xgb_results['batch_size'], xgb_results['throughput_per_sec'],
            marker='o', label='XGBoost', linewidth=2)
    ax.plot(nn_results['batch_size'], nn_results['throughput_per_sec'],
            marker='s', label='Neural Network', linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput (predictions/sec)', fontsize=12)
    ax.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # 3. Latency per sample vs batch size
    ax = axes[1, 0]
    ax.plot(xgb_results['batch_size'], xgb_results['latency_per_sample_ms'],
            marker='o', label='XGBoost', linewidth=2)
    ax.plot(nn_results['batch_size'], nn_results['latency_per_sample_ms'],
            marker='s', label='Neural Network', linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Latency per Sample (ms)', fontsize=12)
    ax.set_title('Latency per Sample vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # 4. Speedup comparison
    ax = axes[1, 1]
    batch_16_xgb = xgb_results[xgb_results['batch_size'] == 16]['throughput_per_sec'].values[0]
    batch_16_nn = nn_results[nn_results['batch_size'] == 16]['throughput_per_sec'].values[0]

    speedup_xgb = xgb_results['throughput_per_sec'] / xgb_results[xgb_results['batch_size'] == 1]['throughput_per_sec'].values[0]
    speedup_nn = nn_results['throughput_per_sec'] / nn_results[nn_results['batch_size'] == 1]['throughput_per_sec'].values[0]

    ax.plot(xgb_results['batch_size'], speedup_xgb,
            marker='o', label='XGBoost', linewidth=2)
    ax.plot(nn_results['batch_size'], speedup_nn,
            marker='s', label='Neural Network', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Speedup vs Batch Size 1', fontsize=12)
    ax.set_title('Throughput Speedup (vs Single Prediction)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'batch_inference_scaling.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary(xgb_results, nn_results):
    """Print summary statistics."""
    print_header("BATCH INFERENCE SUMMARY")

    # Find optimal batch size for throughput
    xgb_best_idx = xgb_results['throughput_per_sec'].idxmax()
    nn_best_idx = nn_results['throughput_per_sec'].idxmax()

    print("XGBoost Performance:")
    print("-" * 80)
    print(f"  Best Batch Size: {xgb_results.loc[xgb_best_idx, 'batch_size']}")
    print(f"  Max Throughput: {xgb_results.loc[xgb_best_idx, 'throughput_per_sec']:.1f} predictions/sec")
    print(f"  Latency at Best Batch: {xgb_results.loc[xgb_best_idx, 'latency_per_sample_ms']:.4f} ms/sample")
    print(f"  Total Time at Best Batch: {xgb_results.loc[xgb_best_idx, 'avg_time_ms']:.3f} ms")

    print("\nNeural Network Performance:")
    print("-" * 80)
    print(f"  Best Batch Size: {nn_results.loc[nn_best_idx, 'batch_size']}")
    print(f"  Max Throughput: {nn_results.loc[nn_best_idx, 'throughput_per_sec']:.1f} predictions/sec")
    print(f"  Latency at Best Batch: {nn_results.loc[nn_best_idx, 'latency_per_sample_ms']:.4f} ms/sample")
    print(f"  Total Time at Best Batch: {nn_results.loc[nn_best_idx, 'avg_time_ms']:.3f} ms")

    # Batch size 16 comparison
    print("\nBatch Size 16 Comparison (Suggested for Real-time Use):")
    print("-" * 80)
    xgb_16 = xgb_results[xgb_results['batch_size'] == 16].iloc[0]
    nn_16 = nn_results[nn_results['batch_size'] == 16].iloc[0]

    print(f"  XGBoost:")
    print(f"    Throughput: {xgb_16['throughput_per_sec']:.1f} predictions/sec")
    print(f"    Latency: {xgb_16['latency_per_sample_ms']:.4f} ms/sample")
    print(f"    Total Time: {xgb_16['avg_time_ms']:.3f} ms")

    print(f"\n  Neural Network:")
    print(f"    Throughput: {nn_16['throughput_per_sec']:.1f} predictions/sec")
    print(f"    Latency: {nn_16['latency_per_sample_ms']:.4f} ms/sample")
    print(f"    Total Time: {nn_16['avg_time_ms']:.3f} ms")

    speedup = xgb_16['throughput_per_sec'] / nn_16['throughput_per_sec']
    print(f"\n  XGBoost is {speedup:.1f}x faster than Neural Network at batch size 16")

    # Scalability analysis
    print("\nScalability Analysis:")
    print("-" * 80)
    xgb_1 = xgb_results[xgb_results['batch_size'] == 1].iloc[0]
    nn_1 = nn_results[nn_results['batch_size'] == 1].iloc[0]

    xgb_speedup_16 = xgb_16['throughput_per_sec'] / xgb_1['throughput_per_sec']
    nn_speedup_16 = nn_16['throughput_per_sec'] / nn_1['throughput_per_sec']

    print(f"  XGBoost batch 16 vs batch 1: {xgb_speedup_16:.2f}x throughput improvement")
    print(f"  Neural Network batch 16 vs batch 1: {nn_speedup_16:.2f}x throughput improvement")

    if nn_speedup_16 > xgb_speedup_16:
        print(f"\n  ✓ Neural Network benefits more from batching ({nn_speedup_16:.2f}x vs {xgb_speedup_16:.2f}x)")
    else:
        print(f"\n  ✓ XGBoost benefits more from batching ({xgb_speedup_16:.2f}x vs {nn_speedup_16:.2f}x)")


def main():
    if len(sys.argv) != 2:
        print("\nUsage: python batch_inference_study.py <compression_csv>")
        print("\nExample:")
        print("  python batch_inference_study.py \\")
        print("    /workspace/build/compression_parameter_study_results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = 'batch_inference_results'

    print_header("BATCH INFERENCE SCALING STUDY")
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")

    # Load models
    models = load_models()
    if models is None:
        print("\n✗ Failed to load models. Exiting.")
        sys.exit(1)

    # Prepare test data
    X = prepare_test_data(csv_path, n_samples=1000)

    # Define batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print_header("RUNNING BENCHMARKS")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup iterations: 10")
    print(f"Benchmark iterations: 100")

    # Benchmark XGBoost
    print_header("XGBOOST COMPRESSION RATIO")
    xgb_results = benchmark_xgboost(models['xgb_ratio'], X, batch_sizes)

    # Benchmark Neural Network
    print_header("NEURAL NETWORK COMPRESSION RATIO")
    nn_results = benchmark_neural_network(
        models['nn_ratio'],
        models['nn_ratio_scaler'],
        X,
        batch_sizes
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    xgb_results.to_csv(os.path.join(output_dir, 'xgboost_batch_results.csv'), index=False)
    nn_results.to_csv(os.path.join(output_dir, 'neural_network_batch_results.csv'), index=False)
    print(f"\n✓ Saved results to {output_dir}/")

    # Create visualizations
    create_visualizations(xgb_results, nn_results, output_dir)

    # Print summary
    print_summary(xgb_results, nn_results)

    print_header("BATCH INFERENCE STUDY COMPLETE")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - xgboost_batch_results.csv")
    print(f"  - neural_network_batch_results.csv")
    print(f"  - batch_inference_scaling.pdf")


if __name__ == '__main__':
    main()
