#!/usr/bin/env python3
"""
Model comparison and paper figure generation.

Generates all figures for the paper:
1. R² comparison across models
2. Inference speed comparison
3. Feature importance analysis
4. Gray-Scott adaptiveness evaluation (MAPE and Kendall's tau)

Usage:
    python compare_models.py                    # Generate all figures
    python compare_models.py --gray-scott      # Run Gray-Scott evaluation
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

PAPER_DIR = 'paper'
os.makedirs(PAPER_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3
})

TARGET_COLS = ['compression_ratio', 'compress_time_ms', 'decompress_time_ms', 'psnr_db']
TARGET_LABELS = ['Compression\nRatio', 'Compress\nTime', 'Decompress\nTime', 'PSNR']

# ============================================================================
# Figure Generation Functions
# ============================================================================

def generate_r2_comparison(results_file='model_comparison_results.json'):
    """Generate R² comparison figure across models and targets."""
    print("\nGenerating R² comparison figure...")

    with open(results_file, 'r') as f:
        results = json.load(f)

    models = results.get('models', results)  # Handle both formats

    # Extract R² values
    model_names = []
    r2_data = {}

    if 'xgboost' in models:
        model_names.append('XGBoost')
        r2_data['XGBoost'] = [models['xgboost']['per_target_r2'][t] for t in TARGET_COLS]

    if 'qtable' in models:
        model_names.append('Q-Table')
        r2_data['Q-Table'] = [models['qtable']['per_target_r2'][t] for t in TARGET_COLS]

    if 'dnn' in models:
        model_names.append('DNN')
        r2_data['DNN'] = [models['dnn']['per_target_r2'][t] for t in TARGET_COLS]

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(TARGET_COLS))
    n_models = len(model_names)
    width = 0.8 / n_models
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:n_models]

    for i, (model, scores) in enumerate(r2_data.items()):
        offset = width * (i - n_models / 2 + 0.5)
        bars = ax.bar(x + offset, scores, width, label=model,
                     color=colors[i], edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('R² Score')
    ax.set_xlabel('Prediction Target')
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    for fmt in ['svg', 'pdf']:
        plt.savefig(f'{PAPER_DIR}/r2_comparison.{fmt}', format=fmt, dpi=150, bbox_inches='tight')
    print(f"  Saved: {PAPER_DIR}/r2_comparison.svg/pdf")

    plt.close()


def generate_inference_speed():
    """Generate inference speed comparison figure."""
    print("\nGenerating inference speed figure...")

    # Inference times in microseconds
    inference_times = {
        'XGBoost': 435,
        'Q-Table (C++)': 1.6,
        'Q-Table (Python)': 27,
        'DNN': 150
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    implementations = list(inference_times.keys())
    times = list(inference_times.values())
    colors = ['#2ecc71', '#3498db', '#5dade2', '#e74c3c']

    bars = ax.barh(implementations, times, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, time in zip(bars, times):
        width = bar.get_width()
        label = f'{time:.1f} μs' if time < 100 else f'{time:.0f} μs'
        ax.annotate(label,
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=10)

    ax.set_xlabel('Inference Time (μs) - Log Scale')
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)

    # Speedup annotation
    ax.annotate('272× faster', xy=(1.6, 1), xytext=(10, 1),
               fontsize=9, color='#2980b9', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5))

    plt.tight_layout()

    for fmt in ['svg', 'pdf']:
        plt.savefig(f'{PAPER_DIR}/inference_speed.{fmt}', format=fmt, dpi=150, bbox_inches='tight')
    print(f"  Saved: {PAPER_DIR}/inference_speed.svg/pdf")

    plt.close()


def generate_feature_importance(data_file='compression_benchmark_results.csv'):
    """Generate feature importance figure using XGBoost."""
    print("\nGenerating feature importance figure...")

    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("  XGBoost not available, skipping feature importance")
        return

    # Load data
    df = pd.read_csv(data_file)

    # Encode categorical features
    le_library = LabelEncoder()
    le_config = LabelEncoder()
    le_datatype = LabelEncoder()

    df['library_id'] = le_library.fit_transform(df['library_name'])
    df['config_id'] = le_config.fit_transform(df['configuration'])
    df['datatype_id'] = le_datatype.fit_transform(df['data_type'])

    feature_cols = ['library_id', 'config_id', 'datatype_id',
                   'data_size', 'shannon_entropy', 'mean_absolute_deviation']
    if 'second_order_derivative' in df.columns:
        feature_cols.append('second_order_derivative')

    feature_labels = ['Library', 'Config', 'Data Type', 'Size',
                     'Entropy', 'MAD', '2nd Derivative'][:len(feature_cols)]

    X = df[feature_cols].values
    y = df['compression_ratio'].values  # Use compression ratio as target

    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)

    # Get feature importance
    importance = model.feature_importances_

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(feature_labels))
    colors = plt.cm.viridis(importance / max(importance))

    bars = ax.barh(y_pos, importance, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Compression Ratio Prediction')

    # Add value labels
    for bar, imp in zip(bars, importance):
        width = bar.get_width()
        ax.annotate(f'{imp:.1%}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=9)

    plt.tight_layout()

    for fmt in ['svg', 'pdf']:
        plt.savefig(f'{PAPER_DIR}/feature_importance.{fmt}', format=fmt, dpi=150, bbox_inches='tight')
    print(f"  Saved: {PAPER_DIR}/feature_importance.svg/pdf")

    # Save results
    results = {feature: float(imp) for feature, imp in zip(feature_labels, importance)}
    with open('feature_importance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: feature_importance_results.json")

    plt.close()


def generate_gray_scott_figures(metrics_file='gray_scott_rl_metrics.csv'):
    """Generate Gray-Scott evaluation figures (MAPE and Kendall's tau)."""
    print("\nGenerating Gray-Scott evaluation figures...")

    if not os.path.exists(metrics_file):
        print(f"  {metrics_file} not found, skipping Gray-Scott figures")
        return

    df = pd.read_csv(metrics_file)

    # Figure 1: MAPE over iterations
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['iteration'], df['mape_ratio'], 'o-', label='Compression Ratio', linewidth=2, markersize=4)
    ax.plot(df['iteration'], df['mape_compress_time'], 's-', label='Compress Time', linewidth=2, markersize=4)
    ax.plot(df['iteration'], df['mape_decompress_time'], '^-', label='Decompress Time', linewidth=2, markersize=4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Mean Absolute Percentage Error Over Time')
    ax.legend()
    ax.set_ylim(0, 150)  # Cap at 150%

    # Annotate initial high value if needed
    if df['mape_ratio'].iloc[0] > 150:
        ax.annotate(f'{df["mape_ratio"].iloc[0]:.0f}%',
                   xy=(1, 150), xytext=(3, 140),
                   fontsize=9, color='#3498db')

    plt.tight_layout()

    for fmt in ['svg', 'pdf']:
        plt.savefig(f'{PAPER_DIR}/gray_scott_mape.{fmt}', format=fmt, dpi=150, bbox_inches='tight')
    print(f"  Saved: {PAPER_DIR}/gray_scott_mape.svg/pdf")
    plt.close()

    # Figure 2: Kendall's tau over iterations
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['iteration'], df['tau_ratio'], 'o-', label='Compression Ratio', linewidth=2, markersize=4)
    ax.plot(df['iteration'], df['tau_compress_time'], 's-', label='Compress Time', linewidth=2, markersize=4)
    ax.plot(df['iteration'], df['tau_decompress_time'], '^-', label='Decompress Time', linewidth=2, markersize=4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel("Kendall's τ")
    ax.set_title('Ranking Correlation Over Time')
    ax.legend()
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()

    for fmt in ['svg', 'pdf']:
        plt.savefig(f'{PAPER_DIR}/gray_scott_kendall_tau.{fmt}', format=fmt, dpi=150, bbox_inches='tight')
    print(f"  Saved: {PAPER_DIR}/gray_scott_kendall_tau.svg/pdf")
    plt.close()


def print_summary(results_file='model_comparison_results.json'):
    """Print summary of model comparison results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    with open(results_file, 'r') as f:
        results = json.load(f)

    models = results.get('models', results)

    print(f"\n{'Model':<15} {'Overall R²':>12} {'Training Time':>15}")
    print("-" * 45)

    for model_name, model_data in models.items():
        r2 = model_data.get('overall_r2', 0)
        time_s = model_data.get('training_time_s', 0)
        print(f"{model_name:<15} {r2:>12.4f} {time_s:>12.1f}s")

    print("\nPer-target R² scores:")
    print(f"{'Target':<25}", end='')
    for model_name in models:
        print(f"{model_name:>15}", end='')
    print()
    print("-" * (25 + 15 * len(models)))

    for target in TARGET_COLS:
        print(f"{target:<25}", end='')
        for model_name, model_data in models.items():
            r2 = model_data.get('per_target_r2', {}).get(target, 0)
            print(f"{r2:>15.4f}", end='')
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--gray-scott', action='store_true',
                       help='Generate Gray-Scott evaluation figures')
    parser.add_argument('--results', default='model_comparison_results.json',
                       help='Path to model comparison results JSON')
    parser.add_argument('--data', default='compression_benchmark_results.csv',
                       help='Path to benchmark data CSV')
    args = parser.parse_args()

    print("=" * 60)
    print("PAPER FIGURE GENERATION")
    print("=" * 60)

    # Generate main comparison figures
    if os.path.exists(args.results):
        generate_r2_comparison(args.results)
        print_summary(args.results)
    else:
        print(f"\nWarning: {args.results} not found")
        print("Run train_models.py first to generate model comparison results")

    generate_inference_speed()

    if os.path.exists(args.data):
        generate_feature_importance(args.data)
    else:
        print(f"\nWarning: {args.data} not found, skipping feature importance")

    # Gray-Scott figures
    if args.gray_scott or os.path.exists('gray_scott_rl_metrics.csv'):
        generate_gray_scott_figures()

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {PAPER_DIR}/")


if __name__ == '__main__':
    main()
