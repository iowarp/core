#!/usr/bin/env python3
"""
CNN vs XGBoost Performance Comparison

This script loads both CNN and XGBoost models and provides a detailed
performance comparison for compression prediction tasks.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

# Try importing both model types
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("WARNING: TensorFlow not available")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not available")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def load_models(model_dir, cnn_dir):
    """Load both CNN and XGBoost models."""
    print_header("LOADING MODELS")

    models = {}

    # Load XGBoost models
    if HAS_XGBOOST:
        xgb_ratio_path = model_dir / 'compression_ratio_model.pkl'
        xgb_psnr_path = model_dir / 'psnr_model.pkl'

        if xgb_ratio_path.exists():
            models['xgb_ratio'] = joblib.load(xgb_ratio_path)
            print(f"✓ Loaded XGBoost compression ratio model")
        else:
            print(f"⚠️  XGBoost ratio model not found: {xgb_ratio_path}")

        if xgb_psnr_path.exists():
            models['xgb_psnr'] = joblib.load(xgb_psnr_path)
            print(f"✓ Loaded XGBoost PSNR model")
        else:
            print(f"⚠️  XGBoost PSNR model not found: {xgb_psnr_path}")

    # Load CNN models
    if HAS_TENSORFLOW:
        cnn_ratio_path = cnn_dir / 'cnn_compression_ratio_model.keras'
        cnn_psnr_path = cnn_dir / 'cnn_psnr_model.keras'
        cnn_ratio_scaler_path = cnn_dir / 'cnn_compression_ratio_scaler.pkl'
        cnn_psnr_scaler_path = cnn_dir / 'cnn_psnr_scaler.pkl'

        if cnn_ratio_path.exists():
            models['cnn_ratio'] = tf.keras.models.load_model(cnn_ratio_path)
            models['cnn_ratio_scaler'] = joblib.load(cnn_ratio_scaler_path)
            print(f"✓ Loaded CNN compression ratio model")
        else:
            print(f"⚠️  CNN ratio model not found: {cnn_ratio_path}")

        if cnn_psnr_path.exists():
            models['cnn_psnr'] = tf.keras.models.load_model(cnn_psnr_path)
            models['cnn_psnr_scaler'] = joblib.load(cnn_psnr_scaler_path)
            print(f"✓ Loaded CNN PSNR model")
        else:
            print(f"⚠️  CNN PSNR model not found: {cnn_psnr_path}")

    return models


def prepare_features(df):
    """Prepare feature matrix (same as training script)."""
    # One-hot encode library
    library_onehot = pd.get_dummies(df['Library'], prefix='Library')

    # One-hot encode data type
    if 'Data Type' in df.columns:
        data_type = df['Data Type']
    else:
        data_type = df['Distribution'].apply(lambda x: 'float' if 'float' in x else 'char')

    datatype_onehot = pd.get_dummies(data_type, prefix='Data Type')

    # Numerical features
    numerical_features = df[[
        'Chunk Size (bytes)',
        'Target CPU Util (%)',
        'Shannon Entropy (bits/byte)',
        'MAD',
        'Second Derivative Mean'
    ]].copy()

    # Combine all features
    X = pd.concat([
        numerical_features,
        library_onehot,
        datatype_onehot
    ], axis=1)

    return X


def compare_models(models, X_test, y_test, model_type='ratio'):
    """Compare performance of CNN vs XGBoost on test set."""
    results = {}

    # XGBoost prediction
    if f'xgb_{model_type}' in models:
        start_time = time.time()
        xgb_pred = models[f'xgb_{model_type}'].predict(X_test)
        xgb_time = time.time() - start_time

        results['XGBoost'] = {
            'predictions': xgb_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'mae': mean_absolute_error(y_test, xgb_pred),
            'r2': r2_score(y_test, xgb_pred),
            'inference_time': xgb_time,
            'time_per_sample': xgb_time / len(X_test) * 1000  # ms
        }

    # CNN prediction
    if f'cnn_{model_type}' in models:
        # Scale features for CNN
        X_test_scaled = models[f'cnn_{model_type}_scaler'].transform(X_test)

        start_time = time.time()
        cnn_pred = models[f'cnn_{model_type}'].predict(X_test_scaled, verbose=0).flatten()
        cnn_time = time.time() - start_time

        results['CNN'] = {
            'predictions': cnn_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, cnn_pred)),
            'mae': mean_absolute_error(y_test, cnn_pred),
            'r2': r2_score(y_test, cnn_pred),
            'inference_time': cnn_time,
            'time_per_sample': cnn_time / len(X_test) * 1000  # ms
        }

    return results


def plot_comparison(results_ratio, results_psnr, y_test_ratio, y_test_psnr, output_dir):
    """Create comparison visualizations."""
    print_header("GENERATING COMPARISON PLOTS")

    # Comparison plot for compression ratio
    if results_ratio:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for i, (model_name, result) in enumerate(results_ratio.items()):
            if i >= 2:
                break

            ax = axes[i]
            ax.scatter(y_test_ratio, result['predictions'], alpha=0.5, s=20)
            ax.plot([y_test_ratio.min(), y_test_ratio.max()],
                   [y_test_ratio.min(), y_test_ratio.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Compression Ratio')
            ax.set_ylabel('Predicted Compression Ratio')
            ax.set_title(f"{model_name}\nR² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
            ax.grid(True, alpha=0.3)

        plt.suptitle('Compression Ratio: CNN vs XGBoost', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_compression_ratio.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: comparison_compression_ratio.pdf")

    # Comparison plot for PSNR
    if results_psnr:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for i, (model_name, result) in enumerate(results_psnr.items()):
            if i >= 2:
                break

            ax = axes[i]
            ax.scatter(y_test_psnr, result['predictions'], alpha=0.5, s=20)
            ax.plot([y_test_psnr.min(), y_test_psnr.max()],
                   [y_test_psnr.min(), y_test_psnr.max()], 'r--', lw=2)
            ax.set_xlabel('Actual PSNR (dB)')
            ax.set_ylabel('Predicted PSNR (dB)')
            ax.set_title(f"{model_name}\nR² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f} dB")
            ax.grid(True, alpha=0.3)

        plt.suptitle('PSNR: CNN vs XGBoost', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_psnr.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: comparison_psnr.pdf")

    # Performance metrics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compression ratio metrics
    if results_ratio:
        models = list(results_ratio.keys())
        metrics = ['R²', 'RMSE', 'MAE', 'Time/Sample (ms)']
        x = np.arange(len(models))
        width = 0.35

        # R² comparison
        r2_values = [results_ratio[m]['r2'] for m in models]
        axes[0, 0].bar(x, r2_values, width, label='R²')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Compression Ratio - R² Score (Higher is Better)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # RMSE comparison
        rmse_values = [results_ratio[m]['rmse'] for m in models]
        axes[0, 1].bar(x, rmse_values, width, label='RMSE', color='orange')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Compression Ratio - RMSE (Lower is Better)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # PSNR metrics
    if results_psnr:
        models = list(results_psnr.keys())

        # R² comparison
        r2_values = [results_psnr[m]['r2'] for m in models]
        axes[1, 0].bar(x, r2_values, width, label='R²', color='green')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('PSNR - R² Score (Higher is Better)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models)
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Inference time comparison
        time_values = [results_psnr[m]['time_per_sample'] for m in models]
        axes[1, 1].bar(x, time_values, width, label='Time/Sample', color='red')
        axes[1, 1].set_ylabel('Time per Sample (ms)')
        axes[1, 1].set_title('Inference Time (Lower is Better)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: comparison_metrics.pdf")


def print_comparison_table(results_ratio, results_psnr):
    """Print formatted comparison table."""
    print_header("PERFORMANCE COMPARISON")

    if results_ratio:
        print("Compression Ratio Prediction:")
        print("-" * 80)
        print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Time/Sample (ms)':<20}")
        print("-" * 80)
        for model_name, result in results_ratio.items():
            print(f"{model_name:<15} {result['r2']:<10.4f} {result['rmse']:<10.4f} "
                  f"{result['mae']:<10.4f} {result['time_per_sample']:<20.6f}")
        print("-" * 80)

    print()

    if results_psnr:
        print("PSNR Prediction (Lossy Compressors):")
        print("-" * 80)
        print(f"{'Model':<15} {'R²':<10} {'RMSE (dB)':<12} {'MAE (dB)':<12} {'Time/Sample (ms)':<20}")
        print("-" * 80)
        for model_name, result in results_psnr.items():
            print(f"{model_name:<15} {result['r2']:<10.4f} {result['rmse']:<12.4f} "
                  f"{result['mae']:<12.4f} {result['time_per_sample']:<20.6f}")
        print("-" * 80)

    # Determine winner
    print("\nSummary:")
    if results_ratio:
        best_ratio_r2 = max(results_ratio.items(), key=lambda x: x[1]['r2'])
        best_ratio_speed = min(results_ratio.items(), key=lambda x: x[1]['time_per_sample'])
        print(f"  Best Compression Ratio Accuracy: {best_ratio_r2[0]} (R² = {best_ratio_r2[1]['r2']:.4f})")
        print(f"  Fastest Compression Ratio Inference: {best_ratio_speed[0]} ({best_ratio_speed[1]['time_per_sample']:.6f} ms/sample)")

    if results_psnr:
        best_psnr_r2 = max(results_psnr.items(), key=lambda x: x[1]['r2'])
        best_psnr_speed = min(results_psnr.items(), key=lambda x: x[1]['time_per_sample'])
        print(f"  Best PSNR Accuracy: {best_psnr_r2[0]} (R² = {best_psnr_r2[1]['r2']:.4f})")
        print(f"  Fastest PSNR Inference: {best_psnr_speed[0]} ({best_psnr_speed[1]['time_per_sample']:.6f} ms/sample)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_cnn_vs_xgboost.py <lossless_csv> <lossy_csv>")
        sys.exit(1)

    lossless_csv = sys.argv[1]
    lossy_csv = sys.argv[2]

    script_dir = Path(__file__).parent
    model_dir = script_dir / 'model_output'
    cnn_dir = script_dir / 'cnn_model_output'
    output_dir = script_dir / 'comparison_output'
    output_dir.mkdir(exist_ok=True)

    print_header("CNN vs XGBOOST MODEL COMPARISON")

    # Load models
    models = load_models(model_dir, cnn_dir)

    if not models:
        print("\n⚠️  No models found. Please train models first:")
        print("    python train_compression_model.py <lossless_csv> <lossy_csv>")
        print("    python train_cnn_compression_model.py <lossless_csv> <lossy_csv>")
        sys.exit(1)

    # Load test data
    print_header("LOADING TEST DATA")
    df_lossless = pd.read_csv(lossless_csv)
    df_lossy = pd.read_csv(lossy_csv)

    # Add PSNR sentinel for lossless
    if 'PSNR (dB)' not in df_lossless.columns:
        df_lossless['PSNR (dB)'] = 999.0

    df_combined = pd.concat([df_lossless, df_lossy], ignore_index=True)
    print(f"✓ Loaded {len(df_combined)} total records")

    # Prepare features
    X = prepare_features(df_combined)
    y_ratio = df_combined['Compression Ratio'].values
    y_psnr = df_combined['PSNR (dB)'].values

    # Use 20% as test set (same split as training)
    X_train, X_test, y_ratio_train, y_ratio_test = train_test_split(
        X, y_ratio, test_size=0.2, random_state=42
    )
    _, _, y_psnr_train, y_psnr_test = train_test_split(
        X, y_psnr, test_size=0.2, random_state=42
    )

    # Filter PSNR test set for lossy only
    lossy_mask = y_psnr_test < 999
    X_test_psnr = X_test[lossy_mask]
    y_psnr_test_filtered = y_psnr_test[lossy_mask]

    print(f"✓ Test set: {len(X_test)} samples (compression ratio)")
    print(f"✓ Test set: {len(X_test_psnr)} samples (PSNR, lossy only)")

    # Compare models
    print_header("EVALUATING MODELS ON TEST SET")

    results_ratio = compare_models(models, X_test, y_ratio_test, 'ratio')
    results_psnr = compare_models(models, X_test_psnr, y_psnr_test_filtered, 'psnr')

    # Print comparison table
    print_comparison_table(results_ratio, results_psnr)

    # Generate plots
    plot_comparison(results_ratio, results_psnr, y_ratio_test, y_psnr_test_filtered, output_dir)

    print_header("COMPARISON COMPLETE")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
