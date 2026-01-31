#!/usr/bin/env python3
"""
Linear Regression Table Training Script

Trains a table of linear regressors indexed by [library][configuration][data_type].
Each regressor takes data_size as input and predicts:
- compress_time_ms
- decompress_time_ms
- compression_ratio

Only trains on lossless algorithms.

Usage:
    python train_linreg_table.py                     # Train and evaluate
    python train_linreg_table.py --export           # Export model for C++
    python train_linreg_table.py --output-dir DIR   # Custom output directory
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# ============================================================================
# Configuration
# ============================================================================

# Lossless compression libraries to include in training
LOSSLESS_LIBRARIES = [
    'ZSTD', 'ZLIB', 'LZMA', 'LZ4', 'BZIP2', 'BROTLI', 'SNAPPY', 'Blosc2'
]

# Lossy libraries to exclude
LOSSY_LIBRARIES = ['ZFP', 'SZ3', 'FPZIP']

# Target columns
TARGET_COLS = ['compress_time_ms', 'decompress_time_ms', 'compression_ratio']


# ============================================================================
# Linear Regression Table Model
# ============================================================================

class LinearRegressionTable:
    """
    Table of linear regressors indexed by [library][configuration][data_type][distribution].

    Each regressor: data_size -> [compress_time, decompress_time, compress_ratio]

    Uses predicted_distribution (from the mathematical classifier) instead of
    ground truth distribution_name for runtime lookup compatibility.
    """

    def __init__(self, min_samples: int = 10):
        """
        Initialize the model.

        Args:
            min_samples: Minimum samples required to fit a model
        """
        self.min_samples = min_samples
        self.models: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self.global_stats = {}

    def fit(self, df: pd.DataFrame) -> 'LinearRegressionTable':
        """
        Fit linear regression models for each (library, config, data_type, predicted_distribution) group.

        Args:
            df: DataFrame with columns: library_name, configuration, data_type,
                predicted_distribution, data_size, compress_time_ms, decompress_time_ms,
                compression_ratio

        Returns:
            self
        """
        # Filter for lossless libraries only
        df_lossless = df[df['library_name'].isin(LOSSLESS_LIBRARIES)].copy()
        print(f"Filtered to {len(df_lossless):,} lossless samples from {len(df):,} total")

        # Group by (library, config, data_type, predicted_distribution)
        # Using predicted_distribution allows runtime lookup with classified distribution
        grouped = df_lossless.groupby(['library_name', 'configuration', 'data_type', 'predicted_distribution'])

        for (library, config, dtype, dist), group in grouped:
            if len(group) < self.min_samples:
                # Don't print skip messages for every combination (too many)
                continue

            key = (library, config, dtype, dist)

            # Extract features (data_size) and targets
            X = group['data_size'].values.reshape(-1, 1)
            y_compress = group['compress_time_ms'].values
            y_decompress = group['decompress_time_ms'].values
            y_ratio = group['compression_ratio'].values

            # Fit separate linear regression for each target
            model_compress = LinearRegression()
            model_compress.fit(X, y_compress)

            model_decompress = LinearRegression()
            model_decompress.fit(X, y_decompress)

            model_ratio = LinearRegression()
            model_ratio.fit(X, y_ratio)

            # Calculate R² scores
            r2_compress = r2_score(y_compress, model_compress.predict(X))
            r2_decompress = r2_score(y_decompress, model_decompress.predict(X))
            r2_ratio = r2_score(y_ratio, model_ratio.predict(X))

            self.models[key] = {
                'compress_time': {
                    'model': model_compress,
                    'slope': float(model_compress.coef_[0]),
                    'intercept': float(model_compress.intercept_),
                    'r2': r2_compress
                },
                'decompress_time': {
                    'model': model_decompress,
                    'slope': float(model_decompress.coef_[0]),
                    'intercept': float(model_decompress.intercept_),
                    'r2': r2_decompress
                },
                'compress_ratio': {
                    'model': model_ratio,
                    'slope': float(model_ratio.coef_[0]),
                    'intercept': float(model_ratio.intercept_),
                    'r2': r2_ratio
                },
                'sample_count': len(group)
            }

        # Compute global statistics
        self.global_stats = {
            'num_models': len(self.models),
            'libraries': list(set(k[0] for k in self.models.keys())),
            'configurations': list(set(k[1] for k in self.models.keys())),
            'data_types': list(set(k[2] for k in self.models.keys())),
            'distributions': list(set(k[3] for k in self.models.keys())),
            'total_samples': sum(m['sample_count'] for m in self.models.values())
        }

        print(f"Trained {len(self.models)} linear regression models")
        return self

    def predict(self, library: str, config: str, dtype: str, dist: str,
                data_size: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict compression metrics for given parameters.

        Args:
            library: Library name
            config: Configuration name
            dtype: Data type name
            dist: Distribution name
            data_size: Array of data sizes

        Returns:
            Dictionary with predicted values for each target
        """
        key = (library, config, dtype, dist)

        if key not in self.models:
            # Return defaults
            return {
                'compress_time_ms': np.full_like(data_size, 0.1, dtype=float),
                'decompress_time_ms': np.full_like(data_size, 0.05, dtype=float),
                'compression_ratio': np.full_like(data_size, 1.5, dtype=float)
            }

        model_data = self.models[key]
        X = data_size.reshape(-1, 1) if data_size.ndim == 1 else data_size

        return {
            'compress_time_ms': np.maximum(0, model_data['compress_time']['model'].predict(X)),
            'decompress_time_ms': np.maximum(0, model_data['decompress_time']['model'].predict(X)),
            'compression_ratio': np.maximum(1.0, model_data['compress_ratio']['model'].predict(X))
        }

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            df: Test DataFrame

        Returns:
            Dictionary with evaluation metrics
        """
        df_lossless = df[df['library_name'].isin(LOSSLESS_LIBRARIES)].copy()

        results = {
            'per_key': {},
            'overall': {target: {'r2': [], 'mae': [], 'mse': []} for target in TARGET_COLS}
        }

        grouped = df_lossless.groupby(['library_name', 'configuration', 'data_type', 'predicted_distribution'])

        for (library, config, dtype, dist), group in grouped:
            key = (library, config, dtype, dist)
            key_str = f"{library}_{config}_{dtype}_{dist}"

            if key not in self.models:
                continue

            X = group['data_size'].values
            predictions = self.predict(library, config, dtype, dist, X)

            key_results = {}
            for target in TARGET_COLS:
                y_true = group[target].values
                y_pred = predictions[target]

                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)

                key_results[target] = {'r2': r2, 'mae': mae, 'mse': mse}
                results['overall'][target]['r2'].append(r2)
                results['overall'][target]['mae'].append(mae)
                results['overall'][target]['mse'].append(mse)

            results['per_key'][key_str] = key_results

        # Compute overall averages
        for target in TARGET_COLS:
            results['overall'][target] = {
                'mean_r2': np.mean(results['overall'][target]['r2']),
                'std_r2': np.std(results['overall'][target]['r2']),
                'mean_mae': np.mean(results['overall'][target]['mae']),
                'mean_mse': np.mean(results['overall'][target]['mse'])
            }

        return results

    def export_json(self, output_dir: str) -> None:
        """
        Export model to JSON format for C++ loading.

        Args:
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build models list
        models_list = []
        for (library, config, dtype, dist), model_data in self.models.items():
            entry = {
                'library': library,
                'config': config,
                'data_type': dtype,
                'distribution': dist,
                'slope_compress_time': model_data['compress_time']['slope'],
                'intercept_compress_time': model_data['compress_time']['intercept'],
                'slope_decompress_time': model_data['decompress_time']['slope'],
                'intercept_decompress_time': model_data['decompress_time']['intercept'],
                'slope_compress_ratio': model_data['compress_ratio']['slope'],
                'intercept_compress_ratio': model_data['compress_ratio']['intercept'],
                'sample_count': model_data['sample_count'],
                'r2_compress_time': model_data['compress_time']['r2'],
                'r2_decompress_time': model_data['decompress_time']['r2'],
                'r2_compress_ratio': model_data['compress_ratio']['r2']
            }
            models_list.append(entry)

        # Sort for consistent output
        models_list.sort(key=lambda x: (x['library'], x['config'], x['data_type'], x['distribution']))

        # Write linreg_table.json
        table_json = {
            'model_type': 'linreg_table',
            'version': '1.0',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_models': len(models_list),
            'models': models_list
        }

        table_path = os.path.join(output_dir, 'linreg_table.json')
        with open(table_path, 'w') as f:
            json.dump(table_json, f, indent=2)

        # Write metadata.json
        metadata = {
            'model_type': 'linreg_table',
            'version': '1.1',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_models': len(self.models),
            'total_samples': self.global_stats['total_samples'],
            'libraries': sorted(self.global_stats['libraries']),
            'configurations': sorted(self.global_stats['configurations']),
            'data_types': sorted(self.global_stats['data_types']),
            'distributions': sorted(self.global_stats['distributions']),
            'features': ['data_size'],
            'targets': TARGET_COLS,
            'lossless_only': True
        }

        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nExported model to {output_dir}/")
        print(f"  - linreg_table.json ({len(models_list)} models)")
        print(f"  - metadata.json")

    def get_statistics(self) -> str:
        """Get model statistics as string."""
        lines = [
            "LinearRegressionTable Statistics:",
            f"  Number of models: {len(self.models)}",
            f"  Libraries: {sorted(self.global_stats['libraries'])}",
            f"  Configurations: {sorted(self.global_stats['configurations'])}",
            f"  Data types: {sorted(self.global_stats['data_types'])}",
            f"  Distributions: {len(self.global_stats['distributions'])} unique",
            f"  Total training samples: {self.global_stats['total_samples']}",
        ]

        # Average R² scores
        avg_r2 = {'compress_time': [], 'decompress_time': [], 'compress_ratio': []}
        for model_data in self.models.values():
            avg_r2['compress_time'].append(model_data['compress_time']['r2'])
            avg_r2['decompress_time'].append(model_data['decompress_time']['r2'])
            avg_r2['compress_ratio'].append(model_data['compress_ratio']['r2'])

        lines.append(f"  Average R² (compress_time): {np.mean(avg_r2['compress_time']):.4f}")
        lines.append(f"  Average R² (decompress_time): {np.mean(avg_r2['decompress_time']):.4f}")
        lines.append(f"  Average R² (compress_ratio): {np.mean(avg_r2['compress_ratio']):.4f}")

        return '\n'.join(lines)


# ============================================================================
# Training Functions
# ============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate benchmark data."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples from {data_path}")

    # Check required columns
    required_cols = ['library_name', 'configuration', 'data_type', 'data_size',
                     'compression_ratio', 'compress_time_ms', 'decompress_time_ms']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Print data summary
    print(f"  Libraries: {sorted(df['library_name'].unique())}")
    print(f"  Configurations: {sorted(df['configuration'].unique())}")
    print(f"  Data types: {sorted(df['data_type'].unique())}")
    print(f"  Data size range: {df['data_size'].min():,} - {df['data_size'].max():,}")

    return df


def cross_validate(df: pd.DataFrame, n_folds: int = 5) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        df: Training data
        n_folds: Number of CV folds

    Returns:
        Dictionary with CV results
    """
    print(f"\nRunning {n_folds}-fold cross-validation...")

    # Filter for lossless only
    df_lossless = df[df['library_name'].isin(LOSSLESS_LIBRARIES)].copy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_lossless), 1):
        df_train = df_lossless.iloc[train_idx]
        df_val = df_lossless.iloc[val_idx]

        # Train model
        model = LinearRegressionTable()
        model.fit(df_train)

        # Evaluate
        eval_results = model.evaluate(df_val)

        fold_r2 = {target: eval_results['overall'][target]['mean_r2']
                   for target in TARGET_COLS}
        fold_results.append(fold_r2)

        mean_r2 = np.mean(list(fold_r2.values()))
        print(f"  Fold {fold}/{n_folds}: mean R² = {mean_r2:.4f}")

    # Aggregate results
    cv_results = {target: [] for target in TARGET_COLS}
    for fold_r2 in fold_results:
        for target in TARGET_COLS:
            cv_results[target].append(fold_r2[target])

    print("\nCV Results (mean ± std):")
    overall_r2 = []
    for target in TARGET_COLS:
        mean = np.mean(cv_results[target])
        std = np.std(cv_results[target])
        overall_r2.append(mean)
        print(f"  {target:25s}: {mean:.4f} ± {std:.4f}")

    print(f"\n  Overall CV R²: {np.mean(overall_r2):.4f}")

    return {
        'per_target': cv_results,
        'overall_r2': np.mean(overall_r2)
    }


def train_and_evaluate(df: pd.DataFrame) -> Tuple[LinearRegressionTable, Dict]:
    """
    Train model and evaluate on test set.

    Args:
        df: Full dataset

    Returns:
        Tuple of (trained model, evaluation results)
    """
    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTraining samples: {len(df_train):,}")
    print(f"Test samples: {len(df_test):,}")

    # Train
    print("\nTraining linear regression table...")
    start_time = time.time()

    model = LinearRegressionTable()
    model.fit(df_train)

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f}s")

    # Print model statistics
    print(f"\n{model.get_statistics()}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = model.evaluate(df_test)

    print("\nTest Set Results:")
    for target in TARGET_COLS:
        stats = eval_results['overall'][target]
        print(f"  {target:25s}: R² = {stats['mean_r2']:.4f}, MAE = {stats['mean_mae']:.4f}")

    overall_r2 = np.mean([eval_results['overall'][t]['mean_r2'] for t in TARGET_COLS])
    print(f"\n  Overall Test R²: {overall_r2:.4f}")

    return model, eval_results


def print_sample_predictions(model: LinearRegressionTable, df: pd.DataFrame) -> None:
    """Print sample predictions for verification."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    # Sample a few (library, config, dtype, predicted_distribution) combinations
    # Using predicted_distribution values: 'uniform', 'normal', 'gamma', 'exponential'
    samples = [
        ('ZSTD', 'fast', 'char', 'normal'),
        ('ZSTD', 'balanced', 'float', 'uniform'),
        ('LZ4', 'fast', 'char', 'gamma'),
        ('BZIP2', 'best', 'int', 'exponential'),
    ]

    test_sizes = np.array([1024, 10240, 102400, 1024000])

    for library, config, dtype, dist in samples:
        key = (library, config, dtype, dist)
        if key not in model.models:
            print(f"\n{library}/{config}/{dtype}/{dist}: No model available")
            continue

        print(f"\n{library}/{config}/{dtype}/{dist}:")
        print(f"  Coefficients:")
        coeffs = model.models[key]
        print(f"    compress_time:   slope={coeffs['compress_time']['slope']:.2e}, "
              f"intercept={coeffs['compress_time']['intercept']:.4f}, "
              f"R²={coeffs['compress_time']['r2']:.4f}")
        print(f"    decompress_time: slope={coeffs['decompress_time']['slope']:.2e}, "
              f"intercept={coeffs['decompress_time']['intercept']:.4f}, "
              f"R²={coeffs['decompress_time']['r2']:.4f}")
        print(f"    compress_ratio:  slope={coeffs['compress_ratio']['slope']:.2e}, "
              f"intercept={coeffs['compress_ratio']['intercept']:.4f}, "
              f"R²={coeffs['compress_ratio']['r2']:.4f}")

        print(f"\n  Predictions for test sizes:")
        preds = model.predict(library, config, dtype, dist, test_sizes)
        for i, size in enumerate(test_sizes):
            print(f"    {size:>10,} bytes: "
                  f"compress={preds['compress_time_ms'][i]:.4f} ms, "
                  f"decompress={preds['decompress_time_ms'][i]:.4f} ms, "
                  f"ratio={preds['compression_ratio'][i]:.2f}x")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train linear regression table for compression prediction'
    )
    parser.add_argument('--data', default='compression_benchmark_results.csv',
                        help='Path to benchmark data CSV')
    parser.add_argument('--export', action='store_true',
                        help='Export model for C++ inference')
    parser.add_argument('--output-dir', default='linreg_table_model',
                        help='Directory for model export')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation')
    parser.add_argument('--no-samples', action='store_true',
                        help='Skip sample predictions')
    args = parser.parse_args()

    print("=" * 80)
    print("LINEAR REGRESSION TABLE - COMPRESSION PREDICTION")
    print("=" * 80)

    # Load data
    df = load_data(args.data)

    # Cross-validation (optional)
    if args.cv:
        cv_results = cross_validate(df)

    # Train and evaluate
    model, eval_results = train_and_evaluate(df)

    # Print sample predictions
    if not args.no_samples:
        print_sample_predictions(model, df)

    # Export model
    if args.export:
        print("\n" + "=" * 60)
        print("EXPORTING MODEL")
        print("=" * 60)

        # Train on full dataset for export
        full_model = LinearRegressionTable()
        full_model.fit(df)
        full_model.export_json(args.output_dir)

    # Save results summary
    results_summary = {
        'model_type': 'linreg_table',
        'version': '1.1',
        'lossless_only': True,
        'libraries': sorted(model.global_stats['libraries']),
        'configurations': sorted(model.global_stats['configurations']),
        'data_types': sorted(model.global_stats['data_types']),
        'num_distributions': len(model.global_stats['distributions']),
        'num_models': len(model.models),
        'total_samples': model.global_stats['total_samples'],
        'test_results': {
            target: {
                'mean_r2': float(eval_results['overall'][target]['mean_r2']),
                'mean_mae': float(eval_results['overall'][target]['mean_mae'])
            }
            for target in TARGET_COLS
        }
    }

    with open('linreg_table_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults saved to: linreg_table_results.json")

    print("\nDone!")


if __name__ == '__main__':
    main()
