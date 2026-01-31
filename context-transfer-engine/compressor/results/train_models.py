#!/usr/bin/env python3
"""
Unified model training script with 5-fold cross-validation.

Trains XGBoost, Q-Table, and DNN models on compression benchmark data.
Exports the Q-Table model for C++ inference.

Usage:
    python train_models.py                    # Train all models
    python train_models.py --export-qtable   # Also export Q-table for C++
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

FEATURE_COLS = [
    'library_id', 'config_id', 'datatype_id',
    'data_size', 'shannon_entropy', 'mean_absolute_deviation',
    'second_order_derivative'
]

TARGET_COLS = ['compression_ratio', 'compress_time_ms', 'decompress_time_ms', 'psnr_db']

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

QTABLE_BINS = 15

# ============================================================================
# Q-Table Implementation
# ============================================================================

class QTableModel:
    """Q-Table model for compression prediction."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.qtable = {}
        self.bin_edges = {}
        self.global_average = None
        self.feature_names = FEATURE_COLS
        self.target_names = TARGET_COLS
        self.label_encoders = {}

    def fit(self, X, y, feature_names=None):
        """Build Q-table from training data."""
        if feature_names:
            self.feature_names = feature_names

        # Compute bin edges for continuous features (indices 3-6)
        continuous_indices = [3, 4, 5, 6] if X.shape[1] > 6 else [3, 4, 5]
        for i in continuous_indices:
            self.bin_edges[i] = np.percentile(
                X[:, i],
                np.linspace(0, 100, self.n_bins + 1)
            )[1:-1]  # Exclude min/max for binning

        # Build Q-table
        self.qtable = {}
        for j in range(len(X)):
            state = self._discretize(X[j])
            if state not in self.qtable:
                self.qtable[state] = {'sum': np.zeros(len(TARGET_COLS)), 'count': 0}
            self.qtable[state]['sum'] += y[j]
            self.qtable[state]['count'] += 1

        # Compute averages
        for state in self.qtable:
            self.qtable[state]['value'] = (
                self.qtable[state]['sum'] / self.qtable[state]['count']
            )

        # Global average for unknown states
        self.global_average = np.mean(y, axis=0)

        return self

    def predict(self, X):
        """Predict using Q-table."""
        predictions = np.zeros((len(X), len(TARGET_COLS)))
        self.unknown_count = 0

        for j in range(len(X)):
            state = self._discretize(X[j])
            if state in self.qtable:
                predictions[j] = self.qtable[state]['value']
            else:
                predictions[j] = self.global_average
                self.unknown_count += 1

        return predictions

    def _discretize(self, x):
        """Convert feature vector to discrete state tuple."""
        state = [int(x[0]), int(x[1]), int(x[2])]  # Categorical features

        # Continuous features
        continuous_indices = [3, 4, 5, 6] if len(x) > 6 else [3, 4, 5]
        for i in continuous_indices:
            if i in self.bin_edges:
                bin_idx = np.searchsorted(self.bin_edges[i], x[i])
                state.append(int(bin_idx))
            else:
                state.append(0)

        return tuple(state)

    def export_json(self, output_dir):
        """Export Q-table to JSON format for C++ loading."""
        os.makedirs(output_dir, exist_ok=True)

        # Export binning parameters
        bin_edges_list = [[] for _ in range(len(self.feature_names))]
        for i, edges in self.bin_edges.items():
            bin_edges_list[i] = edges.tolist()

        binning_params = {
            'n_bins': self.n_bins,
            'use_nearest_neighbor': False,
            'nn_k': 5,
            'bin_edges': bin_edges_list,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'label_encoders': self.label_encoders
        }

        with open(f'{output_dir}/binning_params.json', 'w') as f:
            json.dump(binning_params, f, indent=2)

        # Export Q-table
        states_list = []
        for state, data in self.qtable.items():
            entry = {
                'state': list(state),
                'compression_ratio': float(data['value'][0]),
                'psnr_db': float(data['value'][3]),
                'compression_time_ms': float(data['value'][1]),
                'sample_count': int(data['count'])
            }
            # Add decompress_time if available
            if len(data['value']) > 2:
                entry['decompress_time_ms'] = float(data['value'][2])
            states_list.append(entry)

        qtable_json = {
            'num_states': len(self.qtable),
            'states': states_list,
            'global_average': {
                'compression_ratio': float(self.global_average[0]),
                'psnr_db': float(self.global_average[3]),
                'compression_time_ms': float(self.global_average[1]),
                'decompress_time_ms': float(self.global_average[2]) if len(self.global_average) > 2 else 0.0,
                'sample_count': sum(d['count'] for d in self.qtable.values())
            }
        }

        with open(f'{output_dir}/qtable.json', 'w') as f:
            json.dump(qtable_json, f, indent=2)

        # Export metadata
        metadata = {
            'model_type': 'qtable',
            'version': '1.0',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_bins': self.n_bins,
            'n_states': len(self.qtable),
            'features': self.feature_names,
            'targets': self.target_names
        }

        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Q-table exported to: {output_dir}/")
        print(f"  - binning_params.json")
        print(f"  - qtable.json ({len(self.qtable):,} states)")
        print(f"  - metadata.json")


# ============================================================================
# Training Functions
# ============================================================================

def load_data(data_path='compression_benchmark_results.csv'):
    """Load and preprocess benchmark data."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples from {data_path}")

    # Encode categorical features
    le_library = LabelEncoder()
    le_config = LabelEncoder()
    le_datatype = LabelEncoder()

    df['library_id'] = le_library.fit_transform(df['library_name'])
    df['config_id'] = le_config.fit_transform(df['configuration'])
    df['datatype_id'] = le_datatype.fit_transform(df['data_type'])

    # Store encoders for export
    label_encoders = {
        'library_name': {str(i): name for i, name in enumerate(le_library.classes_)},
        'configuration': {str(i): name for i, name in enumerate(le_config.classes_)},
        'data_type': {str(i): name for i, name in enumerate(le_datatype.classes_)}
    }

    # Check for second_order_derivative column
    feature_cols = FEATURE_COLS.copy()
    if 'second_order_derivative' not in df.columns:
        feature_cols.remove('second_order_derivative')
        print("Note: second_order_derivative not in dataset, using 6 features")

    X = df[feature_cols].values
    y = df[TARGET_COLS].values

    return X, y, label_encoders, feature_cols


def train_xgboost(X_train, y_train, X_test, y_test, cv_folds=5):
    """Train XGBoost with cross-validation."""
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("XGBOOST - 5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = {target: [] for target in TARGET_COLS}

    print(f"\nRunning {cv_folds}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        fold_scores = []
        for i, target in enumerate(TARGET_COLS):
            model = xgb.XGBRegressor(**XGBOOST_PARAMS)
            model.fit(X_fold_train, y_fold_train[:, i], verbose=False)
            y_pred = model.predict(X_fold_val)
            r2 = r2_score(y_fold_val[:, i], y_pred)
            cv_scores[target].append(r2)
            fold_scores.append(r2)

        print(f"  Fold {fold}/{cv_folds}: mean R² = {np.mean(fold_scores):.4f}")

    # Print CV results
    print("\nCV Results (mean ± std):")
    cv_means = {}
    for target in TARGET_COLS:
        mean = np.mean(cv_scores[target])
        std = np.std(cv_scores[target])
        cv_means[target] = mean
        print(f"  {target:25s}: {mean:.4f} ± {std:.4f}")
    print(f"\n  Overall CV R²: {np.mean(list(cv_means.values())):.4f}")

    # Train final models on full training set
    print("\nTraining final models on full training set...")
    start_time = time.time()
    models = {}
    test_scores = {}

    for i, target in enumerate(TARGET_COLS):
        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        model.fit(X_train, y_train[:, i], verbose=False)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test[:, i], y_pred)
        mae = mean_absolute_error(y_test[:, i], y_pred)
        test_scores[target] = {'r2': r2, 'mae': mae}
        models[target] = model
        print(f"  {target:25s}: R² = {r2:.4f}, MAE = {mae:.4f}")

    training_time = time.time() - start_time
    overall_r2 = np.mean([s['r2'] for s in test_scores.values()])
    print(f"\n  Overall Test R²: {overall_r2:.4f}")
    print(f"  Training time: {training_time:.1f}s")

    return {
        'models': models,
        'cv_scores': cv_scores,
        'test_scores': test_scores,
        'overall_r2': overall_r2,
        'training_time': training_time
    }


def train_qtable(X_train, y_train, X_test, y_test, feature_names, label_encoders, cv_folds=5):
    """Train Q-Table with cross-validation."""
    print("\n" + "=" * 60)
    print("Q-TABLE - 5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = {target: [] for target in TARGET_COLS}
    unknown_rates = []

    print(f"\nRunning {cv_folds}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        model = QTableModel(n_bins=QTABLE_BINS)
        model.fit(X_fold_train, y_fold_train, feature_names)
        y_pred = model.predict(X_fold_val)

        unknown_rate = model.unknown_count / len(X_fold_val) * 100
        unknown_rates.append(unknown_rate)

        fold_scores = []
        for i, target in enumerate(TARGET_COLS):
            r2 = r2_score(y_fold_val[:, i], y_pred[:, i])
            cv_scores[target].append(r2)
            fold_scores.append(r2)

        print(f"  Fold {fold}/{cv_folds}: mean R² = {np.mean(fold_scores):.4f}, unknown = {unknown_rate:.1f}%")

    # Print CV results
    print("\nCV Results (mean ± std):")
    cv_means = {}
    for target in TARGET_COLS:
        mean = np.mean(cv_scores[target])
        std = np.std(cv_scores[target])
        cv_means[target] = mean
        print(f"  {target:25s}: {mean:.4f} ± {std:.4f}")
    print(f"\n  Overall CV R²: {np.mean(list(cv_means.values())):.4f}")
    print(f"  Average unknown rate: {np.mean(unknown_rates):.1f}%")

    # Train final model on full training set
    print("\nTraining final Q-table on full training set...")
    start_time = time.time()

    final_model = QTableModel(n_bins=QTABLE_BINS)
    final_model.fit(X_train, y_train, feature_names)
    final_model.label_encoders = label_encoders

    y_pred = final_model.predict(X_test)

    test_scores = {}
    for i, target in enumerate(TARGET_COLS):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        test_scores[target] = {'r2': r2, 'mae': mae}
        print(f"  {target:25s}: R² = {r2:.4f}, MAE = {mae:.4f}")

    training_time = time.time() - start_time
    overall_r2 = np.mean([s['r2'] for s in test_scores.values()])
    unknown_rate_test = final_model.unknown_count / len(X_test) * 100

    print(f"\n  Overall Test R²: {overall_r2:.4f}")
    print(f"  Test unknown rate: {unknown_rate_test:.1f}%")
    print(f"  Unique states: {len(final_model.qtable):,}")
    print(f"  Training time: {training_time:.1f}s")

    return {
        'model': final_model,
        'cv_scores': cv_scores,
        'test_scores': test_scores,
        'overall_r2': overall_r2,
        'unknown_rate': unknown_rate_test,
        'n_states': len(final_model.qtable),
        'training_time': training_time
    }


def train_dnn(X_train, y_train, X_test, y_test):
    """Train Deep Neural Network (PyTorch)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("\nDNN: PyTorch not available, skipping DNN training")
        return None

    print("\n" + "=" * 60)
    print("DEEP NEURAL NETWORK (PyTorch)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Model
    class CompressionDNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )

        def forward(self, x):
            return self.network(x)

    model = CompressionDNN(X_train.shape[1], len(TARGET_COLS)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    start_time = time.time()
    epochs = 50

    print("\nTraining DNN...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

    training_time = time.time() - start_time

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    test_scores = {}
    for i, target in enumerate(TARGET_COLS):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        test_scores[target] = {'r2': r2, 'mae': mae}
        print(f"  {target:25s}: R² = {r2:.4f}, MAE = {mae:.4f}")

    overall_r2 = np.mean([s['r2'] for s in test_scores.values()])
    print(f"\n  Overall Test R²: {overall_r2:.4f}")
    print(f"  Training time: {training_time:.1f}s")

    return {
        'model': model,
        'scaler': scaler,
        'test_scores': test_scores,
        'overall_r2': overall_r2,
        'training_time': training_time
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train compression prediction models')
    parser.add_argument('--data', default='compression_benchmark_results.csv',
                       help='Path to benchmark data CSV')
    parser.add_argument('--export-qtable', action='store_true',
                       help='Export Q-table model for C++ inference')
    parser.add_argument('--output-dir', default='qtable_model',
                       help='Directory for Q-table export')
    args = parser.parse_args()

    print("=" * 80)
    print("COMPRESSION PREDICTION MODEL TRAINING")
    print("=" * 80)

    # Load data
    X, y, label_encoders, feature_cols = load_data(args.data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")

    # Train models
    results = {}

    # XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {
        'overall_r2': xgb_results['overall_r2'],
        'per_target_r2': {t: s['r2'] for t, s in xgb_results['test_scores'].items()},
        'training_time_s': xgb_results['training_time']
    }

    # Q-Table
    qtable_results = train_qtable(X_train, y_train, X_test, y_test, feature_cols, label_encoders)
    results['qtable'] = {
        'overall_r2': qtable_results['overall_r2'],
        'per_target_r2': {t: s['r2'] for t, s in qtable_results['test_scores'].items()},
        'unknown_rate': qtable_results['unknown_rate'],
        'n_states': qtable_results['n_states'],
        'training_time_s': qtable_results['training_time']
    }

    # DNN
    dnn_results = train_dnn(X_train, y_train, X_test, y_test)
    if dnn_results:
        results['dnn'] = {
            'overall_r2': dnn_results['overall_r2'],
            'per_target_r2': {t: s['r2'] for t, s in dnn_results['test_scores'].items()},
            'training_time_s': dnn_results['training_time']
        }

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<15} {'Overall R²':>12} {'Training Time':>15}")
    print("-" * 45)
    for model_name, model_results in results.items():
        print(f"{model_name:<15} {model_results['overall_r2']:>12.4f} "
              f"{model_results['training_time_s']:>12.1f}s")

    # Save results
    with open('model_comparison_results.json', 'w') as f:
        json.dump({'models': results}, f, indent=2)
    print("\nResults saved to: model_comparison_results.json")

    # Export Q-table if requested
    if args.export_qtable:
        print("\n" + "=" * 60)
        print("EXPORTING Q-TABLE MODEL")
        print("=" * 60)
        qtable_results['model'].export_json(args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
