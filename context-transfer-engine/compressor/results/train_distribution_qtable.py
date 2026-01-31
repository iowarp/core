#!/usr/bin/env python3
"""
Q-Table Distribution Classifier

Trains a Q-table to predict parameterized distribution names (e.g., binned_normal_w1_p0)
from statistical features. Uses more features than the basic mathematical classifier.

Usage:
    python train_distribution_qtable.py                    # Train and evaluate
    python train_distribution_qtable.py --export           # Export model
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# Configuration
# ============================================================================

# Features for distribution prediction (more than the basic classifier)
FEATURE_COLS = [
    'data_size',
    'shannon_entropy',
    'mean_absolute_deviation',
    'first_order_derivative',
    'second_order_derivative',
    'block_entropy_mean',
    'block_entropy_std',
    'block_mad_mean',
    'block_mad_std',
    'block_skewness',
    'block_kurtosis',
    'block_deriv1_mean',
    'block_deriv1_std',
    'block_value_range',
    'block_value_concentration',
]

# Number of bins for discretization
N_BINS = 10


# ============================================================================
# Q-Table Distribution Classifier
# ============================================================================

class QTableDistributionClassifier:
    """Q-Table classifier for predicting parameterized distribution names."""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.qtable = {}  # state -> {distribution: count}
        self.bin_edges = {}
        self.distributions = []
        self.feature_names = []

    def fit(self, X, y, feature_names=None):
        """
        Build Q-table from training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Distribution labels (n_samples,)
            feature_names: List of feature names
        """
        if feature_names:
            self.feature_names = feature_names

        self.distributions = sorted(list(set(y)))
        print(f"Training Q-table with {len(self.distributions)} distribution classes")

        # Compute bin edges for each feature using percentiles
        for i in range(X.shape[1]):
            self.bin_edges[i] = np.percentile(
                X[:, i],
                np.linspace(0, 100, self.n_bins + 1)
            )[1:-1]  # Exclude min/max

        # Build Q-table: state -> distribution vote counts
        self.qtable = {}
        for j in range(len(X)):
            state = self._discretize(X[j])
            if state not in self.qtable:
                self.qtable[state] = defaultdict(int)
            self.qtable[state][y[j]] += 1

        # Convert to majority vote
        for state in self.qtable:
            counts = self.qtable[state]
            total = sum(counts.values())
            # Store distribution with highest count and confidence
            best_dist = max(counts, key=counts.get)
            confidence = counts[best_dist] / total
            self.qtable[state] = {
                'distribution': best_dist,
                'confidence': confidence,
                'total_count': total,
                'vote_counts': dict(counts)
            }

        print(f"Q-table has {len(self.qtable):,} unique states")
        return self

    def predict(self, X):
        """Predict distribution for each sample."""
        predictions = []
        confidences = []
        self.unknown_count = 0

        # Compute most common distribution as fallback
        all_dists = [v['distribution'] for v in self.qtable.values()]
        if all_dists:
            from collections import Counter
            self.fallback_dist = Counter(all_dists).most_common(1)[0][0]
        else:
            self.fallback_dist = self.distributions[0] if self.distributions else 'unknown'

        for j in range(len(X)):
            state = self._discretize(X[j])
            if state in self.qtable:
                predictions.append(self.qtable[state]['distribution'])
                confidences.append(self.qtable[state]['confidence'])
            else:
                predictions.append(self.fallback_dist)
                confidences.append(0.0)
                self.unknown_count += 1

        return np.array(predictions), np.array(confidences)

    def _discretize(self, x):
        """Convert feature vector to discrete state tuple."""
        state = []
        for i in range(len(x)):
            if i in self.bin_edges:
                bin_idx = np.searchsorted(self.bin_edges[i], x[i])
                state.append(int(bin_idx))
            else:
                state.append(0)
        return tuple(state)

    def export_json(self, output_dir):
        """Export Q-table to JSON for C++ loading."""
        os.makedirs(output_dir, exist_ok=True)

        # Export binning parameters
        bin_edges_list = {}
        for i, edges in self.bin_edges.items():
            bin_edges_list[str(i)] = edges.tolist()

        binning_params = {
            'n_bins': self.n_bins,
            'bin_edges': bin_edges_list,
            'feature_names': self.feature_names,
            'distributions': self.distributions
        }

        with open(f'{output_dir}/binning_params.json', 'w') as f:
            json.dump(binning_params, f, indent=2)

        # Export Q-table
        states_list = []
        for state, data in self.qtable.items():
            entry = {
                'state': list(state),
                'distribution': data['distribution'],
                'confidence': data['confidence'],
                'total_count': data['total_count']
            }
            states_list.append(entry)

        qtable_json = {
            'num_states': len(self.qtable),
            'num_distributions': len(self.distributions),
            'states': states_list
        }

        with open(f'{output_dir}/distribution_qtable.json', 'w') as f:
            json.dump(qtable_json, f, indent=2)

        # Export metadata
        metadata = {
            'model_type': 'distribution_qtable',
            'version': '1.0',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_bins': self.n_bins,
            'n_states': len(self.qtable),
            'n_distributions': len(self.distributions),
            'features': self.feature_names
        }

        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nExported to {output_dir}/")
        print(f"  - binning_params.json")
        print(f"  - distribution_qtable.json ({len(self.qtable):,} states)")
        print(f"  - metadata.json")


# ============================================================================
# Training Functions
# ============================================================================

def load_data(data_path='compression_benchmark_results.csv'):
    """Load and preprocess benchmark data."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples from {data_path}")

    # Get unique samples (one per distribution/data_size/data_type combo)
    # We want features that describe the data, not compression results
    unique_df = df.drop_duplicates(subset=['distribution_name', 'data_size', 'data_type'])
    print(f"Unique data samples: {len(unique_df):,}")

    # Check which features are available
    available_features = [f for f in FEATURE_COLS if f in unique_df.columns]
    missing_features = [f for f in FEATURE_COLS if f not in unique_df.columns]

    if missing_features:
        print(f"Missing features (will skip): {missing_features}")

    print(f"Using {len(available_features)} features: {available_features}")

    X = unique_df[available_features].values
    y = unique_df['distribution_name'].values

    return X, y, available_features, unique_df


def cross_validate(X, y, feature_names, n_folds=5):
    """Perform k-fold cross-validation."""
    print(f"\nRunning {n_folds}-fold cross-validation...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_unknown_rates = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = QTableDistributionClassifier(n_bins=N_BINS)
        model.fit(X_train, y_train, feature_names)
        y_pred, _ = model.predict(X_val)

        accuracy = (y_pred == y_val).mean()
        unknown_rate = model.unknown_count / len(X_val) * 100

        fold_accuracies.append(accuracy)
        fold_unknown_rates.append(unknown_rate)

        print(f"  Fold {fold}/{n_folds}: accuracy = {accuracy:.1%}, unknown = {unknown_rate:.1f}%")

    print(f"\nCV Results: {np.mean(fold_accuracies):.1%} ± {np.std(fold_accuracies):.1%}")
    print(f"Average unknown rate: {np.mean(fold_unknown_rates):.1f}%")

    return {
        'accuracies': fold_accuracies,
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'unknown_rates': fold_unknown_rates
    }


def train_and_evaluate(X, y, feature_names):
    """Train on full data and evaluate."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # Train
    model = QTableDistributionClassifier(n_bins=N_BINS)
    model.fit(X_train, y_train, feature_names)

    # Evaluate
    y_pred, confidences = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    unknown_rate = model.unknown_count / len(X_test) * 100

    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Unknown rate: {unknown_rate:.1f}%")
    print(f"  Mean confidence: {confidences.mean():.2f}")

    # Per-base-distribution accuracy
    print(f"\nPer-Distribution Accuracy:")
    for base in ['uniform', 'normal', 'gamma', 'exponential']:
        mask = np.array([base in d for d in y_test])
        if mask.sum() > 0:
            base_acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"  {base:12s}: {base_acc:.1%}")

    return model, accuracy, unknown_rate


def analyze_confusion(y_true, y_pred):
    """Analyze common misclassifications."""
    print(f"\nMost Common Misclassifications:")

    misclass = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            # Extract base distributions
            true_base = true.split('_')[1]  # e.g., 'normal' from 'binned_normal_w0_p0'
            pred_base = pred.split('_')[1]
            misclass[(true_base, pred_base)] += 1

    sorted_misclass = sorted(misclass.items(), key=lambda x: -x[1])[:10]
    for (true_base, pred_base), count in sorted_misclass:
        print(f"  {true_base} -> {pred_base}: {count}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Q-Table distribution classifier')
    parser.add_argument('--data', default='compression_benchmark_results.csv',
                        help='Path to benchmark data CSV')
    parser.add_argument('--export', action='store_true',
                        help='Export model')
    parser.add_argument('--output-dir', default='distribution_qtable_model',
                        help='Directory for model export')
    parser.add_argument('--bins', type=int, default=N_BINS,
                        help=f'Number of bins (default: {N_BINS})')
    args = parser.parse_args()

    global N_BINS
    N_BINS = args.bins

    print("=" * 70)
    print("Q-TABLE DISTRIBUTION CLASSIFIER")
    print("=" * 70)

    # Load data
    X, y, feature_names, df = load_data(args.data)

    print(f"\nNumber of distribution classes: {len(set(y))}")

    # Cross-validation
    cv_results = cross_validate(X, y, feature_names)

    # Train and evaluate
    model, accuracy, unknown_rate = train_and_evaluate(X, y, feature_names)

    # Export if requested
    if args.export:
        model.export_json(args.output_dir)

    # Save results
    results = {
        'model_type': 'distribution_qtable',
        'n_bins': N_BINS,
        'n_features': len(feature_names),
        'features': feature_names,
        'n_distributions': len(set(y)),
        'cv_accuracy': cv_results['mean_accuracy'],
        'cv_std': cv_results['std_accuracy'],
        'test_accuracy': accuracy,
        'unknown_rate': unknown_rate,
        'n_states': len(model.qtable)
    }

    with open('distribution_qtable_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: distribution_qtable_results.json")

    print("\nDone!")


if __name__ == '__main__':
    main()
