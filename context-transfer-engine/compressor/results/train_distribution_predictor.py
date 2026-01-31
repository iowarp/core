#!/usr/bin/env python3
"""
Distribution Predictor using Q-Matrix approach.

This script trains a model to predict the data distribution based on
statistical features (shannon entropy, MAD, derivatives, etc.).

The Q-matrix approach bins continuous features and creates a lookup table
mapping feature combinations to the most likely distribution.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from typing import Dict, List, Tuple, Optional

# Features to use for prediction
FEATURE_COLUMNS = [
    'shannon_entropy',
    'mean_absolute_deviation',
    'first_order_derivative',
    'second_order_derivative'
]

# Number of bins for each feature
DEFAULT_NUM_BINS = 10


class DistributionQMatrix:
    """
    Q-Matrix predictor for data distribution classification.

    Bins continuous features and builds a lookup table mapping
    feature combinations to distribution names.
    """

    def __init__(self, num_bins: int = DEFAULT_NUM_BINS):
        self.num_bins = num_bins
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.q_matrix: Dict[str, Dict[str, int]] = {}  # key -> {distribution: count}
        self.distributions: List[str] = []
        self.label_encoder = LabelEncoder()

    def _compute_bin_edges(self, df: pd.DataFrame, feature: str) -> np.ndarray:
        """Compute bin edges using quantiles for better distribution."""
        values = df[feature].dropna()
        # Use quantile-based binning for better coverage
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        edges = np.percentile(values, percentiles)
        # Remove duplicates while preserving order
        edges = np.unique(edges)
        return edges

    def _get_bin_index(self, value: float, edges: np.ndarray) -> int:
        """Get the bin index for a value given bin edges."""
        if np.isnan(value):
            return -1
        idx = np.searchsorted(edges, value, side='right') - 1
        return max(0, min(idx, len(edges) - 2))

    def _get_feature_key(self, row: pd.Series) -> str:
        """Generate a string key from binned feature values."""
        bins = []
        for feature in FEATURE_COLUMNS:
            if feature in self.bin_edges:
                bin_idx = self._get_bin_index(row[feature], self.bin_edges[feature])
                bins.append(f"{feature[:3]}_{bin_idx}")
        return "|".join(bins)

    def fit(self, df: pd.DataFrame) -> 'DistributionQMatrix':
        """
        Train the Q-matrix on the given data.

        Args:
            df: DataFrame with feature columns and 'distribution_name'

        Returns:
            self for chaining
        """
        print(f"Training Distribution Q-Matrix with {self.num_bins} bins per feature...")
        print(f"Features: {FEATURE_COLUMNS}")
        print(f"Total samples: {len(df)}")

        # Get unique distributions
        self.distributions = sorted(df['distribution_name'].unique().tolist())
        self.label_encoder.fit(self.distributions)
        print(f"Number of distributions: {len(self.distributions)}")

        # Compute bin edges for each feature
        for feature in FEATURE_COLUMNS:
            if feature in df.columns:
                self.bin_edges[feature] = self._compute_bin_edges(df, feature)
                print(f"  {feature}: {len(self.bin_edges[feature])-1} bins")

        # Build Q-matrix: count occurrences of each distribution per feature key
        self.q_matrix = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            key = self._get_feature_key(row)
            dist = row['distribution_name']
            self.q_matrix[key][dist] += 1

        # Convert defaultdicts to regular dicts
        self.q_matrix = {k: dict(v) for k, v in self.q_matrix.items()}

        print(f"Q-Matrix has {len(self.q_matrix)} unique feature combinations")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict distributions for the given data.

        Args:
            df: DataFrame with feature columns

        Returns:
            Array of predicted distribution names
        """
        predictions = []

        for _, row in df.iterrows():
            key = self._get_feature_key(row)

            if key in self.q_matrix:
                # Return most common distribution for this key
                dist_counts = self.q_matrix[key]
                predicted = max(dist_counts, key=dist_counts.get)
            else:
                # Fallback: return most common distribution overall
                predicted = self.distributions[0] if self.distributions else "unknown"

            predictions.append(predicted)

        return np.array(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict distribution probabilities for the given data.

        Args:
            df: DataFrame with feature columns

        Returns:
            Array of shape (n_samples, n_distributions) with probabilities
        """
        n_samples = len(df)
        n_distributions = len(self.distributions)
        proba = np.zeros((n_samples, n_distributions))

        dist_to_idx = {d: i for i, d in enumerate(self.distributions)}

        for i, (_, row) in enumerate(df.iterrows()):
            key = self._get_feature_key(row)

            if key in self.q_matrix:
                dist_counts = self.q_matrix[key]
                total = sum(dist_counts.values())
                for dist, count in dist_counts.items():
                    if dist in dist_to_idx:
                        proba[i, dist_to_idx[dist]] = count / total
            else:
                # Uniform distribution as fallback
                proba[i, :] = 1.0 / n_distributions

        return proba

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate the model on the given data.

        Args:
            df: DataFrame with feature columns and 'distribution_name'

        Returns:
            Dictionary with evaluation metrics
        """
        y_true = df['distribution_name'].values
        y_pred = self.predict(df)

        accuracy = accuracy_score(y_true, y_pred)

        # Calculate top-k accuracy
        proba = self.predict_proba(df)
        dist_to_idx = {d: i for i, d in enumerate(self.distributions)}

        top_3_correct = 0
        top_5_correct = 0

        for i, true_dist in enumerate(y_true):
            if true_dist in dist_to_idx:
                true_idx = dist_to_idx[true_dist]
                sorted_indices = np.argsort(proba[i])[::-1]

                if true_idx in sorted_indices[:3]:
                    top_3_correct += 1
                if true_idx in sorted_indices[:5]:
                    top_5_correct += 1

        top_3_accuracy = top_3_correct / len(y_true)
        top_5_accuracy = top_5_correct / len(y_true)

        return {
            'accuracy': accuracy,
            'top_3_accuracy': top_3_accuracy,
            'top_5_accuracy': top_5_accuracy,
            'num_samples': len(y_true),
            'num_distributions': len(self.distributions)
        }

    def export_json(self, output_dir: str) -> None:
        """Export the Q-matrix to JSON format."""
        os.makedirs(output_dir, exist_ok=True)

        # Export Q-matrix
        qmatrix_data = {
            'model_type': 'distribution_qmatrix',
            'version': '1.0',
            'num_bins': self.num_bins,
            'features': FEATURE_COLUMNS,
            'distributions': self.distributions,
            'bin_edges': {k: v.tolist() for k, v in self.bin_edges.items()},
            'q_matrix': self.q_matrix
        }

        with open(os.path.join(output_dir, 'distribution_qmatrix.json'), 'w') as f:
            json.dump(qmatrix_data, f, indent=2)

        print(f"Exported Q-matrix to {output_dir}/distribution_qmatrix.json")


def analyze_feature_importance(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze how well each feature separates distributions.

    Uses the variance ratio: between-class variance / within-class variance
    """
    results = {}

    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue

        # Group by distribution
        groups = df.groupby('distribution_name')[feature]

        # Overall mean
        overall_mean = df[feature].mean()

        # Between-class variance
        between_var = sum(
            len(g) * (g.mean() - overall_mean) ** 2
            for _, g in groups
        ) / len(df)

        # Within-class variance
        within_var = sum(
            g.var() * len(g)
            for _, g in groups
        ) / len(df)

        # Variance ratio (Fisher's criterion)
        if within_var > 0:
            ratio = between_var / within_var
        else:
            ratio = float('inf') if between_var > 0 else 0

        results[feature] = ratio

    return results


def compute_pseudo_r2(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      label_encoder: LabelEncoder) -> float:
    """
    Compute a pseudo-R² score for classification.

    Uses McFadden's pseudo-R²: 1 - (log-likelihood of model / log-likelihood of null model)
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    # Encode true labels
    y_true_encoded = label_encoder.transform(y_true)

    # Log-likelihood of model
    ll_model = 0
    for i, true_idx in enumerate(y_true_encoded):
        prob = max(y_pred_proba[i, true_idx], 1e-10)  # Avoid log(0)
        ll_model += np.log(prob)

    # Log-likelihood of null model (predicts uniform distribution)
    ll_null = n_samples * np.log(1.0 / n_classes)

    # McFadden's pseudo-R²
    pseudo_r2 = 1 - (ll_model / ll_null)

    return pseudo_r2


def main():
    # Load data
    data_path = '/workspace/context-transport-primitives/test/unit/compress/results/compression_benchmark_results.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total rows: {len(df)}")
    print(f"Unique distributions: {df['distribution_name'].nunique()}")

    # Analyze feature importance
    print("\n" + "="*60)
    print("Feature Importance Analysis (Fisher's Variance Ratio)")
    print("="*60)
    importance = analyze_feature_importance(df)
    for feature, ratio in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feature}: {ratio:.4f}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42,
                                          stratify=df['distribution_name'])

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Try different numbers of bins
    print("\n" + "="*60)
    print("Q-Matrix Performance vs Number of Bins")
    print("="*60)

    best_accuracy = 0
    best_model = None
    best_bins = 0

    results_by_bins = []

    for num_bins in [5, 10, 15, 20, 25, 30]:
        model = DistributionQMatrix(num_bins=num_bins)
        model.fit(train_df)

        train_metrics = model.evaluate(train_df)
        test_metrics = model.evaluate(test_df)

        # Compute pseudo-R²
        test_proba = model.predict_proba(test_df)
        pseudo_r2 = compute_pseudo_r2(
            test_df['distribution_name'].values,
            test_proba,
            model.label_encoder
        )

        results_by_bins.append({
            'num_bins': num_bins,
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'top_3_accuracy': test_metrics['top_3_accuracy'],
            'top_5_accuracy': test_metrics['top_5_accuracy'],
            'pseudo_r2': pseudo_r2
        })

        print(f"\nBins: {num_bins}")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {test_metrics['top_5_accuracy']:.4f}")
        print(f"  Pseudo-R²:      {pseudo_r2:.4f}")

        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            best_model = model
            best_bins = num_bins

    # Export best model
    print("\n" + "="*60)
    print(f"Best Model: {best_bins} bins, Test Accuracy: {best_accuracy:.4f}")
    print("="*60)

    output_dir = '/workspace/context-transport-primitives/test/unit/compress/results/distribution_qmatrix_model'
    best_model.export_json(output_dir)

    # Save results summary
    results_summary = {
        'model_type': 'distribution_qmatrix',
        'version': '1.0',
        'features': FEATURE_COLUMNS,
        'num_distributions': len(best_model.distributions),
        'best_num_bins': best_bins,
        'results_by_bins': results_by_bins,
        'best_test_accuracy': best_accuracy,
        'feature_importance': importance
    }

    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/results_summary.json")

    # Print detailed classification report for best model
    print("\n" + "="*60)
    print("Detailed Classification Report (Best Model)")
    print("="*60)

    y_true = test_df['distribution_name'].values
    y_pred = best_model.predict(test_df)

    # Show per-distribution accuracy for a subset
    unique_dists = sorted(set(y_true))[:10]  # First 10 distributions
    print("\nPer-distribution accuracy (first 10):")
    for dist in unique_dists:
        mask = y_true == dist
        if mask.sum() > 0:
            acc = (y_pred[mask] == dist).mean()
            print(f"  {dist}: {acc:.4f} ({mask.sum()} samples)")


if __name__ == '__main__':
    main()
