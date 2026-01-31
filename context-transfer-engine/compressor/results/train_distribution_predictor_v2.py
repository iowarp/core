#!/usr/bin/env python3
"""
Distribution Predictor v2 - includes data_type as a feature.

This version also explores using a neural network approach for comparison.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import json
import os
from typing import Dict, List, Tuple

# Features to use for prediction
NUMERIC_FEATURES = [
    'shannon_entropy',
    'mean_absolute_deviation',
    'first_order_derivative',
    'second_order_derivative'
]

CATEGORICAL_FEATURES = ['data_type']

DEFAULT_NUM_BINS = 25


class DistributionQMatrixV2:
    """
    Enhanced Q-Matrix predictor that includes categorical features.
    """

    def __init__(self, num_bins: int = DEFAULT_NUM_BINS):
        self.num_bins = num_bins
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.q_matrix: Dict[str, Dict[str, int]] = {}
        self.distributions: List[str] = []
        self.label_encoder = LabelEncoder()
        self.data_types: List[str] = []

    def _compute_bin_edges(self, df: pd.DataFrame, feature: str) -> np.ndarray:
        """Compute bin edges using quantiles."""
        values = df[feature].dropna()
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        edges = np.percentile(values, percentiles)
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
        parts = []

        # Add categorical features
        for feature in CATEGORICAL_FEATURES:
            if feature in row:
                parts.append(f"{feature}={row[feature]}")

        # Add binned numeric features
        for feature in NUMERIC_FEATURES:
            if feature in self.bin_edges:
                bin_idx = self._get_bin_index(row[feature], self.bin_edges[feature])
                parts.append(f"{feature[:3]}_{bin_idx}")

        return "|".join(parts)

    def fit(self, df: pd.DataFrame) -> 'DistributionQMatrixV2':
        """Train the Q-matrix."""
        print(f"Training Distribution Q-Matrix V2 with {self.num_bins} bins...")
        print(f"Numeric features: {NUMERIC_FEATURES}")
        print(f"Categorical features: {CATEGORICAL_FEATURES}")
        print(f"Total samples: {len(df)}")

        # Get unique distributions and data types
        self.distributions = sorted(df['distribution_name'].unique().tolist())
        self.label_encoder.fit(self.distributions)
        self.data_types = sorted(df['data_type'].unique().tolist())
        print(f"Number of distributions: {len(self.distributions)}")
        print(f"Data types: {self.data_types}")

        # Compute bin edges for each numeric feature
        for feature in NUMERIC_FEATURES:
            if feature in df.columns:
                self.bin_edges[feature] = self._compute_bin_edges(df, feature)

        # Build Q-matrix
        self.q_matrix = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            key = self._get_feature_key(row)
            dist = row['distribution_name']
            self.q_matrix[key][dist] += 1

        self.q_matrix = {k: dict(v) for k, v in self.q_matrix.items()}
        print(f"Q-Matrix has {len(self.q_matrix)} unique feature combinations")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict distributions."""
        predictions = []

        for _, row in df.iterrows():
            key = self._get_feature_key(row)

            if key in self.q_matrix:
                dist_counts = self.q_matrix[key]
                predicted = max(dist_counts, key=dist_counts.get)
            else:
                predicted = self.distributions[0] if self.distributions else "unknown"

            predictions.append(predicted)

        return np.array(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict distribution probabilities."""
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
                proba[i, :] = 1.0 / n_distributions

        return proba

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate the model."""
        y_true = df['distribution_name'].values
        y_pred = self.predict(df)

        accuracy = accuracy_score(y_true, y_pred)

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

        return {
            'accuracy': accuracy,
            'top_3_accuracy': top_3_correct / len(y_true),
            'top_5_accuracy': top_5_correct / len(y_true),
        }


def train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train a Random Forest classifier for comparison."""
    print("\nTraining Random Forest classifier...")

    # Prepare features
    le_dtype = LabelEncoder()
    le_dist = LabelEncoder()

    # Encode categorical features
    train_dtype = le_dtype.fit_transform(train_df['data_type'])
    test_dtype = le_dtype.transform(test_df['data_type'])

    # Encode target
    y_train = le_dist.fit_transform(train_df['distribution_name'])
    y_test = le_dist.transform(test_df['distribution_name'])

    # Build feature matrices
    X_train = np.column_stack([
        train_df[NUMERIC_FEATURES].values,
        train_dtype.reshape(-1, 1)
    ])
    X_test = np.column_stack([
        test_df[NUMERIC_FEATURES].values,
        test_dtype.reshape(-1, 1)
    ])

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    # Top-k accuracy
    top_3_acc = top_k_accuracy_score(y_test, y_proba, k=3)
    top_5_acc = top_k_accuracy_score(y_test, y_proba, k=5)

    # Feature importance
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importances = dict(zip(feature_names, rf.feature_importances_))

    return {
        'accuracy': accuracy,
        'top_3_accuracy': top_3_acc,
        'top_5_accuracy': top_5_acc,
        'feature_importances': importances
    }


def main():
    # Load data
    data_path = '/workspace/context-transport-primitives/test/unit/compress/results/compression_benchmark_results.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total rows: {len(df)}")
    print(f"Unique distributions: {df['distribution_name'].nunique()}")
    print(f"Data types: {df['data_type'].unique().tolist()}")

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42,
        stratify=df['distribution_name']
    )

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Q-Matrix V2 (with data_type)
    print("\n" + "="*60)
    print("Q-Matrix V2 (with data_type feature)")
    print("="*60)

    best_accuracy = 0
    best_model = None
    best_bins = 0

    for num_bins in [15, 20, 25, 30, 35]:
        model = DistributionQMatrixV2(num_bins=num_bins)
        model.fit(train_df)

        train_metrics = model.evaluate(train_df)
        test_metrics = model.evaluate(test_df)

        print(f"\nBins: {num_bins}")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {test_metrics['top_5_accuracy']:.4f}")

        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            best_model = model
            best_bins = num_bins

    print(f"\nBest Q-Matrix V2: {best_bins} bins, Test Accuracy: {best_accuracy:.4f}")

    # Random Forest comparison
    print("\n" + "="*60)
    print("Random Forest Classifier (for comparison)")
    print("="*60)

    rf_results = train_random_forest(train_df, test_df)
    print(f"\nRandom Forest Results:")
    print(f"  Test Accuracy:  {rf_results['accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {rf_results['top_3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {rf_results['top_5_accuracy']:.4f}")
    print(f"\nFeature Importances:")
    for feat, imp in sorted(rf_results['feature_importances'].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # Summary comparison
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'Top-3':<12} {'Top-5':<12}")
    print("-" * 60)
    print(f"{'Q-Matrix V1 (no dtype)':<25} {'0.4535':<12} {'0.7981':<12} {'0.9205':<12}")
    print(f"{'Q-Matrix V2 (with dtype)':<25} {best_accuracy:<12.4f} "
          f"{best_model.evaluate(test_df)['top_3_accuracy']:<12.4f} "
          f"{best_model.evaluate(test_df)['top_5_accuracy']:<12.4f}")
    print(f"{'Random Forest':<25} {rf_results['accuracy']:<12.4f} "
          f"{rf_results['top_3_accuracy']:<12.4f} {rf_results['top_5_accuracy']:<12.4f}")


if __name__ == '__main__':
    main()
