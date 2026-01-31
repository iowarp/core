#!/usr/bin/env python3
"""
Distribution Predictor using Block Sampling Features.

This script trains a classifier to predict data distribution using
the new block sampling features from the compression benchmark.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import json
import os
from typing import Dict, List

# Original features (computed on full data)
ORIGINAL_FEATURES = [
    'shannon_entropy',
    'mean_absolute_deviation',
    'first_order_derivative',
    'second_order_derivative'
]

# New block sampling features
BLOCK_FEATURES = [
    'block_entropy_mean',
    'block_entropy_std',
    'block_mad_mean',
    'block_mad_std',
    'block_skewness',
    'block_kurtosis',
    'block_deriv1_mean',
    'block_deriv1_std',
    'block_value_range',
    'block_value_concentration'
]

# Combined feature sets for comparison
FEATURE_SETS = {
    'original_only': ORIGINAL_FEATURES,
    'block_only': BLOCK_FEATURES,
    'combined': ORIGINAL_FEATURES + BLOCK_FEATURES,
}


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       feature_cols: List[str], name: str) -> Dict:
    """Train Random Forest and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Features: {len(feature_cols)}")
    print(f"{'='*60}")

    # Prepare target
    le_dist = LabelEncoder()
    y_train = le_dist.fit_transform(train_df['distribution_name'])
    y_test = le_dist.transform(test_df['distribution_name'])

    # Prepare features
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

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
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    top_3_acc = top_k_accuracy_score(y_test, y_proba, k=3)
    top_5_acc = top_k_accuracy_score(y_test, y_proba, k=5)

    # Feature importance
    importances = dict(zip(feature_cols, rf.feature_importances_))

    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Top-3 Accuracy: {top_3_acc:.4f}")
    print(f"Top-5 Accuracy: {top_5_acc:.4f}")

    print("\nTop 5 Feature Importances:")
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])[:5]
    for feat, imp in sorted_imp:
        print(f"  {feat}: {imp:.4f}")

    return {
        'name': name,
        'num_features': len(feature_cols),
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

    # Check if block features exist
    has_block_features = all(col in df.columns for col in BLOCK_FEATURES)

    if not has_block_features:
        print("\nERROR: Block sampling features not found in CSV!")
        print("Please run the updated compression benchmark first.")
        print("\nExpected columns:")
        for col in BLOCK_FEATURES:
            status = "✓" if col in df.columns else "✗"
            print(f"  {status} {col}")
        return

    print(f"Total rows: {len(df)}")
    print(f"Unique distributions: {df['distribution_name'].nunique()}")

    # Get unique samples (one per distribution/data_size/data_type combination)
    # This avoids having the same data profiled with different compressors
    unique_df = df.drop_duplicates(subset=['distribution_name', 'data_size', 'data_type'])
    print(f"Unique data samples: {len(unique_df)}")

    # Split data
    train_df, test_df = train_test_split(
        unique_df, test_size=0.3, random_state=42,
        stratify=unique_df['distribution_name']
    )

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Train and evaluate each feature set
    results = []

    for name, features in FEATURE_SETS.items():
        # Check if all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"\nSkipping {name}: missing features {missing}")
            continue

        result = train_and_evaluate(train_df, test_df, features, name)
        results.append(result)

    # Summary comparison
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Feature Set':<20} {'Features':<10} {'Accuracy':<12} {'Top-3':<12} {'Top-5':<12}")
    print("-" * 66)

    for r in results:
        print(f"{r['name']:<20} {r['num_features']:<10} {r['accuracy']:<12.4f} "
              f"{r['top_3_accuracy']:<12.4f} {r['top_5_accuracy']:<12.4f}")

    # Save results
    output_dir = '/workspace/context-transport-primitives/test/unit/compress/results/distribution_block_model'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/comparison_results.json")


if __name__ == '__main__':
    main()
