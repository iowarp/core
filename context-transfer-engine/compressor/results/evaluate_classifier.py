#!/usr/bin/env python3
"""
Evaluate the mathematical distribution classifier against ground truth.

This script applies the same mathematical rules as the C++ classifier
to the benchmark data and measures accuracy.
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

def compute_moments(values):
    """Compute mean, variance, skewness, kurtosis from data."""
    n = len(values)
    if n < 4:
        return 0, 0, 0, 0

    mean = np.mean(values)
    variance = np.var(values)

    if variance < 1e-10:
        return mean, variance, 0, 0

    # Standardized moments
    std = np.sqrt(variance)
    centered = values - mean
    skewness = np.mean((centered / std) ** 3)
    kurtosis = np.mean((centered / std) ** 4) - 3  # Excess kurtosis

    return mean, variance, skewness, kurtosis


def classify_by_moments(skewness, kurtosis):
    """
    Classify distribution based on skewness and kurtosis.

    Thresholds calibrated on synthetic binned distributions:
      Observed moments:
      - Uniform:     skew≈0.1,  kurt≈-0.75
      - Normal:      skew≈0.17, kurt≈-0.30
      - Gamma:       skew≈0.66, kurt≈-0.02
      - Exponential: skew≈2.9,  kurt≈8.2
    """
    # Exponential: strongly right-skewed
    if skewness > 1.5:
        return 'exponential'

    # Gamma: moderately right-skewed
    if skewness > 0.4:
        return 'gamma'

    # Uniform vs Normal: both symmetric
    # Uniform has more negative kurtosis
    if kurtosis < -0.5:
        return 'uniform'

    # Normal (default for symmetric, mesokurtic)
    return 'normal'


def extract_base_distribution(dist_name):
    """Extract base distribution type from full name like 'binned_normal_w0_p0'."""
    if 'uniform' in dist_name:
        return 'uniform'
    elif 'normal' in dist_name:
        return 'normal'
    elif 'gamma' in dist_name:
        return 'gamma'
    elif 'exponential' in dist_name:
        return 'exponential'
    return 'unknown'


def main():
    # Load benchmark data
    data_path = '/workspace/context-transport-primitives/test/unit/compress/results/compression_benchmark_results.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total rows: {len(df)}")

    # Get unique data samples (one per distribution/data_size/data_type)
    unique_df = df.drop_duplicates(subset=['distribution_name', 'data_size', 'data_type'])
    print(f"Unique data samples: {len(unique_df)}")

    # Extract ground truth base distribution
    unique_df = unique_df.copy()
    unique_df['true_base'] = unique_df['distribution_name'].apply(extract_base_distribution)

    # Use block sampling features to compute moments
    # skewness and kurtosis are directly available
    if 'block_skewness' in unique_df.columns:
        print("\nUsing block sampling features for classification...")
        unique_df['predicted'] = unique_df.apply(
            lambda row: classify_by_moments(row['block_skewness'], row['block_kurtosis']),
            axis=1
        )
    else:
        print("\nBlock sampling features not available, using entropy/MAD heuristics...")
        # Fallback: use simple heuristics based on available features
        def heuristic_classify(row):
            entropy = row['shannon_entropy']
            mad = row['mean_absolute_deviation']
            deriv1 = row['first_order_derivative']

            # High entropy + high MAD + high derivative = likely uniform
            if entropy > 7.5 and mad > 50:
                return 'uniform'
            # Low entropy + low MAD = likely exponential (concentrated)
            if entropy < 5 and mad < 30:
                return 'exponential'
            # Medium entropy = normal or gamma
            if entropy > 6:
                return 'normal'
            return 'gamma'

        unique_df['predicted'] = unique_df.apply(heuristic_classify, axis=1)

    # Calculate accuracy
    correct = (unique_df['predicted'] == unique_df['true_base']).sum()
    total = len(unique_df)
    accuracy = correct / total

    print(f"\n{'='*60}")
    print("Classification Results (4 base distribution types)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{total})")

    # Per-distribution accuracy
    print(f"\nPer-Distribution Accuracy:")
    for dist in ['uniform', 'normal', 'gamma', 'exponential']:
        mask = unique_df['true_base'] == dist
        if mask.sum() > 0:
            dist_correct = (unique_df.loc[mask, 'predicted'] == dist).sum()
            dist_total = mask.sum()
            print(f"  {dist:12s}: {dist_correct/dist_total:.1%} ({dist_correct}/{dist_total})")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    confusion = defaultdict(lambda: defaultdict(int))
    for _, row in unique_df.iterrows():
        confusion[row['true_base']][row['predicted']] += 1

    # Print header
    dists = ['uniform', 'normal', 'gamma', 'exponential']
    print(f"{'True \\ Pred':<14}", end='')
    for d in dists:
        print(f"{d:>12}", end='')
    print()

    for true_dist in dists:
        print(f"{true_dist:<14}", end='')
        for pred_dist in dists:
            count = confusion[true_dist][pred_dist]
            print(f"{count:>12}", end='')
        print()

    # Analyze misclassifications
    print(f"\nMost Common Misclassifications:")
    misclass = unique_df[unique_df['predicted'] != unique_df['true_base']]
    if len(misclass) > 0:
        misclass_pairs = misclass.groupby(['true_base', 'predicted']).size().sort_values(ascending=False)
        for (true, pred), count in misclass_pairs.head(5).items():
            pct = count / len(unique_df) * 100
            print(f"  {true} → {pred}: {count} ({pct:.1f}%)")

    # Moment statistics by distribution
    if 'block_skewness' in unique_df.columns:
        print(f"\nMoment Statistics by True Distribution:")
        print(f"{'Distribution':<14} {'Skewness':>12} {'Kurtosis':>12}")
        for dist in dists:
            mask = unique_df['true_base'] == dist
            if mask.sum() > 0:
                skew_mean = unique_df.loc[mask, 'block_skewness'].mean()
                kurt_mean = unique_df.loc[mask, 'block_kurtosis'].mean()
                print(f"{dist:<14} {skew_mean:>12.3f} {kurt_mean:>12.3f}")

        print(f"\nTheoretical Moments:")
        print(f"  Uniform:     skewness=0.00,  kurtosis=-1.20")
        print(f"  Normal:      skewness=0.00,  kurtosis= 0.00")
        print(f"  Gamma(k=2):  skewness=1.41,  kurtosis= 3.00")
        print(f"  Exponential: skewness=2.00,  kurtosis= 6.00")


if __name__ == '__main__':
    main()
