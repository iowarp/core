#!/usr/bin/env python3
"""
General Distribution Generator for Compression Benchmarks

Generates data with controlled distributions using a bin-based approach:
- n bins, each with a target percentage and value range
- Perturbation parameter controls spatial correlation (repeat probability)
- Can generate uniform, normal, gamma, constant-like distributions
"""

import numpy as np
from scipy import stats
import time

def generate_binned_distribution(
    size: int,
    bin_percentages: np.ndarray,
    bin_ranges: list,
    perturbation: float = 0.0,
    dtype=np.float64,
    seed: int = None
) -> np.ndarray:
    """
    Generate data using bin-based distribution with perturbation.

    Args:
        size: Total number of values to generate
        bin_percentages: Array of percentages for each bin (must sum to 1.0)
        bin_ranges: List of (min, max) tuples for each bin's value range
        perturbation: Probability of repeating the same bin (0.0 = no correlation)
        dtype: Output data type
        seed: Random seed for reproducibility

    Returns:
        Array of generated values
    """
    if seed is not None:
        np.random.seed(seed)

    n_bins = len(bin_percentages)
    assert len(bin_ranges) == n_bins
    assert abs(sum(bin_percentages) - 1.0) < 1e-6, f"Percentages must sum to 1.0, got {sum(bin_percentages)}"

    # Calculate target count for each bin
    bin_targets = (bin_percentages * size).astype(int)
    # Adjust for rounding errors
    bin_targets[-1] += size - bin_targets.sum()

    bin_counts = np.zeros(n_bins, dtype=int)
    result = np.zeros(size, dtype=dtype)

    # Track which bins are still active
    active_bins = set(range(n_bins))

    # Start with a random bin
    current_bin = np.random.choice(list(active_bins))

    for i in range(size):
        # Generate value from current bin
        lo, hi = bin_ranges[current_bin]
        if lo == hi:
            result[i] = lo
        else:
            result[i] = np.random.uniform(lo, hi)

        bin_counts[current_bin] += 1

        # Check if bin reached its quota
        if bin_counts[current_bin] >= bin_targets[current_bin]:
            active_bins.discard(current_bin)

        if not active_bins:
            break

        # Choose next bin
        if np.random.random() < perturbation and current_bin in active_bins:
            # Repeat same bin
            pass
        else:
            # Choose different bin uniformly from remaining active bins
            other_bins = active_bins - {current_bin}
            if other_bins:
                current_bin = np.random.choice(list(other_bins))
            elif active_bins:
                current_bin = np.random.choice(list(active_bins))

    return result


def create_uniform_percentages(n_bins: int) -> np.ndarray:
    """Each bin gets equal percentage."""
    return np.ones(n_bins) / n_bins


def create_normal_percentages(n_bins: int, std_bins: float = 2.0) -> np.ndarray:
    """Percentages follow normal distribution centered on middle bins."""
    x = np.linspace(-3, 3, n_bins)
    percentages = stats.norm.pdf(x, 0, std_bins / 3 * 2)
    return percentages / percentages.sum()


def create_gamma_percentages(n_bins: int, shape: float = 2.0) -> np.ndarray:
    """Percentages follow gamma distribution (skewed toward lower bins)."""
    x = np.linspace(0.1, 6, n_bins)
    percentages = stats.gamma.pdf(x, shape)
    return percentages / percentages.sum()


def create_bin_ranges(n_bins: int, bin_width: int, value_range: tuple = (0, 255)) -> list:
    """
    Create bin ranges based on bin width.

    Args:
        n_bins: Number of bins
        bin_width: Width of each bin in value units
        value_range: (min, max) of overall value range

    Returns:
        List of (lo, hi) tuples for each bin
    """
    lo, hi = value_range
    total_range = hi - lo

    # Calculate bin boundaries
    if bin_width * n_bins <= total_range:
        # Bins fit within range - space them out
        gap = (total_range - bin_width * n_bins) / (n_bins - 1) if n_bins > 1 else 0
        ranges = []
        current = lo
        for i in range(n_bins):
            bin_lo = current
            bin_hi = current + bin_width
            ranges.append((bin_lo, min(bin_hi, hi)))
            current = bin_hi + gap
    else:
        # Bins overlap - distribute evenly
        step = total_range / n_bins
        ranges = []
        for i in range(n_bins):
            bin_lo = lo + i * step
            bin_hi = min(bin_lo + bin_width, hi)
            ranges.append((bin_lo, bin_hi))

    return ranges


def compute_entropy(arr: np.ndarray) -> float:
    """Compute byte-level Shannon entropy."""
    data = arr.astype(np.float64).tobytes()
    byte_arr = np.frombuffer(data, dtype=np.uint8)
    _, counts = np.unique(byte_arr, return_counts=True)
    probs = counts / len(byte_arr)
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_mad(arr: np.ndarray) -> float:
    """Compute mean absolute deviation."""
    return np.mean(np.abs(arr - np.mean(arr)))


# =============================================================================
# MAIN GENERATION SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GENERAL DISTRIBUTION GENERATOR - PARAMETER SWEEP")
    print("=" * 80)
    print()

    # Parameters
    N_BINS = 16
    BIN_WIDTHS = [1, 16, 64, 128, 256]
    PERTURBATIONS = [0.0, 0.05, 0.1, 0.25, 0.325, 0.5]
    DISTRIBUTION_TYPES = ['uniform', 'normal', 'gamma']

    SIZE = 128 * 128 * 128  # 2M values (like Gray-Scott chunk)
    VALUE_RANGE = (0, 255)  # Byte-compatible range

    print(f"Configuration:")
    print(f"  N_BINS: {N_BINS}")
    print(f"  BIN_WIDTHS: {BIN_WIDTHS}")
    print(f"  PERTURBATIONS: {PERTURBATIONS}")
    print(f"  DISTRIBUTION_TYPES: {DISTRIBUTION_TYPES}")
    print(f"  SIZE: {SIZE:,} values")
    print(f"  VALUE_RANGE: {VALUE_RANGE}")
    print()

    # Generate percentage distributions
    percentages = {
        'uniform': create_uniform_percentages(N_BINS),
        'normal': create_normal_percentages(N_BINS),
        'gamma': create_gamma_percentages(N_BINS),
    }

    print("Bin Percentages:")
    for name, pct in percentages.items():
        print(f"  {name}: {np.round(pct * 100, 1)}")
    print()

    # Run parameter sweep
    print("=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)
    print()
    print(f"{'Dist':<8} {'Width':>6} {'Perturb':>8} {'Entropy':>10} {'MAD':>12} {'Unique%':>10}")
    print(f"{'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

    results = []

    for dist_type in DISTRIBUTION_TYPES:
        pct = percentages[dist_type]

        for bin_width in BIN_WIDTHS:
            bin_ranges = create_bin_ranges(N_BINS, bin_width, VALUE_RANGE)

            for perturbation in PERTURBATIONS:
                arr = generate_binned_distribution(
                    size=SIZE,
                    bin_percentages=pct,
                    bin_ranges=bin_ranges,
                    perturbation=perturbation,
                    seed=42
                )

                entropy = compute_entropy(arr)
                mad = compute_mad(arr)
                unique_pct = len(np.unique(arr)) / len(arr) * 100

                print(f"{dist_type:<8} {bin_width:>6} {perturbation:>8.3f} {entropy:>10.4f} {mad:>12.4f} {unique_pct:>9.2f}%")

                results.append({
                    'distribution': dist_type,
                    'bin_width': bin_width,
                    'perturbation': perturbation,
                    'entropy': entropy,
                    'mad': mad,
                    'unique_pct': unique_pct
                })

        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    import pandas as pd
    df = pd.DataFrame(results)

    print("Entropy range by distribution type:")
    for dist in DISTRIBUTION_TYPES:
        subset = df[df['distribution'] == dist]
        print(f"  {dist}: {subset['entropy'].min():.4f} - {subset['entropy'].max():.4f}")

    print()
    print("Entropy range by bin width:")
    for width in BIN_WIDTHS:
        subset = df[df['bin_width'] == width]
        print(f"  width={width}: {subset['entropy'].min():.4f} - {subset['entropy'].max():.4f}")

    print()
    print(f"Total configurations: {len(results)}")
    print(f"Entropy range achieved: {df['entropy'].min():.4f} - {df['entropy'].max():.4f}")
