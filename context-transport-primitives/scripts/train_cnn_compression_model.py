#!/usr/bin/env python3
"""
CNN-Based Compression Performance Prediction

This script trains multiple CNN architectures to predict compression performance
and compares them with the XGBoost tree-based model.

Model Input:
  - Compressor type (one-hot encoded)
  - Data type (char/float, one-hot encoded)
  - Chunk size (bytes)
  - Target CPU utilization (%)
  - Shannon entropy (bits/byte)
  - MAD (Mean Absolute Deviation)
  - Second derivative mean (data curvature)

Model Output:
  - Compression ratio
  - PSNR (for lossy compressors only)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("WARNING: TensorFlow not available. Please install: pip install tensorflow")
    print("Exiting...")
    sys.exit(1)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def load_data(lossless_csv, lossy_csv):
    """Load and combine lossless and lossy compression benchmark data."""
    print_header("LOADING DATA")

    # Load lossless data
    df_lossless = pd.read_csv(lossless_csv)
    print(f"✓ Loaded {len(df_lossless)} lossless records")

    # Load lossy data
    df_lossy = pd.read_csv(lossy_csv)
    print(f"✓ Loaded {len(df_lossy)} lossy records")

    # Combine datasets
    # Lossless data doesn't have PSNR/SNR, fill with sentinel value
    if 'PSNR (dB)' not in df_lossless.columns:
        df_lossless['PSNR (dB)'] = 999.0  # Sentinel for lossless
        df_lossless['SNR (dB)'] = 999.0

    df_combined = pd.concat([df_lossless, df_lossy], ignore_index=True)
    print(f"✓ Total records: {len(df_combined)}\n")

    return df_combined, df_lossy


def prepare_features(df):
    """Prepare feature matrix for CNN training."""
    print_header("FEATURE ENGINEERING")

    # One-hot encode library
    library_encoder = LabelEncoder()
    library_encoded = library_encoder.fit_transform(df['Library'])
    library_onehot = pd.get_dummies(df['Library'], prefix='Library')

    # One-hot encode data type (char vs float)
    # Infer data type from distribution name or explicit column
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

    # Target variables
    y_ratio = df['Compression Ratio'].values
    y_psnr = df['PSNR (dB)'].values

    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"✓ Features:")
    for i, col in enumerate(X.columns, 1):
        print(f"    {i}. {col}")

    return X, y_ratio, y_psnr, X.columns.tolist(), library_encoder


def create_cnn_model(input_dim, hidden_layers, dropout_rate, learning_rate, l2_reg):
    """
    Create a dense neural network for compression prediction on tabular data.

    Note: Despite the name "cnn_model", this uses a dense architecture which is
    more appropriate for tabular data with categorical features. CNNs work well for
    spatial/sequential data, but dense networks are better for mixed tabular features.

    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes (e.g., [128, 64, 32])
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        l2_reg: L2 regularization strength

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden dense layers with batch normalization and dropout
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    # Output layer (single neuron for regression)
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile model with AdamW optimizer (decoupled weight decay)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_reg),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def train_cnn_models(X_train, X_val, y_train, y_val, target_name):
    """Train optimized dense neural network with best hyperparameters."""
    print(f"\nTraining CNN model for {target_name}...")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")

    # Standardize features (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Best configuration for compression ratio: High learning rate (0.01)
    # Best configuration for PSNR: Low dropout (0.1)
    # Using task-specific optimal hyperparameters
    if 'PSNR' in target_name:
        # Optimized for PSNR prediction
        config = {
            'name': 'DenseNN-Optimized-PSNR',
            'hidden_layers': [128, 64, 32, 16],
            'dropout_rate': 0.1,  # Low dropout works best for PSNR
            'learning_rate': 0.001,
            'l2_reg': 0.0001,
            'batch_size': 16,
            'epochs': 200
        }
    else:
        # Optimized for compression ratio prediction
        config = {
            'name': 'DenseNN-Optimized-Ratio',
            'hidden_layers': [128, 64, 32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.01,  # High learning rate works best for ratio
            'l2_reg': 0.0001,
            'batch_size': 16,
            'epochs': 200
        }

    print(f"\n  Architecture: {config['hidden_layers']}")
    print(f"  Dropout: {config['dropout_rate']}, LR: {config['learning_rate']}, Batch: {config['batch_size']}")

    # Create model
    model = create_cnn_model(
        input_dim=X_train_scaled.shape[1],
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate'],
        l2_reg=config['l2_reg']
    )

    # Early stopping callback with reduced learning rate on plateau
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    # Train model
    print(f"  Training for up to {config['epochs']} epochs...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    val_pred = model.predict(X_val_scaled, verbose=0).flatten()

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    print(f"\n  Final Performance:")
    print(f"    Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"    Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

    results = [{
        'name': config['name'],
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'history': history.history
    }]

    return model, scaler, config, results


def train_compression_ratio_model(X, y, feature_names):
    """Train CNN models for compression ratio prediction."""
    print_header("TRAINING COMPRESSION RATIO CNN")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train CNN models
    model, scaler, config, results = train_cnn_models(
        X_train, X_val, y_train, y_val, "Compression Ratio"
    )

    # Final evaluation
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    val_pred = model.predict(X_val_scaled, verbose=0).flatten()

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    print("\nBest Model Performance:")
    print(f"  Training RMSE:   {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    print(f"  Training MAE:    {train_mae:.4f}")
    print(f"  Validation MAE:  {val_mae:.4f}")
    print(f"  Training R²:     {train_r2:.4f}")
    print(f"  Validation R²:   {val_r2:.4f}")

    # Check for overfitting
    if train_r2 - val_r2 > 0.1:
        print("\n⚠️  WARNING: Potential overfitting detected")
    else:
        print("\n✓ No significant overfitting detected")

    return model, scaler, config, results, (train_pred, val_pred, y_train, y_val)


def train_psnr_model(X, y, feature_names, df_lossy):
    """Train CNN models for PSNR prediction (lossy compressors only)."""
    print_header("TRAINING PSNR CNN (Lossy Compressors Only)")

    # Filter for lossy data only
    lossy_mask = y < 999  # Sentinel value for lossless
    X_lossy = X[lossy_mask]
    y_lossy = y[lossy_mask]

    print(f"✓ Using {len(X_lossy)} lossy compression records\n")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_lossy, y_lossy, test_size=0.2, random_state=42
    )

    # Train CNN models
    model, scaler, config, results = train_cnn_models(
        X_train, X_val, y_train, y_val, "PSNR"
    )

    # Final evaluation
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    val_pred = model.predict(X_val_scaled, verbose=0).flatten()

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    print("\nBest Model Performance:")
    print(f"  Training RMSE:   {train_rmse:.4f} dB")
    print(f"  Validation RMSE: {val_rmse:.4f} dB")
    print(f"  Training MAE:    {train_mae:.4f} dB")
    print(f"  Validation MAE:  {val_mae:.4f} dB")
    print(f"  Training R²:     {train_r2:.4f}")
    print(f"  Validation R²:   {val_r2:.4f}")

    # Check for overfitting
    if train_r2 - val_r2 > 0.1:
        print("\n⚠️  WARNING: Potential overfitting detected")
    else:
        print("\n✓ No significant overfitting detected")

    return model, scaler, config, results, (train_pred, val_pred, y_train, y_val)


def plot_predictions(train_pred, val_pred, y_train, y_val, title, output_path):
    """Plot predicted vs actual values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training set
    ax1.scatter(y_train, train_pred, alpha=0.5, s=20)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'{title} - Training Set')
    ax1.grid(True, alpha=0.3)

    # Validation set
    ax2.scatter(y_val, val_pred, alpha=0.5, s=20, color='orange')
    ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'{title} - Validation Set')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_training_history(results, title, output_path):
    """Plot training history for all CNN architectures."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, result in enumerate(results):
        if i >= len(axes):
            break

        ax = axes[i]
        history = result['history']

        # Plot loss
        ax.plot(history['loss'], label='Training Loss', alpha=0.8)
        ax.plot(history['val_loss'], label='Validation Loss', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f"{result['name']}\nVal R²: {result['val_r2']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def compare_with_xgboost(output_dir):
    """Load XGBoost results and compare with CNN."""
    print_header("COMPARING CNN vs XGBOOST")

    # Try to load XGBoost results
    xgb_metadata_path = output_dir / 'model_metadata.json'

    if not xgb_metadata_path.exists():
        print("⚠️  XGBoost model metadata not found. Run train_compression_model.py first.")
        print(f"    Expected: {xgb_metadata_path}")
        return

    with open(xgb_metadata_path, 'r') as f:
        xgb_metadata = json.load(f)

    print("XGBoost Model Performance:")
    print(f"  Compression Ratio - Validation R²: {xgb_metadata.get('compression_ratio_val_r2', 'N/A')}")
    print(f"  PSNR - Validation R²: {xgb_metadata.get('psnr_val_r2', 'N/A')}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python train_cnn_compression_model.py <lossless_csv> <lossy_csv>")
        print("\nExample:")
        print("  python train_cnn_compression_model.py \\")
        print("    compression_parameter_study_results.csv \\")
        print("    compression_lossy_parameter_study_results.csv")
        sys.exit(1)

    lossless_csv = sys.argv[1]
    lossy_csv = sys.argv[2]

    # Create output directory
    output_dir = Path(__file__).parent / 'cnn_model_output'
    output_dir.mkdir(exist_ok=True)

    print_header("CNN-BASED COMPRESSION PERFORMANCE PREDICTION")
    print(f"Lossless data: {lossless_csv}")
    print(f"Lossy data: {lossy_csv}")
    print(f"Output directory: {output_dir}")

    # Load data
    df_combined, df_lossy = load_data(lossless_csv, lossy_csv)

    # Prepare features
    X, y_ratio, y_psnr, feature_names, library_encoder = prepare_features(df_combined)

    # Train compression ratio model
    ratio_model, ratio_scaler, ratio_config, ratio_results, ratio_preds = \
        train_compression_ratio_model(X.values, y_ratio, feature_names)

    # Train PSNR model
    psnr_model, psnr_scaler, psnr_config, psnr_results, psnr_preds = \
        train_psnr_model(X.values, y_psnr, feature_names, df_lossy)

    # Generate visualizations
    print_header("GENERATING VISUALIZATIONS")

    plot_predictions(
        ratio_preds[0], ratio_preds[1], ratio_preds[2], ratio_preds[3],
        "Compression Ratio CNN Predictions",
        output_dir / "cnn_compression_ratio_predictions.pdf"
    )

    plot_predictions(
        psnr_preds[0], psnr_preds[1], psnr_preds[2], psnr_preds[3],
        "PSNR CNN Predictions",
        output_dir / "cnn_psnr_predictions.pdf"
    )

    plot_training_history(
        ratio_results,
        "Compression Ratio CNN - Training History",
        output_dir / "cnn_compression_ratio_training_history.pdf"
    )

    plot_training_history(
        psnr_results,
        "PSNR CNN - Training History",
        output_dir / "cnn_psnr_training_history.pdf"
    )

    # Save models
    print_header("SAVING MODELS")

    # Save Keras models
    ratio_model.save(output_dir / 'cnn_compression_ratio_model.keras')
    print(f"✓ Saved compression ratio model: cnn_compression_ratio_model.keras")

    psnr_model.save(output_dir / 'cnn_psnr_model.keras')
    print(f"✓ Saved PSNR model: cnn_psnr_model.keras")

    # Save scalers
    joblib.dump(ratio_scaler, output_dir / 'cnn_compression_ratio_scaler.pkl')
    joblib.dump(psnr_scaler, output_dir / 'cnn_psnr_scaler.pkl')
    print(f"✓ Saved scalers: cnn_compression_ratio_scaler.pkl, cnn_psnr_scaler.pkl")

    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'ratio_config': ratio_config,
        'psnr_config': psnr_config,
        'ratio_results': [
            {k: v for k, v in r.items() if k != 'history'}
            for r in ratio_results
        ],
        'psnr_results': [
            {k: v for k, v in r.items() if k != 'history'}
            for r in psnr_results
        ]
    }

    with open(output_dir / 'cnn_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: cnn_model_metadata.json")

    # Compare with XGBoost
    compare_with_xgboost(Path(__file__).parent / 'model_output')

    print_header("CNN TRAINING COMPLETE")

    print("\nGenerated files:")
    print(f"  • cnn_compression_ratio_model.keras - Keras model for compression ratio")
    print(f"  • cnn_psnr_model.keras - Keras model for PSNR (lossy only)")
    print(f"  • cnn_compression_ratio_scaler.pkl - Feature scaler for ratio model")
    print(f"  • cnn_psnr_scaler.pkl - Feature scaler for PSNR model")
    print(f"  • cnn_model_metadata.json - Model configurations and results")
    print(f"  • cnn_compression_ratio_predictions.pdf - Prediction visualizations")
    print(f"  • cnn_psnr_predictions.pdf - PSNR prediction visualizations")
    print(f"  • cnn_compression_ratio_training_history.pdf - Training curves")
    print(f"  • cnn_psnr_training_history.pdf - PSNR training curves")

    print("\nModels can be loaded for inference using:")
    print("  import tensorflow as tf")
    print("  import joblib")
    print("  ")
    print("  # Load models")
    print("  ratio_model = tf.keras.models.load_model('cnn_compression_ratio_model.keras')")
    print("  psnr_model = tf.keras.models.load_model('cnn_psnr_model.keras')")
    print("  ratio_scaler = joblib.load('cnn_compression_ratio_scaler.pkl')")
    print("  ")
    print("  # Prepare features and predict")
    print("  X_scaled = ratio_scaler.transform(features)")
    print("  ratio_pred = ratio_model.predict(X_scaled)")


if __name__ == "__main__":
    main()
