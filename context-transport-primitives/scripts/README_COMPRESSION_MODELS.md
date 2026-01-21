# Compression Performance Prediction Models

This directory contains machine learning models for predicting compression performance (compression ratio and PSNR) based on data characteristics and system parameters.

## Overview

Two types of models are provided:
1. **XGBoost Models** (Recommended) - Tree-based gradient boosting models
2. **Dense Neural Networks** - Deep learning models for comparison

### Model Performance Comparison

| Model | Compression Ratio R² | PSNR R² | Inference Speed |
|-------|---------------------|---------|-----------------|
| XGBoost | 0.9949 | 0.9999 | 0.04-0.09 ms/sample |
| Dense NN | 0.9900 | 0.9913 | 2.3-2.5 ms/sample |

**Recommendation**: Use XGBoost models for production deployment due to superior accuracy and 25-60× faster inference.

## Prerequisites

### Required Packages

```bash
# Core dependencies
pip install numpy pandas scikit-learn xgboost

# For neural network models
pip install tensorflow

# For visualization
pip install matplotlib seaborn
```

## Directory Structure

```
scripts/
├── train_compression_model.py          # Train XGBoost models
├── train_cnn_compression_model.py      # Train dense neural network models
├── compare_cnn_vs_xgboost.py          # Compare model performance
├── model_output/                       # XGBoost model outputs
│   ├── compression_ratio_model.pkl
│   ├── psnr_model.pkl
│   └── model_metadata.json
├── cnn_model_output/                   # Neural network model outputs
│   ├── cnn_compression_ratio_model.keras
│   ├── cnn_psnr_model.keras
│   ├── cnn_compression_ratio_scaler.pkl
│   └── cnn_psnr_scaler.pkl
└── comparison_output/                  # Comparison plots
    ├── comparison_compression_ratio.pdf
    ├── comparison_psnr.pdf
    └── comparison_metrics.pdf
```

## Data Requirements

### Input CSV Format

The models require two CSV files:

**Lossless Compression Results** (compression_parameter_study_results.csv):
```csv
Library,Distribution,Chunk Size (bytes),Target CPU Util (%),Compress Time (ms),Decompress Time (ms),Compression Ratio,Compress CPU %,Decompress CPU %,Shannon Entropy (bits/byte),MAD,Second Derivative Mean,Success
BZIP2,uniform_7,4096,0.0,0.605,0.079,2.4925,92.85,93.43,2.9984,1.42,4.652,YES
```

**Lossy Compression Results** (compression_lossy_parameter_study_results.csv):
```csv
Library,Distribution,Chunk Size (bytes),Target CPU Util (%),Compress Time (ms),Decompress Time (ms),Compression Ratio,PSNR (dB),Compress CPU %,Decompress CPU %,Shannon Entropy (bits/byte),MAD,Second Derivative Mean,Success
ZFP_tol_0.010000,uniform_7,4096,0.0,0.125,0.088,5.8182,52.73,91.23,94.56,2.9984,1.42,4.652,YES
```

### Features Used

The models use the following input features:
- **Chunk Size (bytes)** - Size of data chunk being compressed
- **Target CPU Util (%)** - Target CPU utilization level
- **Shannon Entropy (bits/byte)** - Data randomness measure
- **MAD** - Mean Absolute Deviation
- **Second Derivative Mean** - Data smoothness measure
- **Library** - Compression library (one-hot encoded)
- **Data Type** - char or float (one-hot encoded)

### Running Benchmarks

To generate the required CSV files, run the compression benchmarks:

```bash
# Navigate to build directory
cd /workspace/build

# Run lossless compression benchmark
./bin/test_compress_parameter_study

# Run lossy compression benchmark
./bin/test_compress_lossy_parameter_study
```

The benchmarks test 7 chunk sizes (4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB) across multiple configurations:
- Multiple distributions (uniform_7, normal_7, exponential_7, etc.)
- Multiple CPU utilization targets (0%, 25%, 50%, 75%, 100%)
- Multiple compression libraries (BZIP2, ZFP with different tolerances)

Estimated runtime: 8-12 hours for complete benchmark suite.

## Training Models

### 1. Train XGBoost Models (Recommended)

```bash
cd /workspace/context-transport-primitives/scripts

python train_compression_model.py \
  /workspace/build/compression_parameter_study_results.csv \
  /workspace/build/compression_lossy_parameter_study_results.csv
```

**Output:**
- `model_output/compression_ratio_model.pkl` - XGBoost model for compression ratio
- `model_output/psnr_model.pkl` - XGBoost model for PSNR
- `model_output/model_metadata.json` - Model configuration and feature names
- Visualization PDFs showing prediction accuracy

**Expected Performance:**
- Compression Ratio: R² > 0.99, RMSE < 0.03
- PSNR: R² > 0.99, RMSE < 0.15 dB

### 2. Train Dense Neural Network Models

```bash
cd /workspace/context-transport-primitives/scripts

python train_cnn_compression_model.py \
  /workspace/build/compression_parameter_study_results.csv \
  /workspace/build/compression_lossy_parameter_study_results.csv
```

**Output:**
- `cnn_model_output/cnn_compression_ratio_model.keras` - TensorFlow model for ratio
- `cnn_model_output/cnn_psnr_model.keras` - TensorFlow model for PSNR
- `cnn_model_output/cnn_*_scaler.pkl` - Feature scalers (StandardScaler)
- `cnn_model_output/cnn_model_metadata.json` - Model configuration
- Training history and prediction visualization PDFs

**Expected Performance:**
- Compression Ratio: R² ≈ 0.99, RMSE < 0.04
- PSNR: R² ≈ 0.99, RMSE < 0.9 dB

**Optimized Hyperparameters:**
- **Compression Ratio Model:**
  - Architecture: [128, 64, 32, 16] dense layers
  - Learning Rate: 0.01 (high LR works best)
  - Dropout: 0.2
  - Batch Size: 16

- **PSNR Model:**
  - Architecture: [128, 64, 32, 16] dense layers
  - Learning Rate: 0.001
  - Dropout: 0.1 (low dropout works best)
  - Batch Size: 16

### 3. Compare Models

```bash
cd /workspace/context-transport-primitives/scripts

python compare_cnn_vs_xgboost.py \
  /workspace/build/compression_parameter_study_results.csv \
  /workspace/build/compression_lossy_parameter_study_results.csv
```

**Output:**
- `comparison_output/comparison_compression_ratio.pdf` - Scatter plots
- `comparison_output/comparison_psnr.pdf` - PSNR comparison
- `comparison_output/comparison_metrics.pdf` - Performance metrics bar charts
- Console output with detailed performance comparison

### 4. Batch Inference Scaling Study

```bash
cd /workspace/context-transport-primitives/scripts

python batch_inference_study.py \
  /workspace/build/compression_parameter_study_results.csv
```

**Output:**
- `batch_inference_results/xgboost_batch_results.csv` - XGBoost results
- `batch_inference_results/neural_network_batch_results.csv` - NN results
- `batch_inference_results/batch_inference_scaling.pdf` - Scaling plots
- Console output with throughput and latency metrics for batch sizes 1-512

## Using Trained Models

### XGBoost Model Inference

```python
import joblib
import pandas as pd
import numpy as np

# Load models
ratio_model = joblib.load('model_output/compression_ratio_model.pkl')
psnr_model = joblib.load('model_output/psnr_model.pkl')

# Load metadata for feature names
import json
with open('model_output/model_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_names = metadata['feature_names']

# Prepare input features (example)
features = pd.DataFrame({
    'Chunk Size (bytes)': [65536],
    'Target CPU Util (%)': [50.0],
    'Shannon Entropy (bits/byte)': [3.5],
    'MAD': [1.2],
    'Second Derivative Mean': [2.5],
    'Library_BZIP2': [1],
    'Library_ZFP_tol_0.010000': [0],
    'Library_ZFP_tol_0.100000': [0],
    'Data Type_char': [0],
    'Data Type_float': [1]
})

# Ensure correct feature order
features = features[feature_names]

# Predict
compression_ratio = ratio_model.predict(features)[0]
psnr = psnr_model.predict(features)[0]

print(f"Predicted Compression Ratio: {compression_ratio:.2f}")
print(f"Predicted PSNR: {psnr:.2f} dB")
```

### Dense Neural Network Inference

```python
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Load models and scalers
ratio_model = tf.keras.models.load_model('cnn_model_output/cnn_compression_ratio_model.keras')
psnr_model = tf.keras.models.load_model('cnn_model_output/cnn_psnr_model.keras')
ratio_scaler = joblib.load('cnn_model_output/cnn_compression_ratio_scaler.pkl')
psnr_scaler = joblib.load('cnn_model_output/cnn_psnr_scaler.pkl')

# Prepare input features (same as above)
features = pd.DataFrame({
    'Chunk Size (bytes)': [65536],
    'Target CPU Util (%)': [50.0],
    'Shannon Entropy (bits/byte)': [3.5],
    'MAD': [1.2],
    'Second Derivative Mean': [2.5],
    'Library_BZIP2': [1],
    'Library_ZFP_tol_0.010000': [0],
    'Library_ZFP_tol_0.100000': [0],
    'Data Type_char': [0],
    'Data Type_float': [1]
})

# Scale features
features_ratio_scaled = ratio_scaler.transform(features)
features_psnr_scaled = psnr_scaler.transform(features)

# Predict
compression_ratio = ratio_model.predict(features_ratio_scaled, verbose=0)[0][0]
psnr = psnr_model.predict(features_psnr_scaled, verbose=0)[0][0]

print(f"Predicted Compression Ratio: {compression_ratio:.2f}")
print(f"Predicted PSNR: {psnr:.2f} dB")
```

## Model Details

### XGBoost Hyperparameters

**Compression Ratio Model:**
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'min_child_weight': 10,
    'reg_lambda': 1.0
}
```

**PSNR Model:**
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8
}
```

### Dense Neural Network Architecture

**Compression Ratio Model:**
```
Input (10 features)
  ↓
Dense(128, relu) + BatchNorm + Dropout(0.2)
  ↓
Dense(64, relu) + BatchNorm + Dropout(0.2)
  ↓
Dense(32, relu) + BatchNorm + Dropout(0.2)
  ↓
Dense(16, relu) + BatchNorm + Dropout(0.2)
  ↓
Dense(1, linear)
```

**PSNR Model:**
```
Input (10 features)
  ↓
Dense(128, relu) + BatchNorm + Dropout(0.1)
  ↓
Dense(64, relu) + BatchNorm + Dropout(0.1)
  ↓
Dense(32, relu) + BatchNorm + Dropout(0.1)
  ↓
Dense(16, relu) + BatchNorm + Dropout(0.1)
  ↓
Dense(1, linear)
```

**Training Configuration:**
- Optimizer: AdamW (decoupled weight decay)
- Loss: Mean Squared Error (MSE)
- Callbacks: Early stopping (patience=30), ReduceLROnPlateau
- Training epochs: Up to 200 with early stopping
- Validation split: 20%

## Key Findings from Hyperparameter Tuning

### Critical Hyperparameters

1. **Learning Rate** - Most critical parameter
   - High LR (0.01) optimal for compression ratio
   - Standard LR (0.001) optimal for PSNR
   - Too low (0.0001) causes training failure

2. **Dropout Rate** - Task-dependent
   - Low dropout (0.1) best for PSNR prediction
   - Standard dropout (0.2) good for compression ratio
   - High dropout (0.4) significantly hurts performance

3. **Batch Size** - Small is better for this dataset
   - Batch size 16 works well
   - Batch size 8 also competitive
   - Large batch (32) reduces performance

4. **Network Depth** - Moderate depth optimal
   - 4-layer network (128→64→32→16) performs best
   - Shallow 2-layer network surprisingly competitive
   - Very deep 5-layer network prone to overfitting

### Performance Impact

| Parameter Variation | Compression Ratio R² Range | PSNR R² Range |
|-------------------|---------------------------|---------------|
| Learning Rate | -12.3 to 0.99 | -140 to 0.99 |
| Dropout | 0.88 to 0.98 | 0.83 to 0.99 |
| Batch Size | 0.93 to 0.98 | -2.6 to 0.97 |
| Network Depth | 0.93 to 0.98 | 0.95 to 0.96 |

## Troubleshooting

### Issue: Models perform poorly (R² < 0.9)

**Solution:**
- Check that CSV files have sufficient data (>500 samples recommended)
- Verify all features are present in input data
- Ensure 'Success' column is filtered to 'YES' only
- Check for missing or NaN values in input data

### Issue: Training takes too long

**Solution:**
- XGBoost: Reduce `n_estimators` from 200 to 100
- Neural Network: Reduce `epochs` from 200 to 100
- Use fewer chunk sizes in benchmarks (e.g., only 64KB, 1MB, 16MB)

### Issue: Neural network training fails to converge

**Solution:**
- Check learning rate (should be 0.001-0.01 range)
- Verify feature scaling is applied (StandardScaler)
- Increase patience for early stopping callback
- Check for data quality issues (outliers, missing values)

### Issue: Feature mismatch error during inference

**Solution:**
- Load feature names from `model_metadata.json`
- Ensure input DataFrame columns match training features exactly
- Use correct feature order: `features = features[feature_names]`

## Citation

If you use these models in your research, please cite:

```
IOWarp Compression Performance Prediction Models
Context Transport Primitives - IOWarp Core Framework
https://github.com/iowarp/iowarp-core
```

## Additional Scripts

- `train_cnn_compression_model.py` - Neural network training (optimized hyperparameters)
- `compare_cnn_vs_xgboost.py` - Model comparison and visualization
- Model training and validation scripts in `/workspace/context-transport-primitives/scripts/`

## Batch Inference Performance

### Running Batch Inference Study

To measure how models perform with different batch sizes:

```bash
cd /workspace/context-transport-primitives/scripts

python batch_inference_study.py \
  /workspace/build/compression_parameter_study_results.csv
```

**Output:**
- `batch_inference_results/xgboost_batch_results.csv` - XGBoost performance by batch size
- `batch_inference_results/neural_network_batch_results.csv` - NN performance by batch size
- `batch_inference_results/batch_inference_scaling.pdf` - Visualization plots

### Batch Inference Results

| Batch Size | XGBoost Throughput | XGBoost Latency/Sample | NN Throughput | NN Latency/Sample |
|------------|-------------------|----------------------|---------------|------------------|
| 1 | 513 pred/s | 1.95 ms | 22 pred/s | 46.18 ms |
| 16 | 10,605 pred/s | 0.09 ms | 342 pred/s | 2.92 ms |
| 64 | 42,046 pred/s | 0.02 ms | 1,382 pred/s | 0.72 ms |
| 256 | 150,043 pred/s | 0.007 ms | 4,958 pred/s | 0.20 ms |
| 512 | 251,680 pred/s | 0.004 ms | 8,930 pred/s | 0.11 ms |

### Key Findings

**Batch Size 16 (Recommended for Real-time Applications):**
- **XGBoost**: 10,605 predictions/sec, 0.09 ms/sample
- **Neural Network**: 342 predictions/sec, 2.92 ms/sample
- **XGBoost is 31× faster** at batch size 16

**Batching Speedup (vs Single Prediction):**
- **XGBoost**: 20.7× throughput improvement (batch 16 vs batch 1)
- **Neural Network**: 15.8× throughput improvement (batch 16 vs batch 1)
- XGBoost scales better with batch size

**Maximum Throughput (Batch Size 512):**
- **XGBoost**: 251,680 predictions/sec (28× faster than NN)
- **Neural Network**: 8,930 predictions/sec

**Recommendation for Production:**
- Use **batch size 16** for optimal latency/throughput trade-off
- XGBoost processes 16 predictions in ~1.5 ms
- Neural Network processes 16 predictions in ~47 ms
- Both models benefit significantly from batching

## Performance Benchmarks

Hardware configuration for reported performance:
- CPU: Intel/AMD x86_64 (AVX2, AVX512 support)
- RAM: Sufficient for dataset (~1GB for 10K samples)
- Python: 3.10-3.13
- TensorFlow: 2.20.0
- XGBoost: Latest version

Inference times measured on single-threaded CPU execution.

## License

Part of IOWarp Core Framework. See main repository LICENSE file.
