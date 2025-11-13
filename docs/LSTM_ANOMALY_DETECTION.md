# LSTM Anomaly Detection for Racing Telemetry

## Overview

The LSTM (Long Short-Term Memory) Anomaly Detection module provides deep learning-based pattern detection for racing telemetry data. This Tier 3 detection method complements the existing statistical (Tier 1) and machine learning (Tier 2) approaches.

## Why LSTM for Anomaly Detection?

### Problem: Traditional Methods Miss Temporal Patterns

Racing telemetry is inherently **time-series data** where the sequence matters:
- A driver's brake-throttle-steer sequence defines their driving style
- Anomalies often manifest as unusual temporal patterns, not just outliers
- Statistical methods (z-scores) only look at individual points
- Isolation Forest considers multivariate relationships but not temporal order

### Solution: LSTM Autoencoders

LSTM autoencoders learn **normal temporal patterns** during training and detect anomalies when the pattern deviates:

```
Normal Pattern:    Brake → Coast → Apex → Throttle → Accelerate
Anomaly:           Brake → Throttle → Brake → Coast (unusual sequence)
```

**Key Advantage**: Detects anomalies based on *sequence* rather than just values.

## How It Works

### Architecture: LSTM Autoencoder

```
Input Sequence (50 timesteps × 6 features)
    ↓
[LSTM Layer 1: 64 units, return sequences]
    ↓
[LSTM Layer 2: 32 units, compress to latent space]
    ↓
[Dropout: 0.2]
    ↓
[Dense: 32 units, ReLU activation]
    ↓
[Dropout: 0.2]
    ↓
[Dense: 6 units, reconstruct features]
    ↓
Output (reconstructed features)
```

### Reconstruction Error = Anomaly Signal

The model is trained to reconstruct **normal driving patterns**. When it encounters an anomalous pattern, the reconstruction error is high.

**Reconstruction Error (MSE)** = Mean Squared Error between input and output

```
If MSE > threshold → Anomaly detected
```

### Training Process

1. **Filter Normal Data**: Use bottom 80% of lap times as "normal" training data
2. **Create Sequences**: Sliding window of 50 timesteps
3. **Train Autoencoder**: 30-50 epochs (30-90 seconds)
4. **Calculate Threshold**: 95th percentile of reconstruction errors
5. **Detect Anomalies**: Flag sequences exceeding threshold

## Features Used

The LSTM detector analyzes 6 telemetry features:

| Feature | Column Mapping | Description |
|---------|---------------|-------------|
| **Speed** | `KPH`, `MPH`, `SPEED` | Vehicle speed |
| **ThrottlePosition** | `aps`, `THROTTLE` | Accelerator pedal position (0-100%) |
| **BrakePressure** | `pbrake_f`, `BRAKE` | Front brake pressure |
| **SteeringAngle** | `Steering_Angle`, `STEERING` | Steering wheel angle |
| **RPM** | `nmot`, `RPM` | Engine revolutions per minute |
| **Gear** | `gear`, `GEAR` | Current gear selection |

**Note**: The detector gracefully handles missing features and normalizes all values to 0-1 range.

## Usage

### Option 1: Direct Use with `LSTMAnomalyDetector`

```python
from src.tactical.anomaly_detector import LSTMAnomalyDetector
import pandas as pd

# Initialize detector
lstm_detector = LSTMAnomalyDetector(
    sequence_length=50,  # Number of timesteps per sequence
    verbose=0            # 0=silent, 1=progress bar, 2=one line per epoch
)

# Load telemetry data
telemetry_df = pd.read_csv('telemetry_data.csv')

# Detect anomalies
result_df = lstm_detector.detect_pattern_anomalies(
    telemetry_df,
    epochs=50,              # Training epochs (more = better but slower)
    batch_size=32,          # Batch size for training
    contamination=0.05      # Expected % of anomalies (5%)
)

# Get anomalies
anomalies = result_df[result_df['lstm_is_anomaly']]
print(f"Found {len(anomalies)} anomalies")

# Examine reconstruction errors
print(result_df[['LAP_NUMBER', 'lstm_reconstruction_error', 'lstm_anomaly_score']])
```

### Option 2: Use via `AnomalyDetector` Wrapper

```python
from src.tactical.anomaly_detector import AnomalyDetector

# Initialize main detector
detector = AnomalyDetector()

# Run LSTM detection
result_df = detector.detect_lstm_anomalies(
    telemetry_df,
    sequence_length=50,
    epochs=50,
    contamination=0.05
)

# Compare with other methods
stat_result = detector.detect_statistical_anomalies(telemetry_df)
ml_result = detector.detect_pattern_anomalies(telemetry_df)
```

### Option 3: Dashboard Integration

The LSTM detector is integrated into the **Tactical Analysis** dashboard:

1. Navigate to **Tactical Analysis** page
2. Select a driver
3. Scroll to **Advanced Anomaly Detection**
4. Click the **"Deep Learning (LSTM)"** tab
5. Adjust parameters:
   - **Sequence Length**: 20-100 timesteps (default: 50)
   - **Training Epochs**: 10-100 (default: 30)
   - **Contamination**: 1-20% (default: 5%)
6. Click **"Run LSTM Detection"**
7. Wait 30-90 seconds for training
8. View results:
   - Anomaly table with reconstruction errors
   - Reconstruction error plot over time
   - Method comparison (Statistical vs ML vs LSTM)

## Output Columns

The LSTM detector adds these columns to your DataFrame:

| Column | Type | Description |
|--------|------|-------------|
| `lstm_reconstruction_error` | float | MSE between input and reconstructed output |
| `lstm_is_anomaly` | bool | True if reconstruction error > threshold |
| `lstm_anomaly_score` | float | Normalized score (0-1), higher = more anomalous |

## Performance Characteristics

### Training Time

| Data Size | Hardware | Training Time (50 epochs) |
|-----------|----------|---------------------------|
| 500 samples | CPU (4 cores) | ~30-45 seconds |
| 500 samples | GPU (CUDA) | ~10-15 seconds |
| 2000 samples | CPU (4 cores) | ~60-90 seconds |
| 2000 samples | GPU (CUDA) | ~20-30 seconds |

**Tip**: Use fewer epochs (20-30) for faster demos with acceptable performance.

### Inference Time

Once trained, inference is **fast**:
- 500 samples: <1 second
- 2000 samples: ~2-3 seconds

### Memory Requirements

- Model size: ~500 KB (small, can be cached)
- Peak memory: ~200-500 MB (during training)

## When to Use LSTM vs Statistical vs Isolation Forest

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **Statistical (Z-Score)** | Single-point outliers, obvious anomalies | ⚡⚡⚡ Fast (ms) | ⭐⭐ Good |
| **Isolation Forest** | Multivariate patterns, feature combinations | ⚡⚡ Medium (seconds) | ⭐⭐⭐ Better |
| **LSTM** | Temporal patterns, sequential anomalies | ⚡ Slow training (30-90s), fast inference | ⭐⭐⭐⭐ Best for sequences |

### Recommendation

**Use all three methods together**:
1. Start with **Statistical** for quick baseline
2. Use **Isolation Forest** for multivariate anomalies with SHAP explanations
3. Apply **LSTM** for complex temporal pattern detection

The dashboard's **Method Comparison** feature shows agreement between methods and helps identify robust anomalies.

## Examples

### Example 1: Detect Unusual Braking Patterns

```python
# LSTM will catch: "Driver braked late, then early, then late again"
# (Statistical methods might miss this pattern)

lstm_detector = LSTMAnomalyDetector(sequence_length=50)
result = lstm_detector.detect_pattern_anomalies(telemetry_df)

# Find high-error laps
high_error_laps = result[result['lstm_reconstruction_error'] > 0.1]
print(high_error_laps[['LAP_NUMBER', 'lstm_reconstruction_error']])
```

### Example 2: Compare Detection Methods

```python
detector = AnomalyDetector()

# Run all methods
stat_df = detector.detect_statistical_anomalies(telemetry_df, threshold=2.5)
ml_df = detector.detect_pattern_anomalies(telemetry_df, contamination=0.05)
lstm_df = detector.detect_lstm_anomalies(telemetry_df, epochs=30)

# Count anomalies
stat_count = (stat_df['anomaly_count'] > 0).sum()
ml_count = (ml_df['is_anomaly'] == -1).sum()
lstm_count = lstm_df['lstm_is_anomaly'].sum()

print(f"Statistical: {stat_count} anomalies")
print(f"Isolation Forest: {ml_count} anomalies")
print(f"LSTM: {lstm_count} anomalies")

# Find anomalies detected by all three methods (high confidence)
high_confidence = (
    (stat_df['anomaly_count'] > 0) &
    (ml_df['is_anomaly'] == -1) &
    lstm_df['lstm_is_anomaly']
)
print(f"\nHigh-confidence anomalies (all methods agree): {high_confidence.sum()}")
```

### Example 3: Visualize Reconstruction Error

```python
import matplotlib.pyplot as plt

result = lstm_detector.detect_pattern_anomalies(telemetry_df)

plt.figure(figsize=(12, 6))
plt.plot(result['LAP_NUMBER'], result['lstm_reconstruction_error'],
         label='Reconstruction Error', alpha=0.7)
plt.scatter(result[result['lstm_is_anomaly']]['LAP_NUMBER'],
           result[result['lstm_is_anomaly']]['lstm_reconstruction_error'],
           color='red', s=100, label='Anomalies', zorder=5)
plt.axhline(y=lstm_detector.threshold, color='orange', linestyle='--',
           label='Threshold')
plt.xlabel('Lap Number')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('LSTM Anomaly Detection: Reconstruction Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Configuration Guidelines

### Sequence Length

**Trade-offs**:
- **Short (20-30)**: Faster training, local patterns, may miss long-term anomalies
- **Medium (50)**: Balanced, captures typical corner sequences
- **Long (80-100)**: Captures full sector patterns, slower training

**Recommendation**: Start with 50, adjust based on data frequency.

### Training Epochs

**Trade-offs**:
- **Few (10-20)**: Fast training, may underfit
- **Medium (30-50)**: Good balance, early stopping helps
- **Many (80-100)**: Better learning, risk of overfitting

**Recommendation**: 30-50 epochs with early stopping (patience=5).

### Contamination

**Expected % of anomalies in your data**:
- **Low (1-3%)**: Clean data, strict threshold
- **Medium (5-10%)**: Typical racing data
- **High (15-20%)**: Noisy data or learning sessions

**Recommendation**: Start with 5%, adjust based on domain knowledge.

## Limitations and Considerations

### When LSTM May Not Help

1. **Too Little Data**: Need at least 100 samples (2 laps with 50 samples/lap)
2. **No Temporal Dependencies**: If order doesn't matter, use Isolation Forest instead
3. **Single-Point Outliers**: Statistical methods are faster and sufficient
4. **Constantly Changing Patterns**: Driver still learning or experimenting

### Handling Edge Cases

#### Missing Features
```python
# Detector gracefully handles missing features
# If only 3 of 6 features available, model adapts automatically
result = lstm_detector.detect_pattern_anomalies(partial_telemetry_df)
```

#### Too Few Laps
```python
# Check data size before running
if len(telemetry_df) < 100:
    print("Warning: Insufficient data for LSTM (need ≥100 samples)")
    # Fall back to statistical methods
else:
    result = lstm_detector.detect_pattern_anomalies(telemetry_df)
```

#### Variable-Length Laps
```python
# LSTM works with variable lap lengths
# Sequences are created by sliding window, not per-lap
# Each sequence is fixed length (e.g., 50 timesteps)
```

## Installation

### Standard TensorFlow (CPU + GPU support)
```bash
pip install tensorflow
```

### CPU-Only (smaller, faster install)
```bash
pip install tensorflow-cpu
```

### With All Dependencies
```bash
pip install tensorflow scikit-learn shap pandas numpy matplotlib
```

## Troubleshooting

### Issue: "TensorFlow not installed"
**Solution**: Install TensorFlow (see Installation section)

### Issue: "Insufficient data" error
**Solution**: Ensure you have at least `sequence_length * 2` samples (e.g., 100 for length=50)

### Issue: Training is very slow
**Solutions**:
1. Reduce `epochs` (try 20-30)
2. Increase `batch_size` (try 64)
3. Reduce `sequence_length` (try 30)
4. Install GPU version of TensorFlow

### Issue: All samples flagged as anomalies
**Solutions**:
1. Increase `contamination` parameter (try 0.10 or 0.15)
2. Check if data has sufficient "normal" samples for training
3. Verify lap time filtering is working correctly

### Issue: No anomalies detected
**Solutions**:
1. Decrease `contamination` parameter (try 0.03)
2. Increase `epochs` for better learning (try 50-80)
3. Check if data actually contains anomalies (visualize manually)

## Demo and Testing

### Run the Demo Script
```bash
python examples/lstm_anomaly_demo.py
```

This will:
1. Create synthetic telemetry data with known anomalies
2. Train LSTM detector and measure performance
3. Compare with statistical methods
4. Generate visualizations

### Expected Output
```
================================================================================
LSTM ANOMALY DETECTION FOR RACING TELEMETRY
RaceIQ Pro - Deep Learning Module Demo
================================================================================

✓ TensorFlow is available

--------------------------------------------------------------------------------
OPTION 1: Using Synthetic Sample Data
--------------------------------------------------------------------------------
Creating sample telemetry data: 500 samples, 25 anomalies
✓ Sample data created with 25 true anomalies

================================================================================
LSTM ANOMALY DETECTION DEMO
================================================================================

1. Initializing LSTM Anomaly Detector...
   ✓ Detector initialized with sequence length = 50

2. Training LSTM autoencoder and detecting anomalies...
   (This may take 30-90 seconds depending on your hardware)

Epoch 1/30 [=====>....] loss: 0.0523
...
   ✓ LSTM training complete!

3. Results:
   - Total samples: 500
   - Anomalies detected: 28
   - Detection rate: 5.6%

4. Performance Metrics:
   - Precision: 0.821
   - Recall: 0.920
   - F1 Score: 0.868
   - True Positives: 23
   - False Positives: 5
   - False Negatives: 2

✓ Visualization saved to: examples/lstm_anomaly_results.png
```

## References

### Papers and Resources

1. **LSTM Autoencoders for Anomaly Detection**:
   - Malhotra, P., et al. (2016). "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

2. **Time Series Anomaly Detection**:
   - Chauhan, S., & Vig, L. (2015). "Anomaly detection in ECG time signals via deep long short-term memory networks"

3. **TensorFlow Documentation**:
   - https://www.tensorflow.org/tutorials/structured_data/time_series

### Related Documentation

- [Anomaly Detection Overview](./ANOMALY_DETECTION.md)
- [Statistical Methods](./STATISTICAL_METHODS.md)
- [SHAP Explainability](./SHAP_EXPLANATIONS.md)
- [Tactical Analysis Guide](./TACTICAL_ANALYSIS.md)

## Summary

**Key Points**:
- LSTM autoencoders detect temporal pattern anomalies in racing telemetry
- Training takes 30-90 seconds, inference is fast (<1 second)
- Best for sequential data where order matters (brake-throttle-steer patterns)
- Complements statistical and ML methods for comprehensive anomaly detection
- Integrated into dashboard with easy parameter tuning
- Use all three methods together for robust anomaly detection

**Next Steps**:
1. Install TensorFlow: `pip install tensorflow`
2. Run demo: `python examples/lstm_anomaly_demo.py`
3. Try in dashboard: Navigate to Tactical Analysis → Deep Learning (LSTM) tab
4. Compare with other methods to find robust anomalies

---

*For questions or issues, see the main documentation or file an issue on GitHub.*
