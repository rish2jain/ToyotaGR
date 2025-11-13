# SHAP Explainability for Anomaly Detection

## Overview

This document describes the SHAP (SHapley Additive exPlanations) explainability implementation for the RaceIQ Pro anomaly detection system. SHAP provides interpretable explanations for why the ML model flagged specific laps or telemetry data as anomalous.

## Installation

SHAP is already included in the requirements.txt file:

```bash
pip install -r requirements.txt
```

Or install SHAP separately:

```bash
pip install shap==0.44.0
```

## Implementation Files

### 1. Enhanced Anomaly Detector (`src/tactical/anomaly_detector.py`)

#### New Methods:

**`explain_anomaly(anomaly_data, telemetry_features=None)`**
- Generates SHAP-based explanation for a single anomaly
- Uses SHAP TreeExplainer for Isolation Forest
- Returns feature importance and human-readable explanations

**Output Format:**
```python
{
    'top_features': [
        {
            'feature': 'S1_SECONDS',
            'contribution': 0.45,
            'shap_value': -0.23,
            'feature_value': 28.5,
            'direction': 'high'
        },
        ...
    ],
    'explanation': "S1 Seconds 45% too high, Top Speed 35% too low",
    'shap_values': array([...]),
    'confidence': 0.85
}
```

**`get_anomaly_explanations(anomalies_df, telemetry_data=None)`**
- Generates SHAP explanations for all detected anomalies
- Returns DataFrame with explanation columns
- Gracefully handles missing SHAP installation

**Output Columns:**
- `explanation`: Human-readable explanation string
- `top_feature_1/2/3`: Names of top contributing features
- `contribution_1/2/3`: Contribution percentages (0-1)
- `confidence`: Explanation confidence score (0-1)

### 2. Example Demo (`examples/shap_anomaly_demo.py`)

Comprehensive demonstration script that:
- Loads real or synthetic telemetry data
- Runs Tier 1 (statistical) and Tier 2 (ML) anomaly detection
- Generates SHAP explanations for anomalies
- Visualizes feature importance
- Saves visualization to `shap_importance.png`

**Usage:**
```bash
python examples/shap_anomaly_demo.py
```

### 3. Dashboard Integration (`dashboard/pages/tactical.py`)

Enhanced tactical analysis page with:
- Two-tab interface: Statistical vs ML detection
- SHAP-powered anomaly explanations
- Interactive expandable sections for each anomaly
- Feature importance bar charts
- Detailed SHAP values viewer

## Usage Examples

### Basic Usage

```python
from src.tactical.anomaly_detector import AnomalyDetector
import pandas as pd

# Initialize detector
detector = AnomalyDetector()

# Load your telemetry data
telemetry_df = pd.read_csv('your_telemetry.csv')

# Detect anomalies using ML
result_df = detector.detect_pattern_anomalies(
    telemetry_df,
    contamination=0.1  # Expect 10% anomalies
)

# Get anomalies
anomalies = result_df[result_df['is_anomaly'] == -1]

# Generate SHAP explanations
explained = detector.get_anomaly_explanations(anomalies, result_df)

# View explanations
print(explained[['LAP_NUMBER', 'explanation', 'confidence']])
```

### Single Anomaly Explanation

```python
# Explain a specific anomaly
anomaly_row = anomalies.iloc[0]
explanation = detector.explain_anomaly(anomaly_row)

print(f"Explanation: {explanation['explanation']}")
print(f"Confidence: {explanation['confidence']:.1%}")

# Top contributing features
for feature in explanation['top_features'][:3]:
    print(f"- {feature['feature']}: {feature['contribution']:.1%} ({feature['direction']})")
```

### Dashboard Usage

1. Navigate to the Tactical Analysis page in the Streamlit dashboard
2. Select a driver
3. Go to "Advanced Anomaly Detection" section
4. Click on "ML Detection with SHAP" tab
5. View anomalies with expandable explanations
6. Each anomaly shows:
   - Human-readable explanation
   - Confidence score
   - Top 3 contributing features
   - Feature importance bar chart
   - Detailed SHAP values (expandable)

## Technical Details

### SHAP Explainer Selection

The implementation uses an intelligent fallback mechanism:

1. **Primary**: `shap.TreeExplainer` - Fast, optimized for tree-based models like Isolation Forest
2. **Fallback**: `shap.KernelExplainer` - Model-agnostic, used if TreeExplainer fails

### Feature Importance Calculation

SHAP values are calculated for each feature, representing their contribution to the anomaly score:

- **Positive SHAP value**: Feature pushes prediction toward "anomaly"
- **Negative SHAP value**: Feature pushes prediction toward "normal"
- **Contribution**: Normalized absolute SHAP value as percentage

### Direction Classification

Each feature is classified as:
- **high**: SHAP value > 0.01 (feature too high)
- **low**: SHAP value < -0.01 (feature too low)
- **normal**: |SHAP value| < 0.01 (feature within normal range)

### Confidence Score

Confidence is calculated from the anomaly score:
```python
confidence = max(0.0, min(1.0, 1.0 - (anomaly_score + 0.5)))
```

Lower anomaly scores (more anomalous) result in higher confidence.

## Error Handling

The implementation includes comprehensive error handling:

### Graceful Degradation
- If SHAP is not installed: Returns warning, continues without explanations
- If model not trained: Returns warning with instructions
- If feature extraction fails: Skips individual anomaly, continues with others

### Import Guards
```python
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
```

All SHAP functionality checks `SHAP_AVAILABLE` before execution.

## Performance Considerations

### TreeExplainer Performance
- **Fast**: O(TLD²) where T=trees, L=leaves, D=depth
- **Typical time**: <100ms for 100 trees, 10 features
- **Recommended for**: Real-time dashboard usage

### KernelExplainer Performance
- **Slower**: O(2^M) where M=features (with approximations)
- **Typical time**: 1-10 seconds depending on background samples
- **Used only**: As fallback when TreeExplainer fails

### Optimization Tips
1. Use smaller contamination values to reduce anomaly count
2. Limit background samples for KernelExplainer (default: 100)
3. Cache SHAP explainer in detector instance (already implemented)
4. Consider batch explanation for large datasets

## Visualization

### Feature Importance Bar Chart

Generated for each anomaly in the dashboard:
- Horizontal bar chart
- Top 3 features by contribution
- Color-coded by importance
- Percentage contributions

### Command-Line Visualization

The demo script generates a comprehensive plot:
```bash
python examples/shap_anomaly_demo.py
# Creates: examples/shap_importance.png
```

## Integration with RaceIQ Pro

### Tier 1: Statistical Detection
- Uses rolling z-scores
- Fast, simple outlier detection
- Good for single-metric anomalies

### Tier 2: ML Detection with SHAP
- Uses Isolation Forest
- Detects complex multivariate patterns
- SHAP explains which features contributed
- Better for understanding root causes

### Recommended Workflow
1. Run Tier 1 for quick anomaly screening
2. Run Tier 2 with SHAP for detailed analysis
3. Use SHAP explanations to generate recommendations
4. Present findings in dashboard with visualizations

## Future Enhancements

### Planned Features
1. **Historical Pattern Analysis**: Build database of common anomaly patterns
2. **Automated Recommendations**: Map SHAP features to actionable advice
3. **Cross-Driver Comparison**: Compare anomaly patterns across drivers
4. **Real-time Alerts**: Stream anomaly detection with live SHAP explanations
5. **SHAP Summary Plots**: Add waterfall and force plots to dashboard

### Potential Improvements
- Custom SHAP kernel for racing domain knowledge
- Feature interaction detection
- Temporal SHAP analysis (how features contribute over time)
- Integration with predictive maintenance

## Troubleshooting

### SHAP Import Errors
```bash
# If SHAP import fails:
pip install --upgrade shap
pip install llvmlite --ignore-installed
```

### TreeExplainer Errors
If TreeExplainer fails, the system automatically falls back to KernelExplainer. Check warnings for details.

### Memory Issues
For large datasets:
```python
# Reduce background samples
detector.shap_explainer = shap.KernelExplainer(
    model.decision_function,
    shap.sample(data, 50)  # Reduce from 100 to 50
)
```

## References

- SHAP Documentation: https://shap.readthedocs.io/
- Original Paper: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- Isolation Forest: Liu et al. (2008) "Isolation Forest"

## Support

For questions or issues:
1. Check the example demo: `examples/shap_anomaly_demo.py`
2. Review error messages and warnings
3. Verify SHAP installation: `pip list | grep shap`
4. Check Python version compatibility (3.9+ recommended)

---

**Last Updated**: 2025-11-13
**RaceIQ Pro Version**: 1.0
**SHAP Version**: 0.44.0
