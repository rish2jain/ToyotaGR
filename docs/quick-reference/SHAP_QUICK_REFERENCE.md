# SHAP Anomaly Detection - Quick Reference

## Installation

```bash
pip install shap==0.44.0
# or
pip install -r requirements.txt
```

## Quick Start (Command Line)

```bash
# Run the demo
python examples/shap_anomaly_demo.py
```

## Quick Start (Python)

```python
from src.tactical.anomaly_detector import AnomalyDetector
import pandas as pd

# 1. Initialize
detector = AnomalyDetector()

# 2. Detect anomalies
df = pd.read_csv('your_telemetry.csv')
result = detector.detect_pattern_anomalies(df, contamination=0.1)
anomalies = result[result['is_anomaly'] == -1]

# 3. Get SHAP explanations
explained = detector.get_anomaly_explanations(anomalies, result)

# 4. View results
print(explained[['LAP_NUMBER', 'explanation', 'confidence']])
```

## Quick Start (Dashboard)

```bash
# Start dashboard
streamlit run dashboard/app.py

# Navigate to:
# Tactical Analysis → Advanced Anomaly Detection → ML Detection with SHAP
```

## Method Reference

### `explain_anomaly(anomaly_data, telemetry_features=None)`
Explain a single anomaly.

**Returns**:
```python
{
    'top_features': [...],      # List of feature contributions
    'explanation': "...",        # Human-readable text
    'shap_values': [...],        # Raw SHAP values
    'confidence': 0.85           # 0-1 confidence score
}
```

### `get_anomaly_explanations(anomalies_df, telemetry_data=None)`
Explain all anomalies in batch.

**Returns**: DataFrame with columns:
- `explanation`: Human-readable explanation
- `top_feature_1/2/3`: Top contributing features
- `contribution_1/2/3`: Contribution percentages (0-1)
- `confidence`: Confidence score (0-1)

## Dashboard Features

**Tab 1: Statistical Detection**
- Z-score based anomaly detection
- Simple, fast outlier detection
- Good for single-metric anomalies

**Tab 2: ML Detection with SHAP**
- Isolation Forest anomaly detection
- SHAP explanations for each anomaly
- Feature importance bar charts
- Human-readable explanations
- Confidence scores

## Example Output

```
Anomaly #1 - Lap 10
Anomaly Score: -0.6739
Confidence: 82.61%

Explanation: S1 Seconds 45% too high, Top Speed 35% too low

Top Contributing Features:
  1. S1_SECONDS: 45.2%
  2. TOP_SPEED: 35.1%
  3. S2_SECONDS: 19.7%
```

## File Locations

```
src/tactical/anomaly_detector.py    # Enhanced module with SHAP
examples/shap_anomaly_demo.py       # Demo script
dashboard/pages/tactical.py         # Dashboard integration
docs/SHAP_EXPLAINABILITY.md        # Full documentation
```

## Troubleshooting

**SHAP not installed**
→ System gracefully degrades, shows warning

**TreeExplainer fails**
→ Automatically falls back to KernelExplainer

**No anomalies detected**
→ Try adjusting `contamination` parameter (0.05-0.15)

**Slow performance**
→ TreeExplainer is fast (~100ms), KernelExplainer slower (~5s)

## Common Parameters

**contamination**: Expected anomaly rate (default: 0.1)
- `0.05` = Expect 5% anomalies (stricter)
- `0.10` = Expect 10% anomalies (balanced)
- `0.15` = Expect 15% anomalies (more sensitive)

## Next Steps

1. Install SHAP: `pip install shap==0.44.0`
2. Run demo: `python examples/shap_anomaly_demo.py`
3. Try dashboard: `streamlit run dashboard/app.py`
4. Read docs: `docs/SHAP_EXPLAINABILITY.md`

---
For detailed documentation, see: `docs/SHAP_EXPLAINABILITY.md`
