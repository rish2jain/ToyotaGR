# RaceIQ Pro - Quick Start Guide

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Verify Installation
```bash
python verify_structure.py
```

### 2. Load and Analyze Data
```python
from pipeline.data_loader import DataLoader
from pipeline.validator import DataValidator
from pipeline.feature_engineer import FeatureEngineer

# Load sample data
loader = DataLoader()
data = loader.load_all_sample_data()

# Validate data quality
validator = DataValidator()
results = validator.validate_all(data)
print(validator.get_summary_report())

# Engineer features
engineer = FeatureEngineer()
engineered_data = engineer.engineer_all_features(data)
```

### 3. Calculate Metrics
```python
from utils.metrics import (
    calculate_theoretical_best_lap,
    calculate_driver_consistency,
    calculate_pace_analysis
)

section_df = data['section_analysis']

# Theoretical best laps
best_laps = calculate_theoretical_best_lap(section_df)
print(best_laps)

# Driver consistency
consistency = calculate_driver_consistency(section_df)
print(consistency)
```

### 4. Create Visualizations
```python
from utils.visualization import plot_lap_times, plot_sector_comparison
import matplotlib.pyplot as plt

# Plot lap times
fig = plot_lap_times(
    section_df,
    driver_col='DRIVER_NUMBER',
    lap_col='LAP_NUMBER',
    time_col='LAP_TIME_SECONDS'
)
plt.show()
```

### 5. Run Tests
```bash
python tests/test_data_loader.py
```

### 6. Explore in Jupyter
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 7. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## Key Modules

- **Pipeline**: `src/pipeline/` - Data loading, validation, feature engineering
- **Utils**: `src/utils/` - Metrics, visualizations, constants
- **Tactical**: `src/tactical/` - Section analysis, anomaly detection, optimal ghost
- **Strategic**: `src/strategic/` - Pit stops, tire degradation, strategy optimization
- **Integration**: `src/integration/` - Cross-module intelligence

See SETUP_SUMMARY.md for detailed documentation.
