# RaceIQ Pro - Quick Start Guide

Get up and running with RaceIQ Pro in minutes.

## Installation

### 1. Create a Virtual Environment (Recommended)

To avoid dependency conflicts with other packages, use a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you see dependency conflict warnings, they're likely from other packages in your global environment. Using a virtual environment (step 1) avoids these conflicts.

### 3. Verify Installation

```bash
python verify_structure.py
```

This checks:

- Directory structure
- File existence
- Module imports
- Sample data files

---

## Quick Start

### Option 1: Launch Dashboard (Recommended)

```bash
streamlit run dashboard/app.py
```

Navigate to:

- **Overview** - Project overview and data selection
- **Tactical Analysis** - Section analysis, anomaly detection, racing line
- **Strategic Analysis** - Pit strategy, tire degradation, Bayesian optimization
- **Race Simulator** - Multi-driver race simulation
- **Integrated Insights** - Cross-module intelligence

### Option 2: Use Python API

```python
from src.pipeline.data_loader import DataLoader
from src.tactical.section_analyzer import SectionAnalyzer
from src.strategic.strategy_optimizer import PitStrategyOptimizer

# Load data
loader = DataLoader()
lap_times = loader.load_lap_time_data()
sections = loader.load_section_analysis()

# Tactical analysis
analyzer = SectionAnalyzer(lap_times, sections)
section_performance = analyzer.analyze_driver('42')

# Strategic analysis
optimizer = PitStrategyOptimizer()
strategy = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data=lap_times,
    tire_model={'degradation_rate': 0.05},
    race_length=25
)
```

---

## Project Structure

```
ToyotaGR/
├── src/
│   ├── pipeline/          # Data loading, validation, feature engineering
│   ├── tactical/          # Section analysis, anomaly detection, racing line
│   ├── strategic/         # Pit strategy, tire degradation, optimization
│   ├── integration/       # Cross-module intelligence
│   └── utils/             # Metrics, visualizations, constants
├── dashboard/             # Streamlit dashboard
├── examples/              # Example scripts and demos
├── tests/                 # Unit tests
├── Data/                  # Race data files
└── docs/                  # Documentation
```

---

## Key Modules

### Data Pipeline (`src/pipeline/`)

**DataLoader** - Load CSV files from race data directories

```python
from src.pipeline.data_loader import DataLoader

loader = DataLoader()
lap_times = loader.load_lap_time_data()
sections = loader.load_section_analysis()
telemetry = loader.load_telemetry_data()
weather = loader.load_weather_data()
```

**DataValidator** - Validate data quality

```python
from src.pipeline.validator import DataValidator

validator = DataValidator()
results = validator.validate_all(data)
print(validator.get_summary_report())
```

**FeatureEngineer** - Create derived metrics

```python
from src.pipeline.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
engineered_data = engineer.engineer_all_features(data)
```

### Tactical Analysis (`src/tactical/`)

**SectionAnalyzer** - Analyze section-by-section performance

```python
from src.tactical.section_analyzer import SectionAnalyzer

analyzer = SectionAnalyzer(lap_times, sections)
performance = analyzer.analyze_driver('42')
```

**AnomalyDetector** - Detect unusual laps/patterns

```python
from src.tactical.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
anomalies = detector.detect_pattern_anomalies(telemetry, contamination=0.1)
explanations = detector.get_anomaly_explanations(anomalies, telemetry)
```

**RacingLineReconstructor** - Reconstruct racing lines from telemetry

```python
from src.tactical.racing_line import RacingLineReconstructor

reconstructor = RacingLineReconstructor()
line = reconstructor.reconstruct_line(telemetry)
comparison = reconstructor.compare_racing_lines(telemetry1, telemetry2)
```

### Strategic Analysis (`src/strategic/`)

**PitStrategyOptimizer** - Optimize pit stop timing with Bayesian uncertainty

```python
from src.strategic.strategy_optimizer import PitStrategyOptimizer

optimizer = PitStrategyOptimizer(pit_loss_seconds=25.0)
result = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data=lap_times,
    tire_model={'degradation_rate': 0.05},
    race_length=25
)
```

**TireDegradationModel** - Model tire degradation over race distance

```python
from src.strategic.tire_degradation import TireDegradationModel

model = TireDegradationModel()
model.fit(lap_times)
predicted_time = model.predict(lap_number=15)
```

**RaceSimulator** - Simulate multi-driver races

```python
from src.strategic.race_simulation import MultiDriverRaceSimulator

simulator = MultiDriverRaceSimulator(race_length=25)
result = simulator.simulate_race(drivers_data, strategies)
```

### Integration Engine (`src/integration/`)

**IntegrationEngine** - Cross-module intelligence and recommendations

```python
from src.integration.intelligence_engine import IntegrationEngine

engine = IntegrationEngine()
insights = engine.generate_insights(tactical_data, strategic_data)
recommendations = engine.build_recommendations(insights)
```

---

## Common Tasks

### 1. Analyze a Driver's Performance

```python
from src.pipeline.data_loader import DataLoader
from src.tactical.section_analyzer import SectionAnalyzer

# Load data
loader = DataLoader()
lap_times = loader.load_lap_time_data()
sections = loader.load_section_analysis()

# Analyze driver
analyzer = SectionAnalyzer(lap_times, sections)
performance = analyzer.analyze_driver('42')

print(f"Best lap: {performance['best_lap_time']}")
print(f"Average: {performance['avg_lap_time']}")
print(f"Consistency: {performance['consistency_score']}")
```

### 2. Detect Anomalies with SHAP Explanations

```python
from src.tactical.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect_pattern_anomalies(telemetry, contamination=0.1)
anomalies = result[result['is_anomaly'] == -1]

# Get SHAP explanations
explained = detector.get_anomaly_explanations(anomalies, result)
print(explained[['LAP_NUMBER', 'explanation', 'confidence']])
```

### 3. Optimize Pit Strategy

```python
from src.strategic.strategy_optimizer import PitStrategyOptimizer

optimizer = PitStrategyOptimizer()
result = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data=lap_times,
    tire_model={'degradation_rate': 0.05},
    race_length=25
)

print(f"Optimal pit lap: {result['optimal_lap']}")
print(f"90% confidence: Laps {result['confidence_90'][0]}-{result['confidence_90'][1]}")
print(f"Risk level: {result['risk_assessment']['risk_level']}")
```

### 4. Simulate a Race

```python
from src.strategic.race_simulation import MultiDriverRaceSimulator

simulator = MultiDriverRaceSimulator(race_length=25)

drivers_data = {
    'A': {'base_lap_time': 95.0, 'tire_deg_rate': 0.05},
    'B': {'base_lap_time': 95.2, 'tire_deg_rate': 0.05}
}

strategies = {
    'A': {'pit_laps': [12]},
    'B': {'pit_laps': [14]}
}

result = simulator.simulate_race(drivers_data, strategies)
print(result['final_results'])
```

---

## Example Scripts

Run example scripts to see features in action:

```bash
# Bayesian strategy demo
python examples/bayesian_strategy_demo.py

# SHAP anomaly detection
python examples/shap_anomaly_demo.py

# Racing line reconstruction
python examples/racing_line_demo.py

# Race simulation
python examples/race_simulation_demo.py

# Weather integration
python examples/weather_integration_demo.py

# Causal analysis
python examples/causal_analysis_demo.py

# LSTM anomaly detection
python examples/lstm_anomaly_demo.py
```

---

## Data Files

Sample data is located in `Data/` directory:

- **Barber Motorsports Park**: `Data/barber/`
- **Circuit of the Americas**: `Data/COTA/`
- **Sonoma Raceway**: `Data/Sonoma/`
- **Road America**: `Data/road-america/`
- **Sebring**: `Data/sebring/`
- **Indianapolis**: `Data/indianapolis/`
- **Virginia International Raceway**: `Data/virginia-international-raceway/`

Each track directory contains:

- Lap time data (`*lap_time*.csv`)
- Section analysis (`*Sections*.CSV`)
- Telemetry data (`*telemetry*.csv`)
- Weather data (`*Weather*.CSV`)
- Race results (`*Results*.CSV`)

---

## Next Steps

1. **Explore the Dashboard** - Launch `streamlit run dashboard/app.py` and navigate through pages
2. **Run Examples** - Try the example scripts in `examples/`
3. **Read Documentation** - Check `docs/` for detailed guides
4. **Review Quick References** - See `docs/quick-reference/` for feature-specific guides

---

## Troubleshooting

### Dependency Conflict Warnings

If you see warnings like:

```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**These are warnings, not errors.** The installation succeeded. To avoid conflicts:

1. Use a virtual environment (recommended - see Installation step 1)
2. Or ignore the warnings if RaceIQ Pro works correctly

### Import Errors

```bash
# Ensure you're in the project root directory
cd /path/to/ToyotaGR

# Activate virtual environment (if using one)
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python verify_structure.py
```

### Data Loading Issues

- Check that data files exist in `Data/` directory
- Verify file naming conventions match expected patterns
- Check file encoding (should be UTF-8)

### Dashboard Not Starting

```bash
# Check Streamlit installation
pip install streamlit

# Try running directly
python -m streamlit run dashboard/app.py
```

---

## Getting Help

- **Documentation**: See `docs/README.md` for complete documentation index
- **Examples**: Check `examples/` directory for working code samples
- **Module READMEs**: Each module has its own README with API documentation
- **Quick References**: Feature-specific quick guides in `docs/quick-reference/`

---

**Version:** 1.0  
**Last Updated:** 2024
