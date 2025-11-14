# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RaceIQ Pro is an AI-powered racing analytics platform for the Toyota GR Cup that combines tactical driver coaching with strategic race planning. It transforms raw telemetry and timing data into actionable insights using machine learning, statistical analysis, and optimization algorithms.

## Key Commands

### Running the Application
```bash
# Main dashboard (primary interface)
streamlit run dashboard/app.py

# Testing commands
python verify_structure.py          # Verify installation and structure
python test_functional.py          # Test with actual data
python test_platform.py            # Comprehensive platform testing
python test_enhancements.py        # Test advanced features (SHAP, Bayesian, etc.)

# Analysis script
python run_comprehensive_analysis.py  # Run full analysis pipeline
```

### Development Workflow
```bash
# Install dependencies
pip install -r requirements.txt

# Run specific demos
python examples/shap_anomaly_demo.py
python examples/bayesian_strategy_demo.py
python examples/race_simulation_demo.py
python examples/track_map_demo.py
```

## Architecture Overview

The system follows a modular architecture with three main analytical layers connected by an integration engine:

### Core Modules

1. **Tactical Module** (`src/tactical/`)
   - **Section Analysis**: Breaks down lap performance by track sections to identify improvement areas
   - **Anomaly Detection**: Uses Isolation Forest and optional LSTM models to detect performance outliers
   - **Racing Line Reconstruction**: Physics-based reconstruction of driver racing lines from telemetry
   - **Optimal Ghost**: Theoretical best lap construction from driver's best sections

2. **Strategic Module** (`src/strategic/`)
   - **Tire Degradation Model**: Predicts tire performance decay with weather and track condition adjustments
   - **Pit Strategy Optimizer**: Monte Carlo simulations with Bayesian uncertainty quantification
   - **Race Simulation**: Multi-driver race simulation with overtaking dynamics and position modeling
   - **Strategy Patterns**: Undercut/overcut analysis and multi-stop optimization

3. **Integration Engine** (`src/integration/`)
   - **Intelligence Engine**: Connects tactical findings to strategic implications
   - **Causal Analysis**: DoWhy-based causal inference for "what-if" scenarios
   - **Weather Adjuster**: Real-time adjustments based on track conditions
   - **Recommendation Builder**: Prioritized, actionable insights from cross-module analysis

### Data Pipeline (`src/pipeline/`)
- **DataLoader**: Handles multi-track, multi-format CSV data loading
- **FeatureEngineer**: Creates derived features for ML models
- **Validator**: Ensures data quality and consistency

### Dashboard (`dashboard/`)
The Streamlit-based dashboard provides multiple specialized views:
- **Overview Page**: Data selection and high-level metrics
- **Tactical Analysis**: Section performance, anomaly detection, SHAP explanations
- **Strategic Analysis**: Pit strategy with Bayesian confidence intervals
- **Race Simulator**: Interactive multi-driver race simulation
- **Integrated Insights**: Cross-module intelligence and recommendations

## Data Structure

The project expects race data organized by track in the `Data/` directory:
```
Data/
├── barber/           # Each track has lap times, sections, weather
├── COTA/
├── Sonoma/
├── indianapolis/
├── road-america/
├── sebring/
└── virginia-international-raceway/
```

Key data files per race:
- `*_lap_time.csv`: Lap timing data
- `*_AnalysisEnduranceWithSections*.CSV`: Section-by-section telemetry
- `*_Weather*.CSV`: Weather conditions
- Large telemetry files (800MB-3GB) are gitignored

## Advanced Features Integration

### SHAP Explainability
- Located in `src/tactical/anomaly_detector.py`
- Provides feature importance for anomaly detection
- Accessed via dashboard Tactical Analysis page

### Bayesian Uncertainty
- Implemented in `src/strategic/strategy_optimizer.py`
- Methods with `_with_uncertainty` suffix provide confidence intervals
- Visualized in Strategic Analysis dashboard

### Weather Integration
- Core logic in `src/integration/weather_adjuster.py`
- Adjusts tire degradation and lap times based on conditions
- Auto-loaded when weather data is available

### Causal Inference
- Module: `src/integration/causal_analysis.py`
- Uses DoWhy for rigorous what-if analysis
- Pre-configured templates for common racing questions

### Track Visualization
- Track layouts in `src/utils/track_layouts.py`
- Interactive maps in dashboard pages
- Heatmap overlays for performance visualization

## Testing Strategy

The project has multiple testing entry points:
1. `verify_structure.py` - Quick structural validation
2. `test_functional.py` - Tests with mock data for CI/CD
3. `test_platform.py` - Comprehensive integration testing
4. `test_enhancements.py` - Advanced feature testing
5. `examples/*.py` - Feature-specific demonstrations

## Key Design Patterns

### Cross-Module Intelligence
The Integration Engine (`src/integration/intelligence_engine.py`) is the core differentiator. It connects tactical anomalies to strategic implications through:
- `connect_anomaly_to_strategy()`: Links driver issues to pit timing
- `analyze_section_impact_on_strategy()`: Quantifies improvement benefits
- `generate_unified_recommendations()`: Creates prioritized action items

### Uncertainty Quantification
Strategic recommendations include confidence intervals using:
- Monte Carlo simulation for variability
- Bayesian methods for statistical rigor
- Multiple confidence levels (80%, 90%, 95%)

### Modular Analytics
Each module can operate independently but gains power through integration:
- Tactical insights inform tire degradation rates
- Strategic constraints guide tactical focus areas
- Weather adjustments cascade through all calculations

## Important Conventions

1. **Data Loading**: Always use `DataLoader` class for consistency
2. **Confidence Scores**: All predictions should include uncertainty measures
3. **Visualization**: Use Plotly for interactive, matplotlib for static plots
4. **Error Handling**: Graceful degradation when optional features unavailable
5. **Caching**: Use Streamlit's `@st.cache_data` for expensive computations

## Documentation Locations

- Main documentation: `docs/` directory
- Module-specific docs: README.md in each module directory
- Quick references: `docs/quick-reference/` for feature guides
- API examples: `examples/` directory with working demos