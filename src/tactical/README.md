# Tactical Analysis Module

The Tactical Analysis Module provides comprehensive section-by-section performance analysis, anomaly detection, and driver coaching capabilities for RaceIQ Pro.

## Overview

This module enables detailed analysis of driver performance at the track section level, identifying strengths, weaknesses, and improvement opportunities through:

- **Optimal Ghost Analysis**: Create theoretical perfect laps by combining the best section times from all drivers
- **Anomaly Detection**: Multi-tier detection of unusual patterns in telemetry and performance data
- **Section Analysis**: Statistical analysis of track sections to identify driver strengths and areas for improvement

## Modules

### 1. OptimalGhostAnalyzer (`optimal_ghost.py`)

Creates optimal "ghost" laps and compares individual driver performance against this benchmark.

#### Key Features

- **create_optimal_ghost()**: Combines the fastest section times (default: top 5%) to create a theoretical perfect lap
- **analyze_driver_vs_ghost()**: Detailed comparison showing gaps and improvement opportunities
- **compare_multiple_drivers()**: Rank all drivers by their gap to the optimal ghost

#### Example Usage

```python
from src.tactical import OptimalGhostAnalyzer

# Initialize analyzer
analyzer = OptimalGhostAnalyzer()

# Create optimal ghost from all driver data
optimal_ghost = analyzer.create_optimal_ghost(section_data, percentile=95)

# Analyze a specific driver
driver_data = section_data[section_data['DRIVER_NUMBER'] == 123]
analysis = analyzer.analyze_driver_vs_ghost(driver_data, optimal_ghost)

# View top improvement opportunities
for opp in analysis['top_3_improvements']:
    print(f"{opp['section']}: +{opp['gap_seconds']:.3f}s ({opp['priority']})")
```

#### Output Structure

```python
{
    'driver_number': 123,
    'total_gap': 2.456,  # seconds
    'sections': {
        'S1': {
            'median_time': 27.234,
            'optimal_time': 26.961,
            'gap_seconds': 0.273,
            'gap_percent': 1.01
        },
        ...
    },
    'top_3_improvements': [
        {
            'section': 'S2',
            'gap_seconds': 0.856,
            'gap_percent': 1.98,
            'priority': 'HIGH'  # HIGH/MEDIUM/LOW
        },
        ...
    ]
}
```

### 2. AnomalyDetector (`anomaly_detector.py`)

Multi-tier anomaly detection for racing telemetry and performance data.

#### Key Features

**Tier 1: Statistical Methods**
- **detect_statistical_anomalies()**: Rolling z-score based anomaly detection
  - Configurable window size (default: 5 laps)
  - Threshold for standard deviations (default: 2.5σ)
  - Per-driver analysis to account for skill differences

**Tier 2: Machine Learning** (requires scikit-learn)
- **detect_pattern_anomalies()**: Isolation Forest for complex multivariate anomalies
  - Detects subtle patterns missed by statistical methods
  - Configurable contamination rate (default: 5%)

**Classification**
- **classify_anomaly_type()**: Categorizes anomalies as:
  - `brake_issue`: Braking anomalies
  - `throttle_issue`: Throttle/acceleration problems
  - `speed_anomaly`: Speed-related issues
  - `section_time_anomaly`: Section timing irregularities
  - `driver_error`: Multiple concurrent anomalies suggesting driver mistake
  - `unknown`: Cannot determine specific type

#### Example Usage

```python
from src.tactical import AnomalyDetector

detector = AnomalyDetector()

# Tier 1: Statistical anomaly detection
anomalies = detector.detect_statistical_anomalies(
    telemetry_data,
    window=5,
    threshold=2.5
)

# Classify anomaly types
anomalies['anomaly_type'] = detector.classify_anomaly_type(anomalies)

# Get summary
summary = detector.get_anomaly_summary(anomalies)
print(f"Anomaly rate: {summary['anomaly_rate']*100:.1f}%")

# Filter high priority anomalies (2+ metrics flagged)
high_priority = detector.filter_high_priority_anomalies(anomalies, min_anomaly_count=2)

# Tier 2: Machine learning detection (optional)
ml_anomalies = detector.detect_pattern_anomalies(telemetry_data, contamination=0.05)
```

#### Output Structure

The detector adds several columns to your dataframe:
- `{metric}_zscore`: Z-score for each metric
- `{metric}_anomaly`: Boolean flag for anomaly
- `anomaly_count`: Total anomalies per row
- `anomaly_type`: Classification of anomaly type

### 3. SectionAnalyzer (`section_analyzer.py`)

Comprehensive statistical analysis of track sections and driver performance.

#### Key Features

- **calculate_section_statistics()**: Comprehensive stats (mean, median, std, min/max, percentiles)
- **identify_driver_strengths()**: Find sections where driver excels (default: top 20%)
- **identify_improvement_areas()**: Sections with >2% gap to optimal ghost
- **analyze_section_consistency()**: Measure driver consistency using coefficient of variation
- **compare_driver_sections()**: Detailed driver vs field comparison
- **get_section_improvement_potential()**: Compare driver's median vs their personal best

#### Example Usage

```python
from src.tactical import SectionAnalyzer

analyzer = SectionAnalyzer()

# Calculate overall statistics
stats = analyzer.calculate_section_statistics(section_data)
print(f"S1 median: {stats['S1']['median']:.3f}s")

# Find driver strengths (top 20%)
strengths = analyzer.identify_driver_strengths(driver_data, all_data, top_percentile=20)
for strength in strengths:
    print(f"{strength['section']}: Top {strength['percentile_rank']:.1f}%")

# Identify improvement areas (>2% gap)
improvements = analyzer.identify_improvement_areas(driver_data, optimal_ghost, gap_threshold=2.0)
for area in improvements:
    print(f"{area['section']}: {area['gap_percent']:.1f}% gap ({area['priority']})")

# Analyze consistency
consistency = analyzer.analyze_section_consistency(driver_data)
for section, metrics in consistency.items():
    print(f"{section}: {metrics['consistency_score']:.1f}/100")

# Self-improvement potential
potential = analyzer.get_section_improvement_potential(driver_data)
for section, seconds in potential.items():
    print(f"{section}: {seconds:.3f}s potential improvement")
```

## Data Requirements

All modules expect pandas DataFrames with the following structure:

### Required Columns
- `DRIVER_NUMBER`: Integer driver identifier
- `LAP_NUMBER`: Lap number
- Section time columns ending with `_SECONDS` (e.g., `S1_SECONDS`, `S2_SECONDS`, `S3_SECONDS`)

### Optional Columns
- `TOP_SPEED`: Maximum speed
- Telemetry columns for anomaly detection (brake, throttle, etc.)
- Intermediate sector times (e.g., `IM1_time`, `IM2_time`)

### Example Data Structure

```csv
DRIVER_NUMBER,LAP_NUMBER,LAP_TIME,S1_SECONDS,S2_SECONDS,S3_SECONDS,TOP_SPEED
1,1,94.123,27.456,43.234,23.433,182.1
1,2,93.892,27.234,43.112,23.546,183.4
2,1,94.567,27.678,43.456,23.433,181.5
...
```

## Installation

### Required Dependencies

```bash
pip install pandas numpy scipy
```

### Optional Dependencies

For machine learning anomaly detection:
```bash
pip install scikit-learn
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Priority Levels

All modules use consistent priority classifications:

- **CRITICAL**: Gap ≥ 5% or 3+ anomalous metrics
- **HIGH**: Gap ≥ 3% or 2 anomalous metrics
- **MEDIUM**: Gap ≥ 2% or 1 anomalous metric
- **LOW**: Gap < 2%

## Demo Script

Run the complete demo to see all features in action:

```bash
python examples/tactical_analysis_demo.py
```

The demo includes:
1. Optimal ghost creation and driver comparison
2. Statistical and ML-based anomaly detection
3. Section analysis with strengths, improvements, and consistency metrics

## Performance Considerations

- **Statistical anomaly detection**: Fast, suitable for real-time analysis
- **Isolation Forest**: Moderate speed, best for batch analysis
- **Per-driver analysis**: Automatically parallelizable for large datasets
- All methods handle missing data gracefully

## Integration with RaceIQ Pro

The Tactical Analysis Module integrates with:

- **Strategic Module**: Optimal ghost insights inform tire strategy decisions
- **Integration Engine**: Anomalies trigger strategic recommendation adjustments
- **Visualization**: All outputs are dashboard-ready with priority levels

## Type Hints and Documentation

All functions include:
- Complete docstrings with examples
- Type hints for parameters and return values
- Detailed parameter descriptions
- Usage examples in docstrings

## Version

Current version: 1.0.0

## Authors

RaceIQ Pro Development Team
Toyota GR Cup Hackathon Project
