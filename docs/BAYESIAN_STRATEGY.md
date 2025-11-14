# Bayesian Uncertainty Quantification for Pit Strategy Optimization

Complete guide to Bayesian pit strategy optimization in RaceIQ Pro, including implementation details, workflow, and usage.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture & Workflow](#system-architecture--workflow)
3. [Implementation Details](#implementation-details)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Usage Guide](#usage-guide)
6. [Dashboard Integration](#dashboard-integration)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Bayesian uncertainty quantification feature provides pit strategy recommendations with explicit confidence intervals and risk assessment. Instead of recommending a single pit lap (e.g., "Pit on lap 14"), the Bayesian approach provides:

- **Optimal window:** "90% confident: pit between laps 12-15"
- **Uncertainty level:** "7.4% relative uncertainty"
- **Risk assessment:** "MODERATE - reasonable confidence with timing flexibility"

### Key Advantages

1. **Explicit Uncertainty Quantification** - Not just "pit on lap 14", but "90% confident: pit between laps 12-15"
2. **Incorporates Prior Knowledge** - Leverages racing experience and updates beliefs with new data
3. **Risk-Aware Decision Making** - Quantifies strategic risk and helps teams make informed choices
4. **Interpretable Confidence Intervals** - Credible intervals have intuitive interpretation
5. **Adaptive to Data Availability** - Works with limited data (prior dominates) and improves with more data

---

## System Architecture & Workflow

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     RACEIQ PRO DASHBOARD                        │
│                   (dashboard/pages/strategic.py)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ User selects driver
                              │ System collects race data
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                             │
│                                                                 │
│  • Parse lap times from race data                              │
│  • Build tire degradation model                                │
│  • Detect pit stops                                            │
│  • Calculate degradation rate via linear regression            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              BAYESIAN STRATEGY OPTIMIZER                        │
│         (src/strategic/strategy_optimizer.py)                   │
│                                                                 │
│  calculate_optimal_pit_window_with_uncertainty()                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│   PRIOR DISTRIBUTION    │    │  MONTE CARLO LIKELIHOOD  │
│                         │    │                         │
│ • Mean: 60% of race     │    │ • Simulate 100 races    │
│ • Std: ±3 laps          │    │ • Test each pit lap     │
│ • Based on experience   │    │ • Account for tire deg  │
└─────────────────────────┘    └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   BAYESIAN UPDATE             │
              │   _update_posterior()         │
              │                               │
              │   Posterior = Prior × Likelihood
              │                               │
              │   Using conjugate normal-normal
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   POSTERIOR DISTRIBUTION      │
              │                               │
              │   • Mean: Optimal pit lap     │
              │   • Std: Uncertainty          │
              │   • Samples: 1000 draws       │
              └───────────────────────────────┘
                              │
                              ▼
      ┌───────────────────────┴───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ CONFIDENCE  │    │  VISUALIZATION  │    │ RISK ASSESSMENT  │
│ INTERVALS   │    │     DATA        │    │                  │
│             │    │                 │    │                  │
│ • 80%       │    │ • PDF curve     │    │ • Risk level     │
│ • 90%       │    │ • Histogram     │    │ • Explanation    │
│ • 95%       │    │ • Violin plot   │    │ • Strategy note  │
└─────────────┘    └─────────────────┘    └──────────────────┘
```

### Data Flow Example

**Input Data:**
```python
race_data = {
    'LAP_NUMBER': [1, 2, 3, ..., 15],
    'LAP_TIME': ['1:35.234', '1:35.456', ..., '1:36.789'],
    'DRIVER_NUMBER': [10, 10, ..., 10]
}

tire_model = {
    'baseline_lap_time': 95.234,
    'degradation_rate': 0.08,
    'model_type': 'linear'
}

race_length = 25
```

**Processing Steps:**

1. **Prior Definition**
   ```python
   prior_mean = 25 * 0.6 = 15.0 laps
   prior_std = 3.0 laps
   prior_precision = 1 / 9.0 = 0.111
   ```

2. **Monte Carlo Simulation**
   ```python
   For pit_lap in [5, 6, 7, ..., 22]:
       Simulate 100 races
       Calculate mean race time
   
   Results:
   Lap 14: 2413.56s ± 1.05s
   Lap 15: 2412.89s ± 1.02s  ← OPTIMAL
   Lap 16: 2413.12s ± 1.03s
   ```

3. **Likelihood Extraction**
   ```python
   likelihood_mean = 15  (optimal lap from simulation)
   likelihood_std = 1.0  (spread of competitive laps)
   likelihood_precision = 1 / 1.0 = 1.0
   ```

4. **Posterior Calculation**
   ```python
   posterior_precision = 0.111 + 1.0 = 1.111
   posterior_std = sqrt(1 / 1.111) = 0.95 laps
   
   posterior_mean = (0.111 * 15.0 + 1.0 * 15) / 1.111
                  = 15.0 laps
   ```

5. **Confidence Intervals**
   ```python
   95% CI: norm.interval(0.95, 15.0, 0.95)
        = (13.1, 16.9) → Laps 13-17
   
   90% CI: norm.interval(0.90, 15.0, 0.95)
        = (13.4, 16.6) → Laps 13-17
   
   80% CI: norm.interval(0.80, 15.0, 0.95)
        = (13.8, 16.2) → Laps 14-16
   ```

6. **Risk Assessment**
   ```python
   posterior_std = 0.95 < 2.0
   → MODERATE risk
   
   time_spread = 2413.56 - 2412.89 = 0.67s < 1.0s
   → "Pit timing not critical - minimal time difference"
   ```

**Output:**
```python
{
    'optimal_lap': 15,
    'confidence_95': (13, 17),
    'confidence_90': (13, 17),
    'confidence_80': (14, 16),
    'posterior_mean': 15.0,
    'posterior_std': 0.95,
    'uncertainty': 0.063,  # 6.3%
    'risk_assessment': {
        'risk_level': 'MODERATE',
        'explanation': 'Reasonable confidence in pit window, some timing flexibility',
        'strategy_note': 'Pit timing not critical - minimal time difference'
    }
}
```

---

## Implementation Details

### Enhanced Strategy Optimizer (`src/strategic/strategy_optimizer.py`)

#### `calculate_optimal_pit_window_with_uncertainty()`

**Purpose:** Calculate optimal pit window with full Bayesian uncertainty quantification

**Approach:**
- **Prior Distribution:** Normal distribution over pit lap times based on racing experience (typically 60% through race, ±3 laps uncertainty)
- **Likelihood:** Derived from Monte Carlo simulations of different pit lap strategies
- **Posterior:** Conjugate normal-normal update combining prior knowledge with simulation data

**Returns:**
```python
{
    'optimal_lap': int,              # Best estimate (posterior mean)
    'confidence_95': (lower, upper),  # 95% credible interval
    'confidence_90': (lower, upper),  # 90% credible interval
    'confidence_80': (lower, upper),  # 80% credible interval
    'posterior_mean': float,          # Mean of posterior distribution
    'posterior_std': float,           # Standard deviation
    'uncertainty': float,             # Relative uncertainty (std/mean)
    'posterior_samples': list,        # 1000 samples for visualization
    'risk_assessment': dict,          # Comprehensive risk analysis
    'simulation_results': dict        # Full simulation data
}
```

**Key Features:**
- Uses scipy.stats only (no pymc3 dependency)
- Conjugate prior approach for analytical solutions
- Provides multiple confidence levels (80%, 90%, 95%)
- Quantifies uncertainty explicitly
- Risk-aware decision making

#### `visualize_posterior_distribution()`

**Purpose:** Create visualization data for posterior distribution

**Returns:**
```python
{
    'histogram': {
        'bin_centers': list,
        'counts': list,
        'bin_edges': list
    },
    'pdf': {
        'x': list,  # 100 points
        'y': list   # probability density
    },
    'samples': list,  # 1000 posterior samples
    'confidence_intervals': {
        '95%': (lower, upper),
        '90%': (lower, upper),
        '80%': (lower, upper)
    },
    'optimal_lap': int,
    'risk_assessment': dict
}
```

**Visualizations Supported:**
- Violin plots of posterior distribution
- Probability density function (PDF) curves
- Confidence interval bars
- Histogram of posterior samples

#### `_run_simulations()`

**Purpose:** Run Monte Carlo simulations for each candidate pit lap

**Process:**
1. Define candidate pit laps (earliest: lap 5, latest: race_length - 3)
2. For each candidate lap:
   - Simulate 100 complete race scenarios
   - Account for tire degradation before and after pit
   - Add random variation to lap times
3. Calculate statistics: mean, std, samples

#### `_update_posterior()`

**Purpose:** Update posterior distribution using conjugate normal-normal model

**Bayesian Math:**
```
Prior:      N(μ₀, σ₀²)
Likelihood: N(μₗ, σₗ²)
Posterior:  N(μₚ, σₚ²)

where:
  τ₀ = 1/σ₀²  (prior precision)
  τₗ = 1/σₗ²  (likelihood precision)
  τₚ = τ₀ + τₗ (posterior precision)

  μₚ = (τ₀·μ₀ + τₗ·μₗ) / τₚ  (weighted average)
  σₚ² = 1/τₚ
```

**Key Insights:**
- More data → higher likelihood precision → posterior closer to likelihood
- Less data → prior dominates → posterior closer to prior
- Uncertainty decreases as more laps are completed

#### `_assess_uncertainty_risk()`

**Purpose:** Assess strategic risk based on posterior uncertainty

**Risk Levels:**

| Posterior Std | Risk Level | Interpretation |
|--------------|-----------|----------------|
| < 1.0 laps   | LOW       | Well-defined optimal window, high confidence |
| 1.0-2.0 laps | MODERATE  | Reasonable confidence, some timing flexibility |
| 2.0-3.0 laps | ELEVATED  | Significant uncertainty, monitor closely |
| > 3.0 laps   | HIGH      | Large uncertainty, timing highly sensitive |

**Strategy Notes:**
- Time spread < 1s: Pit timing not critical
- Time spread 1-3s: Moderate advantage from optimal timing
- Time spread > 3s: Critical to hit optimal window

---

## Mathematical Foundation

### Conjugate Normal-Normal Model

**Prior:**
```
π(μ) ~ N(μ₀, σ₀²)
```

**Likelihood (from data):**
```
p(x|μ) ~ N(μ, σ²)
```

**Posterior:**
```
π(μ|x) ~ N(μₚ, σₚ²)

where:
  τ₀ = 1/σ₀²  (prior precision)
  τ  = 1/σ²   (data precision)

  τₚ = τ₀ + τ

  μₚ = (τ₀μ₀ + τx̄) / τₚ

  σₚ² = 1/τₚ
```

### Why Conjugate Priors?

1. **Analytical Solution:** No MCMC needed, instant computation
2. **Interpretable:** Precision naturally combines prior + data
3. **Efficient:** Suitable for real-time race strategy
4. **Theoretically Sound:** Proper Bayesian inference

### Credible Intervals

```
P(μ ∈ [L, U] | data) = α

For α = 0.90:
  L = μₚ - 1.645σₚ
  U = μₚ + 1.645σₚ
```

Direct probability interpretation: "90% probability the true optimal lap is in this interval"

### Comparison: Traditional vs Bayesian

| Feature | Traditional Monte Carlo | Bayesian Approach |
|---------|------------------------|-------------------|
| Output | Single optimal lap | Posterior distribution |
| Uncertainty | Standard deviation | Credible intervals |
| Confidence | Implicit | Explicit (80%, 90%, 95%) |
| Prior Knowledge | Not used | Incorporated |
| Risk Assessment | Manual | Automated |
| Interpretability | Point estimate | Probability statements |
| Data Requirements | Moderate | Works with limited data |
| Computational Cost | Low | Low (conjugate priors) |

---

## Usage Guide

### Basic Usage

```python
from src.strategic.strategy_optimizer import PitStrategyOptimizer

# Initialize optimizer
optimizer = PitStrategyOptimizer(
    pit_loss_seconds=25.0,
    simulation_iterations=100,
    uncertainty_model='bayesian'
)

# Calculate optimal pit window with uncertainty
result = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data=driver_data,
    tire_model=tire_degradation_model,
    race_length=25
)

# Get recommendation
print(f"Optimal pit lap: {result['optimal_lap']}")
print(f"90% confidence: Laps {result['confidence_90'][0]}-{result['confidence_90'][1]}")
print(f"Risk level: {result['risk_assessment']['risk_level']}")
```

### Visualization

```python
# Get visualization data
viz_data = optimizer.visualize_posterior_distribution(result)

# Create plots (in dashboard)
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Violin(
    y=result['posterior_samples'],
    name='Posterior Distribution',
    box_visible=True,
    meanline_visible=True
))
```

### Risk-Based Strategy

```python
risk = result['risk_assessment']

if risk['risk_level'] == 'LOW':
    # High confidence - stick to optimal lap
    recommended_lap = result['optimal_lap']
elif risk['risk_level'] == 'MODERATE':
    # Use 90% confidence interval for flexibility
    window = result['confidence_90']
    recommended_lap = f"Laps {window[0]}-{window[1]}"
else:
    # High uncertainty - be conservative
    window = result['confidence_95']
    recommended_lap = f"Laps {window[0]}-{window[1]} (conservative)"
```

### Interpreting Confidence Intervals

**Example:** "90% Confidence: Laps 12-15"

**Means:** We are 90% certain the optimal pit lap is between 12 and 15.

**Strategy:**
- **Ideal:** Pit on lap 13 or 14 (center of window)
- **Acceptable:** Any lap from 12-15
- **Avoid:** Pitting outside this window (only 10% chance of being optimal)

### Risk Assessment Guide

#### 🟢 LOW Risk
- **Posterior Std:** < 1.0 laps
- **What it means:** Optimal window is well-defined with high confidence
- **Strategy:** Stick to the optimal lap for best results
- **Example:** "Optimal: Lap 15, 90% CI: 14-16" (tight window)

#### 🟡 MODERATE Risk
- **Posterior Std:** 1.0-2.0 laps
- **What it means:** Reasonable confidence, some timing flexibility
- **Strategy:** Use 90% confidence window, monitor tire condition
- **Example:** "Optimal: Lap 14, 90% CI: 12-16" (moderate window)

#### 🟠 ELEVATED Risk
- **Posterior Std:** 2.0-3.0 laps
- **What it means:** Significant uncertainty, monitor closely
- **Strategy:** Be prepared to adjust, watch tire degradation carefully
- **Example:** "Optimal: Lap 13, 90% CI: 10-16" (wide window)

#### 🔴 HIGH Risk
- **Posterior Std:** > 3.0 laps
- **What it means:** Large uncertainty, timing highly sensitive
- **Strategy:** Very conservative, use 95% confidence, adapt to real-time
- **Example:** "Optimal: Lap 15, 90% CI: 9-21" (very wide)

---

## Dashboard Integration

### Enhanced Pit Window Analysis

**New Features:**

#### Bayesian Pit Strategy Recommendation
- Displays optimal pit lap with posterior mean
- Shows uncertainty percentage
- Color-coded risk level indicators:
  - 🟢 LOW
  - 🟡 MODERATE
  - 🟠 ELEVATED
  - 🔴 HIGH

#### Interactive Confidence Level Slider
```python
confidence_level = st.select_slider(
    "Select Confidence Level",
    options=[80, 90, 95],
    value=90
)
```
- Adjust confidence level dynamically
- See how interval width changes
- Understand trade-off between confidence and precision

#### Confidence Intervals Display
- Table showing all three confidence levels
- Lower bound, upper bound, and window size
- Clear interpretation: "90% confident the optimal pit lap falls within this range"

#### Posterior Distribution Visualizations

**1. Violin Plot**
- Shows full distribution of posterior samples
- Box plot overlay for quartiles
- Mean line indicator
- Confidence interval markers

**2. PDF Curve**
- Smooth probability density function
- Shaded confidence interval regions
- Optimal lap marked with star
- Interactive hovering for exact probabilities

**3. Simulation Results Comparison**
- Expected race time vs pit lap
- Error bars showing simulation uncertainty
- Optimal lap highlighted
- Confidence interval shaded region

#### Risk Assessment Panel
**Left Column:**
- Risk level with color-coded indicator
- Detailed explanation
- Strategic recommendation

**Right Column:**
- Posterior standard deviation
- Relative uncertainty percentage
- Time spread across strategies

---

## Examples

### Example 1: Point Estimate vs Bayesian Approach

Side-by-side comparison:
- Traditional Monte Carlo (single optimal lap)
- Bayesian approach (confidence intervals + uncertainty)
- Shows advantages of Bayesian method

### Example 2: Uncertainty Narrows with More Data

Tests with 5, 10, 15, 20 laps of data:
- Demonstrates posterior std decreasing
- Shows confidence intervals narrowing
- Illustrates learning from data

### Example 3: Understanding Confidence Intervals

Explains interpretation of credible intervals:
- Practical strategy recommendations
- When to use each confidence level
- Risk vs precision trade-off

### Example 4: Visualizing the Posterior Distribution

Creates publication-quality matplotlib figures:
- PDF with shaded confidence intervals
- Histogram of posterior samples
- Saves visualization to disk

### Example 5: Risk Assessment in Different Scenarios

- High certainty scenario (20 laps, stable conditions)
- Moderate uncertainty (12 laps, mid-race)
- High uncertainty (5 laps, variable conditions)
- Shows risk assessment in action

**Run Examples:**
```bash
python examples/bayesian_strategy_demo.py
python examples/test_bayesian_strategy.py
```

---

## Troubleshooting

### Issue: Very Wide Confidence Intervals
**Cause:** Limited data or high variability in lap times
**Solution:**
- Collect more laps before deciding
- Use 95% confidence for safety
- Monitor tire condition closely

### Issue: Risk Level is HIGH
**Cause:** Significant uncertainty in optimal pit timing
**Solution:**
- Be prepared to adjust strategy
- Watch for changing conditions (weather, tire wear)
- Consider conservative pit timing

### Issue: Uncertainty Not Decreasing
**Cause:** Inconsistent lap times or changing conditions
**Solution:**
- Check for traffic, weather changes, driver errors
- May need to reassess tire model
- Consider current conditions vs historical data

### Best Practices

**✓ DO**
- Use 90% confidence as default
- Consider risk level in decision making
- Monitor how uncertainty changes over race
- Combine with driver feedback on tire condition
- Adjust confidence level based on race criticality

**✗ DON'T**
- Ignore uncertainty and use only optimal lap
- Pit outside confidence interval without good reason
- Overlook risk assessment warnings
- Rely solely on early-race predictions (high uncertainty)
- Forget to update with latest lap data

---

## Quick Reference

For a quick reference card, see: [Bayesian Strategy Quick Reference](./quick-reference/BAYESIAN_STRATEGY_QUICK_REFERENCE.md)

---

**Version:** 1.0  
**Last Updated:** 2024  
**Module:** `src/strategic/strategy_optimizer.py`  
**Dashboard:** `dashboard/pages/strategic.py`

