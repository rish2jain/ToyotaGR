# Bayesian Uncertainty Quantification for Pit Strategy Optimization

## Implementation Summary

This document summarizes the Bayesian uncertainty quantification features added to the RaceIQ Pro pit strategy optimizer for the Toyota GR Cup Hackathon.

---

## 1. Enhanced Strategy Optimizer (`src/strategic/strategy_optimizer.py`)

### New Methods Added

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

---

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

---

#### `_run_simulations()`

**Purpose:** Run Monte Carlo simulations for each candidate pit lap

**Process:**
1. Define candidate pit laps (earliest: lap 5, latest: race_length - 3)
2. For each candidate lap:
   - Simulate 100 complete race scenarios
   - Account for tire degradation before and after pit
   - Add random variation to lap times
3. Calculate statistics: mean, std, samples

---

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

---

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

## 2. Dashboard Integration (`dashboard/pages/strategic.py`)

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

## 3. Demonstration Examples

### `examples/bayesian_strategy_demo.py`

**Comprehensive demonstration with 5 demos:**

#### Demo 1: Point Estimate vs Bayesian Approach
- Side-by-side comparison
- Traditional Monte Carlo (single optimal lap)
- Bayesian approach (confidence intervals + uncertainty)
- Shows advantages of Bayesian method

#### Demo 2: Uncertainty Narrows with More Data
- Tests with 5, 10, 15, 20 laps of data
- Demonstrates posterior std decreasing
- Shows confidence intervals narrowing
- Illustrates learning from data

#### Demo 3: Understanding Confidence Intervals
- Explains interpretation of credible intervals
- Practical strategy recommendations
- When to use each confidence level
- Risk vs precision trade-off

#### Demo 4: Visualizing the Posterior Distribution
- Creates publication-quality matplotlib figures
- PDF with shaded confidence intervals
- Histogram of posterior samples
- Saves visualization to disk

#### Demo 5: Risk Assessment in Different Scenarios
- High certainty scenario (20 laps, stable conditions)
- Moderate uncertainty (12 laps, mid-race)
- High uncertainty (5 laps, variable conditions)
- Shows risk assessment in action

---

### `examples/test_bayesian_strategy.py`

**Quick automated test:**
- Generates synthetic race data
- Tests optimal pit window calculation
- Verifies visualization methods
- Validates all outputs
- Can be run in CI/CD pipeline

---

## 4. Technical Implementation Details

### Dependencies
- **scipy.stats:** For statistical distributions and intervals
- **numpy:** For numerical computations
- **pandas:** For data handling
- **plotly/matplotlib:** For visualizations

**Note:** No pymc3 required - uses analytical solutions via conjugate priors

### Bayesian Approach Advantages

1. **Explicit Uncertainty Quantification**
   - Not just "pit on lap 14"
   - But "90% confident: pit between laps 12-15"

2. **Incorporates Prior Knowledge**
   - Leverages racing experience
   - Updates beliefs with new data
   - Balances prior and likelihood

3. **Risk-Aware Decision Making**
   - Quantifies strategic risk
   - Helps teams make informed choices
   - Shows when to be conservative vs aggressive

4. **Interpretable Confidence Intervals**
   - Credible intervals have intuitive interpretation
   - "90% probability the true value is in this range"
   - Direct probability statements

5. **Adaptive to Data Availability**
   - Works with limited data (prior dominates)
   - Improves with more data (likelihood dominates)
   - Graceful handling of uncertainty

---

## 5. Usage Examples

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

---

## 6. Comparison: Traditional vs Bayesian

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

## 7. Statistical Rigor

### Bayesian Framework

**Prior Distribution:**
```python
prior_mean = race_length * 0.6  # Historical optimal around 60%
prior_std = 3.0                  # ±3 laps uncertainty
```

**Likelihood from Simulations:**
- Optimal lap from Monte Carlo: maximum likelihood estimate
- Spread of competitive laps: likelihood uncertainty
- Simulation variance: data quality indicator

**Posterior Update:**
- Conjugate normal-normal model
- Analytical solution (no MCMC needed)
- Efficient computation

### Interpretation

**Credible Intervals (Bayesian):**
- "90% probability the parameter is in this interval"
- Direct probability statement
- Accounts for all sources of uncertainty

**Confidence Intervals (Frequentist):**
- "90% of such intervals would contain the parameter"
- Indirect interpretation
- Long-run frequency property

**For strategy:** Credible intervals are more intuitive and actionable

---

## 8. Future Enhancements

### Potential Improvements

1. **Hierarchical Bayesian Model**
   - Learn from multiple drivers/races
   - Pool information across sessions
   - Driver-specific priors

2. **Dynamic Prior Updates**
   - Update prior based on track/weather
   - Incorporate real-time telemetry
   - Adapt to changing conditions

3. **Multi-Stop Strategies**
   - Extend to 2-stop, 3-stop strategies
   - Joint optimization of multiple pit laps
   - Account for correlations

4. **Competitor Modeling**
   - Bayesian game theory
   - Predict competitor strategies
   - Strategic undercut/overcut analysis

5. **Real-Time Updating**
   - Sequential Bayesian updates each lap
   - Online learning during race
   - Adaptive strategy recommendations

---

## 9. Key Files Modified/Created

### Modified Files
1. `/home/user/ToyotaGR/src/strategic/strategy_optimizer.py`
   - Added `calculate_optimal_pit_window_with_uncertainty()`
   - Added `visualize_posterior_distribution()`
   - Added `_run_simulations()`
   - Added `_update_posterior()`
   - Added `_assess_uncertainty_risk()`

2. `/home/user/ToyotaGR/dashboard/pages/strategic.py`
   - Integrated Bayesian analysis into dashboard
   - Added confidence interval displays
   - Added uncertainty slider
   - Added risk assessment panel
   - Added violin plot and PDF visualizations

### New Files
1. `/home/user/ToyotaGR/examples/bayesian_strategy_demo.py`
   - Comprehensive demonstration (5 demos)
   - Point estimate vs Bayesian comparison
   - Uncertainty with data demo
   - Confidence interval interpretation
   - Posterior visualization
   - Risk scenario analysis

2. `/home/user/ToyotaGR/examples/test_bayesian_strategy.py`
   - Automated testing script
   - Validates implementation
   - Quick verification

3. `/home/user/ToyotaGR/BAYESIAN_STRATEGY_IMPLEMENTATION.md`
   - This documentation file

---

## 10. Testing and Validation

### Test Results

```
Testing Bayesian Pit Strategy Optimization...
======================================================================

Optimal Pit Lap: 16
Posterior Mean: 15.97
Posterior Std: 0.49 laps

Confidence Intervals:
  80%: Laps 15-16
  90%: Laps 15-16
  95%: Laps 15-16

Uncertainty: 3.1%

Risk Level: LOW
Explanation: Optimal pit window is well-defined with high confidence
Strategy Note: Critical to hit optimal window - large time advantage

======================================================================
All tests passed successfully!
======================================================================
```

### Validation
- All methods tested and working
- Visualization data generated correctly
- Dashboard integration functional
- Statistical properties verified
- Edge cases handled

---

## Conclusion

The Bayesian uncertainty quantification enhancement provides RaceIQ Pro with:

1. **More Informed Decisions:** Teams know not just what to do, but how confident to be
2. **Risk Management:** Explicit assessment of strategic risk
3. **Flexibility:** Confidence intervals provide acceptable timing windows
4. **Scientific Rigor:** Statistically sound recommendations
5. **Interpretability:** Clear communication of uncertainty to drivers and engineers

This implementation uses scipy.stats exclusively, avoiding heavy dependencies while maintaining statistical rigor through conjugate prior methods. The result is a production-ready, efficient Bayesian optimization system for race strategy.
