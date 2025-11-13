# Bayesian Pit Strategy Workflow

## System Architecture

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
      │                       │                       │
      └───────────────────────┴───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DASHBOARD DISPLAY                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Optimal Pit Lap: 15 (±1.2 laps)                         │  │
│  │  90% Confidence: Laps 13-16                              │  │
│  │  Risk Level: 🟡 MODERATE                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  [Confidence Level Slider: 80% ←→ 90% ←→ 95%]                 │
│                                                                 │
│  [Violin Plot of Posterior Distribution]                       │
│  [PDF Curve with Shaded Confidence Intervals]                  │
│  [Simulation Results by Pit Lap]                               │
│                                                                 │
│  [Risk Assessment Panel]                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGIC DECISION                           │
│                                                                 │
│  Team/Driver uses information to decide:                       │
│  • When to pit (optimal window)                                │
│  • How flexible timing can be (confidence interval width)      │
│  • Level of strategic risk (risk assessment)                   │
│  • Whether to be conservative or aggressive                    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Example

### Input Data
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

### Processing Steps

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

### Output
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

## Bayesian Updating Visualization

```
Prior Knowledge          Simulation Data           Posterior Belief
(Racing Experience)      (Monte Carlo)             (Combined)

     │                        │                         │
     │                        │                         │
     │      ╱─╲               │   ╱╲                    │     ╱─╲
     │     ╱   ╲              │  ╱  ╲                   │    ╱   ╲
     │    ╱     ╲             │ ╱    ╲                  │   ╱     ╲
     │   ╱       ╲            │╱      ╲                 │  ╱       ╲
─────┴───────────────── × ────┴────────────── = ────────┴─────────────────
     10  15  20             12  15  18             13  15  17
   (Wide, uncertain)      (Data-driven)          (Narrower, confident)

Prior Std: 3.0 laps    Likelihood Std: 1.0    Posterior Std: 0.95 laps
```

## Interactive Dashboard Features

### 1. Confidence Level Adjustment
```
User moves slider: 80% ←→ 90% ←→ 95%

80% Confidence:
├────────────────┤
  Laps 14-16
  Narrow window, higher risk of being wrong

90% Confidence:
├──────────────────────┤
  Laps 13-17
  Balanced window, typical recommendation

95% Confidence:
├────────────────────────────┤
  Laps 13-17
  Wide window, very conservative
```

### 2. Violin Plot
```
         │
    17   ├───────────────
         │    ╱─────╲
    16   ├───╱       ╲───
         │  │         │
    15   ├──┼─────────┼──  ← Mean
         │  │         │
    14   ├───╲       ╱───
         │    ╲─────╱
    13   ├───────────────
         │
         └───────────────
         Optimal Pit Lap
```

### 3. Risk Indicators
```
🟢 LOW       Std < 1.0 laps    → High confidence, precise window
🟡 MODERATE  Std 1.0-2.0 laps  → Reasonable confidence, some flexibility
🟠 ELEVATED  Std 2.0-3.0 laps  → Significant uncertainty, monitor closely
🔴 HIGH      Std > 3.0 laps    → Large uncertainty, very sensitive
```

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

## Comparison to Traditional Approach

### Traditional Monte Carlo
```
Input: race_data, tire_model
  ↓
Run 100 simulations per candidate lap
  ↓
Find lap with minimum mean time
  ↓
Output: optimal_lap = 15
```

**Limitations:**
- Single point estimate
- No uncertainty quantification
- No confidence intervals
- No risk assessment

### Bayesian Approach
```
Input: race_data, tire_model
  ↓
Define prior distribution (experience)
  ↓
Run 100 simulations per candidate lap (likelihood)
  ↓
Update posterior = prior × likelihood
  ↓
Generate samples from posterior
  ↓
Calculate confidence intervals
  ↓
Assess risk based on posterior spread
  ↓
Output: {
  optimal_lap: 15,
  confidence_90: (13, 17),
  uncertainty: 6.3%,
  risk: MODERATE,
  samples: [15.2, 14.8, 15.1, ...]
}
```

**Advantages:**
- Full probability distribution
- Explicit uncertainty
- Multiple confidence levels
- Automated risk assessment
- Visualization support

## Performance Characteristics

### Computational Complexity
- Monte Carlo: O(n × k) where n = candidate laps, k = iterations
- Bayesian update: O(1) analytical solution
- Total: Still O(n × k), dominated by simulation

### Typical Runtime
- 100 simulations × 15 candidate laps = 1500 total simulations
- ~1-2 seconds on modern hardware
- Real-time suitable for pit wall decisions

### Accuracy
- Depends on:
  - Prior quality (racing experience)
  - Data quantity (laps completed)
  - Simulation fidelity (tire model)
- Generally: more data → narrower intervals → higher precision

## Integration Points

### Dashboard → Optimizer
```python
from src.strategic.strategy_optimizer import PitStrategyOptimizer

optimizer = PitStrategyOptimizer(
    pit_loss_seconds=25.0,
    simulation_iterations=100,
    uncertainty_model='bayesian'
)

result = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data, tire_model, race_length=25
)
```

### Optimizer → Visualization
```python
viz_data = optimizer.visualize_posterior_distribution(result)

# Create plotly violin plot
fig = go.Violin(y=result['posterior_samples'], ...)
```

### Result → Decision
```python
if result['risk_assessment']['risk_level'] == 'LOW':
    # High confidence - precise recommendation
    recommend(result['optimal_lap'])
else:
    # Provide window for flexibility
    window = result['confidence_90']
    recommend(f"Laps {window[0]}-{window[1]}")
```

## Future Extensions

### 1. Sequential Updating
Update posterior each lap as new data arrives:
```python
# Lap 10: posterior₁₀
# Lap 11: new data → posterior₁₁ = posterior₁₀ × likelihood₁₁
# Lap 12: new data → posterior₁₂ = posterior₁₁ × likelihood₁₂
```

### 2. Hierarchical Models
Learn from multiple drivers/sessions:
```python
# Global prior: all drivers
# Driver-specific adjustment: individual tendencies
# Session-specific: current race conditions
```

### 3. Multi-Dimensional
Optimize multiple aspects:
```python
posterior(pit_lap, fuel_load, tire_compound | data)
```

---

This workflow provides a complete Bayesian framework for pit strategy optimization, combining theoretical rigor with practical usability for real-time race decisions.
