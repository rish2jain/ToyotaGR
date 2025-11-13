# Bayesian Pit Strategy - Quick Reference Card

## At a Glance

### What is Bayesian Pit Strategy?
Instead of recommending a single pit lap (e.g., "Pit on lap 14"), the Bayesian approach provides:
- **Optimal window:** "90% confident: pit between laps 12-15"
- **Uncertainty level:** "7.4% relative uncertainty"
- **Risk assessment:** "MODERATE - reasonable confidence with timing flexibility"

---

## Dashboard Usage

### 1. Access the Feature
1. Launch RaceIQ Pro dashboard: `streamlit run dashboard/app.py`
2. Navigate to **Strategic Analysis** page
3. Select a driver from dropdown
4. Scroll to **"Optimal Pit Window Analysis with Bayesian Uncertainty"**

### 2. Read the Metrics

| Metric | What It Means | How to Use |
|--------|---------------|------------|
| **Optimal Pit Lap** | Best estimate (posterior mean) | Primary target lap number |
| **Uncertainty** | Relative uncertainty (%) | Lower = more confident |
| **Risk Level** | 🟢 LOW / 🟡 MODERATE / 🟠 ELEVATED / 🔴 HIGH | Overall confidence assessment |

### 3. Use the Confidence Slider

Move slider to adjust confidence level: **80% ←→ 90% ←→ 95%**

| Level | Use When | Trade-off |
|-------|----------|-----------|
| **80%** | Aggressive strategy, tight window | Narrower window, higher risk |
| **90%** | Balanced approach (recommended) | Good balance |
| **95%** | Conservative, accounting for uncertainty | Wider window, safer |

### 4. Interpret Confidence Intervals

**Example:** "90% Confidence: Laps 12-15"

**Means:** We are 90% certain the optimal pit lap is between 12 and 15.

**Strategy:**
- **Ideal:** Pit on lap 13 or 14 (center of window)
- **Acceptable:** Any lap from 12-15
- **Avoid:** Pitting outside this window (only 10% chance of being optimal)

---

## Risk Assessment Guide

### 🟢 LOW Risk
- **Posterior Std:** < 1.0 laps
- **What it means:** Optimal window is well-defined with high confidence
- **Strategy:** Stick to the optimal lap for best results
- **Example:** "Optimal: Lap 15, 90% CI: 14-16" (tight window)

### 🟡 MODERATE Risk
- **Posterior Std:** 1.0-2.0 laps
- **What it means:** Reasonable confidence, some timing flexibility
- **Strategy:** Use 90% confidence window, monitor tire condition
- **Example:** "Optimal: Lap 14, 90% CI: 12-16" (moderate window)

### 🟠 ELEVATED Risk
- **Posterior Std:** 2.0-3.0 laps
- **What it means:** Significant uncertainty, monitor closely
- **Strategy:** Be prepared to adjust, watch tire degradation carefully
- **Example:** "Optimal: Lap 13, 90% CI: 10-16" (wide window)

### 🔴 HIGH Risk
- **Posterior Std:** > 3.0 laps
- **What it means:** Large uncertainty, timing highly sensitive
- **Strategy:** Very conservative, use 95% confidence, adapt to real-time
- **Example:** "Optimal: Lap 15, 90% CI: 9-21" (very wide)

---

## Visualizations

### Violin Plot
Shows the full probability distribution of optimal pit laps.

**How to Read:**
- **Wider = More uncertainty** in that range
- **Mean line:** Optimal lap
- **Box:** 50% of probability (interquartile range)
- **Horizontal lines:** Confidence interval bounds

### PDF Curve
Smooth probability density function.

**How to Read:**
- **Peak:** Most likely pit lap
- **Height:** Relative probability
- **Shaded area:** Selected confidence interval
- **Red star:** Optimal lap (posterior mean)

### Simulation Results Plot
Expected race time for each pit lap.

**How to Read:**
- **Lowest point:** Optimal pit lap (minimum time)
- **Error bars:** Simulation uncertainty
- **Shaded region:** Confidence interval
- **Steep changes:** Critical pit timing
- **Flat region:** Flexible pit timing

---

## Common Scenarios

### Early Race (Few Laps Completed)
```
Laps completed: 5
Uncertainty: ~13%
Risk: MODERATE to ELEVATED
```
**Interpretation:** Limited data, prior dominates, wider intervals
**Strategy:** Use 95% confidence for safety

### Mid Race (Adequate Data)
```
Laps completed: 12
Uncertainty: ~7%
Risk: MODERATE
```
**Interpretation:** Good data, balanced prior/likelihood, reasonable confidence
**Strategy:** Use 90% confidence (recommended)

### Late Race (Abundant Data)
```
Laps completed: 20
Uncertainty: ~2%
Risk: LOW
```
**Interpretation:** Lots of data, likelihood dominates, high confidence
**Strategy:** Can use 80% confidence, tight window

---

## Decision Framework

### Step 1: Check Risk Level
- 🟢 LOW → Proceed with confidence
- 🟡 MODERATE → Standard approach
- 🟠 ELEVATED → Be cautious
- 🔴 HIGH → Very conservative

### Step 2: Choose Confidence Level
- **Aggressive:** 80% confidence (tighter window)
- **Balanced:** 90% confidence (recommended)
- **Conservative:** 95% confidence (wider window)

### Step 3: Set Target Window
- **Primary target:** Optimal lap (center of window)
- **Acceptable range:** Confidence interval bounds
- **Monitor:** Tire condition, competitor strategies

### Step 4: Execute
- **Before window:** Monitor and prepare
- **In window:** Ready to pit when optimal
- **Passed optimal:** Still acceptable within interval
- **Outside window:** Reassess (conditions changed?)

---

## Key Differences from Traditional Approach

| Traditional | Bayesian |
|------------|----------|
| "Pit on lap 14" | "90% confident: pit laps 12-15" |
| Single number | Probability distribution |
| No uncertainty info | Explicit uncertainty (7.4%) |
| No risk assessment | Automated risk level |
| Fixed recommendation | Adaptive to data availability |
| Hard to communicate uncertainty | Clear confidence statements |

---

## Python API (For Developers)

### Basic Usage
```python
from src.strategic.strategy_optimizer import PitStrategyOptimizer

# Initialize
optimizer = PitStrategyOptimizer(
    pit_loss_seconds=25.0,
    simulation_iterations=100,
    uncertainty_model='bayesian'
)

# Calculate
result = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data=driver_data,
    tire_model=tire_model,
    race_length=25
)

# Access results
optimal_lap = result['optimal_lap']
confidence_90 = result['confidence_90']  # (lower, upper)
uncertainty = result['uncertainty']
risk = result['risk_assessment']['risk_level']
```

### Visualization
```python
# Get viz data
viz_data = optimizer.visualize_posterior_distribution(result)

# Use in Plotly
import plotly.graph_objects as go

fig = go.Violin(y=result['posterior_samples'])
fig.show()
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

---

## Best Practices

### ✓ DO
- Use 90% confidence as default
- Consider risk level in decision making
- Monitor how uncertainty changes over race
- Combine with driver feedback on tire condition
- Adjust confidence level based on race criticality

### ✗ DON'T
- Ignore uncertainty and use only optimal lap
- Pit outside confidence interval without good reason
- Overlook risk assessment warnings
- Rely solely on early-race predictions (high uncertainty)
- Forget to update with latest lap data

---

## Example Interpretation

### Scenario
```
Optimal Pit Lap: 15
90% Confidence: Laps 13-17
Uncertainty: 5.2%
Risk Level: 🟡 MODERATE
```

### Interpretation
**What the team should know:**
- Best estimate is lap 15
- Can pit anytime from lap 13-17 with 90% confidence
- Uncertainty is low (5.2%) - reasonable confidence
- Risk is moderate - standard approach, some flexibility

### Strategy Decision
- **Plan A:** Target lap 15 (optimal)
- **Plan B:** Acceptable window is laps 13-17
- **Monitor:** Tire condition, competitor moves
- **Flexibility:** Have 4-lap window to work with
- **Confidence:** 90% sure this is the right range

---

## Quick Command Reference

### Run Tests
```bash
# Quick validation
python examples/test_bayesian_strategy.py

# Full integration test
python examples/test_full_integration.py

# Interactive demo (requires user input)
python examples/bayesian_strategy_demo.py
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Access Documentation
- Full technical docs: `BAYESIAN_STRATEGY_IMPLEMENTATION.md`
- System workflow: `docs/BAYESIAN_WORKFLOW.md`
- Implementation summary: `IMPLEMENTATION_SUMMARY.md`
- This quick reference: `BAYESIAN_STRATEGY_QUICK_REFERENCE.md`

---

## Support

### Understanding Results
- **Optimal lap:** Use as primary target
- **Confidence interval:** Acceptable pit window
- **Uncertainty:** How confident we are
- **Risk level:** Overall assessment

### Making Decisions
1. Check risk level
2. Choose confidence level (90% default)
3. Note the pit window
4. Execute within window
5. Monitor and adjust if needed

### Getting Help
- Review documentation files
- Run test scripts for examples
- Check visualizations in dashboard
- Consult Bayesian workflow diagram

---

**Remember:** The Bayesian approach doesn't just tell you WHAT to do, but HOW CONFIDENT you should be in that decision. Use the uncertainty information to make smarter, risk-aware strategic choices!
