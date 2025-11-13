# Multi-Driver Race Simulation

## Overview

The Multi-Driver Race Simulation feature enables comprehensive modeling of multi-car races with realistic physics, tire degradation, pit strategies, and position battles. This advanced simulation tool helps teams understand race dynamics, test strategic scenarios, and optimize pit stop timing.

## Key Features

### 1. Full Race Simulation
- **Multi-car dynamics**: Simulate 2-10 drivers simultaneously
- **Position changes**: Track overtakes and position battles throughout the race
- **Lap-by-lap analysis**: Detailed state tracking for every lap
- **Strategy comparison**: Evaluate effectiveness of different pit strategies

### 2. Undercut Analysis
- **Fresh tire advantage**: Model the benefit of pitting earlier on new tires
- **Position prediction**: Determine if an undercut will succeed
- **Gap evolution**: Track time gaps through the undercut sequence
- **Success probability**: Calculate likelihood of overtaking

### 3. Overcut Analysis
- **Extended stint simulation**: Model staying out longer on old tires
- **Track position value**: Balance tire performance vs. track position
- **Critical lap identification**: Find the optimal moment to pit
- **Degradation modeling**: Account for increasing tire wear

### 4. Team Strategy Optimization
- **Multi-car optimization**: Find optimal strategy for multiple team cars
- **Strategic objectives**: Maximize points, guarantee win, or block opponents
- **Split strategies**: Test coordinated vs. independent strategies
- **Competitive advantage**: Optimize against specific opponent strategies

## How It Works

### Simulation Physics

The simulator uses realistic racing physics models:

#### Tire Degradation
```python
lap_time = base_lap_time + (tire_age × degradation_rate)
```

**Key Parameters:**
- `base_lap_time`: Lap time on fresh tires (typically 93-96 seconds for Toyota GR Cup)
- `tire_age`: Number of laps since last pit stop
- `degradation_rate`: Time lost per lap on tires (typically 0.04-0.06 s/lap)

**Example:**
- Fresh tire lap: 95.0s
- After 10 laps with 0.05 s/lap degradation: 95.5s
- After 15 laps: 95.75s

#### Fuel Effect
```python
fuel_benefit = (laps_completed / total_laps) × fuel_effect
```

Cars get lighter as fuel burns off, typically ~0.3s faster by race end.

**Example:**
- At lap 13 of 25: 13/25 × 0.3s = 0.156s faster
- At lap 25 of 25: 25/25 × 0.3s = 0.300s faster

#### Pit Stop Loss
Typical pit stop time loss: **23-27 seconds**

Components:
- Entry to pit lane: ~8s
- Pit stop itself: ~12s
- Exit to racing line: ~5-7s
- **Default used: 25.0s**

#### Overtaking Logic

Overtake succeeds when:
1. Gap closes to < 1.0 second
2. Pursuing driver is consistently faster
3. Pace advantage > 0.3s (track position penalty)

### Race State Management

The simulator maintains detailed state for each driver:
- **Position**: Current track position (P1, P2, etc.)
- **Cumulative time**: Total race time including pit stops
- **Tire age**: Laps on current tire set
- **Lap times**: Complete lap time history
- **Strategy**: Planned pit stops

## Usage Guide

### Python API

#### Basic Race Simulation

```python
from strategic.race_simulation import MultiDriverRaceSimulator

# Create simulator
simulator = MultiDriverRaceSimulator(
    race_length=25,
    pit_loss_time=25.0
)

# Define drivers
drivers_data = {
    'A': {
        'name': 'Driver A',
        'base_lap_time': 95.0,
        'tire_deg_rate': 0.05,
        'consistency': 0.10
    },
    'B': {
        'name': 'Driver B',
        'base_lap_time': 95.2,
        'tire_deg_rate': 0.05,
        'consistency': 0.10
    }
}

# Define strategies
strategies = {
    'A': {'pit_laps': [12]},
    'B': {'pit_laps': [14]}
}

# Run simulation
result = simulator.simulate_race(drivers_data, strategies)

# Access results
print(result['final_results'])
print(result['position_changes'])
```

#### Undercut Analysis

```python
# Test if undercut will work
driver_a_config = {
    'base_lap_time': 95.0,
    'tire_deg_rate': 0.05
}

driver_b_config = {
    'base_lap_time': 95.0,
    'tire_deg_rate': 0.05
}

result = simulator.simulate_undercut_scenario(
    driver_a_config,
    driver_b_config,
    pit_lap_a=10,  # A pits early
    pit_lap_b=12   # B pits later
)

print(f"Undercut success: {result['success']}")
print(f"Overtake lap: {result['overtake_lap']}")
print(f"Final gap: {result['time_delta']:.2f}s")
```

#### Team Strategy Optimization

```python
# Define team and opponents
team_drivers = {
    'T1': {'name': 'Team Car 1', 'base_lap_time': 94.5, 'tire_deg_rate': 0.05},
    'T2': {'name': 'Team Car 2', 'base_lap_time': 94.8, 'tire_deg_rate': 0.05}
}

opponents = {
    'O1': {'name': 'Opponent 1', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05},
    'O2': {'name': 'Opponent 2', 'base_lap_time': 95.2, 'tire_deg_rate': 0.05}
}

# Optimize team strategy
result = simulator.optimize_team_strategy(
    team_drivers,
    opponents,
    objective='maximize_team_points'
)

print(result['recommendation'])
print(f"Team score: {result['team_score']}")
```

### Dashboard Interface

The interactive Streamlit dashboard provides four main tools:

#### 1. Race Animation
- Configure 2-10 drivers with custom parameters
- Set pit strategies for each driver
- Run full race simulation
- Visualize position changes over time
- Download results as CSV

**Usage:**
1. Select number of drivers
2. Configure each driver's pace and pit strategy
3. Click "Run Race Simulation"
4. View animated position chart and results

#### 2. Undercut Analyzer
- Configure two drivers head-to-head
- Set different pit stop laps
- Analyze if undercut succeeds
- View gap evolution chart

**Usage:**
1. Configure Driver A (attempting undercut)
2. Configure Driver B (defending)
3. Ensure Driver A pits earlier
4. Click "Analyze Undercut"
5. Review success probability and timing

#### 3. Strategy Optimizer
- Set up team cars vs. opponents
- Choose optimization objective
- Find optimal pit strategies
- View expected race results

**Usage:**
1. Add 1-3 team cars
2. Add 1-5 opponent cars
3. Select objective (maximize points, guarantee win, etc.)
4. Click "Optimize Team Strategy"
5. Implement recommended strategies

#### 4. What-If Scenarios
- Test custom scenarios
- Equal pace battles
- High degradation races
- Fast pit stops
- Custom configurations

**Usage:**
1. Select scenario template or create custom
2. Configure race parameters
3. Run simulation
4. Compare different scenarios

## Realistic Parameters for Toyota GR Cup

Based on actual race data analysis:

### Typical Lap Times
- **Fast lap**: 93.5 - 94.5 seconds
- **Average lap**: 94.5 - 95.5 seconds
- **Slow lap**: 95.5 - 97.0 seconds

### Tire Degradation Rates
- **Low degradation**: 0.03 - 0.04 s/lap (smooth driving, cool conditions)
- **Medium degradation**: 0.045 - 0.055 s/lap (typical conditions)
- **High degradation**: 0.06 - 0.08 s/lap (aggressive driving, hot conditions)

### Driver Consistency
- **Very consistent**: 0.05 - 0.08s standard deviation
- **Typical consistency**: 0.08 - 0.12s standard deviation
- **Inconsistent**: 0.12 - 0.20s standard deviation

### Pit Strategy Windows
- **Early pit**: Laps 8-10
- **Mid-race pit**: Laps 11-14
- **Late pit**: Laps 15-18

### Example Configurations

#### Barber Motorsports Park (2024 Race 1)
```python
drivers = {
    'Car30': {
        'base_lap_time': 93.5,
        'tire_deg_rate': 0.04,
        'consistency': 0.08,
        'pit_lap': 12
    },
    'Car32': {
        'base_lap_time': 93.7,
        'tire_deg_rate': 0.05,
        'consistency': 0.10,
        'pit_lap': 10
    },
    'Car21': {
        'base_lap_time': 93.9,
        'tire_deg_rate': 0.045,
        'consistency': 0.09,
        'pit_lap': 14
    }
}
```

## Assumptions and Limitations

### Assumptions Made

1. **Tire Performance**: Linear degradation model
   - Real tires may have non-linear "cliff" behavior
   - Model assumes gradual, consistent degradation

2. **Fuel Effect**: Linear weight reduction
   - Assumes constant fuel consumption rate
   - Actual benefit may vary with driving style

3. **Overtaking**: Simplified proximity-based model
   - Real overtaking depends on track layout
   - Some tracks harder to pass than others

4. **Weather**: Not included in base model
   - Temperature affects tire degradation
   - Rain changes everything

5. **Traffic**: No lapped car handling
   - All cars assumed to be on lead lap
   - Lapping traffic can affect strategy

6. **Incidents**: No random events
   - No yellow flags, safety cars, or crashes
   - Real races have unpredictable elements

### Known Limitations

1. **Two-car battles only**: Overtaking logic simplified for 1-on-1
2. **Single pit stop**: Multiple pit stops possible but not optimized
3. **Grid start**: All cars assumed to start in order defined
4. **No qualifying**: Starting positions not based on lap times
5. **Deterministic fuel**: Same fuel consumption for all drivers

## Tips for Best Results

### 1. Use Realistic Parameters
- Base lap times on actual qualifying data
- Set degradation rates from race data analysis
- Use typical pit loss times (23-27s)

### 2. Test Multiple Scenarios
- Run simulations with different pit laps
- Test ±2 laps around expected pit window
- Compare early vs. late pit strategies

### 3. Account for Track Position
- Being ahead is worth ~0.3-0.5s per lap
- Undercuts need significant tire advantage
- Consider traffic when choosing pit lap

### 4. Validate Results
- Compare to actual race results when available
- Adjust parameters if results seem unrealistic
- Use multiple simulations to account for randomness

### 5. Strategy Considerations
- Pit when tires are degraded (not too early)
- Avoid pitting same lap as competitors (lose positions)
- Consider undercut 1-2 laps before opponent
- Late pit requires low degradation rate

## Future Enhancements

Potential improvements for future versions:

1. **Weather Integration**: Temperature, humidity, rain effects
2. **Track-Specific Modeling**: Overtaking difficulty by track
3. **Multiple Compounds**: Different tire types with unique degradation
4. **Yellow Flags**: Random safety car periods
5. **Qualifying**: Grid positions based on lap times
6. **Lapped Traffic**: Modeling backmarkers
7. **Tire Cliff Detection**: Non-linear degradation curves
8. **Driver Skill Variation**: Overtaking ability differences
9. **Fuel Strategy**: Optimal fuel saving opportunities
10. **Telemetry Integration**: Use real telemetry to tune models

## Examples and Demonstrations

### Run Demo Script
```bash
cd examples
python race_simulation_demo.py
```

This runs 5 comprehensive scenarios:
1. Simple 2-driver battle
2. Undercut demonstration
3. 5-driver strategy battle
4. Team strategy optimization
5. Overcut analysis

### Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

Navigate to: **🏎️ Race Simulator**

## Technical Details

### Algorithm Complexity
- **Time complexity**: O(n × m) where n = drivers, m = laps
- **Space complexity**: O(n × m) for storing lap-by-lap data
- **Typical runtime**: <1s for 10 drivers × 25 laps

### Data Structures
- Race state: Dictionary of driver states
- Lap history: List of race states per lap
- Results: Sorted list of final positions

### Performance Optimization
- NumPy arrays for calculations
- Efficient position updates (single sort per lap)
- Minimal memory allocation during simulation

## Support and Questions

For questions or issues:
1. Check this documentation
2. Review example scenarios in `examples/race_simulation_demo.py`
3. Examine source code in `src/strategic/race_simulation.py`
4. Test with interactive dashboard

## Summary

The Multi-Driver Race Simulation provides a powerful tool for understanding race dynamics and optimizing strategy. By modeling realistic tire degradation, fuel effects, and position battles, teams can:

- ✅ Test pit strategies before the race
- ✅ Analyze undercut/overcut opportunities
- ✅ Optimize multi-car team strategies
- ✅ Understand race dynamics and position changes
- ✅ Make data-driven strategic decisions

**Key Success Factors:**
- Use realistic parameters from actual race data
- Test multiple scenarios to find optimal strategy
- Account for track position value
- Validate results against real races
- Iterate and refine models based on results

---

**Version**: 1.0
**Last Updated**: 2025-11-13
**Module**: `src/strategic/race_simulation.py`
**Dashboard**: `dashboard/pages/race_simulator.py`
