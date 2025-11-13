# RaceIQ Pro Integration Engine

## Overview

The Integration Engine is the **key differentiator** for RaceIQ Pro. While many telemetry systems can show data and identify issues, the Integration Engine *connects the dots* between tactical driver performance and strategic race planning to generate unified, actionable recommendations.

### The Value Proposition

**Without Integration Engine:**
- "Driver is braking 0.8s too early in Section 3"
- "Optimal pit window is Lap 14-16"

**With Integration Engine:**
- "Fix Section 3 brake anomaly → Save 0.8s/lap → Delay pit to Lap 16 → Gain P3"

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RaceIQ Pro Integration Layer                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐          ┌─────────────────────────┐     │
│  │ Tactical Module  │          │   Strategic Module      │     │
│  │                  │          │                         │     │
│  │ • Section perf.  │          │ • Tire degradation      │     │
│  │ • Anomalies      │          │ • Pit optimization      │     │
│  │ • Driver coach   │          │ • Position analysis     │     │
│  └────────┬─────────┘          └───────────┬─────────────┘     │
│           │                                 │                    │
│           └────────────┬────────────────────┘                    │
│                        │                                         │
│              ┌─────────▼──────────┐                             │
│              │ Integration Engine │                             │
│              │                    │                             │
│              │ • Connect insights │                             │
│              │ • Calculate impact │                             │
│              │ • Generate chain   │                             │
│              └─────────┬──────────┘                             │
│                        │                                         │
│              ┌─────────▼─────────────┐                          │
│              │ Recommendation Builder│                          │
│              │                       │                          │
│              │ • Format for display  │                          │
│              │ • Prioritize actions  │                          │
│              │ • Dashboard output    │                          │
│              └───────────────────────┘                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Intelligence Engine (`intelligence_engine.py`)

The brain of the integration system. Contains three critical functions:

#### `connect_anomaly_to_strategy(anomaly, tire_model, strategy_optimizer)`

Connects anomaly detection to pit strategy optimization:
1. Estimates lap time impact if anomaly is corrected
2. Recalculates tire degradation with improved pace
3. Updates optimal pit window based on improved performance
4. Returns integrated insight with projected position gain

**Example:**
```python
impact = engine.connect_anomaly_to_strategy(
    anomaly={
        'section': 'Section 3',
        'type': 'brake_early',
        'magnitude': 0.8,
        'current_time': 13.3,
        'baseline_time': 12.5
    },
    tire_model=tire_model,
    strategy_optimizer=strategy
)

# Result: AnomalyImpact(
#   section='Section 3',
#   lap_time_loss=0.56,
#   optimal_pit_lap=16,
#   position_gain_potential=3,
#   confidence=0.85
# )
```

#### `connect_section_improvement_to_strategy(section_improvement, race_data, strategy)`

Connects section-level improvements to overall race strategy:
1. Calculates how section improvements affect overall lap time
2. Adjusts pit window timing based on improved pace
3. Returns strategic recommendation with position impact

**Example:**
```python
analysis = engine.connect_section_improvement_to_strategy(
    section_improvement={
        'section': 'Section 5',
        'current_time': 18.7,
        'potential_time': 18.3,
        'type': 'apex'
    },
    race_data={'current_lap': 10, 'total_laps': 25},
    strategy=strategy
)

# Result: SectionImpactAnalysis(
#   section='Section 5',
#   time_gain_per_lap=0.36,
#   adjusted_pit_lap=17,
#   position_impact=2,
#   confidence=0.78
# )
```

#### `generate_unified_recommendation(tactical_insights, strategic_insights)`

Generates unified recommendations combining tactical and strategic insights:
1. Combines insights from both modules
2. Prioritizes recommendations by expected impact
3. Formats as actionable items with confidence scores
4. Returns full chain of impact: "Fix X → Save Y → Strategy Z → Gain P3"

**Example:**
```python
insights = engine.generate_unified_recommendation(
    tactical_insights=[...],
    strategic_insights=[...]
)

# Result: [IntegratedInsight(
#   priority=1,
#   tactical_element='Section 3 brake anomaly',
#   strategic_element='Pit Lap 16',
#   chain_of_impact='Fix brake → 0.8s/lap → Pit L16 → Gain P3',
#   confidence=0.85
# )]
```

### 2. Recommendation Builder (`recommendation_builder.py`)

Formats insights for dashboard display with action items and visual hints.

#### `build_tactical_recommendation(section_analysis, anomalies)`

Creates driver coaching recommendations:
- Prioritizes by time gain potential
- Generates specific coaching messages
- Includes reference lap comparisons

#### `build_strategic_recommendation(pit_strategy, tire_state)`

Creates pit timing recommendations:
- Includes confidence intervals
- Shows optimal pit window
- Explains strategic rationale

#### `build_integrated_recommendation(tactical, strategic, integration_engine)`

Combines all insights with cross-module intelligence:
- Shows connections between tactical and strategic
- Formats for dashboard display
- Generates action items for driver, engineer, and strategist

## Data Structures

### Input Data (from Tactical Module)
```python
anomaly = {
    'section': str,              # e.g., 'Section 3 (Turn 7)'
    'type': str,                 # 'brake_early', 'apex_wide', etc.
    'magnitude': float,          # seconds lost
    'baseline_time': float,      # reference section time
    'current_time': float,       # current section time
    'lap': int,                  # lap number
    'confidence': float          # 0.0-1.0
}

section_improvement = {
    'section': str,
    'current_time': float,
    'potential_time': float,
    'type': str,                 # 'brake', 'apex', 'throttle'
    'confidence': float
}
```

### Input Data (from Strategic Module)
```python
tire_model = {
    'get_degradation_rate': function,  # Returns deg rate at lap N
    'get_tire_life': function          # Returns remaining tire life 0-1
}

strategy_optimizer = {
    'get_optimal_pit_lap': function,   # Returns optimal lap number
    'get_total_laps': function         # Returns total race laps
}
```

### Output Data
```python
IntegratedInsight = {
    'insight_type': str,
    'priority': int,                    # 1 (highest) to 5 (lowest)
    'tactical_element': str,
    'strategic_element': str,
    'expected_impact': str,
    'action_items': List[str],
    'confidence': float,
    'projected_position_gain': int,
    'chain_of_impact': str             # Full chain visualization
}
```

## Usage Examples

### Basic Usage

```python
from integration import IntegrationEngine, RecommendationBuilder

# Initialize
engine = IntegrationEngine(config={
    'lap_time_threshold': 0.1,
    'position_value': 2.0,
    'confidence_threshold': 0.7
})
builder = RecommendationBuilder()

# Connect anomaly to strategy
impact = engine.connect_anomaly_to_strategy(
    anomaly=detected_anomaly,
    tire_model=tire_model,
    strategy_optimizer=strategy
)

# Build recommendations
tactical_recs = builder.build_tactical_recommendation(
    section_analysis=section_analysis,
    anomalies=anomalies
)

strategic_recs = builder.build_strategic_recommendation(
    pit_strategy=pit_strategy,
    tire_state=tire_state
)

integrated_recs = builder.build_integrated_recommendation(
    tactical=tactical_recs,
    strategic=strategic_recs,
    integration_engine=engine
)

# Format for dashboard
dashboard_data = builder.format_for_dashboard(integrated_recs)
```

### Running the Examples

```bash
cd src/integration
python example_usage.py
```

This will run four comprehensive examples showing:
1. Anomaly integration with strategy
2. Section improvement impact analysis
3. Unified recommendations generation
4. Dashboard-ready formatting

## Configuration Options

```python
config = {
    # Minimum lap time gain to consider (seconds)
    'lap_time_threshold': 0.1,

    # Seconds per position (for position gain calculation)
    'position_value': 2.0,

    # Minimum confidence to include recommendation
    'confidence_threshold': 0.7
}
```

## Key Algorithms

### Position Gain Calculation

```
total_time_gain = lap_time_gain × laps_remaining
position_gain = total_time_gain ÷ position_value

Example: 0.56s/lap × 15 laps = 8.4s = 4 positions
```

### Pit Window Adjustment

```
lap_delay = lap_time_gain ÷ 0.5

If lap_time_gain > 0.5s/lap:
    adjusted_pit_lap = current_pit_lap + lap_delay
Else:
    adjusted_pit_lap = current_pit_lap
```

### Priority Calculation

- Priority 1: >1.0s/lap or >2 positions (CRITICAL)
- Priority 2: 0.5-1.0s/lap or 1-2 positions (HIGH)
- Priority 3: 0.3-0.5s/lap or 1 position (MEDIUM)
- Priority 4: 0.1-0.3s/lap (LOW)
- Priority 5: <0.1s/lap (MINIMAL)

## Integration with Other Modules

### Tactical Module Interface

The integration engine expects the tactical module to provide:
- Anomaly detection results
- Section-by-section performance analysis
- Reference lap comparisons
- Driver consistency metrics

### Strategic Module Interface

The integration engine expects the strategic module to provide:
- Tire degradation models
- Pit strategy optimizer
- Position tracking
- Race state data

## Dashboard Display Format

The recommendation builder outputs dashboard-ready JSON:

```json
{
  "timestamp": "2025-11-13T10:30:00",
  "total_recommendations": 3,
  "priority_breakdown": {
    "critical": 1,
    "high": 2,
    "medium": 0
  },
  "recommendations": [
    {
      "id": "INT_0001",
      "priority": 1,
      "priority_label": "CRITICAL",
      "title": "Section 3 Optimization + Strategy Adjustment",
      "chain_of_impact": "Fix brake point → Save 0.56s/lap → Pit L16 → Gain P3",
      "action_items": [
        {"role": "DRIVER", "action": "Brake later into Section 3", "priority": "HIGH"},
        {"role": "STRATEGY", "action": "Adjust pit window to Lap 16±2", "priority": "MEDIUM"}
      ],
      "expected_impact": {
        "time_per_lap": "0.56s",
        "total_time": "8.4s",
        "positions": "P3"
      },
      "confidence": "85%",
      "status": "new"
    }
  ],
  "summary": {
    "total_time_gain_potential": "15.8s",
    "max_position_gain": "P4",
    "avg_confidence": "82%"
  }
}
```

## Testing

To test the integration engine with mock data:

```python
# Run examples
python example_usage.py

# Or import and test individual functions
from intelligence_engine import IntegrationEngine
from recommendation_builder import RecommendationBuilder

engine = IntegrationEngine()
# ... run tests
```

## Future Enhancements

1. **Machine Learning Integration**: Use historical data to improve position gain predictions
2. **Multi-Car Strategy**: Consider competitor positions and strategies
3. **Weather Integration**: Factor weather changes into recommendations
4. **Real-time Updates**: Stream recommendations as new data arrives
5. **Driver Feedback Loop**: Learn from which recommendations drivers actually implement

## Performance Considerations

- All calculations are designed to run in <100ms for real-time updates
- Confidence scoring uses Bayesian inference for robustness
- Prioritization algorithm is O(n log n) where n = number of recommendations
- Dashboard formatting is cached for repeated access

## License

Part of RaceIQ Pro - Toyota GR Cup Hackathon Project

---

**Created by:** RaceIQ Pro Team
**Last Updated:** November 13, 2025
**Version:** 0.1.0
