# Weather Integration - Quick Start Guide

## Running the Demo
```bash
cd /home/user/ToyotaGR
python examples/weather_integration_demo.py
```

## Testing with Real Data
```bash
python -c "
import sys
sys.path.insert(0, '/home/user/ToyotaGR')
from pathlib import Path
from src.pipeline.data_loader import DataLoader
from src.integration.weather_adjuster import WeatherAdjuster

# Load weather data
loader = DataLoader(Path('/home/user/ToyotaGR/Data/barber'))
weather_df = loader.load_weather_data()

# Get current conditions
adjuster = WeatherAdjuster()
conditions = adjuster.get_current_conditions(weather_df)

# Generate recommendations
recommendations = adjuster.generate_weather_recommendations(weather_df)

print('Weather Conditions:')
print(f'  Temperature: {conditions.temperature:.1f}°C')
print(f'  Track Temp: {conditions.track_temp:.1f}°C')
print(f'  Humidity: {conditions.humidity:.0f}%')
print()
print('Strategic Impact:')
print(f'  {recommendations[\"tire_adjustment\"]}')
print(f'  Lap modifier: {recommendations[\"lap_time_modifier\"]:.3f}x')
"
```

## Dashboard Usage

### Start Dashboard
```bash
cd /home/user/ToyotaGR
streamlit run dashboard/app.py
```

### View Weather Integration
1. Navigate to **Overview** page → See weather widget with current conditions
2. Navigate to **Strategic** page → See weather-adjusted tire degradation

## Key Files

### Implementation
- `src/pipeline/data_loader.py` - Weather data loading
- `src/integration/weather_adjuster.py` - Weather adjustment engine (NEW)
- `src/integration/intelligence_engine.py` - Weather integration method
- `dashboard/pages/overview.py` - Weather widget
- `dashboard/pages/strategic.py` - Weather-adjusted analysis

### Examples
- `examples/weather_integration_demo.py` - Comprehensive demo (NEW)

### Documentation
- `WEATHER_INTEGRATION_SUMMARY.md` - Full documentation
- `WEATHER_QUICK_START.md` - This guide

## Quick API Reference

```python
# Load weather data
from src.pipeline.data_loader import DataLoader
loader = DataLoader()
weather_df = loader.load_weather_data()

# Adjust tire degradation
from src.integration.weather_adjuster import WeatherAdjuster
adjuster = WeatherAdjuster()
conditions = adjuster.get_current_conditions(weather_df)
adjusted_deg, explanation = adjuster.adjust_tire_degradation(0.05, conditions)

# Adjust lap times
adjusted_lap, explanation = adjuster.adjust_lap_times(100.0, conditions)

# Get recommendations
recommendations = adjuster.generate_weather_recommendations(weather_df)
print(recommendations['tire_adjustment'])
print(recommendations['strategic_notes'])

# Integrate with intelligence engine
from src.integration.intelligence_engine import IntegrationEngine
engine = IntegrationEngine()
weather_adjusted = engine.integrate_weather_impact(recommendations, weather_df)
```

## Weather Impact Summary

| Condition | Tire Degradation | Lap Time | Risk |
|-----------|------------------|----------|------|
| Hot Track (>40°C) | +10-20% | -2% (better grip) | Medium (brake fade) |
| Cold Track (<25°C) | -5-10% | +2-4% | Medium (less grip) |
| High Humidity (>70%) | -3% | Neutral | Low |
| High Winds (>25 km/h) | Neutral | +4% | Medium-High |
| Rain | -15% | +10% | High |

## Verification

Run verification test:
```bash
python -c "
import sys
sys.path.insert(0, '/home/user/ToyotaGR')
from pathlib import Path
from src.pipeline.data_loader import DataLoader
from src.integration.weather_adjuster import WeatherAdjuster
from src.integration.intelligence_engine import IntegrationEngine

loader = DataLoader(Path('/home/user/ToyotaGR/Data/barber'))
weather_df = loader.load_weather_data()
adjuster = WeatherAdjuster()
engine = IntegrationEngine()

assert weather_df is not None, 'Weather data loading failed'
assert adjuster.get_current_conditions(weather_df) is not None, 'Conditions extraction failed'
assert len(adjuster.generate_weather_recommendations(weather_df)['strategic_notes']) > 0, 'Recommendations failed'

print('✓ All weather integration components verified!')
"
```

## Support

For detailed documentation, see `WEATHER_INTEGRATION_SUMMARY.md`
For demo walkthrough, run `python examples/weather_integration_demo.py`
