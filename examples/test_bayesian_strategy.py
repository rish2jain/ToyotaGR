"""
Quick test of Bayesian Strategy Optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.strategic.strategy_optimizer import PitStrategyOptimizer

# Set random seed
np.random.seed(42)

# Generate synthetic race data
num_laps = 15
lap_numbers = np.arange(1, num_laps + 1)
lap_times = 95.0 + 0.08 * lap_numbers + np.random.normal(0, 0.3, num_laps)

lap_time_strings = []
for lt in lap_times:
    minutes = int(lt // 60)
    seconds = lt % 60
    lap_time_strings.append(f"{minutes}:{seconds:06.3f}")

race_data = pd.DataFrame({
    'LAP_NUMBER': lap_numbers,
    'LAP_TIME': lap_time_strings,
    'DRIVER_NUMBER': [10] * num_laps
})

# Create tire model
tire_model = {
    'baseline_lap_time': 95.0,
    'degradation_rate': 0.08,
    'model_type': 'linear'
}

# Test Bayesian optimization
print("Testing Bayesian Pit Strategy Optimization...")
print("=" * 70)

optimizer = PitStrategyOptimizer(
    pit_loss_seconds=25.0,
    simulation_iterations=100,
    uncertainty_model='bayesian'
)

result = optimizer.calculate_optimal_pit_window_with_uncertainty(
    race_data, tire_model, race_length=25
)

print(f"\nOptimal Pit Lap: {result['optimal_lap']}")
print(f"Posterior Mean: {result['posterior_mean']:.2f}")
print(f"Posterior Std: {result['posterior_std']:.2f} laps")
print(f"\nConfidence Intervals:")
print(f"  80%: Laps {result['confidence_80'][0]}-{result['confidence_80'][1]}")
print(f"  90%: Laps {result['confidence_90'][0]}-{result['confidence_90'][1]}")
print(f"  95%: Laps {result['confidence_95'][0]}-{result['confidence_95'][1]}")
print(f"\nUncertainty: {result['uncertainty']*100:.1f}%")

risk = result['risk_assessment']
print(f"\nRisk Level: {risk['risk_level']}")
print(f"Explanation: {risk['explanation']}")
print(f"Strategy Note: {risk['strategy_note']}")

# Test visualization
print("\n" + "=" * 70)
print("Testing Visualization Methods...")
print("=" * 70)

viz_data = optimizer.visualize_posterior_distribution(result)

print(f"\nVisualization data generated successfully:")
print(f"  - PDF points: {len(viz_data['pdf']['x'])}")
print(f"  - Histogram bins: {len(viz_data['histogram']['bin_centers'])}")
print(f"  - Posterior samples: {len(viz_data['samples'])}")
print(f"  - Confidence intervals: {list(viz_data['confidence_intervals'].keys())}")

print("\n" + "=" * 70)
print("All tests passed successfully!")
print("=" * 70)
