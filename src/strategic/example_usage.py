"""
Example usage of the Strategic Analysis Module for RaceIQ Pro

This script demonstrates how to use the pit detection, tire degradation,
and strategy optimization components with real race data.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategic import PitStopDetector, TireDegradationModel, PitStrategyOptimizer


def load_race_data(file_path: str) -> pd.DataFrame:
    """Load race data from CSV file."""
    try:
        df = pd.read_csv(file_path, delimiter=';')
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def demo_pit_detection(race_data: pd.DataFrame, driver_number: int = 1):
    """Demonstrate pit stop detection functionality."""
    print("\n" + "="*80)
    print("PIT STOP DETECTION DEMO")
    print("="*80)

    # Filter data for specific driver
    driver_data = race_data[race_data['DRIVER_NUMBER'] == driver_number].copy()
    print(f"\nAnalyzing driver #{driver_number} - {len(driver_data)} laps")

    # Initialize detector
    detector = PitStopDetector(window_size=5, confidence_threshold=0.6)

    # Detect pit stops
    detections = detector.detect_pit_stops(driver_data, sensitivity=2.5)
    print(f"\nInitial detections: {detections['is_pit_stop'].sum()} potential pit stops")

    # Refine detections
    refined = detector.refine_detections(detections)
    print(f"After refinement: {refined['is_pit_stop'].sum()} confirmed pit stops")

    # Get summary
    summary = detector.get_pit_stop_summary(refined)
    print(f"\nPit Stop Summary:")
    print(f"  Total pit stops: {summary['total_pit_stops']}")
    print(f"  Pit stop laps: {summary['pit_stop_laps']}")
    print(f"  Average confidence: {summary['average_confidence']:.2%}")
    if summary['estimated_pit_time_loss']:
        print(f"  Estimated pit time loss: {[f'{t:.1f}s' for t in summary['estimated_pit_time_loss']]}")

    return refined, summary


def demo_tire_degradation(race_data: pd.DataFrame, driver_number: int = 1):
    """Demonstrate tire degradation modeling functionality."""
    print("\n" + "="*80)
    print("TIRE DEGRADATION ANALYSIS DEMO")
    print("="*80)

    # Filter data for specific driver
    driver_data = race_data[race_data['DRIVER_NUMBER'] == driver_number].copy()

    # Initialize model
    tire_model = TireDegradationModel(model_type='polynomial', degree=2)

    # Estimate degradation
    degradation = tire_model.estimate_degradation(driver_data, exclude_outliers=True)

    print(f"\nDegradation Analysis:")
    print(f"  Model type: {degradation['model_type']}")
    print(f"  Baseline lap time: {degradation['baseline_lap_time']:.3f}s")
    print(f"  Current lap time: {degradation['current_lap_time']:.3f}s")
    print(f"  Degradation rate: {degradation['degradation_rate']:.4f} sec/lap")
    print(f"  Current tire performance: {degradation['current_performance_pct']:.1f}%")
    print(f"  Model R²: {degradation['r_squared']:.3f}")

    # Predict tire cliff
    cliff_prediction = tire_model.predict_cliff_point(driver_data)

    print(f"\nTire Cliff Prediction:")
    print(f"  Current lap: {cliff_prediction['current_lap']}")
    print(f"  Predicted cliff lap: {cliff_prediction['cliff_lap']}")
    print(f"  Warning laps: {cliff_prediction['warning_laps']}")
    print(f"  Confidence: {cliff_prediction['cliff_confidence']:.2%}")
    print(f"  Status: {cliff_prediction['status']}")

    # Analyze corner speed degradation
    if 'S1_SECONDS' in driver_data.columns:
        corner_analysis = tire_model.analyze_corner_speed_degradation(driver_data)
        if 'section_analysis' in corner_analysis:
            print(f"\nCorner Speed Degradation:")
            print(f"  Average degradation: {corner_analysis['average_corner_degradation_pct']:.2f}%")
            print(f"  Severity: {corner_analysis['severity']}")
            print(f"  Corner degradation detected: {corner_analysis['corner_degradation_detected']}")

    return degradation, cliff_prediction


def demo_strategy_optimization(race_data: pd.DataFrame, tire_model_data: dict,
                               driver_number: int = 1):
    """Demonstrate pit strategy optimization functionality."""
    print("\n" + "="*80)
    print("PIT STRATEGY OPTIMIZATION DEMO")
    print("="*80)

    # Filter data for specific driver
    driver_data = race_data[race_data['DRIVER_NUMBER'] == driver_number].copy()

    # Initialize optimizer
    optimizer = PitStrategyOptimizer(
        pit_loss_seconds=25.0,
        simulation_iterations=100,
        uncertainty_model='gaussian'
    )

    # Calculate optimal pit window
    race_length = race_data['LAP_NUMBER'].max()
    current_lap = 10  # Simulate being at lap 10

    optimal_strategy = optimizer.calculate_optimal_pit_window(
        driver_data.head(current_lap),
        tire_model_data,
        race_length=race_length,
        current_lap=current_lap
    )

    print(f"\nOptimal Pit Strategy (from lap {current_lap}):")
    print(f"  Optimal pit lap: {optimal_strategy['optimal_pit_lap']}")
    print(f"  Pit window: Laps {optimal_strategy['pit_window'][0]}-{optimal_strategy['pit_window'][1]}")
    print(f"  Expected time gain: {optimal_strategy['expected_time_gain']:.2f}s")
    print(f"  Laps until window: {optimal_strategy['laps_until_window']}")
    print(f"  Optimal expected time: {optimal_strategy['optimal_expected_time']:.2f}s")
    print(f"  Uncertainty (±): {optimal_strategy['optimal_time_uncertainty']:.2f}s")

    # Simulate undercut opportunity
    undercut = optimizer.simulate_undercut_opportunity(
        driver_data.head(current_lap),
        gap_to_competitor=2.0,
        pit_lap_difference=1
    )

    print(f"\nUndercut Analysis:")
    print(f"  Success probability: {undercut['undercut_success_probability']:.2%}")
    print(f"  Current gap: {undercut['current_gap']:.2f}s")
    print(f"  Expected gap after stops: {undercut['expected_gap_after_stops']:.2f}s")
    print(f"  Time gained on track: {undercut['time_gained_on_track']:.2f}s")
    print(f"  Recommendation: {undercut['recommendation']}")
    print(f"  Risk assessment: {undercut['risk_assessment']}")

    return optimal_strategy, undercut


def main():
    """Main demonstration function."""
    print("="*80)
    print("RaceIQ Pro - Strategic Analysis Module Demonstration")
    print("="*80)

    # Define data path
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Data', 'barber', '23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV'
    )

    # Load data
    print(f"\nLoading race data from: {data_path}")
    race_data = load_race_data(data_path)

    if race_data is None:
        print("Failed to load race data. Please check the file path.")
        return

    # Display available drivers
    drivers = race_data['DRIVER_NUMBER'].unique()
    print(f"\nAvailable drivers: {sorted(drivers)}")

    # Select driver for analysis (use driver #1 as example)
    test_driver = 1
    if test_driver not in drivers:
        test_driver = drivers[0]

    print(f"\nRunning analysis for Driver #{test_driver}")

    # Demo 1: Pit Stop Detection
    try:
        pit_detections, pit_summary = demo_pit_detection(race_data, test_driver)
    except Exception as e:
        print(f"Error in pit detection demo: {e}")
        import traceback
        traceback.print_exc()

    # Demo 2: Tire Degradation
    try:
        degradation, cliff = demo_tire_degradation(race_data, test_driver)
    except Exception as e:
        print(f"Error in tire degradation demo: {e}")
        import traceback
        traceback.print_exc()
        degradation = {'baseline_lap_time': 100.0, 'degradation_rate': 0.05}

    # Demo 3: Strategy Optimization
    try:
        strategy, undercut = demo_strategy_optimization(race_data, degradation, test_driver)
    except Exception as e:
        print(f"Error in strategy optimization demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe Strategic Analysis Module is ready for integration with RaceIQ Pro!")


if __name__ == '__main__':
    main()
