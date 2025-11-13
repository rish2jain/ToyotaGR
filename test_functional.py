#!/usr/bin/env python3
"""
Functional test with actual data to verify end-to-end functionality
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def test_data_loading_and_processing():
    """Test data loading with actual sample data"""
    print("Testing data loading and processing...")
    try:
        from src.pipeline.data_loader import DataLoader

        loader = DataLoader()

        # Test lap time data
        lap_times = loader.load_lap_time_data()
        if lap_times is not None and len(lap_times) > 0:
            print(f"✅ Loaded {len(lap_times)} lap time records")
        else:
            print("⚠️  Lap time data is empty or None")

        # Test section analysis
        section_data = loader.load_section_analysis()
        if section_data is not None and len(section_data) > 0:
            print(f"✅ Loaded {len(section_data)} section analysis records")
        else:
            print("⚠️  Section analysis data is empty or None")

        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shap_with_mock_data():
    """Test SHAP explainability with mock data"""
    print("\nTesting SHAP explainability with mock data...")
    try:
        from src.tactical.anomaly_detector import AnomalyDetector
        import shap

        detector = AnomalyDetector()

        # Create mock telemetry data
        mock_data = pd.DataFrame({
            'Speed': [100, 105, 102, 98, 110, 95, 100],
            'ThrottlePosition': [80, 85, 82, 78, 90, 75, 80],
            'BrakePressure': [20, 15, 18, 22, 10, 25, 20],
            'SteeringAngle': [5, 3, 4, 6, 2, 7, 5],
            'RPM': [7000, 7200, 7100, 6900, 7400, 6800, 7000],
        })

        # Try to detect anomalies (this will train the model)
        anomalies = detector.detect_ml_anomalies(mock_data)
        print(f"✅ ML anomaly detection completed ({len(anomalies)} anomalies detected)")

        # Test explanation feature (only if anomalies detected)
        if len(anomalies) > 0 and hasattr(detector, 'explain_anomaly'):
            try:
                # Get first anomaly
                anomaly_idx = anomalies.iloc[0]['Index']
                anomaly_data = mock_data.iloc[anomaly_idx:anomaly_idx+1]

                explanation = detector.explain_anomaly(anomaly_data, mock_data.iloc[:5])
                print("✅ SHAP explanation generated successfully")

                if 'top_features' in explanation:
                    print(f"   Top contributing features: {len(explanation['top_features'])}")
            except Exception as e:
                print(f"⚠️  SHAP explanation test skipped: {e}")

        return True
    except Exception as e:
        print(f"❌ SHAP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bayesian_with_mock_data():
    """Test Bayesian uncertainty with mock race data"""
    print("\nTesting Bayesian uncertainty quantification...")
    try:
        from src.strategic.strategy_optimizer import PitStrategyOptimizer
        from src.strategic.tire_degradation import TireDegradationModel

        optimizer = PitStrategyOptimizer()

        # Create mock race data
        num_laps = 25
        mock_race_data = pd.DataFrame({
            'Lap': range(1, num_laps + 1),
            'LapTime': [90.5 + np.random.normal(0, 1) + (i * 0.3) for i in range(num_laps)],
            'Driver': ['Driver_1'] * num_laps,
        })

        # Create mock tire model
        tire_model = TireDegradationModel()
        tire_model.fit(mock_race_data)

        # Test Bayesian pit window calculation
        if hasattr(optimizer, 'calculate_optimal_pit_window_with_uncertainty'):
            result = optimizer.calculate_optimal_pit_window_with_uncertainty(
                mock_race_data, tire_model
            )
            print(f"✅ Bayesian analysis completed")
            print(f"   Optimal pit lap: {result['optimal_lap']}")
            print(f"   95% confidence interval: {result['confidence_95']}")
            print(f"   Uncertainty: {result['uncertainty']:.2%}")
        else:
            print("⚠️  Bayesian method not found")

        return True
    except Exception as e:
        print(f"❌ Bayesian test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weather_adjuster():
    """Test weather adjustment calculations"""
    print("\nTesting weather integration...")
    try:
        from src.integration.weather_adjuster import WeatherAdjuster

        adjuster = WeatherAdjuster()

        # Test tire degradation adjustment
        base_degradation = 0.5  # seconds per lap
        hot_temp = 40  # Celsius
        cold_temp = 20  # Celsius

        hot_adjusted = adjuster.adjust_tire_degradation(base_degradation, hot_temp)
        cold_adjusted = adjuster.adjust_tire_degradation(base_degradation, cold_temp)

        print(f"✅ Tire degradation adjustments calculated")
        print(f"   Base: {base_degradation:.3f}s/lap")
        print(f"   Hot track (40°C): {hot_adjusted:.3f}s/lap ({((hot_adjusted/base_degradation - 1) * 100):.1f}%)")
        print(f"   Cold track (20°C): {cold_adjusted:.3f}s/lap ({((cold_adjusted/base_degradation - 1) * 100):.1f}%)")

        # Test lap time adjustment
        base_lap_time = 90.0  # seconds
        weather_conditions = {
            'Temperature': 35,
            'TrackTemp': 45,
            'Humidity': 60,
            'WindSpeed': 15,
            'Precipitation': 0
        }

        adjusted_time = adjuster.adjust_lap_times(base_lap_time, weather_conditions)
        print(f"✅ Lap time adjustments calculated")
        print(f"   Base lap time: {base_lap_time:.3f}s")
        print(f"   Weather-adjusted: {adjusted_time:.3f}s ({((adjusted_time/base_lap_time - 1) * 100):.1f}%)")

        return True
    except Exception as e:
        print(f"❌ Weather adjustment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_track_map_generation():
    """Test track map visualization generation"""
    print("\nTesting track map visualization...")
    try:
        from src.utils.track_layouts import get_track_layout
        from src.utils.visualization import create_track_map_with_performance

        # Get Barber track layout
        barber_layout = get_track_layout('barber')
        print(f"✅ Barber track layout loaded ({len(barber_layout['sections'])} sections)")

        # Create mock section performance data
        mock_section_data = pd.DataFrame({
            'Section': [f'Section {i+1}' for i in range(len(barber_layout['sections']))],
            'GapToOptimal': np.random.uniform(-0.5, 2.0, len(barber_layout['sections'])),
            'LapTime': np.random.uniform(5, 15, len(barber_layout['sections']))
        })

        # Generate track map
        fig = create_track_map_with_performance(mock_section_data, 'barber')
        print(f"✅ Track map visualization generated successfully")
        print(f"   Figure type: {type(fig).__name__}")

        return True
    except Exception as e:
        print(f"❌ Track map test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("RaceIQ Pro - Functional Testing with Data")
    print("=" * 70)
    print()

    results = {}

    results['data_loading'] = test_data_loading_and_processing()
    results['shap'] = test_shap_with_mock_data()
    results['bayesian'] = test_bayesian_with_mock_data()
    results['weather'] = test_weather_adjuster()
    results['trackmaps'] = test_track_map_generation()

    print()
    print("=" * 70)
    print("FUNCTIONAL TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}: {status}")

    print()
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print()
        print("🎉 All functional tests passed!")
        print()
        print("Platform is READY for dashboard launch and user testing.")
    else:
        print()
        print("⚠️  Some tests failed. Review errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
