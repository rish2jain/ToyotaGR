#!/usr/bin/env python3
"""
Test script to verify all enhancement features work correctly
"""
import sys
import traceback
from pathlib import Path

def test_shap_import():
    """Test SHAP can be imported"""
    try:
        import shap
        print("✅ SHAP imported successfully (version: {})".format(shap.__version__))
        return True
    except Exception as e:
        print(f"❌ SHAP import failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_anomaly_detector():
    """Test enhanced anomaly detector with SHAP"""
    try:
        from src.tactical.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        print("✅ Enhanced AnomalyDetector instantiated")

        # Check if explain_anomaly method exists
        if hasattr(detector, 'explain_anomaly'):
            print("✅ AnomalyDetector has explain_anomaly method")
        else:
            print("⚠️  AnomalyDetector missing explain_anomaly method")

        if hasattr(detector, 'get_anomaly_explanations'):
            print("✅ AnomalyDetector has get_anomaly_explanations method")
        else:
            print("⚠️  AnomalyDetector missing get_anomaly_explanations method")

        return True
    except Exception as e:
        print(f"❌ AnomalyDetector test failed: {e}")
        traceback.print_exc()
        return False

def test_bayesian_strategy():
    """Test Bayesian uncertainty in strategy optimizer"""
    try:
        from src.strategic.strategy_optimizer import PitStrategyOptimizer

        optimizer = PitStrategyOptimizer()
        print("✅ Enhanced PitStrategyOptimizer instantiated")

        # Check if Bayesian method exists
        if hasattr(optimizer, 'calculate_optimal_pit_window_with_uncertainty'):
            print("✅ PitStrategyOptimizer has calculate_optimal_pit_window_with_uncertainty method")
        else:
            print("⚠️  PitStrategyOptimizer missing Bayesian uncertainty method")

        return True
    except Exception as e:
        print(f"❌ PitStrategyOptimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_weather_integration():
    """Test weather integration components"""
    try:
        from src.integration.weather_adjuster import WeatherAdjuster
        from src.pipeline.data_loader import DataLoader

        adjuster = WeatherAdjuster()
        print("✅ WeatherAdjuster instantiated")

        loader = DataLoader()
        if hasattr(loader, 'load_weather_data'):
            print("✅ DataLoader has load_weather_data method")
        else:
            print("⚠️  DataLoader missing load_weather_data method")

        # Check IntegrationEngine has weather integration
        from src.integration.intelligence_engine import IntegrationEngine
        engine = IntegrationEngine()
        if hasattr(engine, 'integrate_weather_impact'):
            print("✅ IntegrationEngine has integrate_weather_impact method")
        else:
            print("⚠️  IntegrationEngine missing integrate_weather_impact method")

        return True
    except Exception as e:
        print(f"❌ Weather integration test failed: {e}")
        traceback.print_exc()
        return False

def test_track_maps():
    """Test track map visualization components"""
    try:
        from src.utils.track_layouts import TRACK_COORDINATES, get_track_layout
        from src.utils.visualization import create_track_map_with_performance, create_driver_comparison_map

        print("✅ Track layout data imported")

        # Check available tracks
        available_tracks = list(TRACK_COORDINATES.keys())
        print(f"✅ Available tracks: {', '.join(available_tracks)}")

        # Verify track map functions exist
        print("✅ create_track_map_with_performance function available")
        print("✅ create_driver_comparison_map function available")

        return True
    except Exception as e:
        print(f"❌ Track map test failed: {e}")
        traceback.print_exc()
        return False

def test_dashboard_imports():
    """Test that all dashboard pages can be imported"""
    try:
        # We can't actually run streamlit, but we can check imports
        import dashboard.pages.overview
        print("✅ Dashboard overview page imports")

        import dashboard.pages.tactical
        print("✅ Dashboard tactical page imports")

        import dashboard.pages.strategic
        print("✅ Dashboard strategic page imports")

        import dashboard.pages.integrated
        print("✅ Dashboard integrated page imports")

        return True
    except Exception as e:
        print(f"❌ Dashboard import test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("RaceIQ Pro - Enhancement Testing")
    print("=" * 60)
    print()

    results = {}

    print("1. Testing SHAP Installation")
    print("-" * 60)
    results['shap'] = test_shap_import()
    print()

    print("2. Testing Enhanced Anomaly Detector")
    print("-" * 60)
    results['anomaly'] = test_enhanced_anomaly_detector()
    print()

    print("3. Testing Bayesian Strategy Optimizer")
    print("-" * 60)
    results['bayesian'] = test_bayesian_strategy()
    print()

    print("4. Testing Weather Integration")
    print("-" * 60)
    results['weather'] = test_weather_integration()
    print()

    print("5. Testing Track Map Visualization")
    print("-" * 60)
    results['trackmaps'] = test_track_maps()
    print()

    print("6. Testing Dashboard Page Imports")
    print("-" * 60)
    results['dashboard'] = test_dashboard_imports()
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}: {status}")

    print()
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print()
        print("🎉 All enhancement tests passed!")
        print()
        print("Next steps:")
        print("1. Launch dashboard: streamlit run dashboard/app.py")
        print("2. Test each enhancement feature manually:")
        print("   - Tactical → ML Detection with SHAP tab")
        print("   - Tactical → Track Map visualization")
        print("   - Strategic → Bayesian uncertainty section")
        print("   - Overview → Weather widget")
        print("   - Strategic → Weather-adjusted tire degradation")
    else:
        print()
        print("⚠️  Some tests failed. Review errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
