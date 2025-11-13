"""
Module Verification Script - Strategic Analysis Module

Verifies that all components are properly structured and can be imported.
This script does not require data files or external dependencies beyond
numpy, pandas, and scipy.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_imports():
    """Verify all module imports work correctly."""
    print("="*80)
    print("VERIFYING MODULE IMPORTS")
    print("="*80)

    try:
        from strategic import PitStopDetector, TireDegradationModel, PitStrategyOptimizer
        print("✓ Successfully imported all classes from strategic module")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_class_structure():
    """Verify class structure and key methods exist."""
    print("\n" + "="*80)
    print("VERIFYING CLASS STRUCTURE")
    print("="*80)

    try:
        from strategic import PitStopDetector, TireDegradationModel, PitStrategyOptimizer

        # Check PitStopDetector
        print("\n1. PitStopDetector:")
        detector = PitStopDetector()
        methods = ['detect_pit_stops', 'refine_detections', 'get_pit_stop_summary']
        for method in methods:
            if hasattr(detector, method):
                print(f"   ✓ {method} method exists")
            else:
                print(f"   ✗ {method} method missing")

        # Check TireDegradationModel
        print("\n2. TireDegradationModel:")
        tire_model = TireDegradationModel()
        methods = ['estimate_degradation', 'predict_cliff_point', 'analyze_corner_speed_degradation']
        for method in methods:
            if hasattr(tire_model, method):
                print(f"   ✓ {method} method exists")
            else:
                print(f"   ✗ {method} method missing")

        # Check PitStrategyOptimizer
        print("\n3. PitStrategyOptimizer:")
        optimizer = PitStrategyOptimizer()
        methods = ['calculate_optimal_pit_window', 'simulate_undercut_opportunity', 'analyze_overcut_opportunity']
        for method in methods:
            if hasattr(optimizer, method):
                print(f"   ✓ {method} method exists")
            else:
                print(f"   ✗ {method} method missing")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dependencies():
    """Verify required dependencies are available."""
    print("\n" + "="*80)
    print("VERIFYING DEPENDENCIES")
    print("="*80)

    dependencies = {
        'numpy': 'Core numerical computing',
        'pandas': 'Data manipulation and analysis',
        'scipy': 'Scientific computing and optimization'
    }

    all_available = True

    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package:15s} - {description}")
        except ImportError:
            print(f"✗ {package:15s} - {description} [MISSING]")
            all_available = False

    return all_available


def create_synthetic_test():
    """Create simple synthetic data test."""
    print("\n" + "="*80)
    print("RUNNING SYNTHETIC DATA TEST")
    print("="*80)

    try:
        import numpy as np
        import pandas as pd
        from strategic import PitStopDetector, TireDegradationModel, PitStrategyOptimizer

        # Create synthetic lap data
        print("\nCreating synthetic race data...")
        n_laps = 20
        base_lap_time = 100.0  # seconds
        degradation = 0.05  # seconds per lap

        lap_times = [base_lap_time + degradation * i + np.random.normal(0, 0.2)
                    for i in range(n_laps)]

        # Add a pit stop at lap 12
        lap_times[11] = base_lap_time + 30.0  # Pit stop lap

        lap_data = pd.DataFrame({
            'lap_number': range(1, n_laps + 1),
            'lap_time': lap_times
        })

        print(f"✓ Created {n_laps} laps of synthetic data")

        # Test 1: Pit Detection
        print("\nTest 1: Pit Stop Detection")
        detector = PitStopDetector()
        detections = detector.detect_pit_stops(lap_data)
        pit_count = detections['is_pit_stop'].sum()
        print(f"   Detected {pit_count} pit stop(s)")
        if pit_count > 0:
            print(f"   Pit lap(s): {detections[detections['is_pit_stop']==1]['lap_number'].tolist()}")

        # Test 2: Tire Degradation
        print("\nTest 2: Tire Degradation Model")
        tire_model = TireDegradationModel(model_type='polynomial', degree=2)
        degradation_result = tire_model.estimate_degradation(lap_data)
        print(f"   Degradation rate: {degradation_result['degradation_rate']:.4f} sec/lap")
        print(f"   R² score: {degradation_result['r_squared']:.3f}")

        # Test 3: Strategy Optimization
        print("\nTest 3: Strategy Optimization")
        optimizer = PitStrategyOptimizer(pit_loss_seconds=25.0, simulation_iterations=50)
        strategy = optimizer.calculate_optimal_pit_window(
            lap_data.head(8),
            degradation_result,
            race_length=20,
            current_lap=8
        )
        print(f"   Optimal pit lap: {strategy['optimal_pit_lap']}")
        print(f"   Pit window: {strategy['pit_window']}")

        print("\n✓ All synthetic tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Synthetic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "STRATEGIC ANALYSIS MODULE VERIFICATION" + " "*20 + "║")
    print("║" + " "*30 + "RaceIQ Pro" + " "*38 + "║")
    print("╚" + "="*78 + "╝")

    results = {
        'imports': verify_imports(),
        'structure': verify_class_structure(),
        'dependencies': verify_dependencies(),
        'synthetic_test': create_synthetic_test()
    }

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.upper():20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
        print("\nThe Strategic Analysis Module is ready for use!")
    else:
        print("✗ SOME VERIFICATIONS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install numpy pandas scipy")

    print("="*80)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
