#!/usr/bin/env python3
"""
RaceIQ Pro Platform Testing Script

Comprehensive testing of all modules, data loading, and dashboard functionality.
Run this before deployment to ensure everything works.
"""

import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

# Test results tracker
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_imports():
    """Test that all required packages can be imported"""
    print_header("Testing Package Imports")

    required_packages = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('scipy', 'Scientific computing'),
        ('sklearn', 'Machine learning'),
        ('streamlit', 'Web framework'),
        ('plotly', 'Visualization'),
        ('matplotlib', 'Static plots'),
    ]

    optional_packages = [
        ('shap', 'Model explainability'),
        ('pymc3', 'Bayesian inference'),
        ('statsmodels', 'Statistical models'),
    ]

    all_passed = True

    # Test required packages
    print(f"{Colors.BOLD}Required Packages:{Colors.END}")
    for package, description in required_packages:
        try:
            __import__(package)
            print_success(f"{package:20} - {description}")
            test_results['passed'].append(f"Import {package}")
        except ImportError as e:
            print_error(f"{package:20} - {description} - FAILED")
            print(f"  Error: {str(e)}")
            test_results['failed'].append(f"Import {package}")
            all_passed = False

    # Test optional packages
    print(f"\n{Colors.BOLD}Optional Packages:{Colors.END}")
    for package, description in optional_packages:
        try:
            __import__(package)
            print_success(f"{package:20} - {description}")
            test_results['passed'].append(f"Import {package} (optional)")
        except ImportError:
            print_warning(f"{package:20} - {description} - Not installed (optional)")
            test_results['warnings'].append(f"Import {package} (optional)")

    return all_passed

def test_project_structure():
    """Test that all expected directories and files exist"""
    print_header("Testing Project Structure")

    expected_dirs = [
        'src',
        'src/pipeline',
        'src/tactical',
        'src/strategic',
        'src/integration',
        'src/utils',
        'dashboard',
        'dashboard/pages',
        'Data',
        'docs',
        'tests',
    ]

    expected_files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'src/__init__.py',
        'src/pipeline/data_loader.py',
        'src/tactical/optimal_ghost.py',
        'src/tactical/anomaly_detector.py',
        'src/strategic/pit_detector.py',
        'src/strategic/tire_degradation.py',
        'src/strategic/strategy_optimizer.py',
        'src/integration/intelligence_engine.py',
        'dashboard/app.py',
    ]

    all_passed = True

    print(f"{Colors.BOLD}Checking Directories:{Colors.END}")
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print_success(f"{dir_path}")
            test_results['passed'].append(f"Directory {dir_path}")
        else:
            print_error(f"{dir_path} - NOT FOUND")
            test_results['failed'].append(f"Directory {dir_path}")
            all_passed = False

    print(f"\n{Colors.BOLD}Checking Key Files:{Colors.END}")
    for file_path in expected_files:
        if Path(file_path).exists():
            print_success(f"{file_path}")
            test_results['passed'].append(f"File {file_path}")
        else:
            print_error(f"{file_path} - NOT FOUND")
            test_results['failed'].append(f"File {file_path}")
            all_passed = False

    return all_passed

def test_data_availability():
    """Test that sample data files are available"""
    print_header("Testing Data Availability")

    data_dir = Path('Data/barber/Samples')

    if not data_dir.exists():
        print_error(f"Sample data directory not found: {data_dir}")
        test_results['failed'].append("Sample data directory")
        return False

    expected_files = [
        'R1_barber_lap_time_sample.csv',
        'R1_barber_lap_start_sample.csv',
        'R1_barber_lap_end_sample.csv',
        'R1_barber_telemetry_data_sample.csv',
        '23_AnalysisEnduranceWithSections_Race_1_sample.CSV',
    ]

    all_passed = True
    for file_name in expected_files:
        file_path = data_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print_success(f"{file_name} ({size:,} bytes)")
            test_results['passed'].append(f"Data file {file_name}")
        else:
            print_error(f"{file_name} - NOT FOUND")
            test_results['failed'].append(f"Data file {file_name}")
            all_passed = False

    return all_passed

def test_module_imports():
    """Test that all custom modules can be imported"""
    print_header("Testing Custom Module Imports")

    modules_to_test = [
        ('src.pipeline.data_loader', 'DataLoader'),
        ('src.pipeline.validator', 'DataValidator'),
        ('src.pipeline.feature_engineer', 'FeatureEngineer'),
        ('src.tactical.optimal_ghost', 'OptimalGhostAnalyzer'),
        ('src.tactical.anomaly_detector', 'AnomalyDetector'),
        ('src.tactical.section_analyzer', 'SectionAnalyzer'),
        ('src.strategic.pit_detector', 'PitStopDetector'),
        ('src.strategic.tire_degradation', 'TireDegradationModel'),
        ('src.strategic.strategy_optimizer', 'PitStrategyOptimizer'),
        ('src.integration.intelligence_engine', 'IntegrationEngine'),
        ('src.integration.recommendation_builder', 'RecommendationBuilder'),
        ('src.utils.constants', None),
        ('src.utils.metrics', None),
        ('src.utils.visualization', None),
    ]

    all_passed = True

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                cls = getattr(module, class_name)
                print_success(f"{module_name}.{class_name}")
            else:
                print_success(f"{module_name}")
            test_results['passed'].append(f"Module {module_name}")
        except Exception as e:
            print_error(f"{module_name} - FAILED")
            print(f"  Error: {str(e)}")
            test_results['failed'].append(f"Module {module_name}")
            all_passed = False

    return all_passed

def test_data_loading():
    """Test data loading functionality"""
    print_header("Testing Data Loading")

    try:
        from src.pipeline.data_loader import DataLoader

        loader = DataLoader('Data/barber/Samples')
        print_success("DataLoader initialized")

        # Try to load each data type
        data_types = [
            ('lap_times', 'load_lap_time_data'),
            ('lap_starts', 'load_lap_start_data'),
            ('lap_ends', 'load_lap_end_data'),
            ('section_analysis', 'load_section_analysis'),
        ]

        all_passed = True
        for name, method in data_types:
            try:
                load_method = getattr(loader, method)
                data = load_method()
                if data is not None and len(data) > 0:
                    print_success(f"{name}: {len(data)} rows loaded")
                    test_results['passed'].append(f"Load {name}")
                else:
                    print_warning(f"{name}: No data found")
                    test_results['warnings'].append(f"Load {name}")
            except Exception as e:
                print_error(f"{name}: {str(e)}")
                test_results['failed'].append(f"Load {name}")
                all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"DataLoader test failed: {str(e)}")
        traceback.print_exc()
        test_results['failed'].append("Data loading")
        return False

def test_tactical_module():
    """Test tactical analysis module"""
    print_header("Testing Tactical Analysis Module")

    try:
        from src.tactical.optimal_ghost import OptimalGhostAnalyzer
        from src.tactical.anomaly_detector import AnomalyDetector
        from src.tactical.section_analyzer import SectionAnalyzer

        # Test class instantiation
        ghost = OptimalGhostAnalyzer()
        print_success("OptimalGhostAnalyzer instantiated")

        detector = AnomalyDetector()
        print_success("AnomalyDetector instantiated")

        analyzer = SectionAnalyzer()
        print_success("SectionAnalyzer instantiated")

        test_results['passed'].append("Tactical module classes")
        return True

    except Exception as e:
        print_error(f"Tactical module test failed: {str(e)}")
        traceback.print_exc()
        test_results['failed'].append("Tactical module")
        return False

def test_strategic_module():
    """Test strategic analysis module"""
    print_header("Testing Strategic Analysis Module")

    try:
        from src.strategic.pit_detector import PitStopDetector
        from src.strategic.tire_degradation import TireDegradationModel
        from src.strategic.strategy_optimizer import PitStrategyOptimizer

        # Test class instantiation
        detector = PitStopDetector()
        print_success("PitStopDetector instantiated")

        tire_model = TireDegradationModel()
        print_success("TireDegradationModel instantiated")

        optimizer = PitStrategyOptimizer()
        print_success("PitStrategyOptimizer instantiated")

        test_results['passed'].append("Strategic module classes")
        return True

    except Exception as e:
        print_error(f"Strategic module test failed: {str(e)}")
        traceback.print_exc()
        test_results['failed'].append("Strategic module")
        return False

def test_integration_module():
    """Test integration engine"""
    print_header("Testing Integration Engine")

    try:
        from src.integration.intelligence_engine import IntegrationEngine
        from src.integration.recommendation_builder import RecommendationBuilder

        # Test class instantiation
        engine = IntegrationEngine()
        print_success("IntegrationEngine instantiated")

        builder = RecommendationBuilder()
        print_success("RecommendationBuilder instantiated")

        test_results['passed'].append("Integration module classes")
        return True

    except Exception as e:
        print_error(f"Integration module test failed: {str(e)}")
        traceback.print_exc()
        test_results['failed'].append("Integration module")
        return False

def test_dashboard_structure():
    """Test dashboard file structure"""
    print_header("Testing Dashboard Structure")

    dashboard_files = [
        'dashboard/app.py',
        'dashboard/pages/__init__.py',
        'dashboard/pages/overview.py',
        'dashboard/pages/tactical.py',
        'dashboard/pages/strategic.py',
        'dashboard/pages/integrated.py',
    ]

    all_passed = True
    for file_path in dashboard_files:
        if Path(file_path).exists():
            # Check if file has content
            size = Path(file_path).stat().st_size
            if size > 0:
                print_success(f"{file_path} ({size:,} bytes)")
                test_results['passed'].append(f"Dashboard file {file_path}")
            else:
                print_warning(f"{file_path} - Empty file")
                test_results['warnings'].append(f"Dashboard file {file_path}")
        else:
            print_error(f"{file_path} - NOT FOUND")
            test_results['failed'].append(f"Dashboard file {file_path}")
            all_passed = False

    return all_passed

def generate_report():
    """Generate final test report"""
    print_header("Test Summary Report")

    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['warnings'])

    print(f"\n{Colors.BOLD}Results:{Colors.END}")
    print(f"  {Colors.GREEN}Passed:   {len(test_results['passed']):3d}{Colors.END}")
    print(f"  {Colors.RED}Failed:   {len(test_results['failed']):3d}{Colors.END}")
    print(f"  {Colors.YELLOW}Warnings: {len(test_results['warnings']):3d}{Colors.END}")
    print(f"  {Colors.BOLD}Total:    {total_tests:3d}{Colors.END}")

    if test_results['failed']:
        print(f"\n{Colors.BOLD}{Colors.RED}Failed Tests:{Colors.END}")
        for failure in test_results['failed']:
            print(f"  • {failure}")

    if test_results['warnings']:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Warnings:{Colors.END}")
        for warning in test_results['warnings']:
            print(f"  • {warning}")

    # Overall status
    print()
    if len(test_results['failed']) == 0:
        print_success("ALL TESTS PASSED!")
        print_info("Platform is ready for testing with Streamlit")
        print_info("Run: streamlit run dashboard/app.py")
        return True
    else:
        print_error("SOME TESTS FAILED")
        print_info("Please fix the issues above before running the dashboard")
        return False

def main():
    """Run all tests"""
    print_header(f"RaceIQ Pro Platform Test Suite")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Availability", test_data_availability),
        ("Module Imports", test_module_imports),
        ("Data Loading", test_data_loading),
        ("Tactical Module", test_tactical_module),
        ("Strategic Module", test_strategic_module),
        ("Integration Module", test_integration_module),
        ("Dashboard Structure", test_dashboard_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {str(e)}")
            traceback.print_exc()
            results.append(False)

    # Generate report
    all_passed = generate_report()

    # Save report to file
    report_file = Path('test_results.txt')
    with open(report_file, 'w') as f:
        f.write(f"RaceIQ Pro Test Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Passed: {len(test_results['passed'])}\n")
        f.write(f"Failed: {len(test_results['failed'])}\n")
        f.write(f"Warnings: {len(test_results['warnings'])}\n\n")

        if test_results['failed']:
            f.write("Failed Tests:\n")
            for failure in test_results['failed']:
                f.write(f"  - {failure}\n")

        if test_results['warnings']:
            f.write("\nWarnings:\n")
            for warning in test_results['warnings']:
                f.write(f"  - {warning}\n")

    print(f"\n{Colors.BLUE}Test report saved to: {report_file}{Colors.END}")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
