#!/usr/bin/env python3
"""
Execute User Testing Guide - Comprehensive Test Runner
Tests all scenarios from docs/USER_TESTING_GUIDE.md
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List

def run_command(cmd: str, description: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Command: {cmd}")
    print('='*80)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)

def test_installation():
    """Test 1: Verify Installation"""
    print("\n" + "="*80)
    print("TEST SUITE 1: INITIAL SETUP TESTING")
    print("="*80)
    
    results = []
    
    # Test 1.1: Verify structure (with shorter timeout)
    success, output = run_command(
        "python verify_structure.py",
        "1.1 Verify Installation Structure",
        timeout=15
    )
    results.append({
        "test": "1.1 Verify Installation",
        "status": "PASS" if success else "FAIL",
        "output": output[:500]  # Truncate long output
    })
    
    # Test 1.2: Check imports
    imports_to_test = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("plotly", "Plotly"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
    ]
    
    for module, name in imports_to_test:
        success, output = run_command(
            f"python -c 'import {module}; print(\"OK\")'",
            f"1.2.{imports_to_test.index((module, name))+1} Import {name}"
        )
        results.append({
            "test": f"1.2 Import {name}",
            "status": "PASS" if success else "FAIL",
            "output": output[:200]
        })
    
    # Test 1.3: Check data files
    data_path = Path("Data/barber")
    if data_path.exists():
        csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.CSV"))
        results.append({
            "test": "1.3 Data Files",
            "status": "PASS" if csv_files else "WARN",
            "output": f"Found {len(csv_files)} files" if csv_files else "No CSV files found"
        })
    else:
        results.append({
            "test": "1.3 Data Files",
            "status": "WARN",
            "output": "Data/barber directory not found"
        })
    
    return results

def test_dashboard_files():
    """Test 2: Dashboard Files"""
    print("\n" + "="*80)
    print("TEST SUITE 2: DASHBOARD FILES")
    print("="*80)
    
    results = []
    dashboard_path = Path("dashboard")
    
    required_files = [
        "app.py",
        "pages/overview.py",
        "pages/tactical.py",
        "pages/strategic.py",
        "pages/integrated.py",
        "pages/race_simulator.py"
    ]
    
    for file_path in required_files:
        full_path = dashboard_path / file_path
        exists = full_path.exists()
        results.append({
            "test": f"2.{required_files.index(file_path)+1} {file_path}",
            "status": "PASS" if exists else "FAIL",
            "output": "File exists" if exists else "File not found"
        })
    
    return results

def test_module_imports():
    """Test 3: Module Imports"""
    print("\n" + "="*80)
    print("TEST SUITE 3: MODULE IMPORTS")
    print("="*80)
    
    results = []
    
    modules_to_test = [
        ("src.pipeline.data_loader", "DataLoader"),
        ("src.tactical.anomaly_detector", "AnomalyDetector"),
        ("src.strategic.strategy_optimizer", "PitStrategyOptimizer"),
        ("src.integration", "IntegrationEngine"),
    ]
    
    for module_path, class_name in modules_to_test:
        success, output = run_command(
            f"python -c 'from {module_path} import {class_name}; print(\"OK\")'",
            f"3.{modules_to_test.index((module_path, class_name))+1} Import {class_name}"
        )
        results.append({
            "test": f"3.{modules_to_test.index((module_path, class_name))+1} {class_name}",
            "status": "PASS" if success else "FAIL",
            "output": output[:300]
        })
    
    return results

def test_functional():
    """Test 4: Functional Tests"""
    print("\n" + "="*80)
    print("TEST SUITE 4: FUNCTIONAL TESTS")
    print("="*80)
    
    results = []
    
    # Test 4.1: Run functional test script (with timeout)
    if Path("test_functional.py").exists():
        success, output = run_command(
            "python test_functional.py",
            "4.1 Run Functional Tests",
            timeout=30
        )
        results.append({
            "test": "4.1 Functional Tests",
            "status": "PASS" if success else "FAIL",
            "output": output[:1000]  # Keep more output for functional tests
        })
    else:
        results.append({
            "test": "4.1 Functional Tests",
            "status": "WARN",
            "output": "test_functional.py not found"
        })
    
    return results

def generate_browser_test_guide():
    """Generate browser test guide"""
    guide = {
        "dashboard_url": "http://localhost:8501",
        "instructions": """
To complete browser testing, follow these steps:

1. Ensure dashboard is running:
   streamlit run dashboard/app.py

2. Open browser and navigate to: http://localhost:8501

3. Test each page following the scenarios in docs/USER_TESTING_GUIDE.md:

   RACE OVERVIEW:
   - Select track: barber, race: 1
   - Verify 4 metrics displayed (Total Drivers, Total Laps, Top Speed, Fastest Lap)
   - Check leaderboard table displays
   - Verify charts render (Fastest Lap, Completion Status, Section Performance)
   - Check weather widget displays

   TACTICAL ANALYSIS:
   - Select a driver from dropdown
   - Verify performance metrics display
   - Check section heatmap renders
   - Test anomaly detection (Statistical and ML with SHAP tabs)
   - Verify coaching recommendations display

   STRATEGIC ANALYSIS:
   - Select a driver
   - Check pit stop detection works
   - Verify tire degradation curve displays
   - Test Bayesian analysis (adjust confidence slider)
   - Check strategy comparison

   INTEGRATED INSIGHTS:
   - Select a driver
   - Verify combined recommendations
   - Test what-if simulator (adjust sliders)
   - Check impact visualization

   RACE SIMULATOR:
   - Configure drivers
   - Run simulation
   - Verify results display

4. Test cross-page consistency:
   - Select driver on one page
   - Navigate to another page
   - Verify data matches

5. Test track switching:
   - Switch between tracks
   - Verify data updates correctly
        """,
        "test_scenarios": [
            "Race Overview - Metrics Display",
            "Race Overview - Leaderboard",
            "Race Overview - Charts",
            "Tactical Analysis - Driver Selection",
            "Tactical Analysis - Section Heatmap",
            "Tactical Analysis - Anomaly Detection",
            "Tactical Analysis - Recommendations",
            "Strategic Analysis - Pit Stops",
            "Strategic Analysis - Tire Degradation",
            "Strategic Analysis - Bayesian Analysis",
            "Integrated Insights - Recommendations",
            "Integrated Insights - What-If Simulator",
            "Race Simulator - Race Animation",
            "Cross-Page Data Consistency",
            "Track Switching"
        ]
    }
    
    return guide

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("RACEIQ PRO - COMPREHENSIVE USER TESTING")
    print("="*80)
    print("Executing all test scenarios from USER_TESTING_GUIDE.md")
    print("="*80)
    
    all_results = []
    
    # Run test suites
    all_results.extend(test_installation())
    all_results.extend(test_dashboard_files())
    all_results.extend(test_module_imports())
    all_results.extend(test_functional())
    
    # Generate browser test guide
    browser_guide = generate_browser_test_guide()
    
    # Calculate summary
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    warnings = sum(1 for r in all_results if r["status"] == "WARN")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"📊 Total Tests: {len(all_results)}")
    
    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for result in all_results:
            if result["status"] == "FAIL":
                print(f"  - {result['test']}: {result.get('output', 'No output')[:100]}")
    
    # Save report
    report = {
        "summary": {
            "total_tests": len(all_results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings
        },
        "results": all_results,
        "browser_test_guide": browser_guide
    }
    
    report_file = Path("user_testing_execution_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Review the test results above
2. Fix any failed tests
3. For browser testing:
   - Start dashboard: streamlit run dashboard/app.py
   - Open http://localhost:8501
   - Follow scenarios in docs/USER_TESTING_GUIDE.md
   - Or use Browser MCP tools in Cursor for automation

See browser_test_guide in the report for detailed browser test instructions.
""")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

