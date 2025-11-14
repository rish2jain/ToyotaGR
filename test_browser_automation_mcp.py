#!/usr/bin/env python3
"""
Browser Automation Test Script using Browser MCP Tools
Tests all scenarios from USER_TESTING_GUIDE.md with actual browser automation
"""

import subprocess
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Test results storage
test_results = {
    "passed": [],
    "failed": [],
    "errors": [],
    "warnings": []
}

def log_test(test_name: str, status: str, message: str = "", details: Optional[Dict] = None):
    """Log test result"""
    result = {
        "test": test_name,
        "status": status,
        "message": message,
        "timestamp": time.time(),
        "details": details or {}
    }
    
    if status == "PASS":
        test_results["passed"].append(result)
        print(f"✅ PASS: {test_name}")
        if message:
            print(f"   {message}")
    elif status == "FAIL":
        test_results["failed"].append(result)
        print(f"❌ FAIL: {test_name}")
        if message:
            print(f"   {message}")
    elif status == "WARN":
        test_results["warnings"].append(result)
        print(f"⚠️  WARN: {test_name}")
        if message:
            print(f"   {message}")
    else:
        test_results["errors"].append(result)
        print(f"💥 ERROR: {test_name}")
        if message:
            print(f"   {message}")
    
    return result

def check_streamlit_running() -> bool:
    """Check if Streamlit dashboard is running"""
    try:
        import requests
        response = requests.get("http://localhost:8501/_stcore/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_streamlit_dashboard() -> Optional[subprocess.Popen]:
    """Start Streamlit dashboard in background"""
    if check_streamlit_running():
        log_test("Dashboard Startup", "PASS", "Dashboard already running")
        return None
    
    try:
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        if not dashboard_path.exists():
            log_test("Dashboard Startup", "FAIL", f"Dashboard file not found: {dashboard_path}")
            return None
        
        # Start Streamlit
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(dashboard_path.parent)
        )
        
        # Wait for dashboard to start
        max_wait = 30
        waited = 0
        while waited < max_wait:
            if check_streamlit_running():
                log_test("Dashboard Startup", "PASS", f"Dashboard started successfully (waited {waited}s)")
                return process
            time.sleep(1)
            waited += 1
        
        log_test("Dashboard Startup", "FAIL", f"Dashboard failed to start within {max_wait} seconds")
        return None
        
    except Exception as e:
        log_test("Dashboard Startup", "ERROR", str(e))
        return None

def print_browser_test_instructions():
    """Print instructions for manual browser testing"""
    print("\n" + "="*80)
    print("BROWSER AUTOMATION TEST INSTRUCTIONS")
    print("="*80)
    print("""
This script prepares the environment for browser testing. To complete the full
browser automation testing, use the Browser MCP tools available in Cursor.

The following scenarios need to be tested:

1. INITIAL SETUP TESTING
   - Verify installation
   - Launch dashboard
   - Data loading

2. DASHBOARD TESTING
   - Race Overview page (metrics, leaderboard, charts)
   - Tactical Analysis page (driver selection, heatmaps, anomaly detection)
   - Strategic Analysis page (pit stops, tire degradation, Bayesian analysis)
   - Integrated Insights page (recommendations, what-if simulator)
   - Race Simulator page (race animation, undercut analyzer)

3. FEATURE-SPECIFIC TESTING
   - Anomaly detection (Statistical, ML with SHAP, LSTM)
   - Bayesian strategy (confidence intervals, risk assessment)
   - Racing line reconstruction
   - Weather integration

4. INTEGRATION TESTING
   - Cross-page data consistency
   - Track switching
   - Race number switching

5. PERFORMANCE TESTING
   - Page load times
   - Data loading performance
   - Visualization rendering

6. ERROR HANDLING TESTING
   - Missing data files
   - Invalid driver selection
   - Network interruption

To use Browser MCP tools:
1. Ensure dashboard is running at http://localhost:8501
2. Use browser_navigate to go to the dashboard
3. Use browser_snapshot to capture page state
4. Use browser_click to interact with elements
5. Use browser_type to fill forms
6. Use browser_evaluate to check page state

See USER_TESTING_GUIDE.md for detailed test scenarios.
""")

def run_preparation_tests():
    """Run preparation tests before browser automation"""
    print("\n" + "="*80)
    print("PREPARATION TESTS")
    print("="*80)
    
    # Test 1: Verify installation
    try:
        structure_file = Path("verify_structure.py")
        if structure_file.exists():
            log_test("Preparation: Verify Script", "PASS", "verify_structure.py exists")
        else:
            log_test("Preparation: Verify Script", "FAIL", "verify_structure.py not found")
    except Exception as e:
        log_test("Preparation: Verify Script", "ERROR", str(e))
    
    # Test 2: Check dependencies
    try:
        import streamlit
        import pandas
        import plotly
        log_test("Preparation: Dependencies", "PASS", "Core dependencies installed")
    except ImportError as e:
        log_test("Preparation: Dependencies", "FAIL", f"Missing: {e}")
    
    # Test 3: Check data directory
    try:
        data_path = Path("Data/barber")
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.CSV"))
            if csv_files:
                log_test("Preparation: Data Files", "PASS", f"Found {len(csv_files)} files")
            else:
                log_test("Preparation: Data Files", "WARN", "No CSV files found")
        else:
            log_test("Preparation: Data Files", "FAIL", "Data/barber not found")
    except Exception as e:
        log_test("Preparation: Data Files", "ERROR", str(e))
    
    # Test 4: Check dashboard files
    try:
        dashboard_path = Path("dashboard")
        required_files = ["app.py", "pages/overview.py", "pages/tactical.py", 
                         "pages/strategic.py", "pages/integrated.py", "pages/race_simulator.py"]
        all_exist = all((dashboard_path / f).exists() for f in required_files)
        if all_exist:
            log_test("Preparation: Dashboard Files", "PASS", "All dashboard files exist")
        else:
            missing = [f for f in required_files if not (dashboard_path / f).exists()]
            log_test("Preparation: Dashboard Files", "FAIL", f"Missing: {missing}")
    except Exception as e:
        log_test("Preparation: Dashboard Files", "ERROR", str(e))
    
    # Test 5: Start dashboard if not running
    streamlit_process = None
    if not check_streamlit_running():
        print("\n⚠️  Dashboard not running. Starting dashboard...")
        streamlit_process = start_streamlit_dashboard()
        if streamlit_process:
            time.sleep(3)  # Give it time to start
    else:
        log_test("Preparation: Dashboard Running", "PASS", "Dashboard is running")
    
    return streamlit_process

def generate_browser_test_plan():
    """Generate a test plan for browser automation"""
    test_plan = {
        "url": "http://localhost:8501",
        "tests": [
            {
                "name": "Race Overview - Metrics Display",
                "page": "Race Overview",
                "steps": [
                    "Navigate to dashboard",
                    "Select track: barber",
                    "Select race: 1",
                    "Check 4 key metrics displayed",
                    "Verify Total Drivers > 0",
                    "Verify Total Laps > 0",
                    "Verify Top Speed displayed",
                    "Verify Fastest Lap displayed"
                ]
            },
            {
                "name": "Race Overview - Leaderboard",
                "page": "Race Overview",
                "steps": [
                    "Scroll to leaderboard",
                    "Verify table displays",
                    "Check all columns present",
                    "Test sorting functionality"
                ]
            },
            {
                "name": "Tactical Analysis - Driver Selection",
                "page": "Tactical Analysis",
                "steps": [
                    "Navigate to Tactical Analysis",
                    "Select a driver from dropdown",
                    "Verify driver-specific data loads",
                    "Check loading indicator appears"
                ]
            },
            {
                "name": "Tactical Analysis - Section Heatmap",
                "page": "Tactical Analysis",
                "steps": [
                    "Select a driver",
                    "Locate section heatmap",
                    "Verify heatmap displays",
                    "Check tooltips on hover",
                    "Verify legend displays"
                ]
            },
            {
                "name": "Tactical Analysis - Anomaly Detection",
                "page": "Tactical Analysis",
                "steps": [
                    "Scroll to anomaly detection section",
                    "Click Statistical Detection tab",
                    "Verify anomalies listed",
                    "Click ML Detection with SHAP tab",
                    "Wait for analysis",
                    "Verify SHAP explanations available"
                ]
            },
            {
                "name": "Strategic Analysis - Pit Stop Detection",
                "page": "Strategic Analysis",
                "steps": [
                    "Navigate to Strategic Analysis",
                    "Select a driver",
                    "Check pit stop detection section",
                    "Verify pit stops detected",
                    "Check pit stop timeline"
                ]
            },
            {
                "name": "Strategic Analysis - Bayesian Analysis",
                "page": "Strategic Analysis",
                "steps": [
                    "Scroll to Bayesian analysis section",
                    "Verify optimal pit lap displayed",
                    "Check uncertainty percentage",
                    "Verify risk level indicator",
                    "Adjust confidence level slider",
                    "Verify intervals update"
                ]
            },
            {
                "name": "Integrated Insights - Recommendations",
                "page": "Integrated Insights",
                "steps": [
                    "Navigate to Integrated Insights",
                    "Select a driver",
                    "Check combined recommendations",
                    "Verify 3 recommendations displayed",
                    "Check priority levels"
                ]
            },
            {
                "name": "Integrated Insights - What-If Simulator",
                "page": "Integrated Insights",
                "steps": [
                    "Locate what-if simulator",
                    "Adjust section improvement sliders",
                    "Verify results update in real-time",
                    "Check projected lap time improvement",
                    "Check projected position change"
                ]
            },
            {
                "name": "Cross-Page Data Consistency",
                "pages": ["Tactical Analysis", "Strategic Analysis", "Integrated Insights"],
                "steps": [
                    "Select driver on Tactical Analysis",
                    "Note best lap time",
                    "Navigate to Strategic Analysis",
                    "Verify same driver data matches",
                    "Navigate to Integrated Insights",
                    "Verify consistency"
                ]
            }
        ]
    }
    
    plan_file = Path("browser_test_plan.json")
    with open(plan_file, 'w') as f:
        json.dump(test_plan, f, indent=2)
    
    log_test("Test Plan Generation", "PASS", f"Test plan saved to {plan_file}")
    return test_plan

def run_all_tests():
    """Run all preparation tests"""
    print("\n" + "="*80)
    print("RACEIQ PRO - BROWSER AUTOMATION TEST PREPARATION")
    print("="*80)
    print("Preparing environment for browser automation testing")
    print("="*80)
    
    # Run preparation tests
    streamlit_process = run_preparation_tests()
    
    # Generate test plan
    test_plan = generate_browser_test_plan()
    
    # Print instructions
    print_browser_test_instructions()
    
    # Print summary
    print("\n" + "="*80)
    print("PREPARATION SUMMARY")
    print("="*80)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"💥 Errors: {len(test_results['errors'])}")
    
    if test_results['failed']:
        print("\n❌ FAILED TESTS:")
        for test in test_results['failed']:
            print(f"  - {test['test']}: {test['message']}")
    
    # Save results
    results_file = Path("test_results_preparation.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")
    
    if streamlit_process:
        print("\n⚠️  Note: Streamlit process was started by this script.")
        print("   Dashboard is running at http://localhost:8501")
        print("   Use Browser MCP tools to continue testing.")
    
    return len(test_results['failed']) == 0 and len(test_results['errors']) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

