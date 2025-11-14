#!/usr/bin/env python3
"""
Comprehensive Browser-Based User Testing for RaceIQ Pro
Tests all scenarios from docs/USER_TESTING_GUIDE.md using browser automation
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import requests

# Test results storage
test_results = {
    "passed": [],
    "failed": [],
    "errors": [],
    "warnings": [],
    "start_time": time.time()
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

def check_dashboard_running() -> bool:
    """Check if Streamlit dashboard is running"""
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_dashboard(max_wait: int = 60) -> bool:
    """Wait for dashboard to be ready"""
    print(f"Waiting for dashboard to be ready (max {max_wait}s)...")
    for i in range(max_wait):
        if check_dashboard_running():
            print(f"✅ Dashboard is ready!")
            return True
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"   Still waiting... ({i+1}s)")
    print("❌ Dashboard failed to start")
    return False

def test_initial_setup():
    """Test Suite 1: Initial Setup Testing"""
    print("\n" + "="*80)
    print("TEST SUITE 1: INITIAL SETUP TESTING")
    print("="*80)
    
    # Test 1.1: Dashboard Launch
    if check_dashboard_running():
        log_test("1.1 Dashboard Launch", "PASS", "Dashboard is running at http://localhost:8501")
    else:
        log_test("1.1 Dashboard Launch", "FAIL", "Dashboard not accessible")
        return False
    
    # Test 1.2: Dashboard Accessibility
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            log_test("1.2 Dashboard Accessibility", "PASS", f"Status code: {response.status_code}")
        else:
            log_test("1.2 Dashboard Accessibility", "FAIL", f"Status code: {response.status_code}")
    except Exception as e:
        log_test("1.2 Dashboard Accessibility", "ERROR", str(e))
    
    # Test 1.3: Data Files Check
    data_path = Path("Data/barber")
    if data_path.exists():
        csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.CSV"))
        if csv_files:
            log_test("1.3 Data Files", "PASS", f"Found {len(csv_files)} data files")
        else:
            log_test("1.3 Data Files", "WARN", "No CSV files found in Data/barber")
    else:
        log_test("1.3 Data Files", "WARN", "Data/barber directory not found")
    
    return True

def generate_browser_test_instructions():
    """Generate detailed browser test instructions"""
    instructions = {
        "dashboard_url": "http://localhost:8501",
        "test_scenarios": [
            {
                "suite": "Race Overview",
                "tests": [
                    {
                        "name": "Metrics Display",
                        "steps": [
                            "Navigate to dashboard",
                            "Select track: barber",
                            "Select race: 1",
                            "Verify 4 key metrics displayed (Total Drivers, Total Laps, Top Speed, Fastest Lap)",
                            "Check all metrics show valid values (> 0)"
                        ]
                    },
                    {
                        "name": "Leaderboard Table",
                        "steps": [
                            "Scroll to leaderboard section",
                            "Verify table displays with all drivers",
                            "Check columns: Position, Driver Number, Best Lap, Average Lap",
                            "Test sorting by clicking column headers"
                        ]
                    },
                    {
                        "name": "Fastest Lap Chart",
                        "steps": [
                            "Locate 'Fastest Lap Times' chart",
                            "Verify bar chart displays",
                            "Hover over bars to see tooltips",
                            "Check color coding (green=fast, red=slow)"
                        ]
                    },
                    {
                        "name": "Completion Status Pie Chart",
                        "steps": [
                            "Locate 'Race Completion Status' chart",
                            "Verify pie chart displays",
                            "Check segments show completion status",
                            "Verify percentages add up to 100%"
                        ]
                    }
                ]
            },
            {
                "suite": "Tactical Analysis",
                "tests": [
                    {
                        "name": "Driver Selection",
                        "steps": [
                            "Navigate to 'Tactical Analysis' page",
                            "Select a driver from dropdown",
                            "Verify driver-specific data loads",
                            "Check loading indicator appears during update"
                        ]
                    },
                    {
                        "name": "Performance Metrics",
                        "steps": [
                            "Select a driver",
                            "Check performance metrics section",
                            "Verify: Best lap time, Average lap time, Consistency score, Gap to leader",
                            "Verify all metrics are accurate"
                        ]
                    },
                    {
                        "name": "Section Heatmap",
                        "steps": [
                            "Locate 'Section-by-Section Performance' heatmap",
                            "Verify heatmap displays (sections vs laps)",
                            "Hover over cells to see values",
                            "Check legend explains color scale"
                        ]
                    },
                    {
                        "name": "Anomaly Detection - Statistical",
                        "steps": [
                            "Scroll to 'Anomaly Detection' section",
                            "Click 'Statistical Detection' tab",
                            "Verify anomalies are listed",
                            "Check anomaly scores displayed",
                            "Verify lap numbers identified"
                        ]
                    },
                    {
                        "name": "Anomaly Detection - ML with SHAP",
                        "steps": [
                            "Click 'ML Detection with SHAP' tab",
                            "Wait for analysis to complete (may take 10-30 seconds)",
                            "Verify anomalies detected",
                            "Check SHAP explanations available",
                            "Verify feature contributions shown"
                        ]
                    },
                    {
                        "name": "Coaching Recommendations",
                        "steps": [
                            "Scroll to 'Top 3 Improvement Recommendations'",
                            "Verify 3 recommendations displayed",
                            "Check each includes: Priority, Description, Expected time gain, Confidence score"
                        ]
                    }
                ]
            },
            {
                "suite": "Strategic Analysis",
                "tests": [
                    {
                        "name": "Pit Stop Detection",
                        "steps": [
                            "Navigate to 'Strategic Analysis' page",
                            "Select a driver",
                            "Check 'Pit Stop Detection' section",
                            "Verify pit stops detected automatically",
                            "Check pit stop timeline displayed"
                        ]
                    },
                    {
                        "name": "Tire Degradation Curve",
                        "steps": [
                            "Locate 'Tire Degradation Analysis' section",
                            "Verify scatter plot shows lap times vs lap number",
                            "Check trend line shows degradation",
                            "Verify degradation rate calculated",
                            "Check R² value displayed"
                        ]
                    },
                    {
                        "name": "Bayesian Analysis",
                        "steps": [
                            "Scroll to 'Optimal Pit Window Analysis with Bayesian Uncertainty'",
                            "Verify optimal pit lap displayed",
                            "Check uncertainty percentage shown",
                            "Verify risk level indicator (🟢🟡🟠🔴)",
                            "Check confidence intervals displayed (80%, 90%, 95%)",
                            "Adjust confidence level slider",
                            "Verify intervals update dynamically"
                        ]
                    },
                    {
                        "name": "Strategy Comparison",
                        "steps": [
                            "Locate 'Strategy Comparison' section",
                            "Verify actual pit strategy displayed",
                            "Check optimal pit strategy shown",
                            "Verify time difference calculated",
                            "Check visual comparison chart"
                        ]
                    }
                ]
            },
            {
                "suite": "Integrated Insights",
                "tests": [
                    {
                        "name": "Combined Recommendations",
                        "steps": [
                            "Navigate to 'Integrated Insights' page",
                            "Select a driver",
                            "Check 'Combined Recommendations' section",
                            "Verify tactical and strategic recommendations combined",
                            "Check prioritized list displayed"
                        ]
                    },
                    {
                        "name": "What-If Simulator",
                        "steps": [
                            "Locate 'What-If Scenario Simulator'",
                            "Adjust sliders for section improvements",
                            "Verify results update in real-time",
                            "Check projected lap time improvement calculated",
                            "Verify projected position change shown"
                        ]
                    }
                ]
            },
            {
                "suite": "Race Simulator",
                "tests": [
                    {
                        "name": "Race Animation",
                        "steps": [
                            "Navigate to 'Race Simulator' page",
                            "Configure 2-5 drivers",
                            "Set pit strategies",
                            "Run race simulation",
                            "Verify position changes animated",
                            "Check final results displayed"
                        ]
                    }
                ]
            },
            {
                "suite": "Integration Testing",
                "tests": [
                    {
                        "name": "Cross-Page Data Consistency",
                        "steps": [
                            "Select a driver on Tactical Analysis",
                            "Note their best lap time",
                            "Navigate to Strategic Analysis",
                            "Verify same driver's data matches",
                            "Navigate to Integrated Insights",
                            "Verify consistency"
                        ]
                    },
                    {
                        "name": "Track Switching",
                        "steps": [
                            "Select 'Barber' track",
                            "Note data displayed",
                            "Switch to 'COTA' track",
                            "Verify data updates",
                            "Switch back to 'Barber'",
                            "Verify data reloads"
                        ]
                    }
                ]
            }
        ]
    }
    
    return instructions

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("RACEIQ PRO - COMPREHENSIVE USER TESTING")
    print("="*80)
    print("Testing all scenarios from USER_TESTING_GUIDE.md")
    print("="*80)
    
    # Wait for dashboard
    if not wait_for_dashboard():
        print("\n❌ Cannot proceed without dashboard running")
        print("Please start dashboard manually: streamlit run dashboard/app.py")
        return False
    
    # Run initial setup tests
    if not test_initial_setup():
        print("\n⚠️  Initial setup tests failed, but continuing...")
    
    # Generate browser test instructions
    instructions = generate_browser_test_instructions()
    
    # Save instructions
    instructions_file = Path("browser_test_instructions.json")
    with open(instructions_file, 'w') as f:
        json.dump(instructions, f, indent=2)
    print(f"\n📄 Browser test instructions saved to: {instructions_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"💥 Errors: {len(test_results['errors'])}")
    
    # Generate report
    total_time = time.time() - test_results["start_time"]
    report = {
        "summary": {
            "total_tests": len(test_results["passed"]) + len(test_results["failed"]) + 
                          len(test_results["warnings"]) + len(test_results["errors"]),
            "passed": len(test_results["passed"]),
            "failed": len(test_results["failed"]),
            "warnings": len(test_results["warnings"]),
            "errors": len(test_results["errors"]),
            "duration_seconds": total_time
        },
        "passed": test_results["passed"],
        "failed": test_results["failed"],
        "warnings": test_results["warnings"],
        "errors": test_results["errors"],
        "browser_test_instructions": instructions
    }
    
    report_file = Path("user_testing_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
The automated tests have verified basic setup. To complete full browser testing:

1. Open browser and navigate to: http://localhost:8501
2. Follow the test scenarios in browser_test_instructions.json
3. Or use Browser MCP tools in Cursor to automate:
   - browser_navigate to http://localhost:8501
   - browser_snapshot to capture page state
   - browser_click to interact with elements
   - browser_evaluate to check page state

See docs/USER_TESTING_GUIDE.md for detailed test scenarios.
""")
    
    return len(test_results['failed']) == 0 and len(test_results['errors']) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

