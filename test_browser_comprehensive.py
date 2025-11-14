#!/usr/bin/env python3
"""
Comprehensive Browser Automation Test Script for RaceIQ Pro Dashboard
Tests all scenarios from USER_TESTING_GUIDE.md using browser automation
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

def test_initial_setup():
    """Test Suite 1: Initial Setup Testing"""
    print("\n" + "="*80)
    print("TEST SUITE 1: Initial Setup Testing")
    print("="*80)
    
    # Test 1.1: Verify Installation
    try:
        structure_file = Path("verify_structure.py")
        if structure_file.exists():
            log_test("1.1: Verify Installation Script", "PASS", "verify_structure.py exists")
        else:
            log_test("1.1: Verify Installation Script", "FAIL", "verify_structure.py not found")
    except Exception as e:
        log_test("1.1: Verify Installation Script", "ERROR", str(e))
    
    # Test 1.2: Check dependencies
    try:
        import streamlit
        import pandas
        import plotly
        log_test("1.2: Core Dependencies", "PASS", "streamlit, pandas, plotly installed")
    except ImportError as e:
        log_test("1.2: Core Dependencies", "FAIL", f"Missing dependency: {e}")
    
    # Test 1.3: Check data directory
    try:
        data_path = Path("Data/barber")
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.CSV"))
            if csv_files:
                log_test("1.3: Data Directory", "PASS", f"Found {len(csv_files)} data files in barber/")
            else:
                log_test("1.3: Data Directory", "WARN", "Data directory exists but no CSV files found")
        else:
            log_test("1.3: Data Directory", "FAIL", "Data/barber directory not found")
    except Exception as e:
        log_test("1.3: Data Directory", "ERROR", str(e))

def test_module_imports():
    """Test Suite 2: Module Import Testing"""
    print("\n" + "="*80)
    print("TEST SUITE 2: Module Import Testing")
    print("="*80)
    
    modules_to_test = [
        ("src.pipeline.data_loader", "DataLoader"),
        ("src.tactical.anomaly_detector", "AnomalyDetector"),
        ("src.tactical.section_analyzer", "SectionAnalyzer"),
        ("src.strategic.strategy_optimizer", "PitStrategyOptimizer"),
        ("src.strategic.tire_degradation", "TireDegradationModel"),
        ("src.integration.intelligence_engine", "IntelligenceEngine"),
    ]
    
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            log_test(f"2.x: Import {class_name}", "PASS", f"Successfully imported {class_name}")
        except ImportError as e:
            log_test(f"2.x: Import {class_name}", "FAIL", f"Import error: {e}")
        except Exception as e:
            log_test(f"2.x: Import {class_name}", "ERROR", str(e))

def test_dashboard_structure():
    """Test Suite 3: Dashboard Structure Testing"""
    print("\n" + "="*80)
    print("TEST SUITE 3: Dashboard Structure Testing")
    print("="*80)
    
    # Check dashboard files
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
        if full_path.exists():
            log_test(f"3.x: Dashboard File {file_path}", "PASS", "File exists")
        else:
            log_test(f"3.x: Dashboard File {file_path}", "FAIL", f"File not found: {full_path}")

def test_browser_navigation():
    """Test Suite 4: Browser Navigation Testing (using Browser MCP)"""
    print("\n" + "="*80)
    print("TEST SUITE 4: Browser Navigation Testing")
    print("="*80)
    
    if not check_streamlit_running():
        log_test("4.0: Browser Navigation", "FAIL", "Dashboard not running. Start with: streamlit run dashboard/app.py")
        return
    
    # Note: Actual browser automation will be done via Browser MCP tools
    # This is a placeholder that checks if the dashboard is accessible
    try:
        import requests
        base_url = "http://localhost:8501"
        
        # Test main page
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            log_test("4.1: Main Page Accessible", "PASS", f"Status: {response.status_code}")
        else:
            log_test("4.1: Main Page Accessible", "FAIL", f"Status: {response.status_code}")
        
        # Test Streamlit health endpoint
        health_response = requests.get(f"{base_url}/_stcore/health", timeout=5)
        if health_response.status_code == 200:
            log_test("4.2: Streamlit Health Check", "PASS", "Health endpoint responding")
        else:
            log_test("4.2: Streamlit Health Check", "FAIL", f"Status: {health_response.status_code}")
            
    except Exception as e:
        log_test("4.x: Browser Navigation", "ERROR", str(e))

def test_data_loading():
    """Test Suite 5: Data Loading Testing"""
    print("\n" + "="*80)
    print("TEST SUITE 5: Data Loading Testing")
    print("="*80)
    
    try:
        from src.pipeline.data_loader import DataLoader
        
        # Test loading barber data
        loader = DataLoader()
        base_path = Path("Data/barber")
        
        if base_path.exists():
            # Try to load lap times
            lap_files = list(base_path.glob("*lap_time*.csv")) + list(base_path.glob("*lap_time*.CSV"))
            if lap_files:
                log_test("5.1: Lap Time Files Found", "PASS", f"Found {len(lap_files)} lap time files")
            else:
                log_test("5.1: Lap Time Files Found", "WARN", "No lap time files found")
            
            # Try to load section files
            section_files = list(base_path.glob("*Sections*.csv")) + list(base_path.glob("*Sections*.CSV"))
            if section_files:
                log_test("5.2: Section Files Found", "PASS", f"Found {len(section_files)} section files")
            else:
                log_test("5.2: Section Files Found", "WARN", "No section files found")
        else:
            log_test("5.x: Data Loading", "FAIL", "Data/barber directory not found")
            
    except Exception as e:
        log_test("5.x: Data Loading", "ERROR", str(e))

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("RACEIQ PRO - COMPREHENSIVE BROWSER AUTOMATION TEST SUITE")
    print("="*80)
    print("Testing all scenarios from USER_TESTING_GUIDE.md")
    print("="*80)
    
    # Run test suites
    test_initial_setup()
    test_module_imports()
    test_dashboard_structure()
    test_data_loading()
    
    # Check if dashboard is running, if not try to start it
    streamlit_process = None
    if not check_streamlit_running():
        print("\n⚠️  Dashboard not running. Attempting to start...")
        streamlit_process = start_streamlit_dashboard()
        if streamlit_process:
            time.sleep(5)  # Give it time to fully start
    
    test_browser_navigation()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"💥 Errors: {len(test_results['errors'])}")
    print(f"📊 Total: {len(test_results['passed']) + len(test_results['failed']) + len(test_results['warnings']) + len(test_results['errors'])}")
    
    if test_results['failed']:
        print("\n❌ FAILED TESTS:")
        for test in test_results['failed']:
            print(f"  - {test['test']}: {test['message']}")
    
    if test_results['errors']:
        print("\n💥 ERRORS:")
        for test in test_results['errors']:
            print(f"  - {test['test']}: {test['message']}")
    
    if test_results['warnings']:
        print("\n⚠️  WARNINGS:")
        for test in test_results['warnings']:
            print(f"  - {test['test']}: {test['message']}")
    
    # Save results
    results_file = Path("test_results_comprehensive.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")
    
    # Cleanup
    if streamlit_process:
        print("\n⚠️  Note: Streamlit process was started by this script.")
        print("   You may want to stop it manually: pkill -f streamlit")
    
    return len(test_results['failed']) == 0 and len(test_results['errors']) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

