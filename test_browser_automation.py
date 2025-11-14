#!/usr/bin/env python3
"""
Browser Automation Test Script for RaceIQ Pro Dashboard
Tests all scenarios from USER_TESTING_GUIDE.md
"""

import time
import json
from pathlib import Path

# Test results storage
test_results = {
    "passed": [],
    "failed": [],
    "errors": []
}

def log_test(test_name, status, message=""):
    """Log test result"""
    result = {
        "test": test_name,
        "status": status,
        "message": message,
        "timestamp": time.time()
    }
    if status == "PASS":
        test_results["passed"].append(result)
        print(f"✅ PASS: {test_name}")
    else:
        test_results["failed"].append(result)
        print(f"❌ FAIL: {test_name} - {message}")
    return result

def test_initial_setup():
    """Test 1: Verify Installation"""
    print("\n" + "="*80)
    print("TEST SUITE 1: Initial Setup Testing")
    print("="*80)
    
    # Test 1.1: Verify structure
    try:
        structure_file = Path("verify_structure.py")
        if structure_file.exists():
            log_test("1.1: Verify Installation", "PASS", "Structure verification script exists")
        else:
            log_test("1.1: Verify Installation", "FAIL", "verify_structure.py not found")
    except Exception as e:
        log_test("1.1: Verify Installation", "FAIL", str(e))
    
    # Test 1.2: Check dashboard can start
    import subprocess
    import requests
    import time
    
    try:
        # Check if dashboard is running
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        if response.status_code == 200:
            log_test("1.2: Dashboard Health Check", "PASS", "Dashboard is running")
        else:
            log_test("1.2: Dashboard Health Check", "FAIL", f"Status code: {response.status_code}")
    except Exception as e:
        log_test("1.2: Dashboard Health Check", "FAIL", str(e))

def test_data_loading():
    """Test 2: Data Loading"""
    print("\n" + "="*80)
    print("TEST SUITE 2: Data Loading Testing")
    print("="*80)
    
    try:
        from src.pipeline.data_loader import DataLoader
        loader = DataLoader()
        
        # Test loading barber data
        data_path = Path("Data/barber")
        if data_path.exists():
            log_test("2.1: Data Directory Exists", "PASS", "Barber data directory found")
        else:
            log_test("2.1: Data Directory Exists", "FAIL", "Barber data directory not found")
            
    except Exception as e:
        log_test("2.1: Data Loading", "FAIL", str(e))

def test_module_imports():
    """Test 3: Module Imports"""
    print("\n" + "="*80)
    print("TEST SUITE 3: Module Import Testing")
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
            log_test(f"3.x: Import {class_name}", "PASS", f"Successfully imported {class_name}")
        except Exception as e:
            log_test(f"3.x: Import {class_name}", "FAIL", str(e))

def test_dashboard_pages():
    """Test 4: Dashboard Pages"""
    print("\n" + "="*80)
    print("TEST SUITE 4: Dashboard Page Testing")
    print("="*80)
    
    pages = [
        ("overview", "Race Overview"),
        ("tactical", "Tactical Analysis"),
        ("strategic", "Strategic Analysis"),
        ("integrated", "Integrated Insights"),
        ("race_simulator", "Race Simulator"),
    ]
    
    import requests
    base_url = "http://localhost:8501"
    
    for page_path, page_name in pages:
        try:
            # Test page accessibility
            url = f"{base_url}/{page_path}"
            response = requests.get(url, timeout=10, allow_redirects=True)
            if response.status_code in [200, 302]:
                log_test(f"4.x: {page_name} Page Accessible", "PASS", f"Status: {response.status_code}")
            else:
                log_test(f"4.x: {page_name} Page Accessible", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            log_test(f"4.x: {page_name} Page Accessible", "FAIL", str(e))

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("RACEIQ PRO - BROWSER AUTOMATION TEST SUITE")
    print("="*80)
    
    test_initial_setup()
    test_data_loading()
    test_module_imports()
    test_dashboard_pages()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"📊 Total: {len(test_results['passed']) + len(test_results['failed'])}")
    
    if test_results['failed']:
        print("\nFAILED TESTS:")
        for test in test_results['failed']:
            print(f"  - {test['test']}: {test['message']}")
    
    # Save results
    results_file = Path("test_results_browser.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")
    
    return len(test_results['failed']) == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

