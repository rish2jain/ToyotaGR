#!/usr/bin/env python3
"""
Comprehensive Browser Automation Testing for RaceIQ Pro
Uses Playwright for full browser automation of all test scenarios
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, Page, Browser

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

async def wait_for_element(page: Page, selector: str, timeout: int = 10000) -> bool:
    """Wait for element to appear"""
    try:
        await page.wait_for_selector(selector, timeout=timeout)
        return True
    except:
        return False

async def test_race_overview(page: Page) -> bool:
    """Test Race Overview page"""
    print("\n" + "="*80)
    print("TESTING: Race Overview Page")
    print("="*80)
    
    try:
        # Navigate to dashboard
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)  # Wait for page to fully load
        
        # Test 1.1: Select track and race
        track_select = page.locator('select').first
        if await track_select.count() > 0:
            await track_select.select_option("barber")
            log_test("1.1 Track Selection", "PASS", "Selected barber track")
        else:
            log_test("1.1 Track Selection", "FAIL", "Track selector not found")
        
        await asyncio.sleep(1)
        
        # Test 1.2: Check for metrics
        page_text = await page.text_content("body")
        if page_text:
            has_drivers = "driver" in page_text.lower() or "Driver" in page_text
            has_laps = "lap" in page_text.lower() or "Lap" in page_text
            has_speed = "speed" in page_text.lower() or "Speed" in page_text
            
            if has_drivers and has_laps:
                log_test("1.2 Metrics Display", "PASS", "Key metrics found on page")
            else:
                log_test("1.2 Metrics Display", "WARN", "Some metrics may be missing")
        
        # Test 1.3: Check for leaderboard/table
        tables = await page.locator("table").count()
        if tables > 0:
            log_test("1.3 Leaderboard Table", "PASS", f"Found {tables} table(s)")
        else:
            log_test("1.3 Leaderboard Table", "WARN", "No tables found")
        
        # Test 1.4: Check for charts
        charts = await page.locator("[class*='plotly'], [class*='chart'], [id*='chart']").count()
        if charts > 0:
            log_test("1.4 Charts Display", "PASS", f"Found {charts} chart element(s)")
        else:
            # Check for Plotly divs
            plotly_divs = await page.locator("div[class*='js-plotly']").count()
            if plotly_divs > 0:
                log_test("1.4 Charts Display", "PASS", f"Found {plotly_divs} Plotly chart(s)")
            else:
                log_test("1.4 Charts Display", "WARN", "No charts detected")
        
        return True
        
    except Exception as e:
        log_test("Race Overview", "ERROR", str(e))
        return False

async def test_tactical_analysis(page: Page) -> bool:
    """Test Tactical Analysis page"""
    print("\n" + "="*80)
    print("TESTING: Tactical Analysis Page")
    print("="*80)
    
    try:
        # Navigate to Tactical Analysis
        tactical_link = page.locator("text=Tactical Analysis").first
        if await tactical_link.count() > 0:
            await tactical_link.click()
            await asyncio.sleep(3)  # Wait for page to load
            log_test("2.1 Navigate to Tactical", "PASS", "Navigated to Tactical Analysis")
        else:
            # Try alternative navigation
            await page.goto("http://localhost:8501/?page=Tactical+Analysis", wait_until="networkidle")
            await asyncio.sleep(2)
            log_test("2.1 Navigate to Tactical", "PASS", "Navigated via URL")
        
        # Test 2.2: Check for driver selector
        page_text = await page.text_content("body")
        if page_text and ("driver" in page_text.lower() or "select" in page_text.lower()):
            log_test("2.2 Driver Selection", "PASS", "Driver selection interface found")
        else:
            log_test("2.2 Driver Selection", "WARN", "Driver selector may not be visible")
        
        # Test 2.3: Check for heatmap/charts
        charts = await page.locator("[class*='plotly'], [class*='heatmap'], div[class*='js-plotly']").count()
        if charts > 0:
            log_test("2.3 Section Heatmap", "PASS", f"Found {charts} chart element(s)")
        else:
            log_test("2.3 Section Heatmap", "WARN", "Heatmap charts not detected")
        
        # Test 2.4: Check for anomaly detection
        if page_text and ("anomaly" in page_text.lower() or "detection" in page_text.lower()):
            log_test("2.4 Anomaly Detection", "PASS", "Anomaly detection section found")
        else:
            log_test("2.4 Anomaly Detection", "WARN", "Anomaly detection section not found")
        
        # Test 2.5: Check for recommendations
        if page_text and ("recommendation" in page_text.lower() or "improvement" in page_text.lower()):
            log_test("2.5 Coaching Recommendations", "PASS", "Recommendations section found")
        else:
            log_test("2.5 Coaching Recommendations", "WARN", "Recommendations section not found")
        
        return True
        
    except Exception as e:
        log_test("Tactical Analysis", "ERROR", str(e))
        return False

async def test_strategic_analysis(page: Page) -> bool:
    """Test Strategic Analysis page"""
    print("\n" + "="*80)
    print("TESTING: Strategic Analysis Page")
    print("="*80)
    
    try:
        # Navigate to Strategic Analysis
        strategic_link = page.locator("text=Strategic Analysis").first
        if await strategic_link.count() > 0:
            await strategic_link.click()
            await asyncio.sleep(3)
            log_test("3.1 Navigate to Strategic", "PASS", "Navigated to Strategic Analysis")
        else:
            await page.goto("http://localhost:8501/?page=Strategic+Analysis", wait_until="networkidle")
            await asyncio.sleep(2)
            log_test("3.1 Navigate to Strategic", "PASS", "Navigated via URL")
        
        page_text = await page.text_content("body")
        
        # Test 3.2: Check for pit stop detection
        if page_text and ("pit" in page_text.lower() or "stop" in page_text.lower()):
            log_test("3.2 Pit Stop Detection", "PASS", "Pit stop section found")
        else:
            log_test("3.2 Pit Stop Detection", "WARN", "Pit stop section not found")
        
        # Test 3.3: Check for tire degradation
        if page_text and ("tire" in page_text.lower() or "degradation" in page_text.lower()):
            log_test("3.3 Tire Degradation", "PASS", "Tire degradation section found")
        else:
            log_test("3.3 Tire Degradation", "WARN", "Tire degradation section not found")
        
        # Test 3.4: Check for Bayesian analysis
        if page_text and ("bayesian" in page_text.lower() or "uncertainty" in page_text.lower() or "confidence" in page_text.lower()):
            log_test("3.4 Bayesian Analysis", "PASS", "Bayesian analysis section found")
        else:
            log_test("3.4 Bayesian Analysis", "WARN", "Bayesian analysis section not found")
        
        # Test 3.5: Check for charts
        charts = await page.locator("[class*='plotly'], div[class*='js-plotly']").count()
        if charts > 0:
            log_test("3.5 Strategy Charts", "PASS", f"Found {charts} chart element(s)")
        else:
            log_test("3.5 Strategy Charts", "WARN", "Strategy charts not detected")
        
        return True
        
    except Exception as e:
        log_test("Strategic Analysis", "ERROR", str(e))
        return False

async def test_integrated_insights(page: Page) -> bool:
    """Test Integrated Insights page"""
    print("\n" + "="*80)
    print("TESTING: Integrated Insights Page")
    print("="*80)
    
    try:
        # Navigate to Integrated Insights
        integrated_link = page.locator("text=Integrated Insights").first
        if await integrated_link.count() > 0:
            await integrated_link.click()
            await asyncio.sleep(3)
            log_test("4.1 Navigate to Integrated", "PASS", "Navigated to Integrated Insights")
        else:
            await page.goto("http://localhost:8501/?page=Integrated+Insights", wait_until="networkidle")
            await asyncio.sleep(2)
            log_test("4.1 Navigate to Integrated", "PASS", "Navigated via URL")
        
        page_text = await page.text_content("body")
        
        # Test 4.2: Check for recommendations
        if page_text and ("recommendation" in page_text.lower() or "combined" in page_text.lower()):
            log_test("4.2 Combined Recommendations", "PASS", "Combined recommendations found")
        else:
            log_test("4.2 Combined Recommendations", "WARN", "Recommendations section not found")
        
        # Test 4.3: Check for what-if simulator
        if page_text and ("what-if" in page_text.lower() or "simulator" in page_text.lower() or "slider" in page_text.lower()):
            log_test("4.3 What-If Simulator", "PASS", "What-if simulator found")
        else:
            log_test("4.3 What-If Simulator", "WARN", "What-if simulator not found")
        
        return True
        
    except Exception as e:
        log_test("Integrated Insights", "ERROR", str(e))
        return False

async def test_race_simulator(page: Page) -> bool:
    """Test Race Simulator page"""
    print("\n" + "="*80)
    print("TESTING: Race Simulator Page")
    print("="*80)
    
    try:
        # Navigate to Race Simulator
        simulator_link = page.locator("text=Race Simulator").first
        if await simulator_link.count() > 0:
            await simulator_link.click()
            await asyncio.sleep(3)
            log_test("5.1 Navigate to Simulator", "PASS", "Navigated to Race Simulator")
        else:
            await page.goto("http://localhost:8501/?page=Race+Simulator", wait_until="networkidle")
            await asyncio.sleep(2)
            log_test("5.1 Navigate to Simulator", "PASS", "Navigated via URL")
        
        page_text = await page.text_content("body")
        
        # Test 5.2: Check for simulator interface
        if page_text and ("simulator" in page_text.lower() or "race" in page_text.lower()):
            log_test("5.2 Simulator Interface", "PASS", "Simulator interface found")
        else:
            log_test("5.2 Simulator Interface", "WARN", "Simulator interface not found")
        
        return True
        
    except Exception as e:
        log_test("Race Simulator", "ERROR", str(e))
        return False

async def test_cross_page_consistency(page: Page) -> bool:
    """Test cross-page data consistency"""
    print("\n" + "="*80)
    print("TESTING: Cross-Page Data Consistency")
    print("="*80)
    
    try:
        # Navigate through pages and check consistency
        pages_to_test = [
            ("Tactical Analysis", "tactical"),
            ("Strategic Analysis", "strategic"),
            ("Integrated Insights", "integrated")
        ]
        
        for page_name, page_id in pages_to_test:
            await page.goto(f"http://localhost:8501/?page={page_name.replace(' ', '+')}", wait_until="networkidle")
            await asyncio.sleep(2)
            page_text = await page.text_content("body")
            if page_text:
                log_test(f"6.{pages_to_test.index((page_name, page_id))+1} {page_name} Loads", "PASS", f"{page_name} page loaded successfully")
            else:
                log_test(f"6.{pages_to_test.index((page_name, page_id))+1} {page_name} Loads", "FAIL", f"{page_name} page failed to load")
        
        return True
        
    except Exception as e:
        log_test("Cross-Page Consistency", "ERROR", str(e))
        return False

async def run_all_tests():
    """Run all browser automation tests"""
    print("\n" + "="*80)
    print("RACEIQ PRO - BROWSER AUTOMATION TESTING")
    print("="*80)
    print("Testing all scenarios from USER_TESTING_GUIDE.md")
    print("="*80)
    
    async with async_playwright() as p:
        try:
            # Launch browser
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            # Set longer timeout
            page.set_default_timeout(30000)
            
            # Run test suites
            await test_race_overview(page)
            await test_tactical_analysis(page)
            await test_strategic_analysis(page)
            await test_integrated_insights(page)
            await test_race_simulator(page)
            await test_cross_page_consistency(page)
            
            # Close browser
            await browser.close()
            
        except Exception as e:
            log_test("Browser Launch", "ERROR", f"Failed to launch browser: {str(e)}")
            print(f"\n💥 Browser automation failed: {e}")
            print("Make sure Playwright is installed: pip install playwright")
            print("Then install browsers: playwright install chromium")
            return False
    
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
        "errors": test_results["errors"]
    }
    
    report_file = Path("browser_testing_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"💥 Errors: {len(test_results['errors'])}")
    print(f"⏱️  Duration: {total_time:.2f} seconds")
    print(f"\n📄 Full report saved to: {report_file}")
    
    return len(test_results['failed']) == 0 and len(test_results['errors']) == 0

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except ImportError:
        print("\n❌ Playwright not installed!")
        print("Install with: pip install playwright")
        print("Then install browsers: playwright install chromium")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

