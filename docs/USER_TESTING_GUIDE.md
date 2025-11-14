# RaceIQ Pro - User Testing Guide

Comprehensive guide for testing RaceIQ Pro features and functionality.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup Testing](#initial-setup-testing)
3. [Dashboard Testing](#dashboard-testing)
4. [Feature-Specific Testing](#feature-specific-testing)
5. [Integration Testing](#integration-testing)
6. [Performance Testing](#performance-testing)
7. [Error Handling Testing](#error-handling-testing)
8. [Test Scenarios](#test-scenarios)
9. [Reporting Issues](#reporting-issues)

---

## Prerequisites

### System Requirements
- Python 3.9 or higher
- Virtual environment (recommended)
- All dependencies installed (`pip install -r requirements.txt`)
- Race data files in `Data/` directory

### Pre-Testing Checklist
- [ ] Virtual environment activated
- [ ] Dependencies installed successfully
- [ ] `python verify_structure.py` passes
- [ ] Dashboard starts without errors
- [ ] Sample data files accessible

---

## Initial Setup Testing

### Test 1: Verify Installation

**Steps:**
1. Run verification script:
   ```bash
   python verify_structure.py
   ```

**Expected Results:**
- ✅ All directories exist
- ✅ All key files present
- ✅ Sample data files found
- ✅ Module imports successful
- ⚠️ `data/processed` directory may be missing (optional)

**Pass Criteria:** All checks pass except optional directories

---

### Test 2: Launch Dashboard

**Steps:**
1. Start dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

**Expected Results:**
- Dashboard opens in browser at `http://localhost:8501`
- Sidebar shows navigation options
- No error messages in terminal or browser console
- Page loads within 5 seconds

**Pass Criteria:** Dashboard loads successfully

---

### Test 3: Data Loading

**Steps:**
1. Select a track from sidebar (e.g., "Barber")
2. Select a race (Race 1 or Race 2)
3. Wait for data to load

**Expected Results:**
- Loading spinner appears briefly
- Data loads successfully
- No error messages
- Dashboard displays content

**Pass Criteria:** Data loads without errors

---

## Dashboard Testing

### Page 1: Race Overview

#### Test 1.1: Metrics Display

**Steps:**
1. Navigate to "🏁 Race Overview"
2. Observe the 4 key metrics at the top

**Expected Results:**
- **Total Drivers:** Shows number > 0
- **Total Laps:** Shows number > 0
- **Top Speed:** Shows speed in km/h or mph
- **Fastest Lap:** Shows time in MM:SS.mmm format

**Pass Criteria:** All metrics display correctly with valid values

---

#### Test 1.2: Leaderboard Table

**Steps:**
1. Scroll to leaderboard section
2. Check table displays
3. Try sorting by different columns

**Expected Results:**
- Table shows all drivers
- Columns: Position, Driver Number, Best Lap, Average Lap, etc.
- Sorting works for all sortable columns
- Data is accurate and formatted correctly

**Pass Criteria:** Leaderboard displays and sorts correctly

---

#### Test 1.3: Fastest Lap Chart

**Steps:**
1. Locate "Fastest Lap Times" chart
2. Hover over bars to see tooltips
3. Verify color coding

**Expected Results:**
- Bar chart displays
- Bars are color-coded (green = fast, red = slow)
- Tooltips show exact values on hover
- Chart is interactive (zoom, pan)

**Pass Criteria:** Chart displays and is interactive

---

#### Test 1.4: Completion Status Pie Chart

**Steps:**
1. Locate "Race Completion Status" chart
2. Check pie chart displays
3. Verify percentages add up to 100%

**Expected Results:**
- Pie chart displays
- Segments show completion status (Finished, DNF, etc.)
- Percentages are accurate
- Chart is interactive

**Pass Criteria:** Pie chart displays correctly

---

#### Test 1.5: Section Performance Comparison

**Steps:**
1. Scroll to "Section Performance Comparison"
2. Check chart for top 5 drivers
3. Verify section times display

**Expected Results:**
- Chart shows top 5 drivers
- Multiple sections displayed
- Colors differentiate drivers
- Tooltips show exact values

**Pass Criteria:** Section comparison chart displays correctly

---

#### Test 1.6: Weather Widget

**Steps:**
1. Locate weather information section
2. Check weather data displays

**Expected Results:**
- Temperature displayed
- Track temperature shown
- Humidity percentage displayed
- Wind speed (if available)
- Weather conditions icon/description

**Pass Criteria:** Weather information displays correctly

---

### Page 2: Tactical Analysis

#### Test 2.1: Driver Selection

**Steps:**
1. Navigate to "🎯 Tactical Analysis"
2. Select a driver from dropdown
3. Verify driver-specific data loads

**Expected Results:**
- Driver dropdown populates with available drivers
- Selecting driver updates all visualizations
- Loading indicator appears during update
- No errors occur

**Pass Criteria:** Driver selection works correctly

---

#### Test 2.2: Performance Metrics

**Steps:**
1. Select a driver
2. Check performance metrics section

**Expected Results:**
- Best lap time displayed
- Average lap time displayed
- Consistency score shown
- Gap to leader calculated
- All metrics are accurate

**Pass Criteria:** Performance metrics display correctly

---

#### Test 2.3: Section Heatmap

**Steps:**
1. Locate "Section-by-Section Performance" heatmap
2. Check heatmap displays
3. Hover over cells to see values

**Expected Results:**
- Heatmap shows sections (rows) vs laps (columns)
- Color intensity represents performance
- Tooltips show exact values
- Legend explains color scale
- Chart is interactive

**Pass Criteria:** Heatmap displays and is interactive

---

#### Test 2.4: Optimal Ghost Comparison

**Steps:**
1. Locate "Driver vs Optimal Ghost" section
2. Check bar chart displays
3. Verify gap calculations

**Expected Results:**
- Bar chart shows gap to optimal for each section
- Positive values = slower than optimal
- Negative values = faster than optimal
- Colors indicate performance (green = good, red = needs improvement)
- Tooltips show exact gaps

**Pass Criteria:** Ghost comparison chart displays correctly

---

#### Test 2.5: Anomaly Detection

**Steps:**
1. Scroll to "Anomaly Detection" section
2. Check anomaly detection tabs:
   - Statistical Detection
   - ML Detection with SHAP
3. Review detected anomalies

**Expected Results:**
- Anomalies are detected and listed
- Anomaly scores displayed
- SHAP explanations available (if ML tab selected)
- Feature contributions shown
- Confidence scores displayed

**Pass Criteria:** Anomaly detection works correctly

---

#### Test 2.6: Track Map Visualization

**Steps:**
1. Locate "Track Map: Performance Heatmap" section
2. Check track map displays
3. Verify color coding by section

**Expected Results:**
- Track map displays (if track layout available)
- Sections color-coded by performance
- Tooltips show section details
- Map is interactive (zoom, pan)

**Pass Criteria:** Track map displays correctly

---

#### Test 2.7: Coaching Recommendations

**Steps:**
1. Scroll to "Top 3 Improvement Recommendations"
2. Review recommendations displayed

**Expected Results:**
- 3 recommendations shown
- Each recommendation includes:
  - Priority level (High/Medium/Low)
  - Description
  - Expected time gain
  - Confidence score
- Recommendations are actionable

**Pass Criteria:** Recommendations display correctly

---

### Page 3: Strategic Analysis

#### Test 3.1: Pit Stop Detection

**Steps:**
1. Navigate to "⚙️ Strategic Analysis"
2. Select a driver
3. Check "Pit Stop Detection" section

**Expected Results:**
- Pit stops detected automatically
- Pit stop timeline displayed
- Pit stop laps identified correctly
- Pit stop duration calculated

**Pass Criteria:** Pit stops detected accurately

---

#### Test 3.2: Tire Degradation Curve

**Steps:**
1. Locate "Tire Degradation Analysis" section
2. Check degradation curve displays
3. Verify trend line

**Expected Results:**
- Scatter plot shows lap times vs lap number
- Trend line shows degradation
- Degradation rate calculated
- R² value displayed (goodness of fit)
- Chart is interactive

**Pass Criteria:** Degradation curve displays correctly

---

#### Test 3.3: Optimal Pit Window (Bayesian)

**Steps:**
1. Scroll to "Optimal Pit Window Analysis with Bayesian Uncertainty"
2. Check Bayesian analysis displays
3. Adjust confidence level slider

**Expected Results:**
- Optimal pit lap displayed
- Uncertainty percentage shown
- Risk level indicator (🟢🟡🟠🔴)
- Confidence intervals displayed (80%, 90%, 95%)
- Slider adjusts intervals dynamically
- Posterior distribution visualizations:
  - Violin plot
  - PDF curve
  - Simulation results

**Pass Criteria:** Bayesian analysis displays correctly

---

#### Test 3.4: Strategy Comparison

**Steps:**
1. Locate "Strategy Comparison" section
2. Check actual vs optimal comparison

**Expected Results:**
- Actual pit strategy displayed
- Optimal pit strategy shown
- Time difference calculated
- Visual comparison chart
- Strategic insights provided

**Pass Criteria:** Strategy comparison displays correctly

---

#### Test 3.5: Strategic Recommendations

**Steps:**
1. Scroll to "Strategic Recommendations"
2. Review recommendations

**Expected Results:**
- Recommendations displayed
- Each includes:
  - Priority
  - Description
  - Expected impact
  - Confidence level
- Recommendations are strategic (not tactical)

**Pass Criteria:** Strategic recommendations display correctly

---

### Page 4: Integrated Insights

#### Test 4.1: Combined Recommendations

**Steps:**
1. Navigate to "🔗 Integrated Insights"
2. Select a driver
3. Check "Combined Recommendations" section

**Expected Results:**
- Tactical and strategic recommendations combined
- Prioritized list displayed
- Conflicts resolved (if any)
- Total potential gain calculated

**Pass Criteria:** Combined recommendations display correctly

---

#### Test 4.2: What-If Scenario Simulator

**Steps:**
1. Locate "What-If Scenario Simulator"
2. Adjust sliders for different improvements:
   - Section 1 improvement
   - Section 2 improvement
   - Section 3 improvement
   - Pit strategy adjustment
3. Check projected results update

**Expected Results:**
- Sliders are interactive
- Results update in real-time
- Projected lap time improvement calculated
- Projected position change shown
- Total time gain displayed

**Pass Criteria:** Simulator works correctly

---

#### Test 4.3: Causal Analysis Tab

**Steps:**
1. Click on "Causal Analysis" tab (if available)
2. Check causal analysis displays

**Expected Results:**
- Causal graph displayed (if implemented)
- Effect sizes shown
- Confidence intervals displayed
- Robustness scores shown
- Counterfactual scenarios available

**Pass Criteria:** Causal analysis displays correctly (if implemented)

---

#### Test 4.4: Impact Visualization

**Steps:**
1. Locate "Impact Analysis" section
2. Check visualization displays

**Expected Results:**
- Chart shows impact of improvements
- Time savings visualized
- Position changes projected
- Multiple scenarios compared

**Pass Criteria:** Impact visualization displays correctly

---

### Page 5: Race Simulator

#### Test 5.1: Race Animation

**Steps:**
1. Navigate to "🏎️ Race Simulator"
2. Configure 2-5 drivers
3. Set pit strategies
4. Run race simulation

**Expected Results:**
- Driver configuration interface works
- Pit strategies can be set
- Simulation runs successfully
- Position changes animated
- Final results displayed

**Pass Criteria:** Race simulation works correctly

---

#### Test 5.2: Undercut Analyzer

**Steps:**
1. Click on "Undercut Analyzer" tab
2. Configure two drivers
3. Set different pit laps
4. Analyze undercut scenario

**Expected Results:**
- Undercut success/failure determined
- Gap evolution chart displayed
- Overtake lap identified
- Success probability calculated

**Pass Criteria:** Undercut analyzer works correctly

---

#### Test 5.3: Strategy Optimizer

**Steps:**
1. Click on "Strategy Optimizer" tab
2. Configure team vs opponents
3. Select optimization objective
4. Run optimization

**Expected Results:**
- Optimal strategies calculated
- Expected results displayed
- Team score calculated
- Recommendations provided

**Pass Criteria:** Strategy optimizer works correctly

---

## Feature-Specific Testing

### Anomaly Detection Testing

#### Test: Statistical Anomaly Detection

**Steps:**
1. Go to Tactical Analysis
2. Select a driver
3. Click "Statistical Detection" tab
4. Review anomalies

**Expected Results:**
- Anomalies detected using z-score method
- Anomaly scores displayed
- Lap numbers identified
- Threshold clearly shown

**Pass Criteria:** Statistical detection works

---

#### Test: ML Anomaly Detection with SHAP

**Steps:**
1. Click "ML Detection with SHAP" tab
2. Wait for analysis to complete
3. Review SHAP explanations

**Expected Results:**
- Isolation Forest model runs
- Anomalies detected
- SHAP explanations available
- Feature contributions shown
- Confidence scores displayed

**Pass Criteria:** ML detection with SHAP works

---

#### Test: LSTM Anomaly Detection

**Steps:**
1. Click "Deep Learning (LSTM)" tab (if available)
2. Configure parameters:
   - Sequence length: 50
   - Epochs: 30
   - Contamination: 5%
3. Run LSTM detection

**Expected Results:**
- LSTM model trains (30-90 seconds)
- Anomalies detected
- Reconstruction errors displayed
- Temporal patterns identified

**Pass Criteria:** LSTM detection works (if TensorFlow installed)

---

### Bayesian Strategy Testing

#### Test: Confidence Interval Adjustment

**Steps:**
1. Go to Strategic Analysis
2. Select a driver
3. Adjust confidence level slider (80%, 90%, 95%)
4. Observe interval changes

**Expected Results:**
- Intervals update dynamically
- Width increases with confidence level
- All three intervals displayed in table
- Visualizations update

**Pass Criteria:** Confidence intervals adjust correctly

---

#### Test: Risk Assessment

**Steps:**
1. Check risk level indicator
2. Review risk assessment panel
3. Verify risk level matches uncertainty

**Expected Results:**
- Risk level displayed (🟢🟡🟠🔴)
- Explanation provided
- Strategy note included
- Risk level matches posterior std

**Pass Criteria:** Risk assessment displays correctly

---

### Racing Line Testing

#### Test: Racing Line Reconstruction

**Steps:**
1. Go to Tactical Analysis
2. Select a driver with telemetry data
3. Locate racing line section
4. Review reconstructed line

**Expected Results:**
- Racing line reconstructed from telemetry
- Corner identification works
- Apex points identified
- Brake/throttle points shown

**Pass Criteria:** Racing line reconstruction works (if telemetry available)

---

#### Test: Driver Comparison

**Steps:**
1. Select two drivers
2. Compare racing lines
3. Review differences

**Expected Results:**
- Two racing lines overlaid
- Differences highlighted
- Corner-by-corner comparison
- Speed traces compared

**Pass Criteria:** Driver comparison works

---

## Integration Testing

### Test: Cross-Page Data Consistency

**Steps:**
1. Select a driver on Tactical Analysis
2. Note their best lap time
3. Navigate to Strategic Analysis
4. Verify same driver's data matches
5. Check Integrated Insights
6. Verify consistency

**Expected Results:**
- Data consistent across pages
- Same driver number works everywhere
- Metrics match between pages
- No data loss between navigation

**Pass Criteria:** Data consistent across all pages

---

### Test: Track Switching

**Steps:**
1. Select "Barber" track
2. Note data displayed
3. Switch to "COTA" track
4. Verify data updates
5. Switch back to "Barber"
6. Verify data reloads

**Expected Results:**
- Track switching works smoothly
- Data updates correctly
- No errors occur
- Previous selections remembered

**Pass Criteria:** Track switching works correctly

---

### Test: Race Number Switching

**Steps:**
1. Select Race 1
2. Note data displayed
3. Switch to Race 2
4. Verify data updates
5. Switch back to Race 1

**Expected Results:**
- Race switching works
- Data updates correctly
- No errors occur

**Pass Criteria:** Race switching works correctly

---

## Performance Testing

### Test: Page Load Times

**Steps:**
1. Measure time to load each page
2. Note any slow pages

**Expected Results:**
- Overview page: < 3 seconds
- Tactical Analysis: < 5 seconds
- Strategic Analysis: < 5 seconds
- Integrated Insights: < 5 seconds
- Race Simulator: < 3 seconds

**Pass Criteria:** All pages load within acceptable time

---

### Test: Data Loading Performance

**Steps:**
1. Load large dataset
2. Measure loading time
3. Check memory usage

**Expected Results:**
- Large datasets load in < 10 seconds
- Memory usage reasonable
- No browser freezing
- Loading indicators shown

**Pass Criteria:** Performance acceptable for large datasets

---

### Test: Visualization Rendering

**Steps:**
1. Generate multiple visualizations
2. Check rendering performance
3. Verify interactivity maintained

**Expected Results:**
- Charts render quickly
- Interactions responsive
- No lag when hovering
- Zoom/pan works smoothly

**Pass Criteria:** Visualizations render and interact smoothly

---

## Error Handling Testing

### Test: Missing Data Files

**Steps:**
1. Temporarily rename a data file
2. Try to load that track/race
3. Check error handling

**Expected Results:**
- Error message displayed
- Error is user-friendly
- Dashboard doesn't crash
- Suggestion to check data files

**Pass Criteria:** Graceful error handling

---

### Test: Invalid Driver Selection

**Steps:**
1. Try to select non-existent driver
2. Check error handling

**Expected Results:**
- Error prevented (dropdown only shows valid drivers)
- Or graceful error message
- Dashboard remains functional

**Pass Criteria:** Invalid selections handled gracefully

---

### Test: Network Interruption

**Steps:**
1. Start dashboard
2. Simulate network interruption (if applicable)
3. Check recovery

**Expected Results:**
- Error message displayed
- Retry option available
- Dashboard remains functional

**Pass Criteria:** Network issues handled gracefully

---

## Test Scenarios

### Scenario 1: New User Workflow

**Objective:** Test complete workflow for a new user

**Steps:**
1. Launch dashboard
2. Select track and race
3. View Race Overview
4. Select a driver
5. Review Tactical Analysis
6. Check Strategic Analysis
7. Explore Integrated Insights
8. Try Race Simulator

**Expected Results:**
- All steps work smoothly
- No confusion about navigation
- Data displays correctly
- Features are discoverable

**Pass Criteria:** Complete workflow successful

---

### Scenario 2: Driver Coaching Session

**Objective:** Simulate a coaching session

**Steps:**
1. Select a driver
2. Review performance metrics
3. Check section heatmap
4. Review anomaly detection
5. Read coaching recommendations
6. Check racing line (if available)
7. Review strategic recommendations

**Expected Results:**
- All coaching features accessible
- Recommendations actionable
- Visualizations helpful
- Data supports recommendations

**Pass Criteria:** Coaching session successful

---

### Scenario 3: Strategy Planning Session

**Objective:** Test strategy planning workflow

**Steps:**
1. Select a driver
2. Review tire degradation
3. Check optimal pit window
4. Adjust confidence level
5. Review risk assessment
6. Compare actual vs optimal strategy
7. Use what-if simulator
8. Run race simulation

**Expected Results:**
- Strategy tools work correctly
- Recommendations clear
- Simulations accurate
- Decision support provided

**Pass Criteria:** Strategy planning successful

---

### Scenario 4: Multi-Driver Comparison

**Objective:** Compare multiple drivers

**Steps:**
1. Select Driver A
2. Note key metrics
3. Switch to Driver B
4. Compare metrics
5. Use racing line comparison (if available)
6. Compare strategies

**Expected Results:**
- Easy to switch drivers
- Comparisons possible
- Data consistent
- Visualizations helpful

**Pass Criteria:** Multi-driver comparison successful

---

## Reporting Issues

### Issue Report Template

When reporting issues, include:

1. **Environment:**
   - Python version
   - Operating system
   - Browser (if dashboard)
   - Virtual environment (yes/no)

2. **Steps to Reproduce:**
   - Detailed steps
   - Expected vs actual behavior

3. **Screenshots:**
   - Error messages
   - Unexpected behavior
   - Console errors (if applicable)

4. **Data:**
   - Track/race being tested
   - Driver number (if applicable)
   - Sample data files (if relevant)

5. **Error Messages:**
   - Full error traceback
   - Console errors
   - Terminal output

### Common Issues and Solutions

#### Issue: Dashboard won't start
**Solution:**
- Check virtual environment activated
- Verify dependencies installed
- Check port 8501 not in use
- Review terminal error messages

#### Issue: Data won't load
**Solution:**
- Verify data files exist
- Check file naming conventions
- Verify file paths correct
- Check file permissions

#### Issue: Visualizations not displaying
**Solution:**
- Check browser console for errors
- Verify Plotly installed
- Try refreshing page
- Check data format

#### Issue: Anomaly detection slow
**Solution:**
- Normal for LSTM (30-90 seconds)
- Reduce contamination parameter
- Use statistical detection for speed
- Check TensorFlow installed (for LSTM)

#### Issue: Bayesian analysis not working
**Solution:**
- Verify scipy installed
- Check data has enough laps
- Verify tire degradation model fitted
- Review error messages

---

## Testing Checklist

### Pre-Release Testing

- [ ] All dashboard pages load
- [ ] All features functional
- [ ] Data loads correctly
- [ ] Visualizations display
- [ ] No console errors
- [ ] Performance acceptable
- [ ] Error handling works
- [ ] Documentation accurate
- [ ] Examples run successfully
- [ ] Cross-browser compatibility (if applicable)

### Feature-Specific Checklist

- [ ] Race Overview metrics accurate
- [ ] Tactical Analysis recommendations helpful
- [ ] Strategic Analysis pit window correct
- [ ] Bayesian uncertainty quantification works
- [ ] Anomaly detection accurate
- [ ] SHAP explanations clear
- [ ] Racing line reconstruction accurate
- [ ] Race simulation realistic
- [ ] What-if simulator functional
- [ ] Integrated insights combine correctly

---

## Success Criteria

### Critical (Must Pass)
- ✅ Dashboard launches successfully
- ✅ Data loads without errors
- ✅ All pages accessible
- ✅ Core features functional
- ✅ No crashes or freezes

### Important (Should Pass)
- ✅ Visualizations display correctly
- ✅ Interactive features work
- ✅ Performance acceptable
- ✅ Error handling graceful
- ✅ Recommendations accurate

### Nice to Have (Optional)
- ⭐ Advanced features work (LSTM, causal analysis)
- ⭐ All visualizations interactive
- ⭐ Performance optimized
- ⭐ Edge cases handled
- ⭐ Documentation complete

---

## Quick Test Script

Run this quick test to verify basic functionality:

```bash
# 1. Verify installation
python verify_structure.py

# 2. Test imports
python -c "from src.pipeline.data_loader import DataLoader; print('✓ Data loader OK')"
python -c "from src.tactical.anomaly_detector import AnomalyDetector; print('✓ Anomaly detector OK')"
python -c "from src.strategic.strategy_optimizer import PitStrategyOptimizer; print('✓ Strategy optimizer OK')"

# 3. Launch dashboard
streamlit run dashboard/app.py
```

**Expected:** All commands succeed, dashboard launches

---

**Version:** 1.0  
**Last Updated:** 2024  
**For:** RaceIQ Pro v1.0

