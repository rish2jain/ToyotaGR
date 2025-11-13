# RaceIQ Pro Dashboard - Quick Start Guide

## 🚀 Getting Started in 3 Minutes

### Step 1: Install Dependencies (30 seconds)

```bash
cd dashboard
pip install -r requirements.txt
```

### Step 2: Launch Dashboard (10 seconds)

**Option A - Using the run script:**
```bash
./run.sh
```

**Option B - Direct command:**
```bash
streamlit run app.py
```

### Step 3: Navigate the Dashboard (2 minutes)

1. **Browser Opens Automatically**
   - Dashboard will open at `http://localhost:8501`
   - If not, manually navigate to that URL

2. **Select Your Race**
   - Left sidebar: Choose track (e.g., "Barber")
   - Left sidebar: Choose race number (1 or 2)

3. **Explore the Pages**

   **🏁 Race Overview** (Start here!)
   - See overall race statistics
   - View final standings
   - Check weather conditions
   - Understand the competitive landscape

   **🎯 Tactical Analysis** (Driver coaching)
   - Select a driver from dropdown
   - View section-by-section performance
   - See anomaly detection (unusual laps)
   - Get top 3 improvement recommendations

   **⚙️ Strategic Analysis** (Race strategy)
   - Select a driver from dropdown
   - See detected pit stops
   - Analyze tire degradation
   - Review optimal pit windows
   - Compare actual vs optimal strategy

   **🔗 Integrated Insights** (Combined intelligence)
   - Select a driver from dropdown
   - View combined recommendations table
   - Use what-if simulator (adjust sliders!)
   - See cross-module impact analysis
   - Estimate position changes

## 🎯 Example Workflow

### For a Driver/Team:

1. Start with **Race Overview** to understand the race
2. Go to **Tactical Analysis**, select your car number
3. Identify weakest sections from the heatmap
4. Check anomalies to find problem laps
5. Review top 3 recommendations
6. Go to **Strategic Analysis** to review pit strategy
7. Check tire degradation trends
8. Compare your strategy to optimal window
9. Go to **Integrated Insights** to simulate improvements
10. Use sliders to see impact of different improvements
11. Check projected position changes

### For a Data Analyst:

1. **Race Overview** - Export leaderboard data
2. **Tactical Analysis** - Compare multiple drivers (select different drivers)
3. **Strategic Analysis** - Analyze pit timing effectiveness
4. **Integrated Insights** - Calculate total improvement potential

## 💡 Pro Tips

- **Sidebar is Always Visible**: Use it to switch tracks, races, and pages
- **Data is Cached**: First load is slow, subsequent loads are fast
- **Interactive Charts**: Hover over plots for detailed information
- **Zoom & Pan**: Click and drag on charts to zoom in
- **Download Charts**: Hover over chart, click camera icon to save as PNG
- **Fullscreen Mode**: Click expand icon on charts for larger view

## 🔍 What to Look For

### In Tactical Analysis:
- **Section Heatmap**: Red = slow, Green = fast
- **Ghost Comparison**: Taller bars = more time to gain
- **Anomalies**: Red X markers show problematic laps
- **Recommendations**: Focus on "High Priority" items first

### In Strategic Analysis:
- **Pit Timeline**: Red diamonds show pit stops
- **Degradation Curve**: Upward slope = tire wear
- **Optimal Window**: Green shaded area
- **Strategy Table**: ✓ Good vs ✗ Suboptimal

### In Integrated Insights:
- **Recommendations Table**: Sorted by priority
- **What-If Charts**: Green line = simulated improvement
- **Impact Analysis**: Higher bars = bigger opportunity
- **Position Projector**: Shows path to better finishing position

## 🛠️ Troubleshooting

**Dashboard won't start?**
```bash
pip install --upgrade streamlit plotly pandas numpy scipy
```

**No data showing?**
- Check that you're in the correct directory
- Verify Data/ folder exists at same level as dashboard/
- Try a different track or race number

**Charts not displaying?**
- Refresh the page (F5)
- Clear cache: Click hamburger menu → "Clear cache"

**Slow performance?**
- Normal on first load (data caching)
- Should be fast after initial load
- Close other browser tabs if needed

## 📊 Understanding the Metrics

- **Lap Time**: Total time to complete one lap
- **Section Time**: Time for specific track section (S1, S2, S3)
- **Gap**: Time difference to leader/optimal
- **Z-Score**: Statistical measure of outliers (>2 = anomaly)
- **Degradation Rate**: How much slower per lap (tire wear)
- **Stint**: Period between pit stops

## 🎓 Next Steps

1. Read the full [README.md](README.md) for detailed features
2. Explore different tracks and races
3. Compare multiple drivers
4. Experiment with the what-if simulator
5. Export insights for team meetings

## 📞 Need Help?

- Check the [README.md](README.md) for detailed documentation
- Review error messages in the dashboard
- Verify data files are in correct format

---

**Happy Racing! 🏁**

*RaceIQ Pro - Intelligent Racing Analytics for Toyota GR Cup*
