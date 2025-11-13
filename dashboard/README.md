# RaceIQ Pro Dashboard

Interactive Streamlit dashboard for Toyota GR Cup race analysis and driver coaching.

## Features

### 🏁 Race Overview
- Race summary statistics (drivers, laps, speeds)
- Final standings leaderboard
- Fastest lap distribution
- Completion status breakdown
- Section performance comparison
- Weather conditions

### 🎯 Tactical Analysis
- Driver-specific performance metrics
- Section-by-section heatmap analysis
- Driver vs Optimal Ghost comparison
- Anomaly detection (outlier lap identification)
- Telemetry visualization (simulated)
- Top 3 improvement recommendations

### ⚙️ Strategic Analysis
- Pit stop detection and timeline
- Tire degradation curves with trend analysis
- Optimal pit window recommendations
- Actual vs optimal strategy comparison
- Strategic insights and recommendations

### 🔗 Integrated Insights
- Combined tactical + strategic recommendations
- What-if scenario simulator
- Cross-module impact visualization
- Projected position change calculator
- Total potential performance gains

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify data structure:
```
ToyotaGR/
├── Data/
│   ├── barber/
│   ├── COTA/
│   ├── sonoma/
│   └── ...
└── dashboard/
    ├── app.py
    ├── pages/
    └── requirements.txt
```

## Running the Dashboard

From the `dashboard/` directory:

```bash
streamlit run app.py
```

Or from the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Usage

1. **Select Track**: Use the sidebar to choose a track (Barber, COTA, Sonoma, etc.)
2. **Select Race**: Choose Race 1 or Race 2
3. **Navigate Pages**: Use the sidebar navigation to switch between:
   - Race Overview
   - Tactical Analysis
   - Strategic Analysis
   - Integrated Insights
4. **Select Driver**: On analysis pages, select a driver number to view detailed insights

## Data Requirements

The dashboard expects CSV files in the following format:
- `03_*Results*.CSV` - Race results
- `23_*Sections*.CSV` - Section-by-section analysis
- `*lap_time*.csv` - Lap timing data
- `99_*Best*.CSV` - Best lap data
- `26_*Weather*.CSV` - Weather conditions

## Features Detail

### Tactical Analysis
- **Section Heatmap**: Visual representation of section times across all laps
- **Ghost Comparison**: Bar chart showing gap to optimal time in each section
- **Anomaly Detection**: Statistical detection of outlier laps using z-score method
- **Recommendations**: AI-generated coaching points based on performance data

### Strategic Analysis
- **Pit Stop Detection**: Automatic identification of pit stops based on lap time spikes
- **Tire Degradation**: Linear regression trend analysis of lap times
- **Optimal Window**: Calculated recommendation for pit stop timing (typically 1/3-2/3 of race)
- **Strategy Comparison**: Side-by-side comparison of actual vs recommended strategy

### Integrated Insights
- **Scenario Simulator**: Interactive sliders to simulate performance improvements
- **Impact Analysis**: Calculate total time savings from various improvements
- **Position Projector**: Estimate position changes based on lap time improvements

## Customization

### Adding Custom Visualizations
Edit the page files in `pages/` directory:
- `overview.py` - Race overview page
- `tactical.py` - Tactical analysis page
- `strategic.py` - Strategic analysis page
- `integrated.py` - Integrated insights page

### Styling
Custom CSS is defined in `app.py`. Modify the `st.markdown()` section to change colors and styles.

### Data Loading
The `load_race_data()` function in `app.py` handles data loading with caching. Modify this function to add new data sources.

## Technical Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy (linear regression, statistical tests)

## Performance

- Data is cached using `@st.cache_data` for fast page navigation
- Lazy loading of visualizations for improved responsiveness
- Optimized dataframe operations with Pandas

## Troubleshooting

### Data not loading
- Check that CSV files exist in the expected directory structure
- Verify file naming matches the glob patterns in `load_race_data()`
- Check file encoding (should be UTF-8 or Latin-1)

### Visualizations not appearing
- Ensure plotly is installed: `pip install plotly`
- Clear Streamlit cache: Click "Clear cache" in hamburger menu

### Performance issues
- Large CSV files may take time to load initially (they are cached after first load)
- Reduce the number of visualizations per page if needed
- Consider sampling data for very large datasets

## Contributing

To add new features:
1. Create a new module in `pages/` for new analysis types
2. Import and route in `app.py`
3. Follow existing patterns for error handling and visualization
4. Add new dependencies to `requirements.txt`

## License

Part of the Toyota GR Cup Hackathon Project

## Version

v1.0 - Initial Release
