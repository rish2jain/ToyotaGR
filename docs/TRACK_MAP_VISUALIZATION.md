# Track Map Visualization with Performance Heatmap Overlay

## Overview

The Track Map Visualization feature provides interactive, color-coded track maps that overlay driver performance data onto track layouts. This helps drivers and teams quickly identify performance gaps and areas for improvement.

## Features

### 1. Performance Heatmap Overlay
- **Color-coded sections**: Green (fast) → Yellow (good) → Orange (average) → Red (slow)
- **Interactive tooltips**: Hover over any section to see detailed metrics
- **Performance ratings**: Automatic classification of section performance
- **Gap analysis**: Visual representation of time lost per section

### 2. Driver Comparison
- **Side-by-side visualization**: Compare two drivers on the same track
- **Differential coloring**: Red (Driver 1 faster) vs Blue (Driver 2 faster)
- **Time differences**: Exact gaps displayed per section
- **Competitive analysis**: Identify where competitors are gaining/losing time

### 3. Multiple Track Support
- **Barber Motorsports Park**: 2.38 miles, 17 turns, Alabama
- **Circuit of the Americas (COTA)**: 3.41 miles, 20 turns, Austin, Texas
- **Sonoma Raceway**: 2.52 miles, 12 turns, Sonoma, California
- **Generic Layout**: Configurable for unknown tracks

## Files Created

### Core Implementation

1. **`src/utils/track_layouts.py`** (NEW)
   - Track coordinate definitions for multiple circuits
   - Barber Motorsports Park detailed layout (15 sections)
   - COTA, Sonoma, and generic layouts
   - Helper functions for track data retrieval

2. **`src/utils/visualization.py`** (UPDATED)
   - `create_track_map_with_performance()`: Single driver performance map
   - `create_driver_comparison_map()`: Two-driver comparison map
   - Performance-to-color mapping functions
   - Interactive Plotly-based visualizations

3. **`dashboard/pages/tactical.py`** (UPDATED)
   - Integrated track map into Tactical Analysis dashboard
   - Automatic track detection from race data
   - Driver comparison interface
   - Interactive tooltips and help text

### Examples and Documentation

4. **`examples/track_map_demo.py`** (NEW)
   - Complete demonstration script
   - Sample data generation
   - Multiple visualization examples
   - Performance analysis workflow

5. **`examples/track_map_driver42.html`** (GENERATED)
   - Example single-driver performance map
   - Interactive HTML file (4.7 MB)
   - Can be opened in any web browser

6. **`examples/track_map_comparison.html`** (GENERATED)
   - Example driver comparison map
   - Interactive HTML file (4.7 MB)
   - Can be opened in any web browser

## Usage

### 1. As Part of Dashboard

The track map is automatically integrated into the Tactical Analysis page:

```bash
# Run the dashboard
streamlit run dashboard/app.py
```

Navigate to: **Tactical Analysis** → **Track Map: Performance Heatmap**

Features:
- Select driver from dropdown
- View performance-colored track map
- Compare with other drivers
- Interactive hover tooltips

### 2. Programmatic Usage

```python
from src.utils.visualization import create_track_map_with_performance
import pandas as pd

# Prepare section data
section_data = pd.DataFrame({
    'Section': [1, 1, 2, 2, 3, 3],
    'Lap': [1, 2, 1, 2, 1, 2],
    'Time': [26.5, 26.2, 23.4, 23.1, 29.8, 29.3],
    'GapToOptimal': [0.3, 0.0, 0.3, 0.0, 0.5, 0.0]
})

# Create track map
fig = create_track_map_with_performance(
    section_data,
    track_name='barber',
    section_col='Section',
    time_col='Time',
    gap_col='GapToOptimal',
    driver_label='Car #42'
)

# Save to HTML
fig.write_html('my_track_map.html')

# Or display in notebook
fig.show()
```

### 3. Driver Comparison

```python
from src.utils.visualization import create_driver_comparison_map

# Prepare data for both drivers
driver1_data = pd.DataFrame({
    'Section': [1, 2, 3],
    'Time': [26.2, 23.1, 29.3]
})

driver2_data = pd.DataFrame({
    'Section': [1, 2, 3],
    'Time': [26.5, 22.9, 29.8]
})

# Create comparison map
fig = create_driver_comparison_map(
    driver1_data,
    driver2_data,
    track_name='barber',
    driver1_label='Car #42',
    driver2_label='Car #17',
    section_col='Section',
    time_col='Time'
)

fig.write_html('driver_comparison.html')
```

### 4. Run Demo

```bash
cd examples
python track_map_demo.py
```

This will:
1. Show available track layouts
2. Generate sample performance data
3. Create single-driver performance map
4. Create driver comparison map
5. Perform performance analysis
6. Save interactive HTML files

## Track Layouts

### Barber Motorsports Park

- **Location**: Leeds, Alabama
- **Length**: 2.38 miles
- **Turns**: 17
- **Direction**: Clockwise
- **Famous Sections**:
  - Turn 1: Museum Corner (uphill right-hander)
  - Turn 5: Charlotte's Web (downhill left sweeper)
  - Turns 15-17: Final complex

**Sections Defined**:
1. Start/Finish Straight
2. Turn 1 (Museum Corner)
3. Turns 2-3 (Uphill esses)
4. Turn 4
5. Charlotte's Web
6. Turns 6-7 (Chicane)
7. Turn 8
8. Turn 9
9. Turn 10
10. Back Straight
11. Turns 11-12
12. Turn 13
13. Turn 14
14. Turns 15-17 (Final complex)
15. Approach to S/F

### Circuit of the Americas (COTA)

- **Location**: Austin, Texas
- **Length**: 3.41 miles
- **Turns**: 20
- **Direction**: Counter-clockwise
- **Notable**: Turn 1 elevation change, technical infield

### Sonoma Raceway

- **Location**: Sonoma, California
- **Length**: 2.52 miles
- **Turns**: 12
- **Direction**: Clockwise
- **Notable**: "Corkscrew" section, elevation changes

## Performance Color Guide

| Color | Gap to Optimal | Rating | Recommendation |
|-------|---------------|---------|----------------|
| 🟢 Green | < 0.05s | Excellent | Maintain current approach |
| 🟡 Yellow | 0.05s - 0.15s | Good | Fine-tune for consistency |
| 🟠 Orange | 0.15s - 0.30s | Average | Review technique |
| 🔴 Red | > 0.30s | Needs Improvement | Focus area - review braking/racing line |

## Interactive Features

### Track Map Controls

- **Hover**: View detailed section metrics
  - Section name and type (straight/corner)
  - Time gap to optimal
  - Performance rating
  - Section description

- **Pan**: Click and drag to move around the map
- **Zoom**: Scroll wheel to zoom in/out
- **Reset**: Double-click to reset view

### Dashboard Integration

- **Driver Selection**: Dropdown to select any driver in the race
- **Driver Comparison**: Compare against any other driver
- **Track Detection**: Automatically detects track from race data
- **Real-time Updates**: Recalculates when different driver selected

## Technical Details

### Dependencies

```python
# Required
pandas
numpy
plotly

# Optional (for dashboard)
streamlit
```

### Data Format

Section performance data should include:

```python
{
    'Section': int or str,      # Section identifier
    'Lap': int,                 # Lap number
    'Time': float,              # Section time in seconds
    'GapToOptimal': float       # Gap to optimal time (optional)
}
```

If `GapToOptimal` is not provided, it will be calculated automatically from the minimum time per section.

### Track Coordinate System

- Normalized 0-100 coordinate space
- X-axis: Horizontal position
- Y-axis: Vertical position
- Coordinates are stylized representations, not GPS coordinates
- Start/Finish at the beginning of the coordinate list

## Customization

### Adding New Tracks

Edit `src/utils/track_layouts.py`:

```python
def create_my_track_layout() -> Dict[str, List[Tuple[float, float]]]:
    sections = []

    # Add section definitions
    sections.append({
        'name': 'Section 1: Start/Finish',
        'x': np.linspace(50, 75, 20).tolist(),
        'y': np.linspace(10, 10, 20).tolist(),
        'type': 'straight',
        'description': 'Main straight'
    })

    # ... more sections ...

    return {
        'sections': sections,
        'total_sections': len(sections),
        'track_info': {
            'name': 'My Track',
            'location': 'Location',
            'length': 'X.XX miles',
            'turns': XX,
            'direction': 'Clockwise'
        }
    }

# Register in TRACK_COORDINATES dictionary
TRACK_COORDINATES['mytrack'] = create_my_track_layout
```

### Customizing Colors

Modify color mapping in `src/utils/visualization.py`:

```python
def _map_performance_to_colors(section_gaps, colorscale='RdYlGn_r'):
    # Adjust thresholds
    if gap < 0.05:
        # Excellent - Green
    elif gap < 0.15:
        # Good - Yellow
    # ... etc
```

### Adjusting Performance Ratings

Modify thresholds in `src/utils/visualization.py`:

```python
def _get_performance_rating(gap, optimal=0.0):
    if gap < 0.05:
        return "Excellent"
    elif gap < 0.15:
        return "Good"
    elif gap < 0.30:
        return "Average"
    else:
        return "Needs Improvement"
```

## Integration with RaceIQ Pro

The track map visualization is a core component of the **Tactical Analysis** module:

1. **Strategic Planning**: Identify which sections need focus in practice
2. **Race Analysis**: Review where time was lost during the race
3. **Driver Coaching**: Visual tool for explaining performance gaps
4. **Competitor Analysis**: See where competitors are faster/slower
5. **Setup Optimization**: Correlate setup changes with section performance

## Future Enhancements

Potential additions:
- [ ] GPS-based track generation from telemetry data
- [ ] 3D elevation profiles
- [ ] Animated racing line visualization
- [ ] Sector time predictions
- [ ] Weather overlay (wet/dry sections)
- [ ] Tire degradation visualization
- [ ] Real-time lap comparison during live sessions

## Troubleshooting

### "Plotly not available" error

```bash
pip install plotly
```

### Track not found

- Check track name is in: 'barber', 'cota', 'sonoma', or 'generic'
- Track names are case-insensitive
- Falls back to generic layout if not found

### No data displayed

- Ensure section data has required columns: `Section`, `Time`
- Check for NaN values in time data
- Verify section numbers are valid (positive integers)

### Map looks distorted

- Track layouts are stylized representations
- Adjust `scaleanchor` and `scaleratio` in layout settings
- Some browser zoom levels may affect rendering

## Credits

**RaceIQ Pro - Track Map Visualization**
- Developed for Toyota GR Cup Hackathon
- Interactive performance analysis tool
- Based on Barber Motorsports Park race data
- Powered by Plotly interactive graphics

## Contact & Support

For questions or improvements:
- See main project README
- Check examples directory for usage patterns
- Review source code comments for detailed explanations
