# Racing Line Reconstruction

## Overview

The Racing Line Reconstruction feature analyzes telemetry data (speed, brake, throttle) to reconstruct the approximate racing line taken by drivers through corners and straights. This enables comparison between drivers, identification of different racing approaches, and visualization of where time is gained or lost.

## How It Works

### 1. Corner Identification

The system identifies corners from telemetry by detecting:
- **Speed local minima**: Points where speed is locally minimum (potential apexes)
- **Brake application**: Significant brake pressure indicating corner entry
- **Sustained low speed**: Extended periods of reduced speed indicating corner duration

**Algorithm:**
```python
# Smooth speed data to reduce noise
speed_smooth = savgol_filter(speed, window_length=11, polyorder=3)

# Find speed local minima using peak detection
minima_indices = find_peaks(-speed_smooth, distance=10, prominence=5)

# For each minimum, find entry (backward) and exit (forward)
```

### 2. Corner Geometry Estimation

Once corners are identified, the system estimates their geometry using physics formulas.

#### Corner Radius Calculation

The cornerradius is estimated from minimum speed using the lateral acceleration formula:

**Formula:**
```
r = v² / (g × lateral_g)
```

Where:
- `r` = corner radius (meters)
- `v` = minimum speed through corner (m/s)
- `g` = gravitational acceleration (9.81 m/s²)
- `lateral_g` = assumed lateral G-force (typically 1.5-2.5g for racing)

**Example:**
- Corner minimum speed: 120 km/h (33.3 m/s)
- Lateral G assumption: 1.8g
- Calculated radius: r = (33.3)² / (9.81 × 1.8) = **62.6 meters**

#### Brake Point Detection

The brake point is identified by:
1. Finding maximum brake pressure in the corner region
2. Tracing backward to find where brake threshold (20% of max) is first crossed
3. Recording the distance at this point as the brake entry

#### Apex Location

The apex is defined as the point of minimum speed within the corner, which typically corresponds to:
- Tightest part of the corner
- Maximum lateral load
- Minimum corner radius

#### Throttle Application Point

The throttle point (corner exit) is identified by:
1. Starting from the apex, looking forward
2. Finding where throttle exceeds 30% (indicating exit acceleration)
3. Recording this as the corner exit point

### 3. Racing Line Trajectory

The complete racing line is built by:

1. **Straight sections**: Assumed to follow track centerline (lateral offset = 0)
2. **Corner sections**: Interpolated smooth curves with estimated lateral position
3. **Lateral offset estimation**: Using sine wave approximation for typical racing line
   - Entry: Outside of track (maximum offset)
   - Apex: Inside of track (minimum offset)
   - Exit: Outside of track (maximum offset)

**Lateral offset formula:**
```python
progress = (distance - corner_entry) / (corner_exit - corner_entry)
lateral_offset = sin(progress × π) × (track_width / 2)
```

This creates a smooth curve from outside → inside (apex) → outside.

## Physics Formulas Reference

### 1. Lateral Acceleration

```
a_lat = v² / r
```

- `a_lat` = lateral acceleration (m/s²)
- `v` = speed (m/s)
- `r` = corner radius (m)

### 2. Lateral G-Force

```
lateral_g = a_lat / g
```

- Typical racing values: 1.5g (slow corners) to 3.5g (high-speed corners)
- Used assumption: **1.8g** (conservative average for road racing)

### 3. Corner Speed Limit

```
v_max = √(r × g × μ × lateral_g_max)
```

- `μ` = tire coefficient of friction (≈1.0-1.4 for racing slicks)
- `lateral_g_max` = maximum lateral G the car can sustain

### 4. Brake Distance Approximation

```
d_brake = v² / (2 × a_brake)
```

- `d_brake` = braking distance
- `a_brake` = brake deceleration (typically 1.0-1.5g for racing)

## Data Requirements

### Minimum Required Data

- **Speed**: GPS speed or wheel speed (km/h or mph)
- **Distance or Time**: To sequence the data points

### Recommended Data

- **Brake Pressure**: For accurate brake point detection
- **Throttle Position**: For accurate exit point detection
- **Gear**: For corner classification (low gear = tight corner)

### Optional Data

- **Steering Angle**: For improved corner detection
- **GPS Coordinates**: For true track mapping (not estimated)
- **Lateral/Longitudinal G**: For validation of calculations

## Assumptions and Limitations

### Assumptions

1. **Lateral G**: Assumed constant at 1.8g
   - Real racing varies from 1.5g (slow corners) to 2.5g+ (fast corners)
   - Can be adjusted in constructor: `RacingLineReconstructor(lateral_g_assumption=2.0)`

2. **Track Width**: Assumed 12 meters
   - Typical road course width: 10-15m
   - Can be adjusted: `RacingLineReconstructor(track_width_m=15.0)`

3. **Lateral Offset Pattern**: Simplified sine wave
   - Real racing lines vary based on:
     - Corner type (increasing/decreasing radius)
     - Next corner location (double apex, chicanes)
     - Driver preference

4. **Track Centerline**: Straights assumed on centerline
   - Real drivers may position for upcoming corners

### Limitations

1. **Without GPS Coordinates**:
   - Cannot show absolute track position
   - Uses percentage distance (0-100%)
   - Track maps are approximations based on known layouts

2. **Corner Detection**:
   - May miss very fast, subtle corners
   - May split chicanes into separate corners
   - Relies on speed variation (constant-speed corners difficult)

3. **Radius Calculation**:
   - Assumes uniform circular arc
   - Real corners may have varying radius
   - Elevation changes not accounted for

4. **Comparison Accuracy**:
   - Assumes both drivers follow similar track (e.g., same lap)
   - Different lap counts may affect corner matching
   - Weather/track conditions not factored

## Interpreting Results

### Corner-by-Corner Comparison

When comparing two drivers:

#### Entry Delta
- **Positive**: Driver 1 brakes later (more aggressive or more confident)
- **Negative**: Driver 1 brakes earlier (more cautious or different line)

#### Apex Speed Delta
- **Positive**: Driver 1 carries more speed (faster line or better exit preparation)
- **Negative**: Driver 1 slower at apex (may be setting up for better exit)

#### Exit Delta
- **Positive**: Driver 1 accelerates later or carries less speed
- **Negative**: Driver 1 accelerates earlier or carries more speed

#### Radius Delta
- **Positive**: Driver 1 takes wider line (larger radius = faster but longer)
- **Negative**: Driver 1 takes tighter line (shorter but potentially slower)

### Speed Trace Analysis

When viewing speed traces:

1. **Parallel lines**: Similar driving style and pace
2. **Converging lines**: One driver catching/pulling away
3. **Early divergence**: Different brake points
4. **Apex gap**: Different minimum speeds through corner
5. **Exit slope**: Different acceleration rates

## Use Cases

### 1. Driver Coaching

**Goal**: Identify where student driver loses time to reference driver

**Workflow**:
1. Run comparison between student and coach/reference
2. Identify corners with largest apex speed delta
3. Focus coaching on brake points and corner entry technique
4. Use visualizations to show ideal racing line

### 2. Setup Optimization

**Goal**: Determine if setup changes improve corner performance

**Workflow**:
1. Compare before/after racing lines
2. Look for:
   - Higher apex speeds (better grip)
   - Earlier throttle application (better traction)
   - Larger corner radius (more confidence)
3. Quantify improvement in each corner

### 3. Driver Style Analysis

**Goal**: Understand different racing approaches

**Workflow**:
1. Compare multiple drivers on same track
2. Classify approaches:
   - **Late braking, geometric**: Late brake, slow apex, early throttle
   - **Early braking, momentum**: Early brake, fast apex, maintain flow
   - **Aggressive**: High apex speeds, maximum lateral G
   - **Conservative**: Lower apex speeds, more margin
3. Determine which style is fastest for each corner

### 4. Competitive Analysis

**Goal**: Understand competitor advantages

**Workflow**:
1. Compare your line to competitor's
2. Identify:
   - Where they gain time (faster corners)
   - Their technique (brake points, apex position)
   - Replicable advantages vs. car/driver limitations
3. Develop strategy to match or counter

## Tips for Accurate Reconstruction

### 1. Data Quality

- **High sampling rate**: 10 Hz minimum, 100 Hz+ ideal
- **Clean data**: Filter outliers and sensor errors
- **Consistent reference**: Use same lap/session for comparison
- **Similar conditions**: Compare dry-to-dry, wet-to-wet

### 2. Parameter Tuning

Adjust reconstructor parameters based on car type:

**Sports Cars / GT**:
```python
reconstructor = RacingLineReconstructor(
    lateral_g_assumption=1.8,
    track_width_m=12.0
)
```

**Formula Cars / High Downforce**:
```python
reconstructor = RacingLineReconstructor(
    lateral_g_assumption=2.5,  # More downforce
    track_width_m=12.0
)
```

**Street Cars / HPDE**:
```python
reconstructor = RacingLineReconstructor(
    lateral_g_assumption=1.2,  # Lower grip
    track_width_m=10.0  # Narrower use of track
)
```

### 3. Corner Classification

The system attempts to classify corners by:
- **Tight corners**: Low minimum speed (< 80 km/h), small radius
- **Medium corners**: Moderate speed (80-120 km/h), medium radius
- **Fast corners**: High speed (> 120 km/h), large radius

Use this to focus analysis on similar corner types.

## Code Examples

### Basic Single Driver Analysis

```python
from src.tactical.racing_line import RacingLineReconstructor
import pandas as pd

# Load telemetry
telemetry = pd.read_csv('driver_telemetry.csv')

# Initialize reconstructor
reconstructor = RacingLineReconstructor()

# Reconstruct line
line = reconstructor.reconstruct_line(
    telemetry,
    speed_col='gps_speed',
    brake_col='brake_f',
    throttle_col='aps'
)

# Access results
corners = line['corners']
trajectory = line['trajectory']
stats = line['statistics']

# Print corner info
for corner in corners:
    print(f"Corner {corner['corner_number']}: "
          f"Apex at {corner['apex']:.1f}%, "
          f"Speed {corner['apex_speed']:.1f} km/h, "
          f"Radius {corner['radius_m']:.1f}m")
```

### Two-Driver Comparison

```python
from src.tactical.racing_line import RacingLineReconstructor

# Load both drivers' telemetry
telem_a = pd.read_csv('driver_a.csv')
telem_b = pd.read_csv('driver_b.csv')

# Initialize and compare
reconstructor = RacingLineReconstructor()
comparison = reconstructor.compare_racing_lines(
    telem_a,
    telem_b,
    driver1_label="Driver A",
    driver2_label="Driver B"
)

# Access comparison results
differences = comparison['differences']
summary = comparison['summary']

# Print corner-by-corner differences
for diff in differences:
    print(f"Corner {diff['corner_number']}: "
          f"Speed Δ = {diff['apex_speed_delta_kph']:+.1f} km/h, "
          f"Faster: {diff['faster_apex_speed']}")
```

### Visualization

```python
from src.utils.visualization import (
    create_racing_line_comparison,
    create_corner_analysis,
    create_speed_trace_comparison
)
from src.utils.track_layouts import get_track_layout

# Get track layout
track_layout = get_track_layout('barber')

# Create visualizations
fig_lines = create_racing_line_comparison(
    comparison['driver1_line'],
    comparison['driver2_line'],
    track_layout,
    "Driver A",
    "Driver B"
)

fig_corners = create_corner_analysis(
    {},
    "Driver A",
    "Driver B",
    differences
)

fig_speed = create_speed_trace_comparison(
    comparison['driver1_line'],
    comparison['driver2_line'],
    "Driver A",
    "Driver B",
    corner_number=5  # Specific corner
)

# Save or display
fig_lines.write_html('racing_line_comparison.html')
fig_corners.show()
```

## Technical Implementation

### Algorithm Complexity

- **Corner identification**: O(n) where n = number of telemetry points
- **Corner geometry**: O(c) where c = number of corners
- **Trajectory building**: O(n)
- **Comparison**: O(min(c1, c2))

**Total**: O(n) linear complexity - very efficient even for large datasets

### Memory Usage

- **Telemetry data**: ~1 KB per 10 points (with 5-6 channels)
- **Reconstructed line**: ~500 bytes per corner
- **Trajectory**: Same size as input telemetry
- **Comparison**: ~2x single reconstruction

**Example**: 10,000 telemetry points ≈ 1 MB, very manageable

### Performance

Typical processing times (on modern CPU):
- **Single driver reconstruction**: 50-200ms
- **Two-driver comparison**: 100-400ms
- **Visualization generation**: 500-1000ms

## Future Enhancements

### Potential Improvements

1. **GPS-based reconstruction**: Use actual GPS coordinates for true track position
2. **Elevation consideration**: Account for uphill/downhill effects on speed
3. **Variable lateral G**: Estimate lateral G from corner speed/radius dynamically
4. **Machine learning**: Learn optimal racing lines from professional drivers
5. **Real-time analysis**: Process telemetry during live sessions
6. **3D visualization**: Show racing lines with elevation changes

### Integration Opportunities

1. **Lap time prediction**: Estimate lap time from reconstructed line
2. **Optimal line calculation**: Calculate theoretical optimal line
3. **Setup correlation**: Link racing line characteristics to car setup
4. **Tire wear modeling**: Estimate tire degradation from corner loads

## References

### Racing Theory

- **Racing Line Fundamentals**: "Going Faster" by Carl Lopez
- **Vehicle Dynamics**: "Race Car Vehicle Dynamics" by Milliken & Milliken
- **Driver Technique**: "Speed Secrets" series by Ross Bentley

### Physics

- **Lateral Dynamics**: Classical mechanics circular motion equations
- **Tire Physics**: Pacejka tire models for grip estimation
- **Vehicle Dynamics**: SAE papers on racing car dynamics

## Support

For questions or issues:
- Check example scripts in `examples/racing_line_demo.py`
- Review test cases in `tests/test_racing_line.py`
- Consult API documentation in source code docstrings

## Version History

- **v1.0.0** (2024): Initial implementation
  - Basic corner identification
  - Physics-based radius calculation
  - Two-driver comparison
  - Track map visualization
  - Corner-by-corner analysis

---

**Note**: This feature reconstructs *approximate* racing lines based on available telemetry. For precise track positioning, GPS coordinates are required. The current implementation provides valuable comparative analysis even without GPS data.
