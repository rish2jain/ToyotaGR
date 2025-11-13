# Racing Telemetry Analyzer
## Data-driven insights from motorsports telemetry for GR Cup Series

### Overview
This skill provides comprehensive telemetry analysis capabilities for processing and analyzing real-time racing data from Toyota GR Cup Series vehicles. It focuses on extracting actionable insights from sensor data including speed, throttle position, brake pressure, steering angle, G-forces, tire temperatures, and lap times.

### Core Capabilities

#### 1. Data Ingestion & Processing
- **Multi-format Support**: Process CSV, JSON, and streaming telemetry formats
- **Time Series Alignment**: Synchronize multi-sensor data streams with millisecond precision
- **Data Validation**: Automatic outlier detection and sensor fault identification
- **Interpolation**: Handle missing data points with racing-aware interpolation methods

#### 2. Performance Metrics Extraction
- **Lap Time Analysis**: Sector splits, ideal lap calculations, consistency metrics
- **Cornering Performance**: Entry/apex/exit speeds, racing line optimization
- **Braking Analysis**: Brake point identification, pressure patterns, temperature monitoring
- **Acceleration Zones**: Throttle application patterns, traction analysis

#### 3. Driver Behavior Profiling
- **Driving Style Classification**: Aggressive vs. smooth, early vs. late braking
- **Consistency Scoring**: Lap-to-lap variance, sector consistency
- **Risk Assessment**: Over-driving indicators, safety margin calculations
- **Improvement Areas**: Specific corner/sector improvement opportunities

#### 4. Vehicle Dynamics Analysis
- **Weight Transfer**: Pitch and roll dynamics during cornering
- **Tire Performance**: Temperature distribution, grip levels, wear patterns
- **Aerodynamic Efficiency**: Drag vs. downforce balance indicators
- **Mechanical Stress**: Component load analysis, failure prediction

### Implementation Components

```python
class TelemetryAnalyzer:
    def __init__(self):
        self.sampling_rate = 100  # Hz
        self.track_map = None
        self.reference_lap = None

    def process_lap(self, telemetry_data):
        """Process single lap telemetry data"""
        # Time series processing
        # Feature extraction
        # Performance calculation
        pass

    def compare_drivers(self, driver1_data, driver2_data):
        """Compare two drivers' performance"""
        # Delta time calculation
        # Corner analysis
        # Optimization suggestions
        pass

    def identify_improvements(self, current_lap, reference_lap):
        """Identify specific areas for improvement"""
        # Braking point optimization
        # Racing line suggestions
        # Throttle application timing
        pass
```

### Key Algorithms

#### Racing Line Optimization
- Geometric line calculation based on track boundaries
- Friction circle utilization analysis
- Entry-apex-exit trajectory optimization
- Speed vs. distance trade-off calculations

#### Predictive Analytics
- Lap time prediction based on sector performance
- Tire degradation modeling
- Fuel consumption estimation
- Weather impact assessment

#### Real-time Processing
- Sliding window analysis for live telemetry
- Event detection (overtaking, lockups, wheel spin)
- Performance delta calculations
- Alert generation for critical events

### Data Structures

```yaml
telemetry_frame:
  timestamp: milliseconds
  position:
    lat: float
    lon: float
    track_distance: meters
  motion:
    speed: km/h
    acceleration:
      longitudinal: g
      lateral: g
      vertical: g
  driver_inputs:
    throttle: percentage
    brake: percentage
    steering: degrees
  vehicle_state:
    tire_temp: [FL, FR, RL, RR]
    tire_pressure: [FL, FR, RL, RR]
    engine:
      rpm: integer
      temp: celsius
      oil_pressure: bar
```

### Integration Points

#### Toyota GR Data Sources
- GR Cup Series official telemetry feeds
- Historical race data archives
- Weather and track condition data
- Official timing and scoring systems

#### Visualization Outputs
- Track map overlays with telemetry traces
- Comparative analysis dashboards
- Real-time performance monitors
- Post-session analysis reports

### Usage Examples

```python
# Initialize analyzer
analyzer = TelemetryAnalyzer()
analyzer.load_track("Sonoma Raceway")

# Process lap data
lap_data = load_telemetry("driver1_qualifying_lap3.csv")
performance = analyzer.process_lap(lap_data)

# Find improvement opportunities
improvements = analyzer.identify_improvements(
    current_lap=lap_data,
    reference_lap=analyzer.track_record
)

# Generate driver comparison
comparison = analyzer.compare_drivers(
    driver1_data=load_session("driver1"),
    driver2_data=load_session("driver2")
)

# Real-time monitoring
stream = TelemetryStream("live_feed")
for frame in stream:
    alerts = analyzer.process_realtime(frame)
    if alerts:
        notify_pit_crew(alerts)
```

### Performance Optimization

#### Computational Efficiency
- Vectorized operations for time series data
- Caching of frequently accessed calculations
- Parallel processing for multi-car analysis
- GPU acceleration for complex algorithms

#### Memory Management
- Circular buffers for streaming data
- Compression for historical data storage
- Selective loading of relevant data segments
- Efficient data structure design

### Output Formats

#### Analysis Reports
- Lap summary statistics
- Sector-by-sector breakdown
- Driver coaching recommendations
- Vehicle setup suggestions

#### Real-time Feeds
- Live delta times
- Performance alerts
- Strategic recommendations
- Predictive outcomes

### Best Practices

1. **Data Quality**: Always validate incoming telemetry for sensor errors
2. **Normalization**: Account for track conditions when comparing performances
3. **Context Awareness**: Consider tire age, fuel load, and weather conditions
4. **Privacy**: Respect driver data confidentiality and team strategies
5. **Accuracy**: Clearly communicate confidence levels in predictions

### Related Skills
- driver-performance-analyzer
- race-strategy-optimizer
- tire-degradation-predictor
- real-time-dashboard-builder