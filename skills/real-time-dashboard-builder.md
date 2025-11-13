# Real-Time Dashboard Builder
## Interactive visualization platform for GR Cup Series race analytics

### Overview
This skill provides comprehensive tools for building real-time, interactive dashboards that visualize telemetry data, strategic decisions, and performance metrics. It enables teams to monitor races live, analyze historical data, and make data-driven decisions through intuitive visualizations.

### Core Capabilities

#### 1. Live Data Streaming
- **WebSocket Integration**: Real-time telemetry feed handling
- **Data Buffering**: Efficient stream processing and caching
- **Latency Optimization**: Sub-second update rates
- **Multi-source Sync**: Coordinate multiple data streams

#### 2. Visualization Components
- **Track Map Overlay**: Live car positions and telemetry traces
- **Time Series Charts**: Performance metrics over time
- **Comparative Analysis**: Multi-driver/multi-lap comparisons
- **3D Visualizations**: Track elevation and racing lines

#### 3. Interactive Features
- **Drill-down Analysis**: Click to explore detailed data
- **Custom Views**: User-configurable dashboard layouts
- **Alerts & Notifications**: Real-time event highlighting
- **Playback Controls**: Session replay with time scrubbing

#### 4. Performance Optimization
- **Efficient Rendering**: WebGL and Canvas optimization
- **Data Aggregation**: Smart sampling for large datasets
- **Lazy Loading**: On-demand data fetching
- **Responsive Design**: Adaptive layouts for any device

### Architecture

```python
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import pandas as pd

class RealTimeDashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.data_buffer = CircularBuffer(maxsize=10000)
        self.active_visualizations = {}
        self.setup_routes()
        self.setup_socketio()

    def setup_routes(self):
        """Configure dashboard routes"""
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')

        @self.app.route('/api/config')
        def get_config():
            return {
                'track': self.track_config,
                'drivers': self.driver_list,
                'telemetry_channels': self.available_channels
            }

    def setup_socketio(self):
        """Configure WebSocket handlers"""
        @self.socketio.on('connect')
        def handle_connect():
            emit('connected', {'status': 'Connected to live feed'})
            self.start_streaming()

        @self.socketio.on('request_data')
        def handle_data_request(data):
            channel = data['channel']
            timeframe = data.get('timeframe', 'live')
            response_data = self.get_channel_data(channel, timeframe)
            emit('data_update', response_data)

    def start_streaming(self):
        """Begin streaming telemetry data"""
        def stream_loop():
            while self.streaming_active:
                latest_data = self.telemetry_source.get_latest()
                processed = self.process_telemetry(latest_data)
                self.socketio.emit('telemetry_update', processed)
                time.sleep(0.1)  # 10Hz update rate

        self.socketio.start_background_task(stream_loop)
```

### Frontend Components

#### React Dashboard Framework
```javascript
import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import Plot from 'react-plotly.js';

const RaceDashboard = () => {
    const [telemetryData, setTelemetryData] = useState({});
    const [selectedDriver, setSelectedDriver] = useState(null);
    const [viewMode, setViewMode] = useState('live');
    const socket = io('http://localhost:5000');

    useEffect(() => {
        // Socket connection handlers
        socket.on('telemetry_update', (data) => {
            setTelemetryData(prev => ({
                ...prev,
                [data.timestamp]: data
            }));
        });

        return () => socket.disconnect();
    }, []);

    return (
        <div className="dashboard-container">
            <HeaderPanel />
            <div className="main-content">
                <TrackMap data={telemetryData} />
                <TelemetryCharts data={telemetryData} driver={selectedDriver} />
                <StrategyPanel />
                <LeaderBoard />
            </div>
            <ControlPanel
                onDriverSelect={setSelectedDriver}
                onViewModeChange={setViewMode}
            />
        </div>
    );
};
```

#### Track Map Visualization
```javascript
const TrackMap = ({ data }) => {
    const [carPositions, setCarPositions] = useState([]);
    const [selectedLap, setSelectedLap] = useState('current');

    const drawTrack = (ctx, trackData) => {
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 20;
        ctx.beginPath();

        trackData.points.forEach((point, idx) => {
            if (idx === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });

        ctx.stroke();
    };

    const drawCars = (ctx, positions) => {
        positions.forEach(car => {
            // Car position
            ctx.fillStyle = car.color;
            ctx.beginPath();
            ctx.arc(car.x, car.y, 5, 0, 2 * Math.PI);
            ctx.fill();

            // Telemetry trace
            if (car.trace) {
                ctx.strokeStyle = car.color;
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                car.trace.forEach((point, idx) => {
                    if (idx === 0) ctx.moveTo(point.x, point.y);
                    else ctx.lineTo(point.x, point.y);
                });
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }
        });
    };

    return (
        <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="track-map"
        />
    );
};
```

### Visualization Library

#### Telemetry Charts
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TelemetryVisualizer:
    def __init__(self):
        self.chart_configs = self.load_chart_configs()
        self.color_scheme = self.load_color_scheme()

    def create_multi_driver_comparison(self, drivers_data, metrics):
        """Create comparative visualization for multiple drivers"""
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=True,
            subplot_titles=metrics
        )

        for row, metric in enumerate(metrics, 1):
            for driver_name, driver_data in drivers_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=driver_data['distance'],
                        y=driver_data[metric],
                        name=driver_name,
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=row, col=1
                )

        fig.update_layout(
            height=200 * len(metrics),
            showlegend=True,
            title="Multi-Driver Telemetry Comparison"
        )

        return fig

    def create_sector_analysis(self, lap_data):
        """Sector time breakdown visualization"""
        sectors = ['Sector 1', 'Sector 2', 'Sector 3']

        fig = go.Figure()

        # Add bars for each sector
        for i, sector in enumerate(sectors):
            fig.add_trace(go.Bar(
                name=sector,
                x=lap_data['lap_numbers'],
                y=lap_data[f'sector_{i+1}_times'],
                text=lap_data[f'sector_{i+1}_times'],
                textposition='auto'
            ))

        # Add theoretical best line
        theoretical_best = [
            min(lap_data[f'sector_{i+1}_times'])
            for i in range(3)
        ]

        fig.add_trace(go.Scatter(
            x=lap_data['lap_numbers'],
            y=[sum(theoretical_best)] * len(lap_data['lap_numbers']),
            mode='lines',
            name='Theoretical Best',
            line=dict(dash='dash', color='red')
        ))

        return fig

    def create_3d_track_visualization(self, telemetry_data):
        """3D track visualization with elevation"""
        fig = go.Figure(data=[
            go.Scatter3d(
                x=telemetry_data['x_position'],
                y=telemetry_data['y_position'],
                z=telemetry_data['elevation'],
                mode='markers+lines',
                marker=dict(
                    size=3,
                    color=telemetry_data['speed'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed (mph)")
                ),
                line=dict(
                    color=telemetry_data['throttle'],
                    width=4
                ),
                text=[f"Speed: {s:.1f} mph" for s in telemetry_data['speed']],
                hovertemplate='%{text}<extra></extra>'
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Elevation',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            title='3D Track Analysis'
        )

        return fig
```

#### Strategy Visualization
```python
def create_strategy_timeline(strategy_data):
    """Interactive strategy timeline"""
    fig = go.Figure()

    # Add pit stop windows
    for window in strategy_data['pit_windows']:
        fig.add_vrect(
            x0=window['start'], x1=window['end'],
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Pit Window"
        )

    # Add tire degradation curves
    for stint in strategy_data['stints']:
        fig.add_trace(go.Scatter(
            x=stint['laps'],
            y=stint['performance'],
            mode='lines',
            name=f"Stint {stint['number']} - {stint['compound']}",
            line=dict(width=3)
        ))

    # Add competitor strategies
    for competitor in strategy_data['competitors']:
        fig.add_trace(go.Scatter(
            x=competitor['pit_laps'],
            y=[100] * len(competitor['pit_laps']),
            mode='markers',
            name=competitor['name'],
            marker=dict(size=10, symbol='triangle-down')
        ))

    fig.update_layout(
        xaxis_title='Lap',
        yaxis_title='Performance %',
        title='Race Strategy Timeline',
        hovermode='x unified'
    )

    return fig
```

### Real-time Processing

#### Stream Processing Pipeline
```python
import asyncio
from collections import deque
import numpy as np

class TelemetryStreamProcessor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        self.processors = self.initialize_processors()

    async def process_stream(self, telemetry_stream):
        """Async processing of telemetry stream"""
        async for frame in telemetry_stream:
            # Buffer management
            self.update_buffers(frame)

            # Parallel processing
            tasks = [
                self.process_performance_metrics(frame),
                self.process_tire_data(frame),
                self.process_position_data(frame),
                self.detect_events(frame)
            ]
            results = await asyncio.gather(*tasks)

            # Emit processed data
            await self.emit_updates(results)

    def update_buffers(self, frame):
        """Update circular buffers with new data"""
        for channel, value in frame.items():
            self.buffers[channel].append(value)

    async def process_performance_metrics(self, frame):
        """Calculate real-time performance metrics"""
        metrics = {}

        # Rolling averages
        if len(self.buffers['speed']) >= 10:
            metrics['avg_speed'] = np.mean(list(self.buffers['speed'])[-10:])
            metrics['speed_variance'] = np.std(list(self.buffers['speed'])[-10:])

        # Lap time prediction
        if 'track_position' in frame:
            metrics['predicted_laptime'] = self.predict_laptime(frame['track_position'])

        return {'type': 'performance', 'data': metrics}

    async def detect_events(self, frame):
        """Detect significant events in telemetry"""
        events = []

        # Lockup detection
        if frame.get('brake_pressure', 0) > 90 and frame.get('wheel_speed', 0) < frame.get('vehicle_speed', 100) * 0.8:
            events.append({
                'type': 'lockup',
                'severity': 'warning',
                'location': frame['track_position']
            })

        # Oversteer detection
        if abs(frame.get('yaw_rate', 0)) > self.oversteer_threshold:
            events.append({
                'type': 'oversteer',
                'severity': 'info',
                'angle': frame['yaw_rate']
            })

        return {'type': 'events', 'data': events}
```

### Dashboard Components

#### Leaderboard Widget
```javascript
const LeaderBoard = ({ raceData }) => {
    const [sortBy, setSortBy] = useState('position');
    const [showDetails, setShowDetails] = useState(false);

    const renderDriverRow = (driver) => (
        <div className="driver-row" key={driver.id}>
            <div className="position">{driver.position}</div>
            <div className="driver-name">{driver.name}</div>
            <div className="gap">{driver.gap}</div>
            <div className="last-lap">{driver.lastLap}</div>
            {showDetails && (
                <>
                    <div className="tire-age">{driver.tireAge}</div>
                    <div className="fuel">{driver.fuelRemaining}%</div>
                </>
            )}
            <div className="mini-chart">
                <SparkLine data={driver.lapHistory} />
            </div>
        </div>
    );

    return (
        <div className="leaderboard-widget">
            <div className="header">
                <h3>Race Standings</h3>
                <button onClick={() => setShowDetails(!showDetails)}>
                    {showDetails ? 'Less' : 'More'}
                </button>
            </div>
            <div className="leaderboard-content">
                {raceData.drivers
                    .sort((a, b) => a[sortBy] - b[sortBy])
                    .map(renderDriverRow)}
            </div>
        </div>
    );
};
```

#### Alert System
```python
class AlertManager:
    def __init__(self):
        self.alert_rules = self.load_alert_rules()
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)

    def check_alerts(self, telemetry_data):
        """Check for alert conditions"""
        new_alerts = []

        for rule in self.alert_rules:
            if self.evaluate_rule(rule, telemetry_data):
                alert = self.create_alert(rule, telemetry_data)
                new_alerts.append(alert)
                self.active_alerts.append(alert)

        return new_alerts

    def create_alert(self, rule, data):
        """Create alert object"""
        return {
            'id': generate_alert_id(),
            'timestamp': time.time(),
            'type': rule['type'],
            'severity': rule['severity'],
            'title': rule['title'],
            'message': rule['message_template'].format(**data),
            'data': self.extract_relevant_data(rule, data),
            'actions': rule.get('actions', [])
        }

    def evaluate_rule(self, rule, data):
        """Evaluate alert rule condition"""
        try:
            # Parse rule expression
            condition = rule['condition']
            # Safe evaluation with limited scope
            return eval(condition, {'data': data, 'np': np})
        except:
            return False
```

### Performance Optimization

#### Data Aggregation
```python
class DataAggregator:
    def __init__(self):
        self.aggregation_levels = {
            'raw': 1,      # Every sample
            'fine': 10,    # Every 10th sample
            'medium': 50,  # Every 50th sample
            'coarse': 200  # Every 200th sample
        }

    def aggregate_for_zoom(self, data, zoom_level):
        """Dynamically aggregate based on zoom level"""
        if zoom_level >= 0.8:
            return data  # Full resolution

        elif zoom_level >= 0.5:
            return self.downsample(data, self.aggregation_levels['fine'])

        elif zoom_level >= 0.2:
            return self.downsample(data, self.aggregation_levels['medium'])

        else:
            return self.downsample(data, self.aggregation_levels['coarse'])

    def downsample(self, data, factor):
        """Intelligent downsampling preserving important features"""
        downsampled = []

        for i in range(0, len(data), factor):
            window = data[i:i+factor]

            # Keep extremes and average
            aggregated = {
                'timestamp': window[0]['timestamp'],
                'avg': np.mean([d['value'] for d in window]),
                'min': min(window, key=lambda x: x['value'])['value'],
                'max': max(window, key=lambda x: x['value'])['value']
            }
            downsampled.append(aggregated)

        return downsampled
```

### Deployment Configuration

```yaml
dashboard_config:
  server:
    host: "0.0.0.0"
    port: 5000
    cors_enabled: true

  streaming:
    update_rate_hz: 10
    buffer_size: 10000
    compression: true

  performance:
    max_concurrent_users: 100
    cache_ttl: 60
    lazy_loading: true

  visualizations:
    default_charts:
      - track_map
      - leaderboard
      - lap_times
      - tire_temps

    optional_charts:
      - sector_analysis
      - fuel_usage
      - weather_data
      - strategy_timeline

  data_sources:
    primary:
      type: "websocket"
      url: "ws://telemetry.gracing.com"

    backup:
      type: "http"
      url: "https://api.gracing.com/telemetry"

  authentication:
    required: true
    method: "oauth2"
    provider: "toyota_sso"
```

### User Experience Features

#### Customizable Layouts
```javascript
const DashboardLayoutManager = {
    layouts: {
        'race_engineer': ['track_map', 'strategy', 'tire_temps', 'fuel'],
        'driver_coach': ['driver_comparison', 'sector_times', 'racing_line'],
        'team_manager': ['leaderboard', 'strategy_overview', 'weather'],
        'broadcast': ['track_map', 'leaderboard', 'highlights', 'replays']
    },

    saveCustomLayout: (userId, layout) => {
        localStorage.setItem(`layout_${userId}`, JSON.stringify(layout));
    },

    loadUserLayout: (userId) => {
        const saved = localStorage.getItem(`layout_${userId}`);
        return saved ? JSON.parse(saved) : layouts['race_engineer'];
    }
};
```

### Best Practices

1. **Performance First**: Optimize for 60fps rendering
2. **Responsive Design**: Support mobile to 4K displays
3. **Accessibility**: WCAG 2.1 compliance for all components
4. **Error Handling**: Graceful degradation for data issues
5. **User Testing**: Iterate based on team feedback

### Related Skills
- telemetry-analyzer
- driver-performance-analyzer
- race-strategy-optimizer
- tire-degradation-predictor