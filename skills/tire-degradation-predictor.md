# Tire Degradation Predictor
## Advanced tire performance modeling for GR Cup Series strategy

### Overview
This skill focuses on predicting tire performance degradation using physics-based models, machine learning, and real-time telemetry analysis. It helps teams optimize tire strategy, predict performance cliffs, and make informed decisions about stint lengths and compound selection.

### Core Capabilities

#### 1. Degradation Modeling
- **Wear Rate Prediction**: Thermal and mechanical wear calculation
- **Performance Cliff Detection**: Identify when tires "fall off"
- **Compound Comparison**: Relative performance across tire types
- **Life Expectancy**: Estimate remaining competitive laps

#### 2. Temperature Management
- **Thermal Modeling**: Core, surface, and bulk temperature tracking
- **Operating Window**: Optimal temperature range identification
- **Heat Generation**: Predict temperature rise from driving style
- **Cooling Strategies**: Recommend techniques to manage temps

#### 3. Grip Evolution
- **Grip Level Estimation**: Real-time available grip calculation
- **Degradation Curves**: Non-linear grip loss modeling
- **Track Evolution**: Account for rubber buildup effects
- **Weather Impact**: Temperature and moisture effects

#### 4. Strategic Planning
- **Stint Optimization**: Calculate ideal stint lengths
- **Compound Selection**: Data-driven tire choice recommendations
- **Push vs. Conserve**: Balance performance and longevity
- **Multi-stint Planning**: Full race tire allocation strategy

### Physics-Based Model

```python
import numpy as np
from scipy.integrate import odeint

class TirePhysicsModel:
    def __init__(self, tire_compound):
        self.compound_properties = self.load_compound_data(tire_compound)
        self.wear_coefficient = self.compound_properties['wear_rate']
        self.thermal_capacity = self.compound_properties['heat_capacity']
        self.optimal_temp_range = self.compound_properties['optimal_temp']

    def calculate_wear(self, load_history, slip_history, temp_history):
        """Calculate tire wear based on physical forces"""
        mechanical_wear = self.mechanical_wear_model(load_history, slip_history)
        thermal_wear = self.thermal_wear_model(temp_history)
        chemical_wear = self.chemical_degradation_model(temp_history)

        total_wear = mechanical_wear + thermal_wear + chemical_wear
        return {
            'total_wear_mm': total_wear,
            'mechanical_component': mechanical_wear,
            'thermal_component': thermal_wear,
            'chemical_component': chemical_wear
        }

    def mechanical_wear_model(self, load, slip):
        """Archard wear equation for mechanical degradation"""
        # W = K * L * S / H
        # W: wear volume, K: wear coefficient, L: load, S: sliding distance, H: hardness
        sliding_distance = np.sum(slip * self.contact_patch_length)
        wear_volume = self.wear_coefficient * load * sliding_distance / self.hardness
        wear_depth = wear_volume / self.contact_area
        return wear_depth

    def thermal_model(self, ambient_temp, speed, slip_angle, load):
        """Heat generation and dissipation model"""
        def temp_dynamics(T, t):
            # Heat generation from friction
            Q_friction = self.friction_coefficient * load * speed * np.abs(slip_angle)

            # Heat dissipation
            Q_convection = self.convection_coefficient * (T - ambient_temp) * speed
            Q_conduction = self.conduction_coefficient * (T - ambient_temp)

            # Temperature change
            dT_dt = (Q_friction - Q_convection - Q_conduction) / self.thermal_capacity
            return dT_dt

        time_points = np.linspace(0, 1, 100)
        temperatures = odeint(temp_dynamics, ambient_temp, time_points)
        return temperatures[-1]
```

### Machine Learning Models

#### Deep Learning Degradation Predictor
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class TireDegradationNN:
    def __init__(self):
        self.model = self.build_model()
        self.scaler = self.load_scaler()

    def build_model(self):
        """LSTM-based degradation prediction model"""
        # Input layers
        telemetry_input = layers.Input(shape=(None, 15), name='telemetry')
        conditions_input = layers.Input(shape=(7,), name='conditions')

        # LSTM for time series
        lstm = layers.LSTM(128, return_sequences=True)(telemetry_input)
        lstm = layers.LSTM(64)(lstm)

        # Dense for conditions
        conditions = layers.Dense(32, activation='relu')(conditions_input)

        # Combine
        combined = layers.concatenate([lstm, conditions])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        combined = layers.Dense(32, activation='relu')(combined)

        # Output layers for different predictions
        wear_output = layers.Dense(1, name='wear')(combined)
        temp_output = layers.Dense(4, name='temperatures')(combined)
        grip_output = layers.Dense(1, name='grip_level')(combined)

        model = Model(
            inputs=[telemetry_input, conditions_input],
            outputs=[wear_output, temp_output, grip_output]
        )

        model.compile(
            optimizer='adam',
            loss={'wear': 'mse', 'temperatures': 'mse', 'grip_level': 'mse'},
            loss_weights={'wear': 1.0, 'temperatures': 0.5, 'grip_level': 1.5}
        )

        return model

    def predict_degradation(self, telemetry_sequence, conditions):
        """Predict tire degradation for next N laps"""
        # Preprocess inputs
        telemetry_scaled = self.scaler.transform(telemetry_sequence)
        conditions_scaled = self.scaler.transform(conditions)

        # Predict
        wear, temps, grip = self.model.predict([telemetry_scaled, conditions_scaled])

        return {
            'predicted_wear_mm': wear[0][0],
            'tire_temperatures': {
                'FL': temps[0][0], 'FR': temps[0][1],
                'RL': temps[0][2], 'RR': temps[0][3]
            },
            'grip_percentage': grip[0][0] * 100
        }
```

#### Ensemble Predictor
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

class EnsembleDegradationPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100),
            'gb': GradientBoostingRegressor(n_estimators=100),
            'linear': LinearRegression(),
            'neural': TireDegradationNN()
        }
        self.weights = {'rf': 0.25, 'gb': 0.25, 'linear': 0.15, 'neural': 0.35}

    def predict(self, features):
        """Weighted ensemble prediction"""
        predictions = {}

        for name, model in self.models.items():
            if name == 'neural':
                pred = model.predict_degradation(features['telemetry'], features['conditions'])
                predictions[name] = pred['predicted_wear_mm']
            else:
                pred = model.predict(features['flat'])
                predictions[name] = pred[0]

        # Weighted average
        final_prediction = sum(
            predictions[name] * self.weights[name]
            for name in predictions
        )

        # Calculate confidence based on prediction variance
        variance = np.var(list(predictions.values()))
        confidence = 1 / (1 + variance)

        return {
            'wear_prediction': final_prediction,
            'confidence': confidence,
            'model_predictions': predictions
        }
```

### Real-time Analysis

#### Live Tire Monitoring
```python
class LiveTireMonitor:
    def __init__(self):
        self.baseline_performance = {}
        self.current_state = {}
        self.alert_thresholds = {
            'temp_critical': 280,  # °F
            'wear_critical': 3.0,   # mm
            'grip_critical': 70     # %
        }

    def process_telemetry_frame(self, telemetry):
        """Process single telemetry frame"""
        # Update tire state
        self.update_temperatures(telemetry['tire_temps'])
        self.update_pressures(telemetry['tire_pressures'])
        self.estimate_wear(telemetry)
        self.calculate_grip_level(telemetry)

        # Check for alerts
        alerts = self.check_alert_conditions()

        # Performance delta
        performance_delta = self.calculate_performance_delta()

        return {
            'state': self.current_state,
            'alerts': alerts,
            'performance_delta': performance_delta,
            'recommendations': self.generate_recommendations()
        }

    def estimate_wear(self, telemetry):
        """Estimate current wear level"""
        # Use sliding detection and load to estimate wear
        slip_ratio = telemetry['wheel_speed'] / telemetry['vehicle_speed'] - 1
        lateral_slip = np.arctan(telemetry['lateral_velocity'] / telemetry['longitudinal_velocity'])

        wear_rate = (
            abs(slip_ratio) * self.wear_factors['longitudinal'] +
            abs(lateral_slip) * self.wear_factors['lateral']
        ) * telemetry['vertical_load']

        self.current_state['wear'] += wear_rate * self.dt

    def calculate_performance_cliff(self):
        """Predict when tire will hit performance cliff"""
        current_wear = self.current_state['wear']
        current_temp = self.current_state['avg_temp']
        deg_rate = self.calculate_current_deg_rate()

        # Cliff typically occurs at 65-75% wear
        cliff_wear = self.compound_properties['cliff_point']
        laps_to_cliff = (cliff_wear - current_wear) / deg_rate

        return {
            'laps_remaining': max(0, laps_to_cliff),
            'confidence': self.cliff_confidence(current_wear, deg_rate),
            'cliff_severity': self.predict_cliff_severity()
        }
```

### Visualization Components

#### Tire Temperature Heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_tire_heatmap(temp_data):
    """Create heatmap showing tire temperature distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    positions = {
        'FL': (0, 0), 'FR': (0, 1),
        'RL': (1, 0), 'RR': (1, 1)
    }

    for tire, (row, col) in positions.items():
        # Create temperature grid (inside, middle, outside)
        temp_grid = np.array([
            temp_data[tire]['inside'],
            temp_data[tire]['middle'],
            temp_data[tire]['outside']
        ]).reshape(3, 1)

        sns.heatmap(
            temp_grid,
            annot=True,
            fmt='.1f',
            cmap='RdYlBu_r',
            vmin=180, vmax=280,
            cbar_kws={'label': '°F'},
            ax=axes[row, col]
        )
        axes[row, col].set_title(f'{tire} Tire')
        axes[row, col].set_yticklabels(['Inside', 'Middle', 'Outside'])

    plt.tight_layout()
    return fig
```

#### Degradation Curves
```python
def plot_degradation_curves(predictions, actual=None):
    """Plot predicted vs actual degradation"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Lap time degradation
    laps = predictions['laps']
    lap_times = predictions['lap_times']

    ax1.plot(laps, lap_times, 'b-', label='Predicted', linewidth=2)
    if actual:
        ax1.plot(actual['laps'], actual['lap_times'], 'r--', label='Actual')

    ax1.fill_between(
        laps,
        lap_times - predictions['confidence_interval'],
        lap_times + predictions['confidence_interval'],
        alpha=0.3
    )
    ax1.set_xlabel('Lap Number')
    ax1.set_ylabel('Lap Time (s)')
    ax1.set_title('Lap Time Degradation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Grip level evolution
    grip_levels = predictions['grip_levels']
    ax2.plot(laps, grip_levels, 'g-', linewidth=2)
    ax2.axhline(y=70, color='orange', linestyle='--', label='Performance Cliff')
    ax2.fill_between(laps, 0, grip_levels, alpha=0.3, color='green')
    ax2.set_xlabel('Lap Number')
    ax2.set_ylabel('Grip Level (%)')
    ax2.set_title('Tire Grip Evolution')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    return fig
```

### Strategy Integration

#### Tire Strategy Optimizer
```python
class TireStrategyOptimizer:
    def __init__(self, race_length, available_sets):
        self.race_length = race_length
        self.available_sets = available_sets
        self.degradation_models = {}

    def optimize_tire_strategy(self, conditions):
        """Find optimal tire strategy for race"""
        strategies = []

        # Generate possible strategies
        for num_stops in range(1, 4):
            for compounds in self.generate_compound_combinations(num_stops):
                for stints in self.generate_stint_lengths(num_stops):
                    strategy = {
                        'stops': num_stops,
                        'compounds': compounds,
                        'stints': stints
                    }
                    time = self.simulate_strategy(strategy, conditions)
                    strategies.append({**strategy, 'total_time': time})

        # Sort by total time
        strategies.sort(key=lambda x: x['total_time'])

        return {
            'optimal': strategies[0],
            'alternatives': strategies[1:5],
            'risk_analysis': self.analyze_strategy_risks(strategies[0])
        }

    def simulate_strategy(self, strategy, conditions):
        """Simulate race with given tire strategy"""
        total_time = 0
        current_lap = 0

        for stint_idx, stint_length in enumerate(strategy['stints']):
            compound = strategy['compounds'][stint_idx]

            # Predict degradation for stint
            deg_curve = self.degradation_models[compound].predict(
                stint_length,
                conditions,
                fresh_tires=(stint_idx > 0)
            )

            # Sum lap times
            stint_time = sum(deg_curve['lap_times'])
            total_time += stint_time

            # Add pit stop time (except last stint)
            if stint_idx < len(strategy['stints']) - 1:
                total_time += PIT_STOP_TIME

            current_lap += stint_length

        return total_time
```

### Calibration System

```python
class TireModelCalibration:
    def __init__(self):
        self.calibration_data = []
        self.model_parameters = self.load_default_parameters()

    def calibrate_from_session(self, session_data):
        """Calibrate model using actual session data"""
        # Extract features and targets
        features = self.extract_calibration_features(session_data)
        targets = self.extract_wear_measurements(session_data)

        # Optimize model parameters
        from scipy.optimize import minimize

        def objective(params):
            self.model_parameters = params
            predictions = self.predict_wear(features)
            error = np.mean((predictions - targets) ** 2)
            return error

        result = minimize(
            objective,
            self.model_parameters,
            method='L-BFGS-B',
            bounds=self.parameter_bounds
        )

        self.model_parameters = result.x
        return {
            'calibrated_parameters': self.model_parameters,
            'error_reduction': 1 - result.fun / self.baseline_error,
            'confidence': self.calculate_calibration_confidence()
        }

    def online_learning(self, new_data):
        """Continuously update model with new data"""
        # Add to calibration buffer
        self.calibration_data.append(new_data)

        # Periodic re-calibration
        if len(self.calibration_data) >= 10:
            self.calibrate_from_session(self.calibration_data)
            self.calibration_data = []  # Reset buffer

        # Immediate adjustment for large errors
        if new_data['prediction_error'] > self.error_threshold:
            self.quick_adjust(new_data)
```

### Output Formats

#### Tire Status Report
```yaml
tire_status:
  timestamp: "Lap 35, 52:30.123"

  temperatures:
    FL: {inner: 245, middle: 250, outer: 248, core: 265}
    FR: {inner: 248, middle: 252, outer: 245, core: 267}
    RL: {inner: 240, middle: 245, outer: 243, core: 260}
    RR: {inner: 243, middle: 247, outer: 240, core: 262}
    status: "Optimal window"

  wear:
    FL: 2.1mm
    FR: 2.2mm
    RL: 1.9mm
    RR: 2.0mm
    average: 2.05mm
    rate: 0.08mm/lap

  performance:
    current_grip: 82%
    lap_time_delta: +0.6s
    laps_to_cliff: 12
    recommended_stint: 8

  alerts:
    - "FR temperature approaching critical"
    - "Wear rate higher than expected"

  recommendations:
    immediate: "Reduce front brake bias 2%"
    next_lap: "Cool tires on straight"
    strategic: "Plan stop within 8 laps"
```

### Best Practices

1. **Regular Calibration**: Update models with each session
2. **Conservative Estimates**: Add safety margin to predictions
3. **Driver Communication**: Clear feedback on tire state
4. **Data Quality**: Validate sensor readings
5. **Weather Awareness**: Adjust for track temperature changes

### Related Skills
- telemetry-analyzer
- driver-performance-analyzer
- race-strategy-optimizer
- real-time-dashboard-builder