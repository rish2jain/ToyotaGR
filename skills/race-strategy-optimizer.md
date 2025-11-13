# Race Strategy Optimizer
## Real-time decision engine for GR Cup Series race strategy

### Overview
This skill provides comprehensive race strategy optimization using predictive modeling, Monte Carlo simulations, and real-time data analysis. It helps teams make critical decisions about pit stops, tire strategy, fuel management, and race positioning to maximize finishing position.

### Core Capabilities

#### 1. Pit Stop Optimization
- **Window Calculation**: Optimal pit timing based on tire deg and traffic
- **Undercut/Overcut Analysis**: Strategic advantages of pit timing
- **Caution Probability**: Yellow flag prediction and response
- **Service Time Modeling**: Pit crew performance factors

#### 2. Tire Strategy Planning
- **Compound Selection**: Optimal tire choice for conditions
- **Stint Length Optimization**: Balancing speed vs. durability
- **Temperature Management**: Strategies to maintain optimal temps
- **Multi-stint Planning**: Full race tire allocation

#### 3. Fuel Management
- **Consumption Modeling**: Lap-by-lap fuel usage prediction
- **Saving Strategies**: Lift-and-coast optimization
- **Splash-and-Go Decisions**: Quick stop timing
- **Economy vs. Performance**: Trade-off analysis

#### 4. Position Management
- **Track Position Value**: Quantifying position importance
- **Overtaking Probability**: Success likelihood modeling
- **Defensive Strategy**: When to defend vs. conserve
- **Points Optimization**: Championship perspective

### Strategy Engine Architecture

```python
import numpy as np
from scipy.optimize import minimize

class RaceStrategyOptimizer:
    def __init__(self, race_config):
        self.race_length = race_config['laps']
        self.track = race_config['track']
        self.weather = race_config['weather']
        self.competitors = race_config['field']
        self.simulations = 10000  # Monte Carlo runs

    def optimize_strategy(self, current_state):
        """Main strategy optimization loop"""
        strategies = self.generate_strategy_space()
        outcomes = self.simulate_strategies(strategies, current_state)
        optimal = self.select_optimal_strategy(outcomes)
        return self.refine_strategy(optimal, current_state)

    def generate_strategy_space(self):
        """Create possible strategy options"""
        strategies = []
        for stops in range(0, 4):  # 0 to 3 stops
            for timing in self.generate_stop_windows(stops):
                for tires in self.generate_tire_choices(stops):
                    strategies.append({
                        'stops': stops,
                        'timing': timing,
                        'tires': tires
                    })
        return strategies

    def simulate_strategies(self, strategies, current_state):
        """Monte Carlo simulation of race outcomes"""
        results = []
        for strategy in strategies:
            outcomes = []
            for _ in range(self.simulations):
                outcome = self.simulate_race(strategy, current_state)
                outcomes.append(outcome)
            results.append({
                'strategy': strategy,
                'avg_position': np.mean([o['position'] for o in outcomes]),
                'win_probability': sum(o['position']==1 for o in outcomes)/len(outcomes),
                'risk_score': np.std([o['position'] for o in outcomes])
            })
        return results
```

### Predictive Models

#### Caution Flag Prediction
```python
class CautionPredictor:
    def __init__(self):
        self.historical_data = self.load_caution_history()
        self.model = self.train_caution_model()

    def predict_caution_probability(self, race_state):
        """Predict probability of caution in next N laps"""
        features = self.extract_features(race_state)
        probabilities = []

        for lap_offset in range(1, 11):  # Next 10 laps
            lap_features = features + [lap_offset]
            prob = self.model.predict_proba([lap_features])[0][1]
            probabilities.append({
                'lap': race_state['current_lap'] + lap_offset,
                'probability': prob
            })

        return probabilities

    def extract_features(self, race_state):
        """Feature engineering for caution prediction"""
        return [
            race_state['current_lap'] / race_state['total_laps'],
            race_state['green_flag_run_length'],
            race_state['cars_on_lead_lap'],
            race_state['closest_margin'],
            race_state['track_temperature'],
            race_state['lap_traffic_density']
        ]
```

#### Tire Degradation Model
```python
class TireDegradationModel:
    def __init__(self):
        self.degradation_curves = {}
        self.load_historical_deg_data()

    def predict_lap_times(self, tire_age, compound, track_temp, fuel_load):
        """Predict lap times based on tire degradation"""
        base_deg = self.degradation_curves[compound]

        # Adjust for conditions
        temp_factor = 1 + (track_temp - 70) * 0.002
        fuel_factor = 1 - (fuel_load * 0.001)

        deg_rate = base_deg * temp_factor * fuel_factor

        lap_times = []
        for lap in range(tire_age, tire_age + 30):
            time_loss = deg_rate * (lap ** 1.5) / 100
            lap_times.append({
                'lap': lap,
                'time_delta': time_loss,
                'grip_remaining': max(0, 100 - (lap * deg_rate))
            })

        return lap_times

    def optimal_stint_length(self, compound, conditions):
        """Calculate optimal stint length for tire compound"""
        lap_times = self.predict_lap_times(0, compound, **conditions)

        # Find cliff point where deg accelerates
        cliff_lap = self.find_performance_cliff(lap_times)

        # Optimal is typically 85% of cliff
        return int(cliff_lap * 0.85)
```

### Decision Trees

#### Pit Stop Decision Logic
```python
class PitStopDecisionTree:
    def __init__(self):
        self.decision_factors = {
            'tire_deg': 0.3,
            'track_position': 0.25,
            'caution_probability': 0.2,
            'fuel_remaining': 0.15,
            'competitor_strategy': 0.1
        }

    def should_pit(self, race_state):
        """Real-time pit stop decision"""
        score = 0

        # Tire degradation factor
        if race_state['tire_age'] > race_state['optimal_stint']:
            score += self.decision_factors['tire_deg']

        # Track position factor
        if race_state['position'] > race_state['target_position']:
            score += self.decision_factors['track_position']

        # Caution probability
        if race_state['caution_prob_next_5'] > 0.7:
            score -= self.decision_factors['caution_probability']

        # Fuel window
        if race_state['fuel_laps_remaining'] < 5:
            score += self.decision_factors['fuel_remaining']

        # Competitor strategy
        if race_state['leaders_pitting']:
            score += self.decision_factors['competitor_strategy']

        return {
            'decision': 'pit' if score > 0.5 else 'stay_out',
            'confidence': abs(score - 0.5) * 2,
            'factors': self.explain_decision(race_state, score)
        }
```

### Optimization Algorithms

#### Dynamic Programming Strategy
```python
def optimize_race_strategy_dp(race_params):
    """Dynamic programming approach to strategy optimization"""
    laps = race_params['total_laps']
    states = {}  # Memoization

    def min_time_to_finish(lap, tire_age, fuel, stops_remaining):
        """Recursive DP solution"""
        state = (lap, tire_age, fuel, stops_remaining)

        if state in states:
            return states[state]

        if lap == laps:
            return 0

        # Option 1: Continue without stopping
        continue_time = (
            lap_time(tire_age, fuel) +
            min_time_to_finish(lap+1, tire_age+1, fuel-1, stops_remaining)
        )

        # Option 2: Pit stop (if stops available)
        if stops_remaining > 0:
            pit_time = (
                PIT_STOP_TIME +
                min_time_to_finish(lap+1, 0, MAX_FUEL, stops_remaining-1)
            )
            states[state] = min(continue_time, pit_time)
        else:
            states[state] = continue_time

        return states[state]

    return min_time_to_finish(0, 0, race_params['starting_fuel'], 2)
```

#### Genetic Algorithm Optimization
```python
import random

class GeneticStrategyOptimizer:
    def __init__(self, population_size=100):
        self.population_size = population_size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def evolve_strategy(self, race_params, generations=50):
        """Evolve optimal strategy using genetic algorithm"""
        population = self.initialize_population()

        for gen in range(generations):
            # Evaluate fitness
            fitness = [self.evaluate_strategy(s, race_params) for s in population]

            # Selection
            parents = self.tournament_selection(population, fitness)

            # Crossover and mutation
            offspring = self.create_offspring(parents)

            # Replace population
            population = self.survivor_selection(population + offspring, race_params)

        return max(population, key=lambda s: self.evaluate_strategy(s, race_params))

    def evaluate_strategy(self, strategy, race_params):
        """Fitness function for strategy"""
        result = simulate_race_with_strategy(strategy, race_params)
        return 100 - result['finish_position']  # Higher is better
```

### Real-time Adjustments

#### Live Strategy Updates
```python
class LiveStrategyManager:
    def __init__(self, initial_strategy):
        self.base_strategy = initial_strategy
        self.adjustments = []
        self.confidence_threshold = 0.7

    def update_strategy(self, live_data):
        """Adjust strategy based on live race developments"""
        current_lap = live_data['lap']

        # Check for strategy triggers
        triggers = self.check_triggers(live_data)

        for trigger in triggers:
            if trigger['confidence'] > self.confidence_threshold:
                adjustment = self.calculate_adjustment(trigger, live_data)
                self.adjustments.append(adjustment)
                self.base_strategy = self.apply_adjustment(adjustment)

        return {
            'current_strategy': self.base_strategy,
            'next_action': self.get_next_action(live_data),
            'contingencies': self.calculate_contingencies(live_data)
        }

    def check_triggers(self, live_data):
        """Identify strategy change triggers"""
        triggers = []

        # Unexpected caution
        if live_data['yellow_flag'] and not live_data['expected_yellow']:
            triggers.append({
                'type': 'unexpected_caution',
                'confidence': 0.95,
                'action': 'evaluate_pit'
            })

        # Competitor strategy change
        if live_data['leader_pitted_early']:
            triggers.append({
                'type': 'leader_undercut',
                'confidence': 0.8,
                'action': 'cover_strategy'
            })

        return triggers
```

### Visualization Dashboard

```python
import dash
import plotly.graph_objects as go

class StrategyDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def create_strategy_timeline(self, strategies):
        """Visualize pit stop windows and strategy options"""
        fig = go.Figure()

        for i, strategy in enumerate(strategies):
            # Add pit stop markers
            for stop in strategy['stops']:
                fig.add_trace(go.Scatter(
                    x=[stop['lap']],
                    y=[i],
                    mode='markers',
                    marker=dict(size=10, color=stop['tire_compound']),
                    name=f"Strategy {i+1}"
                ))

        return fig

    def create_position_forecast(self, predictions):
        """Show predicted position evolution"""
        fig = go.Figure()

        for scenario in predictions:
            fig.add_trace(go.Scatter(
                x=scenario['laps'],
                y=scenario['positions'],
                mode='lines',
                name=scenario['strategy_name'],
                line=dict(width=2)
            ))

        return fig

    def create_risk_reward_matrix(self, strategies):
        """Risk vs. reward visualization"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[s['risk_score'] for s in strategies],
            y=[s['expected_position'] for s in strategies],
            mode='markers',
            marker=dict(
                size=[s['confidence']*20 for s in strategies],
                color=[s['win_probability'] for s in strategies],
                colorscale='Viridis'
            ),
            text=[s['name'] for s in strategies],
            hovertemplate='%{text}<br>Risk: %{x}<br>Position: %{y}'
        ))

        return fig
```

### Communication Protocols

#### Pit Crew Integration
```python
class PitCrewCommunicator:
    def __init__(self):
        self.message_queue = []
        self.priority_levels = ['critical', 'high', 'normal', 'info']

    def send_pit_alert(self, lap_window, tire_choice, fuel_load):
        """Alert pit crew of upcoming stop"""
        message = {
            'priority': 'high',
            'type': 'pit_preparation',
            'window': lap_window,
            'setup': {
                'tires': tire_choice,
                'fuel': fuel_load,
                'adjustments': self.calculate_adjustments()
            },
            'timing': f"Box in {lap_window[0] - current_lap} laps"
        }
        self.broadcast(message)

    def send_strategy_update(self, change_type, details):
        """Communicate strategy changes"""
        message = {
            'priority': 'critical' if change_type == 'immediate' else 'normal',
            'type': 'strategy_change',
            'change': change_type,
            'details': details,
            'confirmation_required': True
        }
        self.broadcast(message)
```

### Output Formats

#### Strategy Report
```yaml
optimal_strategy:
  type: "2-stop"
  confidence: 0.85
  expected_position: 3.2

  pit_stops:
    - lap: 25
      tire: "soft"
      fuel: "full"
      time_loss: 22.5
      position_after: 8

    - lap: 50
      tire: "medium"
      fuel: "to_finish"
      time_loss: 18.2
      position_after: 5

  key_phases:
    opening: "Push hard on fresh softs"
    middle: "Manage gap, prepare undercut"
    closing: "Attack for podium position"

  risks:
    - "Early caution could compromise strategy"
    - "Tire deg higher than expected"
    - "Fuel saving may be required"

  contingencies:
    caution_lap_20:
      action: "Pit immediately for mediums"
    safety_car_lap_45:
      action: "Stay out, short final stint"
```

### Best Practices

1. **Continuous Calibration**: Update models with real-time data
2. **Risk Management**: Always have contingency plans
3. **Clear Communication**: Ensure driver understands strategy
4. **Flexibility**: Be ready to adapt to race developments
5. **Data Validation**: Verify sensor data before decisions

### Related Skills
- telemetry-analyzer
- driver-performance-analyzer
- tire-degradation-predictor
- real-time-dashboard-builder