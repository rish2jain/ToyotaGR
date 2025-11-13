# Driver Performance Analyzer
## AI-powered coaching insights for GR Cup Series drivers

### Overview
This skill focuses on analyzing individual driver performance, identifying strengths and weaknesses, and providing personalized coaching recommendations. It uses machine learning to understand driving patterns, compare against optimal performance benchmarks, and generate actionable improvement strategies.

### Core Capabilities

#### 1. Performance Profiling
- **Driving Style Classification**: Categorize drivers into style archetypes
- **Skill Level Assessment**: Quantify expertise across different aspects
- **Consistency Analysis**: Measure repeatability and reliability
- **Adaptability Scoring**: Evaluate performance under varying conditions

#### 2. Comparative Analysis
- **Peer Benchmarking**: Compare against similar skill-level drivers
- **Elite Comparison**: Gap analysis with top performers
- **Historical Progression**: Track improvement over time
- **Team Comparison**: Intra-team performance evaluation

#### 3. Weakness Identification
- **Technical Gaps**: Specific skill deficiencies
- **Mental Patterns**: Psychological performance factors
- **Physical Limitations**: Endurance and reaction time issues
- **Equipment Misuse**: Suboptimal vehicle utilization

#### 4. Coaching Recommendations
- **Prioritized Improvements**: Highest-impact areas first
- **Personalized Training Plans**: Custom exercises and drills
- **Mental Coaching**: Focus and consistency techniques
- **Progress Tracking**: Measurable improvement metrics

### Implementation Architecture

```python
class DriverPerformanceAnalyzer:
    def __init__(self, driver_id):
        self.driver_id = driver_id
        self.performance_model = self.load_model()
        self.baseline_metrics = {}
        self.improvement_tracking = []

    def analyze_session(self, session_data):
        """Complete performance analysis for a session"""
        metrics = self.extract_metrics(session_data)
        style = self.classify_driving_style(metrics)
        weaknesses = self.identify_weaknesses(metrics)
        recommendations = self.generate_recommendations(weaknesses)
        return {
            'metrics': metrics,
            'style': style,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        }

    def extract_metrics(self, session_data):
        """Extract key performance indicators"""
        return {
            'corner_entry_speed': self.analyze_corner_entry(session_data),
            'brake_efficiency': self.analyze_braking(session_data),
            'throttle_control': self.analyze_throttle(session_data),
            'racing_line': self.analyze_line(session_data),
            'consistency': self.calculate_consistency(session_data)
        }

    def generate_training_plan(self, analysis_results):
        """Create personalized training program"""
        # Weakness prioritization
        # Exercise selection
        # Progress milestones
        pass
```

### Performance Metrics Framework

#### Cornering Performance
```yaml
corner_metrics:
  entry_phase:
    - brake_point_consistency
    - trail_braking_efficiency
    - turn_in_timing
    - entry_speed_optimization

  apex_phase:
    - minimum_speed_achieved
    - apex_hitting_accuracy
    - throttle_pickup_point
    - line_precision

  exit_phase:
    - acceleration_timing
    - traction_utilization
    - track_out_usage
    - exit_speed_maximization
```

#### Longitudinal Performance
```yaml
straight_line_metrics:
  acceleration:
    - throttle_application_rate
    - wheel_spin_management
    - shift_timing_accuracy
    - draft_utilization

  braking:
    - brake_pressure_buildup
    - peak_pressure_timing
    - release_smoothness
    - lock_up_frequency
```

#### Racecraft Metrics
```yaml
racing_metrics:
  overtaking:
    - attempt_success_rate
    - risk_assessment
    - opportunity_recognition
    - execution_quality

  defending:
    - position_retention
    - line_protection
    - counter_move_timing
    - pressure_handling
```

### Machine Learning Models

#### Driving Style Classifier
```python
from sklearn.ensemble import RandomForestClassifier

class DrivingStyleClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.style_categories = [
            'smooth_efficient',
            'aggressive_attacking',
            'conservative_consistent',
            'adaptive_strategic'
        ]

    def classify(self, driver_metrics):
        # Feature engineering
        features = self.extract_style_features(driver_metrics)
        # Prediction
        style = self.model.predict(features)[0]
        confidence = self.model.predict_proba(features).max()
        return {
            'style': style,
            'confidence': confidence,
            'characteristics': self.get_style_traits(style)
        }
```

#### Performance Predictor
```python
import tensorflow as tf

class LapTimePredictor:
    def __init__(self):
        self.model = self.build_neural_network()

    def build_neural_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Lap time prediction
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_improvement(self, current_metrics, proposed_changes):
        # Simulate performance with changes
        modified_metrics = self.apply_changes(current_metrics, proposed_changes)
        current_time = self.model.predict(current_metrics)
        improved_time = self.model.predict(modified_metrics)
        return {
            'time_gain': current_time - improved_time,
            'confidence_interval': self.calculate_confidence()
        }
```

### Coaching Algorithm

#### Weakness Priority Scoring
```python
def prioritize_improvements(weaknesses, driver_profile):
    """Rank improvements by potential impact"""
    scores = []

    for weakness in weaknesses:
        impact = calculate_time_impact(weakness)
        difficulty = estimate_learning_curve(weakness, driver_profile)
        current_gap = measure_performance_gap(weakness)

        priority_score = (impact * current_gap) / difficulty
        scores.append({
            'area': weakness,
            'score': priority_score,
            'estimated_gain': impact,
            'training_weeks': difficulty
        })

    return sorted(scores, key=lambda x: x['score'], reverse=True)
```

#### Personalized Recommendations
```python
def generate_coaching_plan(driver_analysis):
    """Create actionable coaching recommendations"""
    plan = {
        'immediate_focus': [],  # Top 3 quick wins
        'technical_development': [],  # Skill building
        'mental_training': [],  # Consistency and focus
        'physical_conditioning': []  # Fitness and reactions
    }

    # Immediate improvements (can be addressed this weekend)
    for improvement in driver_analysis['quick_wins'][:3]:
        plan['immediate_focus'].append({
            'area': improvement['area'],
            'specific_action': improvement['action'],
            'expected_gain': improvement['time_gain'],
            'practice_drill': improvement['drill']
        })

    return plan
```

### Visualization Components

#### Performance Spider Chart
```python
import plotly.graph_objects as go

def create_performance_spider(driver_metrics):
    """Create radar chart of driver capabilities"""
    categories = ['Braking', 'Cornering', 'Acceleration',
                  'Consistency', 'Racecraft', 'Tire Management']

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=driver_metrics,
        theta=categories,
        fill='toself',
        name='Driver Performance'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[85, 85, 85, 85, 85, 85],  # Target performance
        theta=categories,
        fill='toself',
        name='Target Level'
    ))

    return fig
```

#### Progress Tracking Dashboard
```python
def create_progress_dashboard(driver_id, date_range):
    """Interactive dashboard showing improvement over time"""
    dashboard_components = {
        'lap_time_trend': plot_lap_time_evolution(),
        'consistency_chart': plot_consistency_improvement(),
        'weakness_heatmap': create_weakness_heatmap(),
        'training_compliance': show_training_adherence(),
        'peer_comparison': plot_peer_rankings()
    }
    return dashboard_components
```

### Training Drill Generator

```python
class TrainingDrillGenerator:
    def __init__(self):
        self.drill_database = self.load_drill_library()

    def generate_session_plan(self, weaknesses, session_duration=60):
        """Create practice session plan"""
        session = {
            'warmup': self.select_warmup_drills(10),
            'main_focus': [],
            'cooldown': self.select_cooldown(5)
        }

        remaining_time = session_duration - 15  # After warmup/cooldown

        for weakness in weaknesses[:3]:  # Top 3 priorities
            drill = self.select_drill(weakness)
            drill['duration'] = remaining_time // 3
            drill['repetitions'] = self.calculate_reps(drill, weakness)
            drill['success_criteria'] = self.define_criteria(weakness)
            session['main_focus'].append(drill)

        return session

    def select_drill(self, weakness):
        """Choose appropriate drill for specific weakness"""
        drills = self.drill_database[weakness['category']]
        # Select based on driver level and weakness severity
        return self.match_drill_to_driver(drills, weakness)
```

### Integration with Race Data

#### Real-time Coaching Feedback
```python
def generate_realtime_feedback(current_lap, reference_lap):
    """Provide immediate coaching during practice"""
    feedback_queue = []

    # Analyze current sectors
    for sector in current_lap.sectors:
        delta = sector.time - reference_lap.sectors[sector.id].time

        if delta > 0.1:  # More than 0.1s slower
            feedback = analyze_sector_loss(sector, reference_lap)
            feedback_queue.append({
                'priority': calculate_priority(delta),
                'message': generate_coaching_message(feedback),
                'timing': 'next_straight'  # When to deliver
            })

    return feedback_queue
```

### Output Formats

#### Driver Report Card
```markdown
# Driver Performance Report
## Session: Sonoma Raceway - Practice 2
## Date: November 15, 2024

### Overall Performance Grade: B+ (82/100)

### Strengths
- Excellent trail braking technique
- Consistent lap times (σ = 0.15s)
- Strong defensive positioning

### Areas for Improvement
1. **Corner Exit Speed** (-0.3s/lap)
   - Early throttle application needed
   - Focus on turns 3, 7, and 11

2. **Tire Management** (-0.5s over stint)
   - Excessive sliding in early laps
   - Recommend smoother inputs

### Personalized Training Plan
Week 1: Throttle Application Drills
- Exercise: Progressive throttle on corner exit
- Duration: 3x 10-minute sessions
- Success Metric: Exit speed +2 mph average

Week 2: Tire Temperature Management
- Exercise: Consistent pace laps
- Duration: 20-lap runs
- Success Metric: Temp variation <5°F
```

### Best Practices

1. **Data Privacy**: Protect individual driver data and team strategies
2. **Positive Reinforcement**: Balance criticism with strengths recognition
3. **Actionable Feedback**: Ensure all recommendations are implementable
4. **Progressive Difficulty**: Scale training with driver improvement
5. **Holistic Approach**: Consider physical, mental, and technical aspects

### Related Skills
- telemetry-analyzer
- race-strategy-optimizer
- tire-degradation-predictor
- real-time-dashboard-builder