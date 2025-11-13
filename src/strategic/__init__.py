"""
Strategic Analysis Module for RaceIQ Pro

This module provides comprehensive race strategy analysis including:
- Pit stop detection using multi-signal analysis
- Tire degradation modeling with predictive analytics
- Pit strategy optimization using Monte Carlo simulation
- Undercut/overcut opportunity analysis
- Multi-driver race simulation with position dynamics

Components:
    - PitStopDetector: Detect pit stops from lap time data
    - TireDegradationModel: Model and predict tire performance
    - PitStrategyOptimizer: Optimize pit strategy decisions
    - MultiDriverRaceSimulator: Simulate multi-car races with strategy interactions

Example Usage:
    >>> from strategic import PitStopDetector, TireDegradationModel, PitStrategyOptimizer
    >>> from strategic import MultiDriverRaceSimulator
    >>> import pandas as pd
    >>>
    >>> # Load race data
    >>> lap_data = pd.read_csv('lap_times.csv')
    >>>
    >>> # Detect pit stops
    >>> detector = PitStopDetector()
    >>> detections = detector.detect_pit_stops(lap_data)
    >>> refined = detector.refine_detections(detections)
    >>>
    >>> # Model tire degradation
    >>> tire_model = TireDegradationModel(model_type='polynomial', degree=2)
    >>> degradation = tire_model.estimate_degradation(lap_data)
    >>> cliff_prediction = tire_model.predict_cliff_point(lap_data)
    >>>
    >>> # Optimize pit strategy
    >>> optimizer = PitStrategyOptimizer(pit_loss_seconds=25.0)
    >>> optimal_strategy = optimizer.calculate_optimal_pit_window(
    ...     lap_data, degradation, race_length=25
    ... )
    >>> undercut = optimizer.simulate_undercut_opportunity(lap_data)
    >>>
    >>> # Simulate multi-driver race
    >>> simulator = MultiDriverRaceSimulator(race_length=25)
    >>> drivers = {
    ...     'A': {'name': 'Driver A', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05},
    ...     'B': {'name': 'Driver B', 'base_lap_time': 95.2, 'tire_deg_rate': 0.05}
    ... }
    >>> strategies = {'A': {'pit_laps': [12]}, 'B': {'pit_laps': [14]}}
    >>> result = simulator.simulate_race(drivers, strategies)
"""

from .pit_detector import PitStopDetector
from .tire_degradation import TireDegradationModel
from .strategy_optimizer import PitStrategyOptimizer
from .race_simulation import MultiDriverRaceSimulator

__all__ = [
    'PitStopDetector',
    'TireDegradationModel',
    'PitStrategyOptimizer',
    'MultiDriverRaceSimulator'
]

__version__ = '1.0.0'
__author__ = 'RaceIQ Pro Team'
