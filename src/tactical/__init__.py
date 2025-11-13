"""
Tactical Analysis Module for RaceIQ Pro

This module provides comprehensive tactical racing analysis including:
- Optimal ghost lap creation and driver comparison
- Multi-tier anomaly detection for telemetry data
- Section-by-section performance analysis and insights

Modules:
    optimal_ghost: Create optimal ghost laps and analyze driver performance gaps
    anomaly_detector: Detect anomalies in telemetry using statistical and ML methods
    section_analyzer: Analyze track sections and identify strengths/weaknesses
"""

from .optimal_ghost import OptimalGhostAnalyzer
from .anomaly_detector import AnomalyDetector
from .section_analyzer import SectionAnalyzer

__all__ = [
    'OptimalGhostAnalyzer',
    'AnomalyDetector',
    'SectionAnalyzer',
]

__version__ = '1.0.0'
