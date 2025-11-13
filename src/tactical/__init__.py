"""
Tactical Analysis Module for RaceIQ Pro

This module provides comprehensive tactical racing analysis including:
- Optimal ghost lap creation and driver comparison
- Multi-tier anomaly detection for telemetry data
- Section-by-section performance analysis and insights
- Racing line reconstruction and comparison

Modules:
    optimal_ghost: Create optimal ghost laps and analyze driver performance gaps
    anomaly_detector: Detect anomalies in telemetry using statistical and ML methods
    section_analyzer: Analyze track sections and identify strengths/weaknesses
    racing_line: Reconstruct and compare racing lines from telemetry data
"""

from .optimal_ghost import OptimalGhostAnalyzer
from .anomaly_detector import AnomalyDetector
from .section_analyzer import SectionAnalyzer
from .racing_line import RacingLineReconstructor

__all__ = [
    'OptimalGhostAnalyzer',
    'AnomalyDetector',
    'SectionAnalyzer',
    'RacingLineReconstructor',
]

__version__ = '1.0.0'
