"""
RaceIQ Pro - Configuration Constants

This module contains all configuration constants and settings for the RaceIQ Pro platform.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "Data"
BARBER_DATA_DIR = DATA_DIR / "barber"
BARBER_SAMPLES_DIR = BARBER_DATA_DIR / "Samples"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Track configurations
TRACKS = {
    "barber": "Barber Motorsports Park",
    "COTA": "Circuit of the Americas",
    "sonoma": "Sonoma Raceway",
    "indianapolis": "Indianapolis Motor Speedway",
    "road-america": "Road America",
    "sebring": "Sebring International Raceway",
}

# Data file patterns
FILE_PATTERNS = {
    "lap_time": "*_lap_time*.csv",
    "lap_start": "*_lap_start*.csv",
    "lap_end": "*_lap_end*.csv",
    "telemetry": "*_telemetry*.csv",
    "section_analysis": "23_AnalysisEnduranceWithSections*.CSV",
    "race_results": "03_*Results*.CSV",
    "weather": "26_Weather*.CSV",
    "best_laps": "99_Best*.CSV",
}

# Data schema definitions
LAP_TIME_COLUMNS = [
    "expire_at", "lap", "meta_event", "meta_session", "meta_source",
    "meta_time", "original_vehicle_id", "outing", "timestamp",
    "vehicle_id", "vehicle_number"
]

TELEMETRY_COLUMNS = [
    "gps_alt", "gps_heading", "gps_lat", "gps_long", "gps_nsat", "gps_quality",
    "gps_speed", "lap", "meta_event", "meta_session", "meta_source", "meta_time",
    "timestamp", "vehicle_id", "vehicle_number"
]

SECTION_ANALYSIS_COLUMNS = [
    "NUMBER", "DRIVER_NUMBER", "LAP_NUMBER", "LAP_TIME", "LAP_IMPROVEMENT",
    "CROSSING_FINISH_LINE_IN_PIT", "S1", "S1_IMPROVEMENT", "S2", "S2_IMPROVEMENT",
    "S3", "S3_IMPROVEMENT", "KPH", "ELAPSED", "HOUR", "S1_LARGE", "S2_LARGE",
    "S3_LARGE", "TOP_SPEED", "PIT_TIME", "CLASS", "GROUP", "MANUFACTURER",
    "FLAG_AT_FL", "S1_SECONDS", "S2_SECONDS", "S3_SECONDS"
]

RACE_RESULTS_COLUMNS = [
    "POSITION", "NUMBER", "STATUS", "LAPS", "TOTAL_TIME", "GAP_FIRST",
    "GAP_PREVIOUS", "FL_LAPNUM", "FL_TIME", "FL_KPH", "CLASS", "GROUP",
    "DIVISION", "VEHICLE", "TIRES"
]

# Performance thresholds
THRESHOLDS = {
    "min_speed": 10.0,  # km/h - minimum valid speed
    "max_speed": 250.0,  # km/h - maximum valid speed
    "min_lap_time": 60.0,  # seconds - minimum valid lap time
    "max_lap_time": 300.0,  # seconds - maximum valid lap time
    "consistency_window": 3,  # Number of laps for consistency calculation
    "anomaly_std_threshold": 3.0,  # Standard deviations for anomaly detection
}

# Feature engineering settings
FEATURE_CONFIG = {
    "rolling_window": 5,  # Rolling window for moving averages
    "percentile_bins": [0.25, 0.5, 0.75, 0.9, 0.95],  # Percentiles for analysis
    "delta_laps": [1, 3, 5],  # Lap deltas for comparison
}

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Visualization settings
VIZ_CONFIG = {
    "default_figsize": (12, 6),
    "color_scheme": {
        "primary": "#E60012",  # Toyota Red
        "secondary": "#0033A0",  # Toyota Blue
        "accent": "#FFB81C",  # Toyota Gold
        "success": "#00A650",  # Green
        "warning": "#FF6600",  # Orange
        "danger": "#CC0000",  # Red
        "neutral": "#6C757D",  # Gray
    },
    "line_width": 2,
    "marker_size": 8,
    "dpi": 100,
}

# Database settings (for future use)
DB_CONFIG = {
    "type": "sqlite",
    "path": PROCESSED_DATA_DIR / "raceiq.db",
}

# API settings (for future use)
API_CONFIG = {
    "host": os.getenv("API_HOST", "localhost"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("API_DEBUG", "False").lower() == "true",
}
