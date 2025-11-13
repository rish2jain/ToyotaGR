"""
Pit Stop Detection Module for RaceIQ Pro

This module provides sophisticated pit stop detection using multiple signal analysis
techniques including rolling median analysis, z-score anomaly detection, and gap analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PitStopDetector:
    """
    Advanced pit stop detector using multi-signal analysis and voting mechanism.

    Combines multiple detection methods:
    - Rolling median anomaly detection (slow-slower-slow pattern)
    - Z-score based statistical outlier detection
    - Gap to leader analysis
    - Voting mechanism with configurable confidence threshold
    """

    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.6):
        """
        Initialize the pit stop detector.

        Args:
            window_size: Rolling window size for median calculation (default: 5)
            confidence_threshold: Minimum voting confidence to flag pit stop (default: 0.6)
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

    def _parse_lap_time(self, lap_time: str) -> float:
        """
        Convert lap time from MM:SS.SSS format to seconds.

        Args:
            lap_time: Lap time string in MM:SS.SSS format

        Returns:
            Lap time in seconds as float
        """
        if pd.isna(lap_time) or lap_time == '':
            return np.nan

        try:
            if isinstance(lap_time, (int, float)):
                return float(lap_time)

            # Handle MM:SS.SSS format
            parts = str(lap_time).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(lap_time)
        except:
            return np.nan

    def _calculate_rolling_median_anomaly(self, lap_times: pd.Series,
                                         sensitivity: float = 2.5) -> np.ndarray:
        """
        Detect anomalies using rolling median with sensitivity threshold.

        Args:
            lap_times: Series of lap times in seconds
            sensitivity: Threshold in seconds above rolling median (default: 2.5)

        Returns:
            Binary array indicating anomalies
        """
        rolling_median = lap_times.rolling(window=self.window_size,
                                          center=True,
                                          min_periods=2).median()

        # Calculate deviation from rolling median
        deviation = lap_times - rolling_median

        # Flag laps that are >sensitivity seconds slower than rolling median
        anomalies = (deviation > sensitivity).astype(int)

        return anomalies.values

    def _calculate_z_score_anomaly(self, lap_times: pd.Series,
                                   threshold: float = 2.0) -> np.ndarray:
        """
        Detect anomalies using z-score statistical method.

        Args:
            lap_times: Series of lap times in seconds
            threshold: Z-score threshold for flagging anomalies (default: 2.0)

        Returns:
            Binary array indicating anomalies
        """
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(lap_times, nan_policy='omit'))

        # Flag laps with z-score above threshold
        anomalies = (z_scores > threshold).astype(int)

        return anomalies

    def _calculate_gap_anomaly(self, lap_data: pd.DataFrame,
                              gap_threshold: float = 5.0) -> np.ndarray:
        """
        Detect pit stops by analyzing gap increases to leader.

        Args:
            lap_data: DataFrame with lap data including 'gap_to_leader' if available
            gap_threshold: Minimum gap increase in seconds to flag (default: 5.0)

        Returns:
            Binary array indicating anomalies
        """
        # If gap data is not available, return zeros
        if 'gap_to_leader' not in lap_data.columns and 'ELAPSED' not in lap_data.columns:
            return np.zeros(len(lap_data))

        # Calculate gap changes
        if 'gap_to_leader' in lap_data.columns:
            gaps = lap_data['gap_to_leader'].values
        else:
            # Estimate gap from elapsed time differences
            gaps = np.zeros(len(lap_data))

        # Calculate gap increase from previous lap
        gap_increase = np.diff(gaps, prepend=gaps[0])

        # Flag large gap increases
        anomalies = (gap_increase > gap_threshold).astype(int)

        return anomalies

    def _voting_mechanism(self, signals: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine multiple detection signals using voting.

        Args:
            signals: List of binary arrays from different detection methods

        Returns:
            Tuple of (pit_stop_flags, confidence_scores)
        """
        # Stack signals and calculate vote percentage
        signal_matrix = np.column_stack(signals)
        confidence = signal_matrix.mean(axis=1)

        # Flag pit stops where confidence exceeds threshold
        pit_stops = (confidence >= self.confidence_threshold).astype(int)

        return pit_stops, confidence

    def detect_pit_stops(self, lap_time_data: pd.DataFrame,
                        sensitivity: float = 2.5) -> pd.DataFrame:
        """
        Detect pit stops using multi-signal analysis and voting.

        Args:
            lap_time_data: DataFrame with lap time data. Expected columns:
                          - 'LAP_NUMBER' or 'lap': Lap number
                          - 'LAP_TIME' or 'lap_time': Lap time (MM:SS.SSS or seconds)
                          - Optional: 'gap_to_leader', 'ELAPSED'
            sensitivity: Sensitivity threshold for rolling median detection (default: 2.5)

        Returns:
            DataFrame with pit stop detections including:
            - lap_number: Lap number
            - lap_time: Lap time in seconds
            - is_pit_stop: Binary flag (1 = pit stop detected)
            - confidence: Confidence score (0-1)
            - method_votes: Dictionary of individual method votes
        """
        # Normalize column names
        df = lap_time_data.copy()

        if 'LAP_NUMBER' in df.columns:
            df['lap_number'] = df['LAP_NUMBER']
        elif 'lap' in df.columns:
            df['lap_number'] = df['lap']

        if 'LAP_TIME' in df.columns:
            df['lap_time_raw'] = df['LAP_TIME']
        elif 'lap_time' in df.columns:
            df['lap_time_raw'] = df['lap_time']

        # Convert lap times to seconds
        df['lap_time'] = df['lap_time_raw'].apply(self._parse_lap_time)

        # Remove invalid lap times
        df = df[df['lap_time'].notna()].reset_index(drop=True)

        if len(df) < 3:
            # Not enough data for analysis
            df['is_pit_stop'] = 0
            df['confidence'] = 0.0
            return df[['lap_number', 'lap_time', 'is_pit_stop', 'confidence']]

        # Apply detection methods
        signals = []
        method_names = []

        # Method 1: Rolling median anomaly
        rolling_anomaly = self._calculate_rolling_median_anomaly(df['lap_time'], sensitivity)
        signals.append(rolling_anomaly)
        method_names.append('rolling_median')

        # Method 2: Z-score anomaly
        zscore_anomaly = self._calculate_z_score_anomaly(df['lap_time'])
        signals.append(zscore_anomaly)
        method_names.append('zscore')

        # Method 3: Gap analysis (if data available)
        gap_anomaly = self._calculate_gap_anomaly(df)
        if gap_anomaly.sum() > 0:  # Only include if it found something
            signals.append(gap_anomaly)
            method_names.append('gap_analysis')

        # Combine signals with voting
        pit_stops, confidence = self._voting_mechanism(signals)

        # Add results to dataframe
        df['is_pit_stop'] = pit_stops
        df['confidence'] = confidence

        # Add individual method votes
        for i, method in enumerate(method_names):
            df[f'vote_{method}'] = signals[i]

        return df

    def refine_detections(self, pit_stops: pd.DataFrame,
                         race_data: Optional[pd.DataFrame] = None,
                         exclude_edge_laps: int = 2) -> pd.DataFrame:
        """
        Refine pit stop detections using race context and heuristics.

        Args:
            pit_stops: DataFrame from detect_pit_stops()
            race_data: Optional additional race context data
            exclude_edge_laps: Number of laps to exclude from start/end (default: 2)

        Returns:
            Refined DataFrame with cleaned pit stop detections
        """
        df = pit_stops.copy()

        # Rule 1: Remove first and last N laps (unlikely to be pit stops)
        min_lap = df['lap_number'].min() + exclude_edge_laps
        max_lap = df['lap_number'].max() - exclude_edge_laps

        df.loc[(df['lap_number'] < min_lap) | (df['lap_number'] > max_lap),
               'is_pit_stop'] = 0
        df.loc[(df['lap_number'] < min_lap) | (df['lap_number'] > max_lap),
               'confidence'] = 0.0

        # Rule 2: Merge consecutive detections
        # If multiple consecutive laps are flagged, keep only the slowest one
        pit_stop_laps = df[df['is_pit_stop'] == 1]['lap_number'].values

        if len(pit_stop_laps) > 1:
            # Find consecutive sequences
            consecutive_groups = []
            current_group = [pit_stop_laps[0]]

            for i in range(1, len(pit_stop_laps)):
                if pit_stop_laps[i] == pit_stop_laps[i-1] + 1:
                    current_group.append(pit_stop_laps[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [pit_stop_laps[i]]

            if len(current_group) > 1:
                consecutive_groups.append(current_group)

            # For each consecutive group, keep only the lap with highest confidence
            for group in consecutive_groups:
                group_df = df[df['lap_number'].isin(group)]
                max_conf_lap = group_df.loc[group_df['confidence'].idxmax(), 'lap_number']

                # Clear other laps in group
                for lap in group:
                    if lap != max_conf_lap:
                        df.loc[df['lap_number'] == lap, 'is_pit_stop'] = 0
                        df.loc[df['lap_number'] == lap, 'confidence'] = 0.0

        # Rule 3: Validate minimum pit stop time
        # Pit stops typically add 30-60 seconds, so lap should be significantly slower
        median_lap_time = df['lap_time'].median()
        min_pit_lap_time = median_lap_time + 10  # At least 10 seconds slower

        df.loc[(df['is_pit_stop'] == 1) & (df['lap_time'] < min_pit_lap_time),
               'is_pit_stop'] = 0
        df.loc[(df['is_pit_stop'] == 1) & (df['lap_time'] < min_pit_lap_time),
               'confidence'] = 0.0

        # Add refinement metadata
        df['refined'] = True

        return df

    def get_pit_stop_summary(self, refined_detections: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for detected pit stops.

        Args:
            refined_detections: DataFrame from refine_detections()

        Returns:
            Dictionary with pit stop summary statistics
        """
        pit_stops = refined_detections[refined_detections['is_pit_stop'] == 1]

        summary = {
            'total_pit_stops': len(pit_stops),
            'pit_stop_laps': pit_stops['lap_number'].tolist(),
            'average_confidence': pit_stops['confidence'].mean() if len(pit_stops) > 0 else 0.0,
            'pit_stop_times': pit_stops['lap_time'].tolist(),
            'average_pit_lap_time': pit_stops['lap_time'].mean() if len(pit_stops) > 0 else 0.0,
            'normal_lap_median': refined_detections[
                refined_detections['is_pit_stop'] == 0
            ]['lap_time'].median(),
            'estimated_pit_time_loss': []
        }

        # Calculate estimated time lost in pit
        if len(pit_stops) > 0 and summary['normal_lap_median'] is not None:
            summary['estimated_pit_time_loss'] = [
                lap_time - summary['normal_lap_median']
                for lap_time in pit_stops['lap_time']
            ]

        return summary
