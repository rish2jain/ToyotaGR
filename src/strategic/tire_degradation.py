"""
Tire Degradation Modeling Module for RaceIQ Pro

This module provides advanced tire degradation analysis using curve fitting,
performance prediction, and tire cliff detection algorithms.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.interpolate import UnivariateSpline
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class TireDegradationModel:
    """
    Advanced tire degradation model using polynomial/exponential curve fitting
    and predictive analytics for tire performance analysis.

    Features:
    - Multiple curve fitting approaches (polynomial, exponential, spline)
    - Degradation rate calculation
    - Remaining tire performance estimation
    - Tire cliff prediction using second derivative analysis
    - Corner speed analysis as leading indicator
    """

    def __init__(self, model_type: str = 'polynomial', degree: int = 2):
        """
        Initialize the tire degradation model.

        Args:
            model_type: Type of curve fit ('polynomial', 'exponential', or 'spline')
            degree: Polynomial degree for fitting (default: 2 for quadratic)
        """
        self.model_type = model_type
        self.degree = degree
        self.fitted_params = None
        self.baseline_performance = None

    def _parse_lap_time(self, lap_time: str) -> float:
        """Convert lap time from MM:SS.SSS format to seconds."""
        if pd.isna(lap_time) or lap_time == '':
            return np.nan

        try:
            if isinstance(lap_time, (int, float)):
                return float(lap_time)

            parts = str(lap_time).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(lap_time)
        except:
            return np.nan

    def _exponential_model(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Exponential degradation model: y = a + b * exp(c * x)

        Args:
            x: Lap numbers
            a: Baseline lap time
            b: Degradation amplitude
            c: Degradation rate

        Returns:
            Predicted lap times
        """
        return a + b * np.exp(c * x)

    def _fit_polynomial(self, laps: np.ndarray, lap_times: np.ndarray) -> np.ndarray:
        """Fit polynomial model to lap time data."""
        coeffs = np.polyfit(laps, lap_times, self.degree)
        self.fitted_params = coeffs
        return coeffs

    def _fit_exponential(self, laps: np.ndarray, lap_times: np.ndarray) -> np.ndarray:
        """Fit exponential model to lap time data."""
        try:
            # Initial guess for parameters
            a0 = np.min(lap_times)
            b0 = np.max(lap_times) - a0
            c0 = 0.01

            popt, _ = optimize.curve_fit(
                self._exponential_model,
                laps,
                lap_times,
                p0=[a0, b0, c0],
                maxfev=5000
            )
            self.fitted_params = popt
            return popt
        except:
            # Fall back to polynomial if exponential fit fails
            return self._fit_polynomial(laps, lap_times)

    def _fit_spline(self, laps: np.ndarray, lap_times: np.ndarray) -> UnivariateSpline:
        """Fit spline model to lap time data."""
        # Use smoothing spline to avoid overfitting
        spline = UnivariateSpline(laps, lap_times, s=len(laps)*0.1, k=3)
        self.fitted_params = spline
        return spline

    def _remove_outliers(self, lap_data: pd.DataFrame,
                        z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outlier laps (likely pit stops or incidents) from analysis.

        Args:
            lap_data: DataFrame with lap data
            z_threshold: Z-score threshold for outlier removal

        Returns:
            Cleaned DataFrame
        """
        df = lap_data.copy()

        if 'lap_time' in df.columns and len(df) > 5:
            z_scores = np.abs(stats.zscore(df['lap_time'], nan_policy='omit'))
            df = df[z_scores < z_threshold]

        return df

    def estimate_degradation(self, lap_data: pd.DataFrame,
                           exclude_outliers: bool = True) -> Dict:
        """
        Estimate tire degradation from lap time data.

        Args:
            lap_data: DataFrame with lap data. Expected columns:
                     - 'LAP_NUMBER' or 'lap_number': Lap number
                     - 'LAP_TIME' or 'lap_time': Lap time
                     - Optional: 'S1', 'S2', 'S3' for section times
                     - Optional: 'TOP_SPEED' or 'KPH' for speed analysis
            exclude_outliers: Whether to remove outlier laps (default: True)

        Returns:
            Dictionary with degradation analysis:
            - degradation_rate: Seconds per lap degradation
            - baseline_lap_time: Estimated fresh tire lap time
            - current_performance_pct: Current tire performance as percentage
            - fitted_curve: Fitted model parameters
            - prediction: Predicted lap times
            - r_squared: Model fit quality
        """
        # Normalize column names
        df = lap_data.copy()

        if 'LAP_NUMBER' in df.columns:
            df['lap_number'] = df['LAP_NUMBER']
        elif 'lap_number' not in df.columns:
            df['lap_number'] = range(1, len(df) + 1)

        if 'LAP_TIME' in df.columns:
            df['lap_time_raw'] = df['LAP_TIME']
        elif 'lap_time' not in df.columns and 'lap_time_raw' in df.columns:
            pass
        else:
            df['lap_time_raw'] = df.get('lap_time', np.nan)

        # Convert lap times to seconds
        df['lap_time'] = df['lap_time_raw'].apply(self._parse_lap_time)

        # Remove invalid data
        df = df[df['lap_time'].notna()].reset_index(drop=True)

        if len(df) < 3:
            return {
                'degradation_rate': 0.0,
                'baseline_lap_time': 0.0,
                'current_performance_pct': 100.0,
                'error': 'Insufficient data for degradation analysis'
            }

        # Remove outliers (pit stops, incidents)
        if exclude_outliers:
            df = self._remove_outliers(df)

        laps = df['lap_number'].values
        lap_times = df['lap_time'].values

        # Fit model
        if self.model_type == 'polynomial':
            coeffs = self._fit_polynomial(laps, lap_times)
            predictions = np.polyval(coeffs, laps)
            # Degradation rate is the linear coefficient (first derivative at mid-point)
            mid_lap = np.mean(laps)
            degradation_rate = np.polyval(np.polyder(coeffs), mid_lap)

        elif self.model_type == 'exponential':
            params = self._fit_exponential(laps, lap_times)
            if len(params) == 3:  # Exponential fit succeeded
                predictions = self._exponential_model(laps, *params)
                # Degradation rate from exponential model
                a, b, c = params
                mid_lap = np.mean(laps)
                degradation_rate = b * c * np.exp(c * mid_lap)
            else:  # Fell back to polynomial
                predictions = np.polyval(params, laps)
                mid_lap = np.mean(laps)
                degradation_rate = np.polyval(np.polyder(params), mid_lap)

        else:  # spline
            spline = self._fit_spline(laps, lap_times)
            predictions = spline(laps)
            # Degradation rate from spline derivative
            mid_lap = np.mean(laps)
            degradation_rate = spline.derivative()(mid_lap)

        # Calculate R-squared
        ss_res = np.sum((lap_times - predictions) ** 2)
        ss_tot = np.sum((lap_times - np.mean(lap_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Estimate baseline performance (lap 1 or minimum predicted)
        if self.model_type == 'polynomial':
            baseline_lap_time = np.polyval(coeffs, 1)
        elif self.model_type == 'exponential' and len(params) == 3:
            baseline_lap_time = self._exponential_model(1, *params)
        else:
            baseline_lap_time = predictions[0]

        self.baseline_performance = baseline_lap_time

        # Current performance percentage (inverse since higher lap time = worse)
        current_lap_time = lap_times[-1]
        if baseline_lap_time > 0:
            current_performance_pct = (baseline_lap_time / current_lap_time) * 100
        else:
            current_performance_pct = 100.0

        # Ensure percentage is reasonable
        current_performance_pct = max(0, min(100, current_performance_pct))

        return {
            'degradation_rate': float(degradation_rate),
            'baseline_lap_time': float(baseline_lap_time),
            'current_lap_time': float(current_lap_time),
            'current_performance_pct': float(current_performance_pct),
            'fitted_curve': self.fitted_params,
            'predictions': predictions.tolist(),
            'actual_times': lap_times.tolist(),
            'lap_numbers': laps.tolist(),
            'r_squared': float(r_squared),
            'model_type': self.model_type
        }

    def predict_cliff_point(self, lap_data: pd.DataFrame,
                           acceleration_threshold: float = 0.05) -> Dict:
        """
        Predict the "tire cliff" - the lap where degradation accelerates sharply.

        Uses second derivative analysis to detect when degradation is accelerating.

        Args:
            lap_data: DataFrame with lap data
            acceleration_threshold: Threshold for acceleration detection (default: 0.05)

        Returns:
            Dictionary with cliff prediction:
            - cliff_lap: Predicted lap number for tire cliff
            - cliff_confidence: Confidence in prediction (0-1)
            - acceleration_profile: Second derivative values
            - warning_laps: Laps before predicted cliff
        """
        # First estimate degradation
        degradation = self.estimate_degradation(lap_data)

        if 'error' in degradation:
            return {
                'cliff_lap': None,
                'cliff_confidence': 0.0,
                'warning': degradation['error']
            }

        laps = np.array(degradation['lap_numbers'])
        predictions = np.array(degradation['predictions'])

        if len(laps) < 5:
            return {
                'cliff_lap': None,
                'cliff_confidence': 0.0,
                'warning': 'Insufficient data for cliff prediction'
            }

        # Calculate first and second derivatives
        # First derivative: rate of change (degradation rate)
        first_derivative = np.gradient(predictions, laps)

        # Second derivative: acceleration of degradation
        second_derivative = np.gradient(first_derivative, laps)

        # Find where second derivative exceeds threshold (acceleration)
        accelerating = second_derivative > acceleration_threshold

        cliff_lap = None
        cliff_confidence = 0.0

        if np.any(accelerating):
            # Find first lap where acceleration is detected
            cliff_lap_idx = np.where(accelerating)[0][0]
            cliff_lap = int(laps[cliff_lap_idx])

            # Confidence based on magnitude of acceleration
            max_acceleration = np.max(second_derivative[accelerating])
            cliff_confidence = min(1.0, max_acceleration / (acceleration_threshold * 5))

        else:
            # Extrapolate to find potential future cliff
            # Fit second derivative trend
            if len(second_derivative) > 3:
                try:
                    # Linear extrapolation of second derivative
                    trend_coeffs = np.polyfit(laps, second_derivative, 1)
                    # Find where it crosses threshold
                    if trend_coeffs[0] > 0:  # Increasing trend
                        cliff_lap = int((acceleration_threshold - trend_coeffs[1]) / trend_coeffs[0])
                        cliff_confidence = 0.3  # Lower confidence for extrapolation

                        # Ensure cliff lap is in reasonable range
                        max_lap = int(laps[-1] * 2)
                        cliff_lap = min(cliff_lap, max_lap)
                except:
                    pass

        current_lap = int(laps[-1])
        warning_laps = cliff_lap - current_lap if cliff_lap else None

        return {
            'cliff_lap': cliff_lap,
            'current_lap': current_lap,
            'cliff_confidence': float(cliff_confidence),
            'warning_laps': warning_laps,
            'acceleration_profile': second_derivative.tolist(),
            'degradation_rate_profile': first_derivative.tolist(),
            'status': self._get_cliff_status(warning_laps, cliff_confidence)
        }

    def _get_cliff_status(self, warning_laps: Optional[int],
                         confidence: float) -> str:
        """Generate human-readable status message for tire cliff."""
        if warning_laps is None:
            return 'No cliff detected - tires stable'
        elif warning_laps < 0:
            return 'Tire cliff already reached - immediate action needed'
        elif warning_laps <= 2 and confidence > 0.6:
            return 'CRITICAL: Tire cliff imminent (1-2 laps)'
        elif warning_laps <= 5 and confidence > 0.5:
            return 'WARNING: Tire cliff approaching (3-5 laps)'
        elif warning_laps <= 10:
            return 'Caution: Potential cliff in 6-10 laps'
        else:
            return f'Tires stable - {warning_laps} laps to predicted cliff'

    def analyze_corner_speed_degradation(self, lap_data: pd.DataFrame) -> Dict:
        """
        Analyze corner speed as a leading indicator of tire degradation.

        Corner speed typically drops before lap time degradation becomes severe.

        Args:
            lap_data: DataFrame with section or speed data

        Returns:
            Dictionary with corner speed analysis
        """
        df = lap_data.copy()

        # Check for available speed/section data
        speed_cols = []
        if 'S1_SECONDS' in df.columns:
            speed_cols.append('S1_SECONDS')
        if 'S2_SECONDS' in df.columns:
            speed_cols.append('S2_SECONDS')
        if 'S3_SECONDS' in df.columns:
            speed_cols.append('S3_SECONDS')

        if not speed_cols:
            return {
                'warning': 'No corner/section speed data available',
                'corner_degradation_detected': False
            }

        # Analyze each section
        section_analysis = {}

        for col in speed_cols:
            section_times = df[col].values
            laps = np.arange(len(section_times))

            # Fit linear trend
            if len(section_times) > 3:
                coeffs = np.polyfit(laps, section_times, 1)
                trend = coeffs[0]  # Positive = getting slower

                # Calculate degradation rate
                baseline = section_times[0] if len(section_times) > 0 else 0
                current = section_times[-1] if len(section_times) > 0 else 0

                pct_degradation = ((current - baseline) / baseline * 100) if baseline > 0 else 0

                section_analysis[col] = {
                    'trend': float(trend),
                    'pct_degradation': float(pct_degradation),
                    'baseline': float(baseline),
                    'current': float(current)
                }

        # Determine overall corner degradation status
        avg_degradation = np.mean([s['pct_degradation'] for s in section_analysis.values()])

        return {
            'section_analysis': section_analysis,
            'average_corner_degradation_pct': float(avg_degradation),
            'corner_degradation_detected': avg_degradation > 2.0,
            'severity': 'HIGH' if avg_degradation > 5 else 'MEDIUM' if avg_degradation > 2 else 'LOW'
        }
