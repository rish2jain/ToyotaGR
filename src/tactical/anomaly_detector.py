"""
Anomaly Detection Module

This module provides multi-tier anomaly detection for racing telemetry data,
including statistical methods and machine learning approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnomalyDetector:
    """
    Multi-tier anomaly detector for racing telemetry and performance data.

    Tier 1: Statistical methods using rolling z-scores
    Tier 2: Machine learning methods using Isolation Forest
    """

    def __init__(self):
        """Initialize the AnomalyDetector."""
        self.anomalies_detected: List[Dict[str, Any]] = []

    def detect_statistical_anomalies(
        self,
        telemetry_data: pd.DataFrame,
        window: int = 5,
        threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Detect anomalies using rolling z-scores (Tier 1 statistical baseline).

        This method calculates rolling mean and standard deviation for telemetry
        metrics and flags values that exceed the threshold number of standard
        deviations from the rolling mean.

        Args:
            telemetry_data: DataFrame containing telemetry data with columns like:
                - LAP_NUMBER or time-based index
                - Telemetry columns (e.g., S1_SECONDS, S2_SECONDS, S3_SECONDS,
                  TOP_SPEED, or other numeric metrics)
                - DRIVER_NUMBER (optional, for per-driver analysis)
            window: Size of rolling window for calculating statistics (default: 5 laps)
            threshold: Number of standard deviations to flag as anomaly (default: 2.5)

        Returns:
            DataFrame with original data plus anomaly flags and z-scores:
            - {column}_zscore: Z-score for each metric
            - {column}_anomaly: Boolean flag indicating anomaly
            - anomaly_count: Total number of anomalies per row

        Example:
            >>> detector = AnomalyDetector()
            >>> anomalies = detector.detect_statistical_anomalies(telemetry_df, window=5, threshold=2.5)
            >>> print(f"Found {anomalies['anomaly_count'].sum()} anomalies")
        """
        if telemetry_data.empty:
            raise ValueError("Telemetry data is empty")

        # Create a copy to avoid modifying original
        result_df = telemetry_data.copy()

        # Identify numeric columns to analyze (exclude identifiers and times)
        exclude_cols = ['DRIVER_NUMBER', 'LAP_NUMBER', 'NUMBER', 'ELAPSED', 'HOUR']
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        analyze_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(analyze_cols) == 0:
            raise ValueError("No numeric columns found for anomaly detection")

        # Check if we should analyze per driver
        per_driver = 'DRIVER_NUMBER' in result_df.columns

        anomaly_count_col = []

        if per_driver:
            # Analyze each driver separately
            for driver_num in result_df['DRIVER_NUMBER'].unique():
                driver_mask = result_df['DRIVER_NUMBER'] == driver_num
                driver_data = result_df[driver_mask].copy()

                for col in analyze_cols:
                    if col not in driver_data.columns or driver_data[col].isna().all():
                        continue

                    # Calculate rolling statistics
                    rolling_mean = driver_data[col].rolling(window=window, min_periods=1).mean()
                    rolling_std = driver_data[col].rolling(window=window, min_periods=1).std()

                    # Calculate z-score
                    z_scores = np.abs((driver_data[col] - rolling_mean) / (rolling_std + 1e-10))

                    # Flag anomalies
                    anomalies = z_scores > threshold

                    # Store results back in main dataframe
                    result_df.loc[driver_mask, f'{col}_zscore'] = z_scores
                    result_df.loc[driver_mask, f'{col}_anomaly'] = anomalies

        else:
            # Analyze all data together
            for col in analyze_cols:
                if result_df[col].isna().all():
                    continue

                # Calculate rolling statistics
                rolling_mean = result_df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = result_df[col].rolling(window=window, min_periods=1).std()

                # Calculate z-score
                z_scores = np.abs((result_df[col] - rolling_mean) / (rolling_std + 1e-10))

                # Flag anomalies
                anomalies = z_scores > threshold

                # Store results
                result_df[f'{col}_zscore'] = z_scores
                result_df[f'{col}_anomaly'] = anomalies

        # Count total anomalies per row
        anomaly_cols = [col for col in result_df.columns if col.endswith('_anomaly')]
        result_df['anomaly_count'] = result_df[anomaly_cols].sum(axis=1)

        # Store detected anomalies
        self.anomalies_detected = result_df[result_df['anomaly_count'] > 0].to_dict('records')

        return result_df

    def detect_pattern_anomalies(
        self,
        telemetry_data: pd.DataFrame,
        contamination: float = 0.05,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest (Tier 2 machine learning).

        This method uses scikit-learn's Isolation Forest algorithm to detect
        complex patterns and multivariate anomalies in the data.

        Args:
            telemetry_data: DataFrame containing telemetry data
            contamination: Expected proportion of anomalies (default: 0.05 = 5%)
            features: List of column names to use as features. If None, uses all numeric columns.

        Returns:
            DataFrame with additional columns:
            - anomaly_score: Anomaly score from Isolation Forest (lower = more anomalous)
            - is_anomaly: Boolean flag (-1 = anomaly, 1 = normal)

        Raises:
            ImportError: If scikit-learn is not installed

        Example:
            >>> detector = AnomalyDetector()
            >>> anomalies = detector.detect_pattern_anomalies(telemetry_df, contamination=0.05)
            >>> anomalous_laps = anomalies[anomalies['is_anomaly'] == -1]
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for pattern anomaly detection. "
                "Install it with: pip install scikit-learn"
            )

        if telemetry_data.empty:
            raise ValueError("Telemetry data is empty")

        result_df = telemetry_data.copy()

        # Select features
        if features is None:
            exclude_cols = ['DRIVER_NUMBER', 'LAP_NUMBER', 'NUMBER', 'ELAPSED', 'HOUR']
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in exclude_cols]

        if len(features) == 0:
            raise ValueError("No features available for pattern anomaly detection")

        # Prepare feature matrix
        X = result_df[features].copy()

        # Handle missing values by filling with column median
        X = X.fillna(X.median())

        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = iso_forest.fit_predict(X)

        # Get anomaly scores (lower = more anomalous)
        scores = iso_forest.score_samples(X)

        # Add results to dataframe
        result_df['is_anomaly'] = predictions
        result_df['anomaly_score'] = scores

        return result_df

    def classify_anomaly_type(
        self,
        anomaly_data: pd.DataFrame,
        row_index: Optional[int] = None
    ) -> Union[str, pd.Series]:
        """
        Classify the type of anomaly based on which metrics are flagged.

        Categories:
        - 'brake_issue': Anomalies in brake-related metrics
        - 'throttle_issue': Anomalies in throttle/acceleration metrics
        - 'speed_anomaly': Anomalies in speed metrics
        - 'section_time_anomaly': Anomalies in section times
        - 'driver_error': Multiple anomalies suggesting driver mistake
        - 'unknown': Cannot determine specific type

        Args:
            anomaly_data: DataFrame with anomaly flags (from detect_statistical_anomalies)
            row_index: Specific row to classify. If None, classifies all rows.

        Returns:
            If row_index is provided: string with anomaly type
            If row_index is None: Series with anomaly type for each row

        Example:
            >>> detector = AnomalyDetector()
            >>> df_with_anomalies = detector.detect_statistical_anomalies(telemetry_df)
            >>> df_with_anomalies['anomaly_type'] = detector.classify_anomaly_type(df_with_anomalies)
        """
        if row_index is not None:
            return self._classify_single_row(anomaly_data, row_index)

        # Classify all rows
        return anomaly_data.apply(
            lambda row: self._classify_single_row(anomaly_data, row.name),
            axis=1
        )

    def _classify_single_row(self, anomaly_data: pd.DataFrame, row_index: int) -> str:
        """
        Classify anomaly type for a single row.

        Args:
            anomaly_data: DataFrame with anomaly flags
            row_index: Index of the row to classify

        Returns:
            String with anomaly type
        """
        row = anomaly_data.loc[row_index]

        # Get all anomaly flag columns
        anomaly_cols = [col for col in anomaly_data.columns if col.endswith('_anomaly')]

        if len(anomaly_cols) == 0 or 'anomaly_count' not in row:
            return 'unknown'

        # If no anomalies, return 'none'
        if row.get('anomaly_count', 0) == 0:
            return 'none'

        # Check which specific metrics are anomalous
        flagged_metrics = [col.replace('_anomaly', '') for col in anomaly_cols if row.get(col, False)]

        if not flagged_metrics:
            return 'unknown'

        # Classification logic based on metric patterns
        brake_keywords = ['brake', 'BRAKE']
        throttle_keywords = ['throttle', 'THROTTLE', 'acceleration', 'ACCEL']
        speed_keywords = ['speed', 'SPEED', 'KPH', 'MPH', 'TOP_SPEED']
        section_keywords = ['S1', 'S2', 'S3', 'SECTION', 'IM']

        brake_flags = sum(1 for m in flagged_metrics if any(k in m for k in brake_keywords))
        throttle_flags = sum(1 for m in flagged_metrics if any(k in m for k in throttle_keywords))
        speed_flags = sum(1 for m in flagged_metrics if any(k in m for k in speed_keywords))
        section_flags = sum(1 for m in flagged_metrics if any(k in m for k in section_keywords))

        # Determine primary anomaly type
        if brake_flags > 0 and brake_flags >= throttle_flags:
            return 'brake_issue'
        elif throttle_flags > 0:
            return 'throttle_issue'
        elif speed_flags > 0:
            return 'speed_anomaly'
        elif section_flags > 0:
            return 'section_time_anomaly'
        elif len(flagged_metrics) >= 3:
            return 'driver_error'
        else:
            return 'unknown'

    def get_anomaly_summary(
        self,
        anomaly_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a summary of detected anomalies.

        Args:
            anomaly_data: DataFrame with anomaly flags and classifications

        Returns:
            Dictionary with anomaly statistics and breakdown by type
        """
        if 'anomaly_count' not in anomaly_data.columns:
            raise ValueError("Data does not contain anomaly_count column. Run detection first.")

        total_rows = len(anomaly_data)
        anomalous_rows = (anomaly_data['anomaly_count'] > 0).sum()

        summary = {
            'total_samples': total_rows,
            'anomalous_samples': int(anomalous_rows),
            'anomaly_rate': float(anomalous_rows / total_rows) if total_rows > 0 else 0.0
        }

        # If anomaly types are classified
        if 'anomaly_type' in anomaly_data.columns:
            type_counts = anomaly_data[anomaly_data['anomaly_count'] > 0]['anomaly_type'].value_counts()
            summary['anomaly_types'] = type_counts.to_dict()

        # Get most anomalous metrics
        anomaly_cols = [col for col in anomaly_data.columns if col.endswith('_anomaly')]
        metric_anomaly_counts = {}
        for col in anomaly_cols:
            metric_name = col.replace('_anomaly', '')
            count = anomaly_data[col].sum()
            if count > 0:
                metric_anomaly_counts[metric_name] = int(count)

        summary['anomalies_by_metric'] = dict(sorted(
            metric_anomaly_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return summary

    def filter_high_priority_anomalies(
        self,
        anomaly_data: pd.DataFrame,
        min_anomaly_count: int = 2
    ) -> pd.DataFrame:
        """
        Filter to show only high-priority anomalies.

        Args:
            anomaly_data: DataFrame with anomaly flags
            min_anomaly_count: Minimum number of anomalous metrics to be considered high priority

        Returns:
            Filtered DataFrame with only high-priority anomalies
        """
        if 'anomaly_count' not in anomaly_data.columns:
            raise ValueError("Data does not contain anomaly_count column. Run detection first.")

        return anomaly_data[anomaly_data['anomaly_count'] >= min_anomaly_count].copy()
