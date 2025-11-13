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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class AnomalyDetector:
    """
    Multi-tier anomaly detector for racing telemetry and performance data.

    Tier 1: Statistical methods using rolling z-scores
    Tier 2: Machine learning methods using Isolation Forest
    """

    def __init__(self):
        """Initialize the AnomalyDetector."""
        self.anomalies_detected: List[Dict[str, Any]] = []
        self.isolation_forest_model = None
        self.feature_names = None
        self.shap_explainer = None

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

        # Store model and features for SHAP explanations
        self.isolation_forest_model = iso_forest
        self.feature_names = features

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

    def explain_anomaly(
        self,
        anomaly_data: pd.Series,
        telemetry_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for a single anomaly.

        Uses SHAP TreeExplainer to identify which features contributed most
        to the anomaly detection and provides human-readable explanations.

        Args:
            anomaly_data: Single row of telemetry data (as Series) with anomaly
            telemetry_features: List of feature names to explain. If None, uses stored features.

        Returns:
            Dictionary containing:
            - top_features: List of dicts with feature name, contribution, and direction
            - explanation: Human-readable explanation string
            - shap_values: Array of SHAP values for all features
            - confidence: Confidence score based on anomaly score

        Raises:
            ImportError: If SHAP is not installed
            ValueError: If model hasn't been trained yet

        Example:
            >>> detector = AnomalyDetector()
            >>> result_df = detector.detect_pattern_anomalies(telemetry_df)
            >>> anomaly_row = result_df[result_df['is_anomaly'] == -1].iloc[0]
            >>> explanation = detector.explain_anomaly(anomaly_row)
            >>> print(explanation['explanation'])
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required for anomaly explanations. "
                "Install it with: pip install shap"
            )

        if self.isolation_forest_model is None:
            raise ValueError(
                "No trained model found. Run detect_pattern_anomalies() first."
            )

        # Use stored feature names if not provided
        if telemetry_features is None:
            telemetry_features = self.feature_names

        if telemetry_features is None:
            raise ValueError("No features available for explanation")

        # Extract feature values for this anomaly
        try:
            feature_values = anomaly_data[telemetry_features].values.reshape(1, -1)
        except KeyError as e:
            raise ValueError(f"Missing features in anomaly data: {e}")

        # Handle missing values
        feature_df = pd.DataFrame(feature_values, columns=telemetry_features)
        feature_df = feature_df.fillna(feature_df.median())

        # Create SHAP explainer if not already created
        if self.shap_explainer is None:
            try:
                # Use TreeExplainer for Isolation Forest
                self.shap_explainer = shap.TreeExplainer(self.isolation_forest_model)
            except Exception as e:
                # Fallback to KernelExplainer if TreeExplainer fails
                import warnings
                warnings.warn(
                    f"TreeExplainer failed ({e}), using KernelExplainer (slower)"
                )
                # Sample background data (use first 100 samples or less)
                background_size = min(100, len(feature_df))
                background = shap.sample(feature_df, background_size)
                self.shap_explainer = shap.KernelExplainer(
                    self.isolation_forest_model.decision_function,
                    background
                )

        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(feature_df)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        # Get feature importance ranking
        feature_importance = []
        for i, feature_name in enumerate(telemetry_features):
            shap_val = shap_values[i]
            feature_val = feature_df[feature_name].iloc[0]

            # Determine direction (high/low/normal)
            if abs(shap_val) < 0.01:
                direction = 'normal'
            elif shap_val > 0:
                direction = 'high'
            else:
                direction = 'low'

            feature_importance.append({
                'feature': feature_name,
                'contribution': abs(shap_val),
                'shap_value': float(shap_val),
                'feature_value': float(feature_val),
                'direction': direction
            })

        # Sort by contribution (absolute SHAP value)
        feature_importance.sort(key=lambda x: x['contribution'], reverse=True)

        # Normalize contributions to percentages
        total_contribution = sum(f['contribution'] for f in feature_importance)
        if total_contribution > 0:
            for f in feature_importance:
                f['contribution'] = (f['contribution'] / total_contribution)

        # Generate human-readable explanation
        top_3 = feature_importance[:3]
        explanation_parts = []

        for f in top_3:
            feature_clean = f['feature'].replace('_', ' ').title()
            contrib_pct = f['contribution'] * 100

            if f['direction'] == 'normal':
                continue
            elif f['direction'] == 'high':
                explanation_parts.append(
                    f"{feature_clean} {contrib_pct:.0f}% too high"
                )
            else:
                explanation_parts.append(
                    f"{feature_clean} {contrib_pct:.0f}% too low"
                )

        explanation = ", ".join(explanation_parts) if explanation_parts else "Anomalous pattern detected"

        # Calculate confidence from anomaly score
        anomaly_score = anomaly_data.get('anomaly_score', -1)
        # Convert anomaly score to confidence (lower score = higher confidence)
        # Typical scores range from -0.5 to 0.5
        confidence = max(0.0, min(1.0, 1.0 - (anomaly_score + 0.5)))

        return {
            'top_features': feature_importance,
            'explanation': explanation,
            'shap_values': shap_values,
            'confidence': float(confidence)
        }

    def get_anomaly_explanations(
        self,
        anomalies_df: pd.DataFrame,
        telemetry_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate SHAP-based explanations for all detected anomalies.

        Args:
            anomalies_df: DataFrame with anomaly flags (from detect_pattern_anomalies)
            telemetry_data: Optional full telemetry data. If None, uses anomalies_df.

        Returns:
            DataFrame with original anomaly data plus explanation columns:
            - explanation: Human-readable explanation
            - top_feature_1, top_feature_2, top_feature_3: Top contributing features
            - contribution_1, contribution_2, contribution_3: Contribution percentages
            - confidence: Explanation confidence score

        Example:
            >>> detector = AnomalyDetector()
            >>> result_df = detector.detect_pattern_anomalies(telemetry_df)
            >>> anomalies = result_df[result_df['is_anomaly'] == -1]
            >>> explained = detector.get_anomaly_explanations(anomalies)
            >>> print(explained[['LAP_NUMBER', 'explanation', 'confidence']])
        """
        if not SHAP_AVAILABLE:
            import warnings
            warnings.warn(
                "SHAP is not installed. Returning original dataframe without explanations. "
                "Install with: pip install shap"
            )
            return anomalies_df.copy()

        if self.isolation_forest_model is None:
            import warnings
            warnings.warn(
                "No trained model found. Run detect_pattern_anomalies() first. "
                "Returning original dataframe without explanations."
            )
            return anomalies_df.copy()

        # Use telemetry_data if provided, otherwise use anomalies_df
        data_to_use = telemetry_data if telemetry_data is not None else anomalies_df

        # Create result dataframe
        result_df = anomalies_df.copy()

        # Initialize explanation columns
        result_df['explanation'] = ''
        result_df['top_feature_1'] = ''
        result_df['top_feature_2'] = ''
        result_df['top_feature_3'] = ''
        result_df['contribution_1'] = 0.0
        result_df['contribution_2'] = 0.0
        result_df['contribution_3'] = 0.0
        result_df['confidence'] = 0.0

        # Generate explanations for each anomaly
        for idx in result_df.index:
            try:
                anomaly_row = data_to_use.loc[idx]
                explanation = self.explain_anomaly(anomaly_row)

                result_df.at[idx, 'explanation'] = explanation['explanation']
                result_df.at[idx, 'confidence'] = explanation['confidence']

                # Add top 3 features
                top_features = explanation['top_features'][:3]
                for i, feature_info in enumerate(top_features, 1):
                    result_df.at[idx, f'top_feature_{i}'] = feature_info['feature']
                    result_df.at[idx, f'contribution_{i}'] = feature_info['contribution']

            except Exception as e:
                import warnings
                warnings.warn(f"Failed to explain anomaly at index {idx}: {e}")
                result_df.at[idx, 'explanation'] = 'Explanation unavailable'
                result_df.at[idx, 'confidence'] = 0.0

        return result_df
