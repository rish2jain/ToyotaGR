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

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMAnomalyDetector:
    """
    LSTM-based anomaly detector for time-series telemetry patterns.

    Uses an LSTM autoencoder to learn normal telemetry patterns and detect
    anomalies based on reconstruction error. High reconstruction error indicates
    that the pattern is anomalous compared to normal driving behavior.

    Features used:
    - Speed (KPH or similar)
    - ThrottlePosition (aps)
    - BrakePressure (pbrake_f)
    - SteeringAngle
    - RPM (nmot)
    - Gear
    """

    def __init__(self, sequence_length: int = 50, verbose: int = 0):
        """
        Initialize the LSTM anomaly detector.

        Args:
            sequence_length: Number of time steps in each sequence (default: 50)
            verbose: Verbosity level for model training (0=silent, 1=progress bar, 2=one line per epoch)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTM anomaly detection. "
                "Install it with: pip install tensorflow"
            )

        self.sequence_length = sequence_length
        self.verbose = verbose
        self.model = None
        self.feature_scaler = None
        self.threshold = None
        self.feature_columns = ['Speed', 'ThrottlePosition', 'BrakePressure',
                               'SteeringAngle', 'RPM', 'Gear']

    def _build_model(self, n_features: int):
        """
        Build LSTM autoencoder model for anomaly detection.

        Architecture:
        - LSTM encoder (64 units, returns sequences)
        - LSTM encoder (32 units, compresses to latent representation)
        - Dense bottleneck (32 units with dropout)
        - Dense decoder (reconstructs n_features)

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True,
                 input_shape=(self.sequence_length, n_features)),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(n_features)  # Reconstruct input features
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Create sliding window sequences for LSTM input.

        Args:
            data: 2D array of shape (n_samples, n_features)

        Returns:
            Tuple of (sequences, original_indices)
            - sequences: 3D array of shape (n_sequences, sequence_length, n_features)
            - original_indices: List of indices mapping sequences to original data points
        """
        sequences = []
        indices = []

        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
            indices.append(i + self.sequence_length - 1)  # Index of last element in sequence

        return np.array(sequences), indices

    def _filter_normal_laps(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only "normal" laps for training.

        Uses bottom 80% of lap times as "normal" to avoid training on outliers.

        Args:
            telemetry_data: DataFrame with lap times

        Returns:
            Filtered DataFrame containing only normal laps
        """
        if 'LAP_NUMBER' not in telemetry_data.columns:
            # If no lap numbers, use all data
            return telemetry_data

        # Calculate lap times if available
        if 'lap_seconds' in telemetry_data.columns:
            lap_times = telemetry_data.groupby('LAP_NUMBER')['lap_seconds'].mean()
            threshold_time = lap_times.quantile(0.8)  # Bottom 80%
            normal_laps = lap_times[lap_times <= threshold_time].index
            return telemetry_data[telemetry_data['LAP_NUMBER'].isin(normal_laps)]

        # Default: return all data
        return telemetry_data

    def _prepare_features(self, telemetry_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare and normalize telemetry features for LSTM.

        Maps available columns to expected feature names and handles missing features.

        Args:
            telemetry_data: Raw telemetry DataFrame

        Returns:
            Tuple of (prepared_features_df, available_feature_names)
        """
        # Map common column names to expected features
        column_mapping = {
            'KPH': 'Speed',
            'MPH': 'Speed',
            'SPEED': 'Speed',
            'aps': 'ThrottlePosition',
            'THROTTLE': 'ThrottlePosition',
            'pbrake_f': 'BrakePressure',
            'BRAKE': 'BrakePressure',
            'Steering_Angle': 'SteeringAngle',
            'STEERING': 'SteeringAngle',
            'nmot': 'RPM',
            'RPM': 'RPM',
            'gear': 'Gear',
            'GEAR': 'Gear'
        }

        prepared_df = pd.DataFrame()
        available_features = []

        for original_col, target_col in column_mapping.items():
            if original_col in telemetry_data.columns:
                if target_col not in prepared_df.columns:  # Avoid duplicates
                    prepared_df[target_col] = telemetry_data[original_col]
                    available_features.append(target_col)

        # Fill missing values with forward fill then backward fill
        prepared_df = prepared_df.fillna(method='ffill').fillna(method='bfill')

        # If still have NaN, fill with 0
        prepared_df = prepared_df.fillna(0)

        return prepared_df, available_features

    def detect_pattern_anomalies(
        self,
        telemetry_data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        contamination: float = 0.05
    ) -> pd.DataFrame:
        """
        Train LSTM autoencoder on normal laps and detect anomalies by reconstruction error.

        Process:
        1. Filter normal laps (bottom 80% of lap times)
        2. Prepare and normalize features
        3. Create sequences for LSTM
        4. Train autoencoder on normal data
        5. Calculate reconstruction errors for all data
        6. Flag anomalies based on threshold

        Args:
            telemetry_data: DataFrame containing telemetry data
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size for training (default: 32)
            contamination: Expected proportion of anomalies (default: 0.05)

        Returns:
            DataFrame with additional columns:
            - lstm_reconstruction_error: MSE reconstruction error
            - lstm_is_anomaly: Boolean flag for anomalies
            - lstm_anomaly_score: Normalized anomaly score (0-1)

        Raises:
            ValueError: If insufficient data or features
            ImportError: If TensorFlow is not installed
        """
        if telemetry_data.empty:
            raise ValueError("Telemetry data is empty")

        if len(telemetry_data) < self.sequence_length * 2:
            raise ValueError(
                f"Insufficient data: need at least {self.sequence_length * 2} samples, "
                f"got {len(telemetry_data)}"
            )

        result_df = telemetry_data.copy()

        # Prepare features
        features_df, available_features = self._prepare_features(telemetry_data)

        if len(available_features) == 0:
            raise ValueError(
                f"No matching features found. Available columns: {telemetry_data.columns.tolist()}"
            )

        # Normalize features (0-1 scaling)
        from sklearn.preprocessing import MinMaxScaler
        self.feature_scaler = MinMaxScaler()
        features_normalized = self.feature_scaler.fit_transform(features_df[available_features])

        # Create sequences
        sequences, sequence_indices = self._create_sequences(features_normalized)

        if len(sequences) < 10:
            raise ValueError(
                f"Insufficient sequences: need at least 10, got {len(sequences)}"
            )

        # Filter normal data for training
        normal_data = self._filter_normal_laps(telemetry_data)

        if len(normal_data) < len(telemetry_data) * 0.5:
            # If filtering removes too much data, use all data
            normal_mask = np.ones(len(sequences), dtype=bool)
        else:
            # Create mask for normal sequences
            normal_indices = normal_data.index
            normal_mask = np.array([
                telemetry_data.index[idx] in normal_indices
                for idx in sequence_indices
            ])

        normal_sequences = sequences[normal_mask]

        if len(normal_sequences) < 5:
            # Not enough normal data, use all sequences
            normal_sequences = sequences

        # Build and train model
        n_features = len(available_features)
        self.model = self._build_model(n_features)

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )

        # Train on normal data
        self.model.fit(
            normal_sequences,
            normal_sequences[:, -1, :],  # Predict last timestep
            epochs=epochs,
            batch_size=batch_size,
            verbose=self.verbose,
            callbacks=[early_stop],
            validation_split=0.2
        )

        # Calculate reconstruction errors for all sequences
        predictions = self.model.predict(sequences, verbose=0)
        actual = sequences[:, -1, :]  # Last timestep of each sequence

        # Mean squared error per sequence
        reconstruction_errors = np.mean(np.square(actual - predictions), axis=1)

        # Determine threshold (based on contamination parameter)
        self.threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)

        # Flag anomalies
        is_anomaly = reconstruction_errors > self.threshold

        # Normalize scores to 0-1 range
        max_error = reconstruction_errors.max()
        min_error = reconstruction_errors.min()
        if max_error > min_error:
            anomaly_scores = (reconstruction_errors - min_error) / (max_error - min_error)
        else:
            anomaly_scores = np.zeros_like(reconstruction_errors)

        # Map back to original dataframe indices
        # Initialize columns
        result_df['lstm_reconstruction_error'] = np.nan
        result_df['lstm_is_anomaly'] = False
        result_df['lstm_anomaly_score'] = 0.0

        # Fill in values for sequences
        for seq_idx, orig_idx in enumerate(sequence_indices):
            result_df.iloc[orig_idx, result_df.columns.get_loc('lstm_reconstruction_error')] = reconstruction_errors[seq_idx]
            result_df.iloc[orig_idx, result_df.columns.get_loc('lstm_is_anomaly')] = is_anomaly[seq_idx]
            result_df.iloc[orig_idx, result_df.columns.get_loc('lstm_anomaly_score')] = anomaly_scores[seq_idx]

        # Forward fill NaN values for rows that weren't in sequences
        result_df['lstm_reconstruction_error'] = result_df['lstm_reconstruction_error'].fillna(method='bfill')
        result_df['lstm_anomaly_score'] = result_df['lstm_anomaly_score'].fillna(method='bfill')

        return result_df


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

    def detect_lstm_anomalies(
        self,
        telemetry_data: pd.DataFrame,
        sequence_length: int = 50,
        epochs: int = 50,
        contamination: float = 0.05
    ) -> pd.DataFrame:
        """
        Wrapper method to use LSTM-based anomaly detection (Tier 3 Deep Learning).

        This method uses an LSTM autoencoder to detect complex temporal patterns
        and anomalies in telemetry data that statistical and ML methods might miss.

        Args:
            telemetry_data: DataFrame containing telemetry data
            sequence_length: Number of time steps in each sequence (default: 50)
            epochs: Number of training epochs (default: 50)
            contamination: Expected proportion of anomalies (default: 0.05)

        Returns:
            DataFrame with LSTM anomaly detection results:
            - lstm_reconstruction_error: MSE reconstruction error
            - lstm_is_anomaly: Boolean flag for anomalies
            - lstm_anomaly_score: Normalized anomaly score (0-1)

        Raises:
            ImportError: If TensorFlow is not installed
            ValueError: If insufficient data or features

        Example:
            >>> detector = AnomalyDetector()
            >>> result = detector.detect_lstm_anomalies(telemetry_df, sequence_length=50, epochs=30)
            >>> lstm_anomalies = result[result['lstm_is_anomaly']]
            >>> print(f"Found {len(lstm_anomalies)} LSTM anomalies")
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTM anomaly detection. "
                "Install it with: pip install tensorflow"
            )

        lstm_detector = LSTMAnomalyDetector(
            sequence_length=sequence_length,
            verbose=0
        )

        return lstm_detector.detect_pattern_anomalies(
            telemetry_data,
            epochs=epochs,
            contamination=contamination
        )
