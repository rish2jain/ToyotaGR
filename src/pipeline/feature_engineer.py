"""
RaceIQ Pro - Feature Engineering

This module creates derived metrics and features from raw race data including
speed deltas, consistency scores, and performance indicators.
"""

import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

from ..utils.constants import FEATURE_CONFIG, THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates derived features and metrics from race data.
    """

    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.config = FEATURE_CONFIG

    def engineer_lap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from lap time data.

        Args:
            df: DataFrame with lap time data

        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Engineering lap time features...")
        df = df.copy()

        if "vehicle_number" not in df.columns or "lap" not in df.columns:
            logger.warning("Missing required columns for lap features")
            return df

        # Calculate lap duration (time between laps)
        if "timestamp" in df.columns:
            df["lap_duration"] = df.groupby("vehicle_number")["timestamp"].diff()
            df["lap_duration_seconds"] = df["lap_duration"].dt.total_seconds()

        # Rolling statistics for consistency
        window = self.config["rolling_window"]
        if "lap_duration_seconds" in df.columns:
            df["lap_time_rolling_mean"] = df.groupby("vehicle_number")[
                "lap_duration_seconds"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

            df["lap_time_rolling_std"] = df.groupby("vehicle_number")[
                "lap_duration_seconds"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).std())

            # Consistency score (lower std = more consistent)
            df["consistency_score"] = 1 / (1 + df["lap_time_rolling_std"])

        # Delta to previous lap
        if "lap_duration_seconds" in df.columns:
            df["delta_prev_lap"] = df.groupby("vehicle_number")[
                "lap_duration_seconds"
            ].diff()

        # Delta to personal best
        if "lap_duration_seconds" in df.columns:
            df["personal_best"] = df.groupby("vehicle_number")[
                "lap_duration_seconds"
            ].transform("min")
            df["delta_to_best"] = df["lap_duration_seconds"] - df["personal_best"]

        # Lap number features
        df["lap_group"] = pd.cut(
            df["lap"],
            bins=[0, 5, 10, 15, 20, 100],
            labels=["early", "mid_early", "mid", "mid_late", "late"]
        )

        logger.info(f"Created lap features for {len(df)} records")
        return df

    def engineer_section_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from section analysis data.

        Args:
            df: DataFrame with section analysis data

        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Engineering section analysis features...")
        df = df.copy()

        section_cols = ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]

        if not all(col in df.columns for col in section_cols):
            logger.warning("Missing section time columns")
            return df

        # Section performance relative to personal best
        for section in section_cols:
            if section in df.columns:
                best_col = f"{section}_best"
                delta_col = f"{section}_delta_to_best"

                df[best_col] = df.groupby("DRIVER_NUMBER")[section].transform("min")
                df[delta_col] = df[section] - df[best_col]

        # Section consistency
        window = self.config["rolling_window"]
        for section in section_cols:
            if section in df.columns:
                df[f"{section}_rolling_mean"] = df.groupby("DRIVER_NUMBER")[
                    section
                ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

                df[f"{section}_rolling_std"] = df.groupby("DRIVER_NUMBER")[
                    section
                ].transform(lambda x: x.rolling(window=window, min_periods=1).std())

        # Overall section consistency score
        if all(f"{s}_rolling_std" in df.columns for s in section_cols):
            avg_std = df[[f"{s}_rolling_std" for s in section_cols]].mean(axis=1)
            df["section_consistency_score"] = 1 / (1 + avg_std)

        # Section performance balance
        if all(col in df.columns for col in section_cols):
            # Coefficient of variation across sections
            section_mean = df[section_cols].mean(axis=1)
            section_std = df[section_cols].std(axis=1)
            df["section_balance"] = section_std / section_mean

        # Speed features
        if "KPH" in df.columns:
            df["speed_delta_to_best"] = df.groupby("DRIVER_NUMBER")["KPH"].transform(
                lambda x: df["KPH"] - x.max()
            )

        if "TOP_SPEED" in df.columns:
            df["top_speed_delta_to_best"] = df.groupby("DRIVER_NUMBER")[
                "TOP_SPEED"
            ].transform(lambda x: df["TOP_SPEED"] - x.max())

        # Lap time features
        if "LAP_TIME_SECONDS" in df.columns:
            df["lap_time_percentile"] = df.groupby("DRIVER_NUMBER")[
                "LAP_TIME_SECONDS"
            ].rank(pct=True)

            # Delta to fastest lap
            df["fastest_lap_in_race"] = df["LAP_TIME_SECONDS"].min()
            df["delta_to_fastest"] = df["LAP_TIME_SECONDS"] - df["fastest_lap_in_race"]

        logger.info(f"Created section features for {len(df)} records")
        return df

    def engineer_telemetry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from telemetry data.

        Args:
            df: DataFrame with telemetry data

        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Engineering telemetry features...")
        df = df.copy()

        # Speed features
        if "gps_speed" in df.columns and "vehicle_number" in df.columns:
            # Speed changes
            df["speed_delta"] = df.groupby("vehicle_number")["gps_speed"].diff()

            # Acceleration (change in speed per second)
            if "timestamp" in df.columns:
                time_delta = df.groupby("vehicle_number")["timestamp"].diff()
                df["time_delta_seconds"] = time_delta.dt.total_seconds()
                df["acceleration"] = df["speed_delta"] / df["time_delta_seconds"]

            # Rolling speed statistics
            window = 10  # 10 telemetry points
            df["speed_rolling_mean"] = df.groupby("vehicle_number")["gps_speed"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            df["speed_rolling_max"] = df.groupby("vehicle_number")["gps_speed"].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )

        # GPS distance calculation
        if all(col in df.columns for col in ["gps_lat", "gps_long", "vehicle_number"]):
            df["distance_delta"] = df.groupby("vehicle_number").apply(
                lambda x: self._calculate_distance_deltas(
                    x["gps_lat"].values, x["gps_long"].values
                )
            ).reset_index(level=0, drop=True)

        # Heading changes (cornering)
        if "gps_heading" in df.columns:
            df["heading_delta"] = df.groupby("vehicle_number")["gps_heading"].diff()
            # Normalize to -180 to 180
            df["heading_delta"] = df["heading_delta"].apply(
                lambda x: ((x + 180) % 360) - 180 if pd.notna(x) else x
            )

            # Cornering intensity
            df["cornering_intensity"] = abs(df["heading_delta"])

        logger.info(f"Created telemetry features for {len(df)} records")
        return df

    def calculate_driver_consistency(
        self, section_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate consistency metrics for each driver.

        Args:
            section_df: DataFrame with section analysis data

        Returns:
            DataFrame with driver-level consistency metrics
        """
        logger.info("Calculating driver consistency metrics...")

        if "DRIVER_NUMBER" not in section_df.columns or "LAP_TIME_SECONDS" not in section_df.columns:
            logger.warning("Missing required columns for consistency calculation")
            return pd.DataFrame()

        # Group by driver
        consistency = section_df.groupby("DRIVER_NUMBER").agg({
            "LAP_TIME_SECONDS": ["mean", "std", "min", "max", "count"],
            "S1_SECONDS": ["mean", "std"],
            "S2_SECONDS": ["mean", "std"],
            "S3_SECONDS": ["mean", "std"],
        }).reset_index()

        # Flatten column names
        consistency.columns = [
            "_".join(col).strip("_") for col in consistency.columns.values
        ]

        # Calculate coefficient of variation (lower = more consistent)
        consistency["lap_time_cv"] = (
            consistency["LAP_TIME_SECONDS_std"] / consistency["LAP_TIME_SECONDS_mean"]
        )

        # Consistency score (0-100, higher is better)
        consistency["consistency_score"] = (
            100 * (1 - consistency["lap_time_cv"])
        ).clip(0, 100)

        # Average section consistency
        section_cvs = []
        for section in ["S1", "S2", "S3"]:
            mean_col = f"{section}_SECONDS_mean"
            std_col = f"{section}_SECONDS_std"
            if mean_col in consistency.columns and std_col in consistency.columns:
                section_cvs.append(consistency[std_col] / consistency[mean_col])

        if section_cvs:
            consistency["avg_section_cv"] = pd.concat(section_cvs, axis=1).mean(axis=1)

        logger.info(f"Calculated consistency for {len(consistency)} drivers")
        return consistency

    def calculate_speed_deltas(
        self, section_df: pd.DataFrame, reference_driver: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate speed deltas relative to reference (fastest driver or specified).

        Args:
            section_df: DataFrame with section analysis data
            reference_driver: Driver number to use as reference (None = fastest)

        Returns:
            DataFrame with speed delta columns
        """
        logger.info("Calculating speed deltas...")
        df = section_df.copy()

        if "DRIVER_NUMBER" not in df.columns or "LAP_TIME_SECONDS" not in df.columns:
            logger.warning("Missing required columns for speed delta calculation")
            return df

        # Find reference (fastest driver's average lap time)
        if reference_driver is None:
            driver_avg_times = df.groupby("DRIVER_NUMBER")["LAP_TIME_SECONDS"].mean()
            reference_driver = driver_avg_times.idxmin()

        logger.info(f"Using driver {reference_driver} as reference")

        # Get reference lap times
        reference_laps = df[df["DRIVER_NUMBER"] == reference_driver].set_index(
            "LAP_NUMBER"
        )["LAP_TIME_SECONDS"]

        # Calculate deltas
        def calc_delta(row):
            if row["LAP_NUMBER"] in reference_laps.index:
                return row["LAP_TIME_SECONDS"] - reference_laps.loc[row["LAP_NUMBER"]]
            return np.nan

        df["delta_to_reference"] = df.apply(calc_delta, axis=1)

        return df

    @staticmethod
    def _calculate_distance_deltas(lats: np.ndarray, longs: np.ndarray) -> pd.Series:
        """
        Calculate distance between consecutive GPS points using Haversine formula.

        Args:
            lats: Array of latitudes
            longs: Array of longitudes

        Returns:
            Series with distance deltas in meters
        """
        if len(lats) < 2:
            return pd.Series([0] * len(lats))

        # Haversine formula
        R = 6371000  # Earth radius in meters

        lat1 = np.radians(lats[:-1])
        lat2 = np.radians(lats[1:])
        dlat = np.radians(np.diff(lats))
        dlon = np.radians(np.diff(longs))

        a = (
            np.sin(dlat / 2) ** 2 +
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c

        # Prepend 0 for first point
        return pd.Series(np.concatenate([[0], distances]))

    def engineer_all_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for all available datasets.

        Args:
            data: Dictionary of DataFrames

        Returns:
            Dictionary of DataFrames with engineered features
        """
        logger.info("Engineering features for all datasets...")
        engineered = {}

        if "lap_time" in data:
            engineered["lap_time"] = self.engineer_lap_features(data["lap_time"])

        if "section_analysis" in data:
            engineered["section_analysis"] = self.engineer_section_features(
                data["section_analysis"]
            )

        if "telemetry" in data:
            engineered["telemetry"] = self.engineer_telemetry_features(
                data["telemetry"]
            )

        # Copy other datasets as-is
        for key, df in data.items():
            if key not in engineered:
                engineered[key] = df

        logger.info("Feature engineering complete")
        return engineered
