"""
RaceIQ Pro - Data Validator

This module performs data quality checks and validation on loaded data.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from ..utils.constants import THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Performs validation and quality checks on race data.
    """

    def __init__(self):
        """Initialize the DataValidator."""
        self.validation_results = {}

    def validate_lap_times(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate lap time data.

        Args:
            df: DataFrame containing lap time data

        Returns:
            Dictionary containing validation results
        """
        results = {
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "passed": True,
        }

        # Check for required columns
        required_cols = ["lap", "vehicle_number", "timestamp"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results["issues"].append(f"Missing required columns: {missing_cols}")
            results["passed"] = False

        # Check for null values in critical columns
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    results["warnings"].append(
                        f"Column '{col}' has {null_count} null values ({null_count/len(df)*100:.2f}%)"
                    )

        # Check for duplicate records
        if all(col in df.columns for col in ["vehicle_number", "lap"]):
            duplicates = df.duplicated(subset=["vehicle_number", "lap"], keep=False).sum()
            if duplicates > 0:
                results["warnings"].append(
                    f"Found {duplicates} duplicate records for vehicle+lap combinations"
                )

        # Check timestamp ordering
        if "timestamp" in df.columns and "vehicle_number" in df.columns:
            for vehicle in df["vehicle_number"].unique():
                vehicle_data = df[df["vehicle_number"] == vehicle].copy()
                if len(vehicle_data) > 1:
                    if not vehicle_data["timestamp"].is_monotonic_increasing:
                        results["warnings"].append(
                            f"Timestamps not monotonic for vehicle {vehicle}"
                        )

        results["summary"] = f"Validated {results['total_records']} lap time records"
        logger.info(results["summary"])

        return results

    def validate_telemetry(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate telemetry data.

        Args:
            df: DataFrame containing telemetry data

        Returns:
            Dictionary containing validation results
        """
        results = {
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "passed": True,
        }

        # Check for required columns
        required_cols = ["timestamp", "vehicle_number", "gps_lat", "gps_long", "gps_speed"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results["issues"].append(f"Missing required columns: {missing_cols}")
            results["passed"] = False

        # Validate GPS coordinates
        if "gps_lat" in df.columns and "gps_long" in df.columns:
            # Check for reasonable latitude (-90 to 90)
            invalid_lat = ((df["gps_lat"] < -90) | (df["gps_lat"] > 90)).sum()
            if invalid_lat > 0:
                results["warnings"].append(
                    f"Found {invalid_lat} records with invalid latitude"
                )

            # Check for reasonable longitude (-180 to 180)
            invalid_long = ((df["gps_long"] < -180) | (df["gps_long"] > 180)).sum()
            if invalid_long > 0:
                results["warnings"].append(
                    f"Found {invalid_long} records with invalid longitude"
                )

        # Validate speed
        if "gps_speed" in df.columns:
            invalid_speed = (
                (df["gps_speed"] < THRESHOLDS["min_speed"]) |
                (df["gps_speed"] > THRESHOLDS["max_speed"])
            ).sum()
            if invalid_speed > 0:
                results["warnings"].append(
                    f"Found {invalid_speed} records with speed outside valid range "
                    f"({THRESHOLDS['min_speed']}-{THRESHOLDS['max_speed']} km/h)"
                )

        # Check GPS quality if available
        if "gps_quality" in df.columns:
            poor_quality = (df["gps_quality"] < 1).sum()
            if poor_quality > 0:
                results["warnings"].append(
                    f"Found {poor_quality} records with poor GPS quality ({poor_quality/len(df)*100:.2f}%)"
                )

        results["summary"] = f"Validated {results['total_records']} telemetry records"
        logger.info(results["summary"])

        return results

    def validate_section_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate section analysis data.

        Args:
            df: DataFrame containing section analysis data

        Returns:
            Dictionary containing validation results
        """
        results = {
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "passed": True,
        }

        # Check for required columns
        required_cols = ["DRIVER_NUMBER", "LAP_NUMBER", "LAP_TIME"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results["issues"].append(f"Missing required columns: {missing_cols}")
            results["passed"] = False

        # Validate section times
        section_cols = ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]
        if all(col in df.columns for col in section_cols):
            # Check if section times sum approximately to lap time
            if "LAP_TIME_SECONDS" in df.columns:
                df_clean = df.dropna(subset=section_cols + ["LAP_TIME_SECONDS"])
                section_sum = df_clean[section_cols].sum(axis=1)
                lap_time_diff = abs(section_sum - df_clean["LAP_TIME_SECONDS"])

                # Allow 1 second tolerance for rounding
                large_diff = (lap_time_diff > 1.0).sum()
                if large_diff > 0:
                    results["warnings"].append(
                        f"Found {large_diff} records where section times don't sum to lap time"
                    )

        # Validate lap times
        if "LAP_TIME_SECONDS" in df.columns:
            invalid_laps = (
                (df["LAP_TIME_SECONDS"] < THRESHOLDS["min_lap_time"]) |
                (df["LAP_TIME_SECONDS"] > THRESHOLDS["max_lap_time"])
            ).sum()
            if invalid_laps > 0:
                results["warnings"].append(
                    f"Found {invalid_laps} records with lap time outside valid range "
                    f"({THRESHOLDS['min_lap_time']}-{THRESHOLDS['max_lap_time']} seconds)"
                )

        # Validate speed data
        if "KPH" in df.columns:
            invalid_speed = (
                (df["KPH"] < THRESHOLDS["min_speed"]) |
                (df["KPH"] > THRESHOLDS["max_speed"])
            ).sum()
            if invalid_speed > 0:
                results["warnings"].append(
                    f"Found {invalid_speed} records with average speed outside valid range"
                )

        results["summary"] = f"Validated {results['total_records']} section analysis records"
        logger.info(results["summary"])

        return results

    def validate_race_results(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate race results data.

        Args:
            df: DataFrame containing race results data

        Returns:
            Dictionary containing validation results
        """
        results = {
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "passed": True,
        }

        # Check for required columns
        required_cols = ["POSITION", "NUMBER", "STATUS", "LAPS"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results["issues"].append(f"Missing required columns: {missing_cols}")
            results["passed"] = False

        # Check position sequence
        if "POSITION" in df.columns:
            positions = df["POSITION"].dropna().astype(int)
            if len(positions) > 0:
                expected = list(range(1, len(positions) + 1))
                if list(positions) != expected:
                    results["warnings"].append(
                        "Position sequence is not continuous (1, 2, 3, ...)"
                    )

        # Check for duplicate numbers
        if "NUMBER" in df.columns:
            duplicates = df["NUMBER"].duplicated().sum()
            if duplicates > 0:
                results["issues"].append(f"Found {duplicates} duplicate car numbers")
                results["passed"] = False

        # Validate fastest lap times
        if "FL_TIME_SECONDS" in df.columns:
            invalid_fl = (
                (df["FL_TIME_SECONDS"] < THRESHOLDS["min_lap_time"]) |
                (df["FL_TIME_SECONDS"] > THRESHOLDS["max_lap_time"])
            ).sum()
            if invalid_fl > 0:
                results["warnings"].append(
                    f"Found {invalid_fl} records with fastest lap outside valid range"
                )

        results["summary"] = f"Validated {results['total_records']} race result records"
        logger.info(results["summary"])

        return results

    def validate_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate all datasets.

        Args:
            data: Dictionary of DataFrames to validate

        Returns:
            Dictionary with validation results for each dataset
        """
        logger.info("Starting validation of all datasets...")
        all_results = {}

        validators = {
            "lap_time": self.validate_lap_times,
            "lap_start": self.validate_lap_times,
            "lap_end": self.validate_lap_times,
            "telemetry": self.validate_telemetry,
            "section_analysis": self.validate_section_analysis,
            "race_results": self.validate_race_results,
        }

        for data_type, df in data.items():
            if data_type in validators:
                logger.info(f"Validating {data_type}...")
                all_results[data_type] = validators[data_type](df)
            else:
                logger.warning(f"No validator found for data type: {data_type}")

        # Summary
        total_issues = sum(len(r.get("issues", [])) for r in all_results.values())
        total_warnings = sum(len(r.get("warnings", [])) for r in all_results.values())

        logger.info(f"Validation complete: {total_issues} issues, {total_warnings} warnings")

        self.validation_results = all_results
        return all_results

    def get_summary_report(self) -> str:
        """
        Generate a summary report of validation results.

        Returns:
            Formatted string with validation summary
        """
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."

        report = ["=" * 80]
        report.append("DATA VALIDATION SUMMARY REPORT")
        report.append("=" * 80)

        for data_type, results in self.validation_results.items():
            report.append(f"\n{data_type.upper()}:")
            report.append(f"  Total Records: {results['total_records']}")
            report.append(f"  Status: {'PASSED' if results['passed'] else 'FAILED'}")

            if results.get("issues"):
                report.append("  Issues:")
                for issue in results["issues"]:
                    report.append(f"    - {issue}")

            if results.get("warnings"):
                report.append("  Warnings:")
                for warning in results["warnings"]:
                    report.append(f"    - {warning}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)
