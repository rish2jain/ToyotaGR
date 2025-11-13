"""
RaceIQ Pro - Data Loader

This module handles loading CSV data files from various sources including lap times,
telemetry data, section analysis, and race results.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict
import pandas as pd
import glob

from ..utils.constants import (
    BARBER_SAMPLES_DIR,
    DATA_DIR,
    FILE_PATTERNS,
    LAP_TIME_COLUMNS,
    SECTION_ANALYSIS_COLUMNS,
    RACE_RESULTS_COLUMNS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and basic preprocessing of race data files.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the DataLoader.

        Args:
            base_path: Base directory for data files. Defaults to BARBER_SAMPLES_DIR.
        """
        self.base_path = Path(base_path) if base_path else BARBER_SAMPLES_DIR
        logger.info(f"DataLoader initialized with base_path: {self.base_path}")

    def load_lap_time_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load lap time data from CSV file.

        Args:
            file_path: Path to the lap time CSV file. If None, searches in base_path.

        Returns:
            DataFrame containing lap time data with parsed timestamps.

        Raises:
            FileNotFoundError: If the file cannot be found.
            pd.errors.EmptyDataError: If the file is empty.
        """
        try:
            if file_path is None:
                # Search for lap time file in base directory
                pattern = str(self.base_path / FILE_PATTERNS["lap_time"])
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(
                        f"No lap time files found matching pattern: {pattern}"
                    )
                file_path = files[0]
                logger.info(f"Auto-detected lap time file: {file_path}")

            # Load the CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded lap time data: {len(df)} rows, {len(df.columns)} columns")

            # Parse timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if "meta_time" in df.columns:
                df["meta_time"] = pd.to_datetime(df["meta_time"], errors="coerce")

            # Convert lap to integer
            if "lap" in df.columns:
                df["lap"] = pd.to_numeric(df["lap"], errors="coerce")

            # Sort by vehicle and lap
            if "vehicle_number" in df.columns and "lap" in df.columns:
                df = df.sort_values(["vehicle_number", "lap"]).reset_index(drop=True)

            return df

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading lap time data: {e}")
            raise

    def load_lap_start_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load lap start time data from CSV file.

        Args:
            file_path: Path to the lap start CSV file. If None, searches in base_path.

        Returns:
            DataFrame containing lap start data with parsed timestamps.
        """
        try:
            if file_path is None:
                pattern = str(self.base_path / FILE_PATTERNS["lap_start"])
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(
                        f"No lap start files found matching pattern: {pattern}"
                    )
                file_path = files[0]
                logger.info(f"Auto-detected lap start file: {file_path}")

            df = pd.read_csv(file_path)
            logger.info(f"Loaded lap start data: {len(df)} rows")

            # Parse timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Convert lap to integer
            if "lap" in df.columns:
                df["lap"] = pd.to_numeric(df["lap"], errors="coerce")

            # Sort by vehicle and lap
            if "vehicle_number" in df.columns and "lap" in df.columns:
                df = df.sort_values(["vehicle_number", "lap"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error loading lap start data: {e}")
            raise

    def load_lap_end_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load lap end time data from CSV file.

        Args:
            file_path: Path to the lap end CSV file. If None, searches in base_path.

        Returns:
            DataFrame containing lap end data with parsed timestamps.
        """
        try:
            if file_path is None:
                pattern = str(self.base_path / FILE_PATTERNS["lap_end"])
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(
                        f"No lap end files found matching pattern: {pattern}"
                    )
                file_path = files[0]
                logger.info(f"Auto-detected lap end file: {file_path}")

            df = pd.read_csv(file_path)
            logger.info(f"Loaded lap end data: {len(df)} rows")

            # Parse timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Convert lap to integer
            if "lap" in df.columns:
                df["lap"] = pd.to_numeric(df["lap"], errors="coerce")

            # Sort by vehicle and lap
            if "vehicle_number" in df.columns and "lap" in df.columns:
                df = df.sort_values(["vehicle_number", "lap"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error loading lap end data: {e}")
            raise

    def load_telemetry_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load telemetry data from CSV file.

        Args:
            file_path: Path to the telemetry CSV file. If None, searches in base_path.

        Returns:
            DataFrame containing telemetry data with parsed timestamps.
        """
        try:
            if file_path is None:
                pattern = str(self.base_path / FILE_PATTERNS["telemetry"])
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(
                        f"No telemetry files found matching pattern: {pattern}"
                    )
                file_path = files[0]
                logger.info(f"Auto-detected telemetry file: {file_path}")

            df = pd.read_csv(file_path)
            logger.info(f"Loaded telemetry data: {len(df)} rows, {len(df.columns)} columns")

            # Parse timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Convert numeric columns
            numeric_cols = [
                "gps_alt", "gps_heading", "gps_lat", "gps_long",
                "gps_speed", "lap"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by vehicle and timestamp
            if "vehicle_number" in df.columns and "timestamp" in df.columns:
                df = df.sort_values(["vehicle_number", "timestamp"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error loading telemetry data: {e}")
            raise

    def load_section_analysis(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load section analysis data from CSV file.

        Args:
            file_path: Path to the section analysis CSV file. If None, searches in base_path.

        Returns:
            DataFrame containing section analysis data.
        """
        try:
            if file_path is None:
                pattern = str(self.base_path / FILE_PATTERNS["section_analysis"])
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(
                        f"No section analysis files found matching pattern: {pattern}"
                    )
                file_path = files[0]
                logger.info(f"Auto-detected section analysis file: {file_path}")

            df = pd.read_csv(file_path, sep=";")
            logger.info(f"Loaded section analysis data: {len(df)} rows, {len(df.columns)} columns")

            # Convert lap time from MM:SS.mmm format to seconds
            if "LAP_TIME" in df.columns:
                df["LAP_TIME_SECONDS"] = df["LAP_TIME"].apply(self._parse_lap_time)

            # Convert section times to numeric
            for col in ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert other numeric columns
            numeric_cols = ["LAP_NUMBER", "KPH", "TOP_SPEED", "DRIVER_NUMBER"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by driver and lap
            if "DRIVER_NUMBER" in df.columns and "LAP_NUMBER" in df.columns:
                df = df.sort_values(["DRIVER_NUMBER", "LAP_NUMBER"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error loading section analysis data: {e}")
            raise

    def load_race_results(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load race results data from CSV file.

        Args:
            file_path: Path to the race results CSV file. If None, searches in base_path.

        Returns:
            DataFrame containing race results data.
        """
        try:
            if file_path is None:
                pattern = str(self.base_path / FILE_PATTERNS["race_results"])
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError(
                        f"No race results files found matching pattern: {pattern}"
                    )
                file_path = files[0]
                logger.info(f"Auto-detected race results file: {file_path}")

            df = pd.read_csv(file_path, sep=";")
            logger.info(f"Loaded race results data: {len(df)} rows, {len(df.columns)} columns")

            # Convert numeric columns
            numeric_cols = ["POSITION", "NUMBER", "LAPS", "FL_KPH"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parse fastest lap time
            if "FL_TIME" in df.columns:
                df["FL_TIME_SECONDS"] = df["FL_TIME"].apply(self._parse_lap_time)

            # Sort by position
            if "POSITION" in df.columns:
                df = df.sort_values("POSITION").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error loading race results data: {e}")
            raise

    @staticmethod
    def _parse_lap_time(time_str: str) -> Optional[float]:
        """
        Parse lap time from MM:SS.mmm format to seconds.

        Args:
            time_str: Time string in MM:SS.mmm format

        Returns:
            Time in seconds as float, or None if parsing fails
        """
        try:
            if pd.isna(time_str) or time_str == "":
                return None

            # Handle format like "1:39.167"
            parts = str(time_str).split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                # Try to parse as float directly
                return float(time_str)
        except (ValueError, AttributeError):
            return None

    def load_all_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available sample data files.

        Returns:
            Dictionary with data type as key and DataFrame as value.
        """
        logger.info("Loading all sample data files...")
        data = {}

        try:
            data["lap_time"] = self.load_lap_time_data()
        except Exception as e:
            logger.warning(f"Could not load lap time data: {e}")

        try:
            data["lap_start"] = self.load_lap_start_data()
        except Exception as e:
            logger.warning(f"Could not load lap start data: {e}")

        try:
            data["lap_end"] = self.load_lap_end_data()
        except Exception as e:
            logger.warning(f"Could not load lap end data: {e}")

        try:
            data["telemetry"] = self.load_telemetry_data()
        except Exception as e:
            logger.warning(f"Could not load telemetry data: {e}")

        try:
            data["section_analysis"] = self.load_section_analysis()
        except Exception as e:
            logger.warning(f"Could not load section analysis data: {e}")

        try:
            data["race_results"] = self.load_race_results()
        except Exception as e:
            logger.warning(f"Could not load race results data: {e}")

        logger.info(f"Loaded {len(data)} data files successfully")
        return data


def load_data_for_track(
    track: str, race: str = "Race 1", use_samples: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all data for a specific track and race.

    Args:
        track: Track name (e.g., 'barber', 'COTA', 'sonoma')
        race: Race identifier (e.g., 'Race 1', 'Race 2')
        use_samples: Whether to use sample data (default: False)

    Returns:
        Dictionary containing all available data for the track/race.
    """
    if use_samples and track.lower() == "barber":
        loader = DataLoader(BARBER_SAMPLES_DIR)
    else:
        track_path = DATA_DIR / track
        if race:
            track_path = track_path / race
        loader = DataLoader(track_path)

    return loader.load_all_sample_data()
