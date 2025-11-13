"""
RaceIQ Pro - Performance Metrics

This module contains functions for calculating various performance metrics
including lap time analysis, sector performance, and driver comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_lap_time_stats(
    df: pd.DataFrame,
    group_by: str = "vehicle_number"
) -> pd.DataFrame:
    """
    Calculate lap time statistics for each group (driver/vehicle).

    Args:
        df: DataFrame with lap time data
        group_by: Column to group by

    Returns:
        DataFrame with lap time statistics
    """
    if "lap_duration_seconds" not in df.columns:
        logger.warning("lap_duration_seconds not found in DataFrame")
        return pd.DataFrame()

    stats = df.groupby(group_by)["lap_duration_seconds"].agg([
        ("best_lap", "min"),
        ("worst_lap", "max"),
        ("avg_lap", "mean"),
        ("median_lap", "median"),
        ("std_lap", "std"),
        ("total_laps", "count"),
    ]).reset_index()

    # Calculate consistency metrics
    stats["lap_time_range"] = stats["worst_lap"] - stats["best_lap"]
    stats["coefficient_of_variation"] = stats["std_lap"] / stats["avg_lap"]

    return stats


def calculate_sector_performance(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER"
) -> pd.DataFrame:
    """
    Calculate sector-by-sector performance metrics.

    Args:
        df: DataFrame with section analysis data
        driver_col: Column name for driver identifier

    Returns:
        DataFrame with sector performance metrics
    """
    sector_cols = ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]

    if not all(col in df.columns for col in sector_cols):
        logger.warning("Missing sector columns")
        return pd.DataFrame()

    results = []

    for driver in df[driver_col].unique():
        driver_data = df[df[driver_col] == driver]

        driver_stats = {driver_col: driver}

        for sector in sector_cols:
            sector_name = sector.replace("_SECONDS", "")
            driver_stats[f"{sector_name}_best"] = driver_data[sector].min()
            driver_stats[f"{sector_name}_avg"] = driver_data[sector].mean()
            driver_stats[f"{sector_name}_std"] = driver_data[sector].std()
            driver_stats[f"{sector_name}_worst"] = driver_data[sector].max()

        results.append(driver_stats)

    return pd.DataFrame(results)


def calculate_pace_analysis(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    time_col: str = "LAP_TIME_SECONDS"
) -> pd.DataFrame:
    """
    Analyze pace evolution throughout the race.

    Args:
        df: DataFrame with lap data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        time_col: Column name for lap time

    Returns:
        DataFrame with pace analysis including stint analysis
    """
    if not all(col in df.columns for col in [driver_col, lap_col, time_col]):
        logger.warning("Missing required columns for pace analysis")
        return pd.DataFrame()

    pace_data = []

    for driver in df[driver_col].unique():
        driver_laps = df[df[driver_col] == driver].sort_values(lap_col)

        if len(driver_laps) == 0:
            continue

        # Early race pace (first 5 laps)
        early_laps = driver_laps[driver_laps[lap_col] <= 5]
        early_pace = early_laps[time_col].mean() if len(early_laps) > 0 else np.nan

        # Mid race pace (laps 6-15)
        mid_laps = driver_laps[(driver_laps[lap_col] > 5) & (driver_laps[lap_col] <= 15)]
        mid_pace = mid_laps[time_col].mean() if len(mid_laps) > 0 else np.nan

        # Late race pace (laps 16+)
        late_laps = driver_laps[driver_laps[lap_col] > 15]
        late_pace = late_laps[time_col].mean() if len(late_laps) > 0 else np.nan

        # Overall statistics
        best_lap = driver_laps[time_col].min()
        avg_lap = driver_laps[time_col].mean()

        # Pace degradation
        pace_degradation = late_pace - early_pace if pd.notna([early_pace, late_pace]).all() else np.nan

        pace_data.append({
            driver_col: driver,
            "best_lap": best_lap,
            "avg_lap": avg_lap,
            "early_pace": early_pace,
            "mid_pace": mid_pace,
            "late_pace": late_pace,
            "pace_degradation": pace_degradation,
            "total_laps": len(driver_laps),
        })

    return pd.DataFrame(pace_data)


def calculate_gap_to_leader(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    time_col: str = "LAP_TIME_SECONDS"
) -> pd.DataFrame:
    """
    Calculate cumulative gap to leader for each lap.

    Args:
        df: DataFrame with lap data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        time_col: Column name for lap time

    Returns:
        DataFrame with gap to leader for each driver/lap
    """
    if not all(col in df.columns for col in [driver_col, lap_col, time_col]):
        logger.warning("Missing required columns for gap calculation")
        return pd.DataFrame()

    # Calculate cumulative time for each driver
    df_sorted = df.sort_values([lap_col, driver_col])
    df_sorted["cumulative_time"] = df_sorted.groupby(driver_col)[time_col].cumsum()

    # Find leader's cumulative time for each lap
    leader_times = df_sorted.groupby(lap_col)["cumulative_time"].min().reset_index()
    leader_times.columns = [lap_col, "leader_cumulative_time"]

    # Merge and calculate gap
    df_with_gap = df_sorted.merge(leader_times, on=lap_col, how="left")
    df_with_gap["gap_to_leader"] = (
        df_with_gap["cumulative_time"] - df_with_gap["leader_cumulative_time"]
    )

    return df_with_gap


def identify_fastest_sectors(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER"
) -> Dict[str, int]:
    """
    Identify which driver was fastest in each sector.

    Args:
        df: DataFrame with section analysis data
        driver_col: Column name for driver identifier

    Returns:
        Dictionary mapping sector names to fastest driver numbers
    """
    sector_cols = ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]

    if not all(col in df.columns for col in sector_cols):
        logger.warning("Missing sector columns")
        return {}

    fastest = {}

    for sector in sector_cols:
        # Find minimum time across all drivers
        best_sector_time = df[sector].min()
        # Find driver who achieved it
        fastest_driver = df[df[sector] == best_sector_time][driver_col].iloc[0]

        sector_name = sector.replace("_SECONDS", "")
        fastest[sector_name] = int(fastest_driver)

    return fastest


def calculate_consistency_score(
    lap_times: pd.Series,
    method: str = "cv"
) -> float:
    """
    Calculate consistency score from lap times.

    Args:
        lap_times: Series of lap times
        method: Method to use ('cv' for coefficient of variation, 'std' for standard deviation)

    Returns:
        Consistency score (lower is more consistent)
    """
    if len(lap_times) < 2:
        return np.nan

    if method == "cv":
        # Coefficient of variation (std / mean)
        return lap_times.std() / lap_times.mean()
    elif method == "std":
        # Standard deviation
        return lap_times.std()
    else:
        raise ValueError(f"Unknown consistency method: {method}")


def detect_pit_stops(
    df: pd.DataFrame,
    time_threshold: float = 30.0,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    time_col: str = "LAP_TIME_SECONDS"
) -> pd.DataFrame:
    """
    Detect pit stops based on abnormally long lap times.

    Args:
        df: DataFrame with lap data
        time_threshold: Threshold (in seconds) above median to consider as pit stop
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        time_col: Column name for lap time

    Returns:
        DataFrame with detected pit stops
    """
    if not all(col in df.columns for col in [driver_col, lap_col, time_col]):
        logger.warning("Missing required columns for pit stop detection")
        return pd.DataFrame()

    pit_stops = []

    for driver in df[driver_col].unique():
        driver_data = df[df[driver_col] == driver].copy()

        if len(driver_data) < 3:
            continue

        # Calculate median lap time for this driver
        median_time = driver_data[time_col].median()

        # Find laps that are significantly longer
        driver_data["is_pit_stop"] = (
            driver_data[time_col] > median_time + time_threshold
        )

        # Extract pit stop information
        pit_laps = driver_data[driver_data["is_pit_stop"]]

        for _, pit_lap in pit_laps.iterrows():
            pit_stops.append({
                driver_col: driver,
                lap_col: pit_lap[lap_col],
                "lap_time": pit_lap[time_col],
                "median_lap_time": median_time,
                "time_lost": pit_lap[time_col] - median_time,
            })

    return pd.DataFrame(pit_stops)


def calculate_position_changes(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    time_col: str = "LAP_TIME_SECONDS"
) -> pd.DataFrame:
    """
    Calculate position changes throughout the race.

    Args:
        df: DataFrame with lap data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        time_col: Column name for lap time

    Returns:
        DataFrame with position for each driver at each lap
    """
    if not all(col in df.columns for col in [driver_col, lap_col, time_col]):
        logger.warning("Missing required columns for position calculation")
        return pd.DataFrame()

    # Calculate cumulative time
    df_sorted = df.sort_values([driver_col, lap_col])
    df_sorted["cumulative_time"] = df_sorted.groupby(driver_col)[time_col].cumsum()

    # Calculate position at each lap
    df_sorted["position"] = df_sorted.groupby(lap_col)["cumulative_time"].rank(
        method="min"
    )

    # Calculate position change from start
    starting_positions = df_sorted[df_sorted[lap_col] == df_sorted[lap_col].min()][
        [driver_col, "position"]
    ].rename(columns={"position": "starting_position"})

    df_with_changes = df_sorted.merge(starting_positions, on=driver_col, how="left")
    df_with_changes["position_change"] = (
        df_with_changes["starting_position"] - df_with_changes["position"]
    )

    return df_with_changes


def calculate_theoretical_best_lap(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER"
) -> pd.DataFrame:
    """
    Calculate theoretical best lap time by combining best sectors.

    Args:
        df: DataFrame with section analysis data
        driver_col: Column name for driver identifier

    Returns:
        DataFrame with theoretical best lap for each driver
    """
    sector_cols = ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]

    if not all(col in df.columns for col in sector_cols + [driver_col]):
        logger.warning("Missing required columns for theoretical best calculation")
        return pd.DataFrame()

    results = []

    for driver in df[driver_col].unique():
        driver_data = df[df[driver_col] == driver]

        best_s1 = driver_data["S1_SECONDS"].min()
        best_s2 = driver_data["S2_SECONDS"].min()
        best_s3 = driver_data["S3_SECONDS"].min()
        theoretical_best = best_s1 + best_s2 + best_s3

        actual_best = driver_data["LAP_TIME_SECONDS"].min() if "LAP_TIME_SECONDS" in df.columns else np.nan
        potential_gain = actual_best - theoretical_best if pd.notna(actual_best) else np.nan

        results.append({
            driver_col: driver,
            "best_S1": best_s1,
            "best_S2": best_s2,
            "best_S3": best_s3,
            "theoretical_best_lap": theoretical_best,
            "actual_best_lap": actual_best,
            "potential_gain": potential_gain,
        })

    return pd.DataFrame(results)
