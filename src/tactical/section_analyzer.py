"""
Section Analysis Module

This module provides detailed statistical analysis of track sections,
identifying driver strengths and areas for improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class SectionAnalyzer:
    """
    Analyzer for track section performance and driver comparison.

    Provides statistical analysis of section times, identifies driver strengths,
    and highlights improvement opportunities.
    """

    def __init__(self):
        """Initialize the SectionAnalyzer."""
        self.section_stats: Optional[Dict[str, Dict[str, float]]] = None

    def calculate_section_statistics(
        self,
        section_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics for each track section.

        Computes mean, median, standard deviation, minimum (best) time,
        and percentiles for each section in the dataset.

        Args:
            section_data: DataFrame containing section time data with columns
                         ending in '_SECONDS' (e.g., S1_SECONDS, S2_SECONDS)

        Returns:
            Dictionary with structure:
            {
                'S1': {
                    'mean': float,
                    'median': float,
                    'std': float,
                    'min': float,  # Best time
                    'max': float,  # Worst time
                    'q25': float,  # 25th percentile
                    'q75': float,  # 75th percentile
                    'count': int   # Number of valid samples
                },
                ...
            }

        Example:
            >>> analyzer = SectionAnalyzer()
            >>> stats = analyzer.calculate_section_statistics(section_df)
            >>> print(f"S1 mean: {stats['S1']['mean']:.3f}s, std: {stats['S1']['std']:.3f}s")
        """
        if section_data.empty:
            raise ValueError("Section data is empty")

        # Find all section columns
        section_columns = [col for col in section_data.columns if col.endswith('_SECONDS')]

        if not section_columns:
            raise ValueError("No section time columns found (expected columns ending with '_SECONDS')")

        statistics = {}

        for section_col in section_columns:
            # Remove '_SECONDS' suffix to get section name
            section_name = section_col.replace('_SECONDS', '')

            # Filter out invalid times
            valid_times = section_data[section_col].dropna()
            valid_times = valid_times[valid_times > 0]

            if len(valid_times) == 0:
                continue

            # Calculate statistics
            statistics[section_name] = {
                'mean': float(valid_times.mean()),
                'median': float(valid_times.median()),
                'std': float(valid_times.std()),
                'min': float(valid_times.min()),
                'max': float(valid_times.max()),
                'q25': float(valid_times.quantile(0.25)),
                'q75': float(valid_times.quantile(0.75)),
                'count': int(len(valid_times))
            }

        self.section_stats = statistics
        return statistics

    def identify_driver_strengths(
        self,
        driver_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        top_percentile: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Identify sections where a driver performs in the top percentile.

        Compares driver's median section times against the overall field
        to determine relative strengths.

        Args:
            driver_data: DataFrame for a specific driver with section times
            reference_data: DataFrame with all drivers for comparison.
                          If None, uses driver_data as reference.
            top_percentile: Percentile threshold for "strength" (default: 20.0 = top 20%)

        Returns:
            List of dictionaries for each strength section:
            [
                {
                    'section': str,
                    'driver_median': float,
                    'field_median': float,
                    'percentile_rank': float,  # Lower is better
                    'advantage_seconds': float,
                    'advantage_percent': float
                },
                ...
            ]
            Sorted by advantage (best sections first)

        Example:
            >>> analyzer = SectionAnalyzer()
            >>> strengths = analyzer.identify_driver_strengths(driver_df, all_drivers_df, top_percentile=20)
            >>> for strength in strengths:
            ...     print(f"{strength['section']}: Top {strength['percentile_rank']:.1f}%")
        """
        if driver_data.empty:
            raise ValueError("Driver data is empty")

        # Use driver_data as reference if not provided
        if reference_data is None:
            reference_data = driver_data

        # Find section columns
        section_columns = [col for col in driver_data.columns if col.endswith('_SECONDS')]

        if not section_columns:
            raise ValueError("No section time columns found")

        strengths = []

        for section_col in section_columns:
            section_name = section_col.replace('_SECONDS', '')

            # Calculate driver's median time
            driver_valid = driver_data[section_col].dropna()
            driver_valid = driver_valid[driver_valid > 0]

            if len(driver_valid) == 0:
                continue

            driver_median = float(driver_valid.median())

            # Calculate field statistics
            field_valid = reference_data[section_col].dropna()
            field_valid = field_valid[field_valid > 0]

            if len(field_valid) == 0:
                continue

            field_median = float(field_valid.median())

            # Calculate percentile rank (what % of field is faster)
            percentile_rank = float((field_valid < driver_median).sum() / len(field_valid) * 100)

            # Check if this is a strength (driver is in top percentile)
            if percentile_rank <= top_percentile:
                advantage_seconds = field_median - driver_median
                advantage_percent = (advantage_seconds / field_median) * 100

                strengths.append({
                    'section': section_name,
                    'driver_median': driver_median,
                    'field_median': field_median,
                    'percentile_rank': percentile_rank,
                    'advantage_seconds': advantage_seconds,
                    'advantage_percent': advantage_percent
                })

        # Sort by advantage (best sections first)
        strengths.sort(key=lambda x: x['advantage_seconds'], reverse=True)

        return strengths

    def identify_improvement_areas(
        self,
        driver_data: pd.DataFrame,
        optimal_ghost: Dict[str, Dict[str, Any]],
        gap_threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Identify sections where driver has significant gap to optimal ghost.

        Finds sections where the driver's median time exceeds the optimal
        ghost time by more than the specified threshold percentage.

        Args:
            driver_data: DataFrame for a specific driver with section times
            optimal_ghost: Optimal ghost data from OptimalGhostAnalyzer
            gap_threshold: Minimum gap percentage to flag as improvement area (default: 2.0%)

        Returns:
            List of dictionaries for each improvement area:
            [
                {
                    'section': str,
                    'driver_median': float,
                    'optimal_time': float,
                    'gap_seconds': float,
                    'gap_percent': float,
                    'priority': str  # 'CRITICAL', 'HIGH', 'MEDIUM'
                },
                ...
            ]
            Sorted by gap (largest gaps first)

        Example:
            >>> analyzer = SectionAnalyzer()
            >>> improvements = analyzer.identify_improvement_areas(driver_df, optimal_ghost, gap_threshold=2.0)
            >>> for area in improvements:
            ...     print(f"{area['section']}: {area['gap_percent']:.1f}% gap ({area['priority']})")
        """
        if driver_data.empty:
            raise ValueError("Driver data is empty")

        if not optimal_ghost:
            raise ValueError("Optimal ghost data is empty")

        improvement_areas = []

        for section_name, ghost_data in optimal_ghost.items():
            section_col = f"{section_name}_SECONDS"

            if section_col not in driver_data.columns:
                continue

            # Calculate driver's median time
            driver_valid = driver_data[section_col].dropna()
            driver_valid = driver_valid[driver_valid > 0]

            if len(driver_valid) == 0:
                continue

            driver_median = float(driver_valid.median())
            optimal_time = ghost_data['time']

            # Calculate gap
            gap_seconds = driver_median - optimal_time
            gap_percent = (gap_seconds / optimal_time) * 100

            # Check if gap exceeds threshold
            if gap_percent >= gap_threshold:
                # Determine priority level
                if gap_percent >= 5.0:
                    priority = 'CRITICAL'
                elif gap_percent >= 3.0:
                    priority = 'HIGH'
                else:
                    priority = 'MEDIUM'

                improvement_areas.append({
                    'section': section_name,
                    'driver_median': driver_median,
                    'optimal_time': optimal_time,
                    'gap_seconds': gap_seconds,
                    'gap_percent': gap_percent,
                    'priority': priority
                })

        # Sort by gap (largest first)
        improvement_areas.sort(key=lambda x: x['gap_seconds'], reverse=True)

        return improvement_areas

    def analyze_section_consistency(
        self,
        driver_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze driver consistency in each section.

        Measures variation in section times using coefficient of variation
        and identifies sections with high inconsistency.

        Args:
            driver_data: DataFrame for a specific driver with section times

        Returns:
            Dictionary with consistency metrics per section:
            {
                'S1': {
                    'mean': float,
                    'std': float,
                    'cv': float,  # Coefficient of variation (std/mean)
                    'range': float,  # max - min
                    'consistency_score': float  # 0-100, higher is more consistent
                },
                ...
            }

        Example:
            >>> analyzer = SectionAnalyzer()
            >>> consistency = analyzer.analyze_section_consistency(driver_df)
            >>> for section, metrics in consistency.items():
            ...     print(f"{section}: {metrics['consistency_score']:.1f}/100 consistency")
        """
        if driver_data.empty:
            raise ValueError("Driver data is empty")

        section_columns = [col for col in driver_data.columns if col.endswith('_SECONDS')]

        if not section_columns:
            raise ValueError("No section time columns found")

        consistency_metrics = {}

        for section_col in section_columns:
            section_name = section_col.replace('_SECONDS', '')

            # Get valid times
            valid_times = driver_data[section_col].dropna()
            valid_times = valid_times[valid_times > 0]

            if len(valid_times) < 2:
                continue

            mean_time = float(valid_times.mean())
            std_time = float(valid_times.std())
            min_time = float(valid_times.min())
            max_time = float(valid_times.max())

            # Coefficient of variation (normalized std)
            cv = (std_time / mean_time) if mean_time > 0 else 0

            # Consistency score (0-100, higher is better)
            # Perfect consistency (cv=0) = 100, high cv = lower score
            consistency_score = max(0, 100 * (1 - min(cv / 0.1, 1)))

            consistency_metrics[section_name] = {
                'mean': mean_time,
                'std': std_time,
                'cv': cv,
                'range': max_time - min_time,
                'consistency_score': consistency_score,
                'sample_count': len(valid_times)
            }

        return consistency_metrics

    def compare_driver_sections(
        self,
        driver_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a detailed comparison of driver performance across all sections.

        Args:
            driver_data: DataFrame for specific driver
            reference_data: DataFrame with all drivers for field comparison

        Returns:
            DataFrame with one row per section containing:
            - section: Section name
            - driver_median: Driver's median time
            - field_median: Field median time
            - driver_best: Driver's best time
            - field_best: Field best time
            - gap_to_field: Gap to field median
            - gap_to_best: Gap to field best
            - percentile_rank: Driver's percentile in field
        """
        if driver_data.empty or reference_data.empty:
            raise ValueError("Driver data or reference data is empty")

        section_columns = [col for col in driver_data.columns if col.endswith('_SECONDS')]

        if not section_columns:
            raise ValueError("No section time columns found")

        comparison_data = []

        for section_col in section_columns:
            section_name = section_col.replace('_SECONDS', '')

            # Driver stats
            driver_valid = driver_data[section_col].dropna()
            driver_valid = driver_valid[driver_valid > 0]

            if len(driver_valid) == 0:
                continue

            driver_median = float(driver_valid.median())
            driver_best = float(driver_valid.min())

            # Field stats
            field_valid = reference_data[section_col].dropna()
            field_valid = field_valid[field_valid > 0]

            if len(field_valid) == 0:
                continue

            field_median = float(field_valid.median())
            field_best = float(field_valid.min())

            # Calculate gaps
            gap_to_field = driver_median - field_median
            gap_to_best = driver_best - field_best

            # Percentile rank
            percentile_rank = float((field_valid < driver_median).sum() / len(field_valid) * 100)

            comparison_data.append({
                'section': section_name,
                'driver_median': driver_median,
                'field_median': field_median,
                'driver_best': driver_best,
                'field_best': field_best,
                'gap_to_field': gap_to_field,
                'gap_to_best': gap_to_best,
                'percentile_rank': percentile_rank
            })

        return pd.DataFrame(comparison_data)

    def get_section_improvement_potential(
        self,
        driver_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate improvement potential for each section based on driver's own variance.

        Compares driver's median time to their best time to identify sections
        where they have shown they can go faster.

        Args:
            driver_data: DataFrame for a specific driver

        Returns:
            Dictionary mapping section names to improvement potential in seconds
            {
                'S1': 0.234,  # Driver could improve by 0.234s based on their best lap
                'S2': 0.156,
                ...
            }
        """
        if driver_data.empty:
            raise ValueError("Driver data is empty")

        section_columns = [col for col in driver_data.columns if col.endswith('_SECONDS')]

        if not section_columns:
            raise ValueError("No section time columns found")

        improvement_potential = {}

        for section_col in section_columns:
            section_name = section_col.replace('_SECONDS', '')

            valid_times = driver_data[section_col].dropna()
            valid_times = valid_times[valid_times > 0]

            if len(valid_times) < 2:
                continue

            median_time = float(valid_times.median())
            best_time = float(valid_times.min())

            # Potential improvement is the gap between median and best
            potential = median_time - best_time

            if potential > 0:
                improvement_potential[section_name] = potential

        return improvement_potential
