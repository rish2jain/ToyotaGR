"""
Optimal Ghost Analyzer Module

This module creates optimal ghost laps by combining the best section times
across all drivers and provides detailed comparisons between individual driver
performance and the optimal ghost.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class OptimalGhostAnalyzer:
    """
    Analyzer for creating optimal ghost laps and comparing driver performance.

    The optimal ghost represents a theoretical perfect lap by combining the best
    section times from all drivers. This provides an absolute performance benchmark
    for identifying improvement opportunities.
    """

    def __init__(self):
        """Initialize the OptimalGhostAnalyzer."""
        self.optimal_ghost: Optional[Dict[str, Dict[str, Any]]] = None

    def create_optimal_ghost(
        self,
        section_data: pd.DataFrame,
        percentile: int = 95
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create an optimal ghost lap by finding the best times for each section.

        The function identifies the fastest section times (by default, the 5th percentile
        representing the fastest 5% of laps) for each section in the dataset.

        Args:
            section_data: DataFrame containing lap data with columns:
                - DRIVER_NUMBER: Driver identifier
                - S1_SECONDS, S2_SECONDS, S3_SECONDS: Section times
                - Or any columns ending with '_SECONDS' for section times
            percentile: Percentile to use for "best" time (default 95 = top 5%).
                       Lower percentile = faster time (5th percentile = fastest 5%)

        Returns:
            Dictionary with structure:
            {
                'S1': {
                    'time': float,  # Best section time in seconds
                    'best_driver': int,  # Driver who achieved this time
                    'percentile': int  # Percentile used
                },
                ...
            }

        Example:
            >>> analyzer = OptimalGhostAnalyzer()
            >>> optimal = analyzer.create_optimal_ghost(section_df, percentile=95)
            >>> print(f"S1 optimal: {optimal['S1']['time']:.3f}s by driver {optimal['S1']['best_driver']}")
        """
        # Find all section columns (columns ending with '_SECONDS')
        section_columns = [col for col in section_data.columns if col.endswith('_SECONDS')]

        if not section_columns:
            raise ValueError("No section time columns found (expected columns ending with '_SECONDS')")

        optimal_ghost = {}

        for section_col in section_columns:
            # Remove the '_SECONDS' suffix to get section name
            section_name = section_col.replace('_SECONDS', '')

            # Filter out invalid times (NaN, zero, or negative)
            valid_times = section_data[section_col].dropna()
            valid_times = valid_times[valid_times > 0]

            if len(valid_times) == 0:
                continue

            # Calculate the target percentile (lower percentile = faster time)
            # 95th percentile means we want the value at 5% from the bottom
            target_percentile = 100 - percentile
            best_time = np.percentile(valid_times, target_percentile)

            # Find the driver who achieved a time closest to this best time
            # (in case exact match doesn't exist)
            time_diffs = np.abs(section_data[section_col] - best_time)
            best_lap_idx = time_diffs.idxmin()
            best_driver = section_data.loc[best_lap_idx, 'DRIVER_NUMBER']
            actual_best_time = section_data.loc[best_lap_idx, section_col]

            optimal_ghost[section_name] = {
                'time': float(actual_best_time),
                'best_driver': int(best_driver),
                'percentile': percentile
            }

        self.optimal_ghost = optimal_ghost
        return optimal_ghost

    def analyze_driver_vs_ghost(
        self,
        driver_data: pd.DataFrame,
        optimal_ghost: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare a driver's performance against the optimal ghost.

        This function analyzes the gap between a driver's median section times
        and the optimal ghost times, identifying the top improvement opportunities.

        Args:
            driver_data: DataFrame containing lap data for a specific driver
                        with section time columns (e.g., S1_SECONDS, S2_SECONDS)
            optimal_ghost: Optimal ghost data from create_optimal_ghost()

        Returns:
            Dictionary containing:
            {
                'driver_number': int,
                'total_gap': float,  # Total time gap in seconds
                'sections': {
                    'S1': {
                        'median_time': float,
                        'optimal_time': float,
                        'gap_seconds': float,
                        'gap_percent': float
                    },
                    ...
                },
                'improvement_opportunities': [
                    {
                        'section': str,
                        'gap_seconds': float,
                        'gap_percent': float,
                        'priority': str  # 'HIGH', 'MEDIUM', or 'LOW'
                    },
                    ...
                ],
                'top_3_improvements': List[Dict]  # Top 3 opportunities
            }

        Example:
            >>> insights = analyzer.analyze_driver_vs_ghost(driver_df, optimal_ghost)
            >>> for opp in insights['top_3_improvements']:
            ...     print(f"{opp['section']}: {opp['gap_seconds']:.3f}s gap ({opp['priority']})")
        """
        if driver_data.empty:
            raise ValueError("Driver data is empty")

        # Get driver number (assuming all rows are for the same driver)
        driver_number = driver_data['DRIVER_NUMBER'].iloc[0]

        sections_analysis = {}
        total_gap = 0.0
        improvement_opportunities = []

        for section_name, ghost_data in optimal_ghost.items():
            section_col = f"{section_name}_SECONDS"

            if section_col not in driver_data.columns:
                continue

            # Calculate median time for this driver in this section
            valid_times = driver_data[section_col].dropna()
            valid_times = valid_times[valid_times > 0]

            if len(valid_times) == 0:
                continue

            median_time = float(valid_times.median())
            optimal_time = ghost_data['time']

            # Calculate gaps
            gap_seconds = median_time - optimal_time
            gap_percent = (gap_seconds / optimal_time) * 100

            sections_analysis[section_name] = {
                'median_time': median_time,
                'optimal_time': optimal_time,
                'gap_seconds': gap_seconds,
                'gap_percent': gap_percent
            }

            total_gap += gap_seconds

            # Determine priority level based on gap percentage
            if gap_percent > 5.0:
                priority = 'HIGH'
            elif gap_percent > 2.0:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            improvement_opportunities.append({
                'section': section_name,
                'gap_seconds': gap_seconds,
                'gap_percent': gap_percent,
                'priority': priority,
                'median_time': median_time,
                'optimal_time': optimal_time
            })

        # Sort by gap_seconds (largest gap first) and get top 3
        improvement_opportunities.sort(key=lambda x: x['gap_seconds'], reverse=True)
        top_3_improvements = improvement_opportunities[:3]

        return {
            'driver_number': int(driver_number),
            'total_gap': total_gap,
            'sections': sections_analysis,
            'improvement_opportunities': improvement_opportunities,
            'top_3_improvements': top_3_improvements
        }

    def get_ghost_total_time(self, optimal_ghost: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
        """
        Calculate the total lap time of the optimal ghost.

        Args:
            optimal_ghost: Optimal ghost data. If None, uses stored optimal_ghost.

        Returns:
            Total optimal lap time in seconds
        """
        ghost = optimal_ghost if optimal_ghost is not None else self.optimal_ghost

        if ghost is None:
            raise ValueError("No optimal ghost available. Run create_optimal_ghost() first.")

        return sum(section['time'] for section in ghost.values())

    def compare_multiple_drivers(
        self,
        section_data: pd.DataFrame,
        optimal_ghost: Dict[str, Dict[str, Any]],
        driver_numbers: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple drivers against the optimal ghost.

        Args:
            section_data: Full dataset with all driver laps
            optimal_ghost: Optimal ghost data from create_optimal_ghost()
            driver_numbers: List of driver numbers to analyze. If None, analyzes all drivers.

        Returns:
            DataFrame with comparison results for each driver, sorted by total gap
        """
        if driver_numbers is None:
            driver_numbers = section_data['DRIVER_NUMBER'].unique()

        results = []

        for driver_num in driver_numbers:
            driver_data = section_data[section_data['DRIVER_NUMBER'] == driver_num]

            if driver_data.empty:
                continue

            analysis = self.analyze_driver_vs_ghost(driver_data, optimal_ghost)

            result_row = {
                'driver_number': analysis['driver_number'],
                'total_gap': analysis['total_gap']
            }

            # Add section-specific gaps
            for section_name, section_data in analysis['sections'].items():
                result_row[f'{section_name}_gap'] = section_data['gap_seconds']
                result_row[f'{section_name}_gap_pct'] = section_data['gap_percent']

            results.append(result_row)

        df_results = pd.DataFrame(results)
        return df_results.sort_values('total_gap')
