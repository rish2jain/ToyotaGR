"""
Racing Line Reconstruction Module

This module reconstructs approximate racing lines from telemetry data
(speed, gear, brake, throttle) and compares lines between drivers.

Method:
1. Identify corner sections from speed/brake patterns
2. Estimate corner geometry from minimum speed
3. Find entry/apex/exit points from brake/throttle timing
4. Interpolate smooth lines through corners
5. Compare racing lines between drivers

Physics formulas used:
- Corner radius: r = v² / (g * lateral_g)
- Lateral g: Assumed ~1.5-2.5g for racing conditions
- Track width: Assumed ~12-15 meters typical
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks, savgol_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RacingLineReconstructor:
    """
    Reconstruct approximate racing lines from speed + gear + brake telemetry.

    This class analyzes telemetry data to estimate the racing line taken through
    corners and straights, allowing for comparison between drivers and identification
    of different approaches to track sections.
    """

    def __init__(
        self,
        lateral_g_assumption: float = 1.8,
        track_width_m: float = 12.0,
        gravity: float = 9.81
    ):
        """
        Initialize the RacingLineReconstructor.

        Args:
            lateral_g_assumption: Assumed lateral G-force for corner speed calculations (default: 1.8g)
            track_width_m: Typical track width in meters (default: 12.0m)
            gravity: Gravitational acceleration in m/s² (default: 9.81)
        """
        self.lateral_g = lateral_g_assumption
        self.track_width = track_width_m
        self.g = gravity

    def reconstruct_line(
        self,
        telemetry_data: pd.DataFrame,
        track_sections: Optional[pd.DataFrame] = None,
        speed_col: str = 'gps_speed',
        brake_col: str = 'brake_f',
        throttle_col: str = 'aps',
        distance_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct racing line from telemetry data.

        Args:
            telemetry_data: DataFrame with speed, brake, throttle data
            track_sections: Optional DataFrame with section definitions
            speed_col: Column name for speed data (km/h)
            brake_col: Column name for brake pressure (0-100 or 0-1)
            throttle_col: Column name for throttle position (0-100 or 0-1)
            distance_col: Optional column for distance traveled

        Returns:
            Dictionary containing:
            - corners: List of identified corner data
            - trajectory: Reconstructed racing line points
            - statistics: Line statistics and metrics
        """
        if telemetry_data.empty:
            raise ValueError("Telemetry data is empty")

        # Ensure required columns exist
        if speed_col not in telemetry_data.columns:
            raise ValueError(f"Speed column '{speed_col}' not found in telemetry data")

        # Create distance column if not provided
        if distance_col is None or distance_col not in telemetry_data.columns:
            telemetry_data = telemetry_data.copy()
            telemetry_data['distance_pct'] = np.linspace(0, 100, len(telemetry_data))
            distance_col = 'distance_pct'

        # Identify corners
        corners = self._identify_corners(
            telemetry_data,
            speed_col=speed_col,
            brake_col=brake_col,
            distance_col=distance_col
        )

        # Estimate geometry for each corner
        for corner in corners:
            self._estimate_corner_geometry(corner, speed_col=speed_col)
            self._find_brake_point(corner, brake_col=brake_col, distance_col=distance_col)
            self._find_apex(corner, speed_col=speed_col, distance_col=distance_col)

            if throttle_col in telemetry_data.columns:
                self._find_throttle_point(corner, throttle_col=throttle_col, distance_col=distance_col)
            else:
                # Estimate throttle point from speed increase
                corner['throttle_point'] = corner['apex'] + (corner['exit'] - corner['apex']) * 0.3

        # Build trajectory
        trajectory = self._build_trajectory(
            corners,
            telemetry_data,
            distance_col=distance_col,
            speed_col=speed_col
        )

        # Calculate statistics
        statistics = self._calculate_line_statistics(corners, trajectory)

        return {
            'corners': corners,
            'trajectory': trajectory,
            'statistics': statistics,
            'telemetry': telemetry_data
        }

    def _identify_corners(
        self,
        telemetry_data: pd.DataFrame,
        speed_col: str = 'gps_speed',
        brake_col: str = 'brake_f',
        distance_col: str = 'distance_pct',
        min_corner_duration: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify corner sections from speed and brake patterns.

        Corners are characterized by:
        - Speed reduction (local minima)
        - Brake application
        - Sustained lower speed

        Args:
            telemetry_data: Telemetry DataFrame
            speed_col: Speed column name
            brake_col: Brake column name
            distance_col: Distance column name
            min_corner_duration: Minimum data points for a corner

        Returns:
            List of corner dictionaries with entry/apex/exit indices
        """
        # Smooth speed data to reduce noise
        speed = telemetry_data[speed_col].values
        window_len = min(21, len(speed) // 2 * 2 - 1) if len(speed) > 21 else min(11, len(speed) // 2 * 2 - 1)
        speed_smooth = savgol_filter(speed, window_length=window_len, polyorder=3)

        # Find speed local minima (potential apexes)
        # Reduced prominence to catch more corners
        minima_indices, _ = find_peaks(-speed_smooth, distance=min_corner_duration * 3, prominence=3)

        corners = []

        for apex_idx in minima_indices:
            # Look backward for corner entry (brake application or speed decrease)
            entry_idx = self._find_corner_entry(
                telemetry_data,
                apex_idx,
                speed_col=speed_col,
                brake_col=brake_col
            )

            # Look forward for corner exit (speed increase)
            exit_idx = self._find_corner_exit(
                telemetry_data,
                apex_idx,
                speed_col=speed_col
            )

            # Validate corner duration
            if exit_idx - entry_idx >= min_corner_duration:
                corner_data = telemetry_data.iloc[entry_idx:exit_idx+1].copy()

                corners.append({
                    'entry_idx': entry_idx,
                    'apex_idx': apex_idx,
                    'exit_idx': exit_idx,
                    'entry': telemetry_data.iloc[entry_idx][distance_col],
                    'apex': telemetry_data.iloc[apex_idx][distance_col],
                    'exit': telemetry_data.iloc[exit_idx][distance_col],
                    'data': corner_data,
                    'corner_number': len(corners) + 1
                })

        logger.info(f"Identified {len(corners)} corners from telemetry")
        return corners

    def _find_corner_entry(
        self,
        telemetry_data: pd.DataFrame,
        apex_idx: int,
        speed_col: str = 'gps_speed',
        brake_col: str = 'brake_f',
        lookback: int = 50
    ) -> int:
        """Find corner entry point by looking backward from apex."""
        start_idx = max(0, apex_idx - lookback)

        # Look for brake application or significant speed decrease
        if brake_col in telemetry_data.columns:
            brake_data = telemetry_data[brake_col].iloc[start_idx:apex_idx]
            brake_threshold = brake_data.max() * 0.3

            # Find first significant brake application
            brake_points = brake_data[brake_data > brake_threshold]
            if len(brake_points) > 0:
                return start_idx + brake_points.index[0]

        # Fallback: find where speed starts decreasing significantly
        speed_data = telemetry_data[speed_col].iloc[start_idx:apex_idx]
        speed_gradient = np.gradient(speed_data)

        # Find sustained speed decrease
        for i in range(len(speed_gradient) - 3):
            if np.mean(speed_gradient[i:i+3]) < -2:  # Sustained deceleration
                return start_idx + i

        # Default to halfway between start and apex
        return start_idx + (apex_idx - start_idx) // 3

    def _find_corner_exit(
        self,
        telemetry_data: pd.DataFrame,
        apex_idx: int,
        speed_col: str = 'gps_speed',
        lookahead: int = 50
    ) -> int:
        """Find corner exit point by looking forward from apex."""
        end_idx = min(len(telemetry_data) - 1, apex_idx + lookahead)

        speed_data = telemetry_data[speed_col].iloc[apex_idx:end_idx]
        speed_gradient = np.gradient(speed_data)

        # Find sustained speed increase
        for i in range(len(speed_gradient) - 3):
            if np.mean(speed_gradient[i:i+3]) > 2:  # Sustained acceleration
                # Continue to find peak acceleration
                for j in range(i + 1, len(speed_gradient) - 1):
                    if speed_gradient[j] < speed_gradient[j-1] * 0.5:
                        return apex_idx + j

        # Default to 2/3 of the way to end
        return apex_idx + (end_idx - apex_idx) * 2 // 3

    def _estimate_corner_geometry(
        self,
        corner: Dict[str, Any],
        speed_col: str = 'gps_speed'
    ):
        """
        Estimate corner radius from minimum speed.

        Formula: radius = v² / (g * lateral_g)
        where:
        - v = minimum speed (m/s)
        - g = gravitational acceleration (9.81 m/s²)
        - lateral_g = assumed lateral G-force (1.5-2.5)

        Args:
            corner: Corner dictionary to update with geometry
            speed_col: Speed column name
        """
        # Get minimum speed in corner
        min_speed_kph = corner['data'][speed_col].min()
        min_speed_ms = min_speed_kph / 3.6  # Convert km/h to m/s

        # Calculate corner radius
        # r = v² / (g * lateral_g)
        radius_m = (min_speed_ms ** 2) / (self.g * self.lateral_g)

        # Estimate corner arc length from entry to exit
        arc_length = corner['exit'] - corner['entry']

        # Estimate corner angle (assuming distance_pct is proportional)
        # For a full lap, 100% = lap length
        # This is approximate without actual GPS coordinates

        corner['min_speed_kph'] = min_speed_kph
        corner['min_speed_ms'] = min_speed_ms
        corner['radius_m'] = radius_m
        corner['arc_length'] = arc_length
        corner['lateral_g_estimated'] = self.lateral_g

        logger.debug(f"Corner {corner['corner_number']}: radius={radius_m:.1f}m, min_speed={min_speed_kph:.1f}kph")

    def _find_brake_point(
        self,
        corner: Dict[str, Any],
        brake_col: str = 'brake_f',
        distance_col: str = 'distance_pct'
    ):
        """
        Find corner brake point from brake application.

        Args:
            corner: Corner dictionary to update
            brake_col: Brake pressure column name
            distance_col: Distance column name
        """
        if brake_col not in corner['data'].columns:
            corner['brake_point'] = corner['entry']
            corner['max_brake_pressure'] = 0
            return

        brake_data = corner['data'][brake_col]

        # Find maximum brake pressure
        max_brake_idx = brake_data.idxmax()
        max_brake = brake_data.max()

        # Find initial brake application (threshold crossing)
        brake_threshold = max_brake * 0.2
        brake_applied = brake_data[brake_data > brake_threshold]

        if len(brake_applied) > 0:
            brake_point_idx = brake_applied.index[0]
            corner['brake_point'] = corner['data'].loc[brake_point_idx, distance_col]
            corner['max_brake_pressure'] = max_brake
        else:
            corner['brake_point'] = corner['entry']
            corner['max_brake_pressure'] = 0

    def _find_apex(
        self,
        corner: Dict[str, Any],
        speed_col: str = 'gps_speed',
        distance_col: str = 'distance_pct'
    ):
        """
        Find corner apex from minimum speed point.

        Args:
            corner: Corner dictionary to update
            speed_col: Speed column name
            distance_col: Distance column name
        """
        # Apex is at minimum speed
        min_speed_idx = corner['data'][speed_col].idxmin()
        corner['apex_point'] = corner['data'].loc[min_speed_idx, distance_col]
        corner['apex_speed'] = corner['data'].loc[min_speed_idx, speed_col]

    def _find_throttle_point(
        self,
        corner: Dict[str, Any],
        throttle_col: str = 'aps',
        distance_col: str = 'distance_pct'
    ):
        """
        Find corner exit throttle application point.

        Args:
            corner: Corner dictionary to update
            throttle_col: Throttle position column name
            distance_col: Distance column name
        """
        if throttle_col not in corner['data'].columns:
            corner['throttle_point'] = corner['exit']
            corner['max_throttle'] = 0
            return

        throttle_data = corner['data'][throttle_col]

        # Find where throttle application begins after apex
        apex_idx = corner['apex_idx']
        post_apex_data = corner['data'].loc[corner['data'].index >= apex_idx]

        if throttle_col in post_apex_data.columns:
            throttle_post_apex = post_apex_data[throttle_col]

            # Find significant throttle application (>30%)
            throttle_threshold = 30
            throttle_applied = throttle_post_apex[throttle_post_apex > throttle_threshold]

            if len(throttle_applied) > 0:
                throttle_point_idx = throttle_applied.index[0]
                corner['throttle_point'] = post_apex_data.loc[throttle_point_idx, distance_col]
            else:
                corner['throttle_point'] = corner['exit']
        else:
            corner['throttle_point'] = corner['exit']

        corner['max_throttle'] = throttle_data.max()

    def _build_trajectory(
        self,
        corners: List[Dict[str, Any]],
        telemetry_data: pd.DataFrame,
        distance_col: str = 'distance_pct',
        speed_col: str = 'gps_speed',
        points_per_section: int = 50
    ) -> pd.DataFrame:
        """
        Build complete racing line trajectory from corners and straights.

        Args:
            corners: List of corner dictionaries
            telemetry_data: Full telemetry DataFrame
            distance_col: Distance column name
            speed_col: Speed column name
            points_per_section: Number of interpolation points per section

        Returns:
            DataFrame with trajectory points including distance, speed, section type
        """
        trajectory_points = []

        # Handle case with no corners
        if not corners:
            return pd.DataFrame({
                'distance': telemetry_data[distance_col],
                'speed': telemetry_data[speed_col],
                'section_type': ['straight'] * len(telemetry_data),
                'lateral_offset': [0] * len(telemetry_data)
            })

        last_exit = 0

        for i, corner in enumerate(corners):
            # Add straight section before corner
            if corner['entry'] > last_exit:
                straight_data = telemetry_data[
                    (telemetry_data[distance_col] >= last_exit) &
                    (telemetry_data[distance_col] < corner['entry'])
                ]

                for _, row in straight_data.iterrows():
                    trajectory_points.append({
                        'distance': row[distance_col],
                        'speed': row[speed_col],
                        'section_type': 'straight',
                        'section_id': f'straight_{i}',
                        'lateral_offset': 0  # Straights assumed on centerline
                    })

            # Add corner section with estimated lateral offset
            corner_data = corner['data']

            for _, row in corner_data.iterrows():
                # Estimate lateral position based on corner geometry
                # This is approximate without GPS data
                progress = (row[distance_col] - corner['entry']) / (corner['exit'] - corner['entry'])

                # Smooth curve from outside to inside (apex) to outside
                # Using sine wave approximation
                lateral_offset = np.sin(progress * np.pi) * (self.track_width / 2)

                trajectory_points.append({
                    'distance': row[distance_col],
                    'speed': row[speed_col],
                    'section_type': 'corner',
                    'section_id': f'corner_{corner["corner_number"]}',
                    'corner_number': corner['corner_number'],
                    'lateral_offset': lateral_offset,
                    'radius': corner['radius_m']
                })

            last_exit = corner['exit']

        # Add final straight section
        if last_exit < telemetry_data[distance_col].max():
            final_straight = telemetry_data[telemetry_data[distance_col] >= last_exit]

            for _, row in final_straight.iterrows():
                trajectory_points.append({
                    'distance': row[distance_col],
                    'speed': row[speed_col],
                    'section_type': 'straight',
                    'section_id': f'straight_final',
                    'lateral_offset': 0
                })

        trajectory_df = pd.DataFrame(trajectory_points)
        logger.info(f"Built trajectory with {len(trajectory_df)} points")

        return trajectory_df

    def _calculate_line_statistics(
        self,
        corners: List[Dict[str, Any]],
        trajectory: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the racing line.

        Args:
            corners: List of corner dictionaries
            trajectory: Trajectory DataFrame

        Returns:
            Dictionary with line statistics
        """
        stats = {
            'total_corners': len(corners),
            'total_distance': trajectory['distance'].max() if not trajectory.empty else 0,
            'avg_speed': trajectory['speed'].mean() if not trajectory.empty else 0,
            'min_speed': trajectory['speed'].min() if not trajectory.empty else 0,
            'max_speed': trajectory['speed'].max() if not trajectory.empty else 0,
        }

        if corners:
            stats['avg_corner_radius'] = np.mean([c['radius_m'] for c in corners])
            stats['min_corner_speed'] = min([c['min_speed_kph'] for c in corners])
            stats['max_corner_speed'] = max([c['min_speed_kph'] for c in corners])
            stats['avg_lateral_g'] = np.mean([c['lateral_g_estimated'] for c in corners])

        return stats

    def compare_racing_lines(
        self,
        driver1_telem: pd.DataFrame,
        driver2_telem: pd.DataFrame,
        driver1_label: str = "Driver 1",
        driver2_label: str = "Driver 2",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare two drivers' racing lines.

        Args:
            driver1_telem: First driver's telemetry DataFrame
            driver2_telem: Second driver's telemetry DataFrame
            driver1_label: Label for first driver
            driver2_label: Label for second driver
            **kwargs: Additional arguments passed to reconstruct_line()

        Returns:
            Dictionary containing:
            - driver1_line: First driver's reconstructed line
            - driver2_line: Second driver's reconstructed line
            - differences: Corner-by-corner comparison
            - summary: Overall comparison summary
        """
        # Reconstruct both lines
        line1 = self.reconstruct_line(driver1_telem, **kwargs)
        line2 = self.reconstruct_line(driver2_telem, **kwargs)

        # Compare corners
        differences = self._compare_corners(
            line1['corners'],
            line2['corners'],
            driver1_label,
            driver2_label
        )

        # Generate summary
        summary = self._generate_comparison_summary(
            line1,
            line2,
            differences,
            driver1_label,
            driver2_label
        )

        return {
            'driver1_line': line1,
            'driver2_line': line2,
            'driver1_label': driver1_label,
            'driver2_label': driver2_label,
            'differences': differences,
            'summary': summary
        }

    def _compare_corners(
        self,
        corners1: List[Dict[str, Any]],
        corners2: List[Dict[str, Any]],
        driver1_label: str,
        driver2_label: str
    ) -> List[Dict[str, Any]]:
        """
        Compare corners between two drivers.

        Args:
            corners1: First driver's corners
            corners2: Second driver's corners
            driver1_label: First driver label
            driver2_label: Second driver label

        Returns:
            List of corner comparison dictionaries
        """
        # Match corners by position (assuming similar number of corners)
        min_corners = min(len(corners1), len(corners2))

        comparisons = []

        for i in range(min_corners):
            c1 = corners1[i]
            c2 = corners2[i]

            # Calculate deltas
            entry_delta = c1['entry'] - c2['entry']
            apex_delta = c1['apex'] - c2['apex']
            exit_delta = c1['exit'] - c2['exit']

            brake_point_delta = c1.get('brake_point', c1['entry']) - c2.get('brake_point', c2['entry'])
            apex_speed_delta = c1.get('apex_speed', 0) - c2.get('apex_speed', 0)
            radius_delta = c1['radius_m'] - c2['radius_m']

            # Determine who's faster/better
            if apex_speed_delta > 0:
                faster_apex = driver1_label
            elif apex_speed_delta < 0:
                faster_apex = driver2_label
            else:
                faster_apex = "Equal"

            if brake_point_delta < 0:
                later_braking = driver1_label
            elif brake_point_delta > 0:
                later_braking = driver2_label
            else:
                later_braking = "Equal"

            comparisons.append({
                'corner_number': i + 1,
                'entry_delta': entry_delta,
                'apex_delta': apex_delta,
                'exit_delta': exit_delta,
                'brake_point_delta': brake_point_delta,
                'apex_speed_delta_kph': apex_speed_delta,
                'radius_delta_m': radius_delta,
                'faster_apex_speed': faster_apex,
                'later_braking': later_braking,
                f'{driver1_label}_brake_point': c1.get('brake_point', c1['entry']),
                f'{driver2_label}_brake_point': c2.get('brake_point', c2['entry']),
                f'{driver1_label}_apex_speed': c1.get('apex_speed', 0),
                f'{driver2_label}_apex_speed': c2.get('apex_speed', 0),
                f'{driver1_label}_radius': c1['radius_m'],
                f'{driver2_label}_radius': c2['radius_m'],
            })

        return comparisons

    def _generate_comparison_summary(
        self,
        line1: Dict[str, Any],
        line2: Dict[str, Any],
        differences: List[Dict[str, Any]],
        driver1_label: str,
        driver2_label: str
    ) -> Dict[str, Any]:
        """Generate overall comparison summary."""
        if not differences:
            return {
                'comparison_possible': False,
                'reason': 'Insufficient corner data for comparison'
            }

        # Calculate aggregate metrics
        avg_apex_speed_delta = np.mean([d['apex_speed_delta_kph'] for d in differences])
        avg_brake_point_delta = np.mean([d['brake_point_delta'] for d in differences])

        # Count advantages
        driver1_faster_corners = sum(1 for d in differences if d['faster_apex_speed'] == driver1_label)
        driver2_faster_corners = sum(1 for d in differences if d['faster_apex_speed'] == driver2_label)

        driver1_later_braking = sum(1 for d in differences if d['later_braking'] == driver1_label)
        driver2_later_braking = sum(1 for d in differences if d['later_braking'] == driver2_label)

        # Overall speed comparison
        avg_speed1 = line1['statistics']['avg_speed']
        avg_speed2 = line2['statistics']['avg_speed']

        return {
            'comparison_possible': True,
            'total_corners_compared': len(differences),
            'avg_apex_speed_delta_kph': avg_apex_speed_delta,
            'avg_brake_point_delta': avg_brake_point_delta,
            f'{driver1_label}_faster_corners': driver1_faster_corners,
            f'{driver2_label}_faster_corners': driver2_faster_corners,
            f'{driver1_label}_later_braking_corners': driver1_later_braking,
            f'{driver2_label}_later_braking_corners': driver2_later_braking,
            f'{driver1_label}_avg_speed': avg_speed1,
            f'{driver2_label}_avg_speed': avg_speed2,
            'avg_speed_delta_kph': avg_speed1 - avg_speed2,
            'dominant_driver_apex': driver1_label if driver1_faster_corners > driver2_faster_corners else driver2_label,
            'dominant_driver_braking': driver1_label if driver1_later_braking > driver2_later_braking else driver2_label,
        }

    def calculate_racing_line_delta(
        self,
        line1: Dict[str, Any],
        line2: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate detailed differences between two racing lines.

        Args:
            line1: First driver's line (from reconstruct_line)
            line2: Second driver's line (from reconstruct_line)

        Returns:
            DataFrame with point-by-point differences
        """
        traj1 = line1['trajectory']
        traj2 = line2['trajectory']

        # Interpolate to common distance points
        common_distance = np.linspace(
            max(traj1['distance'].min(), traj2['distance'].min()),
            min(traj1['distance'].max(), traj2['distance'].max()),
            500
        )

        # Interpolate speeds
        speed1_interp = np.interp(common_distance, traj1['distance'], traj1['speed'])
        speed2_interp = np.interp(common_distance, traj2['distance'], traj2['speed'])

        # Interpolate lateral offsets if available
        if 'lateral_offset' in traj1.columns and 'lateral_offset' in traj2.columns:
            lateral1_interp = np.interp(common_distance, traj1['distance'], traj1['lateral_offset'])
            lateral2_interp = np.interp(common_distance, traj2['distance'], traj2['lateral_offset'])
        else:
            lateral1_interp = np.zeros_like(common_distance)
            lateral2_interp = np.zeros_like(common_distance)

        # Calculate deltas
        delta_df = pd.DataFrame({
            'distance': common_distance,
            'speed_line1': speed1_interp,
            'speed_line2': speed2_interp,
            'speed_delta': speed1_interp - speed2_interp,
            'lateral_line1': lateral1_interp,
            'lateral_line2': lateral2_interp,
            'lateral_delta': lateral1_interp - lateral2_interp,
        })

        return delta_df
