"""
Racing Line Reconstruction Demo

This script demonstrates the Racing Line Reconstruction feature with comprehensive
scenarios including single driver analysis, two-driver comparison, and corner-specific analysis.

Usage:
    python examples/racing_line_demo.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tactical.racing_line import RacingLineReconstructor
from src.utils.visualization import (
    create_racing_line_comparison,
    create_corner_analysis,
    create_speed_trace_comparison
)
from src.utils.track_layouts import get_track_layout


def generate_sample_telemetry(num_points=1000, num_corners=8, seed=42):
    """
    Generate synthetic telemetry data for demonstration purposes.

    This creates realistic-looking telemetry with speed variations,
    brake applications, and throttle inputs.

    Args:
        num_points: Number of telemetry samples
        num_corners: Number of corners on track
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic telemetry
    """
    np.random.seed(seed)

    # Distance along track (0-100%)
    distance_pct = np.linspace(0, 100, num_points)

    # Create corner locations
    corner_locations = np.linspace(10, 90, num_corners)

    # Initialize arrays
    speed = np.ones(num_points) * 180  # Base speed 180 km/h
    brake = np.zeros(num_points)
    throttle = np.ones(num_points) * 100
    gear = np.ones(num_points) * 5

    # Add corners with speed reduction
    for corner_center in corner_locations:
        # Corner influence spans ±7% distance for more pronounced effect
        corner_width = 7
        corner_mask = np.abs(distance_pct - corner_center) < corner_width

        # Speed profile (parabolic dip) - more pronounced
        distance_from_center = distance_pct - corner_center
        speed_reduction = 80 * (1 - (distance_from_center / corner_width)**2)
        speed_reduction = np.clip(speed_reduction, 0, 80)

        speed[corner_mask] -= speed_reduction[corner_mask]

        # Brake application before apex
        brake_zone = (distance_pct >= corner_center - 3) & (distance_pct <= corner_center)
        brake[brake_zone] = 80 + np.random.normal(0, 5, brake_zone.sum())

        # Throttle reduction in corner
        throttle_zone = (distance_pct >= corner_center - 2) & (distance_pct <= corner_center + 2)
        throttle[throttle_zone] = 40 + (distance_from_center[throttle_zone]**2) * 5

        # Gear reduction in corner
        gear[corner_mask] = np.clip(5 - speed_reduction[corner_mask] / 25, 2, 5)

    # Add noise
    speed += np.random.normal(0, 2, num_points)
    brake = np.clip(brake + np.random.normal(0, 3, num_points), 0, 100)
    throttle = np.clip(throttle + np.random.normal(0, 5, num_points), 0, 100)
    gear = np.clip(gear, 1, 6).astype(int)

    # Create DataFrame
    telemetry_df = pd.DataFrame({
        'distance_pct': distance_pct,
        'gps_speed': speed,
        'brake_f': brake,
        'aps': throttle,
        'gear': gear,
        'timestamp': np.arange(num_points) / 100.0  # 10 Hz sampling
    })

    return telemetry_df


def demo_single_driver_reconstruction():
    """
    Scenario 1: Single driver line reconstruction

    Demonstrates:
    - Loading telemetry data
    - Identifying corners
    - Reconstructing racing line
    - Visualizing on track map
    """
    print("\n" + "=" * 80)
    print("SCENARIO 1: Single Driver Line Reconstruction")
    print("=" * 80)

    # Generate sample telemetry
    print("\n1. Generating sample telemetry data...")
    telemetry = generate_sample_telemetry(num_points=1000, num_corners=8)
    print(f"   Generated {len(telemetry)} telemetry points")
    print(f"   Speed range: {telemetry['gps_speed'].min():.1f} - {telemetry['gps_speed'].max():.1f} km/h")

    # Initialize reconstructor
    print("\n2. Initializing Racing Line Reconstructor...")
    reconstructor = RacingLineReconstructor(
        lateral_g_assumption=1.8,  # Typical racing lateral G
        track_width_m=12.0  # Standard track width
    )
    print("   Configured with 1.8g lateral assumption, 12m track width")

    # Reconstruct line
    print("\n3. Reconstructing racing line from telemetry...")
    line = reconstructor.reconstruct_line(
        telemetry,
        speed_col='gps_speed',
        brake_col='brake_f',
        throttle_col='aps',
        distance_col='distance_pct'
    )

    # Display results
    corners = line['corners']
    print(f"\n   Identified {len(corners)} corners:")
    print("\n   Corner | Entry | Apex  | Exit  | Min Speed | Radius | Brake Pressure")
    print("   " + "-" * 75)

    for corner in corners:
        print(f"     {corner['corner_number']:2d}   | "
              f"{corner['entry']:5.1f} | "
              f"{corner['apex']:5.1f} | "
              f"{corner['exit']:5.1f} | "
              f"{corner['min_speed_kph']:7.1f} kph | "
              f"{corner['radius_m']:5.1f}m | "
              f"{corner.get('max_brake_pressure', 0):5.1f}%")

    # Display statistics
    stats = line['statistics']
    print(f"\n   Racing Line Statistics:")
    print(f"   - Total corners: {stats['total_corners']}")
    print(f"   - Average speed: {stats['avg_speed']:.1f} km/h")
    print(f"   - Speed range: {stats['min_speed']:.1f} - {stats['max_speed']:.1f} km/h")
    if 'avg_corner_radius' in stats:
        print(f"   - Average corner radius: {stats['avg_corner_radius']:.1f} m")
        print(f"   - Average corner speed: {stats['min_corner_speed']:.1f} km/h")

    print(f"\n   Trajectory contains {len(line['trajectory'])} points")
    print(f"   - {len(line['trajectory'][line['trajectory']['section_type'] == 'straight'])} straight points")
    print(f"   - {len(line['trajectory'][line['trajectory']['section_type'] == 'corner'])} corner points")

    return line


def demo_two_driver_comparison():
    """
    Scenario 2: Two-driver comparison

    Demonstrates:
    - Loading both drivers' telemetry
    - Comparing lines through each corner
    - Identifying differences
    - Calculating delta (entry, apex, exit)
    """
    print("\n" + "=" * 80)
    print("SCENARIO 2: Two-Driver Racing Line Comparison")
    print("=" * 80)

    # Generate telemetry for two drivers with different styles
    print("\n1. Generating telemetry for Driver A (aggressive) and Driver B (smooth)...")

    # Driver A: Aggressive (later braking, higher corner speeds)
    telem_a = generate_sample_telemetry(num_points=1000, num_corners=8, seed=42)
    telem_a['gps_speed'] += 5  # Slightly faster overall
    telem_a['brake_f'] *= 1.2  # Harder braking
    telem_a['gps_speed'] = telem_a['gps_speed'].clip(lower=0, upper=250)
    telem_a['brake_f'] = telem_a['brake_f'].clip(lower=0, upper=100)

    # Driver B: Smooth (earlier braking, smoother inputs)
    telem_b = generate_sample_telemetry(num_points=1000, num_corners=8, seed=123)
    telem_b['brake_f'] *= 0.9  # Lighter braking
    telem_b['aps'] *= 1.05  # More throttle smoothness

    print(f"   Driver A: {len(telem_a)} points, avg speed {telem_a['gps_speed'].mean():.1f} km/h")
    print(f"   Driver B: {len(telem_b)} points, avg speed {telem_b['gps_speed'].mean():.1f} km/h")

    # Reconstruct both lines
    print("\n2. Reconstructing racing lines for both drivers...")
    reconstructor = RacingLineReconstructor()

    comparison = reconstructor.compare_racing_lines(
        telem_a,
        telem_b,
        driver1_label="Driver A (Aggressive)",
        driver2_label="Driver B (Smooth)",
        speed_col='gps_speed',
        brake_col='brake_f',
        throttle_col='aps'
    )

    # Display comparison results
    differences = comparison['differences']
    summary = comparison['summary']

    print(f"\n3. Comparison Results:")

    if not summary.get('comparison_possible', True):
        print(f"   {summary.get('reason', 'Unable to compare')}")
        return comparison

    print(f"   Total corners compared: {summary.get('total_corners_compared', 0)}")

    print("\n   Corner-by-Corner Breakdown:")
    print("\n   Corner | Entry Δ | Apex Δ  | Exit Δ  | Speed Δ    | Faster Apex     | Later Braking")
    print("   " + "-" * 95)

    for diff in differences:
        print(f"     {diff['corner_number']:2d}   | "
              f"{diff['entry_delta']:+6.2f}  | "
              f"{diff['apex_delta']:+6.2f}  | "
              f"{diff['exit_delta']:+6.2f}  | "
              f"{diff['apex_speed_delta_kph']:+6.2f} kph | "
              f"{diff['faster_apex_speed']:16s} | "
              f"{diff['later_braking']:s}")

    # Summary statistics
    print(f"\n   Overall Summary:")
    print(f"   - Average apex speed delta: {summary['avg_apex_speed_delta_kph']:+.2f} km/h")
    print(f"   - Average brake point delta: {summary['avg_brake_point_delta']:+.2f}%")
    print(f"   - Average speed delta: {summary['avg_speed_delta_kph']:+.2f} km/h")

    print(f"\n   Performance Advantages:")
    print(f"   - Driver A faster in {summary['Driver A (Aggressive)_faster_corners']} corners")
    print(f"   - Driver B faster in {summary['Driver B (Smooth)_faster_corners']} corners")
    print(f"   - Driver A later braking in {summary['Driver A (Aggressive)_later_braking_corners']} corners")
    print(f"   - Driver B later braking in {summary['Driver B (Smooth)_later_braking_corners']} corners")

    print(f"\n   Dominant driver (apex speed): {summary['dominant_driver_apex']}")
    print(f"   Dominant driver (braking): {summary['dominant_driver_braking']}")

    # Generate visualizations if plotly is available
    try:
        from src.utils.visualization import PLOTLY_AVAILABLE

        if PLOTLY_AVAILABLE:
            print("\n4. Generating visualizations...")

            # Get track layout
            track_layout = get_track_layout('barber')

            # Create racing line comparison
            print("   - Racing line comparison map...")
            fig_lines = create_racing_line_comparison(
                comparison['driver1_line'],
                comparison['driver2_line'],
                track_layout,
                "Driver A (Aggressive)",
                "Driver B (Smooth)"
            )

            # Create corner analysis
            print("   - Corner-by-corner analysis charts...")
            fig_corners = create_corner_analysis(
                {},  # Not needed for differences
                "Driver A (Aggressive)",
                "Driver B (Smooth)",
                differences
            )

            # Create speed trace comparison
            print("   - Speed trace comparison...")
            fig_speed = create_speed_trace_comparison(
                comparison['driver1_line'],
                comparison['driver2_line'],
                "Driver A (Aggressive)",
                "Driver B (Smooth)"
            )

            # Save figures
            print("\n   Saving visualizations...")
            fig_lines.write_html('racing_line_comparison.html')
            fig_corners.write_html('corner_analysis.html')
            fig_speed.write_html('speed_trace_comparison.html')

            print("   ✓ Saved racing_line_comparison.html")
            print("   ✓ Saved corner_analysis.html")
            print("   ✓ Saved speed_trace_comparison.html")

        else:
            print("\n   Note: Plotly not available. Skipping visualizations.")

    except Exception as e:
        print(f"\n   Warning: Could not generate visualizations: {e}")

    return comparison


def demo_corner_specific_analysis():
    """
    Scenario 3: Corner-specific analysis

    Demonstrates:
    - Focus on Turn 5 (example corner)
    - Show detailed speed traces
    - Brake/throttle application points
    - Visualize line differences
    """
    print("\n" + "=" * 80)
    print("SCENARIO 3: Corner-Specific Analysis (Turn 5)")
    print("=" * 80)

    # Generate telemetry
    print("\n1. Generating telemetry for detailed corner analysis...")
    telem_a = generate_sample_telemetry(num_points=1000, num_corners=8, seed=42)
    telem_b = generate_sample_telemetry(num_points=1000, num_corners=8, seed=123)

    # Modify Driver B to be different in Turn 5
    turn5_region = (telem_b['distance_pct'] >= 45) & (telem_b['distance_pct'] <= 55)
    telem_b.loc[turn5_region, 'gps_speed'] += 3
    telem_b.loc[turn5_region, 'brake_f'] *= 0.8

    # Reconstruct lines
    print("\n2. Reconstructing racing lines...")
    reconstructor = RacingLineReconstructor()
    comparison = reconstructor.compare_racing_lines(
        telem_a,
        telem_b,
        driver1_label="Driver A",
        driver2_label="Driver B",
        speed_col='gps_speed',
        brake_col='brake_f',
        throttle_col='aps'
    )

    # Focus on corner 5
    corner_number = 5
    print(f"\n3. Analyzing Turn {corner_number} in detail...")

    differences = comparison['differences']
    if corner_number <= len(differences):
        corner_diff = differences[corner_number - 1]

        print(f"\n   Turn {corner_number} Comparison:")
        print(f"   - Entry point delta: {corner_diff['entry_delta']:+.2f}%")
        print(f"   - Apex point delta: {corner_diff['apex_delta']:+.2f}%")
        print(f"   - Exit point delta: {corner_diff['exit_delta']:+.2f}%")

        print(f"\n   Brake Points:")
        print(f"   - Driver A: {corner_diff['Driver A_brake_point']:.2f}%")
        print(f"   - Driver B: {corner_diff['Driver B_brake_point']:.2f}%")
        print(f"   - Delta: {corner_diff['brake_point_delta']:+.2f}% (negative = A brakes later)")

        print(f"\n   Apex Speeds:")
        print(f"   - Driver A: {corner_diff['Driver A_apex_speed']:.1f} km/h")
        print(f"   - Driver B: {corner_diff['Driver B_apex_speed']:.1f} km/h")
        print(f"   - Delta: {corner_diff['apex_speed_delta_kph']:+.1f} km/h")

        print(f"\n   Corner Geometry:")
        print(f"   - Driver A radius: {corner_diff['Driver A_radius']:.1f} m")
        print(f"   - Driver B radius: {corner_diff['Driver B_radius']:.1f} m")
        print(f"   - Delta: {corner_diff['radius_delta_m']:+.1f} m")

        print(f"\n   Analysis:")
        print(f"   - Faster apex speed: {corner_diff['faster_apex_speed']}")
        print(f"   - Later braking: {corner_diff['later_braking']}")

        # Visualize corner-specific speed trace
        try:
            from src.utils.visualization import PLOTLY_AVAILABLE

            if PLOTLY_AVAILABLE:
                print("\n4. Generating corner-specific visualization...")
                fig = create_speed_trace_comparison(
                    comparison['driver1_line'],
                    comparison['driver2_line'],
                    "Driver A",
                    "Driver B",
                    corner_number=corner_number
                )

                fig.write_html(f'corner_{corner_number}_analysis.html')
                print(f"   ✓ Saved corner_{corner_number}_analysis.html")

        except Exception as e:
            print(f"   Warning: Could not generate visualization: {e}")

    else:
        print(f"   Error: Corner {corner_number} not found in comparison")


def main():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 80)
    print("Racing Line Reconstruction Feature Demo")
    print("RaceIQ Pro - Enhancement #6")
    print("=" * 80)

    print("\nThis demo showcases the Racing Line Reconstruction feature with:")
    print("  1. Single driver line reconstruction")
    print("  2. Two-driver comparison")
    print("  3. Corner-specific analysis")

    # Run scenarios
    print("\n" + "=" * 80)
    print("Running Demonstrations...")
    print("=" * 80)

    # Scenario 1: Single driver
    line1 = demo_single_driver_reconstruction()

    # Scenario 2: Two-driver comparison
    comparison = demo_two_driver_comparison()

    # Scenario 3: Corner-specific
    demo_corner_specific_analysis()

    # Summary
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)

    print("\nKey Capabilities Demonstrated:")
    print("  ✓ Corner identification from speed/brake patterns")
    print("  ✓ Physics-based corner radius estimation")
    print("  ✓ Brake point and apex detection")
    print("  ✓ Racing line trajectory reconstruction")
    print("  ✓ Two-driver line comparison")
    print("  ✓ Corner-by-corner performance analysis")
    print("  ✓ Interactive visualizations (if Plotly available)")

    print("\nPhysics Formulas Used:")
    print("  - Corner radius: r = v² / (g * lateral_g)")
    print("  - Lateral G: Assumed ~1.8g for racing")
    print("  - Track width: Assumed ~12m typical")

    print("\nOutput Files Generated (if Plotly available):")
    print("  - racing_line_comparison.html")
    print("  - corner_analysis.html")
    print("  - speed_trace_comparison.html")
    print("  - corner_5_analysis.html")

    print("\nNext Steps:")
    print("  - Run with real telemetry data for actual racing analysis")
    print("  - Use dashboard integration for interactive exploration")
    print("  - Export results for driver coaching sessions")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
