"""
Tactical Analysis Module Demo

This script demonstrates the usage of the Tactical Analysis Module
on real Toyota GR Cup race data.

Usage:
    python examples/tactical_analysis_demo.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tactical import OptimalGhostAnalyzer, AnomalyDetector, SectionAnalyzer


def load_sample_data():
    """Load sample race data from Barber Motorsports Park."""
    data_path = os.path.join(
        os.path.dirname(__file__),
        '../Data/barber/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV'
    )

    if not os.path.exists(data_path):
        print(f"Error: Sample data not found at {data_path}")
        return None

    # Load data
    df = pd.read_csv(data_path, delimiter=';')
    print(f"Loaded {len(df)} laps from Barber Race 1")
    print(f"Drivers: {df['DRIVER_NUMBER'].nunique()}")
    print(f"Columns: {', '.join(df.columns[:10])}...")
    return df


def demo_optimal_ghost(data):
    """Demonstrate optimal ghost analysis."""
    print("\n" + "=" * 80)
    print("OPTIMAL GHOST ANALYSIS")
    print("=" * 80)

    analyzer = OptimalGhostAnalyzer()

    # Create optimal ghost
    optimal_ghost = analyzer.create_optimal_ghost(data, percentile=95)

    print("\nOptimal Ghost Lap (Top 5% times per section):")
    total_time = 0
    for section, info in optimal_ghost.items():
        print(f"  {section}: {info['time']:.3f}s by driver {info['best_driver']}")
        total_time += info['time']

    print(f"\nTotal optimal lap time: {total_time:.3f}s")

    # Analyze a specific driver vs ghost
    driver_num = data['DRIVER_NUMBER'].iloc[0]
    driver_data = data[data['DRIVER_NUMBER'] == driver_num]

    print(f"\nAnalyzing driver {driver_num} vs optimal ghost...")
    analysis = analyzer.analyze_driver_vs_ghost(driver_data, optimal_ghost)

    print(f"\nDriver {analysis['driver_number']} Performance:")
    print(f"  Total gap to optimal: {analysis['total_gap']:.3f}s")

    print("\nTop 3 Improvement Opportunities:")
    for i, opp in enumerate(analysis['top_3_improvements'], 1):
        print(f"  {i}. {opp['section']}: "
              f"+{opp['gap_seconds']:.3f}s ({opp['gap_percent']:.1f}%) - {opp['priority']}")

    # Compare all drivers
    print("\nComparing all drivers to optimal ghost...")
    comparison = analyzer.compare_multiple_drivers(data, optimal_ghost)
    print("\nTop 5 drivers (closest to optimal):")
    print(comparison.head()[['driver_number', 'total_gap']].to_string(index=False))


def demo_anomaly_detection(data):
    """Demonstrate anomaly detection."""
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION")
    print("=" * 80)

    detector = AnomalyDetector()

    # Statistical anomaly detection
    print("\nRunning statistical anomaly detection (rolling z-scores)...")
    anomalies = detector.detect_statistical_anomalies(data, window=5, threshold=2.5)

    print(f"\nFound {(anomalies['anomaly_count'] > 0).sum()} laps with anomalies")

    # Classify anomaly types
    anomalies['anomaly_type'] = detector.classify_anomaly_type(anomalies)

    # Get summary
    summary = detector.get_anomaly_summary(anomalies)
    print(f"\nAnomaly Summary:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Anomalous samples: {summary['anomalous_samples']}")
    print(f"  Anomaly rate: {summary['anomaly_rate']*100:.1f}%")

    if 'anomaly_types' in summary:
        print("\n  Anomalies by type:")
        for atype, count in summary['anomaly_types'].items():
            print(f"    {atype}: {count}")

    if 'anomalies_by_metric' in summary:
        print("\n  Top 5 metrics with anomalies:")
        for i, (metric, count) in enumerate(list(summary['anomalies_by_metric'].items())[:5], 1):
            print(f"    {i}. {metric}: {count}")

    # Show high priority anomalies
    high_priority = detector.filter_high_priority_anomalies(anomalies, min_anomaly_count=2)
    if len(high_priority) > 0:
        print(f"\nHigh priority anomalies (2+ metrics flagged): {len(high_priority)}")
        sample = high_priority.head(3)[['DRIVER_NUMBER', 'LAP_NUMBER', 'anomaly_count', 'anomaly_type']]
        print(sample.to_string(index=False))


def demo_section_analysis(data):
    """Demonstrate section analysis."""
    print("\n" + "=" * 80)
    print("SECTION ANALYSIS")
    print("=" * 80)

    analyzer = SectionAnalyzer()

    # Calculate overall section statistics
    print("\nCalculating section statistics...")
    stats = analyzer.calculate_section_statistics(data)

    print("\nSection Statistics (all drivers):")
    for section, metrics in stats.items():
        if section.startswith('S'):  # Only show main sections
            print(f"\n  {section}:")
            print(f"    Best: {metrics['min']:.3f}s")
            print(f"    Median: {metrics['median']:.3f}s")
            print(f"    Mean: {metrics['mean']:.3f}s ± {metrics['std']:.3f}s")
            print(f"    Samples: {metrics['count']}")

    # Analyze specific driver
    driver_num = data['DRIVER_NUMBER'].iloc[0]
    driver_data = data[data['DRIVER_NUMBER'] == driver_num]

    # Identify strengths
    print(f"\nIdentifying strengths for driver {driver_num}...")
    strengths = analyzer.identify_driver_strengths(driver_data, data, top_percentile=20)

    if strengths:
        print(f"\nDriver {driver_num} Strengths (Top 20%):")
        for strength in strengths:
            print(f"  {strength['section']}: "
                  f"Top {strength['percentile_rank']:.1f}% "
                  f"({strength['advantage_seconds']:+.3f}s vs field median)")
    else:
        print(f"\nDriver {driver_num} has no sections in top 20%")

    # Analyze consistency
    print(f"\nAnalyzing consistency for driver {driver_num}...")
    consistency = analyzer.analyze_section_consistency(driver_data)

    print(f"\nDriver {driver_num} Consistency:")
    for section, metrics in consistency.items():
        if section.startswith('S'):  # Only show main sections
            print(f"  {section}: {metrics['consistency_score']:.1f}/100 "
                  f"(CV: {metrics['cv']:.3f}, Range: {metrics['range']:.3f}s)")

    # Improvement potential
    print(f"\nCalculating improvement potential for driver {driver_num}...")
    potential = analyzer.get_section_improvement_potential(driver_data)

    if potential:
        print(f"\nDriver {driver_num} Self-Improvement Potential:")
        sorted_potential = sorted(potential.items(), key=lambda x: x[1], reverse=True)
        for section, seconds in sorted_potential[:5]:
            if section.startswith('S'):
                print(f"  {section}: {seconds:.3f}s (median vs personal best)")


def main():
    """Run all demos."""
    print("=" * 80)
    print("RaceIQ Pro - Tactical Analysis Module Demo")
    print("=" * 80)

    # Load data
    data = load_sample_data()
    if data is None:
        return

    # Run demos
    demo_optimal_ghost(data)
    demo_anomaly_detection(data)
    demo_section_analysis(data)

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
