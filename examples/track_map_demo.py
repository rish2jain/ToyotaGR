"""
Track Map Visualization Demo
Demonstrates the track map with performance heatmap overlay

This example shows how to:
1. Load section performance data
2. Create interactive track maps
3. Compare driver performance on track maps
4. Visualize performance gaps by section
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.visualization import (
    create_track_map_with_performance,
    create_driver_comparison_map
)
from src.utils.track_layouts import get_track_layout


def generate_sample_section_data(driver_number: int, num_laps: int = 10) -> pd.DataFrame:
    """
    Generate sample section performance data for demonstration

    Args:
        driver_number: Driver number/ID
        num_laps: Number of laps to generate

    Returns:
        DataFrame with section performance data
    """
    data = []

    # Simulate 3 sections with realistic times
    # Section 1: ~25-27 seconds
    # Section 2: ~22-24 seconds
    # Section 3: ~28-30 seconds

    base_times = {1: 26.0, 2: 23.0, 3: 29.0}
    variations = {1: 0.5, 2: 0.4, 3: 0.6}

    # Add driver-specific bias
    driver_bias = (driver_number % 3) * 0.1

    for lap in range(1, num_laps + 1):
        for section in [1, 2, 3]:
            # Add random variation and occasional anomaly
            is_anomaly = np.random.random() < 0.1  # 10% chance of anomaly

            if is_anomaly:
                time = base_times[section] + np.random.uniform(1.5, 3.0)
            else:
                time = base_times[section] + np.random.normal(0, variations[section])

            time += driver_bias

            data.append({
                'Driver': driver_number,
                'Lap': lap,
                'Section': section,
                'Time': time
            })

    df = pd.DataFrame(data)

    # Calculate gap to optimal for each section
    optimal_times = df.groupby('Section')['Time'].min()
    df['GapToOptimal'] = df.apply(
        lambda row: row['Time'] - optimal_times[row['Section']],
        axis=1
    )

    return df


def demo_single_driver_map():
    """Demonstrate single driver track map with performance overlay"""
    print("\n" + "="*70)
    print("DEMO 1: Single Driver Track Map")
    print("="*70)

    # Generate sample data
    print("\n1. Generating sample section data for Driver #42...")
    driver_data = generate_sample_section_data(driver_number=42, num_laps=15)

    print(f"   Generated {len(driver_data)} section records")
    print(f"   Sections: {sorted(driver_data['Section'].unique())}")
    print(f"   Laps: {driver_data['Lap'].min()} to {driver_data['Lap'].max()}")

    # Show sample data
    print("\n2. Sample data (first 5 rows):")
    print(driver_data.head().to_string(index=False))

    # Calculate statistics
    print("\n3. Performance statistics by section:")
    stats = driver_data.groupby('Section').agg({
        'Time': ['mean', 'min', 'max', 'std'],
        'GapToOptimal': 'mean'
    }).round(3)
    print(stats)

    # Create track map
    print("\n4. Creating interactive track map with performance overlay...")
    fig = create_track_map_with_performance(
        driver_data,
        track_name='barber',
        section_col='Section',
        time_col='Time',
        gap_col='GapToOptimal',
        driver_label='Car #42'
    )

    if fig:
        # Save to HTML file
        output_path = os.path.join(os.path.dirname(__file__), 'track_map_driver42.html')
        fig.write_html(output_path)
        print(f"   Track map saved to: {output_path}")
        print("   Open this file in a web browser to view the interactive map!")
    else:
        print("   ERROR: Failed to create track map")

    print("\n" + "-"*70)


def demo_driver_comparison():
    """Demonstrate driver comparison on track map"""
    print("\n" + "="*70)
    print("DEMO 2: Driver Comparison Track Map")
    print("="*70)

    # Generate data for two drivers
    print("\n1. Generating sample data for two drivers...")
    driver1_data = generate_sample_section_data(driver_number=42, num_laps=12)
    driver2_data = generate_sample_section_data(driver_number=17, num_laps=12)

    print(f"   Driver #42: {len(driver1_data)} section records")
    print(f"   Driver #17: {len(driver2_data)} section records")

    # Show performance comparison
    print("\n2. Average section times comparison:")
    comparison = pd.DataFrame({
        'Section': sorted(driver1_data['Section'].unique()),
        'Driver #42': driver1_data.groupby('Section')['Time'].mean().values,
        'Driver #17': driver2_data.groupby('Section')['Time'].mean().values
    })
    comparison['Difference'] = comparison['Driver #42'] - comparison['Driver #17']
    comparison['Faster Driver'] = comparison['Difference'].apply(
        lambda x: 'Driver #17' if x > 0 else ('Driver #42' if x < 0 else 'Equal')
    )
    print(comparison.to_string(index=False))

    # Create comparison map
    print("\n3. Creating driver comparison track map...")
    fig = create_driver_comparison_map(
        driver1_data,
        driver2_data,
        track_name='barber',
        driver1_label='Car #42',
        driver2_label='Car #17',
        section_col='Section',
        time_col='Time'
    )

    if fig:
        # Save to HTML file
        output_path = os.path.join(os.path.dirname(__file__), 'track_map_comparison.html')
        fig.write_html(output_path)
        print(f"   Comparison map saved to: {output_path}")
        print("   Open this file in a web browser to view the interactive comparison!")
    else:
        print("   ERROR: Failed to create comparison map")

    print("\n" + "-"*70)


def demo_track_layouts():
    """Demonstrate available track layouts"""
    print("\n" + "="*70)
    print("DEMO 3: Available Track Layouts")
    print("="*70)

    track_names = ['barber', 'cota', 'sonoma', 'generic']

    print("\nAvailable tracks:")
    for track in track_names:
        print(f"\n{track.upper()}:")
        layout = get_track_layout(track)

        print(f"   Name: {layout['track_info']['name']}")
        print(f"   Location: {layout['track_info']['location']}")
        print(f"   Length: {layout['track_info']['length']}")
        print(f"   Turns: {layout['track_info']['turns']}")
        print(f"   Direction: {layout['track_info']['direction']}")
        print(f"   Sections defined: {layout['total_sections']}")

        # Show first section as example
        if layout['sections']:
            first_section = layout['sections'][0]
            print(f"   Example section: {first_section['name']}")
            print(f"      Type: {first_section['type']}")
            print(f"      Description: {first_section['description']}")

    print("\n" + "-"*70)


def demo_performance_analysis():
    """Demonstrate performance analysis using track map data"""
    print("\n" + "="*70)
    print("DEMO 4: Performance Analysis from Track Map Data")
    print("="*70)

    # Generate sample data
    print("\n1. Generating sample data for performance analysis...")
    driver_data = generate_sample_section_data(driver_number=42, num_laps=20)

    # Identify problem sections
    print("\n2. Identifying sections needing improvement:")
    section_gaps = driver_data.groupby('Section').agg({
        'Time': ['mean', 'min'],
        'GapToOptimal': ['mean', 'max']
    }).round(3)

    section_gaps.columns = ['Avg_Time', 'Best_Time', 'Avg_Gap', 'Max_Gap']
    section_gaps['Improvement_Potential'] = section_gaps['Avg_Gap']
    section_gaps = section_gaps.sort_values('Improvement_Potential', ascending=False)

    print("\nSections ranked by improvement potential:")
    print(section_gaps.to_string())

    # Generate recommendations
    print("\n3. Performance recommendations:")
    for idx, (section, row) in enumerate(section_gaps.iterrows(), 1):
        if row['Avg_Gap'] > 0.3:
            priority = "HIGH"
        elif row['Avg_Gap'] > 0.15:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        print(f"\n   Section {section} - Priority: {priority}")
        print(f"   - Average gap: {row['Avg_Gap']:.3f}s")
        print(f"   - Maximum gap: {row['Max_Gap']:.3f}s")
        print(f"   - Consistency: {'Good' if row['Max_Gap'] < 0.5 else 'Needs improvement'}")

        if priority == "HIGH":
            print(f"   → FOCUS HERE: Review braking points and racing line")
        elif priority == "MEDIUM":
            print(f"   → Fine-tune apex speed and throttle application")
        else:
            print(f"   → Maintain current approach")

    print("\n" + "-"*70)


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "TRACK MAP VISUALIZATION DEMO" + " "*25 + "║")
    print("║" + " "*20 + "RaceIQ Pro Platform" + " "*29 + "║")
    print("╚" + "="*68 + "╝")

    print("\nThis demo showcases the track map visualization capabilities:")
    print("- Interactive track maps with performance heatmap overlays")
    print("- Driver-to-driver comparison visualizations")
    print("- Multiple track layouts (Barber, COTA, Sonoma)")
    print("- Performance analysis and recommendations")

    # Run demonstrations
    demo_track_layouts()
    demo_single_driver_map()
    demo_driver_comparison()
    demo_performance_analysis()

    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - track_map_driver42.html: Single driver performance map")
    print("  - track_map_comparison.html: Driver comparison map")
    print("\nOpen these HTML files in your web browser to explore the interactive maps!")
    print("\nKey features:")
    print("  ✓ Color-coded performance overlay (Green=Fast, Red=Slow)")
    print("  ✓ Interactive hover tooltips with detailed metrics")
    print("  ✓ Pan and zoom capabilities")
    print("  ✓ Section-by-section performance breakdown")
    print("  ✓ Driver comparison visualization")
    print("\nIntegration:")
    print("  - These functions are integrated into the Tactical Analysis dashboard")
    print("  - Access via dashboard/pages/tactical.py in the main application")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
