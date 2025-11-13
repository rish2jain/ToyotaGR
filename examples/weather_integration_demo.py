"""
Weather Integration Demo for RaceIQ Pro

This script demonstrates how weather data is integrated into the RaceIQ Pro platform
to adjust tire degradation predictions, lap time estimates, and strategic recommendations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pipeline.data_loader import DataLoader
from src.integration.weather_adjuster import WeatherAdjuster, WeatherConditions


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demonstrate_weather_loading():
    """Demonstrate loading weather data from files."""
    print_section_header("1. LOADING WEATHER DATA")

    # Initialize data loader
    loader = DataLoader()
    print(f"Data loader initialized with base path: {loader.base_path}")

    # Load weather data
    weather_df = loader.load_weather_data()

    if weather_df is not None and not weather_df.empty:
        print(f"\n✓ Successfully loaded weather data: {len(weather_df)} records")
        print(f"\nColumns: {list(weather_df.columns)}")

        # Display sample data
        print("\nSample weather data (first 5 rows):")
        print(weather_df.head())

        # Display statistics
        print("\nWeather Statistics:")
        if 'AIR_TEMP' in weather_df.columns:
            print(f"  Air Temperature: {weather_df['AIR_TEMP'].min():.1f}°C - {weather_df['AIR_TEMP'].max():.1f}°C (avg: {weather_df['AIR_TEMP'].mean():.1f}°C)")
        if 'HUMIDITY' in weather_df.columns:
            print(f"  Humidity: {weather_df['HUMIDITY'].min():.0f}% - {weather_df['HUMIDITY'].max():.0f}% (avg: {weather_df['HUMIDITY'].mean():.0f}%)")
        if 'WIND_SPEED' in weather_df.columns:
            print(f"  Wind Speed: {weather_df['WIND_SPEED'].min():.1f} - {weather_df['WIND_SPEED'].max():.1f} km/h (avg: {weather_df['WIND_SPEED'].mean():.1f} km/h)")

        return weather_df
    else:
        print("\n✗ No weather data available")
        print("\nGenerating synthetic weather data for demonstration...")

        # Create synthetic weather data
        weather_df = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-06 18:00:00', periods=50, freq='1min'),
            'AIR_TEMP': np.random.uniform(28, 32, 50),
            'TRACK_TEMP': np.random.uniform(38, 45, 50),
            'HUMIDITY': np.random.uniform(50, 65, 50),
            'WIND_SPEED': np.random.uniform(5, 15, 50),
            'RAIN': np.zeros(50)
        })

        print(f"\n✓ Generated synthetic weather data: {len(weather_df)} records")
        return weather_df


def demonstrate_weather_conditions(weather_df):
    """Demonstrate extracting weather conditions."""
    print_section_header("2. WEATHER CONDITIONS ANALYSIS")

    adjuster = WeatherAdjuster()

    # Get current conditions
    conditions = adjuster.get_current_conditions(weather_df)

    if conditions:
        print("Current Weather Conditions:")
        print(f"  Air Temperature:   {conditions.temperature:.1f}°C")
        print(f"  Track Temperature: {conditions.track_temp:.1f}°C")
        print(f"  Humidity:          {conditions.humidity:.0f}%")
        print(f"  Wind Speed:        {conditions.wind_speed:.1f} km/h")
        print(f"  Precipitation:     {'DRY' if conditions.precipitation == 0 else 'WET'}")

        # Assess conditions
        print("\nCondition Assessment:")
        if conditions.track_temp > 40:
            print("  ⚠ HOT track conditions - increased tire degradation expected")
        elif conditions.track_temp < 25:
            print("  ⚠ COLD track conditions - reduced grip expected")
        else:
            print("  ✓ IDEAL track temperature")

        if conditions.wind_speed > 25:
            print("  ⚠ HIGH winds - stability concerns")
        elif conditions.wind_speed > 15:
            print("  ⚠ MODERATE winds - minor impact expected")
        else:
            print("  ✓ LIGHT winds")

        if conditions.humidity > 70:
            print("  ⚠ HIGH humidity")
        else:
            print("  ✓ ACCEPTABLE humidity")

        return conditions
    else:
        print("✗ Could not extract weather conditions")
        return None


def demonstrate_tire_degradation_adjustment(conditions):
    """Demonstrate tire degradation adjustment based on weather."""
    print_section_header("3. TIRE DEGRADATION ADJUSTMENT")

    if conditions is None:
        print("✗ No weather conditions available")
        return

    adjuster = WeatherAdjuster()

    # Test with various baseline degradation rates
    baseline_rates = [0.03, 0.05, 0.08]  # seconds per lap

    print("Tire Degradation Impact Analysis:\n")

    for base_rate in baseline_rates:
        adjusted_rate, explanation = adjuster.adjust_tire_degradation(base_rate, conditions)

        change_pct = ((adjusted_rate / base_rate) - 1) * 100

        print(f"Baseline: {base_rate:.4f}s/lap")
        print(f"  → Adjusted: {adjusted_rate:.4f}s/lap ({change_pct:+.1f}%)")
        print(f"  → Explanation: {explanation}")

        # Calculate impact over a stint
        stint_laps = 20
        baseline_total = base_rate * stint_laps
        adjusted_total = adjusted_rate * stint_laps
        time_difference = adjusted_total - baseline_total

        print(f"  → Impact over {stint_laps} laps: {time_difference:+.2f}s")
        print()


def demonstrate_lap_time_adjustment(conditions):
    """Demonstrate lap time adjustment based on weather."""
    print_section_header("4. LAP TIME ADJUSTMENT")

    if conditions is None:
        print("✗ No weather conditions available")
        return

    adjuster = WeatherAdjuster()

    # Test with various baseline lap times
    baseline_times = [95.0, 100.0, 105.0]  # seconds

    print("Lap Time Impact Analysis:\n")

    for base_time in baseline_times:
        adjusted_time, explanation = adjuster.adjust_lap_times(base_time, conditions)

        change_pct = ((adjusted_time / base_time) - 1) * 100
        time_difference = adjusted_time - base_time

        print(f"Baseline: {base_time:.2f}s")
        print(f"  → Adjusted: {adjusted_time:.2f}s ({change_pct:+.2f}% / {time_difference:+.3f}s)")
        print(f"  → Explanation: {explanation}")

        # Calculate impact over remaining race
        laps_remaining = 15
        total_difference = time_difference * laps_remaining

        print(f"  → Impact over {laps_remaining} laps: {total_difference:+.2f}s")
        print()


def demonstrate_weather_recommendations(weather_df):
    """Demonstrate generating weather-based strategic recommendations."""
    print_section_header("5. WEATHER-BASED STRATEGIC RECOMMENDATIONS")

    adjuster = WeatherAdjuster()

    # Generate recommendations
    race_data = {
        'base_tire_degradation': 0.05,  # 0.05 seconds/lap baseline
        'current_lap': 10,
        'total_laps': 30
    }

    recommendations = adjuster.generate_weather_recommendations(weather_df, race_data)

    print("Strategic Recommendations:\n")

    print(f"Tire Adjustment:")
    print(f"  {recommendations['tire_adjustment']}")
    print()

    print(f"Lap Time Modifier:")
    print(f"  {recommendations['lap_time_modifier']:.3f}x baseline")
    print(f"  ({(recommendations['lap_time_modifier']-1)*100:+.1f}% change)")
    print()

    if recommendations['risky_sections']:
        print(f"Risky Sections:")
        print(f"  {', '.join(map(str, recommendations['risky_sections']))}")
        print()

    print("Strategic Notes:")
    for i, note in enumerate(recommendations['strategic_notes'], 1):
        print(f"  {i}. {note}")
    print()

    # Display impact summary
    if recommendations['impact']:
        impact = recommendations['impact']
        print(f"Overall Weather Impact:")
        print(f"  Risk Level: {impact.risk_level}")
        print(f"  Description: {impact.impact_description}")


def demonstrate_before_after_comparison(weather_df):
    """Demonstrate before/after comparison of strategy with weather adjustments."""
    print_section_header("6. BEFORE/AFTER STRATEGY COMPARISON")

    adjuster = WeatherAdjuster()
    conditions = adjuster.get_current_conditions(weather_df)

    if conditions is None:
        print("✗ No weather conditions available")
        return

    # Simulate race parameters
    baseline_lap_time = 100.0  # seconds
    baseline_degradation = 0.05  # seconds/lap
    total_laps = 30
    pit_loss = 25.0  # seconds

    print("Race Scenario:")
    print(f"  Total Laps: {total_laps}")
    print(f"  Baseline Lap Time: {baseline_lap_time:.2f}s")
    print(f"  Baseline Tire Degradation: {baseline_degradation:.4f}s/lap")
    print(f"  Pit Stop Loss: {pit_loss:.0f}s")
    print()

    # Calculate without weather adjustment
    print("WITHOUT Weather Adjustment:")
    total_time_no_weather = 0
    for lap in range(1, total_laps + 1):
        lap_time = baseline_lap_time + (baseline_degradation * lap)
        if lap == 15:  # Pit on lap 15
            lap_time += pit_loss
        total_time_no_weather += lap_time

    print(f"  Optimal Pit Lap: 15")
    print(f"  Total Race Time: {total_time_no_weather:.2f}s ({total_time_no_weather/60:.2f} min)")
    print()

    # Calculate with weather adjustment
    print("WITH Weather Adjustment:")
    adjusted_deg, _ = adjuster.adjust_tire_degradation(baseline_degradation, conditions)
    adjusted_lap, _ = adjuster.adjust_lap_times(baseline_lap_time, conditions)

    total_time_with_weather = 0
    # Adjust pit lap based on degradation change
    optimal_pit_lap = 15
    if adjusted_deg > baseline_degradation * 1.1:
        optimal_pit_lap = 13  # Earlier pit due to increased degradation
    elif adjusted_deg < baseline_degradation * 0.95:
        optimal_pit_lap = 17  # Later pit due to reduced degradation

    for lap in range(1, total_laps + 1):
        lap_time = adjusted_lap + (adjusted_deg * lap)
        if lap == optimal_pit_lap:
            lap_time += pit_loss
        total_time_with_weather += lap_time

    print(f"  Adjusted Tire Degradation: {adjusted_deg:.4f}s/lap ({((adjusted_deg/baseline_degradation)-1)*100:+.0f}%)")
    print(f"  Adjusted Lap Time: {adjusted_lap:.2f}s ({((adjusted_lap/baseline_lap_time)-1)*100:+.1f}%)")
    print(f"  Optimal Pit Lap: {optimal_pit_lap} (adjusted from 15)")
    print(f"  Total Race Time: {total_time_with_weather:.2f}s ({total_time_with_weather/60:.2f} min)")
    print()

    # Compare
    time_difference = total_time_with_weather - total_time_no_weather
    print(f"Impact of Weather Adjustment:")
    print(f"  Time Difference: {time_difference:+.2f}s")
    print(f"  Strategy Change: Pit lap moved from 15 to {optimal_pit_lap}")

    if time_difference > 0:
        print(f"  ⚠ Weather conditions slow the race by {time_difference:.2f}s")
    else:
        print(f"  ✓ Weather conditions improve pace by {abs(time_difference):.2f}s")


def create_visualization(weather_df):
    """Create visualizations of weather data and impact."""
    print_section_header("7. WEATHER VISUALIZATION")

    try:
        adjuster = WeatherAdjuster()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Weather Data Analysis - Barber Motorsports Park', fontsize=16, fontweight='bold')

        # Temperature plot
        if 'AIR_TEMP' in weather_df.columns and 'timestamp' in weather_df.columns:
            ax = axes[0, 0]
            ax.plot(weather_df['timestamp'], weather_df['AIR_TEMP'], 'o-', color='red', linewidth=2, label='Air Temp')

            if 'TRACK_TEMP_ESTIMATED' in weather_df.columns:
                ax.plot(weather_df['timestamp'], weather_df['TRACK_TEMP_ESTIMATED'], 's-',
                       color='orange', linewidth=2, label='Track Temp (Est.)')

            ax.set_xlabel('Time')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Temperature Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Humidity plot
        if 'HUMIDITY' in weather_df.columns and 'timestamp' in weather_df.columns:
            ax = axes[0, 1]
            ax.plot(weather_df['timestamp'], weather_df['HUMIDITY'], 'o-', color='blue', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Humidity (%)')
            ax.set_title('Humidity Over Time')
            ax.grid(True, alpha=0.3)

        # Wind speed plot
        if 'WIND_SPEED' in weather_df.columns and 'timestamp' in weather_df.columns:
            ax = axes[1, 0]
            ax.plot(weather_df['timestamp'], weather_df['WIND_SPEED'], 'o-', color='green', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Wind Speed (km/h)')
            ax.set_title('Wind Speed Over Time')
            ax.grid(True, alpha=0.3)

        # Tire degradation impact plot
        ax = axes[1, 1]
        conditions_over_time = []
        degradation_multipliers = []

        for idx in range(len(weather_df)):
            temp_conditions = adjuster.get_current_conditions(weather_df.iloc[[idx]])
            if temp_conditions:
                base_deg = 0.05
                adj_deg, _ = adjuster.adjust_tire_degradation(base_deg, temp_conditions)
                multiplier = adj_deg / base_deg
                degradation_multipliers.append(multiplier)
            else:
                degradation_multipliers.append(1.0)

        if 'timestamp' in weather_df.columns:
            ax.plot(weather_df['timestamp'], degradation_multipliers, 'o-', color='purple', linewidth=2)
            ax.axhline(y=1.0, color='gray', linestyle='--', label='Baseline')
            ax.set_xlabel('Time')
            ax.set_ylabel('Degradation Multiplier')
            ax.set_title('Tire Degradation Impact Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = project_root / 'examples' / 'weather_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")

        # Try to display if in interactive environment
        try:
            plt.show()
        except:
            pass

    except Exception as e:
        print(f"\n✗ Could not create visualization: {e}")


def main():
    """Main demonstration script."""
    print("\n" + "=" * 80)
    print("  WEATHER INTEGRATION DEMO - RaceIQ Pro")
    print("  Toyota GR Cup Hackathon 2025")
    print("=" * 80)

    # 1. Load weather data
    weather_df = demonstrate_weather_loading()

    if weather_df is None or weather_df.empty:
        print("\n✗ Cannot proceed without weather data")
        return

    # 2. Analyze weather conditions
    conditions = demonstrate_weather_conditions(weather_df)

    # 3. Demonstrate tire degradation adjustment
    demonstrate_tire_degradation_adjustment(conditions)

    # 4. Demonstrate lap time adjustment
    demonstrate_lap_time_adjustment(conditions)

    # 5. Generate strategic recommendations
    demonstrate_weather_recommendations(weather_df)

    # 6. Before/after comparison
    demonstrate_before_after_comparison(weather_df)

    # 7. Create visualizations
    create_visualization(weather_df)

    # Summary
    print_section_header("SUMMARY")
    print("Weather integration demonstration completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Weather data loading and parsing")
    print("  ✓ Weather condition extraction and analysis")
    print("  ✓ Tire degradation adjustment based on temperature")
    print("  ✓ Lap time adjustment based on weather conditions")
    print("  ✓ Strategic recommendations generation")
    print("  ✓ Before/after strategy comparison")
    print("  ✓ Weather data visualization")
    print()
    print("The weather adjuster is now fully integrated into RaceIQ Pro!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
