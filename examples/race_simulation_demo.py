"""
Multi-Driver Race Simulation Demonstration

This script demonstrates the capabilities of the MultiDriverRaceSimulator
with various scenarios including undercuts, overcurts, and team strategies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategic.race_simulation import MultiDriverRaceSimulator
import pandas as pd
import numpy as np


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_results(results):
    """Print final race results in a formatted table."""
    print("\n📊 FINAL RESULTS:")
    print("-" * 80)
    print(f"{'Pos':<5} {'Driver':<20} {'Time':<12} {'Gap':<10} {'Avg Lap':<10} {'Pit Stops'}")
    print("-" * 80)

    for result in results:
        pos = result['position']
        driver = result['driver_name']
        total_time = result['total_time']
        gap = f"+{result['gap_to_leader']:.2f}s" if result['gap_to_leader'] > 0 else "Leader"
        avg_lap = f"{result['avg_lap_time']:.2f}s"
        pits = result['pit_stops']

        print(f"{pos:<5} {driver:<20} {total_time:>10.2f}s {gap:<10} {avg_lap:<10} {pits}")

    print("-" * 80)


def print_position_changes(position_changes):
    """Print position changes during the race."""
    if not position_changes:
        print("\n⚠️  No position changes during the race")
        return

    print("\n🔄 POSITION CHANGES:")
    print("-" * 60)
    print(f"{'Lap':<6} {'Driver':<15} {'Change'}")
    print("-" * 60)

    for change in position_changes[:10]:  # Limit to first 10
        lap = change['lap']
        driver = change['driver_id']
        old_pos = change['old_position']
        new_pos = change['new_position']

        arrow = "↑" if new_pos < old_pos else "↓"
        change_text = f"P{old_pos} {arrow} P{new_pos}"

        print(f"{lap:<6} {driver:<15} {change_text}")

    if len(position_changes) > 10:
        print(f"\n... and {len(position_changes) - 10} more position changes")

    print("-" * 60)


def scenario_1_simple_2_driver():
    """Scenario 1: Simple 2-driver race with different strategies."""
    print_header("SCENARIO 1: Simple 2-Driver Battle")

    print("Setup:")
    print("  • Driver A: Early pit strategy (lap 10)")
    print("  • Driver B: Late pit strategy (lap 14)")
    print("  • Both have similar pace and tire degradation")
    print("  • 25-lap race\n")

    # Create simulator
    simulator = MultiDriverRaceSimulator(race_length=25, pit_loss_time=25.0)

    # Define drivers
    drivers_data = {
        'A': {
            'name': 'Driver A (Early Pit)',
            'base_lap_time': 95.0,
            'tire_deg_rate': 0.05,
            'consistency': 0.08
        },
        'B': {
            'name': 'Driver B (Late Pit)',
            'base_lap_time': 95.0,
            'tire_deg_rate': 0.05,
            'consistency': 0.08
        }
    }

    # Define strategies
    strategies = {
        'A': {'pit_laps': [10]},
        'B': {'pit_laps': [14]}
    }

    # Run simulation
    print("🏁 Running simulation...")
    result = simulator.simulate_race(drivers_data, strategies)

    # Print results
    print_results(result['final_results'])
    print_position_changes(result['position_changes'])

    # Analysis
    winner = result['final_results'][0]
    print(f"\n✅ WINNER: {winner['driver_name']}")
    print(f"   Winning margin: {result['final_results'][1]['gap_to_leader']:.2f} seconds")

    # Strategy effectiveness
    print("\n📈 STRATEGY EFFECTIVENESS:")
    for driver_id, effectiveness in result['strategy_effectiveness'].items():
        pit_lap = effectiveness['pit_laps'][0]
        final_pos = effectiveness['final_position']
        print(f"   {driver_id}: Pitted lap {pit_lap}, finished P{final_pos}")

    return result


def scenario_2_undercut_demo():
    """Scenario 2: Undercut demonstration."""
    print_header("SCENARIO 2: Undercut Strategy Demonstration")

    print("Setup:")
    print("  • Driver A: Attempts undercut, pits lap 10")
    print("  • Driver B: Defends position, pits lap 12")
    print("  • Goal: Can A pass B on fresh tires?\n")

    # Create simulator
    simulator = MultiDriverRaceSimulator(race_length=25)

    # Define driver configs
    driver_a_config = {
        'base_lap_time': 95.0,
        'tire_deg_rate': 0.05,
        'consistency': 0.08
    }

    driver_b_config = {
        'base_lap_time': 95.0,
        'tire_deg_rate': 0.05,
        'consistency': 0.08
    }

    # Run undercut analysis
    print("⚡ Analyzing undercut scenario...")
    result = simulator.simulate_undercut_scenario(
        driver_a_config, driver_b_config,
        pit_lap_a=10, pit_lap_b=12
    )

    # Print results
    print("\n📊 UNDERCUT ANALYSIS RESULTS:")
    print("-" * 60)
    print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")

    if result['overtake_lap']:
        print(f"Overtake occurred: Lap {result['overtake_lap']}")
    else:
        print("Overtake occurred: Never")

    print(f"Final gap: {result['time_delta']:.3f}s")
    print("-" * 60)

    print(f"\n{result['summary']}")

    # Gap evolution
    print("\n📉 GAP EVOLUTION (First 10 laps after undercut attempt):")
    print("-" * 60)
    print(f"{'Lap':<6} {'Gap (A to B)':<15} {'Position'}")
    print("-" * 60)

    for gap_data in result['gap_evolution'][9:19]:  # Show laps around undercut
        lap = gap_data['lap']
        gap = gap_data['gap']
        pos_a = gap_data['position_a']
        pos_b = gap_data['position_b']

        gap_str = f"{gap:+.3f}s"
        pos_str = f"A: P{pos_a}, B: P{pos_b}"

        marker = ""
        if lap == result['critical_laps']['driver_a_pit']:
            marker = "🔧 A pits"
        elif lap == result['critical_laps']['driver_b_pit']:
            marker = "🔧 B pits"
        elif lap == result['overtake_lap']:
            marker = "🏁 Overtake!"

        print(f"{lap:<6} {gap_str:<15} {pos_str:<20} {marker}")

    print("-" * 60)

    return result


def scenario_3_multi_driver_chaos():
    """Scenario 3: Multi-driver race with different strategies."""
    print_header("SCENARIO 3: 5-Driver Strategy Battle")

    print("Setup:")
    print("  • 5 drivers with varying strategies")
    print("  • Pit windows from lap 8 to lap 18")
    print("  • Different tire degradation rates")
    print("  • Who has the optimal strategy?\n")

    # Create simulator
    simulator = MultiDriverRaceSimulator(race_length=25)

    # Define drivers with varying characteristics
    drivers_data = {
        'Car30': {
            'name': 'Car #30 (Very Early)',
            'base_lap_time': 94.5,
            'tire_deg_rate': 0.06,  # High degradation
            'consistency': 0.10
        },
        'Car32': {
            'name': 'Car #32 (Early)',
            'base_lap_time': 94.8,
            'tire_deg_rate': 0.05,
            'consistency': 0.08
        },
        'Car21': {
            'name': 'Car #21 (Mid)',
            'base_lap_time': 94.6,
            'tire_deg_rate': 0.045,  # Low degradation
            'consistency': 0.12
        },
        'Car14': {
            'name': 'Car #14 (Late)',
            'base_lap_time': 94.9,
            'tire_deg_rate': 0.05,
            'consistency': 0.09
        },
        'Car8': {
            'name': 'Car #8 (Very Late)',
            'base_lap_time': 95.1,
            'tire_deg_rate': 0.048,
            'consistency': 0.11
        }
    }

    # Different pit strategies
    strategies = {
        'Car30': {'pit_laps': [8]},
        'Car32': {'pit_laps': [11]},
        'Car21': {'pit_laps': [13]},
        'Car14': {'pit_laps': [15]},
        'Car8': {'pit_laps': [18]}
    }

    # Run simulation
    print("🏁 Running 5-driver simulation...")
    result = simulator.simulate_race(drivers_data, strategies)

    # Print results
    print_results(result['final_results'])
    print_position_changes(result['position_changes'])

    # Identify optimal strategy
    winner = result['final_results'][0]
    winner_pit = strategies[winner['driver_id']]['pit_laps'][0]

    print(f"\n✅ WINNER: {winner['driver_name']}")
    print(f"   Optimal pit lap: {winner_pit}")
    print(f"   Winning margin: {result['final_results'][1]['gap_to_leader']:.2f}s")

    # Strategy analysis
    print("\n📈 STRATEGY COMPARISON:")
    print("-" * 60)
    print(f"{'Car':<10} {'Pit Lap':<10} {'Final Pos':<12} {'Avg Lap Time'}")
    print("-" * 60)

    for res in result['final_results']:
        driver_id = res['driver_id']
        pit_lap = strategies[driver_id]['pit_laps'][0]
        final_pos = res['position']
        avg_lap = f"{res['avg_lap_time']:.2f}s"

        print(f"{driver_id:<10} {pit_lap:<10} P{final_pos:<11} {avg_lap}")

    print("-" * 60)

    return result


def scenario_4_team_strategy():
    """Scenario 4: Team strategy optimization."""
    print_header("SCENARIO 4: Team Strategy Optimization")

    print("Setup:")
    print("  • 2 team cars vs 3 opponents")
    print("  • Goal: Maximize team points")
    print("  • Finding optimal strategy for both team cars\n")

    # Create simulator
    simulator = MultiDriverRaceSimulator(race_length=25)

    # Define team drivers
    team_drivers = {
        'T1': {
            'name': 'Team Car #1',
            'base_lap_time': 94.5,
            'tire_deg_rate': 0.05,
            'consistency': 0.08
        },
        'T2': {
            'name': 'Team Car #2',
            'base_lap_time': 94.7,
            'tire_deg_rate': 0.05,
            'consistency': 0.09
        }
    }

    # Define opponents
    opponents = {
        'O1': {
            'name': 'Opponent #1',
            'base_lap_time': 94.8,
            'tire_deg_rate': 0.05,
            'consistency': 0.10
        },
        'O2': {
            'name': 'Opponent #2',
            'base_lap_time': 95.0,
            'tire_deg_rate': 0.05,
            'consistency': 0.11
        },
        'O3': {
            'name': 'Opponent #3',
            'base_lap_time': 95.2,
            'tire_deg_rate': 0.05,
            'consistency': 0.12
        }
    }

    # Optimize team strategy
    print("🎯 Optimizing team strategy...")
    result = simulator.optimize_team_strategy(
        team_drivers, opponents,
        objective='maximize_team_points'
    )

    # Print results
    print("\n📊 OPTIMAL TEAM STRATEGY:")
    print("-" * 60)
    print(result['recommendation'])
    print("-" * 60)

    print("\n🏁 PIT STRATEGIES:")
    for driver_id, strategy in result['optimal_strategies'].items():
        pit_lap = strategy['pit_laps'][0]
        is_team = driver_id.startswith('T')
        marker = "🏎️ " if is_team else "🚗 "
        print(f"   {marker}{driver_id}: Pit on lap {pit_lap}")

    print(f"\n📈 Expected Team Score: {result['team_score']:.1f} points")

    # Show expected results
    if result['expected_result']:
        print("\n📊 EXPECTED RACE RESULTS:")
        print_results(result['expected_result']['final_results'])

    return result


def scenario_5_overcut_analysis():
    """Scenario 5: Overcut strategy demonstration."""
    print_header("SCENARIO 5: Overcut Strategy (Staying Out Longer)")

    print("Setup:")
    print("  • Driver A: Stays out, pits lap 14")
    print("  • Driver B: Pits early, lap 10")
    print("  • Can A gain track position by staying out?\n")

    # Create simulator
    simulator = MultiDriverRaceSimulator(race_length=25)

    # Define driver configs
    driver_a_config = {
        'base_lap_time': 95.0,
        'tire_deg_rate': 0.04,  # Lower degradation (can stay out longer)
        'consistency': 0.08
    }

    driver_b_config = {
        'base_lap_time': 95.0,
        'tire_deg_rate': 0.05,
        'consistency': 0.08
    }

    # Run overcut analysis
    print("⚡ Analyzing overcut scenario...")
    result = simulator.simulate_overcut_scenario(
        driver_a_config, driver_b_config,
        pit_lap_a=14, pit_lap_b=10
    )

    # Print results
    print("\n📊 OVERCUT ANALYSIS RESULTS:")
    print("-" * 60)
    print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")
    print(f"Laps stayed out: {result['laps_stayed_out']}")

    if result['overtake_lap']:
        print(f"Overtake occurred: Lap {result['overtake_lap']}")
    else:
        print("Overtake occurred: Never")

    print(f"Final gap: {result['time_delta']:.3f}s")
    print("-" * 60)

    print(f"\n{result['summary']}")

    return result


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 80)
    print("  MULTI-DRIVER RACE SIMULATION - COMPREHENSIVE DEMONSTRATION")
    print("  RaceIQ Pro - Toyota GR Cup Racing Intelligence Platform")
    print("=" * 80)

    # Run each scenario
    scenario_1_simple_2_driver()
    input("\nPress Enter to continue to Scenario 2...")

    scenario_2_undercut_demo()
    input("\nPress Enter to continue to Scenario 3...")

    scenario_3_multi_driver_chaos()
    input("\nPress Enter to continue to Scenario 4...")

    scenario_4_team_strategy()
    input("\nPress Enter to continue to Scenario 5...")

    scenario_5_overcut_analysis()

    # Final summary
    print_header("DEMONSTRATION COMPLETE")
    print("✅ All scenarios executed successfully!")
    print("\nKey Capabilities Demonstrated:")
    print("  1. ✓ Multi-driver race simulation with position changes")
    print("  2. ✓ Undercut strategy analysis")
    print("  3. ✓ Overcut strategy analysis")
    print("  4. ✓ Team strategy optimization")
    print("  5. ✓ Complex multi-driver scenarios")
    print("\nFor interactive simulation, run the Streamlit dashboard:")
    print("  $ streamlit run dashboard/app.py")
    print("\nNavigate to: 🏎️ Race Simulator")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_scenarios()
