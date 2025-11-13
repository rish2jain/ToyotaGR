"""
Causal Inference Analysis Demonstration
========================================

Comprehensive demonstration of causal inference for racing strategy analysis.

This demo shows:
1. Simple causal effect estimation (Section improvement on lap time)
2. Pit strategy counterfactual ("What if we pitted 2 laps earlier?")
3. Section improvement cascading (Section → Lap Time → Position)
4. Confounder analysis (Controlling for tire age)

Usage:
    python examples/causal_analysis_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.integration.causal_analysis import (
    CausalStrategyAnalyzer,
    prepare_race_data_for_causal_analysis,
    DOWHY_AVAILABLE
)


def generate_synthetic_race_data(
    num_drivers: int = 3,
    laps_per_driver: int = 25,
    baseline_lap_time: float = 95.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic race data with realistic causal structure.

    Args:
        num_drivers: Number of drivers
        laps_per_driver: Number of laps per driver
        baseline_lap_time: Base lap time in seconds
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic race data
    """
    np.random.seed(seed)

    data = []

    for driver_id in range(1, num_drivers + 1):
        # Driver skill (latent variable)
        driver_skill = np.random.normal(0, 1)

        # Starting position
        starting_position = driver_id

        for lap in range(1, laps_per_driver + 1):
            # Tire age (laps since last pit)
            if lap <= 12:
                tire_age = lap
            else:
                tire_age = lap - 12  # Assume pit at lap 12

            # Fuel load (decreases linearly)
            fuel_load = 1.0 - (lap / laps_per_driver)

            # Track temperature (increases during race)
            track_temp = 30 + lap * 0.2 + np.random.normal(0, 1)

            # Section times with causal effects
            # Section 1: Affected by tire age and fuel
            section_1 = (
                30.0
                + tire_age * 0.02  # Tire degradation
                + fuel_load * 0.5  # Fuel weight
                - driver_skill * 0.3  # Driver skill
                + np.random.normal(0, 0.2)  # Random noise
            )

            # Section 2: Affected by driver skill and track temp
            section_2 = (
                32.0
                + tire_age * 0.03
                + track_temp * 0.01
                - driver_skill * 0.4
                + np.random.normal(0, 0.2)
            )

            # Section 3: Most affected by tire age
            section_3 = (
                33.0
                + tire_age * 0.05  # High tire sensitivity
                + fuel_load * 0.3
                - driver_skill * 0.5
                + np.random.normal(0, 0.2)
            )

            # Lap time is sum of sections
            lap_time = section_1 + section_2 + section_3

            # Pit lap indicator
            pit_lap = 1 if lap == 12 else 0

            data.append({
                'driver_number': driver_id,
                'lap_number': lap,
                'section_1_time': section_1,
                'section_2_time': section_2,
                'section_3_time': section_3,
                'lap_time': lap_time,
                'tire_age': tire_age,
                'fuel_load': fuel_load,
                'track_temp': track_temp,
                'pit_lap': pit_lap,
                'starting_position': starting_position,
                'driver_skill': driver_skill  # For validation only
            })

    df = pd.DataFrame(data)

    # Calculate final position (inverse of avg lap time)
    avg_lap_times = df.groupby('driver_number')['lap_time'].mean()
    positions = avg_lap_times.rank().astype(int)
    position_map = positions.to_dict()
    df['final_position'] = df['driver_number'].map(position_map)

    return df


def demo_1_simple_causal_effect():
    """
    Demo 1: Simple Causal Effect Estimation
    ========================================
    Estimate the causal effect of improving Section 3 on lap time.
    """
    print("\n" + "=" * 80)
    print("DEMO 1: Simple Causal Effect Estimation")
    print("=" * 80)
    print("\nQuestion: Does improving Section 3 time causally reduce lap time?")
    print("Method: Backdoor adjustment controlling for confounders (tire age, fuel)")
    print("-" * 80)

    # Generate data
    race_data = generate_synthetic_race_data(num_drivers=5, laps_per_driver=25)

    print(f"\nData: {len(race_data)} observations from {race_data['driver_number'].nunique()} drivers")

    # Initialize analyzer
    analyzer = CausalStrategyAnalyzer(min_data_points=15)

    # Analyze Section 3 improvement effect
    try:
        effect = analyzer.analyze_section_improvement_effect(
            race_data,
            section_id=3,
            outcome='lap_time'
        )

        print("\nRESULTS:")
        print(f"  Effect Size: {effect.effect_size:.4f} seconds/second")
        print(f"  Interpretation: A 1-second improvement in Section 3 causes a")
        print(f"                  {abs(effect.effect_size):.4f}s improvement in lap time")
        print(f"\n  Confidence Interval (95%): [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]")
        print(f"  P-Value: {effect.p_value:.4f}")
        print(f"  Statistical Significance: {'Yes' if effect.p_value < 0.05 else 'No'}")
        print(f"\n  Robustness Score: {effect.robustness_score:.2f}")

        print("\nINTERPRETATION:")
        print(f"  {effect.interpretation}")

        # Show comparison: Naive correlation vs Causal effect
        print("\n" + "-" * 80)
        print("COMPARISON: Naive Correlation vs Causal Effect")
        print("-" * 80)

        # Naive correlation
        correlation = race_data[['section_3_time', 'lap_time']].corr().iloc[0, 1]
        print(f"  Naive Correlation: {correlation:.4f}")
        print(f"  Causal Effect: {effect.effect_size:.4f}")
        print(f"\n  Note: Causal effect accounts for confounding by tire age and fuel load.")

    except Exception as e:
        print(f"Error in Demo 1: {e}")
        import traceback
        traceback.print_exc()


def demo_2_pit_strategy_counterfactual():
    """
    Demo 2: Pit Strategy Counterfactual
    ====================================
    "What if we pitted 2 laps earlier?"
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Pit Strategy Counterfactual")
    print("=" * 80)
    print("\nQuestion: What if we pitted at lap 10 instead of lap 12?")
    print("Method: Counterfactual estimation with intervention")
    print("-" * 80)

    # Generate data
    race_data = generate_synthetic_race_data(num_drivers=5, laps_per_driver=25)

    print(f"\nData: {len(race_data)} observations")
    print(f"Current pit strategy: Lap 12")
    print(f"Counterfactual scenario: Pit at lap 10")

    # Initialize analyzer
    analyzer = CausalStrategyAnalyzer(min_data_points=15)

    try:
        # Estimate counterfactual for earlier pit
        # We'll intervene on tire age at lap 15 (after pit)
        # Original: tire_age = 15 - 12 = 3
        # Counterfactual: tire_age = 15 - 10 = 5

        # Select data from lap 15 onwards
        post_pit_data = race_data[race_data['lap_number'] >= 15].copy()

        if len(post_pit_data) > 20:
            counterfactual = analyzer.estimate_counterfactual(
                data=post_pit_data,
                treatment='tire_age',
                outcome='lap_time',
                intervention_value=5.0,  # Tire age if pitted at lap 10
                common_causes=['fuel_load', 'track_temp']
            )

            print("\nRESULTS:")
            print(f"  Original Scenario (Pit Lap 12):")
            print(f"    Avg tire age: {post_pit_data['tire_age'].mean():.2f} laps")
            print(f"    Avg lap time: {counterfactual.original_outcome:.3f}s")
            print(f"\n  Counterfactual Scenario (Pit Lap 10):")
            print(f"    Avg tire age: 5.00 laps (higher degradation)")
            print(f"    Predicted lap time: {counterfactual.counterfactual_outcome:.3f}s")
            print(f"\n  Effect of pitting 2 laps earlier:")
            print(f"    Lap time change: {counterfactual.effect_size:+.3f}s per lap")
            print(f"    Total effect: {counterfactual.effect_size * 10:+.2f}s over 10 laps")
            print(f"\n  Confidence Interval: [{counterfactual.confidence_interval[0]:.3f}, "
                  f"{counterfactual.confidence_interval[1]:.3f}]")

            print("\nINTERPRETATION:")
            print(f"  {counterfactual.practical_interpretation}")

            print("\nRECOMMENDATION:")
            if counterfactual.effect_size < 0:
                print("  ✓ Earlier pit stop (lap 10) would IMPROVE lap times")
            else:
                print("  ✗ Earlier pit stop (lap 10) would WORSEN lap times")
                print("    Current pit strategy (lap 12) is better")

        else:
            print("\n⚠️  Insufficient post-pit data for analysis")

    except Exception as e:
        print(f"Error in Demo 2: {e}")
        import traceback
        traceback.print_exc()


def demo_3_cascading_effects():
    """
    Demo 3: Cascading Causal Effects
    =================================
    Section 3 improvement → Lap Time → Final Position
    """
    print("\n" + "=" * 80)
    print("DEMO 3: Cascading Causal Effects")
    print("=" * 80)
    print("\nQuestion: If we improve Section 3, how does it cascade to final position?")
    print("Method: Multi-step causal chain")
    print("-" * 80)

    # Generate data
    race_data = generate_synthetic_race_data(num_drivers=5, laps_per_driver=25)

    print(f"\nData: {len(race_data)} observations")

    # Initialize analyzer
    analyzer = CausalStrategyAnalyzer(min_data_points=15)

    try:
        # Step 1: Section 3 → Lap Time
        print("\nSTEP 1: Section 3 → Lap Time")
        print("-" * 40)

        effect_s3_to_lap = analyzer.identify_causal_effect(
            data=race_data,
            treatment='section_3_time',
            outcome='lap_time',
            common_causes=['tire_age', 'fuel_load']
        )

        print(f"  Effect: {effect_s3_to_lap.effect_size:.4f} s/s")
        print(f"  Interpretation: 1s improvement in Section 3 → "
              f"{abs(effect_s3_to_lap.effect_size):.4f}s improvement in lap time")

        # Step 2: Lap Time → Final Position
        print("\nSTEP 2: Lap Time → Final Position")
        print("-" * 40)

        # Aggregate to driver level
        driver_data = race_data.groupby('driver_number').agg({
            'lap_time': 'mean',
            'final_position': 'first',
            'section_3_time': 'mean',
            'tire_age': 'mean'
        }).reset_index()

        if len(driver_data) >= 3:
            effect_lap_to_pos = analyzer.identify_causal_effect(
                data=driver_data,
                treatment='lap_time',
                outcome='final_position',
                common_causes=None  # No confounders at driver level
            )

            print(f"  Effect: {effect_lap_to_pos.effect_size:.4f} positions/s")
            print(f"  Interpretation: 1s improvement in avg lap time → "
                  f"{abs(effect_lap_to_pos.effect_size):.4f} position improvement")

            # Step 3: Total cascading effect
            print("\nTOTAL CASCADING EFFECT")
            print("-" * 40)

            # Section 3 → Position (through Lap Time)
            total_effect = effect_s3_to_lap.effect_size * effect_lap_to_pos.effect_size

            print(f"  Section 3 → Lap Time: {effect_s3_to_lap.effect_size:.4f}")
            print(f"  Lap Time → Position: {effect_lap_to_pos.effect_size:.4f}")
            print(f"  Total Effect (Section 3 → Position): {total_effect:.4f}")
            print(f"\n  Practical Interpretation:")
            print(f"    Improving Section 3 by 1.0s causes {abs(total_effect):.4f} position improvement")
            print(f"    Improving Section 3 by 0.5s causes {abs(total_effect * 0.5):.4f} position improvement")

        else:
            print("  ⚠️  Insufficient drivers for position analysis")

    except Exception as e:
        print(f"Error in Demo 3: {e}")
        import traceback
        traceback.print_exc()


def demo_4_confounder_analysis():
    """
    Demo 4: Confounder Analysis
    ============================
    Show importance of controlling for tire age.
    """
    print("\n" + "=" * 80)
    print("DEMO 4: Confounder Analysis - Importance of Controlling Variables")
    print("=" * 80)
    print("\nQuestion: What happens if we DON'T control for tire age?")
    print("Method: Compare naive vs controlled analysis")
    print("-" * 80)

    # Generate data
    race_data = generate_synthetic_race_data(num_drivers=5, laps_per_driver=25)

    print(f"\nData: {len(race_data)} observations")

    # Initialize analyzer
    analyzer = CausalStrategyAnalyzer(min_data_points=15)

    try:
        # Analysis WITHOUT controlling for tire age (WRONG)
        print("\nANALYSIS 1: WITHOUT Confounder Control (Naive)")
        print("-" * 40)

        effect_naive = analyzer.identify_causal_effect(
            data=race_data,
            treatment='section_3_time',
            outcome='lap_time',
            common_causes=None  # No confounders!
        )

        print(f"  Effect Size: {effect_naive.effect_size:.4f}")
        print(f"  P-Value: {effect_naive.p_value:.4f}")

        # Analysis WITH controlling for tire age (CORRECT)
        print("\nANALYSIS 2: WITH Confounder Control (Correct)")
        print("-" * 40)

        effect_controlled = analyzer.identify_causal_effect(
            data=race_data,
            treatment='section_3_time',
            outcome='lap_time',
            common_causes=['tire_age', 'fuel_load', 'track_temp']  # Control confounders
        )

        print(f"  Effect Size: {effect_controlled.effect_size:.4f}")
        print(f"  P-Value: {effect_controlled.p_value:.4f}")
        print(f"  Robustness: {effect_controlled.robustness_score:.2f}")

        # Comparison
        print("\nCOMPARISON")
        print("-" * 40)
        difference = abs(effect_naive.effect_size - effect_controlled.effect_size)
        bias_pct = (difference / abs(effect_controlled.effect_size)) * 100 if effect_controlled.effect_size != 0 else 0

        print(f"  Naive Effect: {effect_naive.effect_size:.4f}")
        print(f"  Controlled Effect: {effect_controlled.effect_size:.4f}")
        print(f"  Difference (Bias): {difference:.4f} ({bias_pct:.1f}%)")

        print("\nCONCLUSION:")
        if bias_pct > 10:
            print(f"  ⚠️  Controlling for confounders makes a {bias_pct:.1f}% difference!")
            print("     The naive analysis is BIASED due to confounding by tire age.")
        else:
            print("  ✓ Confounding bias is minimal in this case.")

        print("\nWHY IT MATTERS:")
        print("  Tire age affects both Section 3 time AND lap time.")
        print("  If we don't control for it, we might think Section 3 improvement")
        print("  has a different effect than it actually does.")

    except Exception as e:
        print(f"Error in Demo 4: {e}")
        import traceback
        traceback.print_exc()


def demo_5_visualize_causal_graph():
    """
    Demo 5: Visualize Causal Graph
    ===============================
    Show the assumed causal structure.
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Causal Graph Visualization")
    print("=" * 80)
    print("\nVisualize the causal relationships between racing variables")
    print("-" * 80)

    # Generate data
    race_data = generate_synthetic_race_data(num_drivers=3, laps_per_driver=20)

    # Initialize analyzer
    analyzer = CausalStrategyAnalyzer()

    try:
        # Build causal graph
        graph = analyzer.build_causal_graph(race_data, include_weather=True)

        print(f"\nCausal Graph Built:")
        print(f"  Nodes: {graph['metadata']['num_nodes']}")
        print(f"  Edges: {graph['metadata']['num_edges']}")

        # Show some key edges
        print("\nKey Causal Relationships:")
        key_edges = [
            ('section_3_time', 'lap_time'),
            ('tire_age', 'lap_time'),
            ('lap_time', 'race_position'),
            ('pit_lap', 'tire_age')
        ]

        for src, dst in key_edges:
            if (src, dst) in graph['edges']:
                print(f"  • {src} → {dst}")

        # Visualize
        print("\nGenerating visualization...")
        output_path = '/home/user/ToyotaGR/examples/causal_graph_demo.png'
        fig = analyzer.visualize_causal_graph(save_path=output_path)

        print(f"  ✓ Causal graph saved to: {output_path}")

        plt.close(fig)

    except Exception as e:
        print(f"Error in Demo 5: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" CAUSAL INFERENCE FOR RACING STRATEGY ANALYSIS ")
    print("=" * 80)
    print("\nThis demo shows how causal inference goes beyond correlation to")
    print("establish cause-and-effect relationships in racing data.")
    print("\nTopics covered:")
    print("  1. Simple causal effect estimation")
    print("  2. Counterfactual analysis (what-if scenarios)")
    print("  3. Cascading causal effects")
    print("  4. Importance of controlling for confounders")
    print("  5. Causal graph visualization")

    if not DOWHY_AVAILABLE:
        print("\n" + "=" * 80)
        print("ERROR: DoWhy not available")
        print("=" * 80)
        print("\nThe DoWhy library is required for causal inference.")
        print("Install it with: pip install dowhy")
        print("\nThen re-run this demo.")
        return

    # Run all demos
    try:
        demo_1_simple_causal_effect()

        input("\nPress Enter to continue to Demo 2...")
        demo_2_pit_strategy_counterfactual()

        input("\nPress Enter to continue to Demo 3...")
        demo_3_cascading_effects()

        input("\nPress Enter to continue to Demo 4...")
        demo_4_confounder_analysis()

        input("\nPress Enter to continue to Demo 5...")
        demo_5_visualize_causal_graph()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return

    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY ")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Causal inference identifies cause-effect relationships, not just correlations")
    print("  2. Counterfactuals answer 'what-if' questions with statistical rigor")
    print("  3. Causal effects can cascade through multiple variables")
    print("  4. Controlling for confounders is CRITICAL for valid conclusions")
    print("  5. Causal graphs make assumptions explicit and transparent")
    print("\nApplications in Racing:")
    print("  • Predict impact of driver improvements on race results")
    print("  • Optimize pit strategy with counterfactual analysis")
    print("  • Identify which changes will have biggest impact")
    print("  • Avoid spurious correlations in telemetry data")
    print("\nStatistical Rigor:")
    print("  • Backdoor adjustment controls for confounding")
    print("  • Confidence intervals quantify uncertainty")
    print("  • Sensitivity analysis tests robustness")
    print("  • P-values indicate statistical significance")
    print("\n" + "=" * 80)
    print("\nFor more information, see docs/CAUSAL_INFERENCE.md")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    main()
