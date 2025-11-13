"""
Bayesian Strategy Optimization Demo
====================================

Demonstrates Bayesian uncertainty quantification for pit strategy optimization.

This example shows:
1. Point estimate vs Bayesian approach comparison
2. How uncertainty narrows with more data
3. Confidence interval interpretation
4. Risk assessment based on posterior distribution

Usage:
    python examples/bayesian_strategy_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.strategic.strategy_optimizer import PitStrategyOptimizer


def generate_synthetic_race_data(num_laps: int = 15,
                                 baseline_lap_time: float = 95.0,
                                 degradation_rate: float = 0.08,
                                 noise_std: float = 0.3) -> pd.DataFrame:
    """
    Generate synthetic race data for demonstration.

    Args:
        num_laps: Number of laps of data
        baseline_lap_time: Base lap time in seconds
        degradation_rate: Tire degradation per lap
        noise_std: Random variation in lap times

    Returns:
        DataFrame with synthetic lap data
    """
    lap_numbers = np.arange(1, num_laps + 1)

    # Generate lap times with degradation and noise
    lap_times = baseline_lap_time + degradation_rate * lap_numbers + \
                np.random.normal(0, noise_std, num_laps)

    # Format as MM:SS.SSS
    lap_time_strings = []
    for lt in lap_times:
        minutes = int(lt // 60)
        seconds = lt % 60
        lap_time_strings.append(f"{minutes}:{seconds:06.3f}")

    data = pd.DataFrame({
        'LAP_NUMBER': lap_numbers,
        'LAP_TIME': lap_time_strings,
        'DRIVER_NUMBER': [10] * num_laps
    })

    return data


def demonstrate_point_vs_bayesian():
    """Compare point estimate vs Bayesian approach."""
    print("\n" + "="*70)
    print("DEMO 1: Point Estimate vs Bayesian Approach")
    print("="*70)

    # Generate race data
    race_data = generate_synthetic_race_data(num_laps=12)

    # Create tire model
    tire_model = {
        'baseline_lap_time': 95.0,
        'degradation_rate': 0.08,
        'model_type': 'linear'
    }

    race_length = 25

    # Point estimate (traditional Monte Carlo)
    print("\n1. TRADITIONAL MONTE CARLO (Point Estimate)")
    print("-" * 70)

    optimizer_traditional = PitStrategyOptimizer(
        pit_loss_seconds=25.0,
        simulation_iterations=100,
        uncertainty_model='gaussian'
    )

    traditional_result = optimizer_traditional.calculate_optimal_pit_window(
        race_data, tire_model, race_length
    )

    print(f"Optimal Pit Lap: {traditional_result['optimal_pit_lap']}")
    print(f"Pit Window: Laps {traditional_result['pit_window'][0]}-{traditional_result['pit_window'][1]}")
    print(f"Expected Time: {traditional_result['optimal_expected_time']:.2f}s")
    print(f"Uncertainty: ±{traditional_result['optimal_time_uncertainty']:.2f}s")

    # Bayesian approach
    print("\n2. BAYESIAN APPROACH (Full Uncertainty Quantification)")
    print("-" * 70)

    optimizer_bayesian = PitStrategyOptimizer(
        pit_loss_seconds=25.0,
        simulation_iterations=100,
        uncertainty_model='bayesian'
    )

    bayesian_result = optimizer_bayesian.calculate_optimal_pit_window_with_uncertainty(
        race_data, tire_model, race_length
    )

    print(f"Optimal Pit Lap: {bayesian_result['optimal_lap']}")
    print(f"Posterior Mean: {bayesian_result['posterior_mean']:.2f}")
    print(f"Posterior Std: {bayesian_result['posterior_std']:.2f} laps")
    print(f"\nConfidence Intervals:")
    print(f"  80% Confidence: Laps {bayesian_result['confidence_80'][0]}-{bayesian_result['confidence_80'][1]}")
    print(f"  90% Confidence: Laps {bayesian_result['confidence_90'][0]}-{bayesian_result['confidence_90'][1]}")
    print(f"  95% Confidence: Laps {bayesian_result['confidence_95'][0]}-{bayesian_result['confidence_95'][1]}")
    print(f"\nRelative Uncertainty: {bayesian_result['uncertainty']*100:.1f}%")

    # Risk assessment
    risk = bayesian_result['risk_assessment']
    print(f"\nRISK ASSESSMENT:")
    print(f"  Level: {risk['risk_level']}")
    print(f"  Explanation: {risk['explanation']}")
    print(f"  Strategy Note: {risk['strategy_note']}")

    # Comparison
    print("\n3. COMPARISON")
    print("-" * 70)
    print("Key Differences:")
    print(f"  - Traditional gives single pit lap: {traditional_result['optimal_pit_lap']}")
    print(f"  - Bayesian gives confidence range: {bayesian_result['confidence_90'][0]}-{bayesian_result['confidence_90'][1]} (90%)")
    print(f"  - Bayesian quantifies uncertainty: {bayesian_result['uncertainty']*100:.1f}%")
    print(f"  - Bayesian provides risk assessment: {risk['risk_level']}")

    return bayesian_result


def demonstrate_uncertainty_with_data():
    """Show how uncertainty narrows with more data."""
    print("\n" + "="*70)
    print("DEMO 2: Uncertainty Narrows with More Data")
    print("="*70)

    tire_model = {
        'baseline_lap_time': 95.0,
        'degradation_rate': 0.08,
        'model_type': 'linear'
    }

    race_length = 25

    optimizer = PitStrategyOptimizer(
        pit_loss_seconds=25.0,
        simulation_iterations=100,
        uncertainty_model='bayesian'
    )

    # Test with different amounts of data
    data_amounts = [5, 10, 15, 20]
    results = []

    print("\nAnalyzing with increasing amounts of race data...\n")

    for num_laps in data_amounts:
        race_data = generate_synthetic_race_data(num_laps=num_laps)

        result = optimizer.calculate_optimal_pit_window_with_uncertainty(
            race_data, tire_model, race_length
        )

        results.append(result)

        print(f"With {num_laps} laps of data:")
        print(f"  Optimal Lap: {result['optimal_lap']}")
        print(f"  Posterior Std: {result['posterior_std']:.2f} laps")
        print(f"  Uncertainty: {result['uncertainty']*100:.1f}%")
        print(f"  90% Confidence: Laps {result['confidence_90'][0]}-{result['confidence_90'][1]}")
        print(f"  Risk Level: {result['risk_assessment']['risk_level']}")
        print()

    print("OBSERVATION:")
    print("As more race data is collected, the posterior uncertainty decreases")
    print("and the confidence intervals narrow, leading to more precise recommendations.")

    return results


def demonstrate_confidence_intervals():
    """Explain confidence interval interpretation."""
    print("\n" + "="*70)
    print("DEMO 3: Understanding Confidence Intervals")
    print("="*70)

    race_data = generate_synthetic_race_data(num_laps=15)

    tire_model = {
        'baseline_lap_time': 95.0,
        'degradation_rate': 0.08,
        'model_type': 'linear'
    }

    optimizer = PitStrategyOptimizer(
        pit_loss_seconds=25.0,
        simulation_iterations=200,
        uncertainty_model='bayesian'
    )

    result = optimizer.calculate_optimal_pit_window_with_uncertainty(
        race_data, tire_model, race_length=25
    )

    print("\nBayesian Credible Intervals (Interpretation):")
    print("-" * 70)

    print(f"\n80% Credible Interval: Laps {result['confidence_80'][0]}-{result['confidence_80'][1]}")
    print("  → We are 80% confident the true optimal pit lap is in this range")
    print("  → Use for: Conservative strategy with some flexibility")

    print(f"\n90% Credible Interval: Laps {result['confidence_90'][0]}-{result['confidence_90'][1]}")
    print("  → We are 90% confident the true optimal pit lap is in this range")
    print("  → Use for: Balanced strategy - typical recommendation")

    print(f"\n95% Credible Interval: Laps {result['confidence_95'][0]}-{result['confidence_95'][1]}")
    print("  → We are 95% confident the true optimal pit lap is in this range")
    print("  → Use for: Very conservative strategy, accounting for most uncertainty")

    print("\nPractical Strategy Recommendations:")
    print("-" * 70)

    optimal = result['optimal_lap']
    ci_90 = result['confidence_90']

    print(f"Primary Target: Lap {optimal} (posterior mean)")
    print(f"Acceptable Window: Laps {ci_90[0]}-{ci_90[1]} (90% confidence)")
    print(f"\nIf you pit within the 90% window, you're making a statistically")
    print(f"sound decision even if it's not exactly on lap {optimal}.")

    return result


def visualize_posterior_distribution(result):
    """Create visualization of posterior distribution."""
    print("\n" + "="*70)
    print("DEMO 4: Visualizing the Posterior Distribution")
    print("="*70)

    optimizer = PitStrategyOptimizer()
    viz_data = optimizer.visualize_posterior_distribution(result)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Probability Density Function
    ax1 = axes[0]
    pdf_x = viz_data['pdf']['x']
    pdf_y = viz_data['pdf']['y']

    ax1.plot(pdf_x, pdf_y, 'b-', linewidth=2, label='Posterior PDF')
    ax1.fill_between(pdf_x, pdf_y, alpha=0.3, color='blue')

    # Mark optimal lap
    optimal = result['optimal_lap']
    optimal_y = stats.norm.pdf(optimal, result['posterior_mean'], result['posterior_std'])
    ax1.plot([optimal], [optimal_y], 'r*', markersize=20, label=f'Optimal Lap: {optimal}')

    # Shade confidence intervals
    ci_95 = result['confidence_95']
    mask_95 = (np.array(pdf_x) >= ci_95[0]) & (np.array(pdf_x) <= ci_95[1])
    ax1.fill_between(
        np.array(pdf_x)[mask_95],
        np.array(pdf_y)[mask_95],
        alpha=0.3, color='red', label='95% Confidence'
    )

    ax1.set_xlabel('Pit Lap Number', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Posterior Distribution of Optimal Pit Lap', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of posterior samples
    ax2 = axes[1]
    samples = result['posterior_samples']
    ax2.hist(samples, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Overlay theoretical PDF
    ax2.plot(pdf_x, pdf_y, 'r-', linewidth=2, label='Theoretical PDF')

    # Mark confidence intervals
    for ci_level, ci_data, color in [
        ('80%', result['confidence_80'], 'green'),
        ('90%', result['confidence_90'], 'orange'),
        ('95%', result['confidence_95'], 'red')
    ]:
        ax2.axvline(ci_data[0], linestyle='--', color=color, alpha=0.7, label=f'{ci_level} CI')
        ax2.axvline(ci_data[1], linestyle='--', color=color, alpha=0.7)

    ax2.set_xlabel('Pit Lap Number', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Posterior Samples vs Theoretical Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = '/home/user/ToyotaGR/examples/bayesian_posterior_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Display statistics
    print("\nPosterior Distribution Statistics:")
    print("-" * 70)
    print(f"Mean: {result['posterior_mean']:.2f} laps")
    print(f"Median: {optimal} laps")
    print(f"Std Dev: {result['posterior_std']:.2f} laps")
    print(f"Coefficient of Variation: {result['uncertainty']*100:.1f}%")

    return fig


def compare_risk_levels():
    """Demonstrate different risk scenarios."""
    print("\n" + "="*70)
    print("DEMO 5: Risk Assessment in Different Scenarios")
    print("="*70)

    optimizer = PitStrategyOptimizer(simulation_iterations=200)

    scenarios = [
        {
            'name': 'High Certainty (Fresh Data, Stable Conditions)',
            'num_laps': 20,
            'degradation_rate': 0.05,
            'noise_std': 0.2
        },
        {
            'name': 'Moderate Uncertainty (Mid-Race)',
            'num_laps': 12,
            'degradation_rate': 0.08,
            'noise_std': 0.4
        },
        {
            'name': 'High Uncertainty (Early Race, Variable Conditions)',
            'num_laps': 5,
            'degradation_rate': 0.10,
            'noise_std': 0.8
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 70)

        race_data = generate_synthetic_race_data(
            num_laps=scenario['num_laps'],
            degradation_rate=scenario['degradation_rate'],
            noise_std=scenario['noise_std']
        )

        tire_model = {
            'baseline_lap_time': 95.0,
            'degradation_rate': scenario['degradation_rate'],
            'model_type': 'linear'
        }

        result = optimizer.calculate_optimal_pit_window_with_uncertainty(
            race_data, tire_model, race_length=25
        )

        risk = result['risk_assessment']

        print(f"Data: {scenario['num_laps']} laps")
        print(f"Optimal Pit Lap: {result['optimal_lap']}")
        print(f"Posterior Std: {result['posterior_std']:.2f} laps")
        print(f"90% Confidence: Laps {result['confidence_90'][0]}-{result['confidence_90'][1]}")
        print(f"\nRISK LEVEL: {risk['risk_level']}")
        print(f"  {risk['explanation']}")
        print(f"  {risk['strategy_note']}")

    print("\nKEY TAKEAWAY:")
    print("Risk assessment helps teams understand how confident they can be")
    print("in their pit strategy and whether they need to be more conservative.")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" BAYESIAN UNCERTAINTY QUANTIFICATION FOR PIT STRATEGY OPTIMIZATION ")
    print("="*70)
    print("\nThis demo shows how Bayesian methods provide richer information than")
    print("traditional point estimates for race strategy decisions.")

    # Run demonstrations
    bayesian_result = demonstrate_point_vs_bayesian()

    input("\nPress Enter to continue to Demo 2...")
    demonstrate_uncertainty_with_data()

    input("\nPress Enter to continue to Demo 3...")
    demonstrate_confidence_intervals()

    input("\nPress Enter to continue to Demo 4...")
    visualize_posterior_distribution(bayesian_result)

    input("\nPress Enter to continue to Demo 5...")
    compare_risk_levels()

    print("\n" + "="*70)
    print(" SUMMARY ")
    print("="*70)
    print("\nBayesian Uncertainty Quantification Advantages:")
    print("  1. Provides confidence intervals, not just point estimates")
    print("  2. Quantifies uncertainty explicitly")
    print("  3. Incorporates prior knowledge and updates with data")
    print("  4. Enables risk-aware decision making")
    print("  5. Shows how confidence improves with more data")
    print("\nUse Case:")
    print("  - Race strategy teams can make more informed decisions")
    print("  - Engineers can communicate uncertainty to drivers")
    print("  - Teams can balance risk vs reward in pit timing")
    print("  - Provides statistical rigor to strategy recommendations")
    print("\n" + "="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    main()
