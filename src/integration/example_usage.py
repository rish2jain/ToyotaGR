"""
Example usage of RaceIQ Pro Integration Engine

This demonstrates how the integration engine connects tactical insights
with strategic recommendations to create the full chain of impact:

"Fix Section 3 brake anomaly → Save 0.8s/lap → Delay pit to Lap 16 → Gain P3"
"""

from intelligence_engine import IntegrationEngine, IntegratedInsight
from recommendation_builder import RecommendationBuilder


def example_anomaly_integration():
    """
    Example: Driver is braking too early in Section 3.

    The integration engine connects this tactical issue to pit strategy:
    1. Identifies 0.8s/lap time loss from early braking
    2. Calculates that fixing this allows extending pit window by 2 laps
    3. Shows potential P3 position gain from combined improvement
    """
    print("=" * 80)
    print("EXAMPLE 1: Anomaly Detection → Strategy Integration")
    print("=" * 80)

    # Initialize engine
    engine = IntegrationEngine(config={
        'lap_time_threshold': 0.1,
        'position_value': 2.0,
        'confidence_threshold': 0.7
    })

    # Mock anomaly data (would come from tactical module)
    anomaly = {
        'id': 'ANOM_001',
        'section': 'Section 3 (Turn 7)',
        'type': 'brake_early',
        'anomaly_type': 'brake_early',
        'magnitude': 0.8,  # 0.8 seconds lost
        'baseline_time': 12.5,
        'current_time': 13.3,
        'lap': 8,
        'confidence': 0.85,
        'consistency': 0.75  # Happens 75% of the time
    }

    # Mock tire model (would come from strategic module)
    class MockTireModel:
        def get_degradation_rate(self, lap):
            return 0.05  # 0.05s/lap degradation

        def get_tire_life(self):
            return 0.4  # 40% tire life remaining

    # Mock strategy optimizer (would come from strategic module)
    class MockStrategyOptimizer:
        def get_optimal_pit_lap(self):
            return 14

        def get_total_laps(self):
            return 25

    tire_model = MockTireModel()
    strategy = MockStrategyOptimizer()

    # Connect anomaly to strategy
    impact = engine.connect_anomaly_to_strategy(
        anomaly=anomaly,
        tire_model=tire_model,
        strategy_optimizer=strategy
    )

    # Display results
    print(f"\n🔍 ANOMALY DETECTED:")
    print(f"   Section: {impact.section}")
    print(f"   Issue: Driver braking {anomaly['magnitude']:.1f}s too early")
    print(f"   Lap Time Loss: {impact.lap_time_loss:.2f}s per lap")
    print(f"   Confidence: {impact.confidence*100:.0f}%")

    print(f"\n🎯 STRATEGIC IMPACT:")
    print(f"   Current Pit Lap: 14")
    print(f"   Adjusted Pit Lap: {impact.optimal_pit_lap}")
    print(f"   Position Gain Potential: P{impact.position_gain_potential}")

    print(f"\n💡 CHAIN OF IMPACT:")
    print(f"   Fix brake point → Save {impact.lap_time_loss:.2f}s/lap →")
    print(f"   Extend stint to Lap {impact.optimal_pit_lap} →")
    print(f"   Gain P{impact.position_gain_potential}")

    print("\n" + "=" * 80)


def example_section_improvement():
    """
    Example: Driver has potential to improve Section 5 apex by 0.4s.

    The integration engine calculates how this tactical improvement
    affects pit strategy and overall race position.
    """
    print("\nEXAMPLE 2: Section Improvement → Strategy Adjustment")
    print("=" * 80)

    engine = IntegrationEngine()

    # Mock section improvement data
    section_improvement = {
        'section': 'Section 5 (Turns 10-11)',
        'current_time': 18.7,
        'potential_time': 18.3,
        'type': 'apex',
        'confidence': 0.78,
        'improvement_type': 'apex'
    }

    # Mock race data
    race_data = {
        'current_lap': 10,
        'total_laps': 25,
        'driver_consistency': 0.85
    }

    # Mock strategy
    class MockStrategy:
        def get_optimal_pit_lap(self):
            return 15

    strategy = MockStrategy()

    # Analyze section improvement impact
    analysis = engine.connect_section_improvement_to_strategy(
        section_improvement=section_improvement,
        race_data=race_data,
        strategy=strategy
    )

    print(f"\n🏁 IMPROVEMENT OPPORTUNITY:")
    print(f"   Section: {analysis.section}")
    print(f"   Current Time: {analysis.current_time:.2f}s")
    print(f"   Potential Time: {analysis.potential_time:.2f}s")
    print(f"   Time Gain per Lap: {analysis.time_gain_per_lap:.2f}s")

    print(f"\n📊 RACE IMPACT:")
    print(f"   Laps Remaining: {analysis.laps_remaining}")
    print(f"   Total Time Gain: {analysis.total_time_gain:.1f}s")
    print(f"   Position Impact: P{analysis.position_impact}")

    print(f"\n🎲 STRATEGY ADJUSTMENT:")
    print(f"   Current Pit Plan: Lap 15")
    print(f"   Adjusted Pit Plan: Lap {analysis.adjusted_pit_lap}")
    print(f"   Rationale: Extend stint to maximize pace advantage")

    print("\n" + "=" * 80)


def example_unified_recommendations():
    """
    Example: Generate unified recommendations combining multiple insights.

    This shows the power of the integration engine: connecting dots
    between multiple tactical and strategic factors.
    """
    print("\nEXAMPLE 3: Unified Recommendations (Full Integration)")
    print("=" * 80)

    engine = IntegrationEngine()
    builder = RecommendationBuilder()

    # Mock tactical insights
    tactical_insights = [
        {
            'type': 'anomaly',
            'section': 'Section 3',
            'anomaly_type': 'brake_early',
            'magnitude': 0.8,
            'lap_impact': 0.56,
            'confidence': 0.85
        },
        {
            'type': 'improvement_opportunity',
            'section': 'Section 7',
            'time_gain_per_lap': 0.35,
            'improvement_type': 'apex',
            'confidence': 0.72
        }
    ]

    # Mock strategic insights
    strategic_insights = [
        {
            'type': 'pit_timing',
            'optimal_lap': 15,
            'laps_remaining': 15,
            'confidence': 0.88
        }
    ]

    # Generate unified recommendations
    unified = engine.generate_unified_recommendation(
        tactical_insights=tactical_insights,
        strategic_insights=strategic_insights
    )

    print(f"\n📋 GENERATED {len(unified)} UNIFIED RECOMMENDATIONS:\n")

    for i, insight in enumerate(unified, 1):
        print(f"   {i}. Priority {insight.priority}: {insight.tactical_element}")
        print(f"      {insight.chain_of_impact}")
        print(f"      Confidence: {insight.confidence*100:.0f}%")
        print(f"      Position Gain: P{insight.projected_position_gain}\n")

    print("=" * 80)


def example_dashboard_format():
    """
    Example: Format recommendations for dashboard display.

    Shows how the recommendation builder transforms insights into
    dashboard-ready format with action items and visual hints.
    """
    print("\nEXAMPLE 4: Dashboard-Ready Recommendations")
    print("=" * 80)

    builder = RecommendationBuilder()

    # Mock section analysis
    section_analysis = {
        'sections': [
            {
                'name': 'Section 3',
                'has_improvement_potential': True,
                'time_gain_potential': 0.6,
                'improvement_type': 'brake',
                'confidence': 0.82
            }
        ],
        'reference_lap_id': 'LAP_005'
    }

    # Mock anomalies
    anomalies = [
        {
            'section': 'Section 3',
            'anomaly_type': 'brake_early',
            'magnitude': 0.8,
            'confidence': 0.85,
            'reference_lap_id': 'LAP_005'
        }
    ]

    # Build tactical recommendations
    tactical_recs = builder.build_tactical_recommendation(
        section_analysis=section_analysis,
        anomalies=anomalies
    )

    # Mock strategic data
    pit_strategy = {
        'optimal_lap': 15,
        'window_start': 13,
        'window_end': 17,
        'confidence': 0.88,
        'pit_loss': 22.5
    }

    tire_state = {
        'degradation_rate': 0.05,
        'tire_life': 0.45
    }

    # Build strategic recommendations
    strategic_recs = builder.build_strategic_recommendation(
        pit_strategy=pit_strategy,
        tire_state=tire_state
    )

    # Build integrated recommendations
    integrated_recs = builder.build_integrated_recommendation(
        tactical=tactical_recs,
        strategic=strategic_recs,
        integration_engine=IntegrationEngine()
    )

    # Format for dashboard
    dashboard_data = builder.format_for_dashboard(integrated_recs)

    print(f"\n📊 DASHBOARD DATA:")
    print(f"   Total Recommendations: {dashboard_data['total_recommendations']}")
    print(f"   Timestamp: {dashboard_data['timestamp']}")

    print(f"\n   Priority Breakdown:")
    for priority, count in dashboard_data['priority_breakdown'].items():
        if count > 0:
            print(f"      {priority.upper()}: {count}")

    print(f"\n   Summary:")
    summary = dashboard_data['summary']
    print(f"      Total Time Gain Potential: {summary['total_time_gain_potential']}")
    print(f"      Max Position Gain: {summary['max_position_gain']}")
    print(f"      Average Confidence: {summary['avg_confidence']}")

    print(f"\n   Quick Actions (Top 3):")
    for i, action in enumerate(dashboard_data['quick_actions'], 1):
        print(f"      {i}. {action}")

    print(f"\n   Detailed Recommendations:")
    for rec in dashboard_data['recommendations']:
        print(f"\n      [{rec['priority_label']}] {rec['title']}")
        print(f"      Chain: {rec['chain_of_impact']}")
        print(f"      Impact: {rec['expected_impact']['time_per_lap']}/lap, "
              f"{rec['expected_impact']['total_time']}, {rec['expected_impact']['positions']}")
        print(f"      Confidence: {rec['confidence']}")
        print(f"      Actions:")
        for action in rec['action_items']:
            print(f"         - {action['role']}: {action['action']}")

    print("\n" + "=" * 80)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "RaceIQ Pro Integration Engine" + " " * 29 + "║")
    print("║" + " " * 26 + "Example Scenarios" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run examples
    example_anomaly_integration()
    example_section_improvement()
    example_unified_recommendations()
    example_dashboard_format()

    print("\n" + "=" * 80)
    print("KEY DIFFERENTIATOR:")
    print("=" * 80)
    print("""
RaceIQ Pro doesn't just show data - it CONNECTS THE DOTS:

✓ Tactical insights (driver performance) → Strategic recommendations (pit timing)
✓ Anomaly detection → Lap time impact → Position gain projection
✓ Section improvements → Pit window adjustments → Race outcome
✓ Unified recommendations with full chain of impact visualization

Example output:
    "Fix Section 3 brake anomaly → Save 0.8s/lap → Delay pit to Lap 16 → Gain P3"

This is what separates RaceIQ Pro from basic telemetry analysis - it provides
ACTIONABLE, CONNECTED insights that drivers and strategists can immediately use
to improve race performance.
""")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
